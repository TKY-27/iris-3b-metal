"""Device-mesh factorization, DDP kwarg resolution and FSDP2 wrap targets.

Every function under test is pure: no process group, no CUDA. What a 2-GPU
smoke cannot catch cheaply is exactly this layer — a mesh that quietly drops
half the world, a DDP flag that never reaches the reducer, or a block class
that is left out of sharding.
"""

import pytest
import torch

from iris3b.config import DistConfig, MeshConfig, ModelConfig
from iris3b.train.parallel import (
    assert_optimizer_sees_shards,
    ddp_kwargs,
    mixed_precision_policy,
    resolve_mesh,
    wrap_class_names,
)


def _dist(**kwargs) -> DistConfig:
    kwargs.setdefault("strategy", "fsdp2")
    return DistConfig(**kwargs)


def test_optimizer_built_before_sharding_is_refused():
    """The ordering trap: fully_shard swaps every parameter for a DTensor, so an
    optimizer built first steps orphaned tensors: it agrees with DDP on step 0,
    diverges from step 1 on, and silently holds a full unsharded optimizer state."""
    plain = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(RuntimeError, match="not DTensors"):
        assert_optimizer_sees_shards(torch.optim.AdamW([plain], lr=1e-3), _dist())


def test_ddp_keeps_plain_parameters():
    """Only fsdp2 uses DTensors; DDP replicates plain parameters, so it must
    not be caught by the DTensor check."""
    plain = torch.nn.Parameter(torch.zeros(4))
    assert_optimizer_sees_shards(torch.optim.AdamW([plain], lr=1e-3), _dist(strategy="ddp"))


def test_resolve_mesh_shards_the_whole_world_by_default():
    """dp_shard=0 on one node of 8: one shard group, no replication."""
    assert resolve_mesh(_dist(), world_size=8, local_world_size=8) == (1, 8)


def test_resolve_mesh_hybrid_derives_replication_from_the_shard_group():
    """16 ranks over 2 nodes, sharded inside a node: 2 replicas of 8."""
    cfg = _dist(sharding="hybrid", mesh=MeshConfig(dp_shard=8))
    assert resolve_mesh(cfg, world_size=16, local_world_size=8) == (2, 8)

    explicit = _dist(sharding="hybrid_grad_op", mesh=MeshConfig(dp_shard=8, dp_replicate=2))
    assert resolve_mesh(explicit, world_size=16, local_world_size=8) == (2, 8)


def test_resolve_mesh_rejects_a_factorization_that_drops_ranks():
    """dp_replicate x dp_shard != world_size trains on a subset in silence."""
    cfg = _dist(sharding="hybrid", mesh=MeshConfig(dp_shard=8, dp_replicate=1))
    with pytest.raises(ValueError, match=r"dp_shard=8.*world_size=16"):
        resolve_mesh(cfg, world_size=16, local_world_size=8)


def test_resolve_mesh_rejects_a_shard_group_across_a_node_boundary():
    cfg = _dist(sharding="hybrid", mesh=MeshConfig(dp_shard=12))
    with pytest.raises(ValueError, match="straddles a node boundary"):
        resolve_mesh(cfg, world_size=24, local_world_size=8)


def test_resolve_mesh_rejects_a_mesh_on_a_non_hybrid_sharding():
    """Full sharding covers every rank; a smaller dp_shard means hybrid."""
    cfg = _dist(mesh=MeshConfig(dp_shard=4))
    with pytest.raises(ValueError, match="shards over every rank"):
        resolve_mesh(cfg, world_size=8, local_world_size=8)


def test_mixed_precision_policy_maps_dtypes():
    """fp32 reduction is the sharded default: bf16 reduction error grows with
    the number of ranks reduced over."""
    assert mixed_precision_policy(_dist()) == (torch.bfloat16, torch.float32)
    assert mixed_precision_policy(_dist(reduce_dtype="bf16")) == (torch.bfloat16, torch.bfloat16)
    assert mixed_precision_policy(_dist(param_dtype="fp32")) == (torch.float32, torch.float32)
    with pytest.raises(ValueError, match="param_dtype"):
        mixed_precision_policy(_dist(param_dtype="int8"))


@pytest.mark.parametrize(
    ("block", "dual_depth", "final_block_text", "expected"),
    [
        ("mmdit", 0, "keep", True),
        ("mmdit", 0, "drop", False),
        # a single-stream block shares every weight between the two halves, so
        # the image rows keep all of them alive and nothing is ever unused
        ("single_stream", 0, "keep", False),
        ("single_stream", 0, "drop", False),
        # dual_depth == depth makes the whole trunk dual, tail included
        ("single_stream", 14, "keep", True),
        # a hybrid trunk is dual only at the FRONT; the tail is still single
        ("single_stream", 6, "keep", False),
    ],
)
def test_ddp_kwargs_auto_resolves_find_unused_from_the_final_block(
    block, dual_depth, final_block_text, expected
):
    """Only a dual-stream final block leaves gradient-free parameters: its text
    tail is computed and discarded. Everything else pays the reducer's
    per-iteration graph walk for nothing."""
    model = ModelConfig(block=block, depth=14, dual_depth=dual_depth, final_block_text=final_block_text)
    assert ddp_kwargs(DistConfig(), model)["find_unused_parameters"] is expected


def test_ddp_kwargs_explicit_setting_overrides_the_model():
    keep = ModelConfig(final_block_text="keep")
    forced_off = DistConfig()
    forced_off.ddp.find_unused_parameters = "false"
    assert ddp_kwargs(forced_off, keep)["find_unused_parameters"] is False

    drop = ModelConfig(final_block_text="drop")
    forced_on = DistConfig()
    forced_on.ddp.find_unused_parameters = "true"
    assert ddp_kwargs(forced_on, drop)["find_unused_parameters"] is True


def test_ddp_kwargs_never_exposes_static_graph():
    """static_graph corrupts the gradient allreduce against this model's unused
    text tail, so it must not be reachable from config."""
    cfg = DistConfig()
    cfg.ddp.gradient_as_bucket_view = True
    cfg.ddp.batched_grad_copy = True
    cfg.ddp.skip_all_reduce_unused_params = True
    assert "static_graph" not in ddp_kwargs(cfg, ModelConfig())


def test_ddp_kwargs_maps_buffer_sync_and_stays_within_the_installed_torch():
    import inspect

    from torch import nn

    accepted = set(inspect.signature(nn.parallel.DistributedDataParallel.__init__).parameters)
    cfg = DistConfig()
    cfg.ddp.forward_sync_buffers = False
    kwargs = ddp_kwargs(cfg, ModelConfig())
    assert kwargs["broadcast_buffers"] is False
    assert set(kwargs) <= accepted


def test_wrap_class_names_covers_both_families_of_a_hybrid_stack():
    """A dual_depth stack runs two block classes; naming one would leave the
    other unsharded and replicated on every rank."""
    text = {"LayerwiseAttentionBlock", "TextAdapterBlock"}  # lap_blocks2 text adapter
    single = ModelConfig(block="single_stream", dual_depth=0)
    assert wrap_class_names(single) == {"SingleStreamBlock", "PiTBlock", *text}

    hybrid = ModelConfig(block="single_stream", dual_depth=4)
    assert wrap_class_names(hybrid) == {"SingleStreamBlock", "MMDiTBlock", "PiTBlock", *text}
