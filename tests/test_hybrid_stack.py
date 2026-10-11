"""Hybrid dual/single trunks and the removable final-block text tail.

Two config-gated topology knobs are guarded here.

``model.dual_depth`` stacks dual-stream blocks under the ``model.block`` family.
A dual block is exactly twice a single-stream one in parameters at the same
FLOPs, so a hybrid must cost exactly the sum of its halves, and it must not
allocate a third shared modulation core.

``model.final_block_text="drop"`` must delete precisely the parameters that
autograd never reaches with ``"keep"`` -- the final block's text output path --
and nothing else, which is what lets DDP stop paying for
``find_unused_parameters``.
"""

import pytest
import torch

from iris3b.models import IrisDiT
from iris3b.models.blocks.mmdit import MMDiTBlock
from iris3b.models.blocks.single_stream import SingleStreamBlock
from tiny_config import tiny_model_config

DEPTH = 4
DUAL = 2
IMAGE = 32

# The final block's text output path: the six parameters that receive no
# gradient at all, plus the two sandwich norms and the content gate when those
# variants are on.
TAIL = {
    "norm_y2.weight",
    "attn.proj_y.weight",
    "attn.proj_y.bias",
    "mlp_y.w1.weight",
    "mlp_y.w3.weight",
    "mlp_y.w2.weight",
}
VARIANT_TAIL = {"attn_post_norm_y.weight", "mlp_post_norm_y.weight", "attn_gate_y.weight"}


def cfg(**overrides):
    return tiny_model_config(depth=DEPTH, **overrides)


def seeded(factory):
    torch.manual_seed(1234)
    return factory()


def meta_model(model_cfg) -> IrisDiT:
    with torch.device("meta"):
        return IrisDiT(model_cfg)


def forward_inputs(model_cfg, batch: int = 2):
    torch.manual_seed(7)
    x = torch.randn(batch, model_cfg.in_channels, IMAGE, IMAGE)
    t = torch.rand(batch) * 1000.0
    y = torch.randn(batch, model_cfg.text_len, model_cfg.text_dim)
    return x, t, y


def n_patches(model_cfg) -> int:
    return (IMAGE // model_cfg.patch_size) ** 2


def dezero(model: IrisDiT) -> None:
    """Give every all-zero tensor a value.

    The output head is zero-initialized, so an untouched model returns zeros and
    every trunk gradient is zero -- which would make a gradient-coverage or
    output-equality assertion vacuous.
    """
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.any():
                parameter.normal_(std=0.02)


def matched_pair(block: str, **overrides) -> tuple[IrisDiT, IrisDiT]:
    """A keep/drop model pair sharing every parameter the drop model still has."""
    keep = seeded(lambda: IrisDiT(cfg(block=block, **overrides)).eval())
    dezero(keep)
    drop = IrisDiT(cfg(block=block, final_block_text="drop", **overrides)).eval()
    shared = {k: v for k, v in keep.state_dict().items() if k in drop.state_dict()}
    assert set(shared) == set(drop.state_dict())
    drop.load_state_dict(shared)
    return keep, drop


def backward_report(model: IrisDiT, **forward_kwargs) -> tuple[list[str], list[str]]:
    """(parameters autograd never reached, parameters reached with a zero gradient)."""
    x, t, y = forward_inputs(model.cfg)
    model(x, t, y, **forward_kwargs).x.square().mean().backward()
    named = list(model.named_parameters())
    return (
        sorted(name for name, p in named if p.grad is None),
        sorted(name for name, p in named if p.grad is not None and not p.grad.any()),
    )


def in_last_block(names: list[str]) -> list[str]:
    return [name for name in names if name.startswith(f"blocks.{DEPTH - 1}.")]


# -- hybrid stack -------------------------------------------------------------


def test_hybrid_stacks_both_families_and_trains():
    model = IrisDiT(cfg(block="single_stream", dual_depth=DUAL))
    assert [type(b) for b in model.blocks] == [MMDiTBlock] * DUAL + [SingleStreamBlock] * (DEPTH - DUAL)

    dezero(model)
    x, t, y = forward_inputs(model.cfg)
    out = model(x, t, y, capture=(DUAL, DEPTH))
    assert out.x.shape == x.shape
    # REPA capture works across the family boundary in both directions
    assert sorted(out.features) == [DUAL, DEPTH]
    assert all(f.shape == (2, n_patches(model.cfg), model.cfg.hidden_size) for f in out.features.values())

    out.x.square().mean().backward()
    assert not [name for name, p in model.named_parameters() if p.grad is None]


def test_hybrid_parameters_are_the_sum_of_its_halves():
    dual_stack = meta_model(cfg(block="mmdit"))
    single_stack = meta_model(cfg(block="single_stream"))
    hybrid = meta_model(cfg(block="single_stream", dual_depth=DUAL))

    dual_block = sum(p.numel() for p in dual_stack.blocks[0].parameters())
    single_block = sum(p.numel() for p in single_stack.blocks[0].parameters())
    # the topology axis is pure weight sharing: 2x the parameters, same FLOPs
    assert dual_block == 2 * single_block

    shell = dual_stack.num_parameters - DEPTH * dual_block
    assert shell == single_stack.num_parameters - DEPTH * single_block
    assert hybrid.num_parameters == shell + DUAL * dual_block + (DEPTH - DUAL) * single_block
    assert 2 * hybrid.num_parameters == dual_stack.num_parameters + single_stack.num_parameters

    # dual_depth == depth is the all-dual stack whatever `block` says
    assert meta_model(cfg(block="single_stream", dual_depth=DEPTH)).num_parameters == (
        dual_stack.num_parameters
    )


@pytest.mark.parametrize("modulation", ["shared_bias", "shared_lowrank"])
def test_hybrid_reuses_the_image_core_instead_of_adding_a_third(modulation):
    model = IrisDiT(cfg(block="single_stream", dual_depth=DUAL, modulation=modulation))
    cores = model.modulation_cores
    assert sorted(cores) == ["adaln_img", "adaln_txt"]
    for block in model.blocks[:DUAL]:
        assert block.adaln_img.core is cores["adaln_img"]
        assert block.adaln_txt.core is cores["adaln_txt"]
    for block in model.blocks[DUAL:]:
        assert block.adaln.core is cores["adaln_img"]

    dim = model.cfg.hidden_size
    assert sum(p.numel() for p in cores.parameters()) == 2 * (dim * 6 * dim + 6 * dim)


# -- final-block text tail ----------------------------------------------------


@pytest.mark.parametrize(
    "overrides, extra",
    [
        ({}, set()),
        ({"sandwich_norm": True, "gated_attention": True}, VARIANT_TAIL),
        ({"num_kv_heads": 2}, set()),  # GQA keeps the y-stream K/V that feed the image half
    ],
)
def test_drop_removes_exactly_the_text_tail(overrides, extra):
    keep = IrisDiT(cfg(**overrides))
    drop = IrisDiT(cfg(final_block_text="drop", **overrides))
    removed = {f"blocks.{DEPTH - 1}.{name}" for name in TAIL | extra}

    assert set(keep.state_dict()) - set(drop.state_dict()) == removed
    assert set(drop.state_dict()) - set(keep.state_dict()) == set()
    assert keep.num_parameters - drop.num_parameters == sum(keep.state_dict()[key].numel() for key in removed)
    # only the last block loses anything
    assert all(
        len(list(k.parameters())) == len(list(d.parameters()))
        for k, d in zip(keep.blocks[:-1], drop.blocks[:-1], strict=True)
    )


def test_drop_is_parameter_neutral_for_a_single_stream_tail():
    """Text and image share every weight there, so nothing can be removed."""
    keep = IrisDiT(cfg(block="single_stream"))
    drop = IrisDiT(cfg(block="single_stream", final_block_text="drop"))
    assert list(keep.state_dict()) == list(drop.state_dict())
    assert keep.num_parameters == drop.num_parameters


@pytest.mark.parametrize("block", ["mmdit", "single_stream"])
def test_drop_reproduces_the_kept_image_output(block):
    """The discarded text half cannot influence the image output, so at matched
    weights the cheaper model must return the same velocity."""
    keep, drop = matched_pair(block)
    x, t, y = forward_inputs(keep.cfg)
    with torch.no_grad():
        reference = keep(x, t, y).x
        assert torch.allclose(drop(x, t, y).x, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("overrides", [{}, {"sandwich_norm": True, "gated_attention": True}])
def test_keep_starves_the_tail_and_drop_leaves_no_dead_parameter(overrides):
    keep, drop = matched_pair("mmdit", **overrides)
    extra = VARIANT_TAIL if overrides else set()

    keep_missing, keep_zero = backward_report(keep)
    assert keep_missing == sorted(f"blocks.{DEPTH - 1}.{name}" for name in TAIL | extra)

    drop_missing, drop_zero = backward_report(drop)
    assert drop_missing == []

    # Dropping the tail introduces no new dead weight anywhere. What stays dead
    # in both is the final block's text *query* path: those attention rows are
    # computed and then discarded, so its gradient arrives as zeros rather than
    # not at all. Removing that too would need a keys/values-only text
    # projection surface, which this knob deliberately does not touch.
    assert drop_zero == keep_zero
    assert in_last_block(drop_zero) == [f"blocks.{DEPTH - 1}.attn.q_norm_y.weight"]


def test_single_stream_drop_has_no_dead_parameter_in_the_tail():
    """Nothing in a shared-weight block can go dead: every weight serves both
    streams, so the image half keeps all of them alive."""
    keep, drop = matched_pair("single_stream")
    for missing, zeroed in (backward_report(keep), backward_report(drop)):
        assert missing == []
        assert in_last_block(zeroed) == []


def test_hybrid_with_a_dropped_dual_tail_trains():
    model = IrisDiT(cfg(block="single_stream", dual_depth=DEPTH, final_block_text="drop"))
    dezero(model)
    missing, zeroed = backward_report(model)
    assert missing == []
    assert in_last_block(zeroed) == [f"blocks.{DEPTH - 1}.attn.q_norm_y.weight"]


# -- validation ---------------------------------------------------------------


@pytest.mark.parametrize("dual_depth", [-1, DEPTH + 1])
def test_dual_depth_bounds_are_validated(dual_depth):
    with pytest.raises(ValueError, match=r"model\.dual_depth must be in \[0, model\.depth=4\]"):
        IrisDiT(cfg(dual_depth=dual_depth))


def test_final_block_text_enum_is_validated():
    with pytest.raises(ValueError, match="model.final_block_text"):
        IrisDiT(cfg(final_block_text="none"))
