"""Checkpoint IO: mode resolution, the shared-directory probe, layout refusal,
retention, and the sharded completion marker.

Nothing here initializes a process group. The plan is a pure function of the
launcher's placement and the probe takes its barrier as an argument, so a
multi-node collective is reproduced with threads and one real directory per
simulated node.
"""

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from iris3b.config import (
    Config,
    DistConfig,
    OptimizerConfig,
    TrainConfig,
)
from iris3b.data.samplers import RangedSampler, ShapeBatchSampler
from iris3b.train.ckpt import (
    CHECKPOINT_IO_MODES,
    DCP_META,
    DCP_SUFFIX,
    PendingCheckpoint,
    checkpoint_plan,
    clone_shards,
    gather_full_state,
    probe_shared_checkpoint_dir,
    prune_checkpoints,
    require_checkpoint_kind,
    require_dense_weights,
    resolve_checkpoint_io,
    resolve_resume,
    save_checkpoint,
)
from iris3b.train.trainer import (
    _agree_batch_count,
    _checkpoint_data_position,
    _effective_batch,
    _load_fsdp2_optimizer_state,
    _resolve_resume_data_position,
)

# ---- mode resolution and writer election ----


def test_auto_resolves_rank0_on_one_node_and_per_node_beyond():
    assert resolve_checkpoint_io("auto", 1) == "rank0"
    assert resolve_checkpoint_io("auto", 2) == "per_node"
    assert resolve_checkpoint_io("auto", 8) == "per_node"
    for mode in ("rank0", "per_node", "dcp"):  # explicit modes are never rewritten
        assert resolve_checkpoint_io(mode, 1) == mode
        assert resolve_checkpoint_io(mode, 4) == mode
    with pytest.raises(ValueError, match="unknown train.dist.checkpoint_io"):
        resolve_checkpoint_io("shared", 1)


def test_single_node_plan_writes_from_rank_zero_without_a_sharing_requirement():
    plan = checkpoint_plan("auto", rank=0, local_rank=0, node_rank=0, num_nodes=1)
    assert plan.mode == "rank0"
    assert plan.writes
    assert not plan.needs_shared_dir
    assert not plan.sharded_layout
    assert plan.tmp_tag == ""


def test_per_node_elects_one_writer_per_node_with_distinct_scratch_files():
    plans = [
        checkpoint_plan("auto", rank=rank, local_rank=rank % 8, node_rank=rank // 8, num_nodes=2)
        for rank in range(16)
    ]
    assert {plan.mode for plan in plans} == {"per_node"}
    assert [plan.rank for plan in plans if plan.writes] == [0, 8]
    assert {plan.tmp_tag for plan in plans if plan.writes} == {".n0", ".n1"}
    # every rank reads its own node's copy, so nothing has to be shared
    assert not any(plan.needs_shared_dir for plan in plans)


def test_rank0_and_dcp_declare_a_sharing_requirement_beyond_one_node():
    rank0 = checkpoint_plan("rank0", rank=8, local_rank=0, node_rank=1, num_nodes=2)
    assert rank0.needs_shared_dir
    assert not rank0.writes  # only global rank 0 writes
    assert not checkpoint_plan("rank0", rank=0, local_rank=0, node_rank=0, num_nodes=1).needs_shared_dir

    sharded = checkpoint_plan("dcp", rank=3, local_rank=3, node_rank=0, num_nodes=2)
    assert sharded.sharded_layout
    assert sharded.needs_shared_dir
    assert not sharded.writes  # every rank writes a shard; rank 0 marks it done


def test_config_gate_and_resolver_accept_the_same_modes():
    """The gate in DistConfig.validate and the resolver must not drift apart."""
    for mode in CHECKPOINT_IO_MODES:
        train = TrainConfig(
            optimizer=OptimizerConfig(name="adamw"),
            dist=DistConfig(strategy="fsdp2", checkpoint_io=mode),
        )
        train.dist.validate(train)
        assert resolve_checkpoint_io(mode, 2) in ("rank0", "per_node", "dcp")
    rejected = TrainConfig(optimizer=OptimizerConfig(name="adamw"), dist=DistConfig(strategy="fsdp2"))
    rejected.dist.checkpoint_io = "shared"
    with pytest.raises(ValueError, match="checkpoint_io"):
        rejected.dist.validate(rejected)


def test_resume_data_policy_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="exact or new_phase"):
        TrainConfig(resume_data_policy="continue").validate_resume_data()


def test_exact_resume_restores_the_saved_position():
    cfg = Config()
    sampler = RangedSampler(6400, rank=0, world_size=64)
    position = {
        **sampler.resume_state(0, cfg.train.batch_size),
        "epoch": 1,
        "batches_consumed": 0,
    }

    step, restored = _resolve_resume_data_position(
        cfg,
        sampler,
        start_step=25,
        checkpoint_epoch=1,
        resumed_data_position=position,
        saved_config=None,
        effective_batch=cfg.train.batch_size * 64,
        steps_per_epoch=100,
        log=lambda _: None,
    )

    assert step == 25
    assert restored == position


def test_new_phase_keeps_checkpoint_step_and_starts_current_dataset_at_zero():
    cfg = Config(train=TrainConfig(batch_size=16, resume_data_policy="new_phase"))
    sampler = RangedSampler(6400, rank=0, world_size=64)
    messages = []

    step, position = _resolve_resume_data_position(
        cfg,
        sampler,
        start_step=22000,
        checkpoint_epoch=0,
        resumed_data_position={"epoch": 3, "batches_consumed": 7},
        saved_config=None,
        effective_batch=1024,
        steps_per_epoch=100,
        log=messages.append,
    )

    assert step == 22000
    assert position["epoch"] == 1
    assert position["batches_consumed"] == 0
    assert position["samples_per_rank"] == 0
    assert position["world_size"] == 64
    sampler.load_resume_state(position)
    assert list(sampler) == list(range(100))
    assert "starting the current dataset at epoch 1 with zero consumed batches" in messages[0]


def test_trainer_checkpoint_keeps_mixed_microbatch_progress_distinct():
    keys = ["a", "b"] * 32
    dataset = SimpleNamespace(key_of=keys.__getitem__)
    source = ShapeBatchSampler(RangedSampler(64), dataset, batch_size=4)
    position = _checkpoint_data_position(
        source,
        4,
        epoch=1,
        batches=3,
        plan_batches=3,
        plan_samples=12,
    )
    cfg = Config(train=TrainConfig(batch_size=8, grad_accum_steps=1))
    current = ShapeBatchSampler(RangedSampler(64), dataset, batch_size=8)
    step, position = _resolve_resume_data_position(
        cfg,
        current,
        start_step=1000,
        checkpoint_epoch=1,
        resumed_data_position=position,
        saved_config=None,
        effective_batch=8,
        steps_per_epoch=8,
        log=lambda _: None,
    )
    current.set_epoch(1)
    current.load_resume_state(position)
    expected = [
        list(range(9, 24, 2)),
        list(range(16, 31, 2)),
        list(range(25, 40, 2)),
        list(range(32, 47, 2)),
        list(range(41, 56, 2)),
        list(range(48, 63, 2)),
    ]
    assert step == 1000
    assert list(current) == expected
    saved_again = _checkpoint_data_position(
        current,
        8,
        epoch=1,
        batches=position["batches_consumed"] + 2,
        plan_batches=2,
        plan_samples=16,
    )
    saved_again.pop("batches_consumed")
    resumed = ShapeBatchSampler(RangedSampler(64), dataset, batch_size=8)
    _, saved_again = _resolve_resume_data_position(
        cfg,
        resumed,
        start_step=1002,
        checkpoint_epoch=1,
        resumed_data_position=saved_again,
        saved_config=None,
        effective_batch=8,
        steps_per_epoch=8,
        log=lambda _: None,
    )
    assert saved_again["batches_consumed"] == 5
    resumed.load_resume_state(saved_again)
    assert list(resumed) == expected[2:]


def test_agreed_shape_limit_applies_to_the_resumed_remainder_and_exhausted_epoch(monkeypatch):
    keys = ["a"] * 32
    dataset = SimpleNamespace(key_of=keys.__getitem__)
    source = ShapeBatchSampler(RangedSampler(32, world_size=2), dataset, batch_size=2)
    migrated = ShapeBatchSampler(RangedSampler(32, world_size=2), dataset, batch_size=4)
    migrated.load_resume_state(source.resume_state(1, 2))
    expected = list(migrated)
    accelerator = SimpleNamespace(num_processes=2, device="cpu", print=lambda _: None)
    agreed = 2
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value, op: value.fill_(agreed))
    _agree_batch_count(accelerator, migrated, migrated, allow_empty=True)
    assert list(migrated) == expected[:2]

    saved = migrated.resume_state(1, 4)
    resumed = ShapeBatchSampler(RangedSampler(32, world_size=2), dataset, batch_size=4)
    resumed.load_resume_state(saved)
    agreed = 1
    _agree_batch_count(accelerator, resumed, resumed, allow_empty=True)
    assert len(resumed) == 1
    assert list(resumed) == expected[1:2]

    exhausted = resumed.resume_state(1, 4)
    resumed.load_resume_state(exhausted)
    agreed = 0
    _agree_batch_count(accelerator, resumed, resumed, allow_empty=True)
    assert len(resumed) == 0
    assert list(resumed) == []

    resumed.set_epoch(2)
    agreed = 4
    _agree_batch_count(accelerator, resumed, resumed)
    assert list(resumed) == [list(range(start, start + 4)) for start in range(0, 16, 4)]


# ---- the startup shared-directory probe ----


def _probe_ranks(plans, dirs) -> dict[int, BaseException | None]:
    """Run the probe on every rank against its node's directory.

    ``dirs`` maps node rank to a directory: one entry for a shared mount, one
    per node for private disks.
    """
    gate = threading.Barrier(len(plans))
    outcome: dict[int, BaseException | None] = {}

    def one(plan):
        try:
            probe_shared_checkpoint_dir(dirs[plan.node_rank], plan, gate.wait)
            outcome[plan.rank] = None
        except BaseException as exc:  # a failed rank must not hang its peers
            outcome[plan.rank] = exc
            gate.abort()

    threads = [threading.Thread(target=one, args=(plan,), name=f"rank{plan.rank}") for plan in plans]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return outcome


def _two_node_plans(mode: str = "rank0"):
    return [checkpoint_plan(mode, rank=rank, local_rank=0, node_rank=rank, num_nodes=2) for rank in (0, 1)]


def test_probe_passes_on_a_shared_directory_and_leaves_nothing_behind(tmp_path):
    shared = tmp_path / "checkpoints"
    outcome = _probe_ranks(_two_node_plans(), {0: shared, 1: shared})
    assert outcome == {0: None, 1: None}
    assert list(shared.iterdir()) == []


def test_probe_is_skipped_when_no_rank_depends_on_sharing(tmp_path):
    shared = tmp_path / "checkpoints"
    # per_node writes a copy per node, and a single-node run has nothing to share
    for plans in (
        _two_node_plans("per_node"),
        [checkpoint_plan("rank0", rank=0, local_rank=0, node_rank=0, num_nodes=1)],
    ):
        assert set(_probe_ranks(plans, {0: shared, 1: shared}).values()) == {None}
    assert not shared.exists()


def test_probe_fails_on_private_per_node_directories(tmp_path):
    """The real misconfiguration: work_dir on each node's local disk."""
    dirs = {node: tmp_path / f"node{node}" / "checkpoints" for node in (0, 1)}
    outcome = _probe_ranks(_two_node_plans(), dirs)
    failure = outcome[1]
    assert isinstance(failure, RuntimeError)
    message = str(failure)
    assert "checkpoint_io=rank0" in message
    assert "rank 1 cannot read it back" in message
    assert "train.dist.checkpoint_io=per_node" in message


def test_probe_is_not_fooled_by_a_stale_sentinel_on_a_private_disk(tmp_path):
    """Every rank clears its own sentinel first, so a leftover from an earlier
    run cannot make a private directory look shared."""
    dirs = {node: tmp_path / f"node{node}" / "checkpoints" for node in (0, 1)}
    stale = dirs[1] / ".shared_ckpt_probe"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"iris shared checkpoint probe\n")
    outcome = _probe_ranks(_two_node_plans(), dirs)
    assert isinstance(outcome[1], RuntimeError)
    assert not stale.exists()


def test_probe_fails_when_a_shared_path_cannot_be_read(tmp_path, monkeypatch):
    """A mount that exists but is not readable from every rank fails the same
    way as a missing one."""
    real_read = Path.read_bytes

    def read(self):
        if threading.current_thread().name != "rank0":
            raise OSError(5, "Input/output error", str(self))
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", read)
    shared = tmp_path / "checkpoints"
    outcome = _probe_ranks(_two_node_plans("dcp"), {0: shared, 1: shared})
    failure = outcome[1]
    assert isinstance(failure, RuntimeError)
    assert "checkpoint_io=dcp" in str(failure)
    assert "Input/output error" in str(failure)
    # dcp cannot fall back to per-node copies, so the fix is the mount
    assert "point cfg.work_dir at a mount every node sees" in str(failure)


def test_probe_rejects_a_sentinel_that_does_not_match(tmp_path, monkeypatch):
    """A shared path serving different bytes to different ranks still fails.

    Only the non-writing rank is corrupted: rank 0 verifies its own write, so
    patching every rank would abort the barrier before the comparison runs.
    """
    real_read = Path.read_bytes

    def read(self):
        if threading.current_thread().name == "rank0":
            return real_read(self)
        return b"someone else's file"

    monkeypatch.setattr(Path, "read_bytes", read)
    outcome = _probe_ranks(_two_node_plans(), {0: tmp_path, 1: tmp_path})
    assert isinstance(outcome[1], RuntimeError)
    assert "sentinel content differs" in str(outcome[1])


# ---- layout refusal ----


def test_sharded_and_single_file_layouts_are_not_interchangeable(tmp_path):
    single = tmp_path / "epoch_1_step_100.pth"
    sharded = tmp_path / f"epoch_1_step_100{DCP_SUFFIX}"

    with pytest.raises(ValueError, match="single-file .pth checkpoint"):
        require_checkpoint_kind(single, "dcp")
    for mode in ("rank0", "per_node"):
        with pytest.raises(ValueError, match="sharded .dcp. checkpoint directory"):
            require_checkpoint_kind(sharded, mode)

    require_checkpoint_kind(single, "rank0")  # matching layouts pass
    require_checkpoint_kind(single, "per_node")
    require_checkpoint_kind(sharded, "dcp")


def test_load_from_always_reads_a_single_file(tmp_path):
    require_dense_weights(tmp_path / "epoch_1_step_100.pth")
    with pytest.raises(ValueError, match="train.load_from"):
        require_dense_weights(tmp_path / f"epoch_1_step_100{DCP_SUFFIX}")


# ---- retention ----


def _write_pth(ckpt_dir: Path, epoch: int, step: int) -> Path:
    path = ckpt_dir / f"epoch_{epoch}_step_{step}.pth"
    path.write_bytes(b"payload")
    return path


def _write_dcp(ckpt_dir: Path, epoch: int, step: int, complete: bool = True) -> Path:
    path = ckpt_dir / f"epoch_{epoch}_step_{step}{DCP_SUFFIX}"
    path.mkdir(parents=True)
    (path / "__0_0.distcp").write_bytes(b"shard")
    if complete:
        (path / DCP_META).write_bytes(b"meta")
    return path


def test_keep_last_zero_keeps_everything(tmp_path):
    made = [_write_pth(tmp_path, epoch, epoch * 100) for epoch in range(1, 6)]
    assert prune_checkpoints(tmp_path, 0) == []
    assert prune_checkpoints(tmp_path, -1) == []
    assert all(path.exists() for path in made)


def test_retention_keeps_the_newest_and_never_prunes_a_milestone(tmp_path):
    made = {step: _write_pth(tmp_path, i + 1, step) for i, step in enumerate((100, 200, 300, 400, 500))}
    deleted = prune_checkpoints(tmp_path, 2, milestones=[200])
    assert sorted(path.name for path in deleted) == [
        "epoch_1_step_100.pth",
        "epoch_3_step_300.pth",
    ]
    assert {step for step, path in made.items() if path.exists()} == {200, 400, 500}


def test_retention_removes_whole_sharded_directories(tmp_path):
    dirs = [_write_dcp(tmp_path, i + 1, step) for i, step in enumerate((100, 200, 300))]
    deleted = prune_checkpoints(tmp_path, 1, milestones=[100])
    assert [path.name for path in deleted] == [f"epoch_2_step_200{DCP_SUFFIX}"]
    assert dirs[0].exists() and not dirs[1].exists() and dirs[2].exists()


# ---- resume resolution ----


def test_latest_link_wins(tmp_path):
    older = _write_pth(tmp_path, 1, 100)
    _write_pth(tmp_path, 2, 200)
    (tmp_path / "latest.pth").symlink_to(older)
    assert resolve_resume(tmp_path) == str(older.resolve())


def test_highest_step_wins_without_a_link(tmp_path):
    _write_pth(tmp_path, 1, 100)
    newest = _write_pth(tmp_path, 2, 200)
    assert resolve_resume(tmp_path) == str(newest)


def test_resolve_resume_skips_a_sharded_save_that_never_finished(tmp_path):
    done = _write_dcp(tmp_path, 1, 100)
    partial = _write_dcp(tmp_path, 2, 200, complete=False)
    assert resolve_resume(tmp_path) == str(done)
    (partial / DCP_META).write_bytes(b"meta")
    assert resolve_resume(tmp_path) == str(partial)


def test_resolve_resume_is_none_on_an_empty_directory(tmp_path):
    assert resolve_resume(tmp_path) is None


# ---- writers ----


def test_per_node_writers_do_not_share_a_scratch_file(tmp_path, monkeypatch):
    """Two node writers whose mount turns out to be shared must not write the
    same .tmp path, or the atomic rename publishes an interleaving."""
    written: list[str] = []
    real_save = torch.save

    def record(payload, path, *args, **kwargs):
        written.append(Path(path).name)
        return real_save(payload, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", record)
    for node in (0, 1):
        plan = checkpoint_plan("per_node", rank=node * 8, local_rank=0, node_rank=node, num_nodes=2)
        save_checkpoint(
            tmp_path / "epoch_1_step_100.pth",
            {"w": torch.zeros(2)},
            epoch=1,
            step=100,
            tmp_tag=plan.tmp_tag,
        )
    assert written == ["epoch_1_step_100.pth.tmp.n0", "epoch_1_step_100.pth.tmp.n1"]
    published = tmp_path / "epoch_1_step_100.pth"
    assert published.exists()
    assert (tmp_path / "latest.pth").resolve() == published.resolve()
    assert not list(tmp_path.glob("*.tmp*"))


def test_sharded_marker_needs_an_explicit_finalize_after_the_barrier(tmp_path):
    """``wait()`` only proves this rank's shards landed; the marker is what
    makes the directory resumable, so it waits for the caller's barrier."""
    path = _write_dcp(tmp_path, 1, 100, complete=False)
    pending = PendingCheckpoint(path=path, meta={"step": 100}, write_meta=True)
    pending.wait()
    assert not (path / DCP_META).exists()
    assert resolve_resume(tmp_path) is None  # not a resume candidate yet

    pending.finalize()
    assert torch.load(path / DCP_META, weights_only=False)["step"] == 100
    assert (tmp_path / f"latest{DCP_SUFFIX}").resolve() == path.resolve()
    assert Path(resolve_resume(tmp_path)) == path.resolve()

    (path / DCP_META).unlink()
    pending.finalize()  # idempotent: published once per save
    assert not (path / DCP_META).exists()


def test_sharded_non_writers_publish_nothing(tmp_path):
    path = _write_dcp(tmp_path, 1, 100, complete=False)
    non_writer = PendingCheckpoint(path=path, meta={"step": 100}, write_meta=False)
    non_writer.wait()
    non_writer.finalize()
    assert not (path / DCP_META).exists()
    assert not (tmp_path / f"latest{DCP_SUFFIX}").exists()


# ---- consolidation helpers ----


def test_clone_shards_detaches_from_live_storage():
    live = torch.ones(3)
    captured = clone_shards({"w": live, "step": 7})
    live.mul_(0)  # the EMA swap unwinding after the capture
    assert torch.equal(captured["w"], torch.ones(3))
    assert captured["step"] == 7


def test_gather_full_state_walks_the_optimizer_structure():
    state = {
        "state": {"model.w": {"exp_avg": torch.ones(2), "step": torch.tensor(4)}},
        "param_groups": [{"lr": 0.1, "params": ["model.w"]}],
    }
    kept = gather_full_state(state, keep=True)
    assert torch.equal(kept["state"]["model.w"]["exp_avg"], torch.ones(2))
    assert kept["state"]["model.w"]["exp_avg"].device.type == "cpu"
    assert kept["param_groups"] == [{"lr": 0.1, "params": ["model.w"]}]
    dropped = gather_full_state(state, keep=False)
    assert dropped["state"]["model.w"]["exp_avg"] is None  # non-writers keep nothing
    assert dropped["param_groups"][0]["lr"] == 0.1


# ---- fsdp2 optimizer resume after REPA is switched off ----


class _FakeOptimizer:
    """Just the two attributes the positional FQN loader touches."""

    def __init__(self, groups: list[dict]):
        self.param_groups = groups
        self.state: dict = {}


def _routed_group(scope: str, route: str, algorithm: str, params, names, lr: float) -> dict:
    return {
        "params": list(params),
        "algorithm": algorithm,
        "lr": lr,
        "iris_scope": scope,
        "iris_route": route,
        "iris_param_names": tuple(names),
    }


def _repa_on_checkpoint() -> dict:
    """What get_optimizer_state_dict wrote for a REPA-on run: FQN-keyed state,
    routed groups, and REPA's projector alone in the trailing aux group."""
    return {
        "state": {
            "model.w": {"momentum": torch.full((2, 2), 3.0)},
            "model.b": {"exp_avg": torch.full((2,), 5.0), "step": torch.tensor(7)},
            "repa.projector.0.weight": {"exp_avg": torch.ones(3)},
        },
        "param_groups": [
            _routed_group("core", "hidden_matrix", "muon", ["model.w"], ["model.w"], 0.1),
            _routed_group("core", "boundary_or_vector", "adamw", ["model.b"], ["model.b"], 0.1),
            _routed_group(
                "aux", "auxiliary", "adamw", ["repa.projector.0.weight"], ["repa.projector.0.weight"], 0.1
            ),
        ],
    }


def _core_params() -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    return torch.nn.Parameter(torch.zeros(2, 2)), torch.nn.Parameter(torch.zeros(2))


def test_fsdp2_resume_drops_the_checkpoint_aux_group_when_repa_is_off():
    """repa.weight=0 still builds the aux group, but empty: the checkpoint's
    projector state is forgotten and every core group loads positionally."""
    w, b = _core_params()
    optimizer = _FakeOptimizer(
        [
            _routed_group("core", "hidden_matrix", "muon", [w], ["model.w"], 0.5),
            _routed_group("core", "boundary_or_vector", "adamw", [b], ["model.b"], 0.5),
            _routed_group("aux", "auxiliary", "adamw", [], [], 0.5),
        ]
    )
    _load_fsdp2_optimizer_state(optimizer, _repa_on_checkpoint())
    assert set(map(id, optimizer.state)) == {id(w), id(b)}
    assert torch.equal(optimizer.state[w]["momentum"], torch.full((2, 2), 3.0))
    assert torch.equal(optimizer.state[b]["exp_avg"], torch.full((2,), 5.0))
    assert int(optimizer.state[b]["step"]) == 7
    assert [group["lr"] for group in optimizer.param_groups] == [0.1, 0.1, 0.1]
    assert optimizer.param_groups[2]["params"] == []


def test_fsdp2_resume_drops_the_aux_group_when_the_live_optimizer_has_none():
    w, b = _core_params()
    optimizer = _FakeOptimizer(
        [
            _routed_group("core", "hidden_matrix", "muon", [w], ["model.w"], 0.5),
            _routed_group("core", "boundary_or_vector", "adamw", [b], ["model.b"], 0.5),
        ]
    )
    _load_fsdp2_optimizer_state(optimizer, _repa_on_checkpoint())
    assert set(map(id, optimizer.state)) == {id(w), id(b)}
    assert len(optimizer.param_groups) == 2


def test_fsdp2_resume_still_rejects_a_changed_aux_group_when_repa_is_on():
    """Only an EMPTY live aux group may forget the checkpoint's; a live
    projector whose parameters differ is a real mismatch."""
    w, b = _core_params()
    projector = torch.nn.Parameter(torch.zeros(3))
    optimizer = _FakeOptimizer(
        [
            _routed_group("core", "hidden_matrix", "muon", [w], ["model.w"], 0.5),
            _routed_group("core", "boundary_or_vector", "adamw", [b], ["model.b"], 0.5),
            _routed_group("aux", "auxiliary", "adamw", [projector], ["repa.projector.2.weight"], 0.5),
        ]
    )
    with pytest.raises(ValueError, match="group 2 parameter names diverge"):
        _load_fsdp2_optimizer_state(optimizer, _repa_on_checkpoint())
    assert projector not in optimizer.state


def test_fsdp2_resume_still_rejects_a_core_mismatch_when_repa_is_off():
    """The aux drop must not loosen the positional contract for core groups."""
    w, b = _core_params()
    optimizer = _FakeOptimizer(
        [
            _routed_group("core", "hidden_matrix", "muon", [w], ["model.w"], 0.5),
            _routed_group("core", "boundary_or_vector", "adamw", [b], ["model.other"], 0.5),
            _routed_group("aux", "auxiliary", "adamw", [], [], 0.5),
        ]
    )
    with pytest.raises(ValueError, match="group 1 parameter names diverge"):
        _load_fsdp2_optimizer_state(optimizer, _repa_on_checkpoint())
    only_core = _FakeOptimizer([_routed_group("core", "hidden_matrix", "muon", [w], ["model.w"], 0.5)])
    with pytest.raises(ValueError, match="group count mismatch"):
        _load_fsdp2_optimizer_state(only_core, _repa_on_checkpoint())


# ---- pinned global batch ----


def test_effective_batch_is_the_product_when_no_pin_is_set():
    train = TrainConfig(batch_size=16, grad_accum_steps=2)
    assert _effective_batch(train, 24) == 768


def test_effective_batch_honours_the_pin_and_names_all_three_factors():
    train = TrainConfig(batch_size=16, grad_accum_steps=2, expected_global_batch=1024)
    assert _effective_batch(train, 32) == 1024
    with pytest.raises(ValueError, match=r"768 \(batch_size 16 x 24 ranks x grad_accum_steps 2\)"):
        _effective_batch(train, 24)
    with pytest.raises(ValueError, match="expected_global_batch=1024"):
        _effective_batch(train, 24)
    with pytest.raises(ValueError, match="must be positive"):
        _effective_batch(TrainConfig(expected_global_batch=0), 8)
