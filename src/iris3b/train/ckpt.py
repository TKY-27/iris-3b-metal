"""Checkpoint IO: single-file and sharded payloads, retention, resume resolution."""

import inspect
import random
import re
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

_CKPT_RE = re.compile(r"epoch_(\d+)_step_(\d+)\.(pth|dcp)$")

CHECKPOINT_IO_MODES = ("auto", "rank0", "per_node", "dcp")
DCP_SUFFIX = ".dcp"
# Written after the sharded payload lands, so a .dcp directory without it is
# the debris of an interrupted save and never a resume candidate.
DCP_META = "meta.pth"
_PROBE_NAME = ".shared_ckpt_probe"
_PROBE_TOKEN = b"iris shared checkpoint probe\n"

_async_notice_shown = False
_cpu_group = None
_cpu_group_resolved = False


def _rng_state() -> dict[str, Any]:
    # The cuda entry is one state per visible device, so it is node-local and
    # is not replayed onto a different device count. Benign either way: the
    # loop reseeds from (seed, epoch, batch index) before every batch, so a
    # restored generator is overwritten before anything draws from it.
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    cuda_states = state.get("cuda") or []
    if torch.cuda.is_available() and len(cuda_states) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda_states)
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])


# ---- checkpoint IO plan: layout, writer election, filesystem requirement ----


def resolve_checkpoint_io(mode: str, num_nodes: int) -> str:
    """Resolve ``train.dist.checkpoint_io``.

    ``auto`` never leaves the only copy of a checkpoint on one node's
    filesystem: beyond a single node it writes a full copy per node.
    """
    if mode not in CHECKPOINT_IO_MODES:
        raise ValueError(f"unknown train.dist.checkpoint_io '{mode}' ({' | '.join(CHECKPOINT_IO_MODES)})")
    if mode != "auto":
        return mode
    return "rank0" if num_nodes <= 1 else "per_node"


@dataclass(frozen=True)
class CheckpointPlan:
    """Who writes a checkpoint, in what layout, resolved once at startup."""

    mode: str  # rank0 | per_node | dcp (never "auto")
    rank: int
    local_rank: int
    node_rank: int
    num_nodes: int

    @property
    def writes(self) -> bool:
        """Ranks that write a single-file payload.

        Under ``dcp`` every rank writes its own shard and this elects the rank
        that publishes the completion marker instead.
        """
        return self.local_rank == 0 if self.mode == "per_node" else self.rank == 0

    @property
    def sharded_layout(self) -> bool:
        return self.mode == "dcp"

    @property
    def needs_shared_dir(self) -> bool:
        """One copy exists and only one node produced it."""
        return self.mode in ("rank0", "dcp") and self.num_nodes > 1

    @property
    def tmp_tag(self) -> str:
        """Per-writer scratch suffix.

        ``per_node`` elects one writer per node; if the directory turns out to
        be a shared mount they would otherwise write the same ``.tmp`` file and
        the atomic rename would publish an interleaving of both.
        """
        return f".n{self.node_rank}" if self.mode == "per_node" else ""

    def describe(self) -> str:
        detail = {
            "rank0": "one file written by global rank 0",
            "per_node": "one file per node written by local rank 0",
            "dcp": "sharded, every rank writes its own shard, written in the background",
        }[self.mode]
        return f"checkpoint_io={self.mode}: {detail} ({self.num_nodes} node(s))"


def checkpoint_plan(
    mode: str, *, rank: int, local_rank: int, node_rank: int, num_nodes: int
) -> CheckpointPlan:
    return CheckpointPlan(resolve_checkpoint_io(mode, num_nodes), rank, local_rank, node_rank, num_nodes)


def _probe_failure(ckpt_dir: str | Path, plan: CheckpointPlan, detail: str) -> str:
    fix = (
        "set train.dist.checkpoint_io=per_node (a full copy per node), or point cfg.work_dir "
        "at a mount every node sees"
        if plan.mode == "rank0"
        else "point cfg.work_dir at a mount every node sees, or set "
        "train.dist.checkpoint_io=per_node (a full copy per node)"
    )
    return (
        f"train.dist.checkpoint_io={plan.mode} keeps one copy of each checkpoint, written by "
        f"rank 0, but rank {plan.rank} cannot read it back from {ckpt_dir} ({detail}). That "
        f"directory is not shared across the {plan.num_nodes} nodes of this job, so the resume "
        f"would fail hours from now instead of here. Fix: {fix}."
    )


def probe_shared_checkpoint_dir(
    ckpt_dir: str | Path, plan: CheckpointPlan, barrier: Callable[[], None]
) -> None:
    """Prove every rank reads back what rank 0 wrote, before the first step.

    Every rank clears its own sentinel first, so a leftover copy on one node's
    private disk cannot make an unshared directory look shared.
    """
    if not plan.needs_shared_dir:
        return
    path = Path(ckpt_dir) / _PROBE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    barrier()
    if plan.rank == 0:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(_PROBE_TOKEN)
        tmp.replace(path)
    barrier()
    try:
        seen = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(_probe_failure(ckpt_dir, plan, f"{type(exc).__name__}: {exc}")) from exc
    if seen != _PROBE_TOKEN:
        raise RuntimeError(_probe_failure(ckpt_dir, plan, "sentinel content differs"))
    barrier()
    if plan.rank == 0:
        path.unlink(missing_ok=True)


def checkpoint_kind(path: str | Path) -> str:
    """``dcp`` for a sharded checkpoint directory, ``pth`` for a single file."""
    return "dcp" if Path(path).suffix == DCP_SUFFIX else "pth"


def require_checkpoint_kind(path: str | Path, mode: str) -> None:
    """Refuse a resume across layouts before ``torch.load`` fails obscurely."""
    kind = checkpoint_kind(path)
    if mode == "dcp" and kind != "dcp":
        raise ValueError(
            f"{path} is a single-file .pth checkpoint but train.dist.checkpoint_io=dcp reads a "
            "sharded checkpoint directory. Resume it with train.dist.checkpoint_io=rank0 or "
            "per_node, or pass it as train.load_from for a weights-only start."
        )
    if mode != "dcp" and kind == "dcp":
        raise ValueError(
            f"{path} is a sharded (dcp) checkpoint directory but train.dist.checkpoint_io="
            f"{mode} reads single-file .pth checkpoints. Set train.dist.checkpoint_io=dcp."
        )


def require_dense_weights(path: str | Path) -> None:
    """``train.load_from`` is a weights-only start and always reads one file."""
    if checkpoint_kind(path) == "dcp":
        raise ValueError(
            f"train.load_from={path} is a sharded (dcp) checkpoint directory; load_from reads a "
            "single-file .pth. Use train.resume_from with train.dist.checkpoint_io=dcp instead."
        )


# ---- single-file payloads ----


def _relink_latest(path: Path, tmp_tag: str = "") -> None:
    """Repoint ``latest.pth`` / ``latest.dcp`` at ``path``, atomically.

    The rename cannot be observed half-done, which matters when per-node
    writers land in the same directory because the mount is shared after all.
    """
    link = path.parent / f"latest{path.suffix}"
    tmp = link.with_name(link.name + f".tmp{tmp_tag}")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(path.resolve())
    tmp.replace(link)


def save_checkpoint(
    path: str | Path,
    model: nn.Module | dict,
    optimizer=None,
    scheduler=None,
    ema=None,
    *,
    epoch: int,
    step: int,
    config_dict: dict | None = None,
    extra_state: dict[str, Any] | None = None,
    tmp_tag: str = "",
) -> Path:
    """Write one ``.pth`` checkpoint and refresh the ``latest.pth`` symlink.

    The payload holds the model state dict plus, when given, EMA / optimizer /
    scheduler state, the config, any ``extra_state`` entries, and the epoch,
    step, and RNG state (torch, all cuda devices, numpy, python).

    ``model``, ``optimizer``, and ``ema`` each accept either a live object or
    an already-materialized state dict — a sharded run consolidates full dicts
    collectively before its elected writers call this. ``tmp_tag`` separates
    concurrent writers' scratch files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_sd = model if isinstance(model, dict) else model.state_dict()
    payload: dict[str, Any] = {"state_dict": model_sd, "epoch": epoch, "step": step}
    if ema is not None:
        payload["state_dict_ema"] = ema if isinstance(ema, dict) else ema.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer if isinstance(optimizer, dict) else optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if config_dict is not None:
        payload["config"] = config_dict
    if extra_state:
        payload.update(extra_state)
    payload["rng_state"] = _rng_state()
    # Write-to-temp + atomic rename: a crash mid-write can never leave a
    # truncated file where resume resolution would find it.
    tmp = path.with_name(path.name + f".tmp{tmp_tag}")
    torch.save(payload, tmp)
    tmp.replace(path)
    _relink_latest(path, tmp_tag)
    return path


def _optimizer_group_signature(group: dict) -> tuple:
    return (
        group.get("iris_scope"),
        group.get("algorithm"),
        group.get("iris_route"),
        tuple(group.get("split_sizes", ())),
        tuple(group.get("iris_param_names", ())),
        len(group.get("params", ())),
    )


def _validate_optimizer_layout(saved: dict, live_groups: list[dict]) -> None:
    saved_groups = saved.get("param_groups", [])
    saved_routed = any(group.get("iris_scope") for group in saved_groups)
    live_routed = any(group.get("iris_scope") for group in live_groups)
    if not saved_routed and not live_routed:
        return
    if not saved_routed or not live_routed:
        raise ValueError(
            "full optimizer resume cannot cross legacy and routed layouts; "
            "use train.load_from for a weights-only optimizer change"
        )
    if [_optimizer_group_signature(group) for group in saved_groups] != [
        _optimizer_group_signature(group) for group in live_groups
    ]:
        raise ValueError(
            "full optimizer resume requires the same ordered parameter routing; "
            "use train.load_from after optimizer or model-routing changes"
        )


def _core_optimizer_state(saved: dict, live_groups: list[dict], legacy_prefix: int) -> tuple[dict, int]:
    saved_groups = saved.get("param_groups", [])
    saved_core = [group for group in saved_groups if group.get("iris_scope") == "core"]
    live_core = [group for group in live_groups if group.get("iris_scope") == "core"]
    if not saved_core and not live_core:
        kept = {
            int(index): state for index, state in saved.get("state", {}).items() if int(index) < legacy_prefix
        }
        return {"state": kept, "param_groups": live_groups}, len(kept)
    if not saved_core or not live_core:
        raise ValueError(
            "optimizer core resume cannot cross legacy and routed optimizer layouts; "
            "use train.load_from for a weights-only optimizer change"
        )

    def signature(group: dict) -> tuple:
        return _optimizer_group_signature(group)

    saved_signature = [signature(group) for group in saved_core]
    live_signature = [signature(group) for group in live_core]
    if saved_signature != live_signature:
        raise ValueError(
            "optimizer core resume requires the same core routing; "
            "use train.load_from for a weights-only optimizer change"
        )
    core_ids = {int(index) for group in saved_core for index in group.get("params", ())}
    kept = {int(index): state for index, state in saved.get("state", {}).items() if int(index) in core_ids}
    return {"state": kept, "param_groups": live_groups}, len(kept)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    ema=None,
    *,
    weights_only_load: bool = False,
    extra_modules: dict[str, nn.Module] | None = None,
    optimizer_prefix: int | None = None,
    extra_out: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Load a checkpoint into live objects.

    Model weights load with ``strict=False`` and a printed report of missing
    and unexpected keys. ``extra_modules`` maps payload keys to modules whose
    state loads whenever the key is present (weights count as weights). With
    ``weights_only_load`` nothing else is touched; otherwise optimizer,
    scheduler, EMA, and RNG state are restored. An EMA passed without a saved
    EMA state is seeded from the freshly loaded model.
    ``optimizer_prefix`` restricts the optimizer load to shared-core state,
    rebuilding groups from the live optimizer for resumes across runs whose
    auxiliary heads differ. Legacy single-group optimizers use the flat prefix;
    routed optimizers persist explicit core-group metadata.

    Returns:
        (epoch, step) recorded in the checkpoint.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[ckpt] missing keys ({len(missing)}): {sorted(missing)}")
    if unexpected:
        print(f"[ckpt] unexpected keys ({len(unexpected)}): {sorted(unexpected)}")
    for name, module in (extra_modules or {}).items():
        if name in payload:
            module.load_state_dict(payload[name])
    if not weights_only_load:
        if optimizer is not None and "optimizer" in payload:
            opt_payload = payload["optimizer"]
            live_groups = optimizer.state_dict()["param_groups"]
            if optimizer_prefix is not None:
                # A core-only resume keeps only shared-core state, takes all
                # groups/hyperparameters from the live optimizer, and leaves
                # new auxiliary parameters stateless until their first step.
                total = len(opt_payload.get("state", {}))
                opt_payload, kept = _core_optimizer_state(opt_payload, live_groups, optimizer_prefix)
                print(f"[ckpt] optimizer core load: kept {kept} param states, dropped {total - kept}")
            else:
                _validate_optimizer_layout(opt_payload, live_groups)
            optimizer.load_state_dict(opt_payload)
        if scheduler is not None and "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        if ema is not None:
            if "state_dict_ema" in payload:
                ema.load_state_dict(payload["state_dict_ema"])
            else:
                ema.load_state_dict(model.state_dict())
        if "rng_state" in payload:
            _restore_rng(payload["rng_state"])
    if extra_out is not None and "data_position" in payload:
        extra_out["data_position"] = payload["data_position"]
    return int(payload.get("epoch", 0)), int(payload.get("step", 0))


# ---- sharded state: consolidation helpers ----


def clone_shards(state_dict: dict) -> dict:
    """Copy shard storage out of the live parameters.

    An EMA state dict is captured while the averaged weights are temporarily
    swapped into the model; without this the payload would alias parameters
    that are restored the moment the swap unwinds.
    """
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in state_dict.items()
    }


def gather_full_state(state, keep: bool):
    """All-gather a per-parameter sharded state dict (DTensor) to CPU.

    One tensor at a time, so the transient full copy is a single parameter
    rather than the whole model. The gather is collective: every rank walks the
    same structure, and ranks that do not write drop their copy immediately.
    """
    from torch.distributed.tensor import DTensor

    if isinstance(state, DTensor):
        gathered = state.full_tensor()
        return gathered.detach().to("cpu", copy=True) if keep else None
    if isinstance(state, torch.Tensor):
        return state.detach().to("cpu", copy=True) if keep else None
    if isinstance(state, dict):
        return {key: gather_full_state(value, keep) for key, value in state.items()}
    if isinstance(state, list):
        return [gather_full_state(value, keep) for value in state]
    if isinstance(state, tuple):
        return tuple(gather_full_state(value, keep) for value in state)
    return state


# ---- sharded (dcp) payloads ----


@dataclass
class PendingCheckpoint:
    """A sharded save in flight.

    ``wait()`` blocks until this rank's shards are on disk. ``finalize()``
    publishes ``meta.pth``, which is what makes the directory a resume
    candidate, and the caller must barrier between the two so the marker means
    *every* rank's shards landed. That is why the marker appears at the next
    drain rather than the instant one rank's write returns: a crash in between
    costs the newest checkpoint instead of producing one that resumes into a
    missing shard.
    """

    path: Path
    meta: dict[str, Any]
    write_meta: bool
    future: Any = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _finalized: bool = False

    def wait(self) -> None:
        if self.future is not None:
            self.future.result()

    def finalize(self) -> None:
        """Publish the completion marker, once."""
        with self._lock:
            if self._finalized:
                return
            if self.write_meta:
                tmp = self.path / (DCP_META + ".tmp")
                torch.save(self.meta, tmp)
                tmp.replace(self.path / DCP_META)
                _relink_latest(self.path)
            self._finalized = True


def _kwargs_if_supported(target, **kwargs) -> dict:
    """Pass newer-torch options only where they exist, instead of pinning."""
    accepted = inspect.signature(target).parameters
    return {name: value for name, value in kwargs.items() if name in accepted}


def _async_fallback_notice(reason: str) -> None:
    global _async_notice_shown
    if _async_notice_shown:
        return
    _async_notice_shown = True
    print(f"[ckpt] background sharded save unavailable ({reason}); writing synchronously")


def dcp_process_group():
    """A gloo group for sharded-checkpoint collectives, or None.

    A staged payload lives in host memory and the checkpoint's own collectives
    are small metadata exchanges. A NCCL-only default group has no CPU backend,
    which the background save path requires, and reusing it would queue the
    write behind the training stream. Collective: every rank must call this
    once, at the same point in startup.
    """
    global _cpu_group, _cpu_group_resolved
    if _cpu_group_resolved:
        return _cpu_group
    _cpu_group_resolved = True
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
    if torch.distributed.get_world_size() == 1 or not torch.distributed.is_gloo_available():
        return None
    _cpu_group = torch.distributed.new_group(backend="gloo")
    return _cpu_group


def _async_dcp_save(state: dict, path: Path, writer, planner, process_group):
    """Start a background sharded write, or None when this torch cannot."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import state_dict_saver

    if not hasattr(dcp, "async_save"):
        _async_fallback_notice("this torch has no torch.distributed.checkpoint.async_save")
        return None
    kwargs = {}
    checkpointer_type = getattr(state_dict_saver, "AsyncCheckpointerType", None)
    if (
        checkpointer_type is not None
        and "async_checkpointer_type" in inspect.signature(dcp.async_save).parameters
    ):
        # a separate process serializes the staged copy instead of holding the
        # trainer's GIL for the length of the write
        kwargs["async_checkpointer_type"] = checkpointer_type.PROCESS
    try:
        return dcp.async_save(
            state,
            checkpoint_id=str(path),
            storage_writer=writer,
            planner=planner,
            process_group=process_group,
            **kwargs,
        )
    except (AssertionError, NotImplementedError, RuntimeError, TypeError, ValueError) as exc:
        # usually a process group without a CPU backend, which staging requires
        _async_fallback_notice(f"{type(exc).__name__}: {exc}")
        return None


def save_dcp_checkpoint(
    path: str | Path,
    state: dict,
    *,
    meta: dict[str, Any],
    write_meta: bool,
    process_group=None,
    prefer_async: bool = True,
) -> PendingCheckpoint:
    """Sharded save: every rank writes its own shard, nothing is gathered.

    The async path stages the payload to host memory before returning, so the
    step loop is released after a copy rather than after a full-model write.
    The caller drains the returned handle (``wait()``, barrier, ``finalize()``)
    before the next save and before the run exits.
    """
    import torch.distributed.checkpoint as dcp

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    writer = dcp.FileSystemWriter(
        path,
        # reuse one staging buffer across saves instead of reallocating the
        # whole payload's worth of host memory every time
        **_kwargs_if_supported(dcp.FileSystemWriter, cache_staged_state_dict=True),
    )
    planner = dcp.DefaultSavePlanner(**_kwargs_if_supported(dcp.DefaultSavePlanner, enable_plan_caching=True))
    future = _async_dcp_save(state, path, writer, planner, process_group) if prefer_async else None
    if future is None:
        dcp.save(
            state,
            checkpoint_id=str(path),
            storage_writer=writer,
            planner=planner,
            process_group=process_group,
        )
    return PendingCheckpoint(path=path, meta=meta, write_meta=write_meta, future=future)


def load_dcp_state(path: str | Path, state: dict, *, process_group=None) -> None:
    """Fill sharded placeholders in ``state`` from a sharded checkpoint."""
    import torch.distributed.checkpoint as dcp

    dcp.load(state, checkpoint_id=str(path), process_group=process_group)


def load_dcp_meta(path: str | Path) -> dict[str, Any]:
    """Read the sidecar that marks a sharded checkpoint complete."""
    marker = Path(path) / DCP_META
    if not marker.exists():
        raise ValueError(
            f"{path} has no {DCP_META}: the sharded save that wrote it never finished. "
            "Resume from an earlier checkpoint."
        )
    return torch.load(str(marker), map_location="cpu", weights_only=False)


# ---- retention and resume resolution ----


def _step_checkpoints(ckpt_dir: Path) -> list[Path]:
    return sorted(ckpt_dir.glob("epoch_*_step_*.pth")) + sorted(ckpt_dir.glob(f"epoch_*_step_*{DCP_SUFFIX}"))


def prune_checkpoints(
    ckpt_dir: str | Path, keep_last: int, milestones: list[int] | tuple[int, ...] = ()
) -> list[Path]:
    """Delete step-checkpoints beyond the newest ``keep_last``, sparing milestones.

    ``keep_last <= 0`` disables pruning: every step-checkpoint is kept. Returns
    the deleted paths.
    """
    if keep_last <= 0:
        return []
    found: list[tuple[int, Path]] = []
    for path in _step_checkpoints(Path(ckpt_dir)):
        match = _CKPT_RE.search(path.name)
        if match:
            found.append((int(match.group(2)), path))
    found.sort(key=lambda item: item[0])
    keep = {step for step, _ in found[-keep_last:]} | set(milestones)
    deleted = []
    for step, path in found:
        if step in keep:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        deleted.append(path)
    return deleted


def resolve_resume(ckpt_dir: str | Path) -> str | None:
    """Pick the resume checkpoint: a ``latest`` link, else the highest step.

    A ``.dcp`` directory counts only once its ``meta.pth`` marker exists;
    without it the sharded write never finished.
    """
    ckpt_dir = Path(ckpt_dir)
    for link in (ckpt_dir / "latest.pth", ckpt_dir / f"latest{DCP_SUFFIX}"):
        if link.exists():
            return str(link.resolve())
    best: tuple[int, int] | None = None
    best_path: Path | None = None
    for f in _step_checkpoints(ckpt_dir):
        m = _CKPT_RE.match(f.name)
        if m is None:
            continue
        if f.suffix == DCP_SUFFIX and not (f / DCP_META).exists():
            continue
        key = (int(m.group(2)), int(m.group(1)))
        if best is None or key > best:
            best, best_path = key, f
    return str(best_path) if best_path is not None else None
