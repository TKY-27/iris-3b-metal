"""Single-run training orchestration on top of HF Accelerate."""

import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn

from iris3b.config import Config, save_config
from iris3b.data.shapes import Shape
from iris3b.flow import RectifiedFlow
from iris3b.models import IrisDiT, ac
from iris3b.registry import TEXT_ENCODERS
from iris3b.sampling import generate
from iris3b.seeding import mix_seed
from iris3b.train import parallel
from iris3b.train.ckpt import (
    DCP_SUFFIX,
    _kwargs_if_supported,
    _restore_rng,
    _rng_state,
    checkpoint_plan,
    clone_shards,
    dcp_process_group,
    gather_full_state,
    load_checkpoint,
    load_dcp_meta,
    load_dcp_state,
    probe_shared_checkpoint_dir,
    prune_checkpoints,
    require_checkpoint_kind,
    require_dense_weights,
    resolve_resume,
    save_checkpoint,
    save_dcp_checkpoint,
)
from iris3b.train.compile import (
    compile_decision,
    compile_model,
    expected_shape_count,
    recompile_budget_warning,
    recompile_count,
)
from iris3b.train.ema import EMA, ShardEMA
from iris3b.train.lr import build_lr_scheduler
from iris3b.train.optim import build_optimizer
from iris3b.train.world import describe_world

VALIDATION_STEPS = 100
VALIDATION_CFG_SCALE = 3.0
SHEET_COLUMNS = 5


def _validate_optimizer_runtime(cfg: Config) -> None:
    cfg.train.perf.validate()
    cfg.train.optimizer.validate()


def _effective_batch(train_cfg, num_processes: int) -> int:
    """Global batch per optimizer step, checked against the launcher's pin when set."""
    effective_batch = train_cfg.batch_size * num_processes * train_cfg.grad_accum_steps
    expected = train_cfg.expected_global_batch
    if expected is None:
        return effective_batch
    if expected <= 0:
        raise ValueError(f"train.expected_global_batch must be positive, got {expected}")
    if effective_batch != expected:
        raise ValueError(
            f"global batch {effective_batch} (batch_size {train_cfg.batch_size} x "
            f"{num_processes} ranks x grad_accum_steps {train_cfg.grad_accum_steps}) "
            f"does not match train.expected_global_batch={expected}"
        )
    return effective_batch


def _checkpoint_data_position(
    sampler,
    batch_size: int,
    *,
    epoch: int,
    batches: int,
    plan_batches: int,
    plan_samples: int,
) -> dict:
    """Serialize the exact sampler cursor with its epoch and yielded batch count."""
    state = sampler.resume_state(
        plan_batches,
        batch_size,
        samples_per_rank=plan_samples,
    )
    return {
        **state,
        "epoch": epoch,
        "batches_consumed": batches,
    }


def _resolve_resume_data_position(
    cfg: Config,
    sampler,
    *,
    start_step: int,
    checkpoint_epoch: int,
    resumed_data_position: dict | None,
    saved_config: dict | None,
    effective_batch: int,
    steps_per_epoch: int,
    log,
) -> tuple[int, dict]:
    """Apply the configured cursor policy without changing restored training state."""
    if cfg.train.resume_data_policy == "new_phase":
        position = _checkpoint_data_position(
            sampler,
            cfg.train.batch_size,
            epoch=1,
            batches=0,
            plan_batches=0,
            plan_samples=0,
        )
        log(
            f"resume_data_policy=new_phase: retained checkpoint step {start_step} and "
            "model/EMA/optimizer/scheduler state; starting the current dataset at "
            "epoch 1 with zero consumed batches"
        )
        return start_step, position

    if checkpoint_epoch <= 0:
        raise ValueError("resume checkpoint has no valid epoch for its data position")
    if resumed_data_position is None:
        old_world_size = cfg.train.resume_data_world_size
        if old_world_size is None:
            raise ValueError(
                "resume checkpoint has no data_position; set "
                "train.resume_data_world_size to the checkpoint's original world size "
                "to migrate this legacy checkpoint"
            )
        if old_world_size <= 0:
            raise ValueError("train.resume_data_world_size must be positive")
        saved_train = saved_config.get("train") if isinstance(saved_config, dict) else None
        if not isinstance(saved_train, dict):
            raise ValueError(
                "legacy resume needs the checkpoint's saved train config to reconstruct its data position"
            )
        try:
            old_batch_size = int(saved_train["batch_size"])
            old_grad_accum = int(saved_train["grad_accum_steps"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "legacy resume needs batch_size and grad_accum_steps in the checkpoint config"
            ) from error
        if old_batch_size <= 0 or old_grad_accum <= 0:
            raise ValueError("legacy checkpoint batch_size and grad_accum_steps must be positive")
        old_global_batch = old_batch_size * old_grad_accum * old_world_size
        if old_global_batch != effective_batch:
            raise ValueError(
                "legacy data-position migration requires an unchanged global batch: "
                f"checkpoint={old_global_batch}, current={effective_batch}"
            )
        steps_in_epoch = start_step - (checkpoint_epoch - 1) * steps_per_epoch
        if not 0 <= steps_in_epoch <= steps_per_epoch:
            raise ValueError(
                "legacy checkpoint epoch/step is incompatible with the current "
                "steps_per_epoch despite an unchanged global batch"
            )
        old_batches = steps_in_epoch * old_grad_accum
        resumed_data_position = sampler.resume_state(
            old_batches,
            old_batch_size,
            world_size=old_world_size,
        )
        resumed_data_position = {
            **resumed_data_position,
            "epoch": checkpoint_epoch,
            "batches_consumed": old_batches,
        }
        log(
            "migrated legacy checkpoint data position using "
            f"train.resume_data_world_size={old_world_size}; progress was approximated "
            "from optimizer steps and assumes no skipped non-finite batches"
        )
    else:
        position_epoch = int(resumed_data_position.get("epoch", checkpoint_epoch))
        if position_epoch != checkpoint_epoch:
            raise ValueError(
                f"checkpoint epoch {checkpoint_epoch} disagrees with data_position epoch {position_epoch}"
            )
        if "batches_consumed" not in resumed_data_position:
            if "batches_per_rank" in resumed_data_position:
                restored_batches = int(resumed_data_position["batches_per_rank"])
                if resumed_data_position.get("kind") == "shape_batches":
                    restored_batches += sum(
                        int(entry["batches_per_rank"])
                        for entry in resumed_data_position.get("regroup_history", [])
                    )
            elif "samples_per_rank" in resumed_data_position:
                saved_train = saved_config.get("train") if isinstance(saved_config, dict) else None
                if not isinstance(saved_train, dict) or "batch_size" not in saved_train:
                    raise ValueError(
                        "checkpoint data_position needs the saved batch_size to recover "
                        "its yielded batch count"
                    )
                saved_batch_size = int(saved_train["batch_size"])
                restored_samples = int(resumed_data_position["samples_per_rank"])
                if saved_batch_size <= 0 or restored_samples % saved_batch_size:
                    raise ValueError("checkpoint samples_per_rank is not divisible by its saved batch_size")
                restored_batches = restored_samples // saved_batch_size
            else:
                raise ValueError("checkpoint data_position has no yielded batch count")
            resumed_data_position = {
                **resumed_data_position,
                "batches_consumed": restored_batches,
            }
        resumed_data_position = {
            **resumed_data_position,
            "epoch": checkpoint_epoch,
        }
    return start_step, resumed_data_position


def _aux_modules(repa: nn.Module | None) -> dict | None:
    """Auxiliary projector heads that must survive save/resume."""
    return None if repa is None else {"repa_projector": repa.projector}


def _dist_summary(dist_cfg, world) -> str:
    """One line naming the strategy and, when sharded, the mesh it runs on."""
    if not dist_cfg.sharded:
        return f"distributed: ddp over {world.world_size} rank(s)"
    dp_replicate, dp_shard = parallel.resolve_mesh(dist_cfg, world.world_size, world.local_world_size)
    return (
        f"distributed: {dist_cfg.strategy} {dist_cfg.sharding} "
        f"mesh dp_replicate={dp_replicate} x dp_shard={dp_shard} "
        f"param={dist_cfg.param_dtype} reduce={dist_cfg.reduce_dtype}"
    )


def _wrapper_state_dict(core_sd: dict, payload: dict) -> dict:
    """Map a single-file checkpoint onto TrainModel keys: the diffusion core
    under ``model.``, aux projector heads under their attribute paths. The
    frozen REPA teacher is never checkpointed, so loads use strict=False."""
    sd = {f"model.{k}": v for k, v in core_sd.items()}
    sd.update({f"repa.projector.{k}": v for k, v in (payload.get("repa_projector") or {}).items()})
    return sd


def _drop_stale_aux_group(saved_groups: list[dict], live_groups: list[dict]) -> list[dict]:
    """Forget the checkpoint's ``aux`` group when this run trains no aux parameters.

    The REPA projector lives in the trailing ``aux`` group. A stage that
    turns those losses off still builds that group, but empty, so the saved
    projector FQNs have no live counterpart: their state is dropped and the
    group loads as empty. Every other group still has to match exactly.
    """
    live_aux = [group for group in live_groups if group.get("iris_scope") == "aux"]
    if any(group["params"] for group in live_aux):
        return saved_groups
    trimmed = []
    for group in saved_groups:
        if group.get("iris_scope") == "aux" and group["params"]:
            if not live_aux:
                continue
            group = {**group, "params": [], "iris_param_names": ()}
        trimmed.append(group)
    return trimmed


def _load_fsdp2_optimizer_state(optimizer, state_dict: dict) -> None:
    """Load an FQN-keyed full/DCP state without torch DSD's shared-param split.

    The shared adaLN cores are referenced by every block. Torch's
    ``set_optimizer_state_dict`` resolves those aliases to an empty param group
    and refuses the load. Routed optimizer groups persist their exact FQN
    order, so validate that contract and shard each full state tensor
    positionally.
    """
    from torch.distributed.tensor import DTensor, distribute_tensor

    saved_groups = _drop_stale_aux_group(state_dict["param_groups"], optimizer.param_groups)
    live_groups = optimizer.param_groups
    if len(saved_groups) != len(live_groups):
        raise ValueError(
            f"optimizer group count mismatch: checkpoint has {len(saved_groups)}, "
            f"live optimizer has {len(live_groups)}"
        )
    dist_kwargs = _kwargs_if_supported(distribute_tensor, src_data_rank=None)
    for gi, (live_group, saved_group) in enumerate(zip(live_groups, saved_groups, strict=True)):
        live_names = live_group.get("iris_param_names")
        saved_names = list(saved_group["params"])
        if live_names is not None and list(live_names) != saved_names:
            raise ValueError(
                f"optimizer group {gi} parameter names diverge from the checkpoint: "
                f"{list(live_names)[:3]}... vs {saved_names[:3]}..."
            )
        if len(live_group["params"]) != len(saved_names):
            raise ValueError(
                f"optimizer group {gi} has {len(live_group['params'])} live params "
                f"but {len(saved_names)} in the checkpoint"
            )
        for key, value in saved_group.items():
            if key not in ("params", "iris_param_names"):
                live_group[key] = value
        for parameter, fqn in zip(live_group["params"], saved_names, strict=True):
            saved_slot = state_dict["state"].get(fqn)
            if saved_slot is None:
                continue
            slot = {}
            for key, value in saved_slot.items():
                if isinstance(value, DTensor):
                    slot[key] = value
                elif torch.is_tensor(value) and value.dim() > 0 and isinstance(parameter, DTensor):
                    if tuple(value.shape) != tuple(parameter.shape):
                        raise ValueError(
                            f"optimizer state {fqn}.{key} shape {tuple(value.shape)} "
                            f"does not match parameter shape {tuple(parameter.shape)}"
                        )
                    slot[key] = distribute_tensor(
                        value,
                        parameter.device_mesh,
                        parameter.placements,
                        **dist_kwargs,
                    )
                elif torch.is_tensor(value):
                    # scalar slots such as a device-side step counter are read by fused CUDA kernels
                    slot[key] = value.to(parameter.device, copy=True)
                else:
                    slot[key] = value
            optimizer.state[parameter] = slot


def _grad_groups(core: nn.Module, repa: nn.Module | None) -> list[tuple[str, list[nn.Parameter]]]:
    """Named parameter groups for per-block gradient-norm logging."""
    groups: list[tuple[str, list[nn.Parameter]]] = []
    for name in (
        "s_embedder",
        "t_embedder",
        "y_embedder",
        "pixel_embedder",
        "final_layer",
        "modulation_cores",  # shared adaLN cores; empty, hence skipped, under per_block
    ):
        module = getattr(core, name, None)
        if module is not None:
            groups.append((name, list(module.parameters())))
    if core.y_pos_embedding is not None:
        groups.append(("y_pos_embedding", [core.y_pos_embedding]))
    for i, block in enumerate(core.blocks):
        groups.append((f"blocks.{i:02d}", list(block.parameters())))
    if core.pixel_blocks is not None:
        for i, block in enumerate(core.pixel_blocks):
            groups.append((f"pixel_blocks.{i}", list(block.parameters())))
    if repa is not None:
        groups.append(("repa_projector", list(repa.projector.parameters())))
    return groups


def _group_grad_norms(groups: list[tuple[str, list[nn.Parameter]]]) -> dict[str, torch.Tensor]:
    """Pre-clip, post-allreduce L2 grad norm per group; parameters without
    grads (the dead text tail) are skipped."""
    norms: dict[str, torch.Tensor] = {}
    for name, params in groups:
        grads = [p.grad for p in params if p.grad is not None]
        if grads:
            norms[name] = torch.norm(torch.stack(torch._foreach_norm(grads)))
    return norms


class _LossLogAccumulator:
    """Reduce rank-local batch means into one global optimizer-step view."""

    def __init__(self) -> None:
        self._sums: dict[str, torch.Tensor] = {}
        self._samples = 0

    def clear(self) -> None:
        self._sums.clear()
        self._samples = 0

    def add(self, flow_loss: torch.Tensor, repa_loss: torch.Tensor | None, *, samples: int) -> None:
        if samples <= 0:
            raise ValueError(f"loss logging requires a positive sample count, got {samples}")

        values = {"loss/flow": flow_loss}
        if repa_loss is not None:
            values["loss/repa_raw"] = repa_loss

        if self._sums and tuple(values) != tuple(self._sums):
            raise RuntimeError("loss metric names changed within one optimizer step")
        if not self._sums:
            self._sums = {name: value.detach().float().mul(samples) for name, value in values.items()}
        else:
            for name, value in values.items():
                self._sums[name].add_(value.detach().float(), alpha=samples)
        self._samples += samples

    def finish(self, accelerator, *, repa_weight: float) -> dict[str, float]:
        if not self._sums:
            raise RuntimeError("cannot log an empty optimizer step")

        names = tuple(self._sums)
        first = next(iter(self._sums.values()))
        packed = torch.stack([*self._sums.values(), first.new_tensor(float(self._samples))])
        totals = accelerator.reduce(packed, reduction="sum")
        global_values = dict(zip(names, (totals[:-1] / totals[-1]).tolist(), strict=True))
        self.clear()

        flow = global_values["loss/flow"]
        logs = {"loss": flow, "loss/flow": flow}
        total = flow
        if "loss/repa_raw" in global_values:
            raw = global_values["loss/repa_raw"]
            logs.update(
                {
                    "repa_loss": raw,
                    "loss/repa_raw": raw,
                    "loss/repa_weighted": repa_weight * raw,
                }
            )
            total += repa_weight * raw
        logs["loss/total"] = total
        return logs


def seed_everything(seed: int) -> None:
    """Seed python, numpy, torch, and every cuda device."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


VAL_GRID_SEED = 0x5AF1  # keys the frozen (noise, t) validation grid
VAL_T_STRIDE = 619  # ~golden-ratio of 1000, coprime: low-discrepancy timestep coverage


def _build_val_loader(cfg: Config, rank: int, world_size: int):
    """Per-rank loader over the leading ``train.val_samples`` holdout pairs.

    Captions are pinned (seed 0, epoch 0), the walk order is the dataset order,
    and each sample carries its absolute index, so the frozen grid is identical
    across steps, runs, batch sizes and world sizes.

    Under a shape-varying policy the loader groups by shape exactly like
    training does -- a dense batch admits one shape -- but the grid's identity
    still comes from the carried index, so grouping cannot re-label it. Note
    that val loss remains comparable only within one (shape policy, image_size,
    flow.shift) triple: the noise field is drawn at the sample's own shape.
    """
    from dataclasses import replace as dc_replace

    from torch.utils.data import DataLoader

    from iris3b.data.builder import collate_batch
    from iris3b.data.samplers import RangedSampler, ShapeBatchSampler
    from iris3b.registry import DATASETS

    val_cfg = dc_replace(cfg.data, data_dirs=list(cfg.data.val_data_dirs), val_data_dirs=[])
    dataset = DATASETS.build(val_cfg.type, val_cfg, cfg.model.patch_size)
    dataset.seed = 0
    dataset.epoch = 0
    limit = len(dataset) if cfg.train.val_samples <= 0 else min(len(dataset), cfg.train.val_samples)
    sampler = RangedSampler(limit, rank=rank, world_size=world_size)
    common = {"num_workers": 2, "pin_memory": True, "collate_fn": collate_batch}
    if dataset.policy.uniform:
        return DataLoader(dataset, batch_size=cfg.train.batch_size, sampler=sampler, **common)
    batch_sampler = ShapeBatchSampler(sampler, dataset, cfg.train.batch_size, drop_last=False)
    return DataLoader(dataset, batch_sampler=batch_sampler, **common)


def _agree_batch_count(accelerator, sampler, dataloader, *, allow_empty: bool = False) -> None:
    """Truncate every rank to the global minimum number of batches.

    A shape-grouping sampler emits ``sum_g floor(n_g / B)`` batches, and ``n_g``
    differs per rank because each rank owns a different contiguous slice of the
    corpus. Ranks that disagree on the batch count issue different numbers of
    collectives and the run hangs -- in DDP's gradient all-reduce, inside Dion
    Muon's ``all_to_all`` during ``optimizer.step()``, or in FSDP's clip-norm
    reduce. One reduction here replaces the heuristic tail break.
    """
    limiter = getattr(sampler, "limit_batches", None)
    if limiter is None or accelerator.num_processes == 1:
        return
    local = torch.tensor([len(dataloader)], device=accelerator.device, dtype=torch.int64)
    torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.MIN)
    agreed = int(local.item())
    if agreed <= 0 and not allow_empty:
        raise ValueError("a rank produced zero shape-homogeneous batches; lower train.batch_size")
    limiter(agreed)
    accelerator.print(f"shape batching: {agreed} remaining batches/rank (global minimum)")


def _to_uint8(images: torch.Tensor) -> np.ndarray:
    """Map [-1, 1] float images [B, C, H, W] to uint8 [B, H, W, C]."""
    arr = (images.float() * 127.5 + 128.0).clamp_(0, 255).to(torch.uint8)
    return arr.permute(0, 2, 3, 1).cpu().numpy()


def _contact_sheet(frames: np.ndarray, columns: int = SHEET_COLUMNS) -> np.ndarray:
    """Tile frames row-major into one image, ``columns`` per row."""
    n, h, w, c = frames.shape
    cols = min(columns, n)
    rows = math.ceil(n / cols)
    sheet = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, col = divmod(i, cols)
        sheet[r * h : (r + 1) * h, col * w : (col + 1) * w] = frame
    return sheet


@dataclass
class TrainOutput:
    x: torch.Tensor
    features: dict[int, torch.Tensor]
    repa_loss: torch.Tensor | None


class TrainModel(nn.Module):
    """Single module handed to the accelerator: diffusion core plus optional aux heads.

    Auxiliary losses are computed INSIDE this forward so DDP sees their
    projectors participate in the wrapped graph (an external call would leave
    their parameters "unused in forward" and break the reducer). Both branches
    run with autocast disabled: teachers and projectors stay fp32.
    """

    def __init__(self, model: IrisDiT, repa: nn.Module | None = None):
        super().__init__()
        self.model = model
        self.repa = repa

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        clean: torch.Tensor | None = None,
        y_mask: torch.Tensor | None = None,
    ):
        capture = () if self.repa is None else (self.model.cfg.repa_layer,)
        out = self.model(x, t, y, capture=capture, y_mask=y_mask)
        repa_loss = None
        if self.repa is not None and clean is not None:
            patch = self.model.cfg.patch_size
            grid = (x.shape[-2] // patch, x.shape[-1] // patch)
            with torch.autocast(x.device.type, enabled=False):
                repa_loss = self.repa(clean, out.features[self.model.cfg.repa_layer], grid)
        return TrainOutput(x=out.x, features=out.features, repa_loss=repa_loss)


class Trainer:
    """Owns one full training run described by a ``Config``."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self) -> None:
        from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

        from iris3b.data.builder import build_dataloader

        cfg = self.cfg
        perf = cfg.train.perf
        _validate_optimizer_runtime(cfg)
        cfg.data.validate(cfg.model.patch_size)
        cfg.train.validate_resume_data()
        ac.validate_policy(cfg.train.activation_checkpointing, cfg.train.ac_selective_every)
        dist_cfg = cfg.train.dist
        dist_cfg.validate(cfg.train)
        sharded = dist_cfg.sharded
        world = describe_world()
        if cfg.train.resume_optimizer not in {"full", "core"}:
            raise ValueError(
                f"train.resume_optimizer must be full or core, got {cfg.train.resume_optimizer!r}"
            )
        work_dir = Path(cfg.work_dir)
        ckpt_dir = work_dir / "checkpoints"
        vis_dir = work_dir / "log_vis"
        log_with = None if cfg.report_to == "none" else cfg.report_to

        # fsdp2 shards explicitly with fully_shard below, so it takes no plugin
        # and no DDP handler: accelerate would otherwise wrap DTensor
        # parameters in DDP, which it refuses outright.
        handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=5400))]
        if not sharded:
            handlers.append(DistributedDataParallelKwargs(**parallel.ddp_kwargs(dist_cfg, cfg.model)))
        accelerator = Accelerator(
            mixed_precision=cfg.train.mixed_precision,
            gradient_accumulation_steps=cfg.train.grad_accum_steps,
            log_with=log_with,
            project_dir=str(work_dir / "logs"),
            kwargs_handlers=handlers,
        )
        if world.world_size != accelerator.num_processes:
            # every mesh factorization is computed from the launcher environment
            raise ValueError(
                f"launcher WORLD_SIZE={world.world_size} disagrees with the process group "
                f"({accelerator.num_processes} ranks); the mesh would be built for the wrong world"
            )
        # fail here, before the model exists, rather than after 3B params are sharded
        effective_batch = _effective_batch(cfg.train, accelerator.num_processes)
        device = accelerator.device
        seed_everything(cfg.train.seed + accelerator.process_index)
        if accelerator.is_main_process:
            for d in (ckpt_dir, vis_dir, work_dir / "logs"):
                d.mkdir(parents=True, exist_ok=True)
            save_config(cfg, work_dir / "config.yaml")
        accelerator.wait_for_everyone()
        accelerator.print(_dist_summary(dist_cfg, world))
        for option in parallel.unsupported_ddp_options(dist_cfg):
            accelerator.print(f"{option} is not available in the installed torch/accelerate; ignored")

        core = IrisDiT(cfg.model)
        core.activation_checkpointing = cfg.train.activation_checkpointing
        core.ac_selective_every = cfg.train.ac_selective_every
        stage_tokens = (cfg.data.image_size // cfg.model.patch_size) ** 2
        flow = RectifiedFlow(cfg.flow, tokens=stage_tokens)
        self.flow_shift = flow.shift
        self.render_shape = Shape(cfg.data.image_size, cfg.data.image_size)
        if flow.shift != cfg.flow.shift:
            accelerator.print(
                f"flow.shift_law={cfg.flow.shift_law}: {cfg.flow.shift} -> {flow.shift:.4f} "
                f"at {stage_tokens} image tokens"
            )

        import iris3b.text  # noqa: F401  (registers text encoders)

        text_encoder = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device)

        repa = None
        if cfg.repa.weight > 0:
            from iris3b.repa import REPALoss

            # rank 0 downloads the hub teacher alone; others then hit the cache
            # (concurrent torch.hub loads race on extraction and corrupt it)
            with accelerator.main_process_first():
                repa = REPALoss(cfg.repa, student_dim=cfg.model.hidden_size)
            # DDP requires identical params on every rank: keep REPA only if
            # the teacher loaded everywhere
            ok = torch.tensor([1.0 if getattr(repa, "available", True) else 0.0], device=device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
            if ok.item() < 1:
                accelerator.print("repa teacher unavailable on some rank; alignment loss disabled")
                repa = None
        train_model = TrainModel(core, repa)
        mesh = None
        if dist_cfg.strategy == "fsdp2":
            # before build_optimizer: fully_shard REPLACES every nn.Parameter
            # with a DTensor-backed one, and the optimizer is built from the
            # raw module's parameter objects
            mesh = parallel.build_device_mesh(dist_cfg, world.world_size, world.local_world_size)
            parallel.shard_model(train_model, dist_cfg, mesh, parallel.wrap_class_names(cfg.model))

        dataloader, sampler = build_dataloader(
            cfg.data,
            cfg.train.batch_size,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            seed=cfg.train.seed,
            patch_size=cfg.model.patch_size,
        )
        if len(dataloader) == 0:
            raise ValueError("dataloader is empty; check data config")
        _agree_batch_count(accelerator, sampler, dataloader)

        val_loader = None
        if cfg.train.val_every_steps > 0 and cfg.data.val_data_dirs:
            val_loader = _build_val_loader(
                cfg, rank=accelerator.process_index, world_size=accelerator.num_processes
            )
            accelerator.print(
                f"frozen-grid val loss every {cfg.train.val_every_steps} steps "
                f"({len(val_loader.sampler)} samples/rank)"
            )

        named_params = [(f"model.{name}", parameter) for name, parameter in core.named_parameters()]
        if repa is not None:
            named_params += [
                (f"repa.projector.{name}", parameter) for name, parameter in repa.projector.named_parameters()
            ]
        parameter_names = [name for name, _ in named_params]
        params = [parameter for _, parameter in named_params]
        optimizer, scaled_lr = build_optimizer(
            cfg.train.optimizer,
            params,
            effective_batch,
            fused=perf.fused_adamw,
            model=core,
            parameter_names=parameter_names,
            # a DeviceMesh here routes Dion's Muon onto its DTensor path so it
            # orthogonalizes whole matrices rather than per-rank shards
            distributed_mesh=mesh,
        )
        parallel.assert_optimizer_sees_shards(optimizer, dist_cfg)
        routing = getattr(optimizer, "iris_routing", None)
        if routing is not None:
            routed = routing["muon"] + routing["adamw"]
            accelerator.print(
                "optimizer: Muon+AdamW "
                f"muon={routing['muon']:,} ({routing['muon'] / routed:.2%}) "
                f"adamw={routing['adamw']:,} ({routing['adamw'] / routed:.2%}) "
                f"momentum={cfg.train.optimizer.muon_momentum} "
                f"nesterov={cfg.train.optimizer.muon_nesterov} "
                f"adjust_lr={cfg.train.optimizer.muon_adjust_lr}"
            )
        n_params_m = core.num_parameters / 1e6  # DTensor params report their global shape

        epoch_batches = len(dataloader)
        steps_per_epoch = math.ceil(epoch_batches / cfg.train.grad_accum_steps)
        total_steps = steps_per_epoch * cfg.train.num_epochs
        if cfg.train.max_steps > 0:
            total_steps = min(total_steps, cfg.train.max_steps)
        scheduler = build_lr_scheduler(cfg.train.optimizer, optimizer, accelerator.num_processes, total_steps)

        # The dataloader's sampler already shards per rank, so it stays out of
        # prepare (accelerate would shard each batch a second time); batches
        # move to the device manually below.
        if dist_cfg.strategy == "fsdp2":
            # the model is already sharded; accelerate refuses DTensor parameters
            # in prepare_model, so it only registers the module (for accumulate,
            # clip_grad_norm_ and the autocast forward wrapper) and skips wrapping
            train_model = accelerator.prepare_model(train_model, device_placement=False, evaluation_mode=True)
            optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
        else:
            train_model, optimizer, scheduler = accelerator.prepare(train_model, optimizer, scheduler)
        core = accelerator.unwrap_model(train_model).model
        repa = accelerator.unwrap_model(train_model).repa

        ema = None
        if cfg.train.ema.enabled and not sharded:  # sharded builds its ShardEMA during restore
            ema = EMA(core, cfg.train.ema.decay, foreach=perf.foreach_ema)
        grad_groups = _grad_groups(core, repa) if cfg.train.log_block_grad_norms else None

        accelerator.print(
            f"model {n_params_m:.1f}M params | effective batch {effective_batch} "
            f"| lr {scaled_lr:.2e} | {steps_per_epoch} steps/epoch | {total_steps} total"
        )

        resume_path = None
        if cfg.train.resume_from:
            resume_path = (
                resolve_resume(ckpt_dir) if cfg.train.resume_from == "latest" else cfg.train.resume_from
            )
        ema, start_step, resumed = self._restore(
            accelerator,
            train_model,
            core,
            repa,
            optimizer,
            scheduler,
            ema,
            ckpt_dir,
            resume_path,
        )
        resumed_data_position = getattr(self, "_resumed_data_position", None)
        new_data_phase = resumed and cfg.train.resume_data_policy == "new_phase"
        if resumed:
            start_step, resumed_data_position = _resolve_resume_data_position(
                cfg,
                sampler,
                start_step=start_step,
                checkpoint_epoch=int(getattr(self, "_resumed_checkpoint_epoch", 0)),
                resumed_data_position=resumed_data_position,
                saved_config=getattr(self, "_resumed_checkpoint_config", None),
                effective_batch=effective_batch,
                steps_per_epoch=steps_per_epoch,
                log=accelerator.print,
            )
            self._resumed_data_position = resumed_data_position
        if resumed and cfg.train.override_lr_on_resume:
            for group in optimizer.param_groups:
                group["lr"] = scaled_lr
                group["initial_lr"] = scaled_lr
            base = getattr(scheduler, "scheduler", scheduler)
            if hasattr(base, "base_lrs"):
                base.base_lrs = [scaled_lr] * len(base.base_lrs)
            accelerator.print(f"lr overridden to {scaled_lr:.2e} after resume")

        if perf.compile:
            # In-place compile keeps state_dict names, DDP hooks, and the EMA
            # shadow (deep-copied eager above) untouched. The shape count comes
            # from the data shape policy so a bucketed run sizes dynamo's cache
            # instead of silently falling back to eager part-way through.
            compile_model(
                core,
                perf,
                shape_count=expected_shape_count(cfg.data, cfg.model.patch_size),
            )
            accelerator.print(compile_decision())

        if log_with is not None:
            wandb_init = {"name": cfg.name, "tags": list(cfg.tags)}
            # Checkpoint resumes continue the SAME wandb run (id persisted in
            # the work_dir): wandb cannot merge runs after the fact. Overlap
            # note: wandb drops non-increasing steps, so on a resume from an
            # older checkpoint the already-logged window keeps its first values.
            run_id_file = work_dir / "wandb_run_id"
            if cfg.report_to == "wandb" and resumed and run_id_file.is_file():
                wandb_init["id"] = run_id_file.read_text().strip()
                wandb_init["resume"] = "must"
                accelerator.print(f"resuming wandb run {wandb_init['id']}")
            try:
                accelerator.init_trackers(
                    cfg.wandb_project, config=asdict(cfg), init_kwargs={"wandb": wandb_init}
                )
            except Exception as e:  # noqa: BLE001 - tracker back-ends vary in config strictness
                accelerator.print(f"tracker init with config failed ({e}); retrying without config")
                accelerator.init_trackers(cfg.wandb_project, init_kwargs={"wandb": wandb_init})
            if cfg.report_to == "wandb":
                # work_dir is per-node and the c10d rendezvous assigns the main
                # rank by sorted FQDN, so it can land on a different node across
                # restarts. Broadcast the id and write it on every node's local
                # writer, or the next resume forks a fresh wandb run.
                run_id_box = [None]
                if accelerator.is_main_process:
                    for tracker in accelerator.trackers:
                        if getattr(tracker, "name", "") == "wandb":
                            run_id_box[0] = str(tracker.tracker.id)
                if accelerator.num_processes > 1:
                    torch.distributed.broadcast_object_list(run_id_box, src=0)
                if accelerator.local_process_index == 0 and run_id_box[0]:
                    run_id_file.write_text(run_id_box[0])

        with torch.no_grad():
            null_enc = text_encoder.null("")
            null_y = null_enc.embeddings.to(device)
            null_mask = null_enc.mask.to(device)

        global_step = start_step
        first_epoch = int(resumed_data_position["epoch"]) if resumed else 1
        active_epoch = first_epoch
        batches_consumed = int(resumed_data_position["batches_consumed"]) if resumed else 0
        plan_batches_consumed = 0
        plan_samples_consumed = 0
        nan_count = 0
        last_saved = start_step if resumed else -1  # the resumed checkpoint already holds start_step
        window: list[float] = []
        loss_accumulator = _LossLogAccumulator()
        window_t0 = time.time()

        reached_max_steps = False

        def _data_position(epoch: int, batches: int, plan_batches: int, plan_samples: int) -> dict:
            return _checkpoint_data_position(
                sampler,
                cfg.train.batch_size,
                epoch=epoch,
                batches=batches,
                plan_batches=plan_batches,
                plan_samples=plan_samples,
            )

        for epoch in range(first_epoch, cfg.train.num_epochs + 1):
            active_epoch = epoch
            sampler.set_epoch(epoch)
            limiter = getattr(sampler, "limit_batches", None)
            if limiter is not None:
                limiter(epoch_batches)
            dataloader.dataset.epoch = epoch
            resume_epoch = resumed and epoch == first_epoch
            skip_batches = int(resumed_data_position["batches_consumed"]) if resume_epoch else 0
            batches_consumed = skip_batches
            plan_batches_consumed = 0
            plan_samples_consumed = 0
            if resume_epoch:
                sampler.load_resume_state(resumed_data_position)
                # Regrouping changes each rank's remaining shape count; a saved
                # migrated plan can also carry a smaller cap than a fresh epoch.
                # An exhausted resumed epoch is valid and advances on all ranks.
                _agree_batch_count(accelerator, sampler, dataloader, allow_empty=True)
                if not new_data_phase:
                    # contiguous states carry world_size at top level, shape-batched ones under layout
                    saved_world = resumed_data_position.get("world_size") or resumed_data_position.get(
                        "layout", {}
                    ).get("world_size")
                    accelerator.print(
                        f"data position restored from a world size of "
                        f"{saved_world} into {world.world_size}; "
                        f"{skip_batches} batches already consumed in epoch {epoch}"
                    )
            else:
                sampler.set_start_batches(0, cfg.train.batch_size)

            accelerator.wait_for_everyone()
            train_model.train()

            timing = os.environ.get("IRIS_TIMING", "") == "1"
            t_prev = time.perf_counter()

            for local_step, batch in enumerate(dataloader):
                batches_consumed = skip_batches + local_step + 1
                plan_batches_consumed = local_step + 1
                if timing:
                    t_data = time.perf_counter()
                # Positional seeding: dropout/timestep/noise draws become a pure
                # function of (seed, rank, epoch, batch position), so a resumed
                # run continues the unbroken draw sequence bit-exactly.
                torch.manual_seed(
                    mix_seed(cfg.train.seed, accelerator.process_index, epoch, skip_batches + local_step)
                )
                images = batch["image"].to(device, non_blocking=True)
                captions = list(batch["caption"])
                plan_samples_consumed += len(captions)
                if perf.skip_dropped_text and cfg.train.text_dropout > 0:
                    # Same cuda rng draw as the post-encode mask below (the
                    # encoder consumes no rng), pulled forward so dropped rows
                    # skip the encoder. Costs one early host sync to gather the
                    # surviving row indices.
                    drop = torch.rand(len(captions), device=device) < cfg.train.text_dropout
                    keep_rows = (~drop).nonzero(as_tuple=True)[0]
                    y = null_y.to(images.dtype).expand(len(captions), *([-1] * (null_y.ndim - 1))).clone()
                    # dropped rows carry the null embedding, so they must carry
                    # the NULL mask too - keeping a caption's mask on a dropped
                    # row would mask the null tokens by the wrong pattern
                    y_mask = null_mask.expand(len(captions), -1).clone()
                    if keep_rows.numel():
                        kept = [captions[i] for i in keep_rows.tolist()]
                        with torch.no_grad():
                            enc = text_encoder.encode(kept)
                        y.index_copy_(0, keep_rows, enc.embeddings.to(device=device, dtype=images.dtype))
                        y_mask.index_copy_(0, keep_rows, enc.mask.to(device=device, dtype=y_mask.dtype))
                else:
                    with torch.no_grad():
                        enc = text_encoder.encode(captions)
                    y = enc.embeddings.to(device=device, dtype=images.dtype)
                    y_mask = enc.mask.to(device=device)
                    if cfg.train.text_dropout > 0:
                        drop = torch.rand(y.shape[0], device=y.device) < cfg.train.text_dropout
                        drop_shape = (drop.shape[0],) + (1,) * (y.ndim - 1)
                        y = torch.where(drop.reshape(drop_shape), null_y.to(y.dtype), y)
                        y_mask = torch.where(drop[:, None], null_mask.to(y_mask.dtype), y_mask)
                if timing:
                    torch.cuda.synchronize()
                    t_text = time.perf_counter()

                stepped = False
                grad_norm = None
                with accelerator.accumulate(train_model):
                    out = flow.training_loss(
                        train_model,
                        images,
                        y,
                        model_kwargs={"clean": images, "y_mask": y_mask},
                    )
                    flow_loss = out.loss
                    repa_loss = out.repa_loss
                    total = flow_loss
                    if repa_loss is not None:
                        total = total + cfg.repa.weight * repa_loss
                    if timing:
                        torch.cuda.synchronize()
                        t_fwd = time.perf_counter()
                    if not perf.lazy_sync and not torch.isfinite(total.detach()):
                        nan_count += 1
                        accelerator.print(
                            f"non-finite loss at step {global_step} "
                            f"({nan_count}/{cfg.train.nan_loss_tolerance})"
                        )
                        if nan_count > cfg.train.nan_loss_tolerance:
                            raise RuntimeError(f"loss was non-finite {nan_count} times; aborting")
                        optimizer.zero_grad(set_to_none=True)
                        loss_accumulator.clear()
                        continue
                    block_norms = None
                    accelerator.backward(total)
                    if timing:
                        torch.cuda.synchronize()
                        t_bwd = time.perf_counter()
                    if accelerator.sync_gradients:
                        block_norms = _group_grad_norms(grad_groups) if grad_groups is not None else None
                        grad_norm = parallel.full_grad_norm(
                            accelerator.clip_grad_norm_(train_model.parameters(), cfg.train.gradient_clip)
                        )
                        if perf.lazy_sync and not torch.isfinite(grad_norm):
                            # The step's single host sync. Post-allreduce grads
                            # are identical on every rank, so the skip is
                            # rank-consistent; the check subsumes non-finite
                            # losses (their grads are non-finite) and also
                            # catches inf grads under a finite loss.
                            nan_count += 1
                            accelerator.print(
                                f"non-finite grad norm at step {global_step} "
                                f"({nan_count}/{cfg.train.nan_loss_tolerance})"
                            )
                            if nan_count > cfg.train.nan_loss_tolerance:
                                raise RuntimeError(f"grads were non-finite {nan_count} times; aborting")
                            optimizer.zero_grad(set_to_none=True)
                            loss_accumulator.clear()
                            continue
                        stepped = True
                        if ema is not None:
                            if sharded:
                                ema.update()
                            else:
                                ema.update(core)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                if timing:
                    torch.cuda.synchronize()
                    t_end = time.perf_counter()
                    print(
                        f"[timing] rank={accelerator.process_index} micro={local_step} "
                        f"data={t_data - t_prev:.2f} text={t_text - t_data:.2f} "
                        f"fwd={t_fwd - t_text:.2f} bwd={t_bwd - t_fwd:.2f} "
                        f"opt={t_end - t_bwd:.2f}",
                        flush=True,
                    )
                    t_prev = t_end
                loss_accumulator.add(flow_loss, repa_loss, samples=len(captions))

                if not stepped:
                    continue
                global_step += 1

                lr_now = optimizer.param_groups[0]["lr"]
                logs = loss_accumulator.finish(accelerator, repa_weight=cfg.repa.weight)
                logs.update(
                    {
                        "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                        "lr": lr_now,
                        "epoch": epoch,
                        "step": global_step,
                    }
                )
                if block_norms:
                    logs.update({f"grad/{name}": v.item() for name, v in block_norms.items()})
                if perf.compile and global_step % cfg.train.log_every == 0:
                    logs["perf/recompiles"] = recompile_count()
                    budget_warning = recompile_budget_warning()
                    if budget_warning is not None:
                        accelerator.print(budget_warning)
                if log_with is not None:
                    accelerator.log(logs, step=global_step)
                window.append(logs["loss"])
                if global_step % cfg.train.log_every == 0:
                    dt = (time.time() - window_t0) / max(1, len(window))
                    eta = timedelta(seconds=int(dt * max(0, total_steps - global_step)))
                    accelerator.print(
                        f"epoch {epoch} | step {global_step}/{total_steps} "
                        f"| loss {sum(window) / len(window):.4f} | grad {logs['grad_norm']:.3f} "
                        f"| lr {lr_now:.3e} | {dt:.2f}s/it | eta {eta}"
                    )
                    window.clear()
                    window_t0 = time.time()

                if cfg.train.save_every_steps > 0 and global_step % cfg.train.save_every_steps == 0:
                    self._save(
                        accelerator,
                        train_model,
                        optimizer,
                        scheduler,
                        ema,
                        ckpt_dir,
                        epoch,
                        global_step,
                        data_state=_data_position(
                            epoch,
                            batches_consumed,
                            plan_batches_consumed,
                            plan_samples_consumed,
                        ),
                    )
                    last_saved = global_step
                if cfg.train.sample_every_steps > 0 and (
                    global_step % cfg.train.sample_every_steps == 0 or global_step == 1
                ):
                    self._validate(accelerator, core, text_encoder, vis_dir, global_step, train_model)
                if (
                    cfg.train.val_every_steps > 0
                    and val_loader is not None
                    and (global_step % cfg.train.val_every_steps == 0)
                ):
                    val_logs = self._val_loss(
                        accelerator, train_model if sharded else core, flow, text_encoder, val_loader
                    )
                    if log_with is not None:
                        accelerator.log(val_logs, step=global_step)
                    accelerator.print(
                        f"val @ {global_step}: "
                        + " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in val_logs.items())
                    )

                if cfg.train.max_steps > 0 and global_step >= cfg.train.max_steps:
                    accelerator.print(f"max_steps {cfg.train.max_steps} reached at epoch {epoch}")
                    reached_max_steps = True
                    break
            if reached_max_steps:
                break

            if (
                cfg.train.save_every_epochs > 0
                and (epoch % cfg.train.save_every_epochs == 0 or epoch == cfg.train.num_epochs)
                and global_step != last_saved
            ):
                self._save(
                    accelerator,
                    train_model,
                    optimizer,
                    scheduler,
                    ema,
                    ckpt_dir,
                    epoch,
                    global_step,
                    data_state=_data_position(
                        epoch,
                        batches_consumed,
                        plan_batches_consumed,
                        plan_samples_consumed,
                    ),
                )
                last_saved = global_step

        if global_step != last_saved:
            self._save(
                accelerator,
                train_model,
                optimizer,
                scheduler,
                ema,
                ckpt_dir,
                active_epoch,
                global_step,
                data_state=_data_position(
                    active_epoch,
                    batches_consumed,
                    plan_batches_consumed,
                    plan_samples_consumed,
                ),
            )
        # a background sharded write is a daemon: leaving one in flight at
        # exit throws the checkpoint away
        self._drain_checkpoint(accelerator)
        accelerator.wait_for_everyone()
        if log_with is not None:
            accelerator.end_training()
        accelerator.print(f"training complete at step {global_step}")

    def _save(
        self,
        accelerator,
        train_model,
        optimizer,
        scheduler,
        ema,
        ckpt_dir: Path,
        epoch,
        step,
        data_state: dict | None = None,
    ) -> None:
        accelerator.wait_for_everyone()
        plan = self._ckpt_plan
        if plan.sharded_layout:
            self._save_dcp(
                accelerator,
                train_model,
                optimizer,
                scheduler,
                ema,
                ckpt_dir,
                epoch,
                step,
                data_state=data_state,
            )
            return
        if self.cfg.train.dist.sharded:
            self._save_sharded(
                accelerator,
                train_model,
                optimizer,
                scheduler,
                ema,
                ckpt_dir,
                epoch,
                step,
                data_state=data_state,
            )
            return
        if not plan.writes:
            return
        unwrapped = accelerator.unwrap_model(train_model)
        extra = _aux_modules(unwrapped.repa)
        if extra is not None:
            extra = {name: module.state_dict() for name, module in extra.items()}
        if data_state is not None:
            # world-size-independent data position: without it a resume at a
            # different rank count re-derives the partition and re-reads data
            extra = {**(extra or {}), "data_position": data_state}
        path = save_checkpoint(
            ckpt_dir / f"epoch_{epoch}_step_{step}.pth",
            unwrapped.model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            epoch=epoch,
            step=step,
            config_dict=asdict(self.cfg),
            extra_state=extra,
            tmp_tag=plan.tmp_tag,
        )
        accelerator.print(f"saved checkpoint {path}")
        prune_checkpoints(
            ckpt_dir,
            self.cfg.train.keep_last_checkpoints,
            self.cfg.train.milestone_steps,
        )

    def _save_sharded(
        self,
        accelerator,
        train_model,
        optimizer,
        scheduler,
        ema,
        ckpt_dir: Path,
        epoch,
        step,
        *,
        data_state: dict | None = None,
    ) -> None:
        """Consolidate a sharded model into the single-file format.

        The gathers are collectives, so every rank enters them and the elected
        writers alone keep the full dicts. ``per_node`` elects one writer per
        node, which is why the gather cannot be rank-0-only there. The frozen
        REPA teacher is held outside the module tree and never appears here.
        """
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
        )

        keep = self._ckpt_plan.writes
        full = gather_full_state(get_model_state_dict(train_model), keep)
        ema_full = None
        if ema is not None:
            # the shadow holds the averaged shards; swapping puts them in the
            # live parameters the gather reads
            with ema.swapped():
                ema_full = gather_full_state(get_model_state_dict(train_model), keep)
        optim_sd = gather_full_state(get_optimizer_state_dict(train_model, optimizer), keep)
        if not keep:
            return

        def subtree(sd: dict, prefix: str) -> dict:
            return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}

        extra = {}
        repa_sd = subtree(full, "repa.projector.")
        if repa_sd:
            extra["repa_projector"] = repa_sd
        if data_state is not None:
            extra["data_position"] = data_state
        path = save_checkpoint(
            ckpt_dir / f"epoch_{epoch}_step_{step}.pth",
            subtree(full, "model."),
            optimizer=optim_sd,
            scheduler=scheduler,
            ema=subtree(ema_full, "model.") if ema_full is not None else None,
            epoch=epoch,
            step=step,
            config_dict=asdict(self.cfg),
            extra_state=extra or None,
            tmp_tag=self._ckpt_plan.tmp_tag,
        )
        accelerator.print(f"saved checkpoint {path}")
        prune_checkpoints(
            ckpt_dir,
            self.cfg.train.keep_last_checkpoints,
            self.cfg.train.milestone_steps,
        )

    def _save_dcp(
        self,
        accelerator,
        train_model,
        optimizer,
        scheduler,
        ema,
        ckpt_dir: Path,
        epoch,
        step,
        *,
        data_state: dict | None = None,
    ) -> None:
        """Sharded save: every rank writes its own shard, nothing is gathered.

        The payload is staged to host memory and written in the background, so
        the step loop pays one copy instead of a full-model write.
        """
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
        )

        self._drain_checkpoint(accelerator)  # one write in flight at a time
        state: dict = {}
        if ema is not None:
            # cloned: the swap unwinds before the writer reads the payload
            with ema.swapped():
                state["model_ema"] = clone_shards(get_model_state_dict(train_model))
        state["model"] = get_model_state_dict(train_model)
        state["optimizer"] = get_optimizer_state_dict(train_model, optimizer)
        path = ckpt_dir / f"epoch_{epoch}_step_{step}{DCP_SUFFIX}"
        meta = {
            "epoch": epoch,
            "step": step,
            "config": asdict(self.cfg),
            "scheduler": scheduler.state_dict(),
            "rng_state": _rng_state(),
        }
        if data_state is not None:
            meta["data_position"] = dict(data_state)
        self._pending_ckpt = save_dcp_checkpoint(
            path,
            state,
            meta=meta,
            write_meta=self._ckpt_plan.writes,
            process_group=self._dcp_group,
        )
        accelerator.print(f"sharded save started: {path.name}")

    def _drain_checkpoint(self, accelerator) -> None:
        """Finish an in-flight sharded write, publish its marker, prune.

        The barrier is what gives ``meta.pth`` its meaning: it is written only
        after every rank's shards are on disk, so resume resolution never picks
        a directory that is missing one.
        """
        pending = self._pending_ckpt
        if pending is None:
            return
        self._pending_ckpt = None
        pending.wait()
        accelerator.wait_for_everyone()
        pending.finalize()
        accelerator.print(f"saved checkpoint {pending.path}")
        if self._ckpt_plan.writes:
            prune_checkpoints(
                pending.path.parent,
                self.cfg.train.keep_last_checkpoints,
                self.cfg.train.milestone_steps,
            )

    def _restore(
        self,
        accelerator,
        train_model,
        core,
        repa,
        optimizer,
        scheduler,
        ema,
        ckpt_dir: Path,
        resume_path,
    ):
        """Resolve checkpoint IO, prove the directory is usable, then restore.

        Returns (ema, start_step, resumed).
        """
        cfg = self.cfg
        self._resumed_data_position = None
        self._resumed_checkpoint_config = None
        self._resumed_checkpoint_epoch = 0
        world = describe_world()
        plan = checkpoint_plan(
            cfg.train.dist.checkpoint_io,
            rank=world.rank,
            local_rank=world.local_rank,
            node_rank=world.node_rank,
            num_nodes=world.num_nodes,
        )
        self._ckpt_plan = plan
        self._pending_ckpt = None
        self._dcp_group = None
        accelerator.print(plan.describe())
        if plan.mode == "per_node" and cfg.train.dist.sharded:
            accelerator.print(
                "checkpoint_io=per_node consolidates the full state on every node's writer; "
                "checkpoint_io=dcp writes shards directly and gathers nothing"
            )
        # turns a silent resume failure hours from now into a startup error
        probe_shared_checkpoint_dir(ckpt_dir, plan, accelerator.wait_for_everyone)
        if plan.sharded_layout:
            self._dcp_group = dcp_process_group()  # collective: every rank, once
        if cfg.train.dist.sharded:
            return self._restore_sharded(accelerator, train_model, optimizer, scheduler, resume_path)
        start_step = 0
        resumed = False
        if resume_path is not None:
            require_checkpoint_kind(resume_path, plan.mode)
            extra = _aux_modules(repa)
            opt_prefix = None
            if cfg.train.resume_optimizer == "core":
                opt_prefix = len(list(core.parameters()))
                accelerator.print(
                    f"resume_optimizer=core: carrying optimizer state for {opt_prefix} core params"
                )
            ckpt_extra: dict = {}
            resume_payload = torch.load(
                str(resume_path),
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            if isinstance(resume_payload, dict):
                self._resumed_checkpoint_config = resume_payload.get("config")
            del resume_payload
            saved_epoch, start_step = load_checkpoint(
                resume_path,
                core,
                optimizer,
                scheduler,
                ema,
                weights_only_load=False,
                extra_modules=extra,
                optimizer_prefix=opt_prefix,
                extra_out=ckpt_extra,
            )
            self._resumed_data_position = ckpt_extra.get("data_position")
            self._resumed_checkpoint_epoch = saved_epoch
            resumed = True
            accelerator.print(f"resumed from {resume_path} at step {start_step}")
        if not resumed and cfg.train.load_from:
            require_dense_weights(cfg.train.load_from)
            extra = _aux_modules(repa)
            load_checkpoint(cfg.train.load_from, core, weights_only_load=True, extra_modules=extra)
            if ema is not None:
                ema.load_state_dict(core.state_dict())
            accelerator.print(f"loaded weights from {cfg.train.load_from}")
        return ema, start_step, resumed

    def _restore_sharded(self, accelerator, train_model, optimizer, scheduler, resume_path):
        """Restore weights/EMA/optimizer under fsdp2; all ranks collectively.

        Two-phase load when EMA is on: the checkpoint's EMA weights go into
        the live (sharded) model first, so the freshly built ShardEMA snapshot
        IS the average; the live weights are then loaded over them.

        Returns (ema, start_step, resumed).
        """
        cfg = self.cfg
        plan = self._ckpt_plan
        path = resume_path if resume_path is not None else cfg.train.load_from
        if path is None:
            ema = ShardEMA(train_model, cfg.train.ema.decay) if cfg.train.ema.enabled else None
            return ema, 0, False
        if resume_path is None:
            require_dense_weights(path)
        else:
            require_checkpoint_kind(path, plan.mode)
            if plan.sharded_layout:
                return self._restore_dcp(accelerator, train_model, optimizer, scheduler, path)

        # Every rank mmaps the payload: one page-cache copy per node instead of
        # one heap copy per rank. Under checkpoint_io=rank0 that file has to be
        # reachable from every node, which the startup probe has proved.
        payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
        if resume_path is not None:
            self._resumed_data_position = payload.get("data_position")
            self._resumed_checkpoint_config = payload.get("config")
            self._resumed_checkpoint_epoch = int(payload.get("epoch", 0))
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        # full_state_dict without broadcast_from_rank0: every rank holds the
        # payload already, so each shards its own slice straight out of it
        model_options = StateDictOptions(full_state_dict=True, strict=False)

        def load_model(state_dict: dict) -> None:
            set_model_state_dict(train_model, _wrapper_state_dict(state_dict, payload), options=model_options)

        ema = None
        if cfg.train.ema.enabled:
            load_model(payload.get("state_dict_ema", payload["state_dict"]))
            ema = ShardEMA(train_model, cfg.train.ema.decay)
        load_model(payload["state_dict"])
        if resume_path is None:
            accelerator.print(f"loaded weights from {path}")
            return ema, 0, False
        if "optimizer" in payload:
            opt_sd = payload["optimizer"]
            state_keys = list(opt_sd.get("state", {}))
            if state_keys and not all(isinstance(k, str) for k in state_keys):
                raise ValueError(
                    "checkpoint optimizer state is index-keyed (written by a replicated run); "
                    "a sharded resume needs a sharded-written checkpoint, or train.load_from "
                    "for weights only"
                )
            _load_fsdp2_optimizer_state(optimizer, opt_sd)
        if "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        if "rng_state" in payload:
            _restore_rng(payload["rng_state"])
        start_step = int(payload.get("step", 0))
        accelerator.print(f"resumed from {path} at step {start_step}")
        return ema, start_step, True

    def _restore_dcp(self, accelerator, train_model, optimizer, scheduler, path):
        """Sharded resume: every rank reads its own shard, nothing is gathered."""
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            get_optimizer_state_dict,
            set_model_state_dict,
        )

        cfg = self.cfg
        meta = load_dcp_meta(path)
        self._resumed_data_position = meta.get("data_position")
        self._resumed_checkpoint_config = meta.get("config")
        self._resumed_checkpoint_epoch = int(meta.get("epoch", 0))
        lenient = StateDictOptions(strict=False)
        ema = None
        if cfg.train.ema.enabled:
            slot = {"model_ema": get_model_state_dict(train_model)}
            load_dcp_state(path, slot, process_group=self._dcp_group)
            set_model_state_dict(train_model, slot["model_ema"], options=lenient)
            ema = ShardEMA(train_model, cfg.train.ema.decay)
        slot = {
            "model": get_model_state_dict(train_model),
            "optimizer": get_optimizer_state_dict(train_model, optimizer),
        }
        load_dcp_state(path, slot, process_group=self._dcp_group)
        set_model_state_dict(train_model, slot["model"], options=lenient)
        _load_fsdp2_optimizer_state(optimizer, slot["optimizer"])
        if "scheduler" in meta:
            scheduler.load_state_dict(meta["scheduler"])
        if "rng_state" in meta:
            _restore_rng(meta["rng_state"])
        start_step = int(meta.get("step", 0))
        accelerator.print(f"resumed from {path} at step {start_step}")
        return ema, start_step, True

    @torch.no_grad()
    def _validate(self, accelerator, core, text_encoder, vis_dir: Path, step: int, train_model=None) -> None:
        if self.cfg.train.dist.strategy == "fsdp2":
            # generate() drives the core module directly, so the root group's
            # embedders and head never see a pre-forward all-gather; unshard
            # them here. The blocks gather inside their own hooks, which are
            # collective, so every rank samples and only rank 0 writes.
            train_model.unshard()
            try:
                self._render_validation(
                    accelerator, core, text_encoder, vis_dir, step, write=accelerator.is_main_process
                )
            finally:
                train_model.reshard()
            return
        if not accelerator.is_main_process:
            return
        self._render_validation(accelerator, core, text_encoder, vis_dir, step)

    @torch.no_grad()
    def _render_validation(
        self, accelerator, core, text_encoder, vis_dir: Path, step: int, write: bool = True
    ) -> None:
        cfg = self.cfg
        was_training = core.training
        core.eval()
        prompts = list(cfg.train.validation_prompts)
        shape = self.render_shape
        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.train.seed)
        with accelerator.autocast():
            images = generate(
                core,
                text_encoder,
                prompts,
                height=shape.height,
                width=shape.width,
                steps=VALIDATION_STEPS,
                order=cfg.sample.order,
                cfg_scale=VALIDATION_CFG_SCALE,
                shift=self.flow_shift,
                generator=generator,
                device=accelerator.device,
                num_train_timesteps=cfg.flow.num_train_timesteps,
                prediction=cfg.flow.prediction,
            )
        core.train(was_training)

        if not write:
            return
        from PIL import Image

        frames = _to_uint8(images)
        out_path = vis_dir / f"vis_{step}.webp"
        Image.fromarray(_contact_sheet(frames)).save(out_path, format="WEBP")
        accelerator.print(f"validation sheet -> {out_path}")
        for tracker in accelerator.trackers:
            if getattr(tracker, "name", "") == "wandb":
                import wandb

                tracker.tracker.log(
                    {"validation": [wandb.Image(f, caption=p) for f, p in zip(frames, prompts, strict=True)]},
                    step=step,
                )

    @torch.no_grad()
    def _val_loss(self, accelerator, model, flow, text_encoder, val_loader) -> dict[str, float]:
        """Flow loss on the frozen holdout grid; runs on EVERY rank.

        Sample k always gets timestep index ``(k * VAL_T_STRIDE) % N`` and noise
        from a generator seeded by (VAL_GRID_SEED, k), where k is the sample's
        ABSOLUTE dataset index carried through collate. The grid is therefore a
        pure function of sample identity — invariant across steps, runs
        (including different timestep densities), world sizes, batch sizes, and any
        batching policy that reorders or regroups samples. Losses are also
        bucketed into four sigma-index quartiles.

        The noise field is drawn at the sample's own shape, so a series is
        comparable only within one (shape policy, image_size, flow.shift)
        triple; a resolution stage opens a new series, it does not continue one.
        """
        cfg = self.cfg
        device = accelerator.device
        n_t = cfg.flow.num_train_timesteps
        was_training = model.training
        model.eval()
        sums = torch.zeros(5, device=device)  # total + 4 quartile sums
        counts = torch.zeros(5, device=device)
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            enc = text_encoder.encode(list(batch["caption"]))
            y = enc.embeddings.to(device=device, dtype=images.dtype)
            y_mask = enc.mask.to(device=device)
            b = images.shape[0]
            ks = batch["index"]
            t_idx = (ks * VAL_T_STRIDE) % n_t
            noise = torch.empty_like(images)
            for j, k in enumerate(ks.tolist()):
                g = torch.Generator(device=device).manual_seed(mix_seed(VAL_GRID_SEED, k))
                noise[j] = torch.randn(images.shape[1:], generator=g, device=device, dtype=images.dtype)
            with accelerator.autocast():
                out = flow.training_loss(
                    model,
                    images,
                    y,
                    timestep_idx=t_idx.to(device),
                    noise=noise,
                    reduction="none",
                    model_kwargs={"y_mask": y_mask},
                )
            band = ((t_idx * 4) // n_t).to(device)
            sums[0] += out.loss.sum()
            counts[0] += b
            for q in range(4):
                mask = (band == q).float()
                sums[1 + q] += (out.loss * mask).sum()
                counts[1 + q] += mask.sum()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums)
            torch.distributed.all_reduce(counts)
        model.train(was_training)
        means = (sums / counts.clamp(min=1)).tolist()
        logs = {"val/loss": means[0]}
        logs.update({f"val/loss_t{q}": means[1 + q] for q in range(4)})
        return logs
