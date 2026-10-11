"""Optimizers and batch-size-aware learning-rate scaling."""

import math
import re
from collections import defaultdict
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn

from iris3b.config import ADAMW_DEFAULT_BETAS, OptimizerConfig
from iris3b.models.blocks.mmdit import MMDiTBlock
from iris3b.models.blocks.pit import PiTBlock
from iris3b.models.blocks.single_stream import SingleStreamBlock
from iris3b.models.dit import FinalLayer
from iris3b.nn.attention import JointAttention, SelfAttention
from iris3b.nn.embeddings import (
    LayerwiseTextEmbedder,
    PatchEmbedder,
    PixelEmbedder,
    TextEmbedder,
    TimestepEmbedder,
    TransformerTextEmbedder,
)
from iris3b.nn.modulation import SharedCoreBias, SharedCoreModulation
from iris3b.registry import OPTIMIZERS


@OPTIMIZERS.register("adamw")
def adamw(
    params,
    lr: float,
    betas=ADAMW_DEFAULT_BETAS,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    fused: bool = False,
):
    """AdamW over the given parameters."""
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=tuple(betas),
        eps=eps,
        weight_decay=weight_decay,
        fused=fused,
    )


def _torch_release() -> tuple[int, int, int]:
    parts = [int(part) for part in re.findall(r"\d+", torch.__version__.split("+", 1)[0])[:3]]
    return tuple([*parts, *([0] * (3 - len(parts)))])


_MUON_MIN_RECOMPILE_LIMIT = 64


def _load_dion_muon():
    if _torch_release() < (2, 7, 1):
        raise RuntimeError(f"optimizer.name=muon requires torch>=2.7.1, found {torch.__version__}")
    torch._dynamo.config.recompile_limit = max(  # type: ignore[attr-defined]
        torch._dynamo.config.recompile_limit,
        _MUON_MIN_RECOMPILE_LIMIT,  # type: ignore[attr-defined]
    )
    try:
        from dion import Muon
    except ImportError as exc:
        raise RuntimeError(
            "optimizer.name=muon requires the pinned Dion dependency; "
            "reinstall the project with `pip install -e .`"
        ) from exc
    return Muon


def _record_split(
    splits: dict[int, tuple[int, ...]], parameter: nn.Parameter, sizes: tuple[int, ...]
) -> None:
    if parameter.ndim != 2 or sum(sizes) != parameter.shape[0]:
        raise ValueError(f"invalid Muon row split {sizes} for matrix shape {tuple(parameter.shape)}")
    previous = splits.get(id(parameter))
    if previous is not None and previous != sizes:
        raise ValueError(f"conflicting Muon row splits {previous} and {sizes}")
    splits[id(parameter)] = sizes


def _record_modulation_split(splits: dict[int, tuple[int, ...]], module: nn.Module, chunks: int) -> None:
    linears: list[nn.Linear] = []
    if isinstance(module, nn.Linear):
        linears.append(module)
    elif isinstance(module, SharedCoreModulation):
        linears.extend((module.core, module.adaln_up))
    elif isinstance(module, SharedCoreBias):
        linears.append(module.core)
    for linear in linears:
        width, remainder = divmod(linear.out_features, chunks)
        if remainder:
            raise ValueError(f"modulation output {linear.out_features} is not divisible into {chunks} chunks")
        _record_split(splits, linear.weight, (width,) * chunks)


def _muon_split_map(model: nn.Module) -> dict[int, tuple[int, ...]]:
    splits: dict[int, tuple[int, ...]] = {}
    for module in model.modules():
        if isinstance(module, SelfAttention):
            width = module.qkv.out_features // 3
            _record_split(splits, module.qkv.weight, (width, width, width))
        elif isinstance(module, JointAttention):
            if module.qkv_x is not None:
                width = module.qkv_x.out_features // 3
                sizes = (width, width, width)
                _record_split(splits, module.qkv_x.weight, sizes)
                _record_split(splits, module.qkv_y.weight, sizes)

        if isinstance(module, MMDiTBlock):
            _record_modulation_split(splits, module.adaln_img, 6)
            _record_modulation_split(splits, module.adaln_txt, 6)
        elif isinstance(module, SingleStreamBlock):
            if module.qkv is not None:
                width = module.qkv.out_features // 3
                _record_split(splits, module.qkv.weight, (width, width, width))
            _record_modulation_split(splits, module.adaln, 6)
    return splits


def _muon_boundary_ids(model: nn.Module) -> set[int]:
    boundary: set[int] = set()
    input_types = (PatchEmbedder, PixelEmbedder, TextEmbedder, TransformerTextEmbedder)
    for module in model.modules():
        if isinstance(module, input_types):
            boundary.add(id(module.proj.weight))
        elif isinstance(module, TimestepEmbedder):
            boundary.update(id(linear.weight) for linear in module.modules() if isinstance(linear, nn.Linear))
        elif isinstance(module, LayerwiseTextEmbedder):
            boundary.add(id(module.layer_pool.weight))
        elif isinstance(module, FinalLayer):
            boundary.add(id(module.linear.weight))
        elif isinstance(module, PiTBlock):
            # Its output rows are pixel-major with modulation chunks
            # interleaved inside each pixel, so contiguous Muon splits cannot
            # isolate shift/scale/gate subspaces without changing checkpoints.
            boundary.add(id(module.adaln.weight))
    return boundary


def build_muon_param_groups(
    model: nn.Module,
    params: Iterable[nn.Parameter],
    parameter_names: Iterable[str] | None = None,
) -> list[dict]:
    """Partition one training parameter set into hidden-matrix Muon and AdamW groups."""
    params = list(params)
    ids = [id(parameter) for parameter in params]
    if len(ids) != len(set(ids)):
        raise ValueError("optimizer parameter list contains duplicate Parameter objects")

    all_ids = set(ids)
    core_ids = {id(parameter) for parameter in model.parameters()}
    missing = core_ids - all_ids
    if missing:
        raise ValueError(f"optimizer parameter list omits {len(missing)} model parameters")
    if parameter_names is None:
        name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
        aux_index = 0
        for parameter in params:
            if id(parameter) not in name_by_id:
                name_by_id[id(parameter)] = f"aux.{aux_index}"
                aux_index += 1
    else:
        names = list(parameter_names)
        if len(names) != len(params) or len(names) != len(set(names)):
            raise ValueError("optimizer parameter names must be unique and align with parameters")
        name_by_id = {id(parameter): name for name, parameter in zip(names, params, strict=True)}

    linear_ids = {id(module.weight) for module in model.modules() if isinstance(module, nn.Linear)}
    boundary_ids = _muon_boundary_ids(model)
    split_map = _muon_split_map(model)

    muon: dict[tuple[int, ...] | None, list[nn.Parameter]] = defaultdict(list)
    adamw_core: list[nn.Parameter] = []
    adamw_aux: list[nn.Parameter] = []
    for parameter in params:
        pid = id(parameter)
        if pid in core_ids and pid in linear_ids and pid not in boundary_ids:
            muon[split_map.get(pid)].append(parameter)
        elif pid in core_ids:
            adamw_core.append(parameter)
        else:
            adamw_aux.append(parameter)

    groups: list[dict] = []
    split_keys = sorted((key for key in muon if key is not None), key=lambda key: (len(key), key))
    for split_sizes in [None, *split_keys]:
        matrices = muon.get(split_sizes)
        if not matrices:
            continue
        group = {
            "params": matrices,
            "algorithm": "muon",
            "iris_scope": "core",
            "iris_route": "hidden_matrix",
            "iris_param_names": tuple(name_by_id[id(parameter)] for parameter in matrices),
        }
        if split_sizes is not None:
            group["split_sizes"] = split_sizes
        groups.append(group)
    if adamw_core:
        groups.append(
            {
                "params": adamw_core,
                "algorithm": "adamw",
                "iris_scope": "core",
                "iris_route": "boundary_or_vector",
                "iris_param_names": tuple(name_by_id[id(parameter)] for parameter in adamw_core),
            }
        )
    groups.append(
        {
            "params": adamw_aux,
            "algorithm": "adamw",
            "iris_scope": "aux",
            "iris_route": "auxiliary",
            "iris_param_names": tuple(name_by_id[id(parameter)] for parameter in adamw_aux),
        }
    )
    return groups


def scale_lr(lr: float, rule: str, effective_batch_size: int, base_batch_size: int) -> float:
    """Rescale lr by the effective/base batch ratio (sqrt, linear, or none)."""
    if not rule or rule == "none":
        return lr
    ratio = effective_batch_size / base_batch_size
    if rule == "sqrt":
        return lr * math.sqrt(ratio)
    if rule == "linear":
        return lr * ratio
    raise ValueError(f"unknown auto_lr rule '{rule}'")


def _distributed_process_group():
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return dist.group.WORLD
    return None


def _muon_distributed_mesh(distributed_mesh):
    """What Dion's Muon should reduce over.

    Dion branches on the type it is handed: a DeviceMesh drives the DTensor
    path that matches per-parameter sharding, a ProcessGroup drives the
    replicated path. Passing a flat group while the parameters are DTensors
    would orthogonalize each local shard instead of the whole matrix, which is
    a different optimizer rather than a slower one.

    Muon only accepts a 1-D mesh. Under hybrid sharding the mesh is
    (dp_replicate, dp_shard) and only the shard dimension splits a parameter,
    so hand over that sub-mesh; the replicate dimension is already covered by
    the gradient reduction before the optimizer runs.
    """
    if distributed_mesh is None:
        return _distributed_process_group()
    if getattr(distributed_mesh, "ndim", 1) > 1:
        names = distributed_mesh.mesh_dim_names or ()
        if "dp_shard" not in names:
            raise ValueError(f"muon needs the 1-D sharded sub-mesh, but the mesh dimensions are {names}")
        return distributed_mesh["dp_shard"]
    return distributed_mesh


def build_optimizer(
    cfg: OptimizerConfig,
    params,
    effective_batch_size: int,
    fused: bool = False,
    *,
    model: nn.Module | None = None,
    parameter_names: Iterable[str] | None = None,
    distributed_mesh=None,
) -> tuple[torch.optim.Optimizer, float]:
    """Construct the configured optimizer with batch-scaled learning rates."""
    cfg.validate()
    params = list(params)
    lr = scale_lr(cfg.lr, cfg.auto_lr, effective_batch_size, cfg.base_batch_size)
    if fused and cfg.name != "adamw":
        raise ValueError(f"perf.fused_adamw only applies to the adamw optimizer, not '{cfg.name}'")
    if cfg.name == "adamw":
        opt = OPTIMIZERS.build(
            "adamw",
            params,
            lr=lr,
            betas=tuple(cfg.betas),
            weight_decay=cfg.weight_decay,
            fused=fused,
        )
    else:
        if model is None:
            raise ValueError("optimizer.name=muon requires the core model for parameter routing")
        muon_cls = _load_dion_muon()
        groups = build_muon_param_groups(model, params, parameter_names)
        process_group = _muon_distributed_mesh(distributed_mesh)
        adjust_lr = None if cfg.muon_adjust_lr == "none" else cfg.muon_adjust_lr
        opt = muon_cls(
            groups,
            distributed_mesh=process_group,
            lr=lr,
            mu=cfg.muon_momentum,
            betas=tuple(cfg.betas)[:2],
            weight_decay=cfg.weight_decay,
            epsilon=1.0e-8,
            nesterov=cfg.muon_nesterov,
            adjust_lr=adjust_lr,
            use_triton=True,
            use_polar_express=False,
        )
        opt.iris_routing = {
            algorithm: sum(
                parameter.numel()
                for group in groups
                if group["algorithm"] == algorithm
                for parameter in group["params"]
            )
            for algorithm in ("muon", "adamw")
        }
    return opt, lr
