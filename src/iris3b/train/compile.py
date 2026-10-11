"""torch.compile application, shape budgeting, and recompilation telemetry.

Dynamo caches one compiled graph per set of guards, and a run that produces
many input shapes needs one cache entry per shape. When a frame exceeds
``torch._dynamo.config.recompile_limit`` (default 8) dynamo gives up on that
frame *permanently* and runs it eager for the rest of the process, after having
already paid autotuning for the first few shapes. Nothing raises and nothing is
printed at warning level unless the user happened to enable dynamo logging, so
the failure mode is a run that quietly gets slower than eager would have been.

Two mechanisms remove that:

- the cache budget is sized from the number of shapes the run will actually
  produce, read from the data shape policy rather than guessed;
- ``recompile_count`` reports guard failures out of dynamo's own bookkeeping so
  the trainer can log recompilations as a metric and shout once when a frame
  crosses its budget.

Compiling per block instead of per model also shrinks the blast radius: each
block is a separate frame with its own cache line, so N shapes cost N small
graphs per block instead of N whole-model graphs, and the autotune cost is paid
once per block shape rather than once per (H, W) grid.
"""

from dataclasses import dataclass

import torch
from torch import nn

from iris3b.config import DataConfig, PerfConfig
from iris3b.data.shapes import build_shape_policy

# Headroom above the shape count: dynamo also specializes on things that are
# not the input grid (autograd state between the train step and validation,
# nn.Module id-match guards), and a frame that hits the limit is unrecoverable.
_BUDGET_SLACK = 8


@dataclass(frozen=True)
class CompileDecision:
    """What ``compile_model`` resolved and applied, for a one-time printout."""

    scope: str
    dynamic: bool
    fullgraph: bool
    mode: str
    limit: int
    shape_count: int
    targets: int

    def __str__(self) -> str:
        shapes = "open" if self.shape_count == 0 else str(self.shape_count)
        return (
            f"compile: scope={self.scope} targets={self.targets} mode={self.mode} "
            f"dynamic={self.dynamic} fullgraph={self.fullgraph} "
            f"shapes={shapes} recompile_limit={self.limit}"
        )


_DECISION: CompileDecision | None = None
_WARNED = False


# -- shape budget ------------------------------------------------------------
def expected_shape_count(data_cfg: DataConfig, patch_size: int) -> int:
    """Number of distinct (H, W) inputs the run will produce; 0 when unbounded.

    The shape policy is the single authority on this: ``fixed`` yields one
    shape, ``bucket`` yields its table, and ``area`` derives shapes from the
    corpus so the set is open.
    """
    policy = build_shape_policy(
        data_cfg.resolved_shape_policy(),
        data_cfg.image_size,
        patch_size,
        data_cfg.aspect_ratio_bucket,
        data_cfg.shape_align,
        data_cfg.shape_max_ratio,
    )
    shapes = policy.shapes()
    if shapes is None:
        return 0
    return len(set(shapes))


def _resolve_dynamic(setting: str, shape_count: int) -> bool:
    if setting == "true":
        return True
    if setting == "false":
        return False
    # "auto": static is only safe when there is exactly one shape to specialize
    # on; an open shape set must be dynamic or every new grid burns a graph.
    return shape_count != 1


def _raise_dynamo_limits(target: int, instances: int) -> int:
    """Raise dynamo's cache budget to ``target``, return the per-frame result.

    ``recompile_limit`` and ``cache_size_limit`` are the new and old names for
    the same knob; on recent torch one aliases the other, on older torch only
    the old one exists. Write whichever are present and read back the max.

    ``instances`` is how many module instances are compiled through the same
    code object at most. The per-frame limit counts only cache entries guarding
    the same id-matched objects, but the accumulated limit counts every entry on
    that code object, so N identical blocks sharing one ``forward`` can need
    N * shapes entries whenever dynamo does fall back to id-match guards.
    """
    cfg = torch._dynamo.config
    names = ("recompile_limit", "cache_size_limit")
    current = max((getattr(cfg, n) for n in names if isinstance(getattr(cfg, n, None), int)), default=0)
    resolved = max(current, target)
    for name in names:
        if isinstance(getattr(cfg, name, None), int):
            setattr(cfg, name, resolved)
    accumulated = resolved * max(2, instances)
    for name in ("accumulated_recompile_limit", "accumulated_cache_size_limit"):
        current_acc = getattr(cfg, name, None)
        if isinstance(current_acc, int):
            setattr(cfg, name, max(current_acc, accumulated))
    return resolved


def _isolate_recompiles() -> None:
    """Give each call site its own recompile budget where torch supports it."""
    for holder in (torch._dynamo, getattr(torch, "compiler", None)):
        fn = getattr(holder, "set_isolate_recompiles", None) if holder is not None else None
        if callable(fn):
            fn()
            return


# -- compilation -------------------------------------------------------------
def _adapter_stacks(core: nn.Module) -> list[tuple[str, nn.Module]]:
    """The text adapter's transformer stacks, whichever adapter is configured."""
    embedder = getattr(core, "y_embedder", None)
    if embedder is None:
        return []
    stacks: list[tuple[str, nn.Module]] = []
    # the layerwise adapter attends over the frozen-encoder layer axis first,
    # then hands its pooled tokens to a nested token-axis refiner
    layer_blocks = getattr(embedder, "layer_blocks", None)
    if layer_blocks is not None:
        stacks.append(("y_embedder.layer_blocks", layer_blocks))
    refiner = getattr(embedder, "refiner", None)
    host, prefix = (refiner, "y_embedder.refiner") if refiner is not None else (embedder, "y_embedder")
    blocks = getattr(host, "blocks", None)
    if blocks is not None:
        stacks.append((f"{prefix}.blocks", blocks))
    return stacks


def compile_targets(core: nn.Module) -> list[tuple[str, nn.Module]]:
    """The submodules block-scope compilation applies to, in forward order."""
    targets: list[tuple[str, nn.Module]] = []
    stacks: list[tuple[str, nn.Module | None]] = [
        ("blocks", getattr(core, "blocks", None)),
        ("pixel_blocks", getattr(core, "pixel_blocks", None)),
    ]
    stacks.extend(_adapter_stacks(core))
    for prefix, stack in stacks:
        if stack is None:
            continue
        targets.extend((f"{prefix}.{i}", block) for i, block in enumerate(stack))
    return targets


def compile_model(core: nn.Module, perf_cfg: PerfConfig, *, shape_count: int) -> None:
    """Apply ``torch.compile`` to ``core`` in place.

    ``nn.Module.compile`` stores the compiled callable outside the module's
    parameter/buffer/submodule dicts, so ``state_dict()`` keys are unchanged and
    a module already wrapped by DDP keeps its wrapper and its hooks. Callers
    must have taken any eager deep copy (the EMA shadow) beforehand.

    ``shape_count`` is the number of distinct input shapes the run will produce
    (see ``expected_shape_count``); 0 means the set is open.
    """
    global _DECISION, _WARNED

    scope = perf_cfg.compile_scope
    targets = compile_targets(core) if scope == "block" else []
    if scope == "block" and not targets:
        # An unrecognized core would otherwise be left entirely uncompiled.
        scope = "model"
    units = len(targets) if scope == "block" else 1

    dynamic = _resolve_dynamic(perf_cfg.compile_dynamic, shape_count)
    limit = _raise_dynamo_limits(max(perf_cfg.recompile_limit, shape_count * 2 + _BUDGET_SLACK), units)
    _isolate_recompiles()

    kwargs = {
        "mode": perf_cfg.compile_mode,
        "dynamic": dynamic,
        "fullgraph": perf_cfg.compile_fullgraph,
    }
    if scope == "block":
        for _, block in targets:
            block.compile(**kwargs)
    else:
        core.compile(**kwargs)

    _WARNED = False
    _DECISION = CompileDecision(
        scope=scope,
        dynamic=dynamic,
        fullgraph=perf_cfg.compile_fullgraph,
        mode=perf_cfg.compile_mode,
        limit=limit,
        shape_count=shape_count,
        targets=units,
    )


def compile_decision() -> str:
    """One-line description of the applied compile settings."""
    if _DECISION is None:
        return "compile: not applied"
    line = str(_DECISION)
    if _DECISION.fullgraph:
        # fullgraph implies one_graph, which makes dynamo raise instead of
        # falling back to eager once a frame exhausts its cache budget
        line += " (limit hit raises)"
    return line


# -- telemetry ---------------------------------------------------------------
def _frame_recompiles(reasons: list) -> int:
    """Recompilations of one frame, from its recorded guard-failure reasons.

    Every reason is prefixed with the compile id of the cache entry that failed,
    and an entry only fails when a newer one replaces it, so the number of
    distinct ids is the number of times the frame was recompiled.
    """
    ids = {str(reason).split(":", 1)[0] for reason in reasons if reason}
    return len(ids)


def _guard_failure_frames() -> list[int] | None:
    failures = getattr(torch._dynamo, "guard_failures", None)
    if failures is None:
        return None
    return [_frame_recompiles(reasons) for reasons in list(failures.values())]


def graph_count() -> int:
    """Total graphs dynamo has compiled in this process."""
    counters = getattr(getattr(torch._dynamo, "utils", None), "counters", None)
    if counters is None:
        return 0
    return int(counters["stats"]["unique_graphs"])


def recompile_count() -> int:
    """Recompilations across every compiled frame."""
    frames = _guard_failure_frames()
    if frames is not None:
        return sum(frames)
    # No guard bookkeeping: every graph past the first per compiled unit is one.
    compiled_units = _DECISION.targets if _DECISION is not None else 0
    return max(0, graph_count() - compiled_units)


def max_frame_recompiles() -> int:
    """Recompilations of the worst single frame - the budget applies per frame."""
    frames = _guard_failure_frames()
    if not frames:
        return recompile_count()
    return max(frames)


def recompile_budget_warning() -> str | None:
    """Message for the first time a frame exhausts its cache budget, else None."""
    global _WARNED
    if _DECISION is None or _WARNED:
        return None
    worst = max_frame_recompiles()
    if worst < _DECISION.limit:
        return None
    _WARNED = True
    return (
        f"WARNING: a compiled frame recompiled {worst} times, at or past the "
        f"dynamo budget of {_DECISION.limit}. Frames that exceed it are dropped "
        f"to eager for the rest of the run. Raise train.perf.recompile_limit or "
        f"set train.perf.compile_dynamic=true."
    )
