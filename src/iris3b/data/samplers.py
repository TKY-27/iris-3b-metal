"""Sequential per-rank samplers and shape-homogeneous batch grouping."""

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator

from torch.utils.data import Sampler

_CACHE_ROOT = "~/.cache/iris_buckets"
# Fixed resume granularity: a rank owns a contiguous block of these chunks, so a
# checkpoint's data position reconstructs at any world size that divides it.
# 640 = lcm(16, 32, 40, 64, 80, 128), i.e. every 8-GPU node count in
# {2, 4, 5, 8, 10, 16}. A world size that does NOT divide it leaves the trailing
# chunks unassigned and loses their position on resume.
CANONICAL_CHUNKS = 640


_STATE_VERSION = 1
_SHAPE_STATE_VERSION = 2


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer, got {value!r}")
    return value


class RangedSampler(Sampler[int]):
    """Walk an ordered physical-range plan with exact world-portable resume."""

    def __init__(
        self,
        num_samples: int,
        rank: int = 0,
        world_size: int = 1,
        canonical_chunks: int = CANONICAL_CHUNKS,
    ):
        num_samples = _require_int("num_samples", num_samples)
        rank = _require_int("rank", rank)
        world_size = _require_int("world_size", world_size, minimum=1)
        canonical_chunks = _require_int("canonical_chunks", canonical_chunks, minimum=1)
        if rank >= world_size:
            raise ValueError(f"rank {rank} out of range for world size {world_size}")

        # Uniform chunks keep the covered samples independent of world size.
        # The tail shorter than one chunk is left unassigned.
        chunks = min(canonical_chunks, num_samples)
        if chunks < world_size:
            raise ValueError(
                f"canonical_chunks={chunks} is below the world size ({world_size}); a rank "
                f"must own at least one chunk"
            )
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.canonical_chunks = chunks
        self.chunk_size = num_samples // chunks
        self.bounds = [k * self.chunk_size for k in range(chunks + 1)]
        per_rank = chunks // world_size
        self.chunks = range(rank * per_rank, (rank + 1) * per_rank)
        self.start = self.bounds[self.chunks.start]
        self.end = self.bounds[self.chunks.stop]
        self.epoch = 0
        self._ranges_by_rank = self._normal_ranges_by_rank(world_size)
        self._progress_samples = 0

    def _normal_ranges_by_rank(self, world_size: int) -> tuple[tuple[tuple[int, int], ...], ...]:
        capacity = (self.canonical_chunks // world_size) * self.chunk_size
        return tuple(((rank * capacity, (rank + 1) * capacity),) for rank in range(world_size))

    @staticmethod
    def _ranges_length(ranges: tuple[tuple[int, int], ...]) -> int:
        return sum(hi - lo for lo, hi in ranges)

    @staticmethod
    def _trim_ranges(ranges: tuple[tuple[int, int], ...], skip: int) -> tuple[tuple[int, int], ...]:
        unread: list[tuple[int, int]] = []
        for lo, hi in ranges:
            span = hi - lo
            if skip >= span:
                skip -= span
                continue
            unread.append((lo + skip, hi))
            skip = 0
        return tuple(unread)

    @staticmethod
    def _split_ranges(
        ranges: list[tuple[int, int]], world_size: int, per_rank: int
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        if per_rank == 0:
            return tuple(() for _ in range(world_size))
        plans: list[list[tuple[int, int]]] = [[] for _ in range(world_size)]
        target_rank = 0
        target_left = per_rank
        for physical_lo, physical_hi in ranges:
            lo = physical_lo
            while lo < physical_hi:
                take = min(physical_hi - lo, target_left)
                hi = lo + take
                plan = plans[target_rank]
                if plan and plan[-1][1] == lo:
                    plan[-1] = (plan[-1][0], hi)
                else:
                    plan.append((lo, hi))
                lo = hi
                target_left -= take
                if target_left == 0:
                    target_rank += 1
                    if target_rank == world_size:
                        break
                    target_left = per_rank
        return tuple(tuple(plan) for plan in plans)

    def _reset_plan(self) -> None:
        self._ranges_by_rank = self._normal_ranges_by_rank(self.world_size)
        self._progress_samples = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _require_int("epoch", epoch)
        self._reset_plan()

    def set_start(self, start: int) -> None:
        start = _require_int("start", start)
        capacity = self._ranges_length(self._ranges_by_rank[self.rank])
        if start > capacity:
            raise ValueError(f"start={start} exceeds this rank's capacity of {capacity} samples")
        self._progress_samples = start

    def set_start_batches(self, batches: int, batch_size: int) -> None:
        """Resume ``batches`` emitted batches in; here that is a sample offset."""
        batches = _require_int("batches_per_rank", batches)
        batch_size = _require_int("batch_size", batch_size, minimum=1)
        self.set_start(batches * batch_size)

    def _require_divisible_world(self, world_size: int, label: str) -> None:
        if self.canonical_chunks % world_size:
            raise ValueError(
                f"canonical_chunks={self.canonical_chunks} must be divisible by the "
                f"{label} world_size={world_size} for an exact resume"
            )

    def _resume_layout(self, world_size: int) -> dict:
        world_size = _require_int("world_size", world_size, minimum=1)
        self._require_divisible_world(world_size, "prior")
        self._require_divisible_world(self.world_size, "current")
        return {
            "canonical_chunks": self.canonical_chunks,
            "num_samples": self.num_samples,
            "world_size": world_size,
        }

    def resume_state(
        self,
        batches_per_rank: int,
        batch_size: int,
        *,
        world_size: int | None = None,
        samples_per_rank: int | None = None,
    ) -> dict:
        """Serialize this plan plus equal progress within every rank's virtual walk."""
        batches_per_rank = _require_int("batches_per_rank", batches_per_rank)
        batch_size = _require_int("batch_size", batch_size, minimum=1)
        prior_world = (
            self.world_size if world_size is None else _require_int("world_size", world_size, minimum=1)
        )
        layout = self._resume_layout(prior_world)
        plans = (
            self._ranges_by_rank
            if prior_world == self.world_size
            else self._normal_ranges_by_rank(prior_world)
        )
        capacities = {self._ranges_length(ranges) for ranges in plans}
        if len(capacities) != 1:
            raise ValueError("resume plan must assign equal sample capacity to every rank")
        capacity = capacities.pop()
        nominal_samples = batches_per_rank * batch_size
        if samples_per_rank is None:
            samples_per_rank = nominal_samples
        else:
            samples_per_rank = _require_int("samples_per_rank", samples_per_rank)
            minimum_samples = max(0, nominal_samples - batch_size + 1)
            if not minimum_samples <= samples_per_rank <= nominal_samples:
                raise ValueError(
                    f"samples_per_rank={samples_per_rank} is incompatible with "
                    f"{batches_per_rank} batches of size {batch_size}; expected "
                    f"{minimum_samples}..{nominal_samples}"
                )
        if samples_per_rank > capacity:
            raise ValueError(
                f"samples_per_rank={samples_per_rank} exceeds the prior rank capacity of {capacity}"
            )
        return {
            "version": _STATE_VERSION,
            "kind": "contiguous_samples",
            **layout,
            "ranges_by_rank": [[[lo, hi] for lo, hi in ranges] for ranges in plans],
            "samples_per_rank": samples_per_rank,
        }

    def _validate_layout(self, layout: dict) -> int:
        if not isinstance(layout, dict):
            raise ValueError(f"resume layout must be a mapping, got {type(layout).__name__}")
        try:
            chunks = _require_int("canonical_chunks", layout["canonical_chunks"], minimum=1)
            num_samples = _require_int("num_samples", layout["num_samples"])
            prior_world = _require_int("world_size", layout["world_size"], minimum=1)
        except KeyError as exc:
            raise ValueError(f"resume state is missing {exc.args[0]!r}") from exc
        if chunks != self.canonical_chunks:
            raise ValueError(
                f"checkpoint used data.canonical_chunks={chunks} but this run uses "
                f"{self.canonical_chunks}; the partition is fixed for a run's lifetime"
            )
        if num_samples != self.num_samples:
            raise ValueError(
                f"checkpoint indexed {num_samples} samples but this dataset has "
                f"{self.num_samples}; the declared shard list is fixed for a run's lifetime"
            )
        self._require_divisible_world(prior_world, "prior")
        self._require_divisible_world(self.world_size, "current")
        return prior_world

    def _validate_plans(self, raw_plans: object, prior_world: int) -> tuple[tuple[tuple[int, int], ...], ...]:
        if not isinstance(raw_plans, (list, tuple)) or len(raw_plans) != prior_world:
            raise ValueError(f"ranges_by_rank must contain exactly {prior_world} rank plans")
        covered_end = self.bounds[-1]
        plans: list[tuple[tuple[int, int], ...]] = []
        all_ranges: list[tuple[int, int]] = []
        capacities: set[int] = set()
        for rank, raw_ranges in enumerate(raw_plans):
            if not isinstance(raw_ranges, (list, tuple)):
                raise ValueError(f"ranges_by_rank[{rank}] must be a sequence")
            plan: list[tuple[int, int]] = []
            previous_hi = -1
            for position, raw_range in enumerate(raw_ranges):
                if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                    raise ValueError(f"ranges_by_rank[{rank}][{position}] must be a [lo, hi] pair")
                lo = _require_int(f"ranges_by_rank[{rank}][{position}][0]", raw_range[0])
                hi = _require_int(f"ranges_by_rank[{rank}][{position}][1]", raw_range[1])
                if lo >= hi:
                    raise ValueError(f"ranges_by_rank[{rank}][{position}] must be nonempty")
                if hi > covered_end:
                    raise ValueError(
                        f"ranges_by_rank[{rank}][{position}] exceeds the covered corpus bound {covered_end}"
                    )
                if lo < previous_hi:
                    raise ValueError(f"ranges_by_rank[{rank}] must be sorted and non-overlapping")
                plan.append((lo, hi))
                all_ranges.append((lo, hi))
                previous_hi = hi
            frozen = tuple(plan)
            plans.append(frozen)
            capacities.add(self._ranges_length(frozen))
        if len(capacities) != 1:
            raise ValueError("ranges_by_rank must assign equal sample capacity to every rank")
        previous_hi = -1
        for lo, hi in sorted(all_ranges):
            if lo < previous_hi:
                raise ValueError("ranges_by_rank contains globally overlapping intervals")
            previous_hi = hi
        return tuple(plans)

    def load_resume_state(self, state: dict) -> None:
        """Trim equal source progress, then balance the exact unread plan."""
        if not isinstance(state, dict):
            raise ValueError(f"resume state must be a mapping, got {type(state).__name__}")
        kind = state.get("kind")
        if kind is not None:
            if kind != "contiguous_samples":
                raise ValueError(f"RangedSampler cannot restore resume state kind {kind!r}")
            version = _require_int("version", state.get("version"), minimum=1)
            if version != _STATE_VERSION:
                raise ValueError(
                    f"unsupported sampler resume state version {version}; expected {_STATE_VERSION}"
                )

        prior_world = self._validate_layout(state)
        try:
            samples_per_rank = _require_int("samples_per_rank", state["samples_per_rank"])
        except KeyError as exc:
            raise ValueError("resume state is missing 'samples_per_rank'") from exc
        if "ranges_by_rank" in state:
            plans = self._validate_plans(state["ranges_by_rank"], prior_world)
        elif kind is None:
            plans = self._normal_ranges_by_rank(prior_world)
        else:
            raise ValueError("resume state is missing 'ranges_by_rank'")

        capacity = self._ranges_length(plans[0])
        if samples_per_rank > capacity:
            raise ValueError(
                f"samples_per_rank={samples_per_rank} exceeds the prior rank capacity of {capacity}"
            )
        unread = [
            physical_range for plan in plans for physical_range in self._trim_ranges(plan, samples_per_rank)
        ]
        total_remaining = prior_world * (capacity - samples_per_rank)
        if total_remaining % self.world_size:
            raise ValueError(
                f"{total_remaining} remaining samples are not divisible by the current "
                f"world_size={self.world_size}; refusing unequal per-rank work"
            )
        remaining_per_rank = total_remaining // self.world_size
        self._ranges_by_rank = self._split_ranges(unread, self.world_size, remaining_per_rank)
        self._progress_samples = 0

    @property
    def ranges(self) -> tuple[tuple[int, int], ...]:
        return self._trim_ranges(self._ranges_by_rank[self.rank], self._progress_samples)

    @property
    def next_index(self) -> int | None:
        ranges = self.ranges
        return ranges[0][0] if ranges else None

    def __len__(self) -> int:
        return self._ranges_length(self.ranges)

    def __iter__(self) -> Iterator[int]:
        ranges = self.ranges
        try:
            for lo, hi in ranges:
                yield from range(lo, hi)
        finally:
            self._progress_samples = 0


def key_cache_path(data_dirs: list[str], world_size: int, rank: int, policy_id: str) -> str:
    """Cache file path keyed by data dirs, world layout, and shape policy."""
    spec = json.dumps([list(map(str, data_dirs)), world_size, rank, policy_id])
    digest = hashlib.sha256(spec.encode()).hexdigest()[:16]
    return os.path.join(os.path.expanduser(_CACHE_ROOT), f"{digest}.json")


class ShapeBatchSampler:
    """Groups a base sampler's indices into shape-homogeneous batches.

    An index joins the group named by the dataset's ``key_of``; a group is
    emitted as a batch once it holds ``batch_size`` indices. Keys are cached in
    memory and optionally persisted, because resolving one costs a metadata read
    against the shard.

    ``__len__`` is EXACT, not an upper bound: the epoch's plan is materialized
    once by walking the base sampler, so ``len()`` and iteration cannot
    disagree. That is load-bearing -- every rank must issue the same number of
    collectives, and DDP's gradient all-reduce, Dion Muon's own ``all_to_all``
    inside ``optimizer.step()`` and FSDP's clip-norm reduce all hang on a
    mismatch. ``limit_batches`` is how the trainer imposes the cross-rank
    minimum.

    A microbatch change replays the saved batch-prefix lineage, then groups the
    unconsumed suffix of each shape at the new size. The lineage is common to
    every rank; consumed per-shape counts are reconstructed locally, never
    serialized as rank-local or per-sample state.

    Indices are consumed in the base sampler's increasing order, so shards are
    still visited front to back; only the grouping is deferred.
    """

    def __init__(
        self,
        sampler: RangedSampler,
        dataset,
        batch_size: int,
        drop_last: bool = True,
        cache_path: str | None = None,
    ):
        batch_size = _require_int("batch_size", batch_size, minimum=1)
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.cache_path = cache_path
        self._key_cache: dict[str, str] = {}
        self._dirty = False
        self._plan: list[list[int]] | None = None
        self._limit: int | None = None
        self._skip_batches = 0
        self._installed_skip_batches = 0
        self._batch_history: list[tuple[int, int]] = []
        self._consumed_by_key: dict[str, int] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    self._key_cache = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._key_cache = {}

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)
        self._plan = None
        self._limit = None
        self._skip_batches = 0
        self._installed_skip_batches = 0
        self._batch_history = []
        self._consumed_by_key = {}

    def _batch_capacity(self) -> int:
        total = len(self.plan())
        return total if self._limit is None else min(total, self._limit)

    def _validate_batch_skip(self, batches: int) -> int:
        batches = _require_int("batches_per_rank", batches)
        capacity = self._batch_capacity()
        if batches > capacity:
            raise ValueError(
                f"batches_per_rank={batches} exceeds this rank's capacity of {capacity} shape batches"
            )
        return batches

    def set_start_batches(self, batches: int, batch_size: int) -> None:
        """Skip exact emitted batch ordinals while retaining partial key groups."""
        batch_size = _require_int("batch_size", batch_size, minimum=1)
        if batch_size != self.batch_size:
            raise ValueError(
                f"resume batch_size={batch_size} does not match the shape sampler's "
                f"batch_size={self.batch_size}"
            )
        self._installed_skip_batches = 0
        self._skip_batches = self._validate_batch_skip(batches)

    def resume_state(
        self,
        batches_per_rank: int,
        batch_size: int,
        *,
        world_size: int | None = None,
        samples_per_rank: int | None = None,
    ) -> dict:
        batch_size = _require_int("batch_size", batch_size, minimum=1)
        if batch_size != self.batch_size:
            raise ValueError(
                f"resume batch_size={batch_size} does not match the shape sampler's "
                f"batch_size={self.batch_size}"
            )
        plan_local_batches = _require_int("batches_per_rank", batches_per_rank)
        prior_world = (
            self.sampler.world_size
            if world_size is None
            else _require_int("world_size", world_size, minimum=1)
        )
        if prior_world != self.sampler.world_size:
            raise ValueError(
                "shape-batched resume requires an unchanged world size; "
                f"checkpoint world_size={prior_world}, current world_size={self.sampler.world_size}"
            )
        if samples_per_rank is not None:
            samples_per_rank = _require_int("samples_per_rank", samples_per_rank)
        batches_per_rank = self._validate_batch_skip(self._installed_skip_batches + plan_local_batches)
        if samples_per_rank is not None:
            expected_samples = (
                plan_local_batches * batch_size
                if self.drop_last
                else sum(len(self.plan()[i]) for i in range(self._installed_skip_batches, batches_per_rank))
            )
            if samples_per_rank != expected_samples:
                raise ValueError(
                    "shape-batched checkpoint samples_per_rank does not match emitted batches: "
                    f"got {samples_per_rank}, expected {expected_samples}"
                )
        return {
            "version": _SHAPE_STATE_VERSION,
            "kind": "shape_batches",
            "batches_per_rank": batches_per_rank,
            "batch_size": batch_size,
            "drop_last": self.drop_last,
            "batch_limit": self._limit,
            "regroup_history": [
                {"batch_size": size, "batches_per_rank": count} for size, count in self._batch_history
            ],
            "layout": self.sampler._resume_layout(prior_world),
        }

    def load_resume_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise ValueError(f"resume state must be a mapping, got {type(state).__name__}")
        if state.get("kind") != "shape_batches":
            raise ValueError(
                "ShapeBatchSampler requires resume state kind 'shape_batches'; "
                "a contiguous sample offset cannot restore grouped batches exactly"
            )
        version = _require_int("version", state.get("version"), minimum=1)
        if version not in (_STATE_VERSION, _SHAPE_STATE_VERSION):
            raise ValueError(f"unsupported shape sampler resume state version {version}")
        try:
            layout = state["layout"]
            batches_per_rank = _require_int("batches_per_rank", state["batches_per_rank"])
            batch_size = _require_int("batch_size", state["batch_size"], minimum=1)
        except KeyError as exc:
            raise ValueError(f"resume state is missing {exc.args[0]!r}") from exc
        prior_world = self.sampler._validate_layout(layout)
        if prior_world != self.sampler.world_size:
            raise ValueError(
                "shape-batched resume requires an unchanged world size; "
                f"checkpoint world_size={prior_world}, current world_size={self.sampler.world_size}"
            )
        history: list[tuple[int, int]] = []
        # Version 1 did not store a tail policy or limit. Its training policy was
        # unchanged on resume, and the trainer installs the full-epoch minimum
        # before loading it. Version 2 pins both, including a migrated epoch cap.
        limit = self._limit
        if version == _SHAPE_STATE_VERSION:
            try:
                drop_last = state["drop_last"]
                raw_history = state["regroup_history"]
                limit = state["batch_limit"]
            except KeyError as exc:
                raise ValueError(f"resume state is missing {exc.args[0]!r}") from exc
            if not isinstance(drop_last, bool) or drop_last != self.drop_last:
                raise ValueError("shape-batched resume requires unchanged drop_last")
            if not isinstance(raw_history, list):
                raise ValueError("regroup_history must be a list of batch-prefix cursors")
            if limit is not None:
                limit = _require_int("batch_limit", limit)
            for entry in raw_history:
                if not isinstance(entry, dict):
                    raise ValueError("regroup_history entries must be mappings")
                size = _require_int("history batch_size", entry.get("batch_size"), minimum=1)
                count = _require_int("history batches_per_rank", entry.get("batches_per_rank"))
                if count:
                    history.append((size, count))
        if limit is not None and batches_per_rank > limit and version == _SHAPE_STATE_VERSION:
            raise ValueError("batches_per_rank exceeds the checkpoint batch_limit")
        if batch_size != self.batch_size:
            if batches_per_rank:
                history.append((batch_size, batches_per_rank))
            batches_per_rank = 0
            # A full-epoch limit at the new size does not describe this regrouped
            # remainder. The trainer must agree its cross-rank minimum anew.
            limit = None

        self.sampler.set_start(0)
        consumed = self._replay_batch_history(history)
        self._batch_history = history
        self._consumed_by_key = consumed
        self._plan = None
        self._limit = limit
        self._installed_skip_batches = self._validate_batch_skip(batches_per_rank)
        self._skip_batches = self._installed_skip_batches

    def _replay_batch_history(self, history: list[tuple[int, int]]) -> dict[str, int]:
        """Recover consumed shape prefixes without materializing old batch plans.

        Only completed, yielded groups count as consumed. Other partially filled
        groups at a saved ordinal remain eligible, even if their indices precede
        the last emitted batch's final index.
        """
        consumed: dict[str, int] = {}
        for size, count in history:
            skip = consumed.copy()
            groups: dict[str, int] = {}
            emitted = 0
            for idx in self.sampler:
                key = self._key(idx)
                if key is None:
                    continue
                if skip.get(key, 0):
                    skip[key] -= 1
                    continue
                pending = groups.get(key, 0) + 1
                groups[key] = pending
                if pending == size:
                    groups[key] = 0
                    consumed[key] = consumed.get(key, 0) + size
                    emitted += 1
                    if emitted == count:
                        break
            else:
                if not self.drop_last:
                    for key, pending in groups.items():
                        if pending:
                            consumed[key] = consumed.get(key, 0) + pending
                            emitted += 1
                            if emitted == count:
                                break
            if emitted != count:
                raise ValueError(
                    f"history batches_per_rank={count} exceeds this rank's capacity "
                    f"of {emitted} shape batches at batch_size={size}"
                )
        return consumed

    def limit_batches(self, limit: int | None) -> None:
        """Truncate the upcoming pass to ``limit`` remaining batches."""
        self._limit = None if limit is None else self._skip_batches + _require_int("limit", limit)

    @property
    def start(self) -> int:
        return self.sampler.start

    @property
    def end(self) -> int:
        return self.sampler.end

    @property
    def ranges(self) -> tuple[tuple[int, int], ...]:
        return self.sampler.ranges

    @property
    def next_index(self) -> int | None:
        return self.sampler.next_index

    def _key(self, idx: int) -> str | None:
        key = self._key_cache.get(str(idx))
        if key is None:
            try:
                key = self.dataset.key_of(idx)
            except Exception:  # noqa: BLE001 - unreadable sample, skip it
                return None
            self._key_cache[str(idx)] = key
            self._dirty = True
        return key

    def plan(self) -> list[list[int]]:
        """The epoch's batches, materialized once and shared by len and iter."""
        if self._plan is not None:
            return self._plan
        groups: dict[str, list[int]] = {}
        batches: list[list[int]] = []
        skip = self._consumed_by_key.copy()
        for idx in self.sampler:
            key = self._key(idx)
            if key is None:
                continue
            if skip.get(key, 0):
                skip[key] -= 1
                continue
            group = groups.setdefault(key, [])
            group.append(idx)
            if len(group) == self.batch_size:
                batches.append(group)
                groups[key] = []
        if not self.drop_last:
            batches.extend(group for group in groups.values() if group)
        self._save_cache()
        self._plan = batches
        return batches

    def __len__(self) -> int:
        return self._batch_capacity() - self._skip_batches

    def __iter__(self) -> Iterator[list[int]]:
        capacity = self._batch_capacity()
        plan = self.plan()
        skip, self._skip_batches = self._skip_batches, 0
        for i in range(skip, capacity):
            yield plan[i]

    def _save_cache(self) -> None:
        if not (self.cache_path and self._dirty):
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.cache_path), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(self._key_cache, f)
            os.replace(tmp, self.cache_path)
            self._dirty = False
        except OSError:
            pass
