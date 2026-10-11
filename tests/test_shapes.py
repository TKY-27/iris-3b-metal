"""Shape policies, shape-homogeneous batching, and the resolution shift law."""

import json
from dataclasses import replace

import pytest
import torch

from iris3b.config import DataConfig, load_config
from iris3b.data.buckets import SHARED_21_512, TRAIN_BUCKETS
from iris3b.data.builder import collate_batch
from iris3b.data.samplers import RangedSampler, ShapeBatchSampler
from iris3b.data.shapes import (
    AreaPolicy,
    BucketPolicy,
    FixedSquarePolicy,
    Shape,
    build_shape_policy,
)
from iris3b.flow.schedule import FlowSchedule, resolution_shift, shift_sigma
from tiny_config import tiny_model_config

STAGE1 = "configs/iris3b/stage1_256.yaml"


# --------------------------------------------------------------------------
# 1. Shape
# --------------------------------------------------------------------------


def test_shape_grid_and_tokens_are_componentwise():
    shape = Shape(256, 1024)
    assert shape.grid(16) == (16, 64)
    assert shape.tokens(16) == 1024
    assert shape.key == "256x1024"
    # the transpose is a DIFFERENT shape with the same token count
    assert Shape(1024, 256).grid(16) == (64, 16)
    assert Shape(1024, 256).key != shape.key


def test_shape_rejects_a_grid_that_is_not_patchable():
    with pytest.raises(ValueError, match="not divisible by patch 16"):
        Shape(258, 256).grid(16)


# --------------------------------------------------------------------------
# 2. policies
# --------------------------------------------------------------------------


def test_fixed_policy_ignores_the_native_size_and_is_uniform():
    policy = FixedSquarePolicy(256)
    assert policy.uniform
    assert policy.target(4000, 3000) == policy.target(100, 900) == Shape(256, 256)
    assert policy.key(4000, 3000) == policy.key(100, 900)


def test_bucket_policy_snaps_to_the_shipped_table():
    policy = BucketPolicy("shared21-512")
    assert not policy.uniform
    assert policy.target(512, 512) == Shape(512, 512)
    # a 2:1 portrait lands on a portrait bucket, not its transpose
    tall = policy.target(1400, 700)
    assert tall.height > tall.width
    assert [float(tall.height), float(tall.width)] in SHARED_21_512.values()
    assert all(s.height % 16 == 0 and s.width % 16 == 0 for s in policy.shapes())


def test_bucket_policy_rejects_an_unknown_table():
    with pytest.raises(ValueError, match="unknown bucket table 'nope'"):
        BucketPolicy("nope")


@pytest.mark.parametrize("table", sorted(TRAIN_BUCKETS))
def test_every_bucket_cell_is_its_own_nearest_bucket(table: str):
    """Compiled records are keyed by their final cell through ``policy.key``.

    That only groups a record with its own cell if snapping a cell's exact
    dimensions returns that cell, for every cell of every shipped table.
    """
    policy = BucketPolicy(table)
    shapes = policy.shapes()
    assert len({s.key for s in shapes}) == len(shapes)
    for shape in shapes:
        assert policy.target(shape.height, shape.width) == shape
        assert policy.key(shape.height, shape.width) == shape.key


def test_area_policy_holds_the_token_budget_without_a_table():
    policy = AreaPolicy(tokens=1024, patch=16, align=32)
    assert not policy.uniform
    assert policy.shapes() is None  # the realized set is open, by construction
    for native in [(512, 512), (683, 1024), (1024, 683), (900, 1600), (4000, 3000)]:
        shape = policy.target(*native)
        assert shape.height % 32 == 0 and shape.width % 32 == 0
        # within one alignment step of the budget in each direction
        assert 0.55 <= shape.tokens(16) / 1024 <= 1.6


def test_area_policy_preserves_orientation_and_clamps_extremes():
    policy = AreaPolicy(tokens=1024, patch=16, align=32, max_ratio=4.0)
    wide = policy.target(400, 1600)
    assert wide.width > wide.height
    tall = policy.target(1600, 400)
    assert (tall.height, tall.width) == (wide.width, wide.height)
    # a 20:1 panorama is clamped to 4:1, not turned into a 1-patch strip
    panorama = policy.target(200, 4000)
    assert panorama.width / panorama.height == pytest.approx(4.0, rel=0.2)


def test_area_policy_requires_align_to_be_a_patch_multiple():
    with pytest.raises(ValueError, match="must be a positive multiple of patch"):
        AreaPolicy(tokens=1024, patch=16, align=24)


def test_build_shape_policy_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown data.shape_policy 'packed'"):
        build_shape_policy("packed", 256, 16, "shared21-512", 32, 4.0)


# --------------------------------------------------------------------------
# 3. config seam
# --------------------------------------------------------------------------


def test_every_shipped_config_resolves_to_a_known_policy():
    import glob

    paths = sorted(glob.glob("configs/iris3b/*.yaml"))
    for path in paths:
        cfg = load_config(path)
        cfg.data.validate(cfg.model.patch_size)


def test_validate_rejects_a_resolution_the_patchifier_cannot_represent():
    cfg = DataConfig(type="pixel", image_size=250)
    with pytest.raises(ValueError, match="not divisible by model.patch_size 16"):
        cfg.validate(16)


# --------------------------------------------------------------------------
# 4. shape-homogeneous batching
# --------------------------------------------------------------------------


class _KeyedDataset:
    """Minimal dataset exposing only what the batch sampler consumes."""

    def __init__(self, keys: list[str]):
        self.keys = keys

    def __len__(self) -> int:
        return len(self.keys)

    def key_of(self, idx: int) -> str:
        return self.keys[idx]


def test_batches_are_shape_homogeneous_and_index_monotone():
    keys = ["a", "b", "a", "b", "a", "b", "a", "b"]
    sampler = ShapeBatchSampler(RangedSampler(8), _KeyedDataset(keys), batch_size=2)
    batches = list(sampler)
    assert batches == [[0, 2], [1, 3], [4, 6], [5, 7]]
    for batch in batches:
        assert len({keys[i] for i in batch}) == 1
        assert batch == sorted(batch)  # tar readers still see a monotone walk


def test_len_equals_the_number_of_emitted_batches():
    """len() must count only emitted batches; over-reporting hangs DDP."""
    keys = ["a"] * 5 + ["b"] * 3  # 8 samples, bs 2 -> 2 + 1 = 3 batches, not 4
    sampler = ShapeBatchSampler(RangedSampler(8), _KeyedDataset(keys), batch_size=2)
    assert len(sampler) == 3 == len(list(sampler))


def test_limit_batches_truncates_both_len_and_iteration():
    keys = ["a"] * 8
    sampler = ShapeBatchSampler(RangedSampler(8), _KeyedDataset(keys), batch_size=2)
    assert len(sampler) == 4
    sampler.limit_batches(2)
    assert len(sampler) == 2 == len(list(sampler))
    assert list(sampler) == [[0, 1], [2, 3]]


def test_shape_resume_by_batch_ordinal_is_exact_at_the_same_world():
    """Shape resume skips emitted batches, not raw contiguous samples."""
    keys = ["a", "b", "a", "b"]
    dataset = _KeyedDataset(keys)
    source = ShapeBatchSampler(RangedSampler(4), dataset, batch_size=2)
    full = list(source)
    assert full == [[0, 2], [1, 3]]
    state = source.resume_state(1, 2)
    assert state["kind"] == "shape_batches"

    resumed = ShapeBatchSampler(RangedSampler(4), dataset, batch_size=2)
    resumed.load_resume_state(state)
    assert len(resumed) == 1
    assert list(resumed) == full[1:]
    assert resumed.resume_state(1, 2)["batches_per_rank"] == 2
    # The skip applies to one pass only.
    assert list(resumed) == full


def test_shape_resume_rejects_a_changed_world():
    dataset = _KeyedDataset(["a"] * 8)
    source = ShapeBatchSampler(RangedSampler(8), dataset, batch_size=2)
    state = source.resume_state(1, 2)

    changed_world = ShapeBatchSampler(RangedSampler(8, rank=0, world_size=2), dataset, batch_size=2)
    with pytest.raises(ValueError, match="unchanged world size"):
        changed_world.load_resume_state(state)


def _remaining_shape_batches(keys, indices, consumed, batch_size, *, drop_last=True):
    groups = {}
    batches = []
    for index in indices:
        if index in consumed:
            continue
        group = groups.setdefault(keys[index], [])
        group.append(index)
        if len(group) == batch_size:
            batches.append(group[:])
            group.clear()
    if not drop_last:
        batches.extend(group[:] for group in groups.values() if group)
    return batches


def test_shape_microbatch_regroup_preserves_partial_buffers_and_recheckpoint():
    keys = ["a", "b"] * 16
    dataset = _KeyedDataset(keys)
    source = ShapeBatchSampler(RangedSampler(32), dataset, batch_size=4)
    source.limit_batches(3)
    state = source.resume_state(1, 4)

    migrated = ShapeBatchSampler(RangedSampler(32), dataset, batch_size=8)
    migrated.limit_batches(1)  # A fresh-plan limit is not a regrouped-plan limit.
    migrated.load_resume_state(state)
    expected = [
        list(range(1, 16, 2)),
        list(range(8, 23, 2)),
        list(range(17, 32, 2)),
    ]
    # The b-buffer [1, 3, 5] was read while forming the old a-batch but not
    # yielded; it must survive. Only the new a-tail [24, 26, 28, 30] is dropped.
    assert len(migrated) == len(expected)
    assert list(migrated) == expected
    for cut in range(len(expected) + 1):
        checkpoint = json.loads(json.dumps(migrated.resume_state(cut, 8, samples_per_rank=cut * 8)))
        resumed = ShapeBatchSampler(RangedSampler(32), dataset, batch_size=8)
        resumed.load_resume_state(checkpoint)
        assert len(resumed) == len(expected) - cut
        assert list(resumed) == expected[cut:]
        saved_again = resumed.resume_state(0, 8, samples_per_rank=0)
        resumed.load_resume_state(saved_again)
        assert list(resumed) == expected[cut:]


def test_shape_regroup_shared_world64_checkpoint_preserves_each_rank_and_limit():
    keys = [key for rank in range(64) for key in ["a"] * (10 + rank % 12) + ["b"] * (70 - rank % 12)]
    dataset = _KeyedDataset(keys)

    def sampler(rank, batch_size):
        return ShapeBatchSampler(RangedSampler(len(keys), rank, 64), dataset, batch_size)

    source = sampler(0, 4)
    source.limit_batches(min(len(sampler(rank, 4)) for rank in range(64)))
    shared = source.resume_state(3, 4)
    migrated = []
    references = []
    for rank in range(64):
        old_batches = list(sampler(rank, 4))
        consumed = {index for batch in old_batches[:3] for index in batch}
        current = sampler(rank, 8)
        current.load_resume_state(shared)
        expected = _remaining_shape_batches(keys, range(current.start, current.end), consumed, 8)
        assert list(current) == expected
        migrated.append(current)
        references.append(expected)

    agreed = min(map(len, references))
    assert len({len(reference) for reference in references}) > 1
    for current, reference in zip(migrated, references, strict=True):
        current.limit_batches(agreed)
        assert list(current) == reference[:agreed]

    shared = json.loads(json.dumps(migrated[0].resume_state(2, 8, samples_per_rank=16)))
    assert len(json.dumps(shared)) < 1024  # No per-sample or per-rank progress vector.
    for rank, reference in enumerate(references):
        resumed = sampler(rank, 8)
        resumed.limit_batches(1)
        resumed.load_resume_state(shared)
        assert len(resumed) == agreed - 2
        assert list(resumed) == reference[2:agreed]
        saved_again = resumed.resume_state(1, 8, samples_per_rank=8)
        resumed.load_resume_state(saved_again)
        assert list(resumed) == reference[3:agreed]

    # A different shape mix means rank 0's derived per-key counts cannot be
    # substituted for another rank's. The same serialized cursor must work.
    shared_again = migrated[0].resume_state(3, 8, samples_per_rank=24)
    final = sampler(63, 8)
    final.load_resume_state(shared_again)
    assert list(final) == references[63][3:agreed]
    final.set_epoch(2)
    normal = list(sampler(63, 8))
    assert list(final) == normal
    final.limit_batches(2)
    checkpoint = final.resume_state(1, 8)
    final.load_resume_state(checkpoint)
    assert list(final) == normal[1:2]


def test_shape_regroup_lineage_survives_further_batch_changes():
    keys = ["a", "b", "b", "c", "a", "c", "a", "b"] * 16
    dataset = _KeyedDataset(keys)
    source = ShapeBatchSampler(RangedSampler(len(keys)), dataset, batch_size=4)
    consumed = {index for batch in list(source)[:2] for index in batch}
    checkpoint = source.resume_state(2, 4)
    for size in (8, 3, 5):
        current = ShapeBatchSampler(RangedSampler(len(keys)), dataset, batch_size=size)
        current.load_resume_state(checkpoint)
        expected = _remaining_shape_batches(keys, range(len(keys)), consumed, size)
        assert list(current) == expected
        consumed.update(index for batch in expected[:2] for index in batch)
        checkpoint = current.resume_state(2, size, samples_per_rank=2 * size)
        current.load_resume_state(json.loads(json.dumps(checkpoint)))
        assert list(current) == expected[2:]


def test_shape_regroup_retains_and_records_partial_final_batches():
    keys = ["a", "b", "a", "b", "c", "c", "c", "d", "d"]
    dataset = _KeyedDataset(keys)
    source = ShapeBatchSampler(RangedSampler(9), dataset, batch_size=2, drop_last=False)
    current = ShapeBatchSampler(RangedSampler(9), dataset, batch_size=4, drop_last=False)
    current.load_resume_state(source.resume_state(2, 2, samples_per_rank=4))
    assert list(current) == [[4, 5, 6], [7, 8]]
    checkpoint = current.resume_state(1, 4, samples_per_rank=3)
    current.load_resume_state(checkpoint)
    assert list(current) == [[7, 8]]
    with pytest.raises(ValueError, match="does not match emitted batches"):
        current.resume_state(1, 4, samples_per_rank=4)
    checkpoint = current.resume_state(1, 4, samples_per_rank=2)
    final = ShapeBatchSampler(RangedSampler(9), dataset, batch_size=1, drop_last=False)
    final.load_resume_state(checkpoint)
    assert list(final) == []


def test_shape_regroup_rejects_unreplayable_progress_and_changed_tail_policy():
    dataset = _KeyedDataset(["a", "b"] * 16)
    source = ShapeBatchSampler(RangedSampler(32), dataset, batch_size=4)
    state = source.resume_state(1, 4)
    current = ShapeBatchSampler(RangedSampler(32), dataset, batch_size=8)
    bad_history = {
        **state,
        "regroup_history": [{"batch_size": 4, "batches_per_rank": 100}],
    }
    with pytest.raises(ValueError, match="exceeds this rank's capacity"):
        current.load_resume_state(bad_history)
    with pytest.raises(ValueError, match="batch_limit"):
        current.load_resume_state({**state, "batch_limit": 0})
    changed_policy = ShapeBatchSampler(RangedSampler(32), dataset, batch_size=8, drop_last=False)
    with pytest.raises(ValueError, match="drop_last"):
        changed_policy.load_resume_state(state)


def test_ranged_sampler_resume_is_still_a_sample_offset():
    sampler = RangedSampler(8)
    sampler.set_start_batches(2, 2)
    assert len(sampler) == 4
    assert list(sampler) == [4, 5, 6, 7]


# --------------------------------------------------------------------------
# 5. collate carries sample identity
# --------------------------------------------------------------------------


def test_collate_carries_the_absolute_index():
    items = [
        {
            "image": torch.zeros(3, 32, 64),
            "caption": f"c{k}",
            "index": k,
            "img_hw": torch.tensor([100, 200]),
            "aspect_ratio": 0.5,
        }
        for k in (7, 3, 11)
    ]
    batch = collate_batch(items)
    assert batch["index"].tolist() == [7, 3, 11]
    assert batch["image"].shape == (3, 3, 32, 64)


# --------------------------------------------------------------------------
# 6. resolution shift law
# --------------------------------------------------------------------------


def test_shift_law_none_is_the_identity():
    assert resolution_shift(4096, "none", 1.0) == 1.0
    assert resolution_shift(4096, "none", 3.0) == 3.0


def test_sd3_law_scales_with_the_linear_side():
    """SD3 Eq.23: alpha = sqrt(m/n), so 2x resolution is 2x shift."""
    assert resolution_shift(256, "sd3", 1.0) == pytest.approx(1.0)
    assert resolution_shift(1024, "sd3", 1.0) == pytest.approx(2.0)
    assert resolution_shift(4096, "sd3", 1.0) == pytest.approx(4.0)


def test_flux_law_matches_the_shipped_calibration():
    """mu affine in token count, anchored on (256, 0.5) and (4096, 1.15)."""
    assert resolution_shift(256, "flux", 1.0) == pytest.approx(1.6487, abs=1e-3)
    assert resolution_shift(1024, "flux", 1.0) == pytest.approx(1.8776, abs=1e-3)
    assert resolution_shift(4096, "flux", 1.0) == pytest.approx(3.1582, abs=1e-3)


def test_shift_law_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown flow.shift_law 'linear'"):
        resolution_shift(1024, "linear", 1.0)


def test_shift_is_a_pure_log_snr_translation():
    """s'/(1-s') == shift * s/(1-s): the property the law is calibrated against."""
    sigma = torch.linspace(0.01, 0.99, 21, dtype=torch.float64)
    for shift in (1.0, 2.0, 3.1582):
        shifted = shift_sigma(sigma, shift)
        torch.testing.assert_close(shifted / (1 - shifted), shift * sigma / (1 - sigma), rtol=1e-9, atol=1e-9)


def test_flow_resolves_the_stage_shift_from_token_count():
    from iris3b.flow import RectifiedFlow

    cfg = load_config(STAGE1)
    flow_cfg = replace(cfg.flow, shift=1.0, shift_law="sd3")
    flow = RectifiedFlow(flow_cfg, tokens=1024)
    assert flow.shift == pytest.approx(2.0)
    # the resolved value is what the schedule was built with, not the raw config
    reference = FlowSchedule(cfg.flow.num_train_timesteps, 2.0)
    torch.testing.assert_close(flow.schedule.sigmas, reference.sigmas)


# --------------------------------------------------------------------------
# 7. shape-varying policies reach the model
# --------------------------------------------------------------------------


def _area_data_cfg() -> DataConfig:
    return DataConfig(type="synthetic", shape_policy="area", image_size=64, shape_align=8, num_workers=0)


def test_area_policy_really_produces_more_than_one_shape():
    """Shape-varying training is only exercised if the area policy yields several shapes."""
    from iris3b.data.datasets import SyntheticDataset

    cfg = _area_data_cfg()
    dataset = SyntheticDataset(cfg, patch_size=4)
    shapes = {tuple(dataset[i]["image"].shape[-2:]) for i in range(len(dataset._NATIVE))}
    assert len(shapes) > 1
    for height, width in shapes:
        assert height % 4 == 0 and width % 4 == 0


def test_model_accepts_every_shape_an_area_policy_can_emit():
    """Non-square grids must reach the head, not just the patch embedder."""
    from iris3b.data.datasets import SyntheticDataset
    from iris3b.models import IrisDiT

    cfg = _area_data_cfg()
    model = IrisDiT(tiny_model_config()).eval()
    dataset = SyntheticDataset(cfg, patch_size=model.cfg.patch_size)
    for i in range(len(dataset._NATIVE)):
        image = dataset[i]["image"][None]
        out = model(image, torch.tensor([5.0]), torch.randn(1, 16, 32))
        assert out.x.shape == image.shape
        assert torch.isfinite(out.x).all()


def test_model_refuses_a_shape_the_patchifier_would_silently_truncate():
    from iris3b.models import IrisDiT

    tiny = tiny_model_config()
    model = IrisDiT(replace(tiny, pixel=replace(tiny.pixel, enabled=False)))
    with pytest.raises(ValueError, match="is not divisible by patch_size 4"):
        model(torch.randn(1, 3, 34, 32), torch.tensor([5.0]), torch.randn(1, 16, 32))


def test_shift_law_requires_the_stage_token_count():
    from iris3b.flow import RectifiedFlow

    cfg = load_config(STAGE1)
    with pytest.raises(ValueError, match="requires the stage token count"):
        RectifiedFlow(replace(cfg.flow, shift_law="sd3"))


_PORTABLE_N = 100_000_000
_PORTABLE_WORLDS = (16, 32, 40, 64, 80, 128)


def _merge_intervals(spans: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def _interval_total(spans) -> int:
    return sum(hi - lo for lo, hi in spans)


def _complement(spans, end: int) -> tuple[tuple[int, int], ...]:
    cursor = 0
    complement = []
    for lo, hi in spans:
        if cursor < lo:
            complement.append((cursor, lo))
        cursor = hi
    if cursor < end:
        complement.append((cursor, end))
    return tuple(complement)


def _state_unread(state: dict) -> tuple[tuple[int, int], ...]:
    progress = state["samples_per_rank"]
    unread = []
    for plan in state["ranges_by_rank"]:
        left = progress
        for lo, hi in plan:
            span = hi - lo
            if left >= span:
                left -= span
            else:
                unread.append((lo + left, hi))
                left = 0
    return _merge_intervals(unread)


def _restored_snapshot(state: dict, world: int):
    from iris3b.data.samplers import CANONICAL_CHUNKS

    samplers = [RangedSampler(_PORTABLE_N, rank, world, CANONICAL_CHUNKS) for rank in range(world)]
    for sampler in samplers:
        sampler.load_resume_state(state)
    raw_unread = [span for sampler in samplers for span in sampler.ranges]
    merged_unread = _merge_intervals(raw_unread)
    assert _interval_total(raw_unread) == _interval_total(merged_unread)
    return samplers, merged_unread


def test_resume_position_survives_every_supported_world_transition():
    """Every NxN transition preserves the exact unread/consumed set and equal work."""
    from iris3b.data.samplers import CANONICAL_CHUNKS

    global_batch, batches = 640, 5000
    for prior_world in _PORTABLE_WORLDS:
        batch_size = global_batch // prior_world
        source = RangedSampler(_PORTABLE_N, 0, prior_world, CANONICAL_CHUNKS)
        state = source.resume_state(batches, batch_size)
        expected_unread = _state_unread(state)
        expected_consumed = _complement(expected_unread, source.bounds[-1])
        assert _interval_total(expected_consumed) == batches * global_batch

        for current_world in _PORTABLE_WORLDS:
            samplers, restored_unread = _restored_snapshot(state, current_world)
            restored_consumed = _complement(restored_unread, source.bounds[-1])
            assert restored_unread == expected_unread
            assert restored_consumed == expected_consumed
            assert _interval_total(restored_consumed) == batches * global_batch
            assert len({len(sampler) for sampler in samplers}) == 1
            if current_world == prior_world:
                for rank, sampler in enumerate(samplers):
                    plan = tuple(map(tuple, state["ranges_by_rank"][rank]))
                    assert sampler.ranges == RangedSampler._trim_ranges(plan, state["samples_per_rank"])


def test_resume_plan_is_closed_under_resize_and_recheckpoint():
    from iris3b.data.samplers import CANONICAL_CHUNKS

    initial = RangedSampler(_PORTABLE_N, 0, 16, CANONICAL_CHUNKS).resume_state(5000, 40)
    mid = RangedSampler(_PORTABLE_N, 0, 32, CANONICAL_CHUNKS)
    mid.load_resume_state(initial)
    recheckpointed = mid.resume_state(100, 2)
    expected_unread = _state_unread(recheckpointed)

    final, restored_unread = _restored_snapshot(recheckpointed, 64)
    assert restored_unread == expected_unread
    assert len({len(sampler) for sampler in final}) == 1
    assert _interval_total(_complement(restored_unread, mid.bounds[-1])) == (5000 * 640 + 100 * 2 * 32)


def test_resume_refuses_invalid_layout_progress_or_divisibility():
    from copy import deepcopy

    from iris3b.data.samplers import CANONICAL_CHUNKS

    source = RangedSampler(_PORTABLE_N, 0, 16, CANONICAL_CHUNKS)
    state = source.resume_state(5000, 40)
    with pytest.raises(ValueError, match="canonical_chunks"):
        RangedSampler(_PORTABLE_N, 0, 16, CANONICAL_CHUNKS // 2).load_resume_state(state)
    with pytest.raises(ValueError, match="samples"):
        RangedSampler(_PORTABLE_N - 1, 0, 16, CANONICAL_CHUNKS).load_resume_state(state)
    with pytest.raises(ValueError, match="nonnegative"):
        source.resume_state(-1, 40)

    over_capacity = deepcopy(state)
    over_capacity["samples_per_rank"] = source.end - source.start + 1
    with pytest.raises(ValueError, match="capacity"):
        source.load_resume_state(over_capacity)

    overlap = deepcopy(state)
    overlap["ranges_by_rank"][1] = overlap["ranges_by_rank"][0]
    with pytest.raises(ValueError, match="overlapping"):
        source.load_resume_state(overlap)

    indivisible_remainder = source.resume_state(1, 1)
    with pytest.raises(ValueError, match="not divisible"):
        RangedSampler(_PORTABLE_N, 0, 32, CANONICAL_CHUNKS).load_resume_state(indivisible_remainder)
    with pytest.raises(ValueError, match="current world_size"):
        RangedSampler(_PORTABLE_N, 0, 3, CANONICAL_CHUNKS).load_resume_state(state)


def test_partial_final_batch_records_actual_sample_progress():
    sampler = RangedSampler(10)
    state = sampler.resume_state(3, 4, samples_per_rank=10)
    assert state["samples_per_rank"] == 10

    restored = RangedSampler(10)
    restored.load_resume_state(state)
    assert list(restored) == []

    with pytest.raises(ValueError, match="exceeds"):
        sampler.resume_state(3, 4)
    with pytest.raises(ValueError, match="incompatible"):
        sampler.resume_state(3, 4, samples_per_rank=8)


def test_canonical_partition_keeps_each_rank_contiguous_and_the_tail_unassigned():
    from iris3b.data.samplers import CANONICAL_CHUNKS

    for world in (1, 2, 8, 16, 64, 128):
        spans = [
            (
                RangedSampler(_PORTABLE_N, rank, world, CANONICAL_CHUNKS).start,
                RangedSampler(_PORTABLE_N, rank, world, CANONICAL_CHUNKS).end,
            )
            for rank in range(world)
        ]
        assert all(
            prev_end == next_start
            for (_, prev_end), (next_start, _) in zip(spans[:-1], spans[1:], strict=True)
        )
    assert [(RangedSampler(12, rank, 5).start, RangedSampler(12, rank, 5).end) for rank in range(5)] == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
        (8, 10),
    ]
