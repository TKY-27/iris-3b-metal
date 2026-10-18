import json

import pytest
import torch
from scripts.validate_turbo import (
    checkpoint_identity,
    load_fixture,
    load_oracle,
    metrics,
    require_new,
    save_fixture,
    trajectory_metrics,
)


def _prepared(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights-v1")
    (checkpoint / "config.yaml").write_text("model: fixture\n")
    (checkpoint / "distill.yaml").write_text("dmd: fixture\n")
    out = tmp_path / "run"
    out.mkdir()
    data = {
        "states": torch.ones(1, 2, 1, 2),
        "mask": torch.ones(1, 2, dtype=torch.bool),
        "noise": torch.zeros(1, 3, 2, 2),
        "prompt": "fox",
        "seed": 3,
        "size": 2,
    }
    identity = checkpoint_identity(checkpoint)
    manifest = save_fixture(out, data, identity, checkpoint, 1.25)
    return out, checkpoint, data, manifest


def test_fixture_manifest_binds_checkpoint_and_tensor_payload(tmp_path):
    out, checkpoint, data, manifest = _prepared(tmp_path)
    loaded, saved, resolved, identity = load_fixture(out)
    assert saved["fixture_sha256"] == manifest["fixture_sha256"]
    assert identity["sha256"] == manifest["checkpoint"]["sha256"]
    assert resolved == checkpoint.resolve()
    assert torch.equal(loaded["noise"], data["noise"])
    assert json.loads((out / "prepare.json").read_text())["tensor_shapes"] == {
        "states": [1, 2, 1, 2],
        "mask": [1, 2],
        "noise": [1, 3, 2, 2],
    }


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"prompt": "cat"}, "prompt conflicts"),
        ({"seed": 4}, "seed conflicts"),
        ({"size": 4}, "size conflicts"),
    ],
)
def test_fixture_rejects_conflicting_run_arguments(tmp_path, kwargs, message):
    out, _, _, _ = _prepared(tmp_path)
    with pytest.raises(SystemExit, match=message):
        load_fixture(out, **kwargs)


def test_fixture_rejects_changed_checkpoint_and_tampered_payload(tmp_path):
    out, checkpoint, _, _ = _prepared(tmp_path)
    (checkpoint / "model.safetensors").write_bytes(b"weights-v2")
    with pytest.raises(SystemExit, match="checkpoint content"):
        load_fixture(out)

    out2, _, _, _ = _prepared(tmp_path / "second")
    with (out2 / "fixture.pt").open("ab") as fixture:
        fixture.write(b"changed")
    with pytest.raises(SystemExit, match="fixture hash"):
        load_fixture(out2)


def test_checkpoint_identity_hashes_required_files_only(tmp_path):
    _, checkpoint, _, manifest = _prepared(tmp_path)
    assert set(manifest["checkpoint"]["files"]) == {
        "model.safetensors",
        "config.yaml",
        "distill.yaml",
    }
    (checkpoint / "notes.txt").write_text("not part of checkpoint")
    assert checkpoint_identity(checkpoint)["sha256"] == manifest["checkpoint"]["sha256"]


def test_saved_oracle_must_match_fixture_provenance(tmp_path):
    out, _, _, manifest = _prepared(tmp_path)
    _, _, _, identity = load_fixture(out)
    result = tmp_path / "oracle.pt"
    provenance = {
        "fixture_sha256": manifest["fixture_sha256"],
        "checkpoint_sha256": identity["sha256"],
    }
    dependencies = {"torch": "2.0", "transformers": "4.0", "safetensors": "0.4"}
    saved = {"provenance": provenance, "dependencies": dependencies}
    torch.save(saved, result)
    assert load_oracle(result, manifest, identity, dependencies)["provenance"] == provenance
    torch.save({**saved, "provenance": {**provenance, "fixture_sha256": "stale"}}, result)
    with pytest.raises(ValueError, match="provenance"):
        load_oracle(result, manifest, identity, dependencies)
    torch.save({**saved, "dependencies": {**dependencies, "torch": "different"}}, result)
    with pytest.raises(ValueError, match="dependencies"):
        load_oracle(result, manifest, identity, dependencies)


def test_old_fixture_without_manifest_requires_a_new_run(tmp_path):
    (tmp_path / "fixture.pt").write_bytes(b"historic")
    with pytest.raises(SystemExit, match="provenance is missing"):
        load_fixture(tmp_path)


def test_metrics_detect_signed_zero_and_report_finiteness_on_both_sides():
    negative_zero = torch.tensor([-0.0])
    positive_zero = torch.tensor([0.0])
    assert metrics(negative_zero, positive_zero)["bitwise_equal"] is False

    left_bad = metrics(torch.tensor([float("nan")]), torch.tensor([1.0]))
    right_bad = metrics(torch.tensor([1.0]), torch.tensor([float("inf")]))
    assert left_bad["finite_a"] is False and left_bad["finite_b"] is True
    assert right_bad["finite_a"] is True and right_bad["finite_b"] is False
    assert left_bad["max_abs"] is None and right_bad["max_abs"] is None


def test_metrics_promotes_half_values_before_subtraction():
    left = torch.tensor([65504.0], dtype=torch.float16)
    right = torch.tensor([-65504.0], dtype=torch.float16)
    assert metrics(left, right)["max_abs"] == 131008.0


def test_metrics_and_trajectory_comparison_reject_shape_or_step_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics(torch.zeros(1), torch.zeros(2))
    with pytest.raises(ValueError):
        trajectory_metrics([torch.zeros(1)], [])


def test_existing_output_is_never_replaced(tmp_path):
    existing = tmp_path / "result.pt"
    existing.write_bytes(b"keep")
    with pytest.raises(SystemExit, match="exists"):
        require_new([existing])
    assert existing.read_bytes() == b"keep"
