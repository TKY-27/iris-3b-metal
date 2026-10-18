import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scripts import denoise_fixture


def test_existing_output_is_rejected_before_checkpoint_resolution(tmp_path, monkeypatch):
    output = tmp_path / "result.npz"
    output.write_bytes(b"preserve")
    monkeypatch.setattr(sys, "argv", ["denoise_fixture.py", "fixture.npz", "--out", str(output)])
    monkeypatch.setattr(
        denoise_fixture,
        "resolve_checkpoint",
        lambda *_: pytest.fail("checkpoint resolution should not run"),
    )

    with pytest.raises(SystemExit, match="exists"):
        denoise_fixture.main()
    assert output.read_bytes() == b"preserve"


def test_result_records_inputs_source_and_dependency_provenance(tmp_path):
    fixture = tmp_path / "fixture.npz"
    fixture.write_bytes(b"saved fixture")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "config.yaml").write_text("model: test\n")
    args = SimpleNamespace(
        fixture=fixture,
        steps=8,
        attention="float16",
        reference="autocast",
    )

    identity = denoise_fixture.provenance(args, checkpoint)

    assert identity["fixture_sha256"] == denoise_fixture.sha256(fixture)
    assert set(identity["checkpoint"]["files"]) == {"model.safetensors", "config.yaml"}
    assert identity["source_sha256"]
    assert {"torch", "numpy", "transformers", "safetensors"} <= identity["dependencies"].keys()
    assert (identity["steps"], identity["attention"], identity["reference"]) == (
        8,
        "float16",
        "autocast",
    )


def test_result_requires_finite_final_and_never_overwrites(tmp_path):
    output = tmp_path / "result.npz"
    identity = {"steps": 1}
    denoise_fixture.save_result(output, torch.ones(1), [0.5], 0.25, identity)
    with np.load(output) as result:
        assert json.loads(str(result["provenance"])) == identity

    with pytest.raises(FileExistsError):
        denoise_fixture.save_result(output, torch.zeros(1), [0.1], 0.2, identity)
    assert np.load(output)["final"].tolist() == [1.0]

    bad = tmp_path / "bad.npz"
    with pytest.raises(ValueError, match="non-finite"):
        denoise_fixture.save_result(bad, torch.tensor([float("nan")]), [0.1], 0.2, identity)
    assert not bad.exists()
