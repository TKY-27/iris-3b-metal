from pathlib import Path

import pytest
import torch

import spec
import worker

REPO = Path(__file__).resolve().parents[2]


def test_spec_uses_existing_prompts():
    built = spec.build(REPO)
    turbo = built["turbo"]["dev"]
    assert len({job["id"] for job in turbo}) == len(turbo) == 12 + 3 + len(spec.TURBO_SIZES)
    assert all(
        job["width"] % 16 == 0 and job["height"] % 16 == 0
        for split in built["turbo"].values()
        for job in split
    )
    assert [j["id"] for j in built["trajectories"]["dev"]] == list(spec.TRAJECTORY_IDS["dev"])
    assert built["ocr"]["dev"]["words"] == ["FRESH", "BREAD"]
    assert "FRESH BREAD" in built["ocr"]["dev"]["prompt"]


def test_tensor_digest_sees_dtype_shape_and_signed_zero():
    zero = torch.zeros(2, 2)
    assert worker.tensor_digest(zero) == worker.tensor_digest(torch.zeros(2, 2))
    assert worker.tensor_digest(zero) != worker.tensor_digest(-zero)
    assert worker.tensor_digest(zero) != worker.tensor_digest(zero.reshape(4))
    assert worker.tensor_digest(zero) != worker.tensor_digest(zero.bfloat16())
    with pytest.raises(SystemExit):
        worker.tensor_digest(torch.tensor([float("nan")]))


def test_clamp_probe_records_first_sample_clamp():
    original = torch.Tensor.clamp
    try:
        probe = worker.ClampProbe()
        torch.ones(2, 3).clamp(max=0)  # not an image batch
        sample = torch.linspace(-2, 2, 48).reshape(1, 3, 4, 4)
        sample.clamp(-1, 1)
        (sample * 0).clamp(min=-1, max=1)  # only the first is the sample
        assert torch.equal(probe.take(), sample)
        with pytest.raises(SystemExit):
            probe.take()
    finally:
        torch.Tensor.clamp = original
