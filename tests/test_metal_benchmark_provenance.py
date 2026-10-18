import subprocess
import sys
from pathlib import Path

import pytest
from scripts import benchmark


@pytest.mark.parametrize("package", ["iris3b", "iris_metal"])
def test_commit_rejects_a_foreign_import(package, monkeypatch, tmp_path):
    module = getattr(benchmark, package)
    foreign = tmp_path / package / "__init__.py"
    monkeypatch.setattr(module, "__file__", str(foreign))
    with pytest.raises(RuntimeError, match=f"{package} loaded from"):
        benchmark.source_commit()


@pytest.mark.parametrize("flag", ["--out", "--image"])
def test_benchmark_rejects_existing_evidence_before_sampling(flag, monkeypatch, tmp_path):
    evidence = tmp_path / ("report.json" if flag == "--out" else "image.png")
    evidence.write_bytes(b"prior run")
    monkeypatch.setattr(sys, "argv", ["benchmark.py", flag, str(evidence)])
    monkeypatch.setattr(benchmark, "gpu_utilization", lambda: pytest.fail("sampling started"))
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
    assert evidence.read_bytes() == b"prior run"


def test_profile_rejects_an_existing_output_directory(tmp_path):
    directory = tmp_path / "profile"
    directory.mkdir()
    evidence = directory / "results.json"
    evidence.write_bytes(b"prior run")
    script = Path(__file__).resolve().parents[1] / "scripts/profile_generation.py"
    result = subprocess.run(
        [sys.executable, str(script), "--checkpoint", str(tmp_path), "--out", str(directory)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "output already exists" in result.stderr
    assert evidence.read_bytes() == b"prior run"
