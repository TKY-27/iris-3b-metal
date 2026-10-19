"""``check`` and ``merge`` end to end in a scratch repository, with the GPU work faked."""

import json
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

import common
import gate
import spec

REPO = Path(__file__).resolve().parents[2]
TRAJECTORY = np.ones((1, 3, 4, 4), np.float32)


def git(tree, *args):
    return subprocess.run(["git", "-C", str(tree), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo, state = tmp_path / "repo", tmp_path / "state"
    for name in common.FROZEN:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / name, repo / name)
    (repo / "src").mkdir()
    (repo / "src/engine.py").write_text("x = 1\n")
    git(tmp_path, "init", "-q", "-b", "opt/loop", str(repo))
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    candidate = tmp_path / "wt"
    git(repo, "worktree", "add", "-q", str(candidate), "-b", "opt/faster")
    (candidate / "src/engine.py").write_text("x = 2\n")
    git(candidate, "commit", "-qam", "faster")

    for module in (common, gate):
        monkeypatch.setattr(module, "STATE", state)
        monkeypatch.setattr(module, "NOTES", state / "notes")
    for name, value in (("LOG", "log.jsonl"), ("BASELINE", "baseline"), ("ORACLE", "oracle")):
        monkeypatch.setattr(gate, name, state / value)
    built = spec.build(REPO)
    common.write_new_json(state / "spec.json", built)
    common.write_new_json(state / "config.json", {"repo": str(repo), "baseline": "v0.6.1", "checkpoints": {}})
    for job in built["trajectories"]["dev"]:
        for folder in ("oracle/dev", "baseline/original-dev"):
            (state / folder).mkdir(parents=True, exist_ok=True)
            np.save(state / folder / f"traj-{job['id']}.npy", TRAJECTORY)
    (state / "oracle/velocity").mkdir(parents=True)
    np.savez(state / "oracle/velocity/vel-a-0.npz", v=TRAJECTORY)
    np.save(state / "baseline/original-dev/vel-a-0.npy", TRAJECTORY)
    captures = {"turbo": {"fox": {"pixels": "p", "preclamp": "q"}}, "original": original_capture()}
    for name, value in (("turbo-dev", captures["turbo"]), ("original-dev", captures["original"])):
        common.write_new_json(state / "baseline" / name / "result.json", value)
    common.write_new_json(state / "baseline/tests.json", {"tests.test_a::test_ok": "passed"})
    common.write_new_json(
        state / "manifest.json", {"frozen": common.frozen_hashes(repo), "state": common.state_hashes()}
    )

    def fake_worker(tree, mode, arg, out, extra_env=None):
        out.mkdir(parents=True, exist_ok=True)
        if mode == "capture-turbo":
            return captures["turbo"]
        if mode == "capture-original":
            for job in built["trajectories"]["dev"]:
                np.save(out / f"traj-{job['id']}.npy", TRAJECTORY)
            np.save(out / "vel-a-0.npy", TRAJECTORY)
            return captures["original"]
        assert mode == "time"
        scale = 0.9 if tree == candidate else 1.0
        return {"cold": 20 * scale, "warm": [10 * scale] * 3, "peak_gib": 10.0, "swapouts": 0}

    monkeypatch.setattr(gate, "worker", fake_worker)
    monkeypatch.setattr(gate, "run_tests", lambda tree, out: {"tests.test_a::test_ok": "passed"})
    monkeypatch.setattr(gate, "code_checks", lambda tree, base: (True, []))
    monkeypatch.setattr(gate, "wait_quiet", lambda: None)
    return repo, candidate, state


def original_capture():
    trajectories = {job: {"encoding": "e", "preclamp": "p"} for job in spec.TRAJECTORY_IDS["dev"]}
    return {"trajectories": trajectories, "velocity": {"vel-a-0": "v"}, "ocr": {"preclamp": "o", "words": []}}


def check_args(candidate, **changes):
    values = {
        "candidate": candidate,
        "category": "attention",
        "targets": ["turbo"],
        "quick": False,
        "note": "",
    }
    return Namespace(**{**values, **changes})


def test_check_accepts_then_merge_needs_review(setup, capsys):
    repo, candidate, state = setup
    gate.check(check_args(candidate))
    entry = json.loads((state / "log.jsonl").read_text().splitlines()[-1])
    assert entry["verdict"] == "ACCEPT", entry["reasons"]
    assert entry["original"]["status"] == "exact"
    assert "GATE G0001 VERDICT=ACCEPT" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="VERDICT: PASS"):
        gate.merge(Namespace(id="G0001"))
    review = state / "notes/reviews/G0001.md"
    review.parent.mkdir(parents=True)
    review.write_text(f"VERDICT: PASS {entry['candidate'][:12]}\n")
    gate.merge(Namespace(id="G0001"))
    assert common.head(repo) == entry["candidate"]


def test_harness_edit_is_refused(setup):
    repo, candidate, state = setup
    (candidate / "loop/decide.py").write_text("MIN_GAIN = 0\n")
    git(candidate, "commit", "-qam", "lower the bar")
    with pytest.raises(SystemExit, match="harness modified"):
        gate.check(check_args(candidate))


def test_turbo_mismatch_rejects_without_timing(setup, monkeypatch):
    repo, candidate, state = setup
    real = gate.worker

    def changed(tree, mode, arg, out, extra_env=None):
        assert mode != "time", "timing must not run after a correctness failure"
        if mode == "capture-turbo":
            return {"fox": {"pixels": "p", "preclamp": "different"}}
        return real(tree, mode, arg, out, extra_env)

    monkeypatch.setattr(gate, "worker", changed)
    gate.check(check_args(candidate))
    entry = json.loads((state / "log.jsonl").read_text().splitlines()[-1])
    assert entry["verdict"] == "REJECT"
    assert entry["turbo_mismatches"] == ["fox.preclamp"]


def test_stale_reference_blocks_merge(setup):
    repo, candidate, state = setup
    gate.check(check_args(candidate))
    entry = json.loads((state / "log.jsonl").read_text().splitlines()[-1])
    review = state / "notes/reviews/G0001.md"
    review.parent.mkdir(parents=True)
    review.write_text(f"VERDICT: PASS {entry['candidate'][:12]}\n")
    (repo / "src/other.py").write_text("y = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "someone else merged first")
    with pytest.raises(SystemExit, match="reference moved"):
        gate.merge(Namespace(id="G0001"))


def test_lost_test_rejects(setup, monkeypatch):
    repo, candidate, state = setup
    monkeypatch.setattr(gate, "run_tests", lambda tree, out: {"tests.test_a::test_ok": "failed"})
    gate.check(check_args(candidate))
    entry = json.loads((state / "log.jsonl").read_text().splitlines()[-1])
    assert entry["verdict"] == "REJECT"
    assert entry["test_failures"] == ["tests.test_a::test_ok"]
