"""``check`` and ``merge`` end to end in a scratch repository, with the GPU work faked."""

import json
import shutil
import subprocess
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

import common
import gate
import spec

REPO = Path(__file__).resolve().parents[2]
TRAJECTORY = np.ones((1, 3, 4, 4), np.float32)
PASSING = {"tests.test_a::test_ok": "passed"}


def git(tree, *args):
    return subprocess.run(["git", "-C", str(tree), *args], check=True, capture_output=True, text=True).stdout


def original_capture():
    trajectories = {
        job: {"encoding": "e", "preclamp": "p", "metadata_keys": ["model"]}
        for job in spec.TRAJECTORY_IDS["dev"]
    }
    return {
        "trajectories": trajectories,
        "velocity": {"vel-a-0": "v"},
        "ocr": {"seed": 0, "words": ["FRESH", "BREAD"]},
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo, state = tmp_path / "repo", tmp_path / "state"
    for name in (*common.FROZEN, "pyproject.toml"):
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
    git(repo, "worktree", "add", "-q", "--detach", str(state / "trees/baseline"), "HEAD")

    for module in (common, gate):
        monkeypatch.setattr(module, "STATE", state)
        monkeypatch.setattr(module, "NOTES", state / "notes")
    for name, value in (("LOG", "log.jsonl"), ("BASELINE", "baseline"), ("ORACLE", "oracle")):
        monkeypatch.setattr(gate, name, state / value)
    built = spec.build(REPO)
    common.write_new_json(state / "spec.json", built)
    config = {
        "repo": str(repo),
        "baseline": "v0.6.1",
        "baseline_commit": common.head(repo),
        "checkpoints": {},
    }
    common.write_new_json(state / "config.json", config)
    for job in built["trajectories"]["dev"]:
        for folder in ("oracle/dev", "baseline/original-dev"):
            (state / folder).mkdir(parents=True, exist_ok=True)
            np.save(state / folder / f"traj-{job['id']}.npy", TRAJECTORY)
    (state / "oracle/velocity").mkdir(parents=True)
    np.savez(state / "oracle/velocity/vel-a-0.npz", v=TRAJECTORY)
    np.save(state / "baseline/original-dev/vel-a-0.npy", TRAJECTORY)
    turbo = {"fox": {"pixels": "p", "preclamp": "q"}}
    common.write_new_json(state / "baseline/turbo-dev/result.json", turbo)
    common.write_new_json(state / "baseline/original-dev/result.json", original_capture())
    common.write_new_json(state / "baseline/tests.json", PASSING)
    common.write_new_json(state / "baseline/peaks.json", dict.fromkeys(built["gate_workloads"], 10.0))
    manifest = {"frozen": common.frozen_hashes(repo), "state": common.state_hashes()}
    common.write_new_json(state / "manifest.json", manifest)

    calls = []

    def fake_worker(tree, mode, arg, out, extra_env=None):
        calls.append((tree.name, mode, arg))
        out.mkdir(parents=True, exist_ok=True)
        if mode == "capture-turbo":
            return turbo
        if mode == "capture-original":
            for job in built["trajectories"]["dev"]:
                np.save(out / f"traj-{job['id']}.npy", TRAJECTORY)
            np.save(out / "vel-a-0.npy", TRAJECTORY)
            return original_capture()
        assert mode == "time"
        scale = 0.9 if tree.name == "candidate" else 1.0
        return {"cold": 20 * scale, "warm": [10 * scale] * 3, "peak_gib": 10.0, "swapouts": 0}

    monkeypatch.setattr(gate, "worker", fake_worker)
    monkeypatch.setattr(gate, "run_tests", lambda tree, out: PASSING)
    monkeypatch.setattr(gate, "code_checks", lambda tree, base: (True, []))
    monkeypatch.setattr(gate, "wait_quiet", lambda: None)
    return Namespace(repo=repo, candidate=candidate, state=state, calls=calls)


def check_args(candidate, **changes):
    values = {
        "candidate": candidate,
        "category": "attention",
        "targets": ["turbo"],
        "quick": False,
        "note": "fuse",
    }
    return Namespace(**{**values, **changes})


def last_entry(state):
    return json.loads((state / "log.jsonl").read_text().splitlines()[-1])


def write_review(state, entry, first_line=None):
    review = state / "notes/reviews" / f"{entry['id']}.md"
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text((first_line or f"VERDICT: PASS {entry['candidate'][:12]}") + "\nfindings\n")


def test_accept_confirm_review_and_merge(setup, capsys):
    gate.check(check_args(setup.candidate))
    entry = last_entry(setup.state)
    assert entry["verdict"] == "ACCEPT", entry["reasons"]
    assert set(entry["confirmation"]) >= {"turbo-repeat", "turbo-new"}
    assert ("reference", "time", "turbo-repeat") in setup.calls  # warm-up
    assert "GATE G0001 VERDICT=ACCEPT" in capsys.readouterr().out
    worktrees = git(setup.repo, "worktree", "list")
    assert "runs/G0001" not in worktrees

    write_review(setup.state, entry, "VERDICT: FAIL x\nVERDICT: PASS " + entry["candidate"][:12])
    with pytest.raises(SystemExit, match="must start with"):
        gate.merge(Namespace(id="G0001"))
    write_review(setup.state, entry)
    gate.merge(Namespace(id="G0001"))
    parents = git(setup.repo, "log", "-1", "--format=%P").split()
    assert entry["candidate"] in parents
    assert "G0001: fuse" in git(setup.repo, "log", "-1", "--format=%B")
    merged = last_entry(setup.state)
    assert merged["type"] == "merge" and merged["review_sha256"]


def test_candidate_harness_change_is_rejected(setup):
    (setup.candidate / "loop/decide.py").write_text("MIN_GAIN = 0\n")
    git(setup.candidate, "commit", "-qam", "lower the bar")
    gate.check(check_args(setup.candidate))
    entry = last_entry(setup.state)
    assert entry["verdict"] == "REJECT"
    assert entry["harness_changed"] == ["loop/decide.py"]


def test_loosened_ruff_config_is_a_harness_change(setup):
    pyproject = setup.candidate / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace('select = ["E"', 'select = ["F"'))
    git(setup.candidate, "commit", "-qam", "quieter lint")
    gate.check(check_args(setup.candidate))
    assert last_entry(setup.state)["harness_changed"] == ["pyproject.toml [tool.ruff] [tool.pytest]"]


def test_turbo_mismatch_rejects_without_timing(setup, monkeypatch):
    real = gate.worker

    def changed(tree, mode, arg, out, extra_env=None):
        if mode == "capture-turbo":
            return {"fox": {"pixels": "p", "preclamp": "different"}}
        return real(tree, mode, arg, out, extra_env)

    monkeypatch.setattr(gate, "worker", changed)
    gate.check(check_args(setup.candidate))
    entry = last_entry(setup.state)
    assert entry["verdict"] == "REJECT"
    assert entry["turbo_mismatches"] == ["fox.preclamp"]
    assert not any(mode == "time" for _, mode, _ in setup.calls)


def test_lost_test_is_broken(setup, monkeypatch):
    monkeypatch.setattr(gate, "run_tests", lambda tree, out: {"tests.test_a::test_ok": "failed"})
    gate.check(check_args(setup.candidate))
    entry = last_entry(setup.state)
    assert entry["verdict"] == "BROKEN"
    assert entry["test_failures"] == ["tests.test_a::test_ok"]


def test_a_crash_is_logged_and_the_next_check_gets_a_new_id(setup, monkeypatch, capsys):
    def crash(tree, mode, arg, out, extra_env=None):
        raise RuntimeError("MPS out of memory")

    monkeypatch.setattr(gate, "worker", crash)
    gate.check(check_args(setup.candidate))
    entry = last_entry(setup.state)
    assert entry["verdict"] == "ERROR" and "MPS out of memory" in entry["reasons"][0]
    assert "GATE G0001 VERDICT=ERROR" in capsys.readouterr().out
    gate.check(check_args(setup.candidate))
    assert last_entry(setup.state)["id"] == "G0002"


def test_changed_evidence_halts(setup):
    (setup.state / "baseline/turbo-dev/result.json").write_text("{}")
    gate.check(check_args(setup.candidate))
    entry = last_entry(setup.state)
    assert entry["verdict"] == "ERROR" and entry["reasons"][0].startswith("HALT")


def test_stale_reference_blocks_merge(setup):
    gate.check(check_args(setup.candidate))
    entry = last_entry(setup.state)
    time.sleep(0.01)
    write_review(setup.state, entry)
    (setup.repo / "src/other.py").write_text("y = 1\n")
    git(setup.repo, "add", ".")
    git(setup.repo, "commit", "-qm", "someone else merged first")
    with pytest.raises(SystemExit, match="reference moved"):
        gate.merge(Namespace(id="G0001"))
