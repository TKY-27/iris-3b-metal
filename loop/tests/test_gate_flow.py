"""The gate end to end in a scratch repository, with the GPU work faked."""

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
TURBO = {"fox": {"pixels": "p", "preclamp": "q"}}


def git(tree, *args):
    return subprocess.run(["git", "-C", str(tree), *args], check=True, capture_output=True, text=True).stdout


def trajectory_files(built, split):
    names = [f"traj-{job['id']}.npy" for job in built["trajectories"][split]]
    if split == "dev":
        names += [f"traj-1mp-{job['id']}.npy" for job in built["velocity"]] + ["vel-a-0.npy"]
    return names


def original_capture(built, split):
    trajectories = {
        name[5:-4]: {"encoding": "e", "preclamp": "p", "metadata_keys": ["model"]}
        for name in trajectory_files(built, split)
        if name.startswith("traj-")
    }
    return {
        "trajectories": trajectories,
        "velocity": {},
        "ocr": {"seed": 0, "words": ["FRESH", "BREAD", "SOUP", "TOMATO"]},
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
        for name, value in (
            ("STATE", ""),
            ("NOTES", "notes"),
            ("BASELINE", "baseline"),
            ("ORACLE", "oracle"),
        ):
            monkeypatch.setattr(module, name, state / value)
    monkeypatch.setattr(gate, "LOG", state / "log.jsonl")
    built = spec.build(REPO)
    common.write_new_json(state / "spec.json", built)
    config = {
        "repo": str(repo),
        "baseline": "v0.6.1",
        "baseline_commit": common.head(repo),
        "checkpoints": {},
    }
    common.write_new_json(state / "config.json", config)
    for split in ("dev", "holdout"):
        for name in trajectory_files(built, split):
            folder = "velocity" if name.startswith(("traj-1mp", "vel")) else split
            for directory in (state / "oracle" / folder, state / "baseline" / f"original-{split}"):
                directory.mkdir(parents=True, exist_ok=True)
                np.save(directory / name, TRAJECTORY)
        common.write_new_json(state / f"baseline/turbo-{split}/result.json", TURBO)
        common.write_new_json(
            state / f"baseline/original-{split}/result.json", original_capture(built, split)
        )
    np.savez(state / "oracle/velocity/vel-a-0.npz", v=TRAJECTORY)
    common.write_new_json(state / "baseline/tests.json", PASSING)
    common.write_new_json(state / "baseline/peaks.json", dict.fromkeys(built["gate_workloads"], 10.0))
    common.write_new_json(
        state / "manifest.json", {"frozen": common.frozen_hashes(repo), "state": common.state_hashes()}
    )
    gate.append_log({"type": "prepare", "head": common.head(repo), "time": gate.now()})

    calls = []

    def fake_worker(tree, mode, arg, out, extra_env=None):
        calls.append((tree.name, mode, arg))
        out.mkdir(parents=True, exist_ok=True)
        if mode == "capture-turbo":
            return TURBO
        if mode == "capture-original":
            for name in trajectory_files(built, arg):
                np.save(out / name, TRAJECTORY)
            return original_capture(built, arg)
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


def last_entry(state, kind="check"):
    entries = [json.loads(line) for line in (state / "log.jsonl").read_text().splitlines()]
    return [e for e in entries if e["type"] == kind][-1]


def write_review(state, entry, first_line=None):
    review = state / "notes/reviews" / f"{entry['id']}.md"
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text((first_line or f"VERDICT: PASS {entry['candidate'][:12]}") + "\nfindings\n")


def test_worker_env_keeps_the_state_and_drops_the_rest(tmp_path, monkeypatch):
    monkeypatch.setenv(common.LOCKED_ENV, "123")
    monkeypatch.setenv("SECRET_FLAG", "1")
    env = gate.worker_env(tmp_path / "tree", "time", tmp_path / "out", {"IRIS_AVAILABLE_MEMORY_GIB": "10"})
    assert env["IRIS_LOOP_HOME"] == str(gate.STATE)
    assert env["HOME"] == str(tmp_path / "out/home") and env["HF_HUB_OFFLINE"] == "1"
    assert env["IRIS_AVAILABLE_MEMORY_GIB"] == "10"
    assert common.LOCKED_ENV not in env and "SECRET_FLAG" not in env


def test_accept_confirm_review_merge_audit_and_finalcheck(setup, capsys):
    gate.check(check_args(setup.candidate))
    entry = last_entry(setup.state)
    assert entry["verdict"] == "ACCEPT", entry["reasons"]
    assert set(entry["confirmation"]) >= {"turbo-repeat", "turbo-new"}
    assert {("reference", "time", "turbo-repeat"), ("candidate", "time", "turbo-repeat")} <= set(setup.calls)
    assert entry["code_lines"] == 2
    assert "GATE G0001 VERDICT=ACCEPT" in capsys.readouterr().out
    assert "runs/G0001" not in git(setup.repo, "worktree", "list")

    write_review(setup.state, entry, "VERDICT: FAIL x\nVERDICT: PASS " + entry["candidate"][:12])
    with pytest.raises(SystemExit, match="must start with"):
        gate.merge(Namespace(id="G0001"))
    write_review(setup.state, entry)
    gate.merge(Namespace(id="G0001"))
    assert entry["candidate"] in git(setup.repo, "log", "-1", "--format=%P").split()
    assert "G0001: fuse" in git(setup.repo, "log", "-1", "--format=%B")

    gate.append_log({"type": "stopcheck", "met": True, "time": gate.now()})
    gate.audit(Namespace())
    assert "GATE AUDIT PASS" in capsys.readouterr().out

    (setup.repo / "benchmarks").mkdir(exist_ok=True)
    (setup.repo / "benchmarks/optimization-loop.md").write_text("report\n")
    git(setup.repo, "add", ".")
    git(setup.repo, "commit", "-qm", "report")
    gate.finalcheck(Namespace())
    assert "GATE FINAL READY" in capsys.readouterr().out
    (setup.repo / "src/engine.py").write_text("x = 3\n")
    git(setup.repo, "commit", "-qam", "sneaked in")
    with pytest.raises(SystemExit, match="changed after the audit"):
        gate.finalcheck(Namespace())


def test_audit_fails_on_commits_the_gate_did_not_merge(setup, capsys):
    (setup.repo / "src/engine.py").write_text("x = 5\n")
    git(setup.repo, "commit", "-qam", "direct commit")
    gate.append_log({"type": "stopcheck", "met": True, "time": gate.now()})
    gate.audit(Namespace())
    assert "no gate merge made" in capsys.readouterr().out


def test_candidate_harness_change_is_rejected(setup):
    (setup.candidate / "loop/decide.py").write_text("MIN_GAIN = 0\n")
    git(setup.candidate, "commit", "-qam", "lower the bar")
    gate.check(check_args(setup.candidate))
    entry = last_entry(setup.state)
    assert entry["verdict"] == "REJECT"
    assert entry["harness_changed"] == ["loop/decide.py"]


def test_added_test_configuration_is_a_harness_change(setup):
    (setup.candidate / "tests").mkdir()
    (setup.candidate / "tests/conftest.py").write_text("collect_ignore = ['test_metal_engine.py']\n")
    git(setup.candidate, "add", ".")
    git(setup.candidate, "commit", "-qm", "skip tests")
    gate.check(check_args(setup.candidate))
    assert last_entry(setup.state)["harness_changed"] == ["tests/conftest.py"]


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


def test_candidate_crash_is_broken_and_reference_crash_is_error(setup, monkeypatch, capsys):
    def crash(tree, mode, arg, out, extra_env=None):
        raise gate.WorkerError(tree, "Metal compile error")

    monkeypatch.setattr(gate, "worker", crash)
    gate.check(check_args(setup.candidate))
    entry = last_entry(setup.state)
    assert entry["verdict"] == "BROKEN" and "Metal compile error" in entry["reasons"][0]
    assert "GATE G0001 VERDICT=BROKEN" in capsys.readouterr().out

    def reference_crash(tree, mode, arg, out, extra_env=None):
        raise gate.WorkerError(Path("/elsewhere"), "disk full")

    monkeypatch.setattr(gate, "worker", reference_crash)
    gate.check(check_args(setup.candidate))
    entry = last_entry(setup.state)
    assert entry["id"] == "G0002" and entry["verdict"] == "ERROR"


def test_interruption_is_logged(setup, monkeypatch, capsys):
    def interrupted(tree, mode, arg, out, extra_env=None):
        raise SystemExit("stopped by signal 15")

    monkeypatch.setattr(gate, "worker", interrupted)
    with pytest.raises(SystemExit):
        gate.check(check_args(setup.candidate))
    assert last_entry(setup.state)["verdict"] == "ERROR"
    assert "GATE G0001 VERDICT=ERROR" in capsys.readouterr().out


def test_stale_candidate(setup):
    (setup.repo / "src/other.py").write_text("y = 1\n")
    git(setup.repo, "add", ".")
    git(setup.repo, "commit", "-qm", "moved on")
    gate.check(check_args(setup.candidate))
    assert last_entry(setup.state)["verdict"] == "STALE"


def test_changed_evidence_halts(setup):
    (setup.state / "baseline/turbo-dev/result.json").write_text("{}")
    gate.check(check_args(setup.candidate))
    entry = last_entry(setup.state)
    assert entry["verdict"] == "ERROR" and entry["reasons"][0].startswith("HALT")


def test_merge_refuses_forged_log_entries(setup):
    gate.check(check_args(setup.candidate))
    entry = last_entry(setup.state)
    forged = {**entry, "id": "G0009"}
    gate.append_log(forged)
    write_review(setup.state, forged)
    with pytest.raises(SystemExit, match="recorded by the gate"):
        gate.merge(Namespace(id="G0009"))


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
