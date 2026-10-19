import pytest

import guard


@pytest.mark.parametrize(
    "command",
    [
        "python loop/gate.py check --candidate ../iris-wt/a --category attention --targets original",
        "python loop/gpu.py run -- python -m pytest -q tests/test_metal_engine.py > /tmp/log.txt",
        "IRIS_SKIP_GPU_TESTS=1 python -m pytest -q",
        "IRIS_CPU_ONLY=1 python scripts/flops.py",
        "cat loop/decide.py",
        "git diff opt/loop -- src",
        "ruff check . && ruff format --check .",
        "echo '# result' >> ~/.iris-loop/notes/LEDGER.md",
        "git worktree add ../iris-wt/fuse-gates -b opt/fuse-gates opt/loop",
    ],
)
def test_allowed(command):
    assert guard.bash_problem(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "sed -i 's/0.005/0.05/' loop/decide.py",
        "echo x > CLAUDE.md",
        "rm -rf ~/.iris-loop/baseline",
        "cp /tmp/fake.json ~/.iris-loop/log.jsonl",
        "git checkout HEAD~3 -- loop/gate.py",
        "python -m pytest tests/test_metal_engine.py",
        "python scripts/benchmark.py --steps 10",
        "pip install mlx",
        "python -c \"open('loop/spec.py', 'w')\"",
    ],
)
def test_blocked(command):
    assert guard.bash_problem(command) is not None


def test_protected_paths(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    state = tmp_path / "state"
    monkeypatch.setattr(guard, "STATE", state)
    monkeypatch.setattr(guard, "NOTES", state / "notes")
    assert guard.protected_path(repo / "loop/gate.py")
    assert guard.protected_path(repo / ".claude/settings.json")
    assert guard.protected_path(repo / "CLAUDE.md")
    assert guard.protected_path(repo / "benchmarks/prompts.json")
    assert guard.protected_path(state / "manifest.json")
    assert not guard.protected_path(state / "notes/LEDGER.md")
    assert not guard.protected_path(repo / "src/iris_metal/model.py")
    assert not guard.protected_path(repo / "benchmarks/optimization-loop-2026-11-01.md")


def test_edit_event(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    event = {"tool_name": "Edit", "tool_input": {"file_path": "loop/decide.py"}, "cwd": str(repo)}
    assert guard.problem(event)
    event["tool_input"]["file_path"] = str(repo / "src/iris_metal/kernels.py")
    assert guard.problem(event) is None


def test_harness_call_cannot_smuggle_writes():
    assert guard.bash_problem("python loop/gpu.py run -- sed -i 's/a/b/' loop/decide.py")
    assert guard.bash_problem(".venv/bin/python loop/gate.py stopcheck") is None
