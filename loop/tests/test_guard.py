import json
import subprocess
import sys
from pathlib import Path

import pytest

import guard


@pytest.mark.parametrize(
    "command",
    [
        "python loop/gate.py check --candidate ../iris-wt/a --category attention --targets original",
        ".venv/bin/python loop/gate.py stopcheck",
        "python loop/gpu.py run -- python -m pytest -q tests/test_metal_engine.py > /tmp/log.txt",
        "nohup python loop/gate.py check --candidate x --category matmul --targets turbo"
        " > ~/.iris-loop/notes/out/x.log 2>&1 &",
        "cd ../iris-wt/a && python loop/gpu.py cpu -- python -m pytest -q",
        ".venv/bin/python loop/gpu.py cpu --wait 5 -- python -m pytest -q tests/test_flow.py",
        "python loop/gpu.py run -- xcrun xctrace record --template 'Metal System Trace'",
        "echo $! > ~/.iris-loop/notes/out/x.pid",
        "cat loop/decide.py",
        "git diff opt/loop -- src",
        "ruff check . && ruff format --check .",
        "echo '# result' >> ~/.iris-loop/notes/LEDGER.md",
        "git worktree add ../iris-wt/fuse-gates -b opt/fuse-gates opt/loop",
        "pip list",
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
        "git -C ~/.iris-loop/trees/baseline switch -d HEAD~1",
        "python -m pytest tests/test_metal_engine.py",
        "python scripts/benchmark.py --steps 10",
        ".venv/bin/python scripts/benchmark.py --steps 10",
        "bash -c 'python scripts/flops.py'",
        "IRIS_SKIP_GPU_TESTS=1 python -m pytest -q",
        "xcrun xctrace record --template 'Metal System Trace'",
        "swift run bench",
        "cp x ~/.iris-loop/notes/../manifest.json",
        "echo '{}' >> ~/.iris-loop/notes/../log.jsonl",
        "echo '{}' > ~/.claude/settings.json",
        "uv run scripts/flops.py",
        "pip install mlx",
        "python -c \"open('loop/spec.py', 'w')\"",
        "python loop/gpu.py run -- sed -i 's/a/b/' loop/decide.py",
    ],
)
def test_blocked(command):
    assert guard.bash_problem(command) is not None


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    return repo


def test_protected_paths(tmp_path, monkeypatch):
    repo, state = make_repo(tmp_path), tmp_path / "state"
    monkeypatch.setattr(guard, "STATE", state)
    monkeypatch.setattr(guard, "NOTES", state / "notes")
    for path in ("loop/gate.py", ".claude/settings.json", "CLAUDE.md", "benchmarks/prompts.json"):
        assert guard.protected_path(repo / path)
    assert guard.protected_path(state / "manifest.json")
    assert not guard.protected_path(state / "notes/LEDGER.md")
    assert not guard.protected_path(repo / "src/iris_metal/model.py")
    assert not guard.protected_path(repo)
    assert guard.protected_path(Path.home() / ".claude/settings.json")


def test_tool_events(tmp_path):
    repo = make_repo(tmp_path)
    edit = {"tool_name": "Edit", "tool_input": {"file_path": "loop/decide.py"}, "cwd": str(repo)}
    assert guard.problem(edit)
    assert guard.problem({**edit, "tool_name": "Read"}) is None
    assert guard.problem({**edit, "tool_name": "mcp__files__write_file", "tool_input": {"path": "loop/x.py"}})
    move = {"source": "src/a.py", "destination": "loop/gate.py"}
    assert guard.problem({**edit, "tool_name": "mcp__files__move_file", "tool_input": move})
    shell = {"tool_name": "mcp__shell__run", "tool_input": {"command": "python scripts/benchmark.py"}}
    assert guard.problem(shell)
    allowed = {"tool_name": "Edit", "tool_input": {"file_path": str(repo / "src/iris_metal/kernels.py")}}
    assert guard.problem({**allowed, "cwd": str(repo)}) is None


def test_hook_fails_closed():
    hook = Path(guard.__file__)
    broken = subprocess.run([sys.executable, str(hook)], input="not json", capture_output=True, text=True)
    assert broken.returncode == 2
    event = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}})
    assert (
        subprocess.run([sys.executable, str(hook)], input=event, capture_output=True, text=True).returncode
        == 0
    )
