"""PreToolUse hook; heuristic, the gate re-hashes the harness anyway."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from common import NOTES, STATE

PROTECTED_TOP = {"loop", ".claude"}
PROTECTED_FILES = {"CLAUDE.md", "benchmarks/prompts.json"}
HARNESS_CALL = re.compile(r"\S*python3?(?:\.\d+)?\s+\S*loop/(?:gate|gpu|lint_comments)\.py\b")
NOTES_PATH = re.compile(r"\S*\.iris-loop/notes\S*")
PROTECTED_TOKEN = re.compile(r"(?:^|[\s'\"=/(])(?:loop/|\.claude\b|CLAUDE\.md|prompts\.json|\.iris-loop)")
WRITES = re.compile(
    r">|<<|\btee\b|\bsed\s+-i|\bperl\s+-\w*i|\b(?:rm|mv|cp|ln|chmod|touch|truncate|patch|rsync)\b"
    r"|\bgit\s+(?:checkout|restore|reset|rm|mv|apply|am|stash|clean|update-index)\b|\bpython3?\s+-[c-]"
)
RUNS_PYTHON = re.compile(r"(?:^|[\s;&|(])(?:python3?|pytest|iris-metal)\b")
CPU_ONLY = re.compile(r"\b(?:IRIS_CPU_ONLY|IRIS_SKIP_GPU_TESTS)=1\b")
INSTALLS = re.compile(
    r"\bpip3?\s+install\b|\bpython3?\s+-m\s+pip\b|\buv\s+(?:pip|add|sync)\b|\bbrew\s+install\b"
)


def repo_root(path: Path) -> Path | None:
    return next((p for p in (path, *path.parents) if (p / ".git").exists()), None)


def protected_path(path: Path) -> bool:
    path = path.expanduser().resolve()
    if path.is_relative_to(STATE.resolve()):
        return not path.is_relative_to(NOTES.resolve())
    root = repo_root(path.parent)
    if root is None:
        return False
    relative = path.relative_to(root)
    return relative.parts[0] in PROTECTED_TOP or relative.as_posix() in PROTECTED_FILES


def bash_problem(command: str) -> str | None:
    if INSTALLS.search(command):
        return "installing packages changes the measured environment; ask the person instead"
    unwrapped = NOTES_PATH.sub(" ", HARNESS_CALL.sub(" ", command))
    if PROTECTED_TOKEN.search(unwrapped) and WRITES.search(unwrapped):
        return "loop/, .claude/, CLAUDE.md, prompts.json and ~/.iris-loop (except notes/) are read-only"
    if RUNS_PYTHON.search(command) and not HARNESS_CALL.search(command) and not CPU_ONLY.search(command):
        return (
            "run GPU work as `python loop/gpu.py run -- <command>`; "
            "prefix CPU-only commands with IRIS_CPU_ONLY=1 and keep them off MPS"
        )
    return None


def problem(event: dict) -> str | None:
    tool, given = event.get("tool_name", ""), event.get("tool_input", {})
    if tool == "Bash":
        return bash_problem(given.get("command", ""))
    target = given.get("file_path") or given.get("notebook_path")
    if target and protected_path(Path(event.get("cwd", ".")) / target):
        return f"{target} is part of the frozen harness"
    return None


def main() -> int:
    reason = problem(json.load(sys.stdin))
    if reason:
        print(f"blocked: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
