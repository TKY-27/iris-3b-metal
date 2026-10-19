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
NOTES_PATH = re.compile(r"\S*\.iris-loop/notes(?:/\S*)?")
PROTECTED_TOKEN = re.compile(r"(?:^|[\s'\"=/(])(?:loop/|\.claude\b|CLAUDE\.md|prompts\.json|\.iris-loop)")
WRITES = re.compile(
    r">|<<|\btee\b|\bsed\s+-i|\bperl\s+-\w*i|\b(?:rm|mv|cp|ln|chmod|touch|truncate|patch|rsync)\b"
    r"|\bgit\b[^;&|]*\b(?:checkout|switch|restore|reset|rm|mv|apply|am|stash|clean|update-index|commit"
    r"|merge|rebase|worktree)\b|\bpython3?\s+-[c-]"
)
RUNS_CODE = re.compile(
    r"(?:^|[\s;&|(/\"'`])(?:python3?(?:\.\d+)?|pytest|iris-metal|xcrun|swift|swiftc)\b|\buv\s+run\b"
)
SEGMENT = re.compile(r";|&&|\|\||\||\n")
INSTALLS = re.compile(
    r"\bpip3?\s+install\b|\bpython3?\s+-m\s+pip\s+install\b|\buv\s+(?:pip\s+install|add|sync)\b|\bbrew\s+install\b"
)
PATH_KEYS = ("file_path", "notebook_path", "path", "source", "destination", "target", "new_path", "old_path")


def repo_root(path: Path) -> Path | None:
    return next((p for p in (path, *path.parents) if (p / ".git").exists()), None)


def protected_path(path: Path) -> bool:
    path = path.expanduser().resolve()
    if path.is_relative_to(STATE.resolve()):
        return not path.is_relative_to(NOTES.resolve())
    if path.is_relative_to((Path.home() / ".claude").resolve()):
        return True
    root = repo_root(path.parent)
    if root is None or path == root:
        return False
    relative = path.relative_to(root)
    return relative.parts[0] in PROTECTED_TOP or relative.as_posix() in PROTECTED_FILES


def bash_problem(command: str) -> str | None:
    if INSTALLS.search(command):
        return "installing packages changes the measured environment; ask the person instead"
    unwrapped = NOTES_PATH.sub(lambda m: m[0] if ".." in m[0] else " ", HARNESS_CALL.sub(" ", command))
    if PROTECTED_TOKEN.search(unwrapped) and WRITES.search(unwrapped):
        return "loop/, .claude/, ~/.claude, CLAUDE.md, prompts.json and ~/.iris-loop (except notes/) are read-only"
    for segment in SEGMENT.split(command):
        if RUNS_CODE.search(segment) and not HARNESS_CALL.search(segment):
            return (
                "run code through `python loop/gpu.py run -- <command>` (GPU) or "
                "`python loop/gpu.py cpu -- <command>` (CPU tests and scripts)"
            )
    return None


def writes_files(tool: str) -> bool:
    return tool in {"Edit", "Write", "MultiEdit", "NotebookEdit"} or bool(
        re.search(r"write|edit|move|create|delete|remove|rename|upload|copy", tool, re.I)
    )


def problem(event: dict) -> str | None:
    given = event.get("tool_input") or {}
    if isinstance(given.get("command"), str):
        return bash_problem(given["command"])
    if not writes_files(event.get("tool_name", "")):
        return None
    cwd = Path(event.get("cwd", "."))
    for key in PATH_KEYS:
        target = given.get(key)
        if isinstance(target, str) and protected_path(cwd / target):
            return f"{target} is part of the frozen harness"
    return None


def main() -> int:
    try:
        reason = problem(json.load(sys.stdin))
    except Exception as error:
        reason = f"the guard could not read this call ({type(error).__name__})"
    if reason:
        print(f"blocked: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
