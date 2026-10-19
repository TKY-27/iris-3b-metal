"""Shared gate helpers. Standard library only: the hook runs on system Python."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

KIT = Path(__file__).resolve().parent
STATE = Path(os.environ.get("IRIS_LOOP_HOME", "~/.iris-loop")).expanduser()
NOTES = STATE / "notes"  # the only agent-writable part of STATE
LOCKED_ENV = "IRIS_LOOP_GPU_LOCKED"

# Any change invalidates every verdict.
FROZEN = (
    "CLAUDE.md",
    "benchmarks/prompts.json",
    ".claude/settings.json",
    ".claude/agents/code-reviewer.md",
    ".claude/agents/loop-auditor.md",
    "loop/GOAL.md",
    "loop/README.md",
    "loop/common.py",
    "loop/decide.py",
    "loop/gate.py",
    "loop/gpu.py",
    "loop/guard.py",
    "loop/lint_comments.py",
    "loop/spec.py",
    "loop/worker.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_of(mapping: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


def frozen_hashes(tree: Path) -> dict[str, str]:
    return {name: sha256_file(tree / name) for name in FROZEN}


def state_hashes() -> dict[str, str]:
    files = [p for p in sorted(STATE.rglob("*")) if p.is_file()]
    keep = [
        p for p in files if p.parts[len(STATE.parts)] in ("oracle", "baseline", "spec.json", "config.json")
    ]
    return {p.relative_to(STATE).as_posix(): sha256_file(p) for p in keep}


def read_json(path: Path):
    return json.loads(path.read_text())


def write_new_json(path: Path, value) -> None:
    """Never replaces: recorded evidence is append-only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def run(*args, cwd: Path | None = None, env: dict | None = None, check: bool = True) -> str:
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(
            f"{' '.join(map(str, args))} failed ({result.returncode}):\n{result.stderr[-4000:]}"
        )
    return result.stdout.strip()


def git(tree: Path, *args: str, check: bool = True) -> str:
    return run("git", "-C", str(tree), *args, check=check)


def head(tree: Path) -> str:
    return git(tree, "rev-parse", "HEAD")


def is_clean(tree: Path) -> bool:
    return git(tree, "status", "--porcelain", "--untracked-files=no") == ""


# -- machine state -----------------------------------------------------------
def gpu_utilization(seconds: float = 3.0) -> float:
    samples = []
    for _ in range(max(1, int(seconds * 2))):
        out = run("ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator")
        samples.append(int(re.search(r'"Device Utilization %"=(\d+)', out)[1]))
        time.sleep(0.5)
    return statistics.fmean(samples)


def swapouts() -> int:
    return int(re.search(r"Swapouts:\s+(\d+)", run("vm_stat"))[1])


def on_ac_power() -> bool:
    return "AC Power" in run("pmset", "-g", "batt")


def speed_limited() -> bool:
    limits = re.findall(r"Speed_Limit\s*=\s*(\d+)", run("pmset", "-g", "therm"))
    return any(int(value) < 100 for value in limits)


def machine() -> dict:
    return {
        "chip": run("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(run("sysctl", "-n", "hw.memsize")),
        "macos": run("sw_vers", "-productVersion"),
        "build": run("sw_vers", "-buildVersion"),
    }


@contextmanager
def gpu_lock(label: str):
    """Machine-wide; a nested holder passes through."""
    if os.environ.get(LOCKED_ENV) == "1":
        yield
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "gpu.lock", "a+") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            file.seek(0)
            print(f"waiting for the GPU (held by: {file.read().strip() or 'unknown'})", flush=True)
            fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0)
        file.truncate()
        file.write(f"{label} pid={os.getpid()}\n")
        file.flush()
        os.environ[LOCKED_ENV] = "1"
        try:
            yield
        finally:
            os.environ.pop(LOCKED_ENV)
            fcntl.flock(file, fcntl.LOCK_UN)
