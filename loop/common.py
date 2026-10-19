"""Shared gate helpers. Standard library only: the hook runs on system Python."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

KIT = Path(__file__).resolve().parent
STATE = Path(os.environ.get("IRIS_LOOP_HOME", "~/.iris-loop")).expanduser()
NOTES = STATE / "notes"  # the only agent-writable part of STATE
LOCKED_ENV = "IRIS_LOOP_GPU_HOLDER"

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
STATE_PARTS = ("oracle", "baseline", "inputs", "spec.json", "config.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_of(mapping: dict) -> str:
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


def frozen_hashes(tree: Path) -> dict[str, str]:
    import tomllib

    hashes = {name: sha256_file(tree / name) for name in FROZEN}
    tool = tomllib.loads((tree / "pyproject.toml").read_text()).get("tool", {})
    hashes["pyproject.toml [tool.ruff] [tool.pytest]"] = digest_of(
        {key: tool.get(key) for key in ("ruff", "pytest")}
    )
    return hashes


def state_hashes() -> dict[str, str]:
    files = [p for p in sorted(STATE.rglob("*")) if p.is_file()]
    kept = [p for p in files if p.relative_to(STATE).parts[0] in STATE_PARTS]
    return {p.relative_to(STATE).as_posix(): sha256_file(p) for p in kept}


def read_json(path: Path):
    return json.loads(path.read_text())


def write_new_json(path: Path, value) -> None:
    """Never replaces: recorded evidence is append-only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def run(*args, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"{' '.join(map(str, args))} failed ({result.returncode}):\n{result.stderr[-4000:]}"
        )
    return result.stdout.strip()


def git(tree: Path, *args: str) -> str:
    return run("git", "-C", str(tree), *args)


def head(tree: Path) -> str:
    return git(tree, "rev-parse", "HEAD")


def is_clean(tree: Path) -> bool:
    return git(tree, "status", "--porcelain", "--untracked-files=no") == ""


def run_child(command: list[str], log: Path, timeout: float, **kwargs) -> int:
    """Kills the whole process group on timeout or when the gate itself is stopped."""
    with open(log, "w") as output:
        child = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, **kwargs
        )

        def stop(signum, frame):
            os.killpg(child.pid, signal.SIGKILL)
            raise SystemExit(f"stopped by signal {signum}")

        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise RuntimeError(f"{command[1:3]} timed out after {timeout / 60:.0f} min; see {log}") from None
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


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


def low_power_mode() -> bool:
    return re.search(r"lowpowermode\s+1", run("pmset", "-g")) is not None


def thermal_state() -> int:
    """0 nominal, 1 fair, 2 serious, 3 critical."""
    from Foundation import NSProcessInfo

    return int(NSProcessInfo.processInfo().thermalState())


def machine() -> dict:
    return {
        "chip": run("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(run("sysctl", "-n", "hw.memsize")),
        "macos": run("sw_vers", "-productVersion"),
        "build": run("sw_vers", "-buildVersion"),
    }


def holds_lock() -> bool:
    """True when an ancestor process holds the GPU lock."""
    holder = os.environ.get(LOCKED_ENV)
    if not holder:
        return False
    pid = os.getpid()
    while pid > 1:
        pid = int(run("ps", "-o", "ppid=", "-p", str(pid)) or 0)
        if str(pid) == holder:
            return True
    return False


@contextmanager
def gpu_lock(label: str, wait: float | None = None):
    """Machine-wide; ``wait`` seconds at most, then SystemExit."""
    if holds_lock():
        yield
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "gpu.lock", "a+") as file:
        deadline = None if wait is None else time.monotonic() + wait
        announced = False
        while True:
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                file.seek(0)
                holder = file.read().strip() or "unknown"
                if deadline is not None and time.monotonic() > deadline:
                    raise SystemExit(f"GPU busy ({holder}); run the command again later") from None
                if not announced:
                    print(f"waiting for the GPU (held by: {holder})", flush=True)
                    announced = True
                time.sleep(5)
        file.seek(0)
        file.truncate()
        file.write(f"{label} pid={os.getpid()} since {time.strftime('%H:%M')}\n")
        file.flush()
        os.environ[LOCKED_ENV] = str(os.getpid())
        try:
            yield
        finally:
            os.environ.pop(LOCKED_ENV)
            fcntl.flock(file, fcntl.LOCK_UN)
