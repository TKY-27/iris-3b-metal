"""End-to-end Iris-3B timing and peak memory on an Apple GPU.

Usage:
    python scripts/benchmark.py --checkpoint iris-3b --size 1024 --steps 100 --out bench.json

Every phase a user waits for is timed separately (text encoding, model load,
sampling) on a cold process. Peak memory is the process's lifetime maximum
physical footprint as reported by the kernel, which on Apple Silicon includes
Metal allocations; it is the number Activity Monitor calls "Memory".
"""

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from iris_metal.pipeline import IrisPipeline

_RUSAGE_INFO_V4 = 4
_LIFETIME_MAX_FOOTPRINT = 28  # uint64 index of ri_lifetime_max_phys_footprint after the uuid


def peak_footprint_gb() -> float:
    libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
    buffer = (ctypes.c_uint64 * 64)()
    if libc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(buffer)):
        raise OSError("proc_pid_rusage failed")
    return buffer[2 + _LIFETIME_MAX_FOOTPRINT] / 1024**3  # 16-byte uuid = two uint64 slots


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used = ")[1].split("M")[0])


def environment() -> dict:
    def command(*args: str) -> str:
        return subprocess.run(args, capture_output=True, text=True).stdout.strip()

    return {
        "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_gb": int(command("sysctl", "-n", "hw.memsize")) / 1024**3,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "commit": command("git", "rev-parse", "--short", "HEAD"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default="iris-3b")
    parser.add_argument("--prompt", default="A red fox sleeping in fresh snow, golden hour")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--image", help="optional path for the generated image")
    parser.add_argument("--out", help="optional JSON report path")
    args = parser.parse_args()

    swap_before = swap_used_mb()
    pipe = IrisPipeline(args.checkpoint, attention_dtype=getattr(torch, args.attention))
    timings = {}

    start = time.perf_counter()
    pipe.encode_prompts([args.prompt])
    timings["text_encoding_s"] = time.perf_counter() - start

    start = time.perf_counter()
    pipe.engine  # noqa: B018  (loads the diffusion model)
    torch.mps.synchronize()
    timings["model_load_s"] = time.perf_counter() - start

    step_times = []
    last = time.perf_counter()

    def on_step(step: int, total: int) -> None:
        nonlocal last
        torch.mps.synchronize()
        now = time.perf_counter()
        step_times.append(now - last)
        last = now

    start = time.perf_counter()
    image = pipe(
        [args.prompt], height=args.size, width=args.size, steps=args.steps, seed=args.seed, on_step=on_step
    )[0]
    timings["sampling_s"] = time.perf_counter() - start  # the prompt encoding is cached by now
    timings["step_median_s"] = sorted(step_times)[len(step_times) // 2]
    timings["first_step_s"] = step_times[0]

    report = {
        "config": vars(args),
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "peak_memory_gb": round(peak_footprint_gb(), 2),
        "swap_growth_mb": round(swap_used_mb() - swap_before, 1),
        "environment": environment(),
    }
    print(json.dumps(report, indent=2))
    if args.image:
        image.save(args.image)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
