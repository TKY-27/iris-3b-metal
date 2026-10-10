"""Iris-3B generation latency and memory on an Apple GPU.

Usage:
    python scripts/benchmark.py --size 1024x1024 --steps 100 --repeats 3 --out bench.json

One fresh process measures, in order: imports, text encoding, model load, the
first image (which includes compiling Metal shaders) and ``--repeats - 1``
warm requests with the models resident and the prompt encoding cached. GPU
work is synchronized only at request boundaries, as in normal use;
``--profile`` adds a separate diagnostic request that synchronizes every step
to report the per-step distribution, and is never mixed into the timings.

Memory is the process's peak physical footprint per phase in GiB, read from
the kernel (on Apple Silicon this includes Metal allocations; it is what
Activity Monitor calls "Memory"). The system's background GPU utilization
before the run and its swap and page-out activity during it are recorded,
because a busy or memory-starved Mac changes every number here.
"""

import time

PROCESS_START = time.perf_counter()

import argparse  # noqa: E402
import ctypes  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from iris_metal.pipeline import IrisPipeline  # noqa: E402

_RUSAGE_INFO_V4 = 4
# rusage_info_v4 fields, as uint64 indices after the 16-byte uuid
_LIFETIME_MAX_FOOTPRINT = 28
_INTERVAL_MAX_FOOTPRINT = 33  # reset by every read


def peak_footprint_gib(field: int) -> float:
    libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
    buffer = (ctypes.c_uint64 * 64)()
    if libc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(buffer)):
        raise OSError("proc_pid_rusage failed")
    return buffer[2 + field] / 1024**3


def command(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout.strip()


def swap_used_mb() -> float:
    return float(command("sysctl", "-n", "vm.swapusage").split("used = ")[1].split("M")[0])


def paging_counters() -> dict[str, int]:
    stats = command("vm_stat")
    return {key: int(re.search(rf"{key}:\s+(\d+)", stats)[1]) for key in ("Pageouts", "Swapouts")}


def gpu_utilization(seconds: float = 5.0) -> float:
    """Mean device utilization reported by the GPU driver, in percent."""
    samples = []
    for _ in range(int(seconds * 2)):
        out = command("ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator")
        samples.append(int(re.search(r'"Device Utilization %"=(\d+)', out)[1]))
        time.sleep(0.5)
    return statistics.fmean(samples)


def environment() -> dict:
    return {
        "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_gb": int(command("sysctl", "-n", "hw.memsize")) / 1e9,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "commit": command("git", "rev-parse", "--short", "HEAD"),
    }


def parse_size(text: str) -> tuple[int, int]:
    width, height = (int(side) for side in text.split("x"))
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default="iris-3b")
    parser.add_argument("--prompt", default="A red fox sleeping in fresh snow, golden hour")
    parser.add_argument("--size", type=parse_size, default=(1024, 1024), help="WIDTHxHEIGHT")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3, help="requests in one process; the first is cold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument(
        "--profile", action="store_true", help="add a per-step synchronized diagnostic request"
    )
    parser.add_argument("--image", help="optional path for the last generated image")
    parser.add_argument("--out", help="optional JSON report path")
    args = parser.parse_args()
    width, height = args.size

    background_gpu = gpu_utilization()
    swap_before, paging_before = swap_used_mb(), paging_counters()
    timings: dict[str, float | list[float]] = {"imports_s": time.perf_counter() - PROCESS_START}
    peaks = {}
    peak_footprint_gib(_INTERVAL_MAX_FOOTPRINT)

    pipe = IrisPipeline(args.checkpoint, attention_dtype=getattr(torch, args.attention))
    start = time.perf_counter()
    pipe.encode([args.prompt, pipe.cfg.sample.negative_prompt])
    timings["text_encoding_s"] = time.perf_counter() - start
    peaks["text_encoding"] = peak_footprint_gib(_INTERVAL_MAX_FOOTPRINT)

    start = time.perf_counter()
    pipe.engine  # noqa: B018  (loads the diffusion model)
    torch.mps.synchronize()
    timings["model_load_s"] = time.perf_counter() - start
    peaks["model_load"] = peak_footprint_gib(_INTERVAL_MAX_FOOTPRINT)

    def generate(on_step=None):
        start = time.perf_counter()
        image = pipe(
            [args.prompt], height=height, width=width, steps=args.steps, seed=args.seed, on_step=on_step
        )[0]  # returns a CPU image, so the GPU work has finished
        return image, time.perf_counter() - start

    image, timings["first_image_s"] = generate()
    warm = [generate()[1] for _ in range(args.repeats - 1)]
    timings["warm_s"] = warm
    if warm:
        timings["warm_median_s"] = statistics.median(warm)
        timings["warm_per_step_s"] = timings["warm_median_s"] / args.steps
    peaks["sampling"] = peak_footprint_gib(_INTERVAL_MAX_FOOTPRINT)
    peaks["process"] = peak_footprint_gib(_LIFETIME_MAX_FOOTPRINT)

    report = {
        "config": {**vars(args), "size": [width, height]},
        "timings": timings,
        "peak_footprint_gib": peaks,
        "system": {
            "background_gpu_utilization_pct": background_gpu,
            "swap_growth_mb": swap_used_mb() - swap_before,
            **{f"{k.lower()}_during_run": v - paging_before[k] for k, v in paging_counters().items()},
        },
        "environment": environment(),
    }

    if args.profile:
        step_times: list[float] = []
        last = time.perf_counter()

        def on_step(step: int, total: int) -> None:
            nonlocal last
            torch.mps.synchronize()
            now = time.perf_counter()
            step_times.append(now - last)
            last = now

        last = time.perf_counter()
        generate(on_step)
        report["profile"] = {
            "synchronized_step_s": {
                "first": step_times[0],
                "median": statistics.median(step_times),
                "first_half_median": statistics.median(step_times[: len(step_times) // 2]),
                "second_half_median": statistics.median(step_times[len(step_times) // 2 :]),
            }
        }

    text = json.dumps(report, indent=2)
    print(text)
    if args.image:
        image.save(args.image)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
