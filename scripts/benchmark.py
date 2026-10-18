"""Iris-3B generation latency and memory on an Apple GPU.

Usage:
    python scripts/benchmark.py --size 1024x1024 --steps 100 --repeats 3 --out bench.json

One fresh process measures, in order: imports, text encoding, model load, the
first image (which includes compiling Metal shaders) and ``--repeats - 1``
warm requests with the models resident and the prompt encoding cached, run
exactly as in normal use (the pipeline waits for each step one step behind,
so the GPU always has the next one queued); ``--profile`` adds a separate
diagnostic request reporting completion callback intervals without global
per-step synchronization, and is never mixed into the timings. The commit recorded is the
one of the code that ran, marked "-dirty" if it had uncommitted changes.

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
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

import iris3b  # noqa: E402
import iris_metal  # noqa: E402
from iris_metal.cli import REPO_ID, parse_size, resolve_checkpoint  # noqa: E402
from iris_metal.pipeline import IrisPipeline, StreamingQwen3VLEncoder  # noqa: E402

IMPORTS_S = time.perf_counter() - PROCESS_START

_RUSAGE_INFO_V4 = 4
# rusage_info_v4 fields, as uint64 indices after the 16-byte uuid
_CURRENT_FOOTPRINT = 7
_LIFETIME_MAX_FOOTPRINT = 28
_INTERVAL_MAX_FOOTPRINT = 33  # explicitly reset after reading by the helper below


def peak_footprint_gib(field: int) -> float:
    libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
    buffer = (ctypes.c_uint64 * 64)()
    if libc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(buffer)):
        raise OSError("proc_pid_rusage failed")
    value = buffer[2 + field] / 1024**3
    # proc_pid_rusage reads the interval maximum without resetting it.
    if field == _INTERVAL_MAX_FOOTPRINT and libc.proc_reset_footprint_interval(os.getpid()):
        raise OSError("proc_reset_footprint_interval failed")
    return value


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
    repo = Path(__file__).resolve().parents[1]
    sources = [
        repo / "scripts/benchmark.py",
        *(repo / "src" / "iris3b").rglob("*.py"),
        *(repo / "src" / "iris_metal").rglob("*.py"),
    ]
    return {
        "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_gb": int(command("sysctl", "-n", "hw.memsize")) / 1e9,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "mps_environment": {k: v for k, v in os.environ.items() if "MPS" in k or k.startswith("PYTORCH")},
        "power": command("pmset", "-g", "batt"),
        "power_modes": command("pmset", "-g", "custom"),
        "commit": source_commit(),
        "source_sha256": {
            path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources)
        },
    }


def source_commit() -> str:
    """The commit of the repository this script (and the engine it imports) belongs to."""
    repo = Path(__file__).resolve().parents[1]
    for module in (iris3b, iris_metal):
        expected = (repo / "src" / module.__name__ / "__init__.py").resolve()
        actual = Path(module.__file__).resolve()
        if actual != expected:
            raise RuntimeError(f"{module.__name__} loaded from {actual}, expected {expected}")
    commit = command("git", "-C", str(repo), "rev-parse", "--short", "HEAD")
    if not commit:
        raise RuntimeError(f"cannot identify the commit of {repo}")
    dirty = command("git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no")
    return f"{commit}-dirty" if dirty else commit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=REPO_ID, help="local directory or Hugging Face repo")
    parser.add_argument("--prompt", default="A red fox sleeping in fresh snow, golden hour")
    parser.add_argument("--size", type=parse_size, default=(1024, 1024), help="WIDTHxHEIGHT")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3, help="requests in one process; the first is cold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--uncached-prompt", default="A ceramic teapot on a wooden table, morning light")
    parser.add_argument("--attention", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--profile", action="store_true", help="add a completion-callback diagnostic request")
    parser.add_argument("--profile-prompt", help="separate diagnostic prompt, e.g. an uncached warm prompt")
    parser.add_argument("--image", help="optional path for the last generated image")
    parser.add_argument("--out", help="optional JSON report path")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    width, height = args.size
    outputs = [Path(path) for path in (args.image, args.out) if path]
    if len({path.resolve() for path in outputs}) != len(outputs):
        parser.error("--image and --out must be different paths")
    for path in outputs:
        if os.path.lexists(path):
            parser.error(f"output already exists: {path}")
    image_format = None
    if args.image:
        from PIL import Image

        image_format = Image.registered_extensions().get(Path(args.image).suffix.lower())
        if image_format not in Image.SAVE:
            parser.error(f"unsupported image output format: {args.image}")
    try:
        source_commit()
    except RuntimeError as exc:
        parser.error(str(exc))

    background_gpu = gpu_utilization()
    swap_before, paging_before = swap_used_mb(), paging_counters()
    timings: dict[str, float | list[float]] = {"imports_s": IMPORTS_S}
    peaks = {}
    peak_footprint_gib(_INTERVAL_MAX_FOOTPRINT)

    first_request_start = time.perf_counter()
    start = time.perf_counter()
    checkpoint = resolve_checkpoint(args.checkpoint, None)
    timings["checkpoint_resolution_s"] = time.perf_counter() - start
    start = time.perf_counter()
    pipe = IrisPipeline(checkpoint, attention_dtype=getattr(torch, args.attention))
    timings["pipeline_config_init_s"] = time.perf_counter() - start
    start = time.perf_counter()
    pipe._text_encoder = StreamingQwen3VLEncoder(pipe.cfg.text_encoder, pipe.device)
    timings["text_encoder_tokenizer_init_s"] = time.perf_counter() - start
    start = time.perf_counter()
    pipe.encode([args.prompt, pipe.cfg.sample.negative_prompt])
    timings["text_encoding_s"] = time.perf_counter() - start
    if args.uncached_prompt in pipe._encodings:
        parser.error("--uncached-prompt must differ from cached prompts")
    peaks["text_encoding"] = peak_footprint_gib(_INTERVAL_MAX_FOOTPRINT)

    start = time.perf_counter()
    pipe.engine  # noqa: B018  (loads the diffusion model)
    torch.mps.synchronize()
    timings["model_load_s"] = time.perf_counter() - start
    peaks["model_load"] = peak_footprint_gib(_INTERVAL_MAX_FOOTPRINT)

    def generate(on_step=None, prompt=None):
        start = time.perf_counter()
        image = pipe(
            [args.prompt if prompt is None else prompt],
            height=height,
            width=width,
            steps=args.steps,
            seed=args.seed,
            on_step=on_step,
        )[0]  # returns a CPU image, so the GPU work has finished
        return image, time.perf_counter() - start

    image, timings["first_image_s"] = generate()
    timings["first_request_including_preparation_s"] = time.perf_counter() - first_request_start
    timings["imports_and_first_request_s"] = IMPORTS_S + timings["first_request_including_preparation_s"]
    warm = [generate()[1] for _ in range(args.repeats - 1)]
    timings["warm_s"] = warm
    if warm:
        timings["warm_median_s"] = statistics.median(warm)
        timings["warm_request_average_per_step_s"] = timings["warm_median_s"] / args.steps
    start = time.perf_counter()
    pipe([args.uncached_prompt], height=height, width=width, steps=args.steps, seed=args.seed)
    timings["warm_engine_uncached_request_s"] = time.perf_counter() - start
    peaks["sampling"] = peak_footprint_gib(_INTERVAL_MAX_FOOTPRINT)
    peaks["process"] = peak_footprint_gib(_LIFETIME_MAX_FOOTPRINT)

    report = {
        "config": {**vars(args), "size": [width, height]},
        "timings": timings,
        "peak_footprint_gib": peaks,
        "system": {
            "background_gpu_utilization_pct": background_gpu,
            "file_cache": "unpurged OS cache; filesystem coldness not established",
            "memory_pressure": command("memory_pressure"),
            "mps_current_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_bytes": torch.mps.driver_allocated_memory(),
            "mps_recommended_bytes": torch.mps.recommended_max_memory(),
            "process_current_footprint_gib": peak_footprint_gib(_CURRENT_FOOTPRINT),
            "swap_growth_mb": swap_used_mb() - swap_before,
            **{f"{k.lower()}_during_run": v - paging_before[k] for k, v in paging_counters().items()},
        },
        "environment": environment(),
    }

    if args.profile:
        import iris_metal.pipeline as pipeline_module

        step_times: list[float] = []
        forwards = []
        adapter_events = []
        phase_times = {}
        first_submission = None
        request_begin = torch.mps.Event(enable_timing=True)
        last = time.perf_counter()
        original_engine = pipe.engine
        original_reporter = pipeline_module.StepReporter
        original_to_images = pipeline_module.to_images
        original_encode = pipe.encode

        def profile_encode(texts):
            start = time.perf_counter()
            result = original_encode(texts)
            phase_times["raw_encoding_call_s"] = time.perf_counter() - start
            return result

        class ProfileReporter(original_reporter):
            def __init__(self, *inputs):
                super().__init__(*inputs)
                self.started = time.perf_counter()
                phase_times["denoising_started"] = self.started

            def finish(self):
                super().finish()
                phase_times["completed_forward_steps_s"] = time.perf_counter() - self.started

        def profile_images(samples, **options):
            start = time.perf_counter()
            # require_finite has completed the final solver update before this
            # boundary; finish() alone only waits for the final model evaluation.
            phase_times["total_denoising_s"] = start - phase_times.pop("denoising_started")
            images = original_to_images(samples, **options)
            phase_times["image_conversion_s"] = time.perf_counter() - start
            return images

        class ProfileEngine:
            def __getattr__(self, name):
                return getattr(original_engine, name)

            def encode_text(self, *inputs):
                begin = torch.mps.Event(enable_timing=True)
                end = torch.mps.Event(enable_timing=True)
                begin.record()
                result = original_engine.encode_text(*inputs)
                end.record()
                adapter_events.append((begin, end))
                return result

            def __call__(self, *inputs):
                nonlocal first_submission
                begin = torch.mps.Event(enable_timing=True)
                end = torch.mps.Event(enable_timing=True)
                submitted = time.perf_counter()
                if first_submission is None:
                    first_submission = submitted - request_start
                begin.record()
                result = original_engine(*inputs)
                end.record()
                forwards.append((begin, end, time.perf_counter() - submitted))
                return result

        def on_step(step: int, total: int) -> None:
            nonlocal last
            now = time.perf_counter()
            step_times.append(now - last)
            last = now

        request_begin.record()
        request_start = last = time.perf_counter()
        pipe._engine = ProfileEngine()
        pipe.encode = profile_encode
        pipeline_module.StepReporter = ProfileReporter
        pipeline_module.to_images = profile_images
        try:
            generate(on_step, prompt=args.profile_prompt)
        finally:
            pipe._engine = original_engine
            pipe.encode = original_encode
            pipeline_module.StepReporter = original_reporter
            pipeline_module.to_images = original_to_images
        event_seconds = []
        for begin, end, _ in forwards:
            try:
                event_seconds.append(begin.elapsed_time(end) / 1000)
            except RuntimeError:
                event_seconds.append(None)
        try:
            # This runtime consumes the end event's notification in elapsed_time.
            # Read each event as an end only once, including the first interval.
            first_begin = request_begin.elapsed_time(forwards[0][0]) / 1000
            first_completion = first_begin + event_seconds[0] if event_seconds[0] is not None else None
        except RuntimeError:
            first_completion = None
        report["profile"] = {
            **phase_times,
            "first_model_host_submission_from_request_s": first_submission,
            "first_model_coarse_completion_from_request_s": first_completion,
            "forward_coarse_mps_event_s": event_seconds,
            "forward_host_submission_s": [submitted for _, _, submitted in forwards],
            "completion_callback_intervals_s": {
                "first": step_times[0],
                "median": statistics.median(step_times),
                "first_half_median": statistics.median(step_times[: max(1, len(step_times) // 2)]),
                "second_half_median": statistics.median(step_times[len(step_times) // 2 :]),
            },
        }
        adapter_seconds = []
        for begin, end in adapter_events:
            try:
                adapter_seconds.append(begin.elapsed_time(end) / 1000)
            except RuntimeError:
                adapter_seconds.append(None)
        report["profile"]["adapter_coarse_mps_event_s"] = adapter_seconds
        report["profile"]["prompt"] = args.prompt if args.profile_prompt is None else args.profile_prompt
        report["profile"]["note"] = (
            "StepReporter completion callback intervals include bounded lookahead; "
            "MPS event intervals are coarse command-completion times, not hardware kernel timestamps. "
            "Events are read after the request; unavailable intervals are null. "
            "No global per-step synchronization. Diagnostics are excluded from acceptance timings."
            " Total denoising includes noise preparation, final solver work and the finite-value check; "
            "completed_forward_steps ends at StepReporter.finish."
        )
        if all(interval is not None for interval in event_seconds):
            report["profile"]["nonforward_denoising_residual_s"] = phase_times["total_denoising_s"] - sum(
                event_seconds
            )
        report["profile"]["note"] += (
            " The residual includes solver/CFG, submission and completion waits; "
            "it is not an isolated GPU solver kernel measurement. "
            "The diagnostic engine wrapper invalidates adapted conditioning, so this request has adapter misses."
        )

    text = json.dumps(report, indent=2)
    print(text)
    if args.image:
        with open(args.image, "xb") as file:
            image.save(file, format=image_format)
    if args.out:
        with open(args.out, "x", encoding="utf-8") as file:
            file.write(text + "\n")


if __name__ == "__main__":
    main()
