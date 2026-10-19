"""One measurement per fresh process, importing the engine from ``--tree``."""

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from common import STATE, read_json, swapouts

DEVICE = torch.device("mps")


def check_imports(tree: Path) -> None:
    import iris3b
    import iris_metal

    for module in (iris3b, iris_metal):
        expected = (tree / "src" / module.__name__ / "__init__.py").resolve()
        if Path(module.__file__).resolve() != expected:
            raise SystemExit(f"{module.__name__} imported from {module.__file__}, expected {expected}")


def lifetime_peak_gib() -> float:
    buffer = (ctypes.c_uint64 * 64)()
    libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
    if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buffer)):
        raise OSError("proc_pid_rusage failed")
    return buffer[2 + 28] / 1024**3  # rusage_info_v4.ri_lifetime_max_phys_footprint


def tensor_digest(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise SystemExit("non-finite tensor in a capture")
    digest = hashlib.sha256(f"{tensor.dtype}{tuple(tensor.shape)}".encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def image_digests(image: Image.Image) -> dict:
    metadata = image.info.get("iris", "")
    return {
        "pixels": hashlib.sha256(f"{image.mode}{image.size}".encode() + image.tobytes()).hexdigest(),
        "metadata": hashlib.sha256(metadata.encode()).hexdigest(),
        "metadata_keys": sorted(json.loads(metadata)) if metadata else [],
    }


def encoding_digest(encodings) -> str:
    return hashlib.sha256("".join(tensor_digest(t) for pair in encodings for t in pair).encode()).hexdigest()


class ClampProbe:
    """First ``clamp(-1, 1)`` input of an image batch."""

    def __init__(self):
        self.sample: torch.Tensor | None = None
        original = torch.Tensor.clamp
        probe = self

        def clamp(tensor, *args, **kwargs):
            bounds = (
                kwargs.get("min", args[0] if args else None),
                kwargs.get("max", args[1] if len(args) > 1 else None),
            )
            if probe.sample is None and tensor.dim() == 4 and tensor.shape[1] == 3 and bounds == (-1, 1):
                probe.sample = tensor.detach().cpu().clone()
            return original(tensor, *args, **kwargs)

        torch.Tensor.clamp = clamp

    def take(self) -> torch.Tensor:
        if self.sample is None:
            raise SystemExit("the request never clamped its sample to [-1, 1]; the probe needs that call")
        sample, self.sample = self.sample, None
        return sample


def ocr_words(path: Path) -> list[str]:
    import Quartz
    import Vision
    from Foundation import NSURL

    source = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(str(path)), None)
    image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, {})
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise SystemExit(f"OCR failed: {error}")
    lines = [obs.topCandidates_(1)[0].string() for obs in request.results() or []]
    return sorted(
        {"".join(c for c in word if c.isalnum()).upper() for line in lines for word in line.split()} - {""}
    )


# -- modes -------------------------------------------------------------------
def resolve(run) -> dict:
    from iris_metal.cache import cached_snapshot
    from iris_metal.cli import REPO_ID, REVISION, resolve_checkpoint
    from iris_metal.tasks import TASK_FOLDERS

    found = {model: str(resolve_checkpoint(REPO_ID, None, model=model)) for model in ("original", "turbo")}
    for task, folder in TASK_FOLDERS.items():
        export = cached_snapshot(REPO_ID, REVISION, task=folder)
        if export is not None:
            found[task] = str(export)
    return found


def oracle(run) -> dict:
    """Upstream FP32 forward and sampler, no autocast."""
    from iris3b.flow.solver import FlowDPMSolver
    from iris_metal.loading import load_config, load_dit
    from iris_metal.pipeline import IrisPipeline, release_memory

    checkpoint = Path(run.config["checkpoints"]["original"])
    cfg = load_config(checkpoint)
    sample = cfg.sample
    jobs = run.spec["velocity"] if run.arg == "velocity" else run.spec["trajectories"][run.arg]
    pipe = IrisPipeline(checkpoint)
    encoded = []
    for job in jobs:
        (cond, cond_mask), (neg, neg_mask) = pipe.encode([job["prompt"], sample.negative_prompt])
        encoded.append((torch.cat([neg, cond]).float(), torch.cat([neg_mask, cond_mask])))
    del pipe
    release_memory()
    model = load_dit(checkpoint, cfg, DEVICE, torch.float32)
    for job, (states, mask) in zip(jobs, encoded, strict=True):
        device_states, device_mask = states.to(DEVICE), mask.to(DEVICE)
        calls = []

        def velocity(x, t, rows, states=device_states, mask=device_mask, calls=calls):
            v = model(x, t, states[rows], y_mask=mask[rows]).x.float()
            if run.arg == "velocity":
                calls.append((x.cpu(), t.cpu(), rows.cpu(), v.cpu()))
            return v

        rows = torch.tensor([0, 1], device=DEVICE)
        solver = FlowDPMSolver(
            velocity,
            num_timesteps=cfg.flow.num_train_timesteps,
            cfg_scale=sample.cfg_scale,
            cfg_interval=tuple(sample.cfg_interval),
            prediction=cfg.flow.prediction,
        )
        noise = torch.randn(
            1, 3, job["height"], job["width"], generator=torch.Generator().manual_seed(job["seed"])
        )
        start = time.perf_counter()
        with torch.no_grad():
            final = solver.sample(
                noise.to(DEVICE),
                rows[1:],
                rows[:1],
                steps=job["steps"],
                order=sample.order,
                shift=cfg.flow.shift,
            ).cpu()
        if run.arg == "velocity":
            k = run.spec["velocity_probes"]
            picks = sorted({round(i * (len(calls) - 1) / (k - 1)) for i in range(k)})
            for j, index in enumerate(picks):
                x, t, rows_used, v = calls[index]
                np.savez(
                    run.out / f"vel-{job['id']}-{j}.npz",
                    x=x,
                    t=t,
                    rows=rows_used,
                    v=v,
                    states=states,
                    mask=mask,
                )
        else:
            np.save(run.out / f"traj-{job['id']}.npy", final.numpy())
        print(f"oracle {run.arg} {job['id']}: {time.perf_counter() - start:.0f}s", flush=True)
    return {"jobs": [job["id"] for job in jobs]}


def capture_turbo(run) -> dict:
    from iris_metal.pipeline import IrisPipeline

    probe, pipe, result = ClampProbe(), IrisPipeline(Path(run.config["checkpoints"]["turbo"])), {}
    for job in run.spec["turbo"][run.arg]:
        [image] = pipe([job["prompt"]], height=job["height"], width=job["width"], seed=job["seed"])
        result[job["id"]] = {
            "preclamp": tensor_digest(probe.take()),
            "encoding": encoding_digest(pipe.encode([job["prompt"]])),
            **image_digests(image),
        }
    return result


def ocr_capture(run, pipe, probe) -> dict:
    """The baseline picks the first seed that renders every word; candidates reuse it."""
    job = run.spec["ocr"][run.arg]
    recorded = STATE / "baseline" / f"original-{run.arg}" / "result.json"
    seeds = [read_json(recorded)["ocr"]["seed"]] if recorded.exists() else job["seeds"]
    for seed in seeds:
        [image] = pipe(
            [job["prompt"]], height=job["height"], width=job["width"], steps=job["steps"], seed=seed
        )
        path = run.out / f"ocr-{job['id']}-{seed}.png"
        image.save(path)
        found = {"seed": seed, "preclamp": tensor_digest(probe.take()), **image_digests(image)}
        found["words"] = ocr_words(path)
        if set(job["words"]) <= set(found["words"]):
            break
    return found


def capture_original(run) -> dict:
    from iris_metal.pipeline import IrisPipeline

    probe, pipe = ClampProbe(), IrisPipeline(Path(run.config["checkpoints"]["original"]))
    negative = pipe.cfg.sample.negative_prompt
    result = {"trajectories": {}, "velocity": {}}
    for job in run.spec["trajectories"][run.arg]:
        [image] = pipe(
            [job["prompt"]], height=job["height"], width=job["width"], steps=job["steps"], seed=job["seed"]
        )
        sample = probe.take()
        np.save(run.out / f"traj-{job['id']}.npy", sample.float().numpy())
        result["trajectories"][job["id"]] = {
            "preclamp": tensor_digest(sample),
            "encoding": encoding_digest(pipe.encode([job["prompt"], negative])),
            **image_digests(image),
        }
    if run.arg == "dev":
        engine = pipe.engine
        for path in sorted((STATE / "oracle" / "velocity").glob("vel-*.npz")):
            with np.load(path) as data:
                x, t, rows = (torch.from_numpy(data[key]).to(DEVICE) for key in ("x", "t", "rows"))
                text = engine.encode_text(torch.from_numpy(data["states"]), torch.from_numpy(data["mask"]))
            v = engine(x, t, text[rows]).cpu()
            np.save(run.out / f"{path.stem}.npy", v.numpy())
            result["velocity"][path.stem] = tensor_digest(v)
    result["ocr"] = ocr_capture(run, pipe, probe)
    return result


def capture_tasks(run) -> dict:
    from iris3b.downstream.restoration import fit_budget
    from iris_metal.pipeline import release_memory
    from iris_metal.tasks import MetalDepth, MetalRestorer

    exports, inputs, result = run.config["checkpoints"], STATE / "inputs", {}
    if "depth" in exports:
        model = MetalDepth(Path(exports["depth"]))
        with Image.open(inputs / "depth" / Path(run.spec["tasks"]["depth"]).name) as image:
            depth = model(image, max_side=1024)
        np.save(run.out / "depth.npy", depth)
        result["depth"] = hashlib.sha256(depth.tobytes()).hexdigest()
        del model
        release_memory()
    if "restoration" in exports:
        model = MetalRestorer(Path(exports["restoration"]))
        with Image.open(inputs / "restoration" / Path(run.spec["tasks"]["restoration"]).name) as image:
            restored = model(fit_budget(image))
        restored.save(run.out / "restoration.png")
        result["restoration"] = image_digests(restored)["pixels"]
    return result


def timing(run) -> dict:
    """Cold covers construction and the first request; unique workloads get a per-process nonce."""
    from iris_metal.pipeline import IrisPipeline

    job = run.spec["workloads"][run.arg]
    width, height = run.spec["workload_size"]
    extra = {"steps": job["steps"]} if job["model"] == "original" else {}
    nonce = os.urandom(3).hex()
    prompts = [f"{p}, study {nonce}{i}" if job.get("unique") else p for i, p in enumerate(job["prompts"])]
    before, times = swapouts(), []
    start = time.perf_counter()
    pipe = IrisPipeline(Path(run.config["checkpoints"][job["model"]]))
    for seed, prompt in enumerate(prompts):
        pipe([prompt], height=height, width=width, seed=seed, **extra)
        times.append(time.perf_counter() - start)
        start = time.perf_counter()
    return {
        "cold": times[0],
        "warm": times[1:],
        "peak_gib": lifetime_peak_gib(),
        "swapouts": swapouts() - before,
    }


MODES = {
    "resolve": resolve,
    "oracle": oracle,
    "capture-turbo": capture_turbo,
    "capture-original": capture_original,
    "capture-tasks": capture_tasks,
    "time": timing,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arg", default="dev", help="split, or the workload for time")
    args = parser.parse_args()
    check_imports(args.tree.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    known = {name: read_json(STATE / f"{name}.json") for name in ("config", "spec") if args.mode != "resolve"}
    run = SimpleNamespace(out=args.out, arg=args.arg, **known)
    value = MODES[args.mode](run)
    (args.out / "result.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    sys.exit(main())
