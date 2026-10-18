"""Serial, saved-noise Turbo numerical and timing study over one fixture.

Modes: ``prepare`` saves the prompt's encoder states and noise; ``fp32`` is the
upstream forward without autocast (the oracle); ``reference`` and
``upstream-bf16`` are the official BF16 autocast recipe with FP32 and BF16
linear storage; ``metal-fp16`` is the rejected Turbo candidate (``MetalIris``, as
Original uses) and ``metal-bf16`` its BF16-trunk variant. Production Turbo
uses the upstream BF16 recipe. Every mode after
``prepare`` records its velocities and reports its distance from the saved
``reference`` and ``fp32`` results, including whether they are bitwise equal.
Run each mode in its own process to keep one model resident. Existing
fixtures and results are never replaced: a rerun needs a new ``--out``. A
non-finite final sample stops a mode before it writes anything.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from iris3b.sampling import generate_few_step
from iris_metal.loading import load_config, load_dit
from iris_metal.model import MetalIris, require_finite
from iris_metal.pipeline import IrisPipeline, to_images
from iris_metal.turbo import TURBO_SIGMAS, checkpoint_revision

DEFAULT_CHECKPOINT = Path("iris-3b/turbo")
DEFAULT_PROMPT = "A red fox sleeping in fresh snow, golden hour"
MANIFEST_VERSION = 1


def sync():
    torch.mps.synchronize()


def create(path: Path):
    """A new file; an existing fixture or result is never replaced."""
    try:
        return open(path, "xb")
    except FileExistsError:
        raise SystemExit(f"{path} exists; write a new run to a new --out") from None


def require_new(paths):
    for path in paths:
        if path.exists():
            raise SystemExit(f"{path} exists; write a new run to a new --out")


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def checkpoint_identity(checkpoint: Path) -> dict:
    root = checkpoint.resolve()
    names = ("model.safetensors", "config.yaml", "distill.yaml")
    files = {name: file_sha256(root / name) for name in names if (root / name).is_file()}
    if len(files) != len(names):
        raise ValueError(f"checkpoint must contain {', '.join(names)}: {root}")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"path": str(root), "sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def validate_fixture_data(data: dict, manifest: dict) -> None:
    names = ("states", "mask", "noise")
    if not all(isinstance(data.get(name), torch.Tensor) for name in names):
        raise ValueError("fixture tensors are missing")
    shapes = {name: list(data[name].shape) for name in names}
    ranks = {"states": 4, "mask": 2, "noise": 4}
    if shapes != manifest["tensor_shapes"] or any(len(shapes[n]) != ranks[n] for n in names):
        raise ValueError("fixture tensor shapes do not match manifest")
    if shapes["states"][0] != 1 or shapes["mask"][0] != 1 or shapes["noise"][:2] != [1, 3]:
        raise ValueError("fixture batch shapes are invalid")
    if shapes["states"][1] != shapes["mask"][1]:
        raise ValueError("fixture states and mask lengths differ")
    if not all(bool(torch.isfinite(data[name]).all()) for name in names):
        raise ValueError("fixture contains non-finite values")
    if data.get("prompt") != manifest["prompt"] or data.get("seed") != manifest["seed"]:
        raise ValueError("fixture payload metadata does not match manifest")
    if data.get("size") != manifest["size"]:
        raise ValueError("fixture size does not match manifest")


def save_fixture(out: Path, data: dict, identity: dict, checkpoint: Path, seconds: float) -> dict:
    fixture = out / "fixture.pt"
    with create(fixture) as file:
        torch.save(data, file)
    manifest = {
        "version": MANIFEST_VERSION,
        "checkpoint": identity,
        "checkpoint_revision": checkpoint_revision(checkpoint),
        "fixture_sha256": file_sha256(fixture),
        **{key: data[key] for key in ("prompt", "seed", "size")},
        "tensor_shapes": {name: list(data[name].shape) for name in ("states", "mask", "noise")},
        "encoding_seconds": seconds,
    }
    with create(out / "prepare.json") as file:
        file.write(json.dumps(manifest, indent=2).encode())
    return manifest


def load_fixture(out: Path, requested_checkpoint=None, prompt=None, seed=None, size=None):
    fixture, manifest_path = out / "fixture.pt", out / "prepare.json"
    if not fixture.is_file() or not manifest_path.is_file():
        raise SystemExit("fixture provenance is missing; prepare a new run in a new --out")
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("version") != MANIFEST_VERSION:
            raise ValueError("unsupported fixture manifest")
        if file_sha256(fixture) != manifest["fixture_sha256"]:
            raise ValueError("fixture hash does not match manifest")
        checkpoint = Path(manifest["checkpoint"]["path"])
        if requested_checkpoint is not None:
            checkpoint = requested_checkpoint.resolve()
        identity = checkpoint_identity(checkpoint)
        if identity["sha256"] != manifest["checkpoint"]["sha256"]:
            raise ValueError("checkpoint content does not match fixture")
        for name, requested in (("prompt", prompt), ("seed", seed), ("size", size)):
            if requested is not None and requested != manifest[name]:
                raise ValueError(f"--{name} conflicts with prepared fixture")
        data = torch.load(fixture, map_location="cpu", weights_only=True)
        validate_fixture_data(data, manifest)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid prepared fixture: {error}") from None
    return data, manifest, checkpoint, identity


def load_oracle(path: Path, manifest: dict, identity: dict, dependencies: dict) -> dict:
    result = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "fixture_sha256": manifest["fixture_sha256"],
        "checkpoint_sha256": identity["sha256"],
    }
    if result.get("provenance") != expected:
        raise ValueError("saved result provenance does not match prepared fixture")
    if result.get("dependencies") != dependencies:
        raise ValueError("saved result dependencies do not match current environment")
    return result


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if a.shape != b.shape:
        raise ValueError(f"tensor shape mismatch: {list(a.shape)} != {list(b.shape)}")
    finite_a, finite_b = bool(torch.isfinite(a).all()), bool(torch.isfinite(b).all())
    equal = a.dtype == b.dtype and a.view(torch.uint8).equal(b.view(torch.uint8))
    result = {
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "shape": list(a.shape),
        "finite_a": finite_a,
        "finite_b": finite_b,
        "bitwise_equal": equal,
    }
    difference = a.float() - b.float() if finite_a and finite_b else None
    if difference is None:
        return {**result, "max_abs": None, "rmse": None, "relative_l2": None}
    norm, distance = torch.linalg.vector_norm(b.float()), torch.linalg.vector_norm(difference)
    return {
        **result,
        "max_abs": difference.abs().max().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "relative_l2": (distance / norm).item() if norm.item() else (0.0 if not distance.item() else None),
    }


def trajectory_metrics(actual, reference):
    return [metrics(a, b) for a, b in zip(actual, reference, strict=True)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("prepare", "reference", "fp32", "metal-bf16", "metal-fp16", "upstream-bf16")
    )
    parser.add_argument(
        "--attention",
        choices=("float32", "float16", "bfloat16"),
        help="Metal modes: attention precision (default: the trunk dtype)",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fixture = args.out / "fixture.pt"
    outputs = (
        [fixture, args.out / "prepare.json"]
        if args.mode == "prepare"
        else [args.out / f"{args.mode}.{suffix}" for suffix in ("pt", "png", "json")]
    )
    require_new(outputs)
    if args.mode == "prepare":
        args.checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
        args.prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPT
        args.size = args.size if args.size is not None else 1024
        args.seed = args.seed if args.seed is not None else 0
        if args.size < 1:
            parser.error("--size must be positive")
        if args.repeats < 1:
            parser.error("--repeats must be positive")
        identity = checkpoint_identity(args.checkpoint)
        start = time.perf_counter()
        pipe = IrisPipeline(args.checkpoint)
        encodings = pipe.encode([args.prompt])
        data = {
            "states": encodings[0][0],
            "mask": encodings[0][1],
            "noise": torch.randn(
                1, 3, args.size, args.size, generator=torch.Generator().manual_seed(args.seed)
            ),
            "prompt": args.prompt,
            "seed": args.seed,
            "size": args.size,
        }
        save_fixture(args.out, data, identity, args.checkpoint, time.perf_counter() - start)
        return
    data, manifest, args.checkpoint, identity = load_fixture(
        args.out, args.checkpoint, args.prompt, args.seed, args.size
    )
    args.prompt, args.seed, args.size = manifest["prompt"], manifest["seed"], manifest["size"]
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    baseline_names = ("reference", "fp32")
    dependencies = {
        "torch": str(torch.__version__),
        "transformers": importlib.metadata.version("transformers"),
        "safetensors": importlib.metadata.version("safetensors"),
    }
    baseline_paths = {name: args.out / f"{name}.pt" for name in baseline_names}
    if args.mode not in baseline_names:
        missing = [name for name, path in baseline_paths.items() if not path.is_file()]
        if missing:
            raise SystemExit(f"prepare saved oracle results first: {', '.join(missing)}")
    baselines = {
        name: load_oracle(path, manifest, identity, dependencies)
        for name, path in baseline_paths.items()
        if name != args.mode and path.is_file()
    }
    cfg = load_config(args.checkpoint)
    start = time.perf_counter()
    dtype = (
        torch.float32
        if args.mode in ("reference", "fp32")
        else torch.bfloat16
        if args.mode in ("metal-bf16", "upstream-bf16")
        else torch.float16
    )
    dit = load_dit(args.checkpoint, cfg, torch.device("mps"), dtype=dtype)
    sync()
    load_seconds = time.perf_counter() - start
    upstream = args.mode in ("reference", "fp32", "upstream-bf16")
    attention = getattr(torch, args.attention) if args.attention else dtype
    engine = None if upstream else MetalIris(dit, attention_dtype=attention)
    trajectories = []

    class Encoder:
        def encode(self, prompts):
            return SimpleNamespace(embeddings=data["states"], mask=data["mask"])

    class Model:
        cfg = dit.cfg

        def parameters(self):
            return dit.parameters()

        def __call__(self, x, t, y, y_mask):
            if engine is None:
                out = dit(x, t, y, y_mask=y_mask)
            else:
                out = SimpleNamespace(x=engine(x, t, adapted))
            if recording:
                sync()
                trajectories.append(out.x.detach().cpu())
            return out

    start = time.perf_counter()
    adapted = None if engine is None else engine.encode_text(data["states"], data["mask"])
    sync()
    adapter_seconds = time.perf_counter() - start

    preclamps = []

    def check_sample(sample):
        require_finite(sample)
        if recording:
            preclamps.append(sample.detach().cpu().clone())

    def run():
        # FP32 reference uses no autocast.
        autocast = engine is None and args.mode != "fp32"
        with torch.no_grad(), torch.autocast("mps", dtype=torch.bfloat16, enabled=autocast):
            return generate_few_step(
                Model(),
                Encoder(),
                [data["prompt"]],
                args.size,
                args.size,
                list(TURBO_SIGMAS),
                noise=data["noise"],
                device="mps",
                num_train_timesteps=cfg.flow.num_train_timesteps,
                prediction=cfg.flow.prediction,
                check=check_sample,
            )

    recording = True
    start = time.perf_counter()
    result = run()
    sync()
    if len(preclamps) != 1:
        raise RuntimeError("sampler did not produce one pre-clamp sample")
    preclamp = preclamps[0]
    recorded_result = result.detach().cpu()
    recorded_seconds = time.perf_counter() - start
    recording = False
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = run()
        sync()
        times.append(time.perf_counter() - start)
    stats = {
        "mode": args.mode,
        "hf_revision": checkpoint_revision(args.checkpoint),
        "checkpoint_sha256": identity["sha256"],
        "fixture_sha256": manifest["fixture_sha256"],
        "prompt": args.prompt,
        "seed": args.seed,
        "size": args.size,
        "checkpoint_dtype": "BF16",
        "linear_storage_dtype": str(dtype),
        "attention_dtype": str(attention) if engine else "fp32" if args.mode == "fp32" else "bf16-autocast",
        "torch": torch.__version__,
        "dependencies": dependencies,
        "source": {
            name: {
                "path": str(Path(module.__file__).resolve()),
                "sha256": file_sha256(Path(module.__file__).resolve()),
            }
            for name, module in sorted(sys.modules.items())
            if name.split(".")[0] in ("iris3b", "iris_metal") and getattr(module, "__file__", None)
        },
        "load_seconds": load_seconds,
        "adapter_seconds": adapter_seconds,
        "recorded_seconds": recorded_seconds,
        "warm_seconds": times,
        "allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_bytes": torch.mps.driver_allocated_memory(),
    }

    # Compare both reference precision recipes.
    for name, baseline in baselines.items():
        key = "deltas" if name == "reference" else "fp32_deltas"
        stats[key] = trajectory_metrics(trajectories, baseline["steps"])
        stats[key.replace("deltas", "final_delta")] = metrics(recorded_result, baseline["final"])
        stats[key.replace("deltas", "preclamp_delta")] = metrics(preclamp, baseline["preclamp"])
    png_path = args.out / f"{args.mode}.png"
    with create(args.out / f"{args.mode}.pt") as file:
        torch.save(
            {
                "steps": trajectories,
                "preclamp": preclamp,
                "final": recorded_result,
                "provenance": {
                    "fixture_sha256": manifest["fixture_sha256"],
                    "checkpoint_sha256": identity["sha256"],
                },
                "source": stats["source"],
                "dependencies": stats["dependencies"],
            },
            file,
        )
    with create(png_path) as file:
        to_images(recorded_result, clamped=True)[0].save(file, format="PNG")
    stats["png_sha256"] = file_sha256(png_path)
    with create(args.out / f"{args.mode}.json") as file:
        file.write(json.dumps(stats, indent=2).encode())
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
