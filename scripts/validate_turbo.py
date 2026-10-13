"""Serial, saved-noise Turbo numerical and timing study over one fixture.

Modes: ``prepare`` saves the prompt's encoder states and noise; ``fp32`` is the
upstream forward without autocast (the oracle); ``reference`` and
``upstream-bf16`` are the official BF16 autocast recipe with FP32 and BF16
linear storage; ``metal-fp16`` is the rejected Turbo candidate (``MetalIris``, as
Original uses) and ``metal-bf16`` its BF16-trunk variant. Production Turbo
uses the upstream BF16 recipe. Every mode after
``prepare`` records its velocities and reports its distance from the saved
``reference`` and ``fp32`` results. Run each mode in its own process to keep
one model resident.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from iris3b.sampling import generate_few_step
from iris_metal.loading import load_config, load_dit
from iris_metal.model import MetalIris, require_finite
from iris_metal.pipeline import IrisPipeline, to_images
from iris_metal.turbo import TURBO_SIGMAS, checkpoint_revision


def sync():
    torch.mps.synchronize()


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
    parser.add_argument("--checkpoint", type=Path, default=Path("iris-3b/turbo"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", default="A red fox sleeping in fresh snow, golden hour")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fixture = args.out / "fixture.pt"
    if args.mode == "prepare":
        start = time.perf_counter()
        pipe = IrisPipeline(args.checkpoint)
        encodings = pipe.encode([args.prompt])
        torch.save(
            {
                "states": encodings[0][0],
                "mask": encodings[0][1],
                "noise": torch.randn(
                    1, 3, args.size, args.size, generator=torch.Generator().manual_seed(args.seed)
                ),
                "prompt": args.prompt,
                "seed": args.seed,
            },
            fixture,
        )
        (args.out / "prepare.json").write_text(json.dumps({"encoding_seconds": time.perf_counter() - start}))
        return
    data = torch.load(fixture, map_location="cpu", weights_only=True)
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
                trajectories.append(out.x.detach().float().cpu())
            return out

    start = time.perf_counter()
    adapted = None if engine is None else engine.encode_text(data["states"], data["mask"])
    sync()
    adapter_seconds = time.perf_counter() - start

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
            )

    recording = True
    start = time.perf_counter()
    result = run()
    sync()
    recorded_seconds = time.perf_counter() - start
    torch.save({"steps": trajectories, "final": result.cpu()}, args.out / f"{args.mode}.pt")
    to_images(require_finite(result))[0].save(args.out / f"{args.mode}.png")
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
        "checkpoint_dtype": "BF16",
        "linear_storage_dtype": str(dtype),
        "attention_dtype": str(attention) if engine else "fp32" if args.mode == "fp32" else "bf16-autocast",
        "torch": torch.__version__,
        "load_seconds": load_seconds,
        "adapter_seconds": adapter_seconds,
        "recorded_seconds": recorded_seconds,
        "warm_seconds": times,
        "allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_bytes": torch.mps.driver_allocated_memory(),
        "png_sha256": hashlib.sha256((args.out / f"{args.mode}.png").read_bytes()).hexdigest(),
    }

    def metrics(a, b):
        difference = (a - b).float()
        return {
            "max_abs": difference.abs().max().item(),
            "rmse": difference.square().mean().sqrt().item(),
            "relative_l2": (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(b)).item(),
        }

    # Compare both reference precision recipes.
    for name in ("reference", "fp32"):
        saved = args.out / f"{name}.pt"
        if args.mode != name and saved.exists():
            baseline = torch.load(saved, map_location="cpu", weights_only=True)
            key = "deltas" if name == "reference" else "fp32_deltas"
            stats[key] = [metrics(a, b) for a, b in zip(trajectories, baseline["steps"], strict=True)]
            stats[key.replace("deltas", "final_delta")] = metrics(result.cpu(), baseline["final"])
    (args.out / f"{args.mode}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
