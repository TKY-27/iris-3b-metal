"""Denoise a saved fixture: fixed conditioning and noise, nothing else varies.

Usage:
    python scripts/denoise_fixture.py fixture.npz --steps 10 --repeats 3 --out result.npz

``fixture.npz`` holds ``y`` (``[2*n, 300, 12, 2560]`` raw Qwen3-VL states,
unconditional rows first), ``mask`` (``[2*n, 300]``) and ``noise``
(``[n, 3, H, W]``). This is how two implementations are compared on identical
inputs: the text encoder and the random number generator are out of the
loop. Each repeat is timed from synchronized start to the finished FP32 state,
without per-step synchronization; text-adapter preparation is included.

``--reference fp32`` or ``--reference autocast`` runs the upstream PyTorch
model instead, with FP32 weights, in FP32 or under the BF16 autocast of
upstream's ``scripts/sample.py``: the baselines the engine is compared with.
"""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from iris3b.flow.solver import FlowDPMSolver
from iris_metal.cli import REPO_ID, resolve_checkpoint
from iris_metal.loading import load_config, load_dit
from iris_metal.model import MetalIris


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity() -> dict[str, str]:
    sources = {Path(__file__).resolve()}
    sources.update(
        Path(module.__file__).resolve()
        for name, module in sys.modules.items()
        if name.split(".")[0] in ("iris3b", "iris_metal") and getattr(module, "__file__", None)
    )
    return {str(path): sha256(path) for path in sorted(sources)}


def checkpoint_identity(checkpoint: Path) -> dict:
    names = ("model.safetensors", "config.yaml")
    files = {name: sha256(checkpoint / name) for name in names}
    return {"path": str(checkpoint.resolve()), "files": files}


def provenance(args, checkpoint: Path) -> dict:
    return {
        "fixture_sha256": sha256(args.fixture),
        "checkpoint": checkpoint_identity(checkpoint),
        "source_sha256": source_identity(),
        "dependencies": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "safetensors": importlib.metadata.version("safetensors"),
        },
        "steps": args.steps,
        "attention": args.attention,
        "reference": args.reference,
    }


def save_result(path: Path, final: torch.Tensor, times: list[float], load_s: float, identity: dict) -> None:
    if not bool(torch.isfinite(final).all()):
        raise ValueError("final sample contains non-finite values")
    with path.open("xb") as result:
        np.savez(
            result,
            final=final.cpu().numpy(),
            seconds=np.array(times),
            load_s=load_s,
            provenance=np.array(json.dumps(identity, sort_keys=True)),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--checkpoint", default=REPO_ID, help="local directory or Hugging Face repo")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--attention", choices=("float16", "bfloat16", "float32"), default="float16", help="the engine only"
    )
    parser.add_argument("--reference", choices=("fp32", "autocast"), help="run the upstream model instead")
    parser.add_argument("--out", type=Path, help="write the final FP32 state and timings")
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    if args.out is not None and args.out.exists():
        raise SystemExit(f"{args.out} exists; use a new output path")
    args.checkpoint = resolve_checkpoint(args.checkpoint, None)

    fixture = np.load(args.fixture)
    device = torch.device("mps")
    cfg = load_config(args.checkpoint)
    states, mask = torch.from_numpy(fixture["y"]), torch.from_numpy(fixture["mask"])
    noise = torch.from_numpy(fixture["noise"]).to(device)
    batch = noise.shape[0]
    if states.shape[0] != 2 * batch or mask.shape[0] != 2 * batch:
        parser.error("fixture conditioning must have 2*n rows in [unconditional, conditional] order")
    sample = cfg.sample
    start = time.perf_counter()
    if args.reference:
        model = load_dit(args.checkpoint, cfg, device, torch.float32)
        states, mask = states.to(device).float(), mask.to(device)

        def velocity(x: torch.Tensor, t: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
            # ``rows`` index the fixture's text states, as ``text`` does for the engine
            with torch.autocast("mps", dtype=torch.bfloat16, enabled=args.reference == "autocast"):
                return model(x, t, states[rows], y_mask=mask[rows]).x.float()

        denoiser, prepare = velocity, lambda: torch.arange(2 * batch, device=device)
    else:
        engine = MetalIris(
            load_dit(args.checkpoint, cfg, device), attention_dtype=getattr(torch, args.attention)
        )
        denoiser, prepare = engine, lambda: engine.encode_text(states, mask)
    torch.mps.synchronize()
    load_s = time.perf_counter() - start

    def denoise() -> torch.Tensor:
        text = prepare()
        solver = FlowDPMSolver(
            denoiser,
            num_timesteps=cfg.flow.num_train_timesteps,
            cfg_scale=sample.cfg_scale,
            cfg_interval=tuple(sample.cfg_interval),
            prediction=cfg.flow.prediction,
        )
        return solver.sample(
            noise, text[batch:], text[:batch], steps=args.steps, order=sample.order, shift=cfg.flow.shift
        )

    times = []
    for _ in range(args.repeats):
        torch.mps.synchronize()
        start = time.perf_counter()
        final = denoise()
        torch.mps.synchronize()
        if not bool(torch.isfinite(final).all()):
            raise ValueError("final sample contains non-finite values")
        times.append(time.perf_counter() - start)
    print(
        f"load {load_s:.1f}s; denoising {args.steps} steps: first {times[0]:.2f}s, "
        f"median of repeats {statistics.median(times[1:] or times):.2f}s ({', '.join(f'{t:.2f}' for t in times)})"
    )
    if args.out:
        save_result(args.out, final, times, load_s, provenance(args, args.checkpoint))


if __name__ == "__main__":
    main()
