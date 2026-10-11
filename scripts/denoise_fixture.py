"""Denoise a saved fixture: fixed conditioning and noise, nothing else varies.

Usage:
    python scripts/denoise_fixture.py fixture.npz --steps 10 --repeats 3 --out result.npz

``fixture.npz`` holds ``y`` (``[2, 300, 12, 2560]`` raw Qwen3-VL states,
unconditional row first), ``mask`` (``[2, 300]``) and ``noise``
(``[1, 3, H, W]``). This is how two implementations are compared on identical
inputs: the text encoder and the random number generator are out of the
loop. Each repeat is timed from synchronized start to the finished FP32 state,
without per-step synchronization; text-adapter preparation is included.

``--reference fp32`` or ``--reference autocast`` runs the upstream PyTorch
model instead, with FP32 weights, in FP32 or under the BF16 autocast of
upstream's ``scripts/sample.py``: the baselines the engine is compared with.
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from iris3b.flow.solver import FlowDPMSolver
from iris_metal.cli import REPO_ID, resolve_checkpoint
from iris_metal.loading import load_config, load_dit
from iris_metal.model import MetalIris


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--checkpoint", default=REPO_ID, help="local directory or Hugging Face repo")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--attention", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--reference", choices=("fp32", "autocast"), help="run the upstream model instead")
    parser.add_argument("--out", type=Path, help="write the final FP32 state and timings")
    args = parser.parse_args()
    args.checkpoint = resolve_checkpoint(args.checkpoint, None)

    fixture = np.load(args.fixture)
    device = torch.device("mps")
    cfg = load_config(args.checkpoint)
    states, mask = torch.from_numpy(fixture["y"]), torch.from_numpy(fixture["mask"])
    noise = torch.from_numpy(fixture["noise"]).to(device)
    sample = cfg.sample
    start = time.perf_counter()
    if args.reference:
        model = load_dit(args.checkpoint, cfg, device, torch.float32)
        states, mask = states.to(device).float(), mask.to(device)

        def velocity(x: torch.Tensor, t: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
            # ``rows`` index the fixture's text states, as ``text`` does for the engine
            with torch.autocast("mps", dtype=torch.bfloat16, enabled=args.reference == "autocast"):
                return model(x, t, states[rows], y_mask=mask[rows]).x.float()

        denoiser, prepare = velocity, lambda: torch.arange(2, device=device)
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
            noise, text[1:], text[:1], steps=args.steps, order=sample.order, shift=cfg.flow.shift
        )

    times = []
    for _ in range(args.repeats):
        torch.mps.synchronize()
        start = time.perf_counter()
        final = denoise()
        torch.mps.synchronize()
        times.append(time.perf_counter() - start)
    print(
        f"load {load_s:.1f}s; denoising {args.steps} steps: first {times[0]:.2f}s, "
        f"median of repeats {statistics.median(times[1:] or times):.2f}s ({', '.join(f'{t:.2f}' for t in times)})"
    )
    if args.out:
        np.savez(args.out, final=final.cpu().numpy(), seconds=np.array(times), load_s=load_s)


if __name__ == "__main__":
    main()
