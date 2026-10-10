"""Denoise a saved fixture: fixed conditioning and noise, nothing else varies.

Usage:
    python scripts/denoise_fixture.py fixture.npz --steps 10 --repeats 3 --out result.npz

``fixture.npz`` holds ``y`` (``[2, 300, 12, 2560]`` raw Qwen3-VL states,
unconditional row first), ``mask`` (``[2, 300]``) and ``noise``
(``[1, 3, H, W]``). This is how two implementations are compared on identical
inputs: the text encoder and the random number generator are out of the
loop. Each repeat is timed from synchronized start to the finished FP32 state,
without per-step synchronization; text-adapter preparation is included.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from iris3b.flow.solver import FlowDPMSolver
from iris_metal.loading import load_config, load_dit
from iris_metal.model import MetalIris


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--checkpoint", default="iris-3b")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--attention", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--out", type=Path, help="write the final FP32 state and timings")
    args = parser.parse_args()

    fixture = np.load(args.fixture)
    device = torch.device("mps")
    cfg = load_config(args.checkpoint)
    start = time.perf_counter()
    engine = MetalIris(load_dit(args.checkpoint, cfg, device), attention_dtype=getattr(torch, args.attention))
    torch.mps.synchronize()
    load_s = time.perf_counter() - start
    states, mask = torch.from_numpy(fixture["y"]), torch.from_numpy(fixture["mask"])
    noise = torch.from_numpy(fixture["noise"]).to(device)
    sample = cfg.sample

    def denoise() -> torch.Tensor:
        text = engine.encode_text(states, mask)
        solver = FlowDPMSolver(
            engine,
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
