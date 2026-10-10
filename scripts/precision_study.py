"""Compare complete generations with the upstream code run in full FP32.

Usage:
    python scripts/precision_study.py fixtures --out runs/precision --ids portrait-fisherman sign-bread
    python scripts/precision_study.py run fp32 --out runs/precision          # the reference
    python scripts/precision_study.py run autocast --out runs/precision      # upstream's BF16 recipe
    python scripts/precision_study.py run engine --out runs/precision        # this engine
    python scripts/precision_study.py score --out runs/precision autocast engine

``fixtures`` saves each prompt's text-encoder states (with the negative
prompt) and its initial noise, so every implementation denoises exactly the
same inputs with the upstream sampler. ``run`` denoises every fixture in
``--out`` and skips results that already exist; the FP32 reference takes
about 8 minutes per 512 x 512 image on an M5. ``score`` reports NRMSE (RMS
error over RMS of the reference) and LPIPS against the FP32 results. Over a
whole trajectory small differences are amplified, so use several prompts.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from iris3b.flow.solver import FlowDPMSolver
from iris_metal.cli import REPO_ID, resolve_checkpoint
from iris_metal.loading import load_config, load_dit
from iris_metal.model import MetalIris

MODES = ("fp32", "autocast", "engine", "engine-bf16")


def make_fixtures(args, cfg) -> None:
    from iris_metal.pipeline import IrisPipeline

    prompts = {p["id"]: p["prompt"] for p in json.loads(Path(args.prompts).read_text())[args.split]}
    args.out.mkdir(parents=True, exist_ok=True)
    pipe = IrisPipeline(args.checkpoint)
    for seed, key in enumerate(args.ids):
        path = args.out / f"fixture-{key}.npz"
        if path.exists():  # results in --out were denoised from it; a new fixture needs a new --out
            print(f"fixture {key} exists, kept")
            continue
        (cond, cond_mask), (neg, neg_mask) = pipe.encode([prompts[key], cfg.sample.negative_prompt])
        noise = torch.randn(1, 3, args.size, args.size, generator=torch.Generator().manual_seed(seed))
        np.savez(
            path,
            y=torch.cat([neg, cond]).float().numpy(),  # unconditional row first
            mask=torch.cat([neg_mask, cond_mask]).numpy(),
            noise=noise.numpy(),
        )
        print(f"fixture {key} (seed {seed})")


def upstream_velocity(model, states: torch.Tensor, mask: torch.Tensor, autocast: bool):
    """The upstream model as the solver's velocity; ``rows`` index the (unconditional, conditional) states."""

    def velocity(x: torch.Tensor, t: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        with torch.autocast("mps", dtype=torch.bfloat16, enabled=autocast):
            return model(x, t, states[rows], y_mask=mask[rows]).x.float()

    return velocity


def denoise(args, cfg) -> None:
    device = torch.device("mps")
    sample = cfg.sample
    if args.mode in ("fp32", "autocast"):
        model = load_dit(args.checkpoint, cfg, device, torch.float32)
    else:
        dtype = torch.bfloat16 if args.mode == "engine-bf16" else torch.float16
        engine = MetalIris(load_dit(args.checkpoint, cfg, device, dtype), attention_dtype=torch.float16)

    for fixture in sorted(args.out.glob("fixture-*.npz")):
        key = fixture.stem.removeprefix("fixture-")
        target = args.out / f"{args.mode}-{key}.npz"
        if target.exists():
            continue
        data = np.load(fixture)
        states, mask = torch.from_numpy(data["y"]), torch.from_numpy(data["mask"])
        noise = torch.from_numpy(data["noise"]).to(device)
        if args.mode in ("fp32", "autocast"):
            model_fn = upstream_velocity(model, states.to(device), mask.to(device), args.mode == "autocast")
            rows = torch.tensor([0, 1], device=device)
            cond, uncond = rows[1:], rows[:1]
        else:
            text = engine.encode_text(states, mask)
            model_fn, cond, uncond = engine, text[1:], text[:1]
        solver = FlowDPMSolver(
            model_fn,
            num_timesteps=cfg.flow.num_train_timesteps,
            cfg_scale=sample.cfg_scale,
            cfg_interval=tuple(sample.cfg_interval),
            prediction=cfg.flow.prediction,
        )
        start = time.perf_counter()
        final = solver.sample(noise, cond, uncond, steps=args.steps, order=sample.order, shift=cfg.flow.shift)
        torch.mps.synchronize()
        np.savez(target, final=final.cpu().numpy())
        print(f"{args.mode} {key}: {time.perf_counter() - start:.0f}s", flush=True)


def score(args) -> None:
    import lpips

    net = lpips.LPIPS(net="alex", verbose=False)
    keys = sorted(p.stem.removeprefix("fp32-") for p in args.out.glob("fp32-*.npz"))
    for mode in args.modes:
        errors, distances = [], []
        for key in keys:
            reference = np.clip(np.load(args.out / f"fp32-{key}.npz")["final"], -1, 1)
            result = np.clip(np.load(args.out / f"{mode}-{key}.npz")["final"], -1, 1)
            errors.append(np.sqrt(((result - reference) ** 2).mean() / (reference**2).mean()))
            with torch.no_grad():
                distances.append(net(torch.from_numpy(result), torch.from_numpy(reference)).item())
            print(f"{mode:12s} {key:24s} NRMSE {errors[-1]:7.2%}  LPIPS {distances[-1]:.4f}")
        print(
            f"{mode:12s} {'mean of ' + str(len(keys)):24s} NRMSE {np.mean(errors):7.2%}  "
            f"LPIPS {np.mean(distances):.4f}  (median NRMSE {np.median(errors):.2%})"
        )


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, required=True, help="directory of fixtures and results")
    common.add_argument("--checkpoint", default=REPO_ID, help="local directory or Hugging Face repo")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fixtures = commands.add_parser("fixtures", parents=[common])
    fixtures.add_argument("--prompts", default="benchmarks/prompts.json")
    fixtures.add_argument("--split", choices=("dev", "holdout"), default="dev")
    fixtures.add_argument("--ids", nargs="+", required=True)
    fixtures.add_argument("--size", type=int, default=512)
    run = commands.add_parser("run", parents=[common])
    run.add_argument("mode", choices=MODES)
    run.add_argument("--steps", type=int, default=100)
    scoring = commands.add_parser("score", parents=[common])
    scoring.add_argument("modes", nargs="+", choices=MODES[1:])
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    if args.command == "score":
        score(args)
        return
    args.checkpoint = resolve_checkpoint(args.checkpoint, None)
    if args.command == "fixtures":
        make_fixtures(args, load_config(args.checkpoint))
    else:
        denoise(args, load_config(args.checkpoint))


if __name__ == "__main__":
    main()
