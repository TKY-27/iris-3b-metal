"""Five AB/BA warm Turbo storage pairs, each in a fresh serial process.

Raw conditioning and noise are fixed. Each process warms its loaded model
once, then measures one complete sampling request with StepReporter. Model
load, storage conversion, warmup and result readback are outside that timer.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark import _INTERVAL_MAX_FOOTPRINT, paging_counters, peak_footprint_gib, swap_used_mb
from iris3b.sampling import generate_few_step
from iris_metal.loading import load_config, load_dit
from iris_metal.pipeline import StepReporter
from iris_metal.turbo import TURBO_SIGMAS


def child(args):
    data = torch.load(args.fixture, map_location="cpu", weights_only=True)
    cfg = load_config(args.checkpoint)
    dtype = torch.float32 if args.mode == "reference" else torch.bfloat16
    model = load_dit(args.checkpoint, cfg, torch.device("mps"), dtype)

    class Encoder:
        def encode(self, prompts):
            return SimpleNamespace(embeddings=data["states"], mask=data["mask"])

    class Model:
        cfg = model.cfg

        def parameters(self):
            return model.parameters()

        def __call__(self, *args, **kwargs):
            out = model(*args, **kwargs)
            reporter.queued()
            return out

    def run():
        nonlocal reporter
        reporter = StepReporter(4, None, torch.device("mps"))
        with torch.no_grad(), torch.autocast("mps", dtype=torch.bfloat16):
            result = generate_few_step(
                Model(),
                Encoder(),
                [data["prompt"]],
                *data["noise"].shape[-2:],
                list(TURBO_SIGMAS),
                noise=data["noise"],
                device="mps",
                num_train_timesteps=cfg.flow.num_train_timesteps,
                prediction=cfg.flow.prediction,
            )
        reporter.finish()
        return result

    reporter = None
    warmup = run().cpu()
    peak_footprint_gib(_INTERVAL_MAX_FOOTPRINT)
    before = paging_counters()
    swap_before = swap_used_mb()
    start = time.perf_counter()
    result = run()
    torch.mps.synchronize()
    seconds = time.perf_counter() - start
    peak = peak_footprint_gib(_INTERVAL_MAX_FOOTPRINT)
    final = result.cpu()
    baseline = torch.load(args.fixture.parent / "reference.pt", map_location="cpu", weights_only=True)[
        "final"
    ]
    row = {
        "mode": args.mode,
        "seconds": seconds,
        "bitwise_equal": torch.equal(final, baseline),
        "repeat_bitwise_equal": torch.equal(final, warmup),
        "physical_peak_gib": peak,
        "allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_bytes": torch.mps.driver_allocated_memory(),
        "paging_before": before,
        "paging_after": paging_counters(),
        "swap_before_mb": swap_before,
        "swap_after_mb": swap_used_mb(),
    }
    args.out.write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)
    assert row["bitwise_equal"] and row["repeat_bitwise_equal"], "parity failed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("iris-3b/turbo"))
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--mode", choices=("reference", "bf16"))
    args = parser.parse_args()
    if args.mode:
        child(args)
        return
    rows = []
    for pair in range(args.pairs):
        order = ("reference", "bf16") if pair % 2 == 0 else ("bf16", "reference")
        for mode in order:
            target = args.out.with_name(f"{args.out.stem}-{pair}-{mode}.json")
            env = dict(os.environ, HF_HUB_OFFLINE="1")
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--fixture",
                    str(args.fixture),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--out",
                    str(target),
                    "--mode",
                    mode,
                ],
                check=True,
                env=env,
            )
            row = json.loads(target.read_text())
            row["pair"] = pair
            rows.append(row)
            args.out.write_text(
                json.dumps(
                    {
                        "scope": "warm sampler+pipeline StepReporter; fixed raw text/noise; fresh serial process per mode",
                        "rows": rows,
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
