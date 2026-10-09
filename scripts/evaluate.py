"""Generate the benchmark prompt set with one sampler configuration.

Usage:
    python scripts/evaluate.py --split dev --out runs/reference
    python scripts/evaluate.py --split dev --out runs/steps50 --steps 50
    python scripts/compare_runs.py runs/reference runs/steps50

Images are written as ``<out>/<prompt id>.png`` with a ``meta.json`` holding
the configuration, per-image sampling time and the environment. Existing
images are kept, so an interrupted run resumes where it stopped. Keep the
``holdout`` split for confirming a decision made on ``dev``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from iris_metal.pipeline import IrisPipeline

PROMPTS = Path(__file__).resolve().parents[1] / "benchmarks" / "prompts.json"


def scaled(size: list[int], scale: float, patch: int = 16) -> tuple[int, int]:
    width, height = (max(patch, round(side * scale / patch) * patch) for side in size)
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default="iris-3b")
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ids", nargs="*", help="subset of prompt ids")
    parser.add_argument("--scale", type=float, default=1.0, help="multiplier on the listed resolutions")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention", choices=("float16", "bfloat16", "float32"), default="float16")
    args = parser.parse_args()

    entries = json.loads(PROMPTS.read_text())[args.split]
    if args.ids:
        entries = [e for e in entries if e["id"] in args.ids]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    config = {key: value for key, value in vars(args).items() if key != "ids"}
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"config": config, "images": {}}
    if meta["config"] != config:
        raise SystemExit(f"{out} holds a run with a different configuration: {meta['config']}")

    pipe = IrisPipeline(args.checkpoint, attention_dtype=getattr(torch, args.attention))
    for entry in entries:
        if (out / f"{entry['id']}.png").exists():
            continue
        # encode and load before the clock starts: "seconds" is sampling time
        pipe.encode([entry["prompt"], pipe.cfg.sample.negative_prompt])
        pipe.engine  # noqa: B018
        width, height = scaled(entry["size"], args.scale)
        began = time.perf_counter()
        image = pipe(
            [entry["prompt"]],
            height=height,
            width=width,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
        )[0]
        seconds = time.perf_counter() - began
        image.save(out / f"{entry['id']}.png")
        meta["images"][entry["id"]] = {"size": [width, height], "seconds": round(seconds, 2)}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{entry['id']}: {width}x{height} in {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
