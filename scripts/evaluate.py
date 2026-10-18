"""Generate the benchmark prompt set with one sampler configuration.

Usage:
    python scripts/evaluate.py --split dev --out runs/reference
    python scripts/evaluate.py --split dev --out runs/steps50 --steps 50
    python scripts/compare_runs.py runs/reference runs/steps50

Images resume only with matching recorded provenance.
Keep ``holdout`` for confirming ``dev`` decisions.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image

import iris3b
import iris_metal
from iris_metal.cli import REPO_ID, resolve_checkpoint
from iris_metal.pipeline import IrisPipeline

PROMPTS = Path(__file__).resolve().parents[1] / "benchmarks" / "prompts.json"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scaled(size: list[int], scale: float, patch: int = 16) -> tuple[int, int]:
    width, height = (max(patch, round(side * scale / patch) * patch) for side in size)
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=REPO_ID, help="local directory or Hugging Face repo")
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ids", nargs="*", help="subset of prompt ids")
    parser.add_argument("--scale", type=float, default=1.0, help="multiplier on the listed resolutions")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention", choices=("float16", "bfloat16", "float32"), default="float16")
    args = parser.parse_args()

    all_entries = json.loads(PROMPTS.read_text())[args.split]
    expected = {
        entry["id"]: {"prompt": entry["prompt"], "size": list(scaled(entry["size"], args.scale))}
        for entry in all_entries
    }
    if len(expected) != len(all_entries):
        raise SystemExit("prompt ids must be unique")
    if args.ids and set(args.ids) - expected.keys():
        raise SystemExit(f"unknown prompt ids: {sorted(set(args.ids) - expected.keys())}")
    entries = [e for e in all_entries if not args.ids or e["id"] in args.ids]
    checkpoint = resolve_checkpoint(args.checkpoint, None)
    sources = [Path(__file__).resolve()]
    for package in (iris3b, iris_metal):
        sources.extend(sorted(Path(package.__file__).resolve().parent.rglob("*.py")))
    provenance = {
        "checkpoint": {
            "config_sha256": sha256(checkpoint / "config.yaml"),
            "weights_sha256": sha256(checkpoint / "model.safetensors"),
        },
        "prompts_sha256": sha256(PROMPTS),
        "source_sha256": {
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): sha256(path)
            for path in sources
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    config = {key: value for key, value in vars(args).items() if key != "ids"}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if any(
            meta.get(key) != value
            for key, value in (("config", config), ("provenance", provenance), ("expected", expected))
        ):
            raise SystemExit(f"{out} holds a run with different configuration or provenance")
    else:
        if list(out.glob("*.png")):
            raise SystemExit(f"{out} has images without meta.json; use a new output directory")
        meta = {"config": config, "provenance": provenance, "expected": expected, "images": {}}
    pngs = {path.stem for path in out.glob("*.png")}
    recorded = set(meta["images"])
    if pngs != recorded or recorded - expected.keys():
        raise SystemExit(f"{out} has incomplete or unexpected saved images; use a new output directory")
    for image_id, record in meta["images"].items():
        if (
            record.get("prompt") != expected[image_id]["prompt"]
            or record.get("size") != expected[image_id]["size"]
        ):
            raise SystemExit(f"{out} has mismatched metadata for {image_id}")
        with Image.open(out / f"{image_id}.png") as saved:
            if list(saved.size) != record["size"]:
                raise SystemExit(f"{out} has mismatched PNG dimensions for {image_id}")

    pipe = IrisPipeline(checkpoint, attention_dtype=getattr(torch, args.attention))
    for entry in entries:
        # meta.json is written after the image, so it marks a finished one
        if entry["id"] in meta["images"] and (out / f"{entry['id']}.png").exists():
            continue
        # encode and load before the clock starts: "seconds" is sampling time
        pipe.encode([entry["prompt"], pipe.cfg.sample.negative_prompt])
        pipe.engine  # noqa: B018
        width, height = expected[entry["id"]]["size"]
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
        if image.size != (width, height):
            raise ValueError(f"{entry['id']}: generated {image.size}, expected {(width, height)}")
        image.save(out / f"{entry['id']}.png")
        meta["images"][entry["id"]] = {
            "prompt": entry["prompt"],
            "size": list(image.size),
            "seconds": round(seconds, 2),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{entry['id']}: {width}x{height} in {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
