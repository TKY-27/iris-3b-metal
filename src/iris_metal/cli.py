"""Generate images with Iris-3B on an Apple Silicon GPU."""

import argparse
import re
import sys
import time
from pathlib import Path

import torch

from iris_metal.pipeline import IrisPipeline

REPO_ID = "speridlabs/iris-3b"
# the checkpoint revision this engine was validated against
REVISION = "7445443349bc9abe3c96f01ff793e2098ca012b3"


def resolve_checkpoint(name: str, revision: str | None) -> Path:
    """A local checkpoint directory, or the text-to-image files of a Hugging Face repo.

    Without an explicit revision, the official repo resolves to the validated one.
    """
    if Path(name).is_dir():
        return Path(name)
    if revision is None and name == REPO_ID:
        revision = REVISION
    from huggingface_hub import snapshot_download

    # the repo also hosts the depth and upscaler fine-tunes (12 GB each)
    return Path(
        snapshot_download(name, revision=revision, allow_patterns=["config.yaml", "model.safetensors"])
    )


def parse_size(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", text)
    if not match:
        raise argparse.ArgumentTypeError("size must look like 1024x1024 (width x height)")
    width, height = int(match[1]), int(match[2])
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise argparse.ArgumentTypeError("width and height must be positive multiples of 16")
    return width, height


def output_paths(output: Path, count: int) -> list[Path]:
    if count == 1:
        return [output]
    return [output.with_name(f"{output.stem}-{i}{output.suffix}") for i in range(count)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="iris-metal", description=__doc__)
    parser.add_argument("prompts", nargs="+", metavar="PROMPT")
    parser.add_argument("-o", "--output", type=Path, default=Path("iris.png"), help="default: iris.png")
    parser.add_argument(
        "--size", type=parse_size, default=(1024, 1024), help="WIDTHxHEIGHT, default 1024x1024"
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--checkpoint", default=REPO_ID, help="local directory or Hugging Face repo")
    parser.add_argument("--revision", help="Hugging Face revision of --checkpoint (default: validated)")
    parser.add_argument(
        "--attention",
        choices=("float16", "bfloat16"),
        default="float16",
        help="attention input precision; bfloat16 is what the upstream CUDA recipe uses",
    )
    args = parser.parse_args(argv)
    if not torch.backends.mps.is_available():
        parser.error("no Apple GPU (MPS) available")

    pipe = IrisPipeline(
        resolve_checkpoint(args.checkpoint, args.revision), attention_dtype=getattr(torch, args.attention)
    )
    width, height = args.size
    for prompt, path in zip(args.prompts, output_paths(args.output, len(args.prompts)), strict=True):
        start = time.perf_counter()
        step_one = 0.0  # models load before the first step, so the rate is timed from there

        def progress(step: int, total: int, path: Path = path) -> None:
            nonlocal step_one
            now = time.perf_counter()
            if step == 1:
                step_one, eta = now, ""
            else:
                eta = f", {(now - step_one) / (step - 1) * (total - step):.0f}s left"
            print(f"\r{path.name}: step {step}/{total}{eta} ", end="", file=sys.stderr, flush=True)

        image = pipe(
            [prompt],
            height=height,
            width=width,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
            on_step=progress,
        )[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        print(f"\r{path}: {width}x{height}, {time.perf_counter() - start:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
