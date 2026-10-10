"""Generate images with Iris-3B on an Apple Silicon GPU."""

import argparse
import functools
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from iris_metal.pipeline import IrisPipeline

REPO_ID = "speridlabs/iris-3b"
# the checkpoint revision this engine was validated against
REVISION = "7445443349bc9abe3c96f01ff793e2098ca012b3"
MAX_SEED = 2**63  # the CPU generator's seed range


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


def int_range(low: int, high: int | None = None):
    """An argparse type for integers in ``[low, high)``."""

    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
        if value < low or (high is not None and value >= high):
            bound = f"in [{low}, {high})" if high is not None else f"at least {low}"
            raise argparse.ArgumentTypeError(f"must be {bound}")
        return value

    return parse


def finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be a finite number")
    return value


def output_paths(output: Path, count: int) -> list[Path]:
    if count == 1:
        return [output]
    return [output.with_name(f"{output.stem}-{i}{output.suffix}") for i in range(count)]


def task_outputs(inputs: list[Path], out: Path, suffix: str) -> list[Path]:
    """``<out>/<stem><suffix>.png`` per input. Inputs that share a stem (``a/photo.jpg``,
    ``b/photo.jpg``, ``photo.png``) get ``-2``, ``-3``, ... in input order instead of
    overwriting each other; names compare case-insensitively, as on APFS."""
    taken: set[str] = set()
    targets = []
    for path in inputs:
        name, n = f"{path.stem}{suffix}", 1
        while name.casefold() in taken:
            n += 1
            name = f"{path.stem}-{n}{suffix}"
        taken.add(name.casefold())
        targets.append(out / f"{name}.png")
    return targets


def check_readable(path: Path) -> Image.Image:
    """Decodes ``path`` fully, raising ValueError if it cannot; the closed image keeps its size."""
    try:
        with Image.open(path) as image:
            image.load()
            return image
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"not a readable image ({exc})") from exc


def check_output(parser: argparse.ArgumentParser, path: Path) -> None:
    """Fail before generating rather than when saving after minutes of work."""
    if path.is_dir():
        parser.error(f"{path} is a directory; give a file name such as {path / 'iris.png'}")
    if path.suffix.lower() not in Image.registered_extensions():
        parser.error(f"{path}: give the file an image extension such as .png")


def load_task(task: str, source: str = REPO_ID, revision: str | None = None):
    from iris_metal.tasks import MetalDepth, MetalRestorer, resolve_task

    revision = revision or (REVISION if source == REPO_ID else None)
    export = resolve_task(source, task, revision)
    return MetalRestorer(export) if task == "restoration" else MetalDepth(export)


def serve_main(argv: list[str]) -> None:
    from iris_metal.server import Models, serve

    parser = argparse.ArgumentParser(prog="iris-metal serve", description="Run the local HTTP API.")
    parser.add_argument("--host", default="127.0.0.1", help="default: localhost only")
    parser.add_argument("--port", type=int_range(1, 65536), default=8765)
    parser.add_argument("--queue", type=int_range(1), default=8, help="maximum queued requests")
    parser.add_argument("--checkpoint", default=REPO_ID, help="local directory or Hugging Face repo")
    parser.add_argument("--revision", help="Hugging Face revision of --checkpoint (default: validated)")
    parser.add_argument("--weights", default=REPO_ID, help="depth and upscaling: local export or repo")
    parser.add_argument("--weights-revision", help="Hugging Face revision of --weights (default: validated)")
    args = parser.parse_args(argv)
    if not torch.backends.mps.is_available():
        parser.error("no Apple GPU (MPS) available")
    pipe = IrisPipeline(resolve_checkpoint(args.checkpoint, args.revision))
    tasks = functools.partial(load_task, source=args.weights, revision=args.weights_revision)
    serve(Models(pipe, tasks), args.host, args.port, args.queue)


def task_main(task: str, argv: list[str]) -> None:
    from iris_metal.tasks import check_upscale_input, depth_image

    upscale = task == "restoration"
    parser = argparse.ArgumentParser(
        prog=f"iris-metal {'upscale' if upscale else 'depth'}",
        description="4x restoration and upscaling" if upscale else "Monocular relative depth",
    )
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("upscaled" if upscale else "depth"))
    parser.add_argument("--weights", default=REPO_ID, help="local task export or Hugging Face repo")
    parser.add_argument("--revision", help="Hugging Face revision of --weights (default: validated)")
    if upscale:
        parser.add_argument("--no-budget", action="store_true", help="do not downscale large inputs first")
        parser.add_argument("--no-color-fix", action="store_true")
    else:
        parser.add_argument(
            "--max-side", type=int_range(0), default=1024, help="long side the model sees; 0 = native"
        )
    args = parser.parse_args(argv)
    if not torch.backends.mps.is_available():
        parser.error("no Apple GPU (MPS) available")
    if args.out.exists() and not args.out.is_dir():
        parser.error(f"--out {args.out} is a file, not a directory")
    for path in args.images:  # before the 12 GB model loads
        try:
            image = check_readable(path)
            if upscale:
                check_upscale_input(image)
        except ValueError as error:
            parser.error(f"{path}: {error}")
    targets = task_outputs(args.images, args.out, "_x4" if upscale else "_depth")

    model = load_task(task, args.weights, args.revision)
    args.out.mkdir(parents=True, exist_ok=True)
    for path, target in zip(args.images, targets, strict=True):
        start = time.perf_counter()
        with Image.open(path) as image:
            if upscale:
                from iris3b.downstream.restoration import fit_budget

                result = model(
                    image if args.no_budget else fit_budget(image), color_fix=not args.no_color_fix
                )
                result.save(target)
            else:
                depth = model(image, max_side=args.max_side)
                np.save(target.with_suffix(".npy"), depth)
                depth_image(depth).save(target)
        print(f"{path} -> {target} ({time.perf_counter() - start:.1f}s)", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["serve"]:
        serve_main(argv[1:])
        return
    if argv[:1] in (["depth"], ["upscale"]):
        task_main("depth" if argv[0] == "depth" else "restoration", argv[1:])
        return
    parser = argparse.ArgumentParser(
        prog="iris-metal",
        description=__doc__,
        epilog="Other commands: iris-metal depth, iris-metal upscale, iris-metal serve (each has --help).",
    )
    parser.add_argument("prompts", nargs="+", metavar="PROMPT")
    parser.add_argument("-o", "--output", type=Path, default=Path("iris.png"), help="default: iris.png")
    parser.add_argument(
        "--size", type=parse_size, default=(1024, 1024), help="WIDTHxHEIGHT, default 1024x1024"
    )
    parser.add_argument("--steps", type=int_range(1), help="default: the checkpoint's (100)")
    parser.add_argument("--cfg-scale", type=finite_float, help="default: the checkpoint's (3)")
    parser.add_argument("--seed", type=int_range(0, MAX_SEED), default=0)
    parser.add_argument("--negative-prompt", help="default: the checkpoint's (empty)")
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
    paths = output_paths(args.output, len(args.prompts))
    for path in paths:
        check_output(parser, path)

    pipe = IrisPipeline(
        resolve_checkpoint(args.checkpoint, args.revision), attention_dtype=getattr(torch, args.attention)
    )
    width, height = args.size
    for prompt, path in zip(args.prompts, paths, strict=True):
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
