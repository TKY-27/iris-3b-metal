"""Generate images with Iris-3B on an Apple Silicon GPU."""

import argparse
import functools
import math
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from PIL import Image

from iris_metal.turbo import TURBO_REVISION, checkpoint_revision, png_info

REPO_ID = "speridlabs/iris-3b"
# the checkpoint revision this engine was validated against
REVISION = "7445443349bc9abe3c96f01ff793e2098ca012b3"
MAX_SEED = 2**63  # the CPU generator's seed range


def resolve_checkpoint(name: str, revision: str | None, model: str = "original") -> Path:
    """A local checkpoint directory, or the text-to-image files of a Hugging Face repo.

    Without an explicit revision, the official repo resolves to the validated one.
    """
    if model not in ("original", "turbo"):
        raise ValueError("model must be original or turbo")
    if Path(name).is_dir():
        root = Path(name)
        if model == "original" and (root / "distill.yaml").exists():
            raise ValueError("this is a Turbo checkpoint; select --model turbo or --turbo-checkpoint")
        selected = root / "turbo" if model == "turbo" and not (root / "distill.yaml").exists() else root
        if model == "turbo" and not (selected / "distill.yaml").is_file():
            raise ValueError("Turbo checkpoint is incomplete: distill.yaml is required")
        return selected
    if revision is None and name == REPO_ID:
        revision = TURBO_REVISION if model == "turbo" else REVISION
    if name == REPO_ID and model == "turbo":
        # Reuse a complete official local-dir download before creating another
        # multi-GB Hub-cache copy. Its recorded immutable revision must match.
        local = Path(__file__).resolve().parents[2] / "iris-3b/turbo"
        required = ("config.yaml", "distill.yaml", "model.safetensors")
        if checkpoint_revision(local) == revision and all(
            (local / filename).is_file() and (local / filename).stat().st_size for filename in required
        ):
            return local
    from huggingface_hub import snapshot_download

    from iris_metal.cache import cached_snapshot

    cached = (
        cached_snapshot(name, revision, turbo=True) if model == "turbo" else cached_snapshot(name, revision)
    )
    if cached is not None:
        return cached

    # the repo also hosts the depth and upscaler fine-tunes (12 GB each)
    patterns = (
        ["turbo/config.yaml", "turbo/distill.yaml", "turbo/model.safetensors"]
        if model == "turbo"
        else ["config.yaml", "model.safetensors"]
    )
    root = Path(snapshot_download(name, revision=revision, allow_patterns=patterns))
    return root / "turbo" if model == "turbo" else root


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


def first_free(name: Callable[[int], Path], taken: set[str], extensions: tuple[str, ...] = ()) -> Path:
    """``name(1)``, ``name(2)``, ...: the first path that is neither chosen earlier in this run nor
    on disk, together with its ``extensions`` siblings. Names compare case-insensitively, as on APFS."""
    n = 1
    while True:
        path = name(n)
        family = [path, *(path.with_suffix(ext) for ext in extensions)]
        # lexists: a dangling symlink also takes the name (exclusive create would fail on it)
        if not any(str(p).casefold() in taken or os.path.lexists(p) for p in family):
            taken.update(str(p).casefold() for p in family)
            return path
        n += 1


def numbered(output: Path) -> Callable[[int], Path]:
    """``output``, then ``<stem>-2<ext>``, ``<stem>-3<ext>``, ..."""
    return lambda n: output if n == 1 else output.with_name(f"{output.stem}-{n}{output.suffix}")


def task_name(out: Path, stem: str, suffix: str) -> Callable[[int], Path]:
    """``<out>/<stem><suffix>.png``, then ``<stem>-2<suffix>.png``, ..."""
    return lambda n: out / f"{stem}{'' if n == 1 else f'-{n}'}{suffix}.png"


def save_new(
    path: Path, write: Callable[[BinaryIO], None], name: Callable[[int], Path], extensions=()
) -> Path:
    """Writes ``path`` only if it does not exist, at the moment of writing; when another run
    took the name meanwhile, the next free one. Returns where the file went."""
    while True:
        try:
            with open(path, "xb") as file:
                write(file)
            return path
        except FileExistsError:
            taken = path
            path = first_free(name, set(), extensions)
            if path == taken:  # cannot happen with lexists; never spin
                raise


def output_paths(output: Path, count: int, overwrite: bool) -> list[Path]:
    """``output`` for one prompt; ``<stem>.png``, ``<stem>-2.png``, ... for several. Unless
    ``overwrite`` (an explicit ``-o``), names already on disk are skipped, so a run never
    replaces an earlier image."""
    name = numbered(output)
    if overwrite:
        return [name(n) for n in range(1, count + 1)]
    taken: set[str] = set()
    return [first_free(name, taken) for _ in range(count)]


def task_outputs(inputs: list[Path], out: Path, suffix: str, extensions: tuple[str, ...] = ()) -> list[Path]:
    """``<out>/<stem><suffix>.png`` per input, numbered ``<stem>-2<suffix>.png``, ... where inputs
    share a stem (``a/photo.jpg``, ``b/photo.jpg``) or an earlier run's output is in the way."""
    taken: set[str] = set()
    return [first_free(task_name(out, path.stem, suffix), taken, extensions) for path in inputs]


def check_readable(path: Path) -> Image.Image:
    """Decodes ``path`` fully, raising ValueError if it cannot; the closed image keeps its size."""
    try:
        with Image.open(path) as image:
            image.load()
            return image
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"not a readable image ({exc})") from exc


# formats Pillow can write an RGB image in
OUTPUT_FORMATS = {"PNG", "JPEG", "WEBP", "TIFF", "BMP"}


def check_output(parser: argparse.ArgumentParser, path: Path) -> None:
    """Fail before generating rather than when saving after minutes of work."""
    if path.is_dir():
        parser.error(f"{path} is a directory; give a file name such as {path / 'iris.png'}")
    if Image.registered_extensions().get(path.suffix.lower()) not in OUTPUT_FORMATS:
        parser.error(f"{path}: give the file a .png, .jpg, .webp, .tif or .bmp extension")


def load_task(task: str, source: str = REPO_ID, revision: str | None = None):
    from iris_metal.tasks import MetalDepth, MetalRestorer, resolve_task

    revision = revision or (REVISION if source == REPO_ID else None)
    export = resolve_task(source, task, revision)
    return MetalRestorer(export) if task == "restoration" else MetalDepth(export)


def serve_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="iris-metal serve", description="Run the local HTTP API.")
    parser.add_argument("--host", default="127.0.0.1", help="default: localhost only")
    parser.add_argument("--port", type=int_range(1, 65536), default=8765)
    parser.add_argument("--queue", type=int_range(1), default=8, help="maximum queued requests")
    parser.add_argument("--checkpoint", default=REPO_ID, help="local directory or Hugging Face repo")
    parser.add_argument("--revision", help="Hugging Face revision of --checkpoint (default: validated)")
    parser.add_argument(
        "--turbo-checkpoint", help="optional local Turbo export or Hugging Face repo; loaded lazily"
    )
    parser.add_argument("--turbo-revision", help="Turbo Hub revision (default: validated official revision)")
    parser.add_argument("--weights", default=REPO_ID, help="depth and upscaling: local export or repo")
    parser.add_argument("--weights-revision", help="Hugging Face revision of --weights (default: validated)")
    args = parser.parse_args(argv)
    import torch

    from iris_metal.pipeline import IrisPipeline
    from iris_metal.server import Models, serve

    if not torch.backends.mps.is_available():
        parser.error("no Apple GPU (MPS) available")
    # Resolve weights lazily on the serial worker.
    defaults = None
    local = Path(args.checkpoint)
    if local.is_dir():
        if (local / "distill.yaml").exists():
            parser.error("--checkpoint must select Original; use --turbo-checkpoint for Turbo")
        if (local / "config.yaml").is_file():
            from iris_metal.loading import load_config

            defaults = load_config(local).sample
    tasks = functools.partial(load_task, source=args.weights, revision=args.weights_revision)

    def load_generation(model):
        if model == "original":
            source, revision = args.checkpoint, args.revision
        else:
            source, revision = args.turbo_checkpoint or args.checkpoint, args.turbo_revision
            if args.turbo_checkpoint is None and not Path(source).is_dir():
                source = REPO_ID
        checkpoint = resolve_checkpoint(source, revision, model=model)
        pipe = IrisPipeline(checkpoint)
        if pipe.model != model:
            raise ValueError(f"checkpoint is {pipe.model}; requested {model}")
        return pipe

    serve(Models(None, tasks, load_generation, defaults=defaults), args.host, args.port, args.queue)


def task_main(task: str, argv: list[str]) -> None:
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
    import numpy as np
    import torch

    from iris_metal.tasks import check_upscale_input, depth_image

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
    targets = task_outputs(
        args.images, args.out, "_x4" if upscale else "_depth", () if upscale else (".npy",)
    )

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
                target = save_new(
                    target,
                    functools.partial(result.save, format="PNG"),
                    task_name(args.out, path.stem, "_x4"),
                )
            else:
                depth = model(image, max_side=args.max_side)
                name = task_name(args.out, path.stem, "_depth")
                target = save_new(
                    target, functools.partial(depth_image(depth).save, format="PNG"), name, (".npy",)
                )
                with open(target.with_suffix(".npy"), "xb") as file:  # never replaces either
                    np.save(file, depth)
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
        epilog="Other commands: iris-metal depth, iris-metal upscale, iris-metal serve (each has --help). "
        "To generate one of those words as a prompt, put -- before it.",
    )
    parser.add_argument("prompts", nargs="+", metavar="PROMPT")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="replaced if it exists (default: iris.png, iris-2.png, ..., never replaced)",
    )
    parser.add_argument(
        "--size", type=parse_size, default=(1024, 1024), help="WIDTHxHEIGHT, default 1024x1024"
    )
    parser.add_argument("--model", choices=("original", "turbo"), default="original")
    parser.add_argument("--steps", type=int_range(1), help="default: the checkpoint's (100)")
    parser.add_argument("--cfg-scale", type=finite_float, help="default: the checkpoint's (3)")
    parser.add_argument("--seed", type=int_range(0, MAX_SEED), default=0)
    parser.add_argument("--negative-prompt", help="default: the checkpoint's (empty)")
    parser.add_argument("--checkpoint", default=REPO_ID, help="local directory or Hugging Face repo")
    parser.add_argument("--revision", help="Hugging Face revision of --checkpoint (default: validated)")
    parser.add_argument(
        "--attention",
        choices=("float16", "bfloat16"),
        default=None,
        help="Original attention precision (default float16); Turbo uses official BF16",
    )
    args = parser.parse_args(argv)
    if args.model == "turbo" and any(
        v is not None for v in (args.steps, args.cfg_scale, args.negative_prompt, args.attention)
    ):
        parser.error("Turbo has fixed steps and embedded guidance; sampling controls do not apply")
    import torch

    from iris_metal.pipeline import IrisPipeline

    if not torch.backends.mps.is_available():
        parser.error("no Apple GPU (MPS) available")
    paths = output_paths(
        args.output or Path("iris.png"), len(args.prompts), overwrite=args.output is not None
    )
    for path in paths:
        check_output(parser, path)

    try:
        pipe = IrisPipeline(
            resolve_checkpoint(args.checkpoint, args.revision, args.model),
            attention_dtype=getattr(torch, args.attention or "float16"),
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if pipe.model != args.model:
        parser.error(f"checkpoint is {pipe.model}; requested {args.model}")
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
        if args.output is not None:
            image.save(path, pnginfo=png_info(image) if path.suffix.lower() == ".png" else None)
        else:  # another run may have taken the name while this one generated
            fmt = Image.registered_extensions()[path.suffix.lower()]
            path = save_new(
                path,
                functools.partial(image.save, format=fmt, pnginfo=png_info(image)),
                numbered(Path("iris.png")),
            )
        print(f"\r{path}: {width}x{height}, {time.perf_counter() - start:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
