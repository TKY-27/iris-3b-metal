"""Batch text-to-image inference.

Usage:
    python scripts/sample.py --checkpoint exported/iris-3b --prompt "a red fox" --prompt "a blue jay"
    python scripts/sample.py --checkpoint output/run/checkpoints/latest.pth --txt-file prompts.txt

``--checkpoint`` is either a directory written by ``scripts/export_checkpoint.py``
(``model.safetensors`` + ``config.yaml``) or a training ``.pth``, whose EMA
weights and embedded config are used. Prompts come from repeated --prompt flags
or a text file (one per line). Images are written as jpgs named after the
prompt plus a grid.png contact sheet.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from iris3b.config import Config, inference_config
from iris3b.models.dit import IrisDiT
from iris3b.registry import TEXT_ENCODERS
from iris3b.sampling import generate, load_for_inference


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "prompt"


def read_prompts(args: argparse.Namespace, cfg: Config) -> list[str]:
    if args.prompt:
        return list(args.prompt)
    if args.txt_file:
        lines = Path(args.txt_file).read_text().splitlines()
        return [line.strip() for line in lines if line.strip()]
    return list(cfg.train.validation_prompts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="exported directory or training .pth")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    parser.add_argument("--prompt", action="append", default=[], help="repeatable prompt")
    parser.add_argument("--txt-file", default=None, help="prompt file, one per line")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=None, help="default: sample.steps (100)")
    parser.add_argument(
        "--order", type=int, default=None, help="DPM-Solver++ order, default: sample.order (2)"
    )
    parser.add_argument("--cfg-scale", type=float, default=None, help="default: sample.cfg_scale (3.0)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument(
        "--shift", type=float, default=None, help="flow shift (default: checkpoint flow.shift)"
    )
    parser.add_argument("--outdir", default="output/samples")
    parser.add_argument("--device", default=None, help="default: cuda when available, else cpu")
    args = parser.parse_args(argv)

    raw, weights = load_for_inference(args.checkpoint)
    cfg = inference_config(raw, args.set)
    steps = args.steps if args.steps is not None else cfg.sample.steps
    order = args.order if args.order is not None else cfg.sample.order
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.sample.cfg_scale
    negative = args.negative_prompt if args.negative_prompt is not None else cfg.sample.negative_prompt
    shift = args.shift if args.shift is not None else cfg.flow.shift
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    prompts = read_prompts(args, cfg)
    if not prompts:
        parser.error("no prompts given (use --prompt or --txt-file)")

    # built on the meta device: the checkpoint tensors become the parameters
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    model.load_state_dict(weights, strict=True, assign=True)
    model = model.eval().to(device=device, dtype=torch.float32)  # FP32 weights, BF16 autocast at inference

    import iris3b.text  # noqa: F401  (registers encoders)

    text_encoder = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        images = generate(
            model,
            text_encoder,
            prompts,
            height=args.height,
            width=args.width,
            steps=steps,
            order=order,
            cfg_scale=cfg_scale,
            cfg_interval=tuple(cfg.sample.cfg_interval),
            shift=shift,
            negative_prompt=negative,
            generator=generator,
            device=device,
            num_train_timesteps=cfg.flow.num_train_timesteps,
            prediction=cfg.flow.prediction,
        )

    from torchvision.utils import save_image

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i, (image, prompt) in enumerate(zip(images, prompts, strict=True)):
        path = outdir / f"{i:03d}_{slugify(prompt)}.jpg"
        save_image(image, str(path), normalize=True, value_range=(-1, 1))
        print(f"wrote {path}")
    grid_path = outdir / "grid.png"
    save_image(
        images, str(grid_path), nrow=max(1, round(len(prompts) ** 0.5)), normalize=True, value_range=(-1, 1)
    )
    print(f"wrote {grid_path}")


if __name__ == "__main__":
    main()
