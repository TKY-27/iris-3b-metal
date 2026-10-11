"""Does the 4x restorer also restore an image at its own size?

The upscaler fine-tune was trained on 4x of 256-pixel inputs. This compares
three ways of using it to repair a degraded image without enlarging it, on
reference images degraded synthetically:

- ``direct``: one pass at scale 1 (the restorer's ``scale`` argument);
- ``via4x``: downscale 4x, restore at 4x (exactly the trained setting);
- ``via2x``: downscale 2x, restore at 4x, downscale 2x.

Each is scored against the clean reference (PSNR, LPIPS) next to the degraded
input itself, and center crops are saved side by side for viewing.

Usage:
    python scripts/restore_study.py runs/holdout-v2/*.png --out runs/restore-study
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from PIL import Image, ImageFilter

DEGRADATIONS = {
    "blur": lambda im: im.filter(ImageFilter.GaussianBlur(2.0)),
    "jpeg": lambda im: jpeg(im, 15),
    "blur-jpeg": lambda im: jpeg(im.filter(ImageFilter.GaussianBlur(1.2)), 30),
    "lowres": lambda im: im.resize((im.width // 2, im.height // 2), Image.Resampling.BICUBIC).resize(
        im.size, Image.Resampling.BICUBIC
    ),
}


def jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return Image.open(buffer).convert("RGB")


def shrink(image: Image.Image, factor: int) -> Image.Image:
    return image.resize((image.width // factor, image.height // factor), Image.Resampling.LANCZOS)


def methods(restorer):
    return {
        "direct": lambda im: restorer(im, scale=1.0),
        "via4x": lambda im: restorer(shrink(im, 4)).resize(im.size, Image.Resampling.LANCZOS),
        "via2x": lambda im: restorer(shrink(im, 2)).resize(im.size, Image.Resampling.LANCZOS),
    }


def tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None]


def psnr(a: Image.Image, b: Image.Image) -> float:
    mse = np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2)
    return float(10 * np.log10(255**2 / max(mse, 1e-12)))


def crop(image: Image.Image, size: int = 384) -> Image.Image:
    left, top = (image.width - size) // 2, (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/restore-study"))
    args = parser.parse_args(argv)

    import lpips

    from iris_metal.cli import load_task

    args.out.mkdir(parents=True, exist_ok=True)
    restorer = load_task("restoration")
    ways = methods(restorer)
    metric = lpips.LPIPS(net="alex", verbose=False)
    rows = []
    for path in args.images:
        clean = Image.open(path).convert("RGB")
        for name, degrade in DEGRADATIONS.items():
            degraded = degrade(clean)
            outputs = {"input": degraded}
            for way, run in ways.items():
                start = time.perf_counter()
                outputs[way] = run(degraded)
                print(f"{path.stem} {name} {way} {time.perf_counter() - start:.1f} s", flush=True)
            for way, image in outputs.items():
                with torch.no_grad():
                    distance = metric(tensor(image), tensor(clean)).item()
                rows.append(
                    {
                        "image": path.stem,
                        "degradation": name,
                        "method": way,
                        "psnr": psnr(image, clean),
                        "lpips": distance,
                    }
                )
            strip = [crop(clean)] + [crop(outputs[k]) for k in ("input", *ways)]
            sheet = Image.new("RGB", (384 * len(strip), 384))
            for i, tile in enumerate(strip):
                sheet.paste(tile, (384 * i, 0))
            sheet.save(args.out / f"{path.stem}-{name}.png")  # clean | input | direct | via4x | via2x

    (args.out / "results.json").write_text(json.dumps(rows, indent=1))
    print(f"\n{'degradation':<10} {'method':<7} {'PSNR':>6} {'LPIPS':>6}  better LPIPS than input")
    for name in DEGRADATIONS:
        base = {r["image"]: r["lpips"] for r in rows if r["degradation"] == name and r["method"] == "input"}
        for way in ("input", *ways):
            picked = [r for r in rows if r["degradation"] == name and r["method"] == way]
            wins = sum(r["lpips"] < base[r["image"]] for r in picked)
            print(
                f"{name:<10} {way:<7} {np.mean([r['psnr'] for r in picked]):6.2f} "
                f"{np.mean([r['lpips'] for r in picked]):6.3f}  {wins}/{len(picked)}"
            )


if __name__ == "__main__":
    main()
