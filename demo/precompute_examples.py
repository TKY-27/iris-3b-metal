"""Precompute the depth and upscaling examples shown in the demo galleries.

Usage (on a GPU, with the package installed):
    python demo/precompute_examples.py --weights speridlabs/iris-3b

Depth runs on preset images; upscaling runs on degraded copies of preset images
(blur, bicubic downscale to a 256 px short side, JPEG quality 35), which are
written to ``examples/lowres/``. Outputs go to ``examples/depth/<name>.npy``
(float16, long side 640) and ``examples/upscale/<name>.webp``.
"""

import argparse
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from iris3b.downstream.depth import DepthPredictor
from iris3b.downstream.restoration import Restorer, fit_budget

ROOT = Path(__file__).parent

DEPTH_EXAMPLES = ["1", "1048", "1006", "57", "70", "128", "10", "46"]
UPSCALE_EXAMPLES = ["1", "57", "20", "157", "152", "79", "71", "116"]


def degrade(image: Image.Image, short_side: int = 256) -> Image.Image:
    scale = short_side / min(image.size)
    small = image.filter(ImageFilter.GaussianBlur(1.2)).resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC
    )
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=35)
    return Image.open(buf).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", default="speridlabs/iris-3b", help="local release dir or Hub repo id")
    args = parser.parse_args()
    for sub in ("depth", "upscale", "lowres"):
        (ROOT / "examples" / sub).mkdir(parents=True, exist_ok=True)

    depth = DepthPredictor(f"{args.weights}/depth")
    for name in DEPTH_EXAMPLES:
        pred = torch.from_numpy(depth(Image.open(ROOT / f"examples/{name}.webp")))[None, None]
        scale = 640 / max(pred.shape[-2:])
        pred = F.interpolate(pred, scale_factor=scale, mode="bilinear", align_corners=False, antialias=True)
        np.save(ROOT / f"examples/depth/{name}.npy", pred[0, 0].numpy().astype(np.float16))
        print("depth", name, tuple(pred.shape[-2:]))
    del depth
    torch.cuda.empty_cache()

    restorer = Restorer(f"{args.weights}/upscaler")
    for name in UPSCALE_EXAMPLES:
        small = degrade(Image.open(ROOT / f"examples/{name}.webp").convert("RGB"))
        small.save(ROOT / f"examples/lowres/{name}.png")
        result = restorer(fit_budget(small))
        result.save(ROOT / f"examples/upscale/{name}.webp", quality=90)
        print("upscale", name, small.size, "->", result.size)


if __name__ == "__main__":
    main()
