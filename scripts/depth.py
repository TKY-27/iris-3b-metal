"""Monocular depth with Iris-3B.

Usage:
    python scripts/depth.py photo.jpg more/*.png --out depth_out
    python scripts/depth.py photo.jpg --weights exported/iris-3b/depth --max-side 0

For each image writes ``<name>.npy`` (relative log depth at the input
resolution, -1 near .. +1 far; affine-invariant, not metric) and
``<name>.png`` (colorized, near = bright). ``--weights`` is a local export or a
Hub folder (default ``speridlabs/iris-3b/depth``). ``--max-side`` caps the long
side the model sees (default 1024, 0 = native resolution).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from iris3b.downstream.depth import DepthPredictor, colorize


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("images", nargs="+")
    parser.add_argument("--out", default="depth_out")
    parser.add_argument("--weights", default="speridlabs/iris-3b/depth")
    parser.add_argument("--max-side", type=int, default=1024)
    args = parser.parse_args(argv)

    predictor = DepthPredictor(args.weights)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.images:
        depth = predictor(Image.open(name), max_side=args.max_side)
        stem = out / Path(name).stem
        np.save(stem.with_suffix(".npy"), depth)
        colorize(depth).save(stem.with_suffix(".png"))
        print(f"{name} -> {stem}.png/.npy  {depth.shape[1]}x{depth.shape[0]}")


if __name__ == "__main__":
    main()
