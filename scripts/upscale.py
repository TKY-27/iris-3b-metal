"""Image restoration and upscaling with Iris-3B.

Usage:
    python scripts/upscale.py photo.jpg more/*.png --out upscaled
    python scripts/upscale.py photo.jpg --no-budget --weights exported/iris-3b/upscaler

The restorer is one forward pass per 1024x1024 tile; larger outputs are tiled
with 50% overlap (see ``iris3b/downstream/restoration.py``). By default inputs
are first downscaled to at most 512 px on the short side and 1024 px on the
long side, so an output is at most 2048x4096 (21 tiles); ``--no-budget`` keeps
the input size, and the run time then grows with the output area.
``--weights`` is a local export or a Hub folder (default
``speridlabs/iris-3b/upscaler``).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from iris3b.downstream.restoration import Restorer, fit_budget


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("images", nargs="+")
    parser.add_argument("--out", default="upscaled")
    parser.add_argument("--weights", default="speridlabs/iris-3b/upscaler")
    parser.add_argument("--no-budget", action="store_true", help="do not downscale large inputs first")
    parser.add_argument("--no-color-fix", action="store_true")
    args = parser.parse_args(argv)

    restorer = Restorer(args.weights)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.images:
        image = Image.open(name)
        if not args.no_budget:
            image = fit_budget(image)
        result = restorer(image, color_fix=not args.no_color_fix)
        path = out / f"{Path(name).stem}_x4.png"
        result.save(path)
        print(f"{name} {image.size[0]}x{image.size[1]} -> {path} {result.size[0]}x{result.size[1]}")


if __name__ == "__main__":
    main()
