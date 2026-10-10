"""Iris draws its own app icon: candidates from Iris-3B, a review sheet, and the app icon.

    python apps/macos/icon/iris_icon.py generate [--server URL] [--only flower --seeds 23 --steps 100]
    python apps/macos/icon/iris_icon.py sheet
    python apps/macos/icon/iris_icon.py install output/icon/raw/flower-23-s100.png

`generate` sends every direction x seed in directions.json to a running
`iris-metal serve` and skips candidates that already exist, so it resumes.
`sheet` masks each candidate into the macOS icon shape and lays them out with
16-128 px previews on light and dark backgrounds. `install` writes the chosen
candidate as the app's Icon Composer document (`Iris/AppIcon.icon`): the system
applies the shape, shadow and appearances, and Xcode derives the icons for
older macOS. With Xcode installed, it also renders previews through `ictool`.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
DIRECTIONS = HERE / "directions.json"
DEFAULT_OUT = HERE.parents[2] / "output" / "icon"
APP_ICON = HERE.parent / "Iris" / "AppIcon.icon"
# Icon Composer's renderer (`xcrun ictool` is a different, asset-catalog tool)
ICTOOL = Path("/Applications/Xcode.app/Contents/Applications/Icon Composer.app/Contents/Executables/ictool")

CANVAS = 1024
BODY = 824  # Apple's macOS icon grid: an 824 px shape centered on a 1024 px canvas
SUPERELLIPSE = 5.0  # close to the continuous-corner macOS icon shape


# --- generate ---------------------------------------------------------------------------------


def request(server: str, method: str, path: str, body: dict | None = None) -> bytes:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"{server}/v1/{path}", data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def generate_one(server: str, params: dict, target: Path) -> None:
    job = json.loads(request(server, "POST", "images", params))["id"]
    while True:
        status = json.loads(request(server, "GET", f"jobs/{job}"))
        print(f"\r{target.name}: {status['status']} {status['step']}/{status['steps']} ", end="", flush=True)
        if status["status"] == "done":
            break
        if status["status"] in ("failed", "cancelled"):
            raise RuntimeError(f"{target.name}: {status['status']} {status.get('error') or ''}")
        time.sleep(2)
    target.write_bytes(request(server, "GET", f"jobs/{job}/image"))
    target.with_suffix(".json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"\r{target.name}: done in {status['seconds']:.0f}s", flush=True)


def generate(args: argparse.Namespace) -> None:
    spec = json.loads(DIRECTIONS.read_text())
    only = set(args.only.split(",")) if args.only else None
    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    width, height = spec["size"]
    for direction in spec["directions"]:
        if only and direction["id"] not in only:
            continue
        for seed in args.seeds or spec["seeds"]:
            suffix = f"-s{args.steps}" if args.steps else ""  # a final render keeps the candidate
            target = raw / f"{direction['id']}-{seed}{suffix}.png"
            if target.exists():
                continue
            params = {
                "prompt": direction["prompt"],
                "negative_prompt": spec["negative_prompt"],
                "width": width,
                "height": height,
                "steps": args.steps or spec["steps"],
                "cfg_scale": spec["cfg_scale"],
                "seed": seed,
            }
            generate_one(args.server, params, target)


# --- shape and previews -----------------------------------------------------------------------


def icon_mask(size: int, body: int) -> Image.Image:
    """An antialiased superellipse of ``body`` px centered on a ``size`` px square."""
    scale = 4  # supersampled, then reduced for the antialiased edge
    big = size * scale
    axis = np.abs((np.arange(big, dtype=np.float32) + 0.5 - big / 2) / (body * scale / 2)) ** SUPERELLIPSE
    inside = (axis[None, :] + axis[:, None]) <= 1
    return Image.fromarray(inside.astype(np.uint8) * 255).resize((size, size), Image.Resampling.LANCZOS)


def masked_icon(artwork: Image.Image, mask: Image.Image) -> Image.Image:
    """The artwork filling the icon shape, with the soft drop shadow of a pre-macOS 26 icon."""
    art = artwork.convert("RGB").resize((BODY, BODY), Image.Resampling.LANCZOS)
    offset = (CANVAS - BODY) // 2
    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 90), (0, 12), mask)
    canvas = Image.alpha_composite(canvas, shadow.filter(ImageFilter.GaussianBlur(14)))
    shaped = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    shaped.paste(art, (offset, offset))
    shaped.putalpha(mask)
    return Image.alpha_composite(canvas, shaped)


def preview_strip(icon: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """The icon at Dock, Finder and menu sizes, as the system would show it."""
    sizes = [128, 64, 32, 16]
    strip = Image.new("RGBA", (sum(sizes) + 24 * (len(sizes) + 1), 128 + 24), background + (255,))
    x = 24
    for size in sizes:
        small = icon.resize((size, size), Image.Resampling.LANCZOS)
        strip.alpha_composite(small, (x, 12 + (128 - size)))
        x += size + 24
    return strip


def sheet(args: argparse.Namespace) -> None:
    raw = sorted((args.out / "raw").glob("*.png"))
    if not raw:
        sys.exit(f"no candidates in {args.out / 'raw'}; run generate first")
    masked_dir = args.out / "masked"
    masked_dir.mkdir(parents=True, exist_ok=True)
    mask = icon_mask(CANVAS, BODY)
    font = ImageFont.load_default(size=22)
    tile_w, tile_h = 560, 300 + 2 * 152 + 40
    columns = 3
    rows = (len(raw) + columns - 1) // columns
    page = Image.new("RGBA", (columns * tile_w, rows * tile_h), (245, 245, 247, 255))
    draw = ImageDraw.Draw(page)
    for index, path in enumerate(raw):
        icon = masked_icon(Image.open(path), mask)
        icon.save(masked_dir / path.name)
        x, y = (index % columns) * tile_w, (index // columns) * tile_h
        page.alpha_composite(icon.resize((280, 280), Image.Resampling.LANCZOS), (x + 140, y + 10))
        page.alpha_composite(preview_strip(icon, (246, 246, 246)), (x + 20, y + 300))
        page.alpha_composite(preview_strip(icon, (30, 30, 32)), (x + 20, y + 452))
        draw.text((x + 24, y + tile_h - 36), path.stem, fill=(40, 40, 40, 255), font=font)
    page.convert("RGB").save(args.out / "sheet.png")
    print(f"{len(raw)} candidates -> {args.out / 'sheet.png'} and {masked_dir}")


# --- install ----------------------------------------------------------------------------------


# the subject a little inside the icon's edge; the feather hides the bitmap's square border
ARTWORK_SCALE = 0.9
FEATHER = 0.1


def feathered(artwork: Image.Image) -> Image.Image:
    """The artwork with its outer border fading out, so it melts into the matching fill."""
    ramp = np.clip(np.minimum(np.arange(CANVAS), np.arange(CANVAS)[::-1]) / (CANVAS * FEATHER), 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)  # smoothstep
    alpha = np.outer(ramp, ramp) * 255
    image = artwork.convert("RGBA")
    image.putalpha(Image.fromarray(alpha.astype(np.uint8)))
    return image


def edge_color(artwork: Image.Image) -> str:
    """The artwork's mean border color as an Icon Composer color, for the fill behind it."""
    pixels = np.asarray(artwork.convert("RGB"), dtype=np.float32) / 255
    border = np.concatenate(
        [
            pixels[:8].reshape(-1, 3),
            pixels[-8:].reshape(-1, 3),
            pixels[:, :8].reshape(-1, 3),
            pixels[:, -8:].reshape(-1, 3),
        ]
    )
    r, g, b = border.mean(axis=0)
    return f"srgb:{r:.5f},{g:.5f},{b:.5f},1.00000"


def install(args: argparse.Namespace) -> None:
    artwork = Image.open(args.candidate).convert("RGB").resize((CANVAS, CANVAS), Image.Resampling.LANCZOS)
    assets = args.dest / "Assets"
    assets.mkdir(parents=True, exist_ok=True)
    for old in assets.iterdir():
        old.unlink()
    feathered(artwork).save(assets / "artwork.png", optimize=True)
    # one full-bleed bitmap layer; the system masks it and adds its own edge light and shadow
    manifest = {
        # the same fill in dark appearance, where the system would otherwise darken it around the bitmap
        "fill-specializations": [
            {"value": {"solid": edge_color(artwork)}},
            {"appearance": "dark", "value": {"solid": edge_color(artwork)}},
        ],
        "groups": [
            {
                "layers": [
                    {
                        "glass": False,
                        "image-name": "artwork.png",
                        "name": "artwork",
                        "position": {"scale": ARTWORK_SCALE, "translation-in-points": [0, 0]},
                    }
                ],
                "shadow": {"kind": "neutral", "opacity": 0.5},
                "translucency": {"enabled": False, "value": 0.5},
            }
        ],
        "supported-platforms": {"squares": "shared"},
    }
    (args.dest / "icon.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.candidate} -> {args.dest}")
    for rendition in ("Default", "Dark"):
        preview = args.preview / f"app-icon-{rendition.lower()}.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        command = [str(ICTOOL), str(args.dest), "--export-image", "--output-file", str(preview)]
        command += ["--platform", "macOS", "--rendition", rendition, "--width", "1024", "--height", "1024"]
        if (
            ICTOOL.exists()
            and subprocess.run(command + ["--scale", "1"], capture_output=True).returncode == 0
        ):
            print(f"preview: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="draw candidates with a running iris-metal serve")
    gen.add_argument("--server", default="http://127.0.0.1:8765")
    gen.add_argument("--only", help="comma-separated direction ids")
    gen.add_argument("--seeds", type=int, nargs="+", help="override directions.json")
    gen.add_argument("--steps", type=int, help="override directions.json, e.g. 100 for a final render")
    gen.add_argument("--out", type=Path, default=DEFAULT_OUT)
    lay = sub.add_parser("sheet", help="mask candidates and build the review sheet")
    lay.add_argument("--out", type=Path, default=DEFAULT_OUT)
    put = sub.add_parser("install", help="make a chosen candidate the app's icon")
    put.add_argument("candidate", type=Path)
    put.add_argument("--dest", type=Path, default=APP_ICON)
    put.add_argument("--preview", type=Path, default=DEFAULT_OUT, help="where ictool previews go")
    args = parser.parse_args()
    {"generate": generate, "sheet": sheet, "install": install}[args.command](args)


if __name__ == "__main__":
    main()
