"""Compare evaluation runs against a reference run of the same prompts.

Usage:
    python scripts/compare_runs.py runs/reference runs/candidate [...] --sheet sheet.png

Per run: fidelity to the reference image of each prompt (PSNR, SSIM on luma,
LPIPS-Alex when the ``lpips`` package is installed), a patch-grid score and
the mean sampling time. Fidelity only says how closely a cheaper sampler
reproduces the reference; whether a difference is better or worse needs the
contact sheet, which shows each image and a 2x crop of its center.

Grid score: mean absolute luma step across 16-pixel patch boundaries divided
by the mean step elsewhere, averaged over both axes. 1.0 means the patch grid
leaves no trace in the gradients; seams push it above 1. Natural images
score close to 1, so it diagnoses patch artifacts but does not rate quality.
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PATCH = 16


def luma(rgb: torch.Tensor) -> torch.Tensor:
    """``[3, H, W]`` in [0, 1] -> ``[1, 1, H, W]`` BT.601 luma."""
    weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    return (rgb * weights).sum(0)[None, None]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * np.log10(1 / mse)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Gaussian-window SSIM (11x11, sigma 1.5) on luma."""
    x, y = luma(a), luma(b)
    coords = torch.arange(11) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    window = (g[:, None] * g[None, :] / g.sum() ** 2)[None, None]

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, window)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x**2
    var_y = blur(y * y) - mu_y**2
    cov = blur(x * y) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return score.mean().item()


def grid_score(rgb: torch.Tensor) -> float:
    y = luma(rgb)[0, 0]
    scores = []
    for steps in ((y[:, 1:] - y[:, :-1]).abs(), (y[1:] - y[:-1]).abs().T):
        crossing = torch.arange(1, steps.shape[1] + 1) % PATCH == 0
        scores.append((steps[:, crossing].mean() / steps[:, ~crossing].mean()).item())
    return sum(scores) / 2


def load(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).float() / 255


def contact_sheet(runs: list[Path], ids: list[str], path: Path, cell: int = 384) -> None:
    sheet = Image.new("RGB", (cell * 2 * len(runs), cell * len(ids)), "white")
    for row, image_id in enumerate(ids):
        for col, run in enumerate(runs):
            image = Image.open(run / f"{image_id}.png").convert("RGB")
            thumb = image.copy()
            thumb.thumbnail((cell, cell))
            sheet.paste(thumb, (2 * col * cell, row * cell))
            w, h = image.size
            crop = image.crop(
                (w // 2 - cell // 4, h // 2 - cell // 4, w // 2 + cell // 4, h // 2 + cell // 4)
            )
            sheet.paste(crop.resize((cell, cell), Image.NEAREST), ((2 * col + 1) * cell, row * cell))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--sheet", type=Path, help="write a contact sheet PNG")
    parser.add_argument("--json", type=Path, help="write per-image metrics")
    args = parser.parse_args()

    try:
        import lpips

        lpips_model = lpips.LPIPS(net="alex", verbose=False)
    except ImportError:
        lpips_model = None

    ids = sorted(p.stem for p in args.reference.glob("*.png"))
    report = {}
    for run in [args.reference, *args.candidates]:
        meta = json.loads((run / "meta.json").read_text())
        rows = {}
        for image_id in ids:
            candidate = load(run / f"{image_id}.png")
            reference = load(args.reference / f"{image_id}.png")
            row = {"grid": grid_score(candidate), "seconds": meta["images"][image_id]["seconds"]}
            if run != args.reference:
                row.update(psnr=psnr(candidate, reference), ssim=ssim(candidate, reference))
                if lpips_model is not None:
                    with torch.no_grad():
                        row["lpips"] = lpips_model(candidate[None] * 2 - 1, reference[None] * 2 - 1).item()
            rows[image_id] = row
        report[str(run)] = rows

        means = {key: statistics.fmean(r[key] for r in rows.values()) for key in next(iter(rows.values()))}
        summary = "  ".join(f"{key} {value:.4f}" for key, value in means.items())
        print(f"{run}: {summary}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    if args.sheet:
        contact_sheet([args.reference, *args.candidates], ids, args.sheet)


if __name__ == "__main__":
    main()
