"""Compare evaluation runs against a reference run of the same prompts.

Usage:
    python scripts/compare_runs.py runs/reference runs/candidate [...] --sheet sheet.png

Whole runs require provenance; subsets are exploratory.

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
VARY = ("steps", "attention", "cfg_scale")


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
        if not crossing.any():
            continue
        boundary = steps[:, crossing].mean().item()
        interior = steps[:, ~crossing].mean().item()
        scores.append(boundary / max(interior, 1e-8) if boundary else (0.0 if interior else 1.0))
    return sum(scores) / len(scores) if scores else 1.0


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


def finished(run: Path) -> tuple[dict, set[str], set[str]]:
    """``meta.json``, the ids of finished images and ids with an image or a record only."""
    meta = json.loads((run / "meta.json").read_text())
    pngs, recorded = {p.stem for p in run.glob("*.png")}, set(meta["images"])
    return meta, pngs & recorded, pngs ^ recorded


def mismatches(reference: Path, run: Path, vary: set[str] | None = None) -> list[str]:
    """Reasons two whole runs cannot be compared."""
    if vary and not vary <= set(VARY):
        raise ValueError(f"unsupported experiment variables: {sorted(vary - set(VARY))}")
    (ref_meta, ref_ids, ref_unpaired), (meta, ids, run_unpaired) = finished(reference), finished(run)
    problems = []
    for name, current, complete, unpaired in (
        (reference, ref_meta, ref_ids, ref_unpaired),
        (run, meta, ids, run_unpaired),
    ):
        identity = current.get("provenance")
        checkpoint = identity.get("checkpoint") if isinstance(identity, dict) else None
        if not (
            isinstance(checkpoint, dict)
            and checkpoint.get("config_sha256")
            and checkpoint.get("weights_sha256")
            and identity.get("prompts_sha256")
            and isinstance(identity.get("source_sha256"), dict)
            and identity["source_sha256"]
        ):
            problems.append(f"{name}: missing checkpoint, prompt or source provenance")
        expected = current.get("expected")
        if not isinstance(expected, dict) or not expected:
            problems.append(f"{name}: missing expected prompt set")
        else:
            if expected.keys() - complete:
                problems.append(f"{name}: incomplete prompt set {sorted(expected.keys() - complete)}")
            if complete - expected.keys():
                problems.append(f"{name}: unexpected prompt ids {sorted(complete - expected.keys())}")
        if unpaired:
            problems.append(f"{name}: image or meta.json entry alone for {sorted(unpaired)}")
        for image_id in sorted(complete):
            record = current["images"][image_id]
            if (
                isinstance(expected, dict)
                and image_id in expected
                and (
                    record.get("prompt") != expected[image_id].get("prompt")
                    or record.get("size") != expected[image_id].get("size")
                )
            ):
                problems.append(f"{name}: {image_id} prompt or declared size differs")
            try:
                with Image.open(name / f"{image_id}.png") as image:
                    actual_size = list(image.size)
                if actual_size != record.get("size"):
                    problems.append(f"{name}: {image_id} PNG dimensions differ from metadata")
            except (OSError, ValueError):
                problems.append(f"{name}: {image_id} PNG cannot be read")
    if ref_ids - ids:
        problems.append(f"missing {sorted(ref_ids - ids)}")
    if ids - ref_ids:
        problems.append(f"unexpected {sorted(ids - ref_ids)}")
    ref_identity, identity = ref_meta.get("provenance"), meta.get("provenance")
    if isinstance(ref_identity, dict) and isinstance(identity, dict):
        for key in ("checkpoint", "prompts_sha256"):
            if ref_identity.get(key) != identity.get(key):
                problems.append(f"{key} provenance differs")
    if ref_meta.get("expected") != meta.get("expected"):
        problems.append("expected prompt sets differ")
    ref_config, config = ref_meta["config"], meta["config"]
    for key in sorted((ref_config.keys() | config.keys()) - {"out"} - (vary or set())):
        if key not in ref_config or key not in config or ref_config[key] != config[key]:
            problems.append(f"{key} {config.get(key)!r} != {ref_config.get(key)!r}")
    for image_id in sorted(ref_ids & ids):
        if ref_meta["images"][image_id]["size"] != meta["images"][image_id]["size"]:
            problems.append(f"{image_id} size differs")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--sheet", type=Path, help="write a contact sheet PNG")
    parser.add_argument("--json", type=Path, help="write per-image metrics")
    parser.add_argument("--subset", action="store_true", help="exploratory: compare shared prompts only")
    parser.add_argument(
        "--vary", choices=VARY, action="append", default=[], help="declared experiment variable"
    )
    args = parser.parse_args()

    try:
        import lpips

        lpips_model = lpips.LPIPS(net="alex", verbose=False)
    except ImportError:
        lpips_model = None

    runs = [args.reference, *args.candidates]
    problems = {str(run): mismatches(args.reference, run, set(args.vary)) for run in runs}
    if args.subset:
        # prompts every run has, so the means of all runs cover the same images
        ids = sorted(set.intersection(*(finished(run)[1] for run in runs)))
        print("EXPLORATORY subset comparison")
    elif any(problems.values()):
        raise SystemExit(
            "runs are not comparable as a whole:\n"
            + "\n".join(f"{run}: {'; '.join(found)}" for run, found in problems.items() if found)
        )
    else:
        ids = sorted(finished(args.reference)[1])
    if not ids:
        raise SystemExit("the runs have no prompt in common")
    metadata = {str(run): json.loads((run / "meta.json").read_text()) for run in runs}
    for image_id in ids:
        sizes = []
        for run in runs:
            with Image.open(run / f"{image_id}.png") as image:
                sizes.append(image.size)
                if list(image.size) != metadata[str(run)]["images"][image_id].get("size"):
                    raise SystemExit(f"{run}: {image_id} PNG dimensions differ from metadata")
        if len(set(sizes)) != 1:
            raise SystemExit(f"{image_id}: actual PNG dimensions differ")
    print(f"{len(ids)} prompts: {', '.join(ids)}")
    report = {
        "complete_comparison": not args.subset,
        "prompts": ids,
        "mismatches": problems,
        "variants": {
            key: {str(run): metadata[str(run)]["config"].get(key) for run in runs} for key in args.vary
        },
        "provenance": {str(run): metadata[str(run)].get("provenance") for run in runs},
        "runs": {},
    }
    for run in runs:
        meta = metadata[str(run)]
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
        report["runs"][str(run)] = rows

        means = {key: statistics.fmean(r[key] for r in rows.values()) for key in next(iter(rows.values()))}
        summary = "  ".join(f"{key} {value:.4f}" for key, value in means.items())
        print(f"{run}: {summary}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    if args.sheet:
        contact_sheet(runs, ids, args.sheet)


if __name__ == "__main__":
    main()
