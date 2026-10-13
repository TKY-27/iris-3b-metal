"""Summarize v0.5.0 measurements and plot Turbo comparison."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def spread(values):
    quartiles = statistics.quantiles(values, method="inclusive") if len(values) > 1 else values * 3
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "iqr": quartiles[2] - quartiles[0],
        "n": len(values),
    }


def summarize(directory):
    groups = {}
    for path in sorted(directory.glob("bench-*.json")):
        report = json.loads(path.read_text())
        metadata = report["metadata"]
        if metadata["model"] == "original":
            key = f"original-{metadata['steps']}"
        elif report.get("implementation") == "v0.5.0-release-bf16":
            key = "turbo-release"
        else:
            key = "turbo-bf16" if "bf16-linear" in metadata["precision"] else "turbo-metal"
        rows = groups.setdefault(key, {"cold_s": [], "warm_s": [], "load_s": [], "peak_gib": []})
        rows["cold_s"].append(report["requests"][0]["seconds"])
        rows["warm_s"].extend(r["seconds"] for r in report["requests"][1:])
        rows["load_s"].extend(s["seconds"] for s in report["stages"] if s["phase"] == "dit_load")
        rows["peak_gib"].append(max(r["physical_peak_gib"] for r in report["requests"]))
    return {key: {metric: spread(values) for metric, values in rows.items()} for key, rows in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("assets/benchmarks/turbo.svg"))
    args = parser.parse_args()
    summary = summarize(args.results)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(left=0.18, right=0.94, top=0.70, bottom=0.24, wspace=0.85)
    fig.text(0.04, 0.93, "Official Turbo · same weights and four-step sampler", fontsize=15, weight="bold")
    fig.text(0.04, 0.84, "M5 · 24 GiB · 1024 × 1024 · fixed prompt and seed · lower is better", fontsize=10)
    labels = ["Previous BF16", "Released BF16", "FP16 (rejected)"]
    keys = ["turbo-bf16", "turbo-release", "turbo-metal"]
    for ax, metric, title, unit in zip(
        axes,
        ["warm_s", "peak_gib"],
        ["Warm generation", "Peak process memory"],
        ["Seconds", "GiB"],
        strict=True,
    ):
        stats = [summary[key][metric] for key in keys]
        values = [s["median"] for s in stats]
        errors = [[s["median"] - s["min"] for s in stats], [s["max"] - s["median"] for s in stats]]
        ax.barh(labels, values, xerr=errors, height=0.48, color=["#9299a1", "#236fb2", "#b1b1b1"], capsize=4)
        ax.invert_yaxis()
        ax.set_xlim(0, max(s["max"] for s in stats) * 1.2)
        ax.set_xlabel(unit)
        ax.set_title(title, loc="left", fontsize=12, pad=14)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#e4e7ea")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0)
        for i, s in enumerate(stats):
            ax.text(s["max"] + ax.get_xlim()[1] * 0.025, i, f"{s['median']:.2f}", va="center", fontsize=11)
    fig.text(
        0.04,
        0.11,
        "Medians; whiskers show min–max. Four warm requests per path; process counts differ.",
        fontsize=10,
    )
    fig.text(
        0.04,
        0.045,
        "FP16 failed the typography quality gate and is not shipped. v0.5.0 keeps the official BF16 recipe.",
        fontsize=9,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, metadata={"Date": None, "Title": "Official Turbo performance, v0.5.0"})
    plt.close(fig)


if __name__ == "__main__":
    main()
