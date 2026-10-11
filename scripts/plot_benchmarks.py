"""Summarize a benchmark campaign and draw the README's charts.

Usage:
    python scripts/plot_benchmarks.py benchmarks/results/2026-10-11 --assets assets/benchmarks

The results directory holds ``pipeline/*.json`` reports of
``scripts/benchmark.py`` and ``fixture/<implementation>-<steps>-r<round>.log``
outputs of the same-input comparison (``scripts/denoise_fixture.py`` or the
iris3b-mlx-optimized CLI, each run under ``/usr/bin/time -l``). The script
writes ``summary.json`` next to them and three SVG charts that follow the
viewer's light or dark appearance. No dependencies beyond the standard library.
"""

import argparse
import json
import re
import statistics
from html import escape
from pathlib import Path

IMPLEMENTATIONS = {  # fixture log prefix -> label, in chart order
    "ours": "Iris-3B Metal (this project)",
    "mlx-speed": "iris3b-mlx-optimized, speed preset",
    "mlx-memory": "iris3b-mlx-optimized, memory preset",
    "upstream-bf16": "Upstream PyTorch, BF16 autocast",
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
STYLE = """
  .surface { fill: #ffffff; } .ink { fill: #0b0b0b; } .ink2 { fill: #52514e; } .ink3 { fill: #6f6e69; }
  .grid { stroke: #e4e3df; } .axis { stroke: #b5b4ae; } .accent { fill: #2a78d6; } .muted { fill: #a3a29d; }
  .whisker { stroke: #0b0b0b; } .s1 { fill: #2a78d6; } .s2 { fill: #eb6834; } .s3 { fill: #1baf7a; }
  .s4 { fill: #4a3aa7; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #0d1117; } .ink { fill: #f0f6fc; } .ink2 { fill: #c3c2b7; } .ink3 { fill: #9d9c95; }
    .grid { stroke: #2b2f36; } .axis { stroke: #4a4f57; } .accent { fill: #3987e5; } .muted { fill: #6e6d68; }
    .whisker { stroke: #f0f6fc; } .s1 { fill: #3987e5; } .s2 { fill: #d95926; } .s3 { fill: #199e70; }
    .s4 { fill: #9085e9; }
  }
"""


def peak_footprint_gib(log: str) -> float:
    return int(re.search(r"(\d+)\s+peak memory footprint", log)[1]) / 1024**3


def fixture_seconds(name: str, log: str) -> list[float]:
    """Warm denoising times of one process (the first request also compiles kernels)."""
    if name.startswith("mlx"):
        records = [json.loads(line) for line in log.splitlines() if line.startswith('{"repeat"')]
        times = [r["request_s"] for r in records if not r["first_execution"]]
    else:
        times = [float(t) for t in re.search(r"\(([\d., ]+)\)", log)[1].split(", ")][1:]
    if not times:
        raise ValueError(f"{name}: no warm repeats")
    return times


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


def summarize(results: Path) -> dict:
    fixture: dict = {}
    for path in sorted((results / "fixture").glob("*-r*.log")):
        name, steps = re.fullmatch(r"(.+)-(\d+)-r\d+", path.stem).groups()
        log = path.read_text()
        entry = fixture.setdefault(steps, {}).setdefault(name, {"seconds": [], "peak_gib": []})
        entry["seconds"] += fixture_seconds(name, log)
        entry["peak_gib"].append(peak_footprint_gib(log))
    for by_name in fixture.values():
        for entry in by_name.values():
            entry["seconds_summary"] = spread(entry["seconds"])
            entry["peak_gib_summary"] = spread(entry["peak_gib"])

    pipeline: dict = {}
    for path in sorted((results / "pipeline").glob("*.json")):
        report = json.loads(path.read_text())
        runs = pipeline.setdefault(str(report["config"]["steps"]), [])
        runs.append({**report["timings"], "peak_gib": report["peak_footprint_gib"]["process"]})
    keys = (
        "imports_s",
        "text_encoder_tokenizer_init_s",
        "text_encoding_s",
        "model_load_s",
        "first_image_s",
        "imports_and_first_request_s",
        "warm_median_s",
        "warm_engine_uncached_request_s",
        "peak_gib",
    )
    pipeline_summary = {
        steps: {key: spread([run[key] for run in runs]) for key in keys} for steps, runs in pipeline.items()
    }
    return {"fixture": fixture, "pipeline": pipeline_summary}


def svg(width: int, height: int, body: list[str], title: str, description: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" font-family="{FONT}">',
            f"<title>{escape(title)}</title><desc>{escape(description)}</desc>",
            f"<style>{STYLE}</style>",
            f'<rect class="surface" width="{width}" height="{height}" rx="8"/>',
            *body,
            "</svg>\n",
        ]
    )


def text(
    x: float,
    y: float,
    content: str,
    cls: str = "ink",
    size: int = 13,
    anchor: str = "start",
    weight: int = 400,
):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{escape(content)}</text>'
    )


def bar(x: float, y: float, length: float, thickness: float, cls: str) -> str:
    """A horizontal bar, square at the baseline and rounded at its end."""
    r = min(4.0, length / 2, thickness / 2)
    return (
        f'<path class="{cls}" d="M{x:.1f},{y:.1f} h{length - r:.1f} a{r},{r} 0 0 1 {r},{r} '
        f'v{thickness - 2 * r:.1f} a{r},{r} 0 0 1 -{r},{r} h-{length - r:.1f} z"/>'
    )


def nice_ticks(maximum: float, count: int = 4) -> list[float]:
    raw = maximum / count
    magnitude = 10 ** len(str(int(raw))) / 10
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    return [step * i for i in range(int(maximum / step) + 2) if step * i <= maximum * 1.0001 + step]


def bar_panel(x0, y0, width, rows, unit, label_width, title, axis_max):
    """Horizontal bars with min-max whiskers; ``rows`` are (label, summary, highlighted)."""
    body = [text(x0, y0, title, "ink", 14, weight=600)]
    plot_x, plot_w = x0 + label_width, width - label_width - 70
    top, band, thick = y0 + 18, 34, 18
    ticks = nice_ticks(axis_max)
    scale = plot_w / ticks[-1]
    bottom = top + band * len(rows)
    for tick in ticks:
        x = plot_x + tick * scale
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}"/>')
        body.append(text(x, bottom + 15, f"{tick:g}", "ink3", 11, "middle"))
    body.append(f'<line class="axis" x1="{plot_x}" y1="{top}" x2="{plot_x}" y2="{bottom}"/>')
    body.append(text(plot_x + plot_w / 2, bottom + 31, unit, "ink3", 11, "middle"))
    for i, (label, summary, highlighted) in enumerate(rows):
        y = top + band * i + (band - thick) / 2
        body.append(text(x0, y + thick / 2 + 4.5, label, "ink" if highlighted else "ink2", 12.5))
        body.append(bar(plot_x, y, summary["median"] * scale, thick, "accent" if highlighted else "muted"))
        if summary["n"] > 1 and summary["max"] > summary["min"]:
            lo, hi, mid = plot_x + summary["min"] * scale, plot_x + summary["max"] * scale, y + thick / 2
            body.append(
                f'<line class="whisker" x1="{lo:.1f}" y1="{mid}" x2="{hi:.1f}" y2="{mid}" stroke-width="1.2"/>'
            )
            for end in (lo, hi):
                body.append(
                    f'<line class="whisker" x1="{end:.1f}" y1="{mid - 4}" x2="{end:.1f}" y2="{mid + 4}" stroke-width="1.2"/>'
                )
        value_x = plot_x + max(summary["median"], summary["max"]) * scale + 8
        weight = 600 if highlighted else 400
        body.append(
            text(value_x, y + thick / 2 + 4.5, f"{summary['median']:.0f}", "ink", 12.5, weight=weight)
        )
    return body, bottom + 40


def latency_chart(summary: dict, footer: str) -> str:
    width, label_width = 860, 250
    body = [text(24, 34, "Time to denoise one 1024 × 1024 image", "ink", 17, weight=600)]
    body.append(
        text(
            24, 54, "Median of warm runs in seconds; whiskers span the fastest and slowest run.", "ink2", 12.5
        )
    )
    axis_max = max(e["seconds_summary"]["max"] for e in summary["fixture"]["20"].values())
    y = 88
    for steps, name in (("10", "10 steps (Fast)"), ("20", "20 steps (Standard)")):
        rows = [
            (label, summary["fixture"][steps][key]["seconds_summary"], key == "ours")
            for key, label in IMPLEMENTATIONS.items()
            if key in summary["fixture"][steps]
        ]
        panel, y = bar_panel(24, y, width - 24, rows, "seconds", label_width, name, axis_max)
        body += panel
        y += 18
    body.append(text(24, y, footer, "ink3", 11))
    rows_text = "; ".join(
        f"{label}: {summary['fixture'][s][key]['seconds_summary']['median']:.0f} s at {s} steps"
        for s in ("10", "20")
        for key, label in IMPLEMENTATIONS.items()
        if key in summary["fixture"][s]
    )
    return svg(width, int(y + 18), body, "Denoising time, 1024 × 1024", f"Bar chart. {rows_text}. {footer}")


def memory_chart(summary: dict, footer: str) -> str:
    width, label_width = 860, 250
    body = [text(24, 34, "Peak memory", "ink", 17, weight=600)]
    body.append(
        text(24, 54, "Peak physical footprint of the process in GiB, 1024 × 1024, 10 steps.", "ink2", 12.5)
    )
    entries = summary["fixture"]["10"]
    rows = [
        (label, entries[key]["peak_gib_summary"], key == "ours")
        for key, label in IMPLEMENTATIONS.items()
        if key in entries
    ]
    panel, y = bar_panel(24, 88, width - 24, rows, "GiB", label_width, "", 24)
    body += panel
    body.append(text(24, y + 4, footer, "ink3", 11))
    rows_text = "; ".join(
        f"{label}: {entries[key]['peak_gib_summary']['median']:.1f} GiB"
        for key, label in IMPLEMENTATIONS.items()
        if key in entries
    )
    return svg(width, int(y + 22), body, "Peak memory, 1024 × 1024", f"Bar chart. {rows_text}. {footer}")


def startup_chart(summary: dict, footer: str) -> str:
    """Where the time of a first and a later request goes, for the engine alone."""
    width, label_width = 860, 250
    segments = (("Startup", "s1"), ("Text encoding", "s2"), ("Model load", "s3"), ("Denoising", "s4"))
    body = [text(24, 34, "From launch to image with Iris-3B Metal", "ink", 17, weight=600)]
    body.append(text(24, 54, "Median seconds per phase, 1024 × 1024, fresh process each run.", "ink2", 12.5))
    legend_x = 24
    for name, cls in segments:
        body.append(f'<rect class="{cls}" x="{legend_x}" y="70" width="12" height="12" rx="2"/>')
        body.append(text(legend_x + 18, 80.5, name, "ink2", 12))
        legend_x += 30 + 7 * len(name) + 24
    rows = []
    for steps, preset in (("10", "Fast"), ("20", "Standard")):
        p = {key: value["median"] for key, value in summary["pipeline"][steps].items()}
        startup = p["imports_s"] + p["text_encoder_tokenizer_init_s"]
        encode_new = p["warm_engine_uncached_request_s"] - p["warm_median_s"]
        rows += [
            (
                f"{steps} steps ({preset}), first image",
                [startup, p["text_encoding_s"], p["model_load_s"], p["first_image_s"]],
            ),
            (f"{steps} steps ({preset}), new prompt", [0, encode_new, 0, p["warm_median_s"]]),
            (f"{steps} steps ({preset}), same prompt", [0, 0, 0, p["warm_median_s"]]),
        ]
    top, band, thick = 104, 30, 16
    axis_max = max(sum(parts) for _, parts in rows)
    ticks = nice_ticks(axis_max)
    plot_x, plot_w = 24 + label_width, width - 24 - label_width - 70
    scale = plot_w / ticks[-1]
    bottom = top + band * len(rows) + 10
    for tick in ticks:
        x = plot_x + tick * scale
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>')
        body.append(text(x, bottom + 15, f"{tick:g}", "ink3", 11, "middle"))
    body.append(text(plot_x + plot_w / 2, bottom + 31, "seconds", "ink3", 11, "middle"))
    descriptions = []
    for i, (label, parts) in enumerate(rows):
        y = top + band * i + (10 if i >= 3 else 0) + (band - thick) / 2
        body.append(text(24, y + thick / 2 + 4.5, label, "ink2", 12.5))
        x = plot_x
        for value, (_, cls) in zip(parts, segments, strict=True):
            if value <= 0:
                continue
            length = value * scale
            body.append(
                f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{max(length - 2, 1):.1f}" height="{thick}" rx="2"/>'
            )
            x += length
        body.append(text(x + 6, y + thick / 2 + 4.5, f"{sum(parts):.0f}", "ink", 12.5, weight=600))
        descriptions.append(f"{label}: {sum(parts):.0f} s")
    body.append(f'<line class="axis" x1="{plot_x}" y1="{top}" x2="{plot_x}" y2="{bottom}"/>')
    body.append(text(24, bottom + 52, footer, "ink3", 11))
    return svg(
        width,
        bottom + 70,
        body,
        "Phases of a request",
        "Stacked bar chart. " + "; ".join(descriptions) + ". " + footer,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("assets/benchmarks"))
    args = parser.parse_args()
    summary = summarize(args.results)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    machine = "MacBook Pro, Apple M5 (10-core GPU), 24 GB, macOS 27.0.1"
    fixture_note = (
        f"{machine}. Same text states and noise for every implementation; CFG 3, DPM-Solver++ (2nd order)."
    )
    args.assets.mkdir(parents=True, exist_ok=True)
    (args.assets / "latency.svg").write_text(latency_chart(summary, fixture_note))
    (args.assets / "memory.svg").write_text(memory_chart(summary, fixture_note))
    (args.assets / "startup.svg").write_text(
        startup_chart(summary, f"{machine}. Warm file cache; prompt to PNG-ready image.")
    )


if __name__ == "__main__":
    main()
