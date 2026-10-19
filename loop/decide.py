"""The gate's decision rules."""

import re
import statistics
from pathlib import Path

import numpy as np

MIN_GAIN = 0.005  # below whole-request noise on this Mac
MEMORY_SLACK_GIB = 0.25
VELOCITY_ITEM_SLACK = 1.02
TRAJECTORY_ITEM_SLACK = 1.10  # trajectories amplify rounding; the means stay strict
STOP_REJECTIONS, STOP_CATEGORIES, STOP_SHARE = 6, 3, 3.0
CATEGORIES = (
    "kernel-fusion",
    "matmul",
    "attention",
    "memory-layout",
    "text-encoder",
    "sampler-loop",
    "loading",
    "caching",
    "allocation",
    "launch-overhead",
    "precision",
    "host-overhead",
)
MODELS = ("original", "turbo")
FAILED = ("REJECT", "NO_GAIN")


def rel_l2(value: np.ndarray, reference: np.ndarray) -> float:
    value, reference = value.astype(np.float64), reference.astype(np.float64)
    return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))


def compare_times(reference: list[list[float]], candidate: list[list[float]]) -> dict:
    ref = [statistics.median(p) for p in reference]
    cand = [statistics.median(p) for p in candidate]
    gain = 1 - statistics.median([t for p in candidate for t in p]) / statistics.median(
        [t for p in reference for t in p]
    )
    if max(cand) < min(ref) and gain >= MIN_GAIN:
        status = "faster"
    elif min(cand) > max(ref) and -gain >= MIN_GAIN:
        status = "slower"
    else:
        status = "same"
    return {"status": status, "gain": gain, "reference": ref, "candidate": cand}


def compare_memory(reference: list[float], candidate: list[float]) -> dict:
    delta = statistics.median(candidate) - statistics.median(reference)
    if delta > MEMORY_SLACK_GIB:
        status = "higher"
    elif delta < -MEMORY_SLACK_GIB and max(candidate) < min(reference):
        status = "lower"
    else:
        status = "same"
    return {"status": status, "delta_gib": delta}


def mismatches(baseline: dict, candidate: dict, prefix: str = "") -> list[str]:
    found = []
    for key, expected in baseline.items():
        actual = candidate.get(key) if isinstance(candidate, dict) else None
        if isinstance(expected, dict):
            found += mismatches(expected, actual or {}, f"{prefix}{key}.")
        elif actual != expected:
            found.append(f"{prefix}{key}")
    return found


def original_quality(baseline: dict, candidate: dict, metrics: dict, words: list[str]) -> dict:
    """``metrics``: {"velocity"|"trajectory": {item: (baseline_error, candidate_error)}}."""
    encodings = [
        k
        for k, v in baseline["trajectories"].items()
        if candidate["trajectories"][k]["encoding"] != v["encoding"]
    ]
    if encodings:
        return {"status": "fail", "reasons": [f"text encodings changed: {encodings}"]}
    if not mismatches(baseline, candidate):
        return {"status": "exact", "reasons": []}
    reasons = []
    for kind, slack in (("velocity", VELOCITY_ITEM_SLACK), ("trajectory", TRAJECTORY_ITEM_SLACK)):
        pairs = metrics[kind]
        base_mean = statistics.fmean(b for b, _ in pairs.values())
        cand_mean = statistics.fmean(c for _, c in pairs.values())
        if cand_mean > base_mean:
            reasons.append(f"{kind} error vs FP32 rose: mean {base_mean:.5g} -> {cand_mean:.5g}")
        reasons += [
            f"{kind} {item}: {b:.5g} -> {c:.5g} (over {slack}x)"
            for item, (b, c) in pairs.items()
            if c > b * slack
        ]
    required = set(words) & set(baseline["ocr"]["words"])
    if missing := required - set(candidate["ocr"]["words"]):
        reasons.append(f"OCR lost {sorted(missing)}")
    return {"status": "fail" if reasons else "within", "reasons": reasons}


def verdict(entry: dict) -> tuple[str, list[str]]:
    reasons = []
    if not entry["lint_ok"]:
        reasons.append("ruff failed")
    elif not entry["tests_ok"]:
        reasons.append(f"tests that passed on v0.6.1 fail: {entry.get('test_failures', [])[:5]}")
    if entry["turbo_mismatches"]:
        reasons.append(f"Turbo not bit-exact: {entry['turbo_mismatches'][:8]}")
    if entry["original"]["status"] == "fail":
        reasons += entry["original"]["reasons"]
    if reasons:
        return "REJECT", reasons
    if entry["noisy"]:
        return "NOISY", entry["noisy"]
    timing, memory = entry["timing"], entry["memory"]
    slower = [name for name, r in timing.items() if r["status"] == "slower"]
    higher = [
        name for name, r in memory.items() if r["status"] == "higher" and not entry["low_memory_ok"].get(name)
    ]
    if slower or higher:
        reasons = [f"slower: {slower}"] if slower else []
        if higher:
            reasons.append(f"more memory when memory is short: {higher}")
        return "REJECT", reasons
    gains = [name for name, r in timing.items() if r["status"] == "faster"]
    gains += [f"{name} memory" for name, r in memory.items() if r["status"] == "lower"]
    if not gains:
        return "NO_GAIN", ["no measurable improvement"]
    return ("ACCEPT" if entry["full"] else "PROMISING"), [f"improved: {gains}"]


# -- stopping ----------------------------------------------------------------
ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|\s*([a-z-]+)\s*\|\s*([^|]*?)\s*\|\s*$")


def ledger_rows(text: str, model: str) -> list[tuple[str, float, str, str]]:
    section = re.search(rf"^## {model}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not section:
        return []
    return [(m[1], float(m[2]), m[3], m[4]) for line in section[1].splitlines() if (m := ROW.match(line))]


def stopcheck(log: list[dict], ledger: str, notes: Path) -> dict[str, list[str]]:
    """Unmet stopping conditions per model."""
    unmet = {}
    for model in MODELS:
        problems = []
        mine = [
            e
            for e in log
            if model in e.get("targets", ()) and e["verdict"] in (*FAILED, "ACCEPT", "PROMISING")
        ]
        last_accept = max((i for i, e in enumerate(mine) if e["verdict"] == "ACCEPT"), default=-1)
        tail = [e for e in mine[last_accept + 1 :] if e["verdict"] in FAILED]
        if len(tail) < STOP_REJECTIONS:
            problems.append(f"{len(tail)}/{STOP_REJECTIONS} rejected candidates since the last ACCEPT")
        if len({e["category"] for e in tail}) < STOP_CATEGORIES:
            problems.append(
                f"rejections span {len({e['category'] for e in tail})}/{STOP_CATEGORIES} categories"
            )
        rows = ledger_rows(ledger, model)
        total = sum(share for _, share, _, _ in rows)
        if not 90 <= total <= 110:
            problems.append(f"ledger shares sum to {total:.0f}%, expected about 100%")
        failed_ids = {e["id"] for e in mine if e["verdict"] in FAILED}
        for name, share, status, evidence in rows:
            if share < STOP_SHARE:
                continue
            if status == "at-bound":
                paths = [p for p in evidence.split() if (notes / p).is_file()]
                if not paths:
                    problems.append(f"{name}: at-bound needs a measurement file under notes/")
            elif status == "exhausted":
                cited = set(re.findall(r"G\d{4}", evidence)) & failed_ids
                if len(cited) < 2:
                    problems.append(f"{name}: exhausted needs two rejected gate ids for {model}")
            else:
                problems.append(f"{name}: {share:.0f}% of time is still {status}")
        unmet[model] = problems
    return unmet


def junit_outcomes(xml: str) -> dict[str, str]:
    import xml.etree.ElementTree as ElementTree

    outcomes = {}
    for case in ElementTree.fromstring(xml).iter("testcase"):
        kinds = {child.tag for child in case}
        outcome = "failed" if kinds & {"failure", "error"} else "skipped" if "skipped" in kinds else "passed"
        outcomes[f"{case.get('classname')}::{case.get('name')}"] = outcome
    return outcomes


def lost_tests(baseline: dict[str, str], candidate: dict[str, str]) -> list[str]:
    """Lost v0.6.1 passes, and failing new tests."""
    lost = [
        node for node, outcome in baseline.items() if outcome == "passed" and candidate.get(node) != "passed"
    ]
    return lost + [
        node for node, outcome in candidate.items() if node not in baseline and outcome == "failed"
    ]
