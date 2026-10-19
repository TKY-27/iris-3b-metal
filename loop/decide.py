"""The gate's decision rules."""

import re
import statistics
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np

MIN_GAIN = 0.005  # below whole-request noise on this Mac
NON_INFERIORITY = 0.01  # a warm slowdown this large rejects even without separation
COLD_SLOWDOWN = 0.02
MEMORY_SLACK_GIB = 0.25
VELOCITY_ITEM_SLACK = 1.02
TRAJECTORY_ITEM_SLACK = 1.10  # trajectories amplify rounding; the means stay strict
STOP_REJECTIONS, STOP_CATEGORIES, STOP_SHARE, STOP_SMALL_TOTAL = 6, 3, 3.0, 10.0
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


def compare_memory(reference: list[float], candidate: list[float], baseline_peak: float) -> dict:
    """``higher`` is measured against v0.6.1, so small increases cannot accumulate."""
    delta = statistics.median(candidate) - statistics.median(reference)
    over = statistics.median(candidate) - baseline_peak
    if delta < -MEMORY_SLACK_GIB and max(candidate) < min(reference):
        status = "lower"
    elif over > MEMORY_SLACK_GIB:
        status = "higher"
    else:
        status = "same"
    return {"status": status, "delta_gib": delta, "over_baseline_gib": over}


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
    pairs = list(baseline["trajectories"].items())
    if changed := [k for k, v in pairs if candidate["trajectories"][k]["encoding"] != v["encoding"]]:
        return {"status": "fail", "reasons": [f"text encodings changed: {changed}"]}
    if changed := [
        k for k, v in pairs if candidate["trajectories"][k]["metadata_keys"] != v["metadata_keys"]
    ]:
        return {"status": "fail", "reasons": [f"metadata keys changed: {changed}"]}
    if not mismatches(baseline, candidate):
        return {"status": "exact", "reasons": []}
    reasons = []
    for kind, slack in (("velocity", VELOCITY_ITEM_SLACK), ("trajectory", TRAJECTORY_ITEM_SLACK)):
        errors = metrics[kind]
        if not errors:
            continue
        base_mean = statistics.fmean(b for b, _ in errors.values())
        cand_mean = statistics.fmean(c for _, c in errors.values())
        if cand_mean > base_mean:
            reasons.append(f"{kind} error vs FP32 rose: mean {base_mean:.5g} -> {cand_mean:.5g}")
        reasons += [
            f"{kind} {item}: {b:.5g} -> {c:.5g} (over {slack}x)"
            for item, (b, c) in errors.items()
            if c > b * slack
        ]
    if missing := set(words) - set(candidate["ocr"]["words"]):
        reasons.append(f"OCR lost {sorted(missing)}")
    return {"status": "fail" if reasons else "within", "reasons": reasons}


def correctness_reasons(found: dict) -> list[str]:
    reasons = [f"Turbo not bit-exact: {found['turbo_mismatches'][:8]}"] if found["turbo_mismatches"] else []
    if found["original"]["status"] == "fail":
        reasons += found["original"]["reasons"]
    return reasons


def counted_gains(entry: dict) -> list[str]:
    gains = []
    for name, result in entry["timing"].items():
        workload, _, phase = name.partition(":")
        if entry["workload_models"][workload] not in entry["targets"]:
            continue
        if phase == "cold" and entry["category"] != "loading":
            continue
        if result["status"] == "faster" and entry["confirmation"].get(name, {}).get("status") == "faster":
            gains.append(name)
    gains += [f"{name} memory" for name, r in entry["memory"].items() if r["status"] == "lower"]
    return gains


def slowdown(name: str, result: dict, category: str) -> bool:
    """Cold starts of non-loading work are noisy single samples, so they need a larger margin."""
    if name.endswith(":cold") and category != "loading":
        return result["status"] == "slower" and result["gain"] <= -COLD_SLOWDOWN
    return result["status"] == "slower" or (not name.endswith(":cold") and result["gain"] < -NON_INFERIORITY)


def suspected_slowdowns(entry: dict) -> list[str]:
    return [name for name, r in entry["timing"].items() if slowdown(name, r, entry["category"])]


def drifted(entry: dict) -> list[str]:
    """Workloads whose estimated time against v0.6.1, over every merge and this one, rose past the bound."""
    return [name for name, ratio in entry["cumulative"].items() if ratio > 1 + NON_INFERIORITY]


def verdict(entry: dict) -> tuple[str, list[str]]:
    if entry["harness_changed"]:
        return "REJECT", [f"the candidate changes the frozen harness: {entry['harness_changed']}"]
    if not entry["lint_ok"]:
        return "BROKEN", ["ruff failed"]
    if not entry["tests_ok"]:
        return "BROKEN", [f"tests that passed on v0.6.1 fail: {entry['test_failures'][:5]}"]
    if reasons := correctness_reasons(entry):
        return "REJECT", reasons
    if entry["low_memory"] and (reasons := correctness_reasons(entry["low_memory"])):
        return "REJECT", [f"with IRIS_AVAILABLE_MEMORY_GIB set: {r}" for r in reasons]
    if entry["noisy"]:
        return "NOISY", entry["noisy"]
    higher = [name for name, r in entry["memory"].items() if r["status"] == "higher"]
    if higher and not (entry["low_memory"] and entry["low_memory"]["peaks_ok"]):
        return "REJECT", [f"peak above v0.6.1 and not released when memory is short: {higher}"]
    confirmed = [
        name
        for name in suspected_slowdowns(entry)
        if name in entry["confirmation"] and slowdown(name, entry["confirmation"][name], entry["category"])
    ]
    if confirmed:
        return "REJECT", [f"slower, confirmed twice: {confirmed}"]
    if drift := drifted(entry):
        return "REJECT", [f"slower than v0.6.1 by more than {NON_INFERIORITY:.0%} after all merges: {drift}"]
    if not (gains := counted_gains(entry)):
        return "NO_GAIN", ["no confirmed improvement on the targeted workloads"]
    return ("ACCEPT" if entry["full"] else "PROMISING"), [f"improved: {gains}"]


# -- stopping ----------------------------------------------------------------
ROW = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*([a-z-]+)\s*\|\s*([a-z-]*)\s*\|\s*([^|]*?)\s*\|\s*$"
)


def ledger_rows(text: str, section: str) -> list[tuple[str, float, str, str, str]]:
    found = re.search(rf"^## {re.escape(section)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not found:
        return []
    return [(m[1], float(m[2]), m[3], m[4], m[5]) for line in found[1].splitlines() if (m := ROW.match(line))]


def measurement_files(evidence: str, notes: Path) -> list[Path]:
    root = (notes / "measurements").resolve()
    paths = [(notes / p).resolve() for p in evidence.split()]
    return [p for p in paths if p.is_relative_to(root) and p.suffix == ".md" and p.is_file()]


def stopcheck(log: list[dict], ledger: str, notes: Path, workloads: dict[str, str]) -> dict[str, list[str]]:
    """Unmet stopping conditions per model; ``workloads`` maps ledger sections to models."""
    unmet = {}
    for model in MODELS:
        problems = []
        mine = [e for e in log if e.get("type") == "check" and model in e["targets"]]
        counted = [e for e in mine if e["full"] and e["correct"] and e.get("code_lines", 0) > 0]
        last_accept = max((i for i, e in enumerate(mine) if e["verdict"] == "ACCEPT"), default=-1)
        tail = [e for e in mine[last_accept + 1 :] if e in counted and e["verdict"] in FAILED]
        distinct = {e["note"].strip().lower() for e in tail}
        if len(distinct) < STOP_REJECTIONS:
            problems.append(
                f"{len(distinct)}/{STOP_REJECTIONS} distinct hypotheses rejected since the last ACCEPT"
            )
        if len({e["category"] for e in tail}) < STOP_CATEGORIES:
            problems.append(
                f"rejections span {len({e['category'] for e in tail})}/{STOP_CATEGORIES} categories"
            )
        failed = {e["id"]: e["category"] for e in counted if e["verdict"] in FAILED}
        for section in (name for name, owner in workloads.items() if owner == model):
            rows = ledger_rows(ledger, section)
            if not 90 <= sum(r[1] for r in rows) <= 110:
                problems.append(
                    f"{section}: ledger shares sum to {sum(r[1] for r in rows):.0f}%, not about 100%"
                )
            if sum(r[1] for r in rows if r[1] < STOP_SHARE) > STOP_SMALL_TOTAL:
                problems.append(
                    f"{section}: rows under {STOP_SHARE:.0f}% add up to more than {STOP_SMALL_TOTAL:.0f}%"
                )
            for name, share, status, category, evidence in rows:
                if share < STOP_SHARE:
                    continue
                if status == "at-bound":
                    if not measurement_files(evidence, notes):
                        problems.append(f"{section} {name}: at-bound needs a notes/measurements/*.md file")
                elif status == "exhausted":
                    cited = [i for i in set(re.findall(r"G\d{4}", evidence)) if failed.get(i) == category]
                    if len(cited) < 2:
                        problems.append(
                            f"{section} {name}: exhausted needs two rejected {category or '?'} checks"
                        )
                else:
                    problems.append(f"{section} {name}: {share:.0f}% of the request is still {status}")
        unmet[model] = problems
    return unmet


def junit_outcomes(xml: str) -> dict[str, str]:
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
