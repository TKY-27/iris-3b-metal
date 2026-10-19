"""Accept or reject an optimization candidate on measured evidence.

    python loop/gate.py prepare                      # once, by a person, ~1-2 h
    python loop/gate.py check --candidate ../wt/c12 --category attention --targets original
    python loop/gate.py merge --id G0012             # after ACCEPT and a PASS review
    python loop/gate.py stopcheck
    python loop/gate.py audit                        # final, holdout and 100 steps

Every verdict ends with one line starting ``GATE`` that the loop's goal reads.
See ``loop/README.md`` for the contract each check enforces.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import decide
import lint_comments
import spec as spec_module
from common import (
    KIT,
    NOTES,
    STATE,
    digest_of,
    frozen_hashes,
    git,
    gpu_lock,
    gpu_utilization,
    head,
    is_clean,
    machine,
    on_ac_power,
    read_json,
    speed_limited,
    state_hashes,
    write_new_json,
)

LOG = STATE / "log.jsonl"
BASELINE = STATE / "baseline"
ORACLE = STATE / "oracle"
QUIET_GPU_PERCENT, QUIET_WAIT_S = 8.0, 600
SWAPOUT_LIMIT = 20_000  # swap-out pages per timing process
LOW_MEMORY_GIB = "10"  # approximates a 16 GB Mac
BLOCKS = (
    ("reference", "candidate", "candidate", "reference"),
    ("candidate", "reference", "reference", "candidate"),
)


def worker(tree: Path, mode: str, arg: str, out: Path, extra_env: dict | None = None) -> dict:
    if (out / "result.json").exists():
        return read_json(out / "result.json")
    env = {**os.environ, "PYTHONPATH": str(tree / "src"), "PYTHONDONTWRITEBYTECODE": "1", **(extra_env or {})}
    command = [
        sys.executable,
        str(KIT / "worker.py"),
        mode,
        "--tree",
        str(tree),
        "--out",
        str(out),
        "--arg",
        arg,
    ]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "worker.log", "w") as log:
        code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=out).returncode
    if code:
        tail = (out / "worker.log").read_text()[-3000:]
        raise RuntimeError(f"worker {mode} {arg} on {tree} exited {code}:\n{tail}")
    return read_json(out / "result.json")


def log_entries() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]


def append_log(entry: dict) -> None:
    with open(LOG, "a") as file:
        file.write(json.dumps(entry, sort_keys=True) + "\n")


def verify(*trees: Path) -> dict:
    manifest = read_json(STATE / "manifest.json")
    for tree in trees:
        changed = [k for k, v in frozen_hashes(tree).items() if manifest["frozen"][k] != v]
        if changed:
            raise SystemExit(f"GATE ERROR harness modified in {tree}: {changed}")
    if state_hashes() != manifest["state"]:
        raise SystemExit("GATE ERROR recorded fixtures or baselines under STATE changed")
    return manifest


# -- prepare -----------------------------------------------------------------
def prepare(args) -> None:
    repo = KIT.parent
    if not is_clean(repo):
        raise SystemExit("commit or stash changes first; prepare freezes this checkout's harness")
    STATE.mkdir(parents=True, exist_ok=True)
    baseline_tree = STATE / "trees" / "baseline"
    if not baseline_tree.exists():
        git(repo, "worktree", "add", "--detach", str(baseline_tree), args.baseline)
    if not (STATE / "config.json").exists():
        found = worker(repo, "resolve", "-", STATE / "work" / "resolve")
        missing = {"depth", "restoration"} - found.keys()
        if missing:
            print(f"task weights not cached, their check is skipped: {sorted(missing)}")
        write_new_json(
            STATE / "config.json",
            {
                "repo": str(repo),
                "baseline": args.baseline,
                "baseline_commit": head(baseline_tree),
                "checkpoints": found,
                "machine": machine(),
                "created": datetime.now(UTC).isoformat(),
            },
        )
    if not (STATE / "spec.json").exists():
        write_new_json(STATE / "spec.json", spec_module.build(baseline_tree))
    with gpu_lock("prepare"):
        for split in ("velocity", "dev", "holdout"):
            started = time.perf_counter()
            worker(baseline_tree, "oracle", split, ORACLE / split)
            print(f"FP32 oracle {split}: {time.perf_counter() - started:.0f}s", flush=True)
        for split in ("dev", "holdout"):
            worker(baseline_tree, "capture-turbo", split, BASELINE / f"turbo-{split}")
            worker(baseline_tree, "capture-original", split, BASELINE / f"original-{split}")
        worker(baseline_tree, "capture-tasks", "dev", BASELINE / "tasks")
        if not (BASELINE / "tests.json").exists():
            write_new_json(BASELINE / "tests.json", run_tests(baseline_tree, STATE / "work" / "tests"))
    manifest = {"frozen": frozen_hashes(repo), "state": state_hashes()}
    if (STATE / "manifest.json").exists():
        if read_json(STATE / "manifest.json") != manifest:
            raise SystemExit("an earlier prepare froze different files; remove STATE to start over")
    else:
        write_new_json(STATE / "manifest.json", manifest)
    print(
        f"prepared. manifest digest {digest_of(manifest['frozen'])[:16]}/{digest_of(manifest['state'])[:16]}"
    )


# -- measurements --------------------------------------------------------------
def original_metrics(base: Path, cand: Path, split: str, spec: dict) -> dict:
    trajectory = {}
    for job in spec["trajectories"][split]:
        name = f"traj-{job['id']}.npy"
        oracle = np.load(ORACLE / split / name)
        trajectory[job["id"]] = (
            decide.rel_l2(np.load(base / name), oracle),
            decide.rel_l2(np.load(cand / name), oracle),
        )
    velocity = {}
    if split == "dev":
        for path in sorted((ORACLE / "velocity").glob("vel-*.npz")):
            with np.load(path) as data:
                oracle = data["v"]
            name = f"{path.stem}.npy"
            velocity[path.stem] = (
                decide.rel_l2(np.load(base / name), oracle),
                decide.rel_l2(np.load(cand / name), oracle),
            )
    return {"trajectory": trajectory, "velocity": velocity}


def correctness(tree: Path, run: Path, split: str, spec: dict, config: dict) -> dict:
    turbo = worker(tree, "capture-turbo", split, run / f"turbo-{split}")
    found = {"turbo_mismatches": decide.mismatches(read_json(BASELINE / f"turbo-{split}/result.json"), turbo)}
    original = worker(tree, "capture-original", split, run / f"original-{split}")
    metrics = original_metrics(BASELINE / f"original-{split}", run / f"original-{split}", split, spec)
    found["original"] = decide.original_quality(
        read_json(BASELINE / f"original-{split}/result.json"), original, metrics, spec["ocr"][split]["words"]
    )
    found["original"]["metrics"] = metrics
    found["tasks_changed"] = []
    if split == "dev" and {"depth", "restoration"} & config["checkpoints"].keys():
        tasks = worker(tree, "capture-tasks", "dev", run / "tasks")
        found["tasks_changed"] = decide.mismatches(read_json(BASELINE / "tasks/result.json"), tasks)
    return found


def wait_quiet() -> str | None:
    deadline = time.monotonic() + QUIET_WAIT_S
    while True:
        problems = []
        if not on_ac_power():
            problems.append("on battery")
        if speed_limited():
            problems.append("thermal speed limit")
        utilization = gpu_utilization()
        if utilization > QUIET_GPU_PERCENT:
            problems.append(f"background GPU {utilization:.0f}%")
        if not problems:
            return None
        if time.monotonic() > deadline:
            return ", ".join(problems)
        time.sleep(30)


def timing(
    reference: Path, candidate: Path, names: list[str], run: Path
) -> tuple[dict, dict, dict, list[str]]:
    samples, noisy = {}, []
    for name in names:
        samples[name] = {"reference": [], "candidate": []}
        for b, block in enumerate(BLOCKS):
            for i, side in enumerate(block):
                if problem := wait_quiet():
                    noisy.append(f"{name}: {problem}")
                tree = reference if side == "reference" else candidate
                result = worker(tree, "time", name, run / f"time-{name}-{b}{i}-{side}")
                if result["swapouts"] > SWAPOUT_LIMIT:
                    noisy.append(f"{name}: {result['swapouts']} swap-outs in one process")
                samples[name][side].append(result)
    times, memory, low_memory_ok = {}, {}, {}
    for name, sides in samples.items():
        ref, cand = sides["reference"], sides["candidate"]
        times[name] = decide.compare_times([r["warm"] for r in ref], [r["warm"] for r in cand])
        times[f"{name}:cold"] = decide.compare_times([[r["cold"]] for r in ref], [[r["cold"]] for r in cand])
        memory[name] = decide.compare_memory([r["peak_gib"] for r in ref], [r["peak_gib"] for r in cand])
        if memory[name]["status"] == "higher":
            low = worker(
                candidate,
                "time",
                name,
                run / f"time-{name}-low-memory",
                {"IRIS_AVAILABLE_MEMORY_GIB": LOW_MEMORY_GIB},
            )
            limit = float(np.median([r["peak_gib"] for r in ref])) + decide.MEMORY_SLACK_GIB
            low_memory_ok[name] = low["peak_gib"] <= limit
    return times, memory, low_memory_ok, noisy


def code_checks(tree: Path, base: str) -> tuple[bool, list[str]]:
    ruff = subprocess.run(["ruff", "check", "."], cwd=tree, capture_output=True, text=True).returncode
    fmt = subprocess.run(
        ["ruff", "format", "--check", "."], cwd=tree, capture_output=True, text=True
    ).returncode
    return ruff == 0 and fmt == 0, lint_comments.flag(tree, base)


def run_tests(tree: Path, out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    report = out / "junit.xml"
    env = {**os.environ, "PYTHONPATH": str(tree / "src"), "HF_HUB_OFFLINE": "1"}
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={report}"]
    with open(out / "pytest.log", "w") as log:
        subprocess.run(command, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT)
    if not report.exists():
        raise RuntimeError(f"pytest wrote no report; see {out / 'pytest.log'}")
    return decide.junit_outcomes(report.read_text())


# -- check ---------------------------------------------------------------------
def check(args) -> None:
    config = read_json(STATE / "config.json")
    spec = read_json(STATE / "spec.json")
    reference, candidate = Path(config["repo"]), args.candidate.resolve()
    if candidate == reference:
        raise SystemExit("GATE ERROR the candidate must be a separate worktree, not the reference checkout")
    verify(reference, candidate)
    for tree in (reference, candidate):
        if not is_clean(tree):
            raise SystemExit(f"GATE ERROR {tree} has uncommitted changes")
    base = head(reference)
    if subprocess.run(["git", "-C", str(candidate), "merge-base", "--is-ancestor", base, "HEAD"]).returncode:
        raise SystemExit(f"GATE ERROR rebase the candidate onto the reference head {base[:12]}")
    if head(candidate) == base:
        raise SystemExit("GATE ERROR the candidate has no commits beyond the reference")
    lint_ok, comments = code_checks(candidate, base)

    with gpu_lock(f"gate check {candidate.name}"):
        gate_id = f"G{1 + sum(1 for e in log_entries() if e.get('type') == 'check'):04d}"
        run = STATE / "runs" / gate_id
        run.mkdir(parents=True)
        entry = {
            "type": "check",
            "id": gate_id,
            "time": datetime.now(UTC).isoformat(),
            "reference": base,
            "candidate": head(candidate),
            "tree": str(candidate),
            "category": args.category,
            "targets": args.targets,
            "full": not args.quick,
            "note": args.note,
            "lint_ok": lint_ok,
            "long_comments": comments,
            "noisy": [],
            "timing": {},
            "memory": {},
            "low_memory_ok": {},
            "turbo_mismatches": [],
            "original": {"status": "skipped", "reasons": []},
            "tasks_changed": [],
        }
        if lint_ok:
            outcomes = run_tests(candidate, run / "tests")
            entry["test_failures"] = decide.lost_tests(read_json(BASELINE / "tests.json"), outcomes)
        entry["tests_ok"] = lint_ok and not entry.get("test_failures")
        if entry["tests_ok"]:
            entry.update(correctness(candidate, run, "dev", spec, config))
        passed = entry["tests_ok"] and not entry["turbo_mismatches"] and entry["original"]["status"] != "fail"
        if passed:
            names = list(spec_module.WORKLOADS)
            if args.quick:
                names = [n for n in names if spec["workloads"][n]["model"] in args.targets]
            times, memory, low, noisy = timing(reference, candidate, names, run)
            entry.update(timing=times, memory=memory, low_memory_ok=low, noisy=noisy)
    entry["verdict"], entry["reasons"] = decide.verdict(entry)
    append_log(entry)
    (run / "entry.json").write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
    report(entry)


def report(entry: dict) -> None:
    for name, result in entry["timing"].items():
        print(
            f"  {name:22} {result['status']:7} {-100 * result['gain']:+6.2f}%  medians {result['reference']} -> {result['candidate']}"
        )
    for name, result in entry["memory"].items():
        print(f"  {name:22} memory {result['status']} {result['delta_gib']:+.2f} GiB")
    if entry["original"].get("metrics"):
        for kind, pairs in entry["original"]["metrics"].items():
            for item, (b, c) in pairs.items():
                print(f"  {kind} {item}: FP32 error {b:.5g} -> {c:.5g}")
    for line in entry["long_comments"]:
        print(f"  comment >= 8 words, reviewer must justify: {line}")
    if entry["tasks_changed"]:
        print(
            f"  depth/restoration output changed {entry['tasks_changed']}: needs human review in runs/{entry['id']}"
        )
    gains = " ".join(f"{n}={-100 * r['gain']:+.2f}%" for n, r in entry["timing"].items() if ":" not in n)
    print(
        f"GATE {entry['id']} VERDICT={entry['verdict']} targets={','.join(entry['targets'])} "
        f"category={entry['category']} original={entry['original']['status']} {gains} | {'; '.join(entry['reasons'])}"
    )


# -- merge, stopcheck, audit ---------------------------------------------------
def merge(args) -> None:
    config = read_json(STATE / "config.json")
    reference = Path(config["repo"])
    verify(reference)
    entry = next((e for e in log_entries() if e.get("id") == args.id and e.get("type") == "check"), None)
    if entry is None or entry["verdict"] != "ACCEPT":
        raise SystemExit(f"{args.id} is not an ACCEPT verdict")
    review = NOTES / "reviews" / f"{args.id}.md"
    text = review.read_text() if review.exists() else ""
    if "VERDICT: PASS" not in text or entry["candidate"][:12] not in text:
        raise SystemExit(
            f"{review} must contain 'VERDICT: PASS' and the candidate commit {entry['candidate'][:12]}"
        )
    if head(reference) != entry["reference"] or not is_clean(reference):
        raise SystemExit(
            "the reference moved or is dirty since this check; rebase the candidate and check again"
        )
    git(reference, "merge", "--ff-only", entry["candidate"])
    append_log(
        {"type": "merge", "id": args.id, "commit": entry["candidate"], "time": datetime.now(UTC).isoformat()}
    )
    print(f"merged {args.id} ({entry['candidate'][:12]}) into {git(reference, 'branch', '--show-current')}")


def stopcheck(args) -> None:
    ledger = NOTES / "LEDGER.md"
    unmet = decide.stopcheck(log_entries(), ledger.read_text() if ledger.exists() else "", NOTES)
    for model, problems in unmet.items():
        print(f"STOPCHECK {model}: {'MET' if not problems else 'NOT MET'}")
        for problem in problems:
            print(f"  - {problem}")
    met = not any(unmet.values())
    print(f"GATE STOPCHECK {'MET' if met else 'NOT MET'}")


def audit(args) -> None:
    """Holdout checks and every workload against v0.6.1."""
    config = read_json(STATE / "config.json")
    spec = read_json(STATE / "spec.json")
    reference, baseline = Path(config["repo"]), STATE / "trees" / "baseline"
    verify(reference)
    run = STATE / "audits" / f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{head(reference)[:12]}"
    run.mkdir(parents=True)
    with gpu_lock("audit"):
        result = {"head": head(reference), "baseline": config["baseline_commit"], "correctness": {}}
        for split in ("dev", "holdout"):
            result["correctness"][split] = correctness(reference, run, split, spec, config)
        times, memory, low, noisy = timing(baseline, reference, list(spec["workloads"]), run)
        result.update(timing=times, memory=memory, low_memory_ok=low, noisy=noisy)
    (run / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, value in times.items():
        print(f"  {name:22} {value['status']:7} {-100 * value['gain']:+6.2f}% vs {config['baseline']}")
    for split, found in result["correctness"].items():
        print(
            f"  {split}: Turbo mismatches {len(found['turbo_mismatches'])}, Original {found['original']['status']}"
        )
    print(f"GATE AUDIT written to {run}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--baseline", default="v0.6.1")
    p = commands.add_parser("check")
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--category", choices=decide.CATEGORIES, required=True)
    p.add_argument(
        "--targets", type=lambda s: s.split(","), required=True, help="original,turbo or one of them"
    )
    p.add_argument(
        "--quick", action="store_true", help="only the targeted workloads; can PROMISE, never ACCEPT"
    )
    p.add_argument("--note", default="", help="one line: the hypothesis")
    p = commands.add_parser("merge")
    p.add_argument("--id", required=True)
    commands.add_parser("stopcheck")
    commands.add_parser("audit")
    args = parser.parse_args()
    if args.command == "check" and set(args.targets) - set(decide.MODELS):
        parser.error(f"targets must be among {decide.MODELS}")
    {"prepare": prepare, "check": check, "merge": merge, "stopcheck": stopcheck, "audit": audit}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
