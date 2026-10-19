"""Accept or reject an optimization candidate on measured evidence.

    python loop/gate.py prepare                      # once, by a person, ~2-3 h
    python loop/gate.py check --candidate ../iris-wt/x --category attention --targets original
    python loop/gate.py merge --id G0012             # after ACCEPT and a PASS review
    python loop/gate.py stopcheck
    python loop/gate.py audit                        # final, holdout and 100 steps

Every command ends with one line starting ``GATE``. See ``loop/README.md``.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
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
    frozen_hashes,
    git,
    gpu_lock,
    gpu_utilization,
    head,
    is_clean,
    low_power_mode,
    machine,
    on_ac_power,
    read_json,
    run_child,
    sha256_file,
    state_hashes,
    thermal_state,
    write_new_json,
)

LOG = STATE / "log.jsonl"
BASELINE = STATE / "baseline"
ORACLE = STATE / "oracle"
QUIET_GPU_PERCENT, QUIET_LOAD, QUIET_WAIT_S = 8.0, 4.0, 900
SWAPOUT_LIMIT = 20_000  # swap-out pages per timing process
LOW_MEMORY_GIB = "10"  # approximates a 16 GB Mac
MINUTES = {"resolve": 10, "oracle": 480, "capture-turbo": 45, "capture-original": 90, "capture-tasks": 45}
TIME_MINUTES = {"original-100": 60}  # other workloads: 30
TEST_MINUTES = 60
BLOCKS = (
    ("reference", "candidate", "candidate", "reference"),
    ("candidate", "reference", "reference", "candidate"),
)


class Halt(Exception):
    """The reference checkout or the recorded evidence changed; a person must look."""


def worker(tree: Path, mode: str, arg: str, out: Path, extra_env: dict | None = None) -> dict:
    if (out / "result.json").exists():
        return read_json(out / "result.json")
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(tree / "src"), "PYTHONDONTWRITEBYTECODE": "1", **(extra_env or {})}
    if mode != "resolve":
        env["HF_HUB_OFFLINE"] = "1"
    if mode == "time":  # no state carried between timing processes except the model files
        home = out / "home"
        home.mkdir()
        env["HF_HOME"] = os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))
        env.update(HOME=str(home), TMPDIR=str(home), XDG_CACHE_HOME=str(home / ".cache"))
    minutes = TIME_MINUTES.get(arg, 30) if mode == "time" else MINUTES[mode]
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
    code = run_child(command, out / "worker.log", minutes * 60, env=env, cwd=out)
    if mode == "time":
        shutil.rmtree(out / "home", ignore_errors=True)
    if code:
        tail = (out / "worker.log").read_text()[-3000:]
        raise RuntimeError(f"worker {mode} {arg} exited {code}:\n{tail}")
    return read_json(out / "result.json")


def log_entries() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]


def append_log(entry: dict) -> None:
    with open(LOG, "a") as file:
        file.write(json.dumps(entry, sort_keys=True) + "\n")


def now() -> str:
    return datetime.now(UTC).isoformat()


def verify_state(config: dict) -> dict:
    manifest = read_json(STATE / "manifest.json")
    if state_hashes() != manifest["state"]:
        raise Halt("recorded fixtures, baselines or inputs under STATE changed")
    baseline = STATE / "trees" / "baseline"
    if head(baseline) != config["baseline_commit"] or not is_clean(baseline):
        raise Halt("the v0.6.1 baseline worktree moved or has changes")
    if changed := harness_changes(KIT.parent, manifest):
        raise Halt(f"the harness running this gate changed: {changed}")
    return manifest


def harness_changes(tree: Path, manifest: dict) -> list[str]:
    return [k for k, v in frozen_hashes(tree).items() if manifest["frozen"].get(k) != v]


def new_run() -> tuple[str, Path]:
    number = 1
    while True:
        path = STATE / "runs" / f"G{number:04d}"
        try:
            path.mkdir(parents=True)
            return path.name, path
        except FileExistsError:
            number += 1


@contextmanager
def snapshots(repo: Path, commits: dict[str, str], run: Path):
    """Detached worktrees, so edits during a long check cannot change what it measures."""
    trees = {}
    try:
        for name, commit in commits.items():
            git(repo, "worktree", "add", "--detach", str(run / name), commit)
            trees[name] = run / name
        yield trees
    finally:
        for tree in trees.values():
            git(repo, "worktree", "remove", "--force", str(tree))


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
        if missing := {"depth", "restoration"} - found.keys():
            print(f"task weights not cached, their check is skipped: {sorted(missing)}")
        write_new_json(
            STATE / "config.json",
            {
                "repo": str(repo),
                "baseline": args.baseline,
                "baseline_commit": head(baseline_tree),
                "checkpoints": found,
                "machine": machine(),
                "created": now(),
            },
        )
    if not (STATE / "spec.json").exists():
        write_new_json(STATE / "spec.json", spec_module.build(baseline_tree))
    spec = read_json(STATE / "spec.json")
    for task, relative in spec["tasks"].items():
        target = STATE / "inputs" / task / Path(relative).name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(baseline_tree / relative, target)
    with gpu_lock("prepare"):
        for split in ("velocity", "dev", "holdout"):
            worker(baseline_tree, "oracle", split, ORACLE / split)
        for split in ("dev", "holdout"):
            worker(baseline_tree, "capture-turbo", split, BASELINE / f"turbo-{split}")
            original = worker(baseline_tree, "capture-original", split, BASELINE / f"original-{split}")
            if missing := set(spec["ocr"][split]["words"]) - set(original["ocr"]["words"]):
                raise SystemExit(f"v0.6.1 never rendered {sorted(missing)} legibly; extend spec OCR seeds")
        worker(baseline_tree, "capture-tasks", "dev", BASELINE / "tasks")
        if not (BASELINE / "tests.json").exists():
            write_new_json(BASELINE / "tests.json", run_tests(baseline_tree, STATE / "work" / "tests"))
        if not (BASELINE / "peaks.json").exists():
            peaks = {
                name: worker(baseline_tree, "time", name, STATE / "work" / f"peak-{name}")["peak_gib"]
                for name in spec["gate_workloads"]
            }
            write_new_json(BASELINE / "peaks.json", peaks)
    manifest = {"frozen": frozen_hashes(repo), "state": state_hashes()}
    if (STATE / "manifest.json").exists():
        if read_json(STATE / "manifest.json") != manifest:
            raise SystemExit("an earlier prepare froze different files; remove STATE to start over")
    else:
        write_new_json(STATE / "manifest.json", manifest)
    print(f"GATE PREPARED manifest sha256 {sha256_file(STATE / 'manifest.json')}")


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


def correctness(tree: Path, out: Path, split: str, spec: dict, config: dict, env: dict | None = None) -> dict:
    turbo = worker(tree, "capture-turbo", split, out / f"turbo-{split}", env)
    found = {"turbo_mismatches": decide.mismatches(read_json(BASELINE / f"turbo-{split}/result.json"), turbo)}
    original = worker(tree, "capture-original", split, out / f"original-{split}", env)
    metrics = original_metrics(BASELINE / f"original-{split}", out / f"original-{split}", split, spec)
    found["original"] = decide.original_quality(
        read_json(BASELINE / f"original-{split}/result.json"), original, metrics, spec["ocr"][split]["words"]
    )
    found["original"]["metrics"] = metrics
    found["tasks_changed"] = []
    if env is None and split == "dev" and {"depth", "restoration"} & config["checkpoints"].keys():
        tasks = worker(tree, "capture-tasks", "dev", out / "tasks")
        found["tasks_changed"] = decide.mismatches(read_json(BASELINE / "tasks/result.json"), tasks)
    return found


def machine_problems() -> list[str]:
    problems = []
    if not on_ac_power():
        problems.append("on battery")
    if low_power_mode():
        problems.append("Low Power Mode")
    if thermal_state() > 1:
        problems.append("thermal state serious or critical")
    if os.getloadavg()[0] > QUIET_LOAD:
        problems.append(f"CPU load {os.getloadavg()[0]:.1f}")
    if (utilization := gpu_utilization()) > QUIET_GPU_PERCENT:
        problems.append(f"background GPU {utilization:.0f}%")
    return problems


def wait_quiet() -> str | None:
    deadline = time.monotonic() + QUIET_WAIT_S
    while problems := machine_problems():
        if time.monotonic() > deadline:
            return ", ".join(problems)
        time.sleep(30)
    return None


def timing(trees: dict[str, Path], names: list[str], out: Path, blocks=BLOCKS) -> tuple[dict, list[str]]:
    """One discarded warm-up per workload, then ``blocks``."""
    samples, noisy = {}, []
    for name in names:
        worker(trees["reference"], "time", name, out / f"time-{name}-warmup")
        samples[name] = {"reference": [], "candidate": []}
        for b, block in enumerate(blocks):
            for i, side in enumerate(block):
                if problem := wait_quiet():
                    noisy.append(f"{name}: {problem}")
                result = worker(trees[side], "time", name, out / f"time-{name}-{b}{i}-{side}")
                if result["swapouts"] > SWAPOUT_LIMIT:
                    noisy.append(f"{name}: {result['swapouts']} swap-outs in one process")
                samples[name][side].append(result)
    return samples, noisy


def compare(samples: dict, peaks: dict[str, float]) -> tuple[dict, dict]:
    times, memory = {}, {}
    for name, sides in samples.items():
        ref, cand = sides["reference"], sides["candidate"]
        times[name] = decide.compare_times([r["warm"] for r in ref], [r["warm"] for r in cand])
        times[f"{name}:cold"] = decide.compare_times([[r["cold"]] for r in ref], [[r["cold"]] for r in cand])
        if name in peaks:
            memory[name] = decide.compare_memory(
                [r["peak_gib"] for r in ref], [r["peak_gib"] for r in cand], peaks[name]
            )
    return times, memory


def low_memory(tree: Path, out: Path, names: list[str], spec: dict, config: dict) -> dict:
    """Exactness and peaks when ``IRIS_AVAILABLE_MEMORY_GIB`` reports a 16 GB Mac."""
    env = {"IRIS_AVAILABLE_MEMORY_GIB": LOW_MEMORY_GIB}
    found = correctness(tree, out / "low-memory", "dev", spec, config, env)
    peaks = read_json(BASELINE / "peaks.json")
    found["peaks"] = {
        name: worker(tree, "time", name, out / f"low-memory-time-{name}", env)["peak_gib"] for name in names
    }
    found["peaks_ok"] = all(found["peaks"][n] <= peaks[n] + decide.MEMORY_SLACK_GIB for n in names)
    return found


def code_checks(tree: Path, base: str) -> tuple[bool, list[str]]:
    results = [
        subprocess.run(command, cwd=tree, capture_output=True, text=True).returncode
        for command in (["ruff", "check", "."], ["ruff", "format", "--check", "."])
    ]
    return not any(results), lint_comments.flag(tree, base)


def run_tests(tree: Path, out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "junit.xml"
    env = {**os.environ, "PYTHONPATH": str(tree / "src"), "HF_HUB_OFFLINE": "1"}
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={report_path}"]
    run_child(command, out / "pytest.log", TEST_MINUTES * 60, env=env, cwd=tree)
    if not report_path.exists():
        raise RuntimeError(f"pytest wrote no report; see {out / 'pytest.log'}")
    return decide.junit_outcomes(report_path.read_text())


def lost_tests(tree: Path, out: Path) -> list[str]:
    """A test counts as lost only if it fails twice."""
    baseline = read_json(BASELINE / "tests.json")
    lost = decide.lost_tests(baseline, run_tests(tree, out))
    if lost:
        second = run_tests(tree, out / "rerun")
        lost = [node for node in lost if second.get(node) != "passed"]
    return lost


# -- check ---------------------------------------------------------------------
def measure(entry: dict, trees: dict[str, Path], out: Path, spec: dict, config: dict, quick: bool) -> None:
    candidate = trees["candidate"]
    entry["lint_ok"], entry["long_comments"] = code_checks(candidate, entry["reference"])
    if not entry["lint_ok"]:
        return
    entry["test_failures"] = lost_tests(candidate, out / "tests")
    entry["tests_ok"] = not entry["test_failures"]
    if not entry["tests_ok"]:
        return
    entry.update(correctness(candidate, out, "dev", spec, config))
    if decide.correctness_reasons(entry):
        return
    entry["correct"] = True
    names = spec["gate_workloads"]
    if quick:
        names = [n for n in names if entry["workload_models"][n] in entry["targets"]]
    samples, entry["noisy"] = timing(trees, names, out)
    entry["timing"], entry["memory"] = compare(samples, read_json(BASELINE / "peaks.json"))
    adaptive = (candidate / "src/iris_metal/memory.py").exists()
    if adaptive or any(r["status"] == "higher" for r in entry["memory"].values()):
        entry["low_memory"] = low_memory(candidate, out, names, spec, config)
    pending = [n for n in decide.counted_gains({**entry, "confirmation": entry["timing"]}) if " " not in n]
    if pending and not entry["noisy"]:
        confirm, noisy = timing(trees, sorted({n.partition(":")[0] for n in pending}), out / "confirm")
        entry["noisy"] += noisy
        entry["confirmation"] = compare(confirm, {})[0]


def new_entry(gate_id: str, args, spec: dict) -> dict:
    return {
        "type": "check",
        "id": gate_id,
        "time": now(),
        "tree": str(args.candidate.resolve()),
        "category": args.category,
        "targets": args.targets,
        "full": not args.quick,
        "note": args.note,
        "workload_models": {n: spec["workloads"][n]["model"] for n in spec["gate_workloads"]},
        "reference": "",
        "candidate": "",
        "harness_changed": [],
        "lint_ok": False,
        "tests_ok": False,
        "test_failures": [],
        "long_comments": [],
        "correct": False,
        "turbo_mismatches": [],
        "original": {"status": "skipped", "reasons": []},
        "tasks_changed": [],
        "low_memory": None,
        "noisy": [],
        "timing": {},
        "memory": {},
        "confirmation": {},
    }


def check(args) -> None:
    config = read_json(STATE / "config.json")
    spec = read_json(STATE / "spec.json")
    repo, candidate = Path(config["repo"]), args.candidate.resolve()
    if candidate == repo:
        raise SystemExit("GATE ERROR the candidate must be a separate worktree, not the reference checkout")
    with gpu_lock(f"gate check {candidate.name}"):
        gate_id, out = new_run()
        entry = new_entry(gate_id, args, spec)
        try:
            manifest = verify_state(config)
            entry["reference"], entry["candidate"] = head(repo), head(candidate)
            ancestry = [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                entry["reference"],
                entry["candidate"],
            ]
            if subprocess.run(ancestry).returncode or entry["candidate"] == entry["reference"]:
                raise RuntimeError(f"rebase the candidate onto the reference head {entry['reference'][:12]}")
            with snapshots(
                repo, {"reference": entry["reference"], "candidate": entry["candidate"]}, out
            ) as trees:
                if changed := harness_changes(trees["reference"], manifest):
                    raise Halt(f"the reference head changes the frozen harness: {changed}")
                entry["harness_changed"] = harness_changes(trees["candidate"], manifest)
                if not entry["harness_changed"]:
                    measure(entry, trees, out, spec, config, args.quick)
            entry["verdict"], entry["reasons"] = decide.verdict(entry)
        except Halt as error:
            entry["verdict"], entry["reasons"] = "ERROR", [f"HALT {error}"]
        except Exception as error:
            entry["verdict"], entry["reasons"] = "ERROR", [f"{type(error).__name__}: {str(error)[-800:]}"]
            (out / "error.txt").write_text(traceback.format_exc())
        append_log(entry)
        (out / "entry.json").write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
    report(entry)


def report(entry: dict) -> None:
    for name, result in entry["timing"].items():
        confirmed = entry["confirmation"].get(name, {}).get("status", "-")
        medians = [round(x, 3) for x in result["reference"]], [round(x, 3) for x in result["candidate"]]
        print(
            f"  {name:22} {result['status']:7} {-100 * result['gain']:+6.2f}%  "
            f"confirm {confirmed:7} medians {medians[0]} -> {medians[1]}"
        )
    for name, result in entry["memory"].items():
        print(f"  {name:22} memory {result['status']} {result['delta_gib']:+.2f} GiB vs head")
    for kind, errors in entry["original"].get("metrics", {}).items():
        for item, (b, c) in errors.items():
            print(f"  {kind} {item}: FP32 error {b:.5g} -> {c:.5g}")
    for line in entry["long_comments"]:
        print(f"  comment >= 8 words, reviewer must justify: {line}")
    if entry["tasks_changed"]:
        print(
            f"  depth/restoration output changed {entry['tasks_changed']}: record it in the ledger for review"
        )
    gains = " ".join(f"{n}={-100 * r['gain']:+.2f}%" for n, r in entry["timing"].items() if ":" not in n)
    print(
        f"GATE {entry['id']} VERDICT={entry['verdict']} targets={','.join(entry['targets'])} "
        f"category={entry['category']} original={entry['original']['status']} {gains} | {'; '.join(entry['reasons'])}"
    )


# -- merge, stopcheck, audit ---------------------------------------------------
def checked_review(entry: dict) -> str:
    """The review's SHA-256; it must start with the exact PASS line and postdate the check."""
    review = NOTES / "reviews" / f"{entry['id']}.md"
    expected = f"VERDICT: PASS {entry['candidate'][:12]}"
    if not review.exists() or review.read_text().splitlines()[:1] != [expected]:
        raise SystemExit(f"GATE ERROR {review} must start with the line '{expected}'")
    if datetime.fromtimestamp(review.stat().st_mtime, UTC) < datetime.fromisoformat(entry["time"]):
        raise SystemExit(f"GATE ERROR {review} is older than the check")
    return sha256_file(review)


def merge(args) -> None:
    config = read_json(STATE / "config.json")
    repo = Path(config["repo"])
    try:
        verify_state(config)
    except Halt as error:
        raise SystemExit(f"GATE ERROR HALT {error}") from None
    entry = next((e for e in log_entries() if e.get("type") == "check" and e["id"] == args.id), None)
    if entry is None or entry["verdict"] != "ACCEPT":
        raise SystemExit(f"GATE ERROR {args.id} is not an ACCEPT verdict")
    review = checked_review(entry)
    if head(repo) != entry["reference"] or not is_clean(repo):
        raise SystemExit(
            "GATE ERROR the reference moved or is dirty since this check; rebase and check again"
        )
    gains = ", ".join(f"{n} {-100 * r['gain']:+.2f}%" for n, r in entry["timing"].items() if ":" not in n)
    message = (
        f"{entry['id']}: {entry['note'] or entry['category']}\n\nGate timing: {gains}\nReview sha256 {review}"
    )
    git(repo, "merge", "--no-ff", "-m", message, entry["candidate"])
    append_log({"type": "merge", "id": args.id, "commit": head(repo), "review_sha256": review, "time": now()})
    print(f"GATE merged {args.id} as {head(repo)[:12]}")


def stopcheck(args) -> None:
    spec = read_json(STATE / "spec.json")
    ledger = NOTES / "LEDGER.md"
    workloads = {n: spec["workloads"][n]["model"] for n in spec["gate_workloads"]}
    unmet = decide.stopcheck(log_entries(), ledger.read_text() if ledger.exists() else "", NOTES, workloads)
    for model, problems in unmet.items():
        print(f"STOPCHECK {model}: {'MET' if not problems else 'NOT MET'}")
        for problem in problems:
            print(f"  - {problem}")
    met = not any(unmet.values())
    append_log({"type": "stopcheck", "met": met, "time": now()})
    print(f"GATE STOPCHECK {'MET' if met else 'NOT MET'}")


def merge_history(entries: list[dict]) -> tuple[list[str], list[str]]:
    lines, problems = [], []
    checks = {e["id"]: e for e in entries if e.get("type") == "check"}
    for merged in (e for e in entries if e.get("type") == "merge"):
        entry = checks.get(merged["id"], {})
        review = NOTES / "reviews" / f"{merged['id']}.md"
        intact = review.exists() and sha256_file(review) == merged["review_sha256"]
        if entry.get("verdict") != "ACCEPT" or not intact:
            problems.append(f"{merged['id']}: not an intact ACCEPT with its reviewed PASS")
        status = "review ok" if intact else "review CHANGED"
        lines.append(f"  merged {merged['id']} {merged['commit'][:12]} {status}: {entry.get('note', '')}")
    return lines, problems


def audit_problems(result: dict, entries: list[dict]) -> list[str]:
    problems = [
        f"{split}: {r}"
        for split, found in result["correctness"].items()
        for r in decide.correctness_reasons(found)
    ]
    if low := result.get("low_memory"):
        problems += [f"low memory: {r}" for r in decide.correctness_reasons(low)]
        if not low["peaks_ok"]:
            problems.append("peak memory above v0.6.1 when memory is short")
    problems += [f"noisy: {n}" for n in result["noisy"]]
    problems += [
        f"{n} slower than v0.6.1"
        for n, r in result["timing"].items()
        if r["status"] == "slower" or r["gain"] < -decide.NON_INFERIORITY
    ]
    last_merge = max((e["time"] for e in entries if e.get("type") == "merge"), default="")
    stops = [e for e in entries if e.get("type") == "stopcheck" and e["time"] > last_merge]
    if not stops or not stops[-1]["met"]:
        problems.append("no GATE STOPCHECK MET after the last merge")
    return problems + merge_history(entries)[1]


def audit(args) -> None:
    """Holdout checks and every workload against v0.6.1."""
    config = read_json(STATE / "config.json")
    spec = read_json(STATE / "spec.json")
    repo = Path(config["repo"])
    long_runs = [n for n in spec["workloads"] if n not in spec["gate_workloads"]]
    with gpu_lock("audit"):
        manifest = verify_state(config)
        final = head(repo)
        out = STATE / "audits" / f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{final[:12]}"
        out.mkdir(parents=True)
        with snapshots(repo, {"final": final}, out) as trees:
            if changed := harness_changes(trees["final"], manifest):
                raise SystemExit(f"GATE AUDIT FAIL the final head changes the harness: {changed}")
            sides = {"reference": STATE / "trees" / "baseline", "candidate": trees["final"]}
            result = {"head": final, "baseline": config["baseline_commit"], "correctness": {}}
            for split in ("dev", "holdout"):
                result["correctness"][split] = correctness(trees["final"], out, split, spec, config)
            samples, noisy = timing(sides, spec["gate_workloads"], out)
            long_samples, long_noisy = timing(sides, long_runs, out, BLOCKS[:1])
            times, memory = compare({**samples, **long_samples}, read_json(BASELINE / "peaks.json"))
            if (trees["final"] / "src/iris_metal/memory.py").exists() or any(
                r["status"] == "higher" for r in memory.values()
            ):
                result["low_memory"] = low_memory(trees["final"], out, spec["gate_workloads"], spec, config)
        result.update(timing=times, memory=memory, noisy=noisy + long_noisy)
    entries = log_entries()
    result["problems"] = audit_problems(result, entries)
    (out / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("\n".join(merge_history(entries)[0]))
    for name, value in times.items():
        print(f"  {name:22} {value['status']:7} {-100 * value['gain']:+6.2f}% vs {config['baseline']}")
    for name, value in memory.items():
        print(f"  {name:22} peak {value['over_baseline_gib']:+.2f} GiB vs {config['baseline']}")
    for split, found in result["correctness"].items():
        print(
            f"  {split}: Turbo mismatches {len(found['turbo_mismatches'])}, Original {found['original']['status']}"
        )
    outcome = "PASS" if not result["problems"] else "FAIL " + "; ".join(result["problems"])
    print(f"GATE AUDIT {outcome} (written to {out})")


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
    if args.command == "check" and (not args.targets or set(args.targets) - set(decide.MODELS)):
        parser.error(f"targets must be among {decide.MODELS}")
    commands = {"prepare": prepare, "check": check, "merge": merge, "stopcheck": stopcheck, "audit": audit}
    try:
        commands[args.command](args)
    except Halt as error:
        raise SystemExit(f"GATE ERROR HALT {error}") from None


if __name__ == "__main__":
    main()
