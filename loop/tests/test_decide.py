import numpy as np
import pytest

import decide


def test_rel_l2():
    reference = np.ones(4, np.float32)
    assert decide.rel_l2(reference, reference) == 0
    assert decide.rel_l2(reference * 1.5, reference) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("reference", "candidate", "status"),
    [
        ([[10.0, 10.1]] * 4, [[9.5, 9.6]] * 4, "faster"),
        ([[10.0, 10.1]] * 4, [[10.5, 10.6]] * 4, "slower"),
        ([[10.0], [10.0], [10.0], [10.0]], [[9.99], [9.99], [9.99], [9.99]], "same"),  # under MIN_GAIN
        ([[10.0], [9.0], [10.0], [10.0]], [[9.5], [9.5], [9.5], [9.5]], "same"),  # medians overlap
    ],
)
def test_compare_times(reference, candidate, status):
    assert decide.compare_times(reference, candidate)["status"] == status


def test_compare_memory_is_bounded_by_v061():
    assert decide.compare_memory([10.0] * 4, [10.2] * 4, 10.0)["status"] == "same"
    assert decide.compare_memory([10.2] * 4, [10.4] * 4, 10.0)["status"] == "higher"  # no ratchet
    assert decide.compare_memory([10.0] * 4, [9.0] * 4, 10.0)["status"] == "lower"
    assert decide.compare_memory([10.0, 9.0, 10.0, 10.0], [9.5, 9.5, 9.5, 9.6], 10.0)["status"] == "same"


def test_mismatches_nested_and_missing():
    baseline = {"a": {"pixels": "1", "preclamp": "2"}, "b": {"pixels": "3"}}
    assert decide.mismatches(baseline, baseline) == []
    assert decide.mismatches(baseline, {"a": {"pixels": "1", "preclamp": "x"}}) == ["a.preclamp", "b.pixels"]


def original(encoding="e", preclamp="p", words=("FRESH", "BREAD"), keys=("model", "seed")):
    return {
        "trajectories": {"t1": {"encoding": encoding, "preclamp": preclamp, "metadata_keys": list(keys)}},
        "velocity": {"v1": "h"},
        "ocr": {"preclamp": preclamp, "words": list(words), "seed": 0},
    }


def metrics(velocity, trajectory):
    return {"velocity": {"v1": velocity}, "trajectory": {"t1": trajectory}}


WORDS = ["FRESH", "BREAD"]


def test_original_exact_skips_metrics():
    assert (
        decide.original_quality(original(), original(), metrics((1, 9), (1, 9)), WORDS)["status"] == "exact"
    )


def test_original_encoding_or_metadata_key_change_fails():
    assert decide.original_quality(original(), original(encoding="x"), {}, WORDS)["status"] == "fail"
    assert decide.original_quality(original(), original(keys=("model",)), {}, WORDS)["status"] == "fail"


def test_original_within_and_worse():
    changed = original(preclamp="q")
    better = decide.original_quality(original(), changed, metrics((0.010, 0.009), (0.05, 0.049)), WORDS)
    assert better["status"] == "within"
    worse = decide.original_quality(original(), changed, metrics((0.010, 0.0101), (0.05, 0.049)), WORDS)
    assert worse["status"] == "fail"


def test_original_holdout_without_velocity_probes():
    changed = original(preclamp="q")
    holdout = {"velocity": {}, "trajectory": {"t1": (0.05, 0.04)}}
    assert decide.original_quality(original(), changed, holdout, WORDS)["status"] == "within"


def test_original_ocr_loss_fails():
    changed = original(preclamp="q", words=("FRESH",))
    result = decide.original_quality(original(), changed, metrics((1, 1), (1, 1)), WORDS)
    assert result["status"] == "fail"
    assert any("BREAD" in reason for reason in result["reasons"])


FASTER = {"status": "faster", "gain": 0.05}
SAME = {"status": "same", "gain": 0.0}


def entry(**changes):
    base = {
        "harness_changed": [],
        "lint_ok": True,
        "tests_ok": True,
        "test_failures": [],
        "turbo_mismatches": [],
        "original": {"status": "exact", "reasons": []},
        "low_memory": None,
        "noisy": [],
        "timing": {"turbo-repeat": FASTER, "turbo-repeat:cold": SAME, "original-10": SAME},
        "confirmation": {"turbo-repeat": FASTER},
        "memory": {"turbo-repeat": {"status": "same"}},
        "workload_models": {"turbo-repeat": "turbo", "original-10": "original"},
        "targets": ["turbo"],
        "category": "kernel-fusion",
        "full": True,
    }
    return {**base, **changes}


def test_verdicts():
    assert decide.verdict(entry())[0] == "ACCEPT"
    assert decide.verdict(entry(full=False))[0] == "PROMISING"
    assert decide.verdict(entry(harness_changed=["loop/decide.py"]))[0] == "REJECT"
    assert decide.verdict(entry(lint_ok=False))[0] == "BROKEN"
    assert decide.verdict(entry(tests_ok=False, test_failures=["t"]))[0] == "BROKEN"
    assert decide.verdict(entry(turbo_mismatches=["x.pixels"]))[0] == "REJECT"
    assert decide.verdict(entry(noisy=["busy"]))[0] == "NOISY"


def test_gains_must_be_targeted_and_confirmed():
    assert decide.verdict(entry(confirmation={}))[0] == "NO_GAIN"
    assert decide.verdict(entry(targets=["original"]))[0] == "NO_GAIN"
    cold = {"turbo-repeat": SAME, "turbo-repeat:cold": FASTER}
    assert decide.verdict(entry(timing=cold, confirmation={"turbo-repeat:cold": FASTER}))[0] == "NO_GAIN"
    loading = entry(timing=cold, confirmation={"turbo-repeat:cold": FASTER}, category="loading")
    assert decide.verdict(loading)[0] == "ACCEPT"


def test_regressions_reject():
    slower = {**entry()["timing"], "original-10": {"status": "slower", "gain": -0.02}}
    assert decide.verdict(entry(timing=slower))[0] == "REJECT"
    drift = {**entry()["timing"], "original-10": {"status": "same", "gain": -0.012}}
    assert decide.verdict(entry(timing=drift))[0] == "REJECT"
    cold_noise = {**entry()["timing"], "turbo-repeat:cold": {"status": "same", "gain": -0.03}}
    assert decide.verdict(entry(timing=cold_noise))[0] == "ACCEPT"


def test_memory_above_v061_needs_the_low_memory_path():
    higher = {"turbo-repeat": {"status": "higher"}}
    assert decide.verdict(entry(memory=higher))[0] == "REJECT"
    released = {"turbo_mismatches": [], "original": {"status": "exact"}, "peaks_ok": True}
    assert decide.verdict(entry(memory=higher, low_memory=released))[0] == "ACCEPT"
    inexact = {**released, "turbo_mismatches": ["fox.pixels"]}
    assert decide.verdict(entry(low_memory=inexact))[0] == "REJECT"


LEDGER = """
## original-10
| component | share % | status | evidence |
|---|---|---|---|
| trunk matmul | 70 | at-bound | measurements/matmul.md |
| attention | 28 | exhausted | G0001 G0002 |
| final layer | 2 | open | |

## turbo-repeat
| component | share % | status | evidence |
|---|---|---|---|
| everything | 100 | open | |
"""
WORKLOADS = {"original-10": "original", "turbo-repeat": "turbo"}


def log(n, model="original", verdict="REJECT", categories=("attention", "matmul", "caching"), **extra):
    return [
        {
            "type": "check",
            "id": f"G{i + 1:04d}",
            "targets": [model],
            "verdict": verdict,
            "category": categories[i % len(categories)],
            "full": True,
            "correct": True,
            **extra,
        }
        for i in range(n)
    ]


@pytest.fixture
def notes(tmp_path):
    (tmp_path / "measurements").mkdir()
    (tmp_path / "measurements/matmul.md").write_text("roofline")
    return tmp_path


def test_stopcheck(notes):
    unmet = decide.stopcheck(log(6), LEDGER, notes, WORKLOADS)
    assert unmet["original"] == []
    assert any("still open" in p for p in unmet["turbo"])


def test_stopcheck_counts_only_after_last_accept(notes):
    entries = log(6) + log(1, verdict="ACCEPT")
    entries[-1]["id"] = "G0007"
    assert decide.stopcheck(entries, LEDGER, notes, WORKLOADS)["original"]


def test_stopcheck_ignores_broken_quick_and_incorrect(notes):
    for weak in (log(6, verdict="BROKEN"), log(6, full=False), log(6, correct=False)):
        assert decide.stopcheck(weak, LEDGER, notes, WORKLOADS)["original"]


def test_stopcheck_needs_evidence_and_limits_small_rows(tmp_path):
    unmet = decide.stopcheck(log(6, categories=("attention",)), LEDGER, tmp_path, WORKLOADS)["original"]
    assert any("categories" in p for p in unmet)
    assert any("measurement file" in p for p in unmet)
    split = LEDGER.replace(
        "| attention | 28 | exhausted | G0001 G0002 |", "\n".join(["| a | 2.8 | open | |"] * 10)
    )
    assert any("rows under" in p for p in decide.stopcheck(log(6), split, tmp_path, WORKLOADS)["original"])


JUNIT = """<testsuites><testsuite>
<testcase classname="tests.test_a" name="test_ok"/>
<testcase classname="tests.test_a" name="test_bad"><failure message="x"/></testcase>
<testcase classname="tests.test_a" name="test_skip"><skipped/></testcase>
</testsuite></testsuites>"""


def test_junit_and_lost_tests():
    outcomes = decide.junit_outcomes(JUNIT)
    assert outcomes == {
        "tests.test_a::test_ok": "passed",
        "tests.test_a::test_bad": "failed",
        "tests.test_a::test_skip": "skipped",
    }
    baseline = {
        "tests.test_a::test_ok": "passed",
        "tests.test_a::test_gone": "passed",
        "t::needs_dion": "failed",
    }
    found = decide.lost_tests(baseline, {**outcomes, "t::needs_dion": "failed"})
    assert found == ["tests.test_a::test_gone", "tests.test_a::test_bad"]
