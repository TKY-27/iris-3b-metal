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


def test_compare_memory():
    assert decide.compare_memory([10.0] * 4, [10.4] * 4)["status"] == "higher"
    assert decide.compare_memory([10.0] * 4, [10.2] * 4)["status"] == "same"
    assert decide.compare_memory([10.0] * 4, [9.0] * 4)["status"] == "lower"
    assert decide.compare_memory([10.0, 9.0, 10.0, 10.0], [9.5, 9.5, 9.5, 9.6])["status"] == "same"


def test_mismatches_nested_and_missing():
    baseline = {"a": {"pixels": "1", "preclamp": "2"}, "b": {"pixels": "3"}}
    assert decide.mismatches(baseline, baseline) == []
    assert decide.mismatches(baseline, {"a": {"pixels": "1", "preclamp": "x"}}) == ["a.preclamp", "b.pixels"]


def original(encoding="e", preclamp="p", words=("FRESH", "BREAD")):
    return {
        "trajectories": {"t1": {"encoding": encoding, "preclamp": preclamp}},
        "velocity": {"v1": "h"},
        "ocr": {"preclamp": preclamp, "words": list(words)},
    }


def metrics(velocity, trajectory):
    return {"velocity": {"v1": velocity}, "trajectory": {"t1": trajectory}}


WORDS = ["FRESH", "BREAD"]


def test_original_exact_skips_metrics():
    result = decide.original_quality(original(), original(), metrics((1, 9), (1, 9)), WORDS)
    assert result["status"] == "exact"


def test_original_encoding_change_fails():
    assert decide.original_quality(original(), original(encoding="x"), {}, WORDS)["status"] == "fail"


def test_original_within_and_worse():
    changed = original(preclamp="q")
    better = decide.original_quality(original(), changed, metrics((0.010, 0.009), (0.05, 0.049)), WORDS)
    assert better["status"] == "within"
    worse = decide.original_quality(original(), changed, metrics((0.010, 0.0101), (0.05, 0.049)), WORDS)
    assert worse["status"] == "fail"


def test_original_ocr_loss_fails():
    changed = original(preclamp="q", words=("FRESH",))
    result = decide.original_quality(original(), changed, metrics((1, 1), (1, 1)), WORDS)
    assert result["status"] == "fail"
    assert any("BREAD" in reason for reason in result["reasons"])


def entry(**changes):
    base = {
        "tests_ok": True,
        "lint_ok": True,
        "turbo_mismatches": [],
        "original": {"status": "exact", "reasons": []},
        "noisy": [],
        "timing": {"turbo-repeat": {"status": "faster"}, "original-10": {"status": "same"}},
        "memory": {"turbo-repeat": {"status": "same"}},
        "low_memory_ok": {},
        "full": True,
    }
    return {**base, **changes}


def test_verdicts():
    assert decide.verdict(entry())[0] == "ACCEPT"
    assert decide.verdict(entry(full=False))[0] == "PROMISING"
    assert decide.verdict(entry(turbo_mismatches=["x.pixels"]))[0] == "REJECT"
    assert decide.verdict(entry(noisy=["busy"]))[0] == "NOISY"
    assert decide.verdict(entry(timing={"a": {"status": "same"}}))[0] == "NO_GAIN"
    slower = {"a": {"status": "faster"}, "b": {"status": "slower"}}
    assert decide.verdict(entry(timing=slower))[0] == "REJECT"
    higher = {"turbo-repeat": {"status": "higher"}}
    assert decide.verdict(entry(memory=higher))[0] == "REJECT"
    assert decide.verdict(entry(memory=higher, low_memory_ok={"turbo-repeat": True}))[0] == "ACCEPT"


def test_correctness_failure_outranks_noise():
    assert decide.verdict(entry(tests_ok=False, noisy=["busy"]))[0] == "REJECT"


LEDGER = """
## original
| component | share % | status | evidence |
|---|---|---|---|
| trunk matmul | 70 | at-bound | measurements/matmul.md |
| attention | 28 | exhausted | G0001 G0002 |
| final layer | 2 | open | |

## turbo
| component | share % | status | evidence |
|---|---|---|---|
| everything | 100 | open | |
"""


def log(n, model="original", verdict="REJECT", categories=("attention", "matmul", "caching")):
    return [
        {
            "id": f"G{i + 1:04d}",
            "targets": [model],
            "verdict": verdict,
            "category": categories[i % len(categories)],
        }
        for i in range(n)
    ]


def test_stopcheck(tmp_path):
    (tmp_path / "measurements").mkdir()
    (tmp_path / "measurements/matmul.md").write_text("roofline")
    unmet = decide.stopcheck(log(6), LEDGER, tmp_path)
    assert unmet["original"] == []
    assert any("still open" in p for p in unmet["turbo"])


def test_stopcheck_counts_only_after_last_accept(tmp_path):
    (tmp_path / "measurements").mkdir()
    (tmp_path / "measurements/matmul.md").write_text("roofline")
    entries = log(6)
    entries.append({"id": "G0007", "targets": ["original"], "verdict": "ACCEPT", "category": "attention"})
    assert any("rejected candidates" in p for p in decide.stopcheck(entries, LEDGER, tmp_path)["original"])


def test_stopcheck_needs_evidence(tmp_path):
    unmet = decide.stopcheck(log(6, categories=("attention",)), LEDGER, tmp_path)["original"]
    assert any("categories" in p for p in unmet)
    assert any("measurement file" in p for p in unmet)


JUNIT = """<testsuites><testsuite>
<testcase classname="tests.test_a" name="test_ok"/>
<testcase classname="tests.test_a" name="test_bad"><failure message="x"/></testcase>
<testcase classname="tests.test_a" name="test_skip"><skipped/></testcase>
</testsuite></testsuites>"""


def test_junit_and_regressions():
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
