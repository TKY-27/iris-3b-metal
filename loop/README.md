# Optimization loop harness

A frozen measurement gate for running Claude Code (ultracode effort,
`/goal`, auto mode) as an autonomous optimizer of this engine. The rules
the agents follow are in [`CLAUDE.md`](../CLAUDE.md).

## What the gate checks

Each check measures detached worktrees of the reference head and the
candidate commit, so edits during the run change nothing.

| Check | Rule |
|---|---|
| Harness | The kit, `CLAUDE.md`, agents, prompts, ruff and pytest settings, fixtures and baselines hash as frozen by `prepare`. A candidate that changes them or adds test or lint configuration (`conftest.py`, `ruff.toml`, …) is rejected; a changed reference halts. |
| Code | `ruff check` and `ruff format --check` with the frozen settings. Every test that passed on v0.6.1 here still passes and new tests pass; a test counts as lost only after failing twice. A crash of the candidate's code is BROKEN, not ERROR. |
| Turbo | SHA-256 of pre-clamp sample, pixels, metadata and text encodings equal v0.6.1 for 12 dev prompts, 3 second visits and 5 more sizes |
| Original | Text encodings and metadata keys equal v0.6.1. Bit-identical output passes; otherwise mean velocity error against upstream FP32 (2 prompts × 3 noise levels, about 1 MP) and mean trajectory error against FP32 sampling (6 prompts at 512² and 20 steps, 2 at about 1 MP and 10 steps) may not rise, each item within 2% and 10%, and OCR must still read `FRESH` and `BREAD` |
| Depth, restoration | Compared with v0.6.1 when their weights are cached; a change is flagged for a person |
| Speed | Per workload (Turbo repeated prompt, Turbo new prompts, Original 10 steps, all 1024²): a discarded warm-up process per side, then fresh processes in ABBA and BAAB order, with CPU work held off. Each process gets a minimal environment, a fresh `HOME` and `TMPDIR`, per-request seeds and, for new prompts, a per-process nonce. A gain needs every candidate process median below every reference median, a pooled gain of at least 0.5%, a targeted workload, and the same result in a confirmation run. Cold time runs from process start and counts only for the `loading` category. A slowdown (separated, or 1% in pooled warm median; 2% for cold) rejects when the confirmation run reproduces it, and so does an estimated drift of more than 1% against v0.6.1 across all merges. |
| Memory | Peak above v0.6.1 + 0.25 GiB is allowed only if, with `IRIS_AVAILABLE_MEMORY_GIB=10`, Turbo stays bit-exact, Original passes, and the peak returns within 0.25 GiB of v0.6.1. Those checks also run whenever `src/iris_metal/memory.py` exists. |
| Machine | Each timing process waits up to 15 min for AC power, no Low Power Mode, nominal or fair thermal state, load average under 4 and under 8% background GPU; otherwise, or with more than 20,000 swap-outs in a process, the verdict is NOISY |

Timing compares with the current integration head, so each change must beat
the previous one; the audit compares the final head with v0.6.1. Holdout
prompts and Original at 100 steps run only in `audit`, which ends in
`GATE AUDIT PASS` or `FAIL`, lists every merge with its review hash, and
requires every commit on the branch to be a gate merge. `finalcheck` then
allows only the report and README to change before the pull request.

## Running it

1. Merge this kit into `main`, then `git checkout -b opt/loop` from that commit.
2. Use the release environment plus OCR:
   `.venv/bin/python -m pip install -c apps/macos/engine-constraints.txt -e ".[dev]" pyobjc-framework-Vision pyobjc-framework-Quartz pyobjc-framework-Cocoa`.
   Download Original and Turbo once (one `iris-metal` run of each), and run
   `iris-metal depth` and `iris-metal upscale` once if those should be checked.
3. Quiet the Mac: AC power, Low Power Mode off, quit heavy apps and the
   Claude desktop app (it used about 40% of the GPU in the October 11
   records). Run Claude Code in Terminal instead.
4. `.venv/bin/python loop/gate.py prepare` builds a v0.6.1 worktree, the FP32
   oracle, baseline captures, test outcomes and peaks under `~/.iris-loop`
   (2–3 hours; the FP32 model swaps on 24 GB). It ends with
   `GATE PREPARED manifest sha256 …`; keep that hash.
5. Check `gh auth status`: the loop opens the pull request with `gh`.
6. In this checkout with the venv active, start `claude`, run
   `/effort ultracode`, switch to auto mode, set the `/config` workflow size
   to small or medium, then `/goal` with the text in [`GOAL.md`](GOAL.md).
7. Follow progress with `tail -f ~/.iris-loop/log.jsonl` and `/goal`.

## Limits

The hook in `loop/guard.py` blocks obvious edits to the harness and code
run outside `loop/gpu.py`, but a determined script could get past it, and
nothing proves a review came from the reviewer agent rather than the main
session. The gate
re-hashes the harness and the recorded evidence before every verdict;
`log.jsonl` itself is not tamper-proof. Before trusting the result, compare
`shasum -a 256 ~/.iris-loop/manifest.json` with the hash `prepare` printed,
rerun `python loop/gate.py audit` yourself, and look at any depth or
restoration output the ledger lists for review. Training tests that need
the `train` extra are covered by the pull request's CI, not by the gate.
