# Optimization loop harness

A frozen measurement gate for running Claude Code (ultracode effort,
`/goal`, auto mode) as an autonomous optimizer of this engine. The rules
the agents follow are in [`CLAUDE.md`](../CLAUDE.md).

## What the gate checks

| Check | Rule |
|---|---|
| Code | `ruff check`, `ruff format --check`; every test that passed on v0.6.1 in this environment (with MPS) still passes, and new tests pass |
| Turbo | SHA-256 of pre-clamp sample, pixels, metadata and text encodings equal v0.6.1 for 12 dev prompts, 3 second visits and 5 more sizes |
| Original | Bit-identical to v0.6.1, or: text encodings identical; mean velocity error against upstream FP32 (2 prompts × 3 noise levels, about 1 MP) and mean 20-step trajectory error against FP32 sampling (6 prompts, 512²) not above v0.6.1; per item within 2% and 10%; OCR keeps `FRESH` and `BREAD` |
| Depth, restoration | Compared with v0.6.1 when their weights are cached; a change is flagged for a person |
| Speed | Fresh processes, ABBA then BAAB, per workload: Turbo repeated prompt, Turbo new prompts, Original 10 steps at 1024². A change counts when every candidate process median is below every reference median and the pooled median moves by at least 0.5%; cold requests are compared the same way. Any slower workload rejects. |
| Memory | Peak physical footprint; more than 0.25 GiB above the reference must disappear when `IRIS_AVAILABLE_MEMORY_GIB=10` |
| Machine | Each timing process waits for AC power, no thermal limit and under 8% background GPU; more than 20,000 swap-outs in a process makes the verdict NOISY |

Timing compares the candidate with the current integration head, so every
accepted change must beat the previous one. Correctness always compares with
v0.6.1. Holdout prompts and Original at 100 steps run only in `audit`.

## Running it

1. Merge this kit, then create the integration branch from that commit:
   `git checkout -b opt/loop`.
2. Use the release environment plus OCR:
   `.venv/bin/python -m pip install -c apps/macos/engine-constraints.txt -e ".[dev]" pyobjc-framework-Vision`.
   Download Original and Turbo once (for example one `iris-metal` run of
   each). Run `iris-metal depth` and `iris-metal upscale` once as well if
   those outputs should be checked.
3. Quiet the Mac: AC power, Low Power Mode off, quit heavy apps and the
   Claude desktop app (it used about 40% of the GPU in the October 11
   records). Run Claude Code in Terminal instead.
4. `.venv/bin/python loop/gate.py prepare` builds a v0.6.1 worktree, the FP32
   oracle and the baseline captures under `~/.iris-loop` (1–2 hours; the
   FP32 model swaps on 24 GB). Keep the digest it prints.
5. In this checkout, with the venv active: `claude --effort ultracode`,
   switch to auto mode, keep `/config` workflow size at small or medium,
   then `/goal` with the text in [`GOAL.md`](GOAL.md).
6. Follow progress with `tail -f ~/.iris-loop/log.jsonl` and `/goal`.

## Limits

The hook in `loop/guard.py` blocks obvious edits to the harness and
unlocked GPU commands, but a determined script could get past it. The gate
re-hashes the harness and the recorded evidence before every verdict, which
catches changes to them; it cannot make `log.jsonl` tamper-proof. Before
trusting the result, rerun `python loop/gate.py audit` yourself from a fresh
checkout of the final `opt/loop`, check that `prepare` would print the same
digest, and look at any depth or restoration output the gate flagged.
