# Iris-3B Metal: optimization loop

This checkout is driven by an autonomous optimization loop. The goal is to
make Original and Turbo inference on Apple Silicon faster, or lighter, until
nothing measurable is left. A measurement gate decides; your judgment does
not. Read this whole file before the first change.

## Scope

- In scope: `src/iris_metal/`, the inference paths of `src/iris3b/` (models,
  `nn`, `flow`, `sampling`, `text`), their tests, and new tests.
- `src/iris3b` is shared with the upstream training code. Training behaviour
  must not change. The gate runs every test that passed on v0.6.1 in this
  environment; the pull request's CI also runs the training suite.
- Out of scope: the macOS app, training, the HTTP and CLI interfaces,
  dependencies, and docs other than the final report and the README
  performance section.

## Contracts

These are fixed. The gate enforces them; do not argue with a verdict.

1. **Turbo is bit-exact** with the official BF16 recipe as v0.6.1 runs it:
   pre-clamp sample, pixels, metadata and text encodings of every fixture,
   including second visits (caches) and nine sizes. Any change of
   accumulation order, operand precision or rounding point breaks this.
   Metal FP16 Turbo was already rejected for turning `BREAD` into `BRIEAD`.
2. **Original may change its output only if it does not move away from
   FP32.** Velocity error against the upstream FP32 forward, and trajectory
   error against FP32 sampling, may not rise on average (per-item slack in
   `loop/decide.py`). The OCR words of the typography fixture must survive,
   and metadata keys stay. Bit-identical output passes without these checks.
3. **Text encodings stay bit-identical** for both models, for the same
   sequence of calls.
4. **Memory.** Peak memory may exceed v0.6.1 only when memory is plentiful.
   Put the budget in `src/iris_metal/memory.py`, honouring
   `IRIS_AVAILABLE_MEMORY_GIB`. Once that file exists the gate also checks
   exactness with `IRIS_AVAILABLE_MEMORY_GIB=10`, and there the peak must
   stay within 0.25 GiB of v0.6.1.
5. **Interfaces stay.** `IrisPipeline.__call__`, `encode`, `last_metadata`,
   metadata keys, CLI and HTTP API are unchanged. For Original,
   `pipe.engine.encode_text(states, mask)` and `pipe.engine(x, t, text)`
   returning FP32 velocity remain; the gate probes them. The final sample
   still passes through `Tensor.clamp(-1, 1)`, where the gate records it.
6. **No benchmark gaming.** Never cache generated samples or images, never
   keep caches of encodings, weights or kernels across processes, never
   detect or special-case the gate's prompts, sizes, seeds or workloads.
   The cold time covers pipeline construction and the first request, so
   moving work into construction or imports gains nothing.
7. Depth and restoration reuse `MetalIris`. When the gate reports their
   output changed, record it in the ledger's `## review` section; a person
   checks those images before the final merge.

## Harness (read-only)

`loop/`, `.claude/`, this file, `benchmarks/prompts.json`, the ruff and
pytest settings in `pyproject.toml`, and `~/.iris-loop/` except
`~/.iris-loop/notes/` are frozen. A hook blocks edits; the gate re-hashes
everything before each verdict. A candidate that changes any of it is
rejected; a reference head that does halts the loop.

- One GPU job at a time, machine-wide. Any Python command runs as
  `python loop/gpu.py run -- <command>`, which waits 60 s for the lock and
  then exits (`--wait SECONDS` changes that). The only exception is CPU
  tests: `IRIS_SKIP_GPU_TESTS=1 python -m pytest ...`.
- While a gate check holds the GPU (often two hours), implementers keep
  writing code and running CPU tests, and try the GPU again later. Start
  micro-benchmarks as background commands and keep them under five minutes.
- `python loop/gate.py check --candidate <worktree> --category <c> --targets original,turbo --note "<hypothesis>"`
  snapshots both commits, then runs lint, tests, correctness and timing in
  fresh processes (ABBA and BAAB after a warm-up). A speedup counts only on
  the targeted workloads, only when a second timing run confirms it, and
  cold starts count only for the `loading` category. `--quick` times only
  the targeted workloads and can return PROMISING, never ACCEPT; a
  PROMISING candidate needs a full check before it can merge.
- Verdicts: ACCEPT, PROMISING, NO_GAIN, REJECT, BROKEN (lint or tests),
  NOISY (the Mac was busy, hot, on battery or swapping), ERROR. The last
  line of every command starts with `GATE`.
- `python loop/gate.py merge --id <G....>` merges an ACCEPT with `--no-ff`
  after a PASS review and records the gate numbers in the merge commit.
  `python loop/gate.py stopcheck` tells whether the stopping rule holds.
- Profile with `xcrun xctrace record --template 'Metal System Trace'` under
  the GPU lock, or with synchronized timers. Never trust one run.

## Loop

The main session orchestrates: it keeps the ledger, picks hypotheses, runs
the gate and merges. Subagents implement. Only the main session runs
`gate.py check`, `merge`, `stopcheck` and `audit`, and quotes each final
`GATE` line in the conversation; the goal is judged from those lines.

Start long gate commands detached, so no tool timeout kills them:
`nohup python loop/gate.py check ... > ~/.iris-loop/notes/out/<slug>.log 2>&1 &`.
Wait for the log's last line to start with `GATE` (a Monitor until-loop, or
a check every ten minutes), then quote that line.

1. **Measure first.** Before the first hypothesis for a workload, record
   where its request time goes (text encoding, load, each block type,
   matmul, attention, elementwise, host gaps) in `~/.iris-loop/notes/LEDGER.md`.
2. **One hypothesis per candidate.** Each lives on its own branch in its own
   worktree: `git worktree add ../iris-wt/<slug> -b opt/<slug> opt/loop`.
   Up to four implementing subagents at once; the GPU queue is serial.
3. Implementers write the change and its tests, run CPU tests and short
   micro-benchmarks, commit, and return the worktree path with the numbers.
   The main session then queues a quick or full check. A hypothesis the
   micro-benchmark does not support is dropped before the gate, with the
   numbers in the ledger.
4. **After ACCEPT**, spawn `code-reviewer` with the gate id and nothing else.
   On PASS, `gate.py merge`. On FAIL, fix on the same branch and check again.
5. **After a merge**, every candidate in flight rebases onto `opt/loop` and
   is checked again; its earlier verdict no longer counts.
6. **After every verdict**, update the ledger: hypothesis, category, gate
   id, verdict, numbers, and what it teaches. Do not retry a rejected idea
   without new evidence, and say what is new.
7. **ERROR** with `HALT` in its reason, three NOISY verdicts in a row, or
   three ERRORs in a row: write `LOOP HALTED: <reason>` in the conversation
   and wait for the person. Any other ERROR: read
   `~/.iris-loop/runs/<id>/error.txt`, fix the cause in the candidate, check
   again.

### Ledger format

The stopping check parses one table per gate workload: `## original-10`,
`## turbo-repeat` and `## turbo-new`.

```
## original-10
| component | share % | status | evidence |
|---|---|---|---|
| trunk matmul | 61 | at-bound | measurements/original-matmul-roofline.md |
| attention | 20 | exhausted | G0004 G0009 |
| text encoding | 4 | open | |
```

Shares are of that workload's warm request and sum to about 100. Rows of
3% or more must be `at-bound`, citing a measurement file under
`~/.iris-loop/notes/` that shows a measured hardware limit, or `exhausted`,
citing two full, correct, rejected checks for that model. Rows under 3% may
add up to 10% at most.

### Stopping

Stop only when `gate.py stopcheck` prints `GATE STOPCHECK MET` after the last
merge, and a fresh `loop-auditor` subagent ends with `AUDIT: AGREE`. If it
disagrees, its hypotheses go into the ledger and the loop continues. Then
run `gate.py audit`. It must end with `GATE AUDIT PASS`; on FAIL, fix what
it names and repeat the stopcheck and audit. Write
`benchmarks/optimization-loop-<date>.md` in the style of the existing
records (conditions, ranges, exactness, what was rejected, no claims beyond
the evidence), update the README performance section only with audited
numbers, and open a pull request from `opt/loop` to `main`. Do not merge it.

## Starting points

Read `benchmarks/optimization-2026-10-18.md` first; its rejected list is
evidence, not a to-do list. These are hypotheses, not findings:

- Turbo changed prompts cost about 4 s more than repeated ones. Qwen is
  streamed from disk layer by layer per request. Measure the split first;
  keeping layers resident when memory allows is exact.
- Original attention is about 22% of a block. A GQA-aware Metal attention
  kernel was deferred for lack of counters, not rejected on evidence.
- Original reads the 12 GB FP32 checkpoint and converts it on every load.
- Remaining exact Turbo fusions: sigmoid attention gates (single- and
  dual-stream), extra casts and copies, per-step host synchronization.
- Peak memory near 10 GiB decides whether 16 GB Macs work at all.

## Code rules

The code must stay something a person can maintain.

- Comments say why, not what. A comment or docstring of eight or more words
  is allowed only when it records what the code cannot: a rounding point, a
  measured number, a hardware or driver limit, an invariant a test locks.
  Never restate the code, narrate the change, or mention the loop or gate.
  The gate lists every such added comment; the reviewer judges each one.
- No dead code, unused parameters, speculative options, one-use wrappers,
  broad `except`, silent fallbacks, debug output or unrelated reformatting.
- Existing tests keep their assertions. Change one only when it pinned an
  internal detail the optimization removes, and say so in the commit.
- A second implementation is allowed only as a `*_reference` that tests
  compare against. Every new Metal kernel has one, and a test on random
  inputs that states exact equality or its tolerance.
- Follow the existing style: argument checks like `_require`, names of the
  surrounding code, `ruff check` and `ruff format`.
- **Speed outranks elegance.** Hand-fused kernels, manual layouts,
  shape-specialized or unrolled paths and duplicated hot code are welcome
  when the gate shows they pay. Give them one short line of why.
- Commit messages: what changed and the micro-benchmark numbers behind it.
  The gate writes its own numbers into the merge commit.
