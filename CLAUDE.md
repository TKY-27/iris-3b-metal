# Iris-3B Metal: optimization loop

This checkout is driven by an autonomous optimization loop. The goal is to
make Original and Turbo inference on Apple Silicon faster, or lighter, until
nothing measurable is left. A measurement gate decides; your judgment does
not. Read this whole file before the first change.

## Scope

- In scope: `src/iris_metal/`, the inference paths of `src/iris3b/` (models,
  `nn`, `flow`, `sampling`, `text`), their tests, and new tests.
- `src/iris3b` is shared with the upstream training code. Training behaviour
  must not change, and the full test suite must pass.
- Out of scope: the macOS app, training, the HTTP and CLI interfaces, docs
  (except the final report), dependencies.

## Contracts

These are fixed. The gate enforces them; do not argue with a verdict.

1. **Turbo is bit-exact** with the official BF16 recipe as v0.6.1 runs it:
   pre-clamp sample, pixels, metadata and text encodings of every fixture,
   including second visits (caches) and seven sizes. Any change of
   accumulation order, operand precision or rounding point breaks this.
   Metal FP16 Turbo was already rejected for turning `BREAD` into `BRIEAD`.
2. **Original may change its output only if it does not move away from
   FP32.** Velocity error against the upstream FP32 forward, and trajectory
   error against FP32 sampling, may not rise on average (per-item slack in
   `loop/decide.py`). The OCR words of the typography fixture must survive.
   Bit-identical Original output passes without these checks.
3. **Text encodings stay bit-identical** for both models, for the same
   sequence of calls.
4. **Memory.** Peak memory may rise only when memory is plentiful. Read the
   budget through one function that honours `IRIS_AVAILABLE_MEMORY_GIB`;
   with `10` the peak must stay within 0.25 GiB of the reference.
5. **Interfaces stay.** `IrisPipeline.__call__`, `encode`, `last_metadata`,
   metadata keys, CLI and HTTP API are unchanged. For Original,
   `pipe.engine.encode_text(states, mask)` and `pipe.engine(x, t, text)`
   returning FP32 velocity remain; the gate probes them. The final sample
   still passes through `Tensor.clamp(-1, 1)`, where the gate records it.
6. **No benchmark gaming.** Never cache generated samples or images between
   requests, never detect or special-case the gate's prompts, sizes, seeds or
   workloads, never move work outside the timed request into imports.
7. Depth and restoration reuse `MetalIris`. The gate flags changed outputs
   for a person to review; they must keep working.

## Harness (read-only)

`loop/`, `.claude/`, this file, `benchmarks/prompts.json` and
`~/.iris-loop/` except `~/.iris-loop/notes/` are frozen. A hook blocks
edits; the gate re-hashes everything before each verdict and refuses to run
if anything changed.

- One GPU job at a time, machine-wide. Anything that may touch MPS runs as
  `python loop/gpu.py run -- <command>`. CPU-only work is prefixed
  `IRIS_CPU_ONLY=1`; CPU test runs use `IRIS_SKIP_GPU_TESTS=1`.
- Keep GPU micro-benchmarks under five minutes; the gate queues behind them.
- `python loop/gate.py check --candidate <worktree> --category <c> --targets original,turbo --note "<hypothesis>"`
  runs tests, correctness and paired ABBA timing (about an hour). `--quick`
  times only the targeted workloads and can only return PROMISING.
- Verdicts: ACCEPT, PROMISING, NO_GAIN, REJECT, NOISY, ERROR. The last
  output line starts with `GATE`.
- `python loop/gate.py merge --id <G....>` fast-forwards the integration
  branch after a PASS review. `python loop/gate.py stopcheck` tells whether
  the stopping rule holds.
- Profile with `xcrun xctrace record --template 'Metal System Trace'` under
  the GPU lock, or with synchronized timers. Never trust one run.

## Loop

The main session orchestrates: it keeps the ledger, picks hypotheses,
runs the gate and merges. Subagents implement. Only the main session runs
`gate.py check`, `merge`, `stopcheck` and `audit`, as background commands,
and quotes each final `GATE` line in the conversation; the goal is judged
from those lines.

1. **Measure first.** Before the first hypothesis for a model, record where
   its request time goes (text encoding, load, each block type, matmul,
   attention, elementwise, host gaps) in `~/.iris-loop/notes/LEDGER.md`.
2. **One hypothesis per candidate.** Each lives on its own branch in its own
   worktree: `git worktree add ../iris-wt/<slug> -b opt/<slug> opt/loop`.
   Up to four implementing subagents at once; the GPU queue is serial.
3. Implementers write the change and its tests, run CPU tests and short
   micro-benchmarks under the lock, commit, and return the worktree path
   with the numbers. The main session then queues a quick or full check.
   A hypothesis that the micro-benchmark does not support is dropped
   before it reaches the gate, with the numbers in the ledger.
4. **After ACCEPT**, spawn `code-reviewer` with the gate id and nothing else.
   On PASS, `gate.py merge`. On FAIL, fix on the same branch and check again.
5. **After a merge**, every candidate in flight rebases onto `opt/loop` and
   is checked again; its earlier verdict no longer counts.
6. **After every verdict**, update the ledger: hypothesis, category, gate
   id, verdict, numbers, and what it teaches. Do not retry a rejected idea
   without new evidence, and say what is new.
7. NOISY means the Mac was busy, hot or on battery. After three NOISY
   verdicts in a row, stop and ask the person.

### Ledger format

The stopping check parses one table per model:

```
## original
| component | share % | status | evidence |
|---|---|---|---|
| trunk matmul | 61 | at-bound | measurements/original-matmul-roofline.md |
| attention | 20 | exhausted | G0004 G0009 |
| text encoding | 4 | open | |
```

`at-bound` cites a measurement file under `~/.iris-loop/notes/` showing the
component at a measured hardware limit; `exhausted` cites at least two
rejected gate ids for that model. Shares are of the warm request and sum to
about 100.

### Stopping

Stop only when `gate.py stopcheck` prints `GATE STOPCHECK MET` and a fresh
`loop-auditor` subagent ends with `AUDIT: AGREE`. If it disagrees, its
hypotheses go into the ledger and the loop continues. Then run
`gate.py audit`, write `benchmarks/optimization-loop-<date>.md` in the style
of the existing records (conditions, ranges, exactness, what was rejected,
no claims beyond the evidence), update the README performance section only
with audited numbers, and open a pull request from `opt/loop` to `main`.
Do not merge it.

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
  when the gate shows they pay. Give them one short line of why, and put
  the measurement in the commit message with the gate id.
- Commit messages: what changed, the measured effect, the gate id.
