# Original and official Turbo performance — 2026-10-14

Measurements on Apple M5, 24 GiB unified memory, macOS 27.0.1,
PyTorch 2.14.1, Python 3.14.8 and Transformers 5.19.0, on AC power.
The baseline is the incoming working tree, including the uncommitted official
Turbo/gallery integration, based on `52075be`. A 296-file source snapshot and
SHA-256 manifest are retained under `.scratch/perf-20261014/baseline/` and
`baseline-files.json`; comparisons do not use the older Git HEAD as Turbo's baseline.

## Conditions and scope

GPU work was serialized. The prior idle test server was stopped during engine
measurements to avoid two resident checkpoints. Other applications were left
running. Driver-reported background GPU utilization was approximately 66–76%;
no thermal warning was reported. These are observations on this busy machine,
not idle-machine or cross-device guarantees. Absolute times drifted during the
session, so optimization decisions use alternating comparisons. Global paging
counters increased during some load/switch tests; the final warm Turbo ABBA
comparison had no additional pageouts or swapouts.

After the user stopped other GPU work, both test app windows and the idle engine
were closed for a bounded follow-up. Driver-reported background utilization fell
to 22%. The lower-load result is reported separately below; it cannot establish
which application caused the earlier load. Only the latest test app and one
local engine were restarted afterward.

Inputs: 1024 × 1024, fox prompt and seed 0; Original uses 20 steps, CFG 3 and
an empty negative prompt. Turbo uses its official four sigmas
`1, .938, .833, .625`, no CFG, BF16 Linear storage and autocast, FP32 non-linear
parameters and flow state. No sampler arithmetic, weights or precision changed.
Cold means a fresh pipeline/process with local files already downloaded; OS file
caches were not flushed. Complete requests end at the CPU RGB image, excluding
Python imports, downloads and PNG file writing.

The [previous validation](turbo-validation-2026-10-13.md) identifies the pinned
upstream, weight and Qwen revisions. Original checkpoint SHA-256 is
`a90e1f4c197bac8491f82078def68884b6a3527ea7fc97a693e3e367372aa80f`;
Turbo is `7148dbba3117567e34c86fadaae6e10325feb2c0ff3f43fe4938c15c75b356e1`.
No weights were downloaded, converted, copied or deleted in this optimization.

## Accepted changes

1. Adapt Turbo's text once per request and evaluate each shared modulation core
   once per timestep. The text and timestep inputs are identical where reused;
   each block still adds its own bias. Text tokens live only for the current
   request, and modulation only for the current forward. No cross-step image
   activation reuse or persistent GPU cache is introduced. Training and ordinary
   upstream forward calls retain their original path. Original's Metal path
   already performs these optimizations.
2. Keep each generation pipeline's bounded CPU prompt cache when switching
   Original/Turbo. The previous model and adapted GPU cache are still unloaded
   first. The old implementation retained Original's pipeline but recreated
   Turbo's, unnecessarily repeating Qwen encoding on every return to Turbo.

Final production implementation, warm complete Turbo requests, three blocks
(ABBA, BAAB, ABBA), two observations per implementation in each block:

| Block | Baseline mean | Candidate mean | Reduction | Baseline A/A spread |
|---|---:|---:|---:|---:|
| 1 | 22.602 s | 22.207 s | 1.75% | 0.292 s |
| 2 | 22.347 s | 21.610 s | 3.30% | 0.074 s |
| 3 | 22.240 s | 21.688 s | 2.48% | 0.126 s |
| All six requests per implementation | 22.396 s | 21.835 s | 2.51% | — |

All twelve images are bitwise equal to the official reference pixels. Peak
physical memory was 8.920–8.928 GiB, including the interpreter and Metal buffers;
there is no meaningful peak-memory increase. Explicit unload returned live MPS
allocation to zero. An earlier prototype also won all three ABBA blocks
(~0.6–0.8 s), with identical pixels; it was replaced by the small explicit
inference API, not retained as monkeypatches or a general caching framework.

The lower-load follow-up used the same saved inputs and production harness,
with two blocks (ABBA, BAAB), four requests per implementation:

| Block | Baseline mean | Candidate mean | Reduction | Baseline A/A spread |
|---|---:|---:|---:|---:|
| 1 | 12.685 s | 12.468 s | 1.71% | 0.131 s |
| 2 | 12.704 s | 12.359 s | 2.72% | 0.032 s |
| All four requests per implementation | 12.695 s | 12.414 s | 2.22% | — |

All eight images were bitwise identical to the reference. Physical peaks were
8.913–8.920 GiB; unload again left zero live MPS allocation. Pageouts stayed
unchanged; global swapouts increased by 6,157 pages, so this is not a claim of
zero system activity. Both blocks exceeded their baseline A/A spread. This
confirms the direction of the earlier result with substantially lower load.

## Baseline bottlenecks

Separate forward microprofiles use the same saved states/noise, timestep 625,
three warm calls per batch. B=2 duplicates the positive input for controlled
shape measurement; it is not a two-prompt quality comparison. Original B=2 is
representative of the CFG batch shape, while Turbo normally uses B=1.

| Model | B=1 median forward | B=2 median forward | DiT load |
|---|---:|---:|---:|
| Original | 2.737 s | 5.144 s | 9.397 s |
| Turbo baseline | 4.682 s | 9.051 s | 4.232 s |

A separate instrumented call measures MPS event spans around Python operations.
These include scheduling/dispatch under background load; they are not isolated
hardware shader durations. Some event pairs did not expose timestamps, so totals
below are available-event lower bounds, not a complete additive decomposition.

| Operation | Original B1 | Original B2 | Turbo B1 | Turbo B2 |
|---|---:|---:|---:|---:|
| GEMM / `F.linear` | 1527 ms | 2891 ms | 1701 ms | 3632 ms |
| SDPA | 488 ms | 975 ms | 519 ms | 1117 ms |
| Metal norm/modulation | 38 ms | 130 ms | — | — |
| Metal Q/K norm + RoPE | 152 ms | 196 ms | — | — |
| Metal gated residual/norm | 62 ms | 113 ms | — | — |
| Metal pixel MLP | 21 ms | 46 ms | — | — |
| Metal pixel RMS | 0.6 ms | 5.3 ms | — | — |

Original has 283 linear calls and 28 SDPA calls; Turbo baseline has 341 and 32.
Original GEMM spans are missing 3/2 events for B1/B2; norm/modulation B1 is missing
4 and residual/norm B1 is missing 2. Turbo B2 GEMM is missing 5. Turbo uses
upstream PyTorch operations, not this engine's custom fused Metal kernels.
The text adapter's inclusive Turbo B1/B2 spans were 130/258 ms; these overlap
linear and attention totals. Original remains dominated by GEMM and SDPA.

## Rejected or unattempted paths

A small alternating native `F.rms_norm` experiment was faster than the explicit
FP32 reduction on representative trunk, Q/K and pixel shapes, but changed
rounding: BF16 maximum absolute differences were 0.00769, 0.02007 and 0.02202;
FP32 differences reached 2.86e-6. It was rejected at the numerical gate, without
claiming full-image quality or end-to-end speedup. No production code uses it.

The installed PyTorch was already 2.14.1, the version used in the previous audit.
Its native Metal changes are already active. The current MPS Linear source does
not expose the historical `PYTORCH_MPS_PREFER_METAL` switch. No unverified tuning
flag, custom GEMM, packed projection, GELU lookup, CFG approximation, block cache,
or previously unsuccessful asynchronous loading scheme was added.

## Quality boundaries

Saved official FP32-master/BF16-autocast MPS references from the previous audit
were reused for fox/seed 0, FRESH BREAD storefront/seed 1, and violin-maker/seed 2.
With identical raw encoder states, initial CPU noise, checkpoint and recipe,
**all four velocities, final tensor and RGB pixels match bitwise for every case**.
Images were directly inspected for typography, face/hands, composition and detail.
The portrait retains the extra hand/arm present in the official reference.
This demonstrates preservation, not anatomical correctness or CUDA parity.

Raw evidence and temporary measurement programs are in
`.scratch/perf-20261014/`: `forward-*.json`, `production-pairs.json`,
`turbo-invariants-followup.json`, `lifecycle.json`, `quality-turbo.json`,
`followup-production-pairs.json`, `switch-images.json`, `gui-results.json`,
`rms-probe.json`, and generated PNGs. Large fixtures and experimental harnesses
are deliberately not added to the package.

## Step and model lifecycle measurements

A later paired real-checkpoint forward comparison (AB, BA, AB for each batch)
measured the recurring Turbo step, with request-local adapted text prepared once
outside that step. The first request still pays this adapter cost (177 ms B1,
282 ms B2 in this diagnostic). All compared velocity tensors were bitwise equal.

| Turbo step | Baseline median | Candidate median | Candidate MPS event span |
|---|---:|---:|---:|
| B1 | 5.575 s | 5.374 s | 5374 ms |
| B2 | 9.729 s | 9.381 s | 9381 ms |

These are later-session times and must not be compared with the earlier forward
microprofile to calculate a speedup. B2 is a diagnostic supported shape, not a
change to Turbo's B1 default. The forward harness released all live MPS memory.

Three fresh manager pairs (AB, BA, AB), each switching
Turbo → Original → Turbo → Original, independently tested pipeline retention:

| Return to model, cached prompt | Baseline ready times | Candidate ready times |
|---|---|---|
| Turbo | 6.621, 6.637, 6.691 s | 3.448, 3.309, 3.542 s |
| Original | 10.419, 9.139, 8.123 s | 11.427, 9.987, 8.513 s |

Ready includes previous-model release, prompt encoding and new DiT loading,
not image generation. Returning to Turbo no longer repeats 2.94–3.15 s of Qwen
work; retained-cache lookup took 35–88 microseconds (Original: 17–20 microseconds).
Original already retained
its cache and shows no load/switch improvement. Loading varied during global
paging. Every switch confirmed the old engine weak reference was dead and live
MPS allocation was zero before loading the next. Final unload: allocated 0,
driver 720,896 bytes, process physical footprint 0.559 GiB.

The tradeoff is keeping Turbo's bounded CPU cache (at most 16 prompt entries,
approximately 281 MiB for this released encoder's BF16 states) in addition to
Original's existing bounded cache and tokenizer. No second DiT or Qwen GPU
weights are retained. Checkpoint storage remains unchanged.

A separate complete switch-and-generate check used the optimized conditioning
path on both sides, isolating CPU prompt-cache retention. Two AB/BA pairs measured
returning from Original to Turbo and producing the cached fox image:

| Pair | Baseline | Candidate |
|---|---:|---:|
| AB | 29.151 s | 26.345 s |
| BA | 29.195 s | 25.668 s |
| Mean | 29.173 s | 26.007 s |

All four pixel arrays matched the reference. Peak physical footprint was
8.973–9.090 GiB. Global paging increased during this check, so the 3.17 s mean
reduction is supported by the independent preparation measurements above, not
treated as a universal percentage. Unload left zero live MPS allocation and
1,769,472 driver bytes. This longer process still reported 6.048 GiB physical
footprint after cleanup; zero live GPU tensors does not mean zero process memory.
The separate preparation-only process ended at 0.559 GiB. Long-duration process
memory behavior was not established by these bounded runs.

Fresh-process Turbo requests, one cold/warm pair per implementation:

| Phase | Baseline | Candidate |
|---|---:|---:|
| Complete first generation | 30.569 s | 29.635 s |
| Complete cached repeat | 22.240 s | 22.059 s |
| Tokenizer setup | 0.818 s | 0.960 s |
| New prompt encoding, including layer transfer | 3.410 s | 3.162 s |
| DiT loading | 3.787 s | 3.592 s |
| Maximum physical footprint across both requests | 9.967 GiB | 10.030 GiB |

These single pairs are lifecycle observations, not evidence of faster loading
or a guaranteed cold-start reduction. Allocator high-water marks differ from
steady warm ABBA measurements because these processes also ran Qwen. Both
unloaded to allocated 0 and driver 1,769,472 bytes. Pixels matched the saved
reference on every request. Loading and Qwen arithmetic were not modified.

Original's unchanged 20-step path was also generated end to end again:

| Original phase | Incoming baseline | Final tree |
|---|---:|---:|
| Complete first request | 120.127 s | 114.884 s |
| Complete cached repeat | 100.522 s | 99.280 s |
| Tokenizer setup | 1.211 s | 0.966 s |
| New prompt encoding | 3.694 s | 3.225 s |
| DiT load | 10.534 s | 8.751 s |
| Peak physical footprint | 10.006 GiB | 10.013 GiB |

These are unpaired single observations, with global paging during the final run;
**no Original acceleration is claimed**. Warm completion intervals were about
4.91–4.99 s per step in the final run. Both first and repeat images match the
incoming baseline pixels bitwise (`1b1488cb81cae7fda454830036779b27a5090262db02a508ab8df59b504e1a5e`)
and were visually inspected. Final unload returned allocated bytes to zero.

## Validation

- CPU suite: 410 passed, 21 GPU cases skipped, six existing MPS pin-memory warnings.
- Relevant on-device engine suite: 30 passed, including all new B1/B2 BF16 MPS
  exact-conditioning cases, rectangular shapes, read-only prepared tokens, distinct
  prompts/timesteps and the existing precision/Metal-kernel gates. The unchanged
  text encoder's earlier MPS results were reused; full requests here also exercised
  streamed Qwen on the real device.
- Swift: 74 passed, including opt-in HTTP cases with the CPU stand-in, native search
  cancel without a field editor, shader compilation and actual Metal drawing.
  The new background test waits for the window server's visibility notification
  before rendering; checking synchronously immediately after ordering the window
  initially failed. No assertion or visual criterion was relaxed.
- Ruff lint and format checks, Debug tests/build and Release build passed.
- The upstream training/default forward tests still pass; prepared conditioning
  rejects training or gradient-enabled use. Turbo cancellation does not retain
  prepared text for the following request.

### Native app checks

The Debug app used a separate copy of the prior test library. On the old app,
searching for Tokyo, selecting an image and then clicking the native search
cancel button left four results even though the field was empty. The new app
returned to all 26 initial images using the same actions. Command-F also focuses
the native search field.

Real local-engine generation completed for Turbo (4 steps, seed 0) and Original
(10 steps, CFG 3, seed 0). Both the running tile and the gallery's background
flower visibly changed during sampling; moving the pointer through the gallery
continued to affect the background after the former three-second idle deadline.
The renderer regression also verifies idle and disabled-animation pausing.

Command-period stopped Turbo after its first reported step. Try Again completed
with the same image and saved settings. All four newly completed GUI images
matched their prior same-setting images bitwise; model, seed, sampler and
checkpoint metadata were preserved. The copied library contains the stopped
entry as history. Existing library entries and all 41 protected source-library
files were unchanged. Restart restored all 30 completed images plus stopped
entries, with one app instance and one ready local engine. Package build succeeded;
wheel/sdist contain no checkpoints,
virtual environment, scratch directory or user library.

Remaining performance opportunities are mostly native MPS GEMM/SDPA and small
Turbo FP32 normalization/elementwise dispatches. Any different reduction/layout
must preserve the existing numerical gates and demonstrate a complete-request
win. No CUDA reference execution, other Apple GPU/RAM configuration, long memory
soak, signing, notarization or public release was performed.
