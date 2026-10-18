# Exactness-first optimization — 2026-10-18

Implementation on top of `1bf1d73` (v0.5.0). Captured results were bit-identical
to the unchanged source; this record lists the
evidence, the measured effect and what was rejected. Raw results, tools and the
source-only baseline snapshot (with SHA-256 manifest) are outside the
repository in `iris-opt-20261018/` next to the checkout.

The independent release audit found two limitations in the original wrapper:
it injected saved Qwen states even for the first request, and replacing those
tensors before every repetition invalidated the adapted-text cache. Thus
"cold" below excludes text encoding, and the repeated runs did not establish
cross-request cache hits. The release audit repeats both checks with preserved
fixture tensors and separately measures requests with the real text encoder.

## Environment and conditions

M5 MacBook Pro (Mac17,2, 10-core CPU and GPU, 24 GiB), macOS 27.0.1 (26A434),
Xcode 27.1 (27A9275, SDK 27.0), Python 3.14.8, PyTorch 2.14.1 (`5c48869`),
Transformers 5.19.0, safetensors 0.8.0, huggingface_hub 1.33.0, Pillow 12.3.0,
NumPy 2.5.3, no MPS or PyTorch environment variables, AC power, Low Power Mode
off, no thermal warning. Checkpoints: Original `a90e1f4c…` @ `7445443`, Turbo
`7148dbba…` @ `adc05b9`, Qwen @ `ebb281e`, all local.

The machine was not quiet. Two Unity editors (about 12 GiB together), Chrome
and the Claude app stayed open; system swap grew from 5.6 to 20.5 GiB and most
model processes recorded hundreds of thousands of swap-outs. Midway, Metal
shared events began failing system-wide (`encodeSignalEvent` received a nil
event in a three-line fresh process), so later full-request runs replaced
`StepReporter`'s MPS events with immediate progress reports in **both** trees.
That changes no arithmetic. Timing conclusions rely on paired, alternating
runs; absolute times are not comparable with earlier records.

Later the same day the Unity editors were closed (swap 2.2–8 GiB; Chrome and
the Claude app still open) and MPS events worked again. The full Python suite
then passed with MPS (496 tests, including the two event-dependent text
tests), and the Original and Turbo runs marked "with events" below used the
unmodified `StepReporter`.

## Exactness

Captures hash every model output, the first `clamp(-1, 1)` input (the
pre-clamp sample) and the pixels in separate processes for baseline and
candidate, with imported module paths checked:

| Workload | Result |
|---|---|
| A/A: baseline twice, Turbo fox/storefront seed 1/portrait and Original 10-step fox/storefront | identical; pixels match the historical official Turbo hashes (`bf9f0d31…`, `e478e758…`) |
| Turbo, 3 fixtures × 2 requests (second request uses the new cross-request cache) | all 4 step outputs, pre-clamp sample, pixels and metadata identical |
| Original 10 steps (CFG batch 2), 2 fixtures × 2 requests | all 10 step outputs, pre-clamp sample, pixels and metadata identical |
| Turbo at 512², 768×1024, 1024², 1024×768, 1280×768, 640², 896×1152 | pixels identical |
| Restoration 1024² → 4096² (49 tiles), 384×512 (6 tiles, colour fix on and off); depth | identical |
| Tiny released topology: CPU and MPS, batch 1–3, prepared and plain paths, FP32-master autocast, FP32, gradients | identical |

The Qwen streamer reproduced the saved fox fixture's states bitwise.

## Turbo

Fresh processes, fox, 1024², one cold and three warm requests each, ABBA then
BAAB:

| | Baseline | Candidate |
|---|---:|---:|
| Warm median (12 requests) | 14.06 s (12.80–15.49) | **11.16 s** (10.44–12.53) |
| ABBA / BAAB warm medians | 13.10 / 14.73 s | 10.80 / 11.60 s |
| Cold median (4) | 19.88 s | 18.05 s |
| Peak physical footprint | 9.95 GiB | 8.95 GiB |
| Live MPS allocation after unload | 0 | 0 |

Within each block every candidate warm request was faster than every baseline
warm request. One more ABBA block with events and no swap-outs confirmed it:
warm 12.86 s (12.64–13.11) → 10.24 s (10.17–10.49), −20%, cold 16.67 →
14.10 s, peak 9.92 → 8.92 GiB, same separation. Forward-level alternating runs on the real model (fixed inputs,
no hooks) attribute the gain:

| Change | Median paired difference per forward | Faster rounds |
|---|---:|---:|
| Fused norm/modulation and gated-residual tails | −926 ms of 4.54 s | 10/10 |
| One FP32 copy in RMSNorm (O01) | −51 ms | 14/16 |
| Shared autocast Linear operands (O02) | −69 ms | 14/16 |
| Joined single-stream tokens and RoPE (O12/O13) | −45 ms | 23/40, inconclusive |

The fused tails keep PyTorch's `rsqrt(mean(x²) + eps)`: a compiled reduction
changed 1,707 of 4,396 rows, while the elementwise tail compiled or
handwritten was bitwise equal. Components of O12/O13 are resolved: each removed
join costs 0.83 ms and joint RoPE takes 1.0 instead of 2.0 ms per block.
Reusing adapted text across requests saves the 85 ms adapter on repeated
prompts.

### Pixel-stage tails (later the same evening)

Two more exact tails, measured with events working and no swap-outs. Both keep
PyTorch's `rsqrt(mean(x²) + eps)` and round wherever the eager expression
does. `norm_operand` writes the PiT and final RMSNorm directly as the BF16
operand that autocast would hand the next Linear. `modulated_residual` fuses
PiT's post-modulated update `x + (branch * (1 + scale) + shift)` on BF16 pixels.

| Change (1024²) | Component | Paired difference per 2.5 s forward | Faster rounds |
|---|---:|---:|---:|
| RMSNorm into the BF16 operand, 9 per forward | 10.18 → 3.34 ms | −70 ms | 15/16 |
| PiT post-modulated residual, 8 per forward | 5.79 → 1.69 ms | −33 ms | 30/40 |

Fox requests against the tree without these two, ABBA then BAAB: warm median
10.38 s (10.00–11.05) → 10.04 s (9.83–10.35), every candidate process median
below every baseline one, block medians −0.09 and −1.01 s. Captures of fox,
typography, portrait and hands against the untouched baseline matched at
every step, the pre-clamp sample and the pixels. Dropping any one rounding
point changes 1.9–4.4 million of 16.7 million elements, which the new tests
catch. The suite passes with MPS (499 tests).

## Original

The Original path changed only in copies and kernel launches. Ten-step warm
requests in ABBA and BAAB blocks under heavy swap were inconclusive (baseline
45.78 s, 42.98–49.93; candidate 46.73 s, 45.28–50.49; n = 8 each). Repeated
with events: baseline 40.40 s (39.58–41.68), candidate 40.41 s (39.69–41.45),
n = 8 each, block differences +0.56 and −0.20 s, 0.19–0.43 million swap-outs
per process, identical pixels — no whole-request difference either way.
Components: strided modulation reads save 19 µs per norm/residual pair at
batch 2 (33/40 rounds), building guidance inputs once saves 110 µs per guided
step (40/40), and image conversion takes 1.9 instead of 3.7 ms per 1024² image.

## Other measurements

- Restoration colour fix: 15.0 → 10.2 ms (1024²), 51.3 → 36.9 ms (2048²),
  102.5 → 74.7 ms (2048 × 4096), 16/16 rounds each.
- Status long polling with the stand-in engine: completion observed after
  1.0 ms instead of 75 ms (medians of 24 jobs each).
- Per-size caches stay at four tables; across seven sizes live MPS allocation
  levelled off instead of growing with each new size.

## Rejected, inconclusive or deferred

| Item | Finding |
|---|---|
| Cached timestep frequencies | exact, 21 → 11 µs per forward; not worth extra state |
| Request metadata caching | 0.12 ms per request |
| CLI `--help` imports | Initially deferred over test monkeypatch aliases; the release audit moved patches to the owning modules and measured lazy imports separately. |
| Kernel launch geometry | exact; `rms_norm_modulate` −3% at batch 2 but residual and batch 1 neutral |
| Inductor quantization / RMSNorm | exact but 2.8 s compile per size / reduction not exact |
| Image-only final queries | exact, 1.1 ms per forward |
| Fused sigmoid attention gates | exact (Metal `1 / (1 + exp(-x))` equals MPS sigmoid on all 65,536 BF16 inputs); 1.12 → 0.77 ms per single-stream gate, about 8 ms per forward expected; forward −6 ms, 22/40, unresolved; the dual-stream half also needs gate logits in `JointAttention` |
| Joint RoPE table once per forward | 0.21 ms per join; removing 15 joins +1.7 ms (19/40), adding 16 +0.9 ms (20/40): no effect |
| Direct un-patchify instead of `F.fold` | loses fold's −0 → +0; 0.46 ms |
| Packed projections | exact, no speed difference |
| FlexAttention | not exact, 128× slower |
| Precompiled metallib | library builds in 7 ms |
| Host-alias weight loading | exact; 0.45 → 0.01 ms per small tensor but slower for an 80 MiB tensor; full-load comparison deferred under memory pressure |
| Event pooling, custom GEMM/SDPA | deferred: no counters, events unavailable |
| One-layer Qwen lookahead | with events: states bitwise identical on MPS (5 prompts, 8 processes) and text tests pass, but uncached encoding 1.75 → 1.65 s median (n = 56 each, Mann–Whitney p = 0.20, per-prompt signs mixed); left reverted |
| Queue byte bound | worst case about 1.5 GiB of decoded uploads; the app sends one job at a time |
| Cross-model raw text cache | deferred: only helps a first visit to the other model |

## Independent release audit — 2026-10-19

Reference: `1bf1d7330a179a47545e45cfd15d50b826f76781`. The received source
was preserved before edits, with manifest SHA-256
`66b3f79a1ad3d3119f8f6d2816c19e4720c9da0e8d2ac413f2e435a75f40854c`.
On the same M5, OS and numerical dependencies listed above, separate-process
captures checked the imported source hashes and full fixture sets. Turbo's
three fixtures × two requests matched all four model outputs, pre-clamp
FP32 values, RGB pixels and metadata; Original's two fixtures × two requests
matched all ten outputs and the same final contracts. Turbo A/A also matched.
Hashes cover contiguous tensor bytes with dtype, shape and finiteness checks,
including signed zero; this is finite fixture evidence, not universal proof.

An additional Turbo ABBA block used the real text encoder and unmodified MPS
events, with fox, typography and portrait prompts repeated twice per process.
The first request includes model loading and text setup; a changed prompt
includes uncached text encoding. These are pipeline request times through
image conversion and metadata, excluding HTTP transport and GUI persistence.

| Request | Reference median (range) | Received median (range) | Samples per variant |
|---|---:|---:|---:|
| First | 20.02 s (19.75–20.30) | 17.18 s (16.87–17.49) | 2 |
| Repeated prompt | 12.52 s (12.09–12.73) | 9.81 s (9.55–10.44) | 6 |
| Changed prompt | 18.30 s (16.68–19.49) | 14.05 s (13.47–15.32) | 4 |

Repeated-prompt latency fell 21.6%; the observed ranges did not overlap.
There were only two independent processes per variant, with background GPU
utilization of 0–8.75% and 51,744–248,880 system swap-outs per process.
Changed-prompt and first-request estimates therefore have substantial system
noise. Peak whole-request physical footprint was 10.08 versus 10.04 GiB: this
run does not establish a peak-memory improvement. Live MPS allocations were
zero after unload. Pixels and metadata matched for every paired request.

Original's corresponding ABBA block used ten steps and two prompts repeated
twice per process. Repeated-request medians were 39.559 s versus 39.586 s
(four samples each; ranges 39.318–40.745 and 39.391–39.890 s). First-request
medians were 51.625 versus 51.604 s, and changed-prompt medians were 43.318
versus 43.516 s (two samples each). Peak physical footprint was 10.097 versus
10.096 GiB. This does not establish an Original speed or memory improvement,
and does not measure the 100-step default. All pixels and metadata matched.

The CLI import decision was revisited because monkeypatch aliases were test
details, not a public API. Moving patches to the owning modules permits lazy
inference imports without changing generation. Seven fresh processes per
variant, in alternating order, produced identical help output:

| Complete command | Received median | Final median | Reduction |
|---|---:|---:|---:|
| `--help` | 654 ms | 77 ms | 88.2% |
| `serve --help` | 752 ms | 85 ms | 88.7% |
| `depth --help` | 825 ms | 94 ms | 88.6% |
| `upscale --help` | 820 ms | 88 ms | 89.2% |

The ordinary-help ranges were 615–838 ms and 71–83 ms. These are command
startup results, not inference speedups. Raw audit captures and timing inputs
are retained outside the checkout in the sibling `iris-audit-20261019-*`
directory; the release report records the final artifact identities.
