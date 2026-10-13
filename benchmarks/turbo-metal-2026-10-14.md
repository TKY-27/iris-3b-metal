# Turbo on the Metal forward — 2026-10-14

**Historical candidate study, not the released path.** The v0.5.0 independent
audit rejected this FP16 candidate for its reproducible storefront spelling
regression. Production Turbo retains the official BF16 recipe. Present-tense
implementation descriptions below refer to the studied candidate. See the
[final measurements and remediation](release-0.5.0.md).

Apple M5 (10-core GPU), 24 GiB unified memory, macOS 27.0.1, Python 3.14.8,
PyTorch 2.14.1, Transformers 5.19.0, AC power. Other applications stayed open
and nothing was terminated; driver-reported background GPU utilization was
0–25% when sampled before the study and was not logged continuously. The
working tree was based on `d95c179`. Weights are the pinned Original and
official Turbo checkpoints of the [previous validation](turbo-validation-2026-10-13.md)
(Turbo SHA-256 `7148dbba…356e1`); nothing was downloaded, converted or deleted.

## Candidate changes (not shipped)

Turbo now runs on the same fused Metal forward as Original (`iris_metal.model.MetalIris`:
FP16 trunk linears and attention, BF16 text adapter and pixel stage, FP32
norms, modulation and residual streams), instead of the upstream `IrisDiT`
forward under BF16 autocast. The official four-step deterministic flow-Euler
sampler (`generate_few_step`, sigmas `1, .938, .833, .625`) and the official
weights are unchanged; only the arithmetic of each forward differs. The
upstream-model changes of `d95c179` (request-local conditioning reuse inside
`IrisDiT.forward`) are no longer needed and were removed, so `src/iris3b`
is back to its upstream form. The prompt cache keeps adapted trunk tokens for
both models, as it already did for Original.

## Why FP16 was investigated

The [2026-10-13 validation](turbo-validation-2026-10-13.md) rejected the Metal
forward for Turbo after one pilot: `MetalIris` with a **BF16 trunk and BF16
attention**, compared against the official BF16 recipe (relative L2 0.36 on the
final tensor). Two things were missing. The pilot used the configuration whose
BF16 query/key rounding Original had already measured as poor (its production
configuration, FP16 trunk and attention, was never run for Turbo), and the
comparison target was itself a BF16 computation, so its distance from a
candidate says nothing about which of the two is closer to the model's
arithmetic. This study adds an FP32 oracle: the upstream forward without
autocast, from the same saved encoder states and CPU noise.

## Numerical study against the FP32 oracle

`scripts/validate_turbo.py` ran `prepare`, `fp32`, `reference` (FP32 master,
BF16 autocast), `upstream-bf16` (the previous production path: BF16 linear
storage, BF16 autocast, bitwise equal to `reference`) and `metal-fp16` (the new
production path) in separate processes, 1024 × 1024, one saved noise and
encoder state per prompt. The three fixtures of the previous validation (fox,
FRESH BREAD storefront, violin maker; seeds 0–2) were reused and five prompts
from `benchmarks/prompts.json` were added (seeds 3–7). Raw results are in
`.scratch/turbo-20261014/<prompt>/`.

Relative L2 distance from the FP32 oracle, first velocity (same input for every
implementation) and final tensor:

| Prompt | Official BF16: step 1 | Official BF16: final | Metal FP16: step 1 | Metal FP16: final |
|---|---:|---:|---:|---:|
| fox (seed 0) | 8.7% | 30.3% | 3.6% | 9.3% |
| FRESH BREAD storefront (1) | 4.4% | 20.3% | 2.0% | 19.1% |
| violin maker portrait (2) | 5.5% | 28.6% | 2.5% | 16.2% |
| potter's hands (3) | 5.5% | 21.0% | 1.0% | 4.7% |
| chalkboard menu (4) | 2.3% | 5.9% | 0.9% | 2.2% |
| tabby cat (5) | 2.9% | 9.9% | 1.1% | 5.8% |
| pocket watches and gears (6) | 6.0% | 24.9% | 1.8% | 13.2% |
| hands on a piano (7) | 3.4% | 18.5% | 1.2% | 5.3% |

Final images against the FP32 oracle's image (LPIPS-Alex, PSNR):

| | Official BF16 path | Metal FP16 path |
|---|---:|---:|
| Mean LPIPS (lower is closer) | 0.0785 | 0.0325 |
| Median LPIPS | 0.0916 | 0.0135 |
| Mean PSNR | 26.2 dB | 33.4 dB |
| Closer to FP32 by LPIPS | — | 8 of 8 prompts |

The two BF16/FP16 paths differ from each other by about as much as the official
path differs from FP32 (mean LPIPS 0.085, 26.0 dB). A four-step trajectory
amplifies any rounding into visible fine-structure differences, so individual
details can go either way: in the storefront fixture the Metal path misspelled
one word that both other paths rendered correctly, while the official path
added garbled text to the window that the oracle does not have. The composition,
subject and lighting were the same in all 24 images (`.scratch/turbo-20261014/sheet-*.png`).
The Metal outputs are closer to the FP32 reference on this bounded set.
This does not establish better images or uniform perceptual quality.

A Metal BF16-trunk variant with FP16 attention (fox only) was further from FP32
than the FP16 trunk (final 12.8% vs 9.3%) and only 0.6 s faster, so the Original
precision plan was kept. FP32 attention was also checked on the fox and storefront fixtures. It did
not fix the storefront spelling, and its storefront warm denoising took
13.8–14.0 s versus 7.7–7.8 s for FP16 attention, so FP16 attention was kept.

**FP16 range.** The largest FP16 trunk linear input over the eight 4-step
trajectories was 956 and the largest output 1936 (`.scratch/turbo-20261014/fp16-range.json`),
about 3% of FP16's range, in line with Original's 2520. Every result passes the
existing non-finite check.

## Timing

Warm four-step denoising from saved inputs in the numerical study (two
repeats per process, same session, uncontrolled background; later prompts
overlapped a GPU test run and are slower on both sides):

| Prompt | Official BF16 path | Metal FP16 path |
|---|---:|---:|
| fox | 12.6 s | 8.5 s |
| storefront | 12.8 s | 7.8 s |
| portrait | 12.8 s | 7.9 s |
| potter's hands | 15.5 s | 9.5 s |
| chalkboard | 14.8 s | 9.2 s |
| cat | 14.1 s | 9.0 s |
| watches | 14.4 s | 8.9 s |
| piano | 13.5 s | 8.1 s |

Production comparison with `scripts/profile_generation.py` (complete requests
through `IrisPipeline`, including metadata attachment but excluding PNG file writing; cold = fresh process with the
checkpoint loaded and the prompt encoded inside the request, warm = cached
prompt), alternating A = committed tree `d95c179` (official BF16 path) and
B = this working tree (Metal path) in fresh processes, order ABBA BAAB:

| Run | Implementation | First request (cold) | Cached prompt (warm) |
|---|---|---:|---:|
| 1 | A official BF16 | 19.49 s | 13.32 s |
| 2 | B Metal FP16 | 14.42 s | 8.08 s |
| 3 | B Metal FP16 | 14.71 s | 8.57 s |
| 4 | A official BF16 | 20.54 s | 14.23 s |
| 5 | B Metal FP16 | 15.13 s | 8.67 s |
| 6 | A official BF16 | 20.49 s | 14.17 s |
| 7 | A official BF16 | 20.46 s | 14.16 s |
| 8 | B Metal FP16 | 15.04 s | 8.80 s |
| Mean | A official BF16 | 20.25 s | 13.97 s |
| Mean | B Metal FP16 | 14.83 s | 8.53 s (−39%) |

Every B run was faster than every A run. Model loading (2.6–2.7 s) and prompt
encoding (2.6–2.7 s) were the same on both sides, so the whole difference is
the four forwards.

Both implementations reproduce their own pixels exactly across processes
(official: `bf9f0d31…`, the hash of the previous audits; Metal: `1b42f0fc…`).
Physical peak memory of the process fell from 10.02 GiB to 7.92 GiB in every pair, because the fused forward materializes fewer FP32 intermediates; unloading left zero live MPS allocation on both sides.

Original's 20-step path is untouched: a fresh-process 1024 × 1024 fox request on the working tree took 97.9 s (cold, prompt encoded inside the request), peaked at 10.01 GiB and produced the same pixels as the previous audits (`1b1488cb…`). No Original speedup is claimed; its remaining time is native MPS GEMM and SDPA at the CFG batch shape, which earlier studies already found at the hardware's measured throughput.

## Validation

- Python: 426 passed with the GPU cases on the M5 (none skipped), and 407 passed with 19 GPU cases skipped under `IRIS_SKIP_GPU_TESTS=1`, as CI runs. The removed upstream-forward tests of `d95c179` went with the removed API; Turbo's pipeline tests now run the toy topology through `MetalIris`.
- Swift: 74 tests, 3 opt-in HTTP tests skipped, 0 failures; Debug test build and Release build succeeded.
- Native app on the real engine (Debug build, isolated library copy, fake engine replaced by `iris-metal serve` from this tree): Turbo generation produced the production pixels (`1b42f0fc…`) in 15 s cold and 11 s after returning from restoration; Original 10 steps was stopped at 3/10 and Try Again completed in 42 s with the pixels of the earlier Original entries; Estimate Depth (12 s) and Restore and Upscale (27 s, 2048 × 2048) ran in between. Search, project views, the model menu, both tuning popovers, the minimum window size and dark appearance were checked by screenshot.
- Ruff lint and format checks passed.

## Not done

No CUDA execution, other Apple chips or memory sizes, long memory soak, or
controlled idle-GPU benchmark. The eight prompts are a numerical study, not a
perceptual user study; the previous validation's three fixtures no longer match
the official recipe bitwise by design, and `scripts/validate_turbo.py reference`
remains the way to reproduce that recipe.
