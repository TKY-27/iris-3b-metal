# Official Turbo validation — 2026-10-13

Apple M5, 24 GiB unified memory, macOS 27.0.1, Python 3.14.8,
PyTorch 2.14.1, Transformers 5.19.0. Battery power, low-power mode off.
These are local observations with background system activity and existing swap,
not controlled idle-machine guarantees. The working tree was based on `52075be`.

## Model and reference

- [Official implementation](https://github.com/speridlabs/iris-3b/tree/cee0c130a62a9c288ad8918f1f7cdf23cc228d69), `generate_few_step`.
- [Official Turbo export](https://huggingface.co/speridlabs/iris-3b/tree/adc05b909fb05d4276d26aae5a52b36406b2ef26/turbo), including `distill.yaml`.
- Checkpoint: 5,975,089,200 bytes, 639 BF16 tensors. SHA-256
  `7148dbba3117567e34c86fadaae6e10325feb2c0ff3f43fe4938c15c75b356e1`,
  verified against the official LFS object digest.
- Four deterministic flow-Euler steps, sigmas `1, .938, .833, .625`;
  embedded guidance, no negative prompt or adjustable CFG/steps.

The production implementation retains official IrisDiT operations under BF16
autocast. Linear parameters are stored in BF16; other parameters and the flow
state remain FP32. Original retains its existing Metal forward and precision.

Saved Qwen states and CPU FP32 noise were shared with the reference on MPS.
At 1024 × 1024, fox (seed 0), FRESH BREAD typography (seed 1) and violin-maker
portrait/hands (seed 2) matched the official FP32-master/BF16-autocast recipe
**bitwise at every velocity, the final tensor and PNG pixels**. This is source
and same-device parity, not a CUDA/H100 comparison. Both portrait outputs have
an extra hand/arm: exact parity does not establish anatomical quality.

The faster existing fused Metal BF16/FP16 path was rejected for Turbo because
of numerical drift (native BF16 final RMSE 0.189725, relative L2 0.360447).
An in-place dtype-mutation timing pilot also failed parity and was discarded.
No new approximation, cache reuse across trajectories, training or LoRA is used.

## Complete request observations

Prompt: `A red fox sleeping in fresh snow, golden hour`, seed 0, 1024 × 1024,
batch 1. Original uses CFG 3 and empty negative prompt. Cold includes model
load, tokenization, text encoding, denoising and RGB conversion; process import,
network download and PNG file I/O are excluded. Warm repeats the cached prompt.

| Model | First request | Warm request | Peak physical memory |
|---|---:|---:|---:|
| Original, 10 steps | 52.73 s; later 65.18 s | 38.77 s | 10.01 GiB |
| Original, 20 steps | 119.34 s | 102.02 s | 10.01 GiB |
| Turbo, 4 steps, first run | 19.65 s | 12.57 s | 9.96 GiB |
| Turbo, 4 steps, later instrumented run | 29.22 s | 22.20 s | 9.97 GiB |

There is one cold/warm pair per Original setting, an additional cold Original
10-step stage profile, and two Turbo pairs. The large
session variation precludes treating these values as stable medians or a
universal speedup. Different checkpoints and step counts are not equal-quality
alternatives. Original 10-step pixels match the pre-change HEAD pipeline bitwise.
Cold/warm repeats of Original 10/20 and Turbo 4 are also identical.

The additional Original profile measured DiT loading at 9.447 s, tokenizer/encoder
setup at 0.993 s and streamed text encoding at 4.257 s. Its pixels still match
the earlier Original run.

The instrumented Turbo run measured tokenizer/encoder setup at 0.932 s,
streamed text encoding at 3.225 s, and DiT loading at 3.876 s. Reloading DiT took
4.690 s; a complete request after unload took 25.803 s and reproduced the same
pixels. Explicit unload returned GPU allocated bytes to zero, including after
reload. These are short lifecycle checks, not an hours-long memory soak.

## Paired storage comparison

Three fresh-process pairs alternate reference/production order (AB, BA, AB),
with identical saved inputs and one warm timed denoise per process. All six
outputs and repeat outputs match bitwise.

| Pair | FP32 master + BF16 autocast | BF16 Linear storage |
|---|---:|---:|
| 1 | 16.142 s | 12.787 s |
| 2 | 13.940 s | 13.234 s |
| 3 | 13.740 s | 13.786 s |
| Peak physical memory, each pair | 14.84 GiB | 8.87 GiB |
| Live GPU allocation | 12.158 GB | 6.111 GB |

The memory reduction is repeatable. Timing does not consistently win, so the
change is retained as a memory optimization, without a general latency claim.
The denoising comparison excludes streamed text encoding and cannot be compared
directly with complete-request physical peaks.

## Native app verification

The Debug app was launched and operated through macOS accessibility and native
input, using a separate library copied from the existing one. All eight inherited
images remained byte-identical, and all ten files in the real user library were
unchanged. Restart restored 15 completed images (eight inherited, one explicitly
labelled mock result, six real results), two projects and the cancelled entry.
An empty library was tested separately; existing development preferences were
retained, so this was not a clean-machine installation test.

The following are server job durations for **actual GUI submissions**, including
local loading and result encoding, excluding application/process startup and
network downloads. The repeat followed an engine restart and is not a warm-cache
measurement. The source image for depth/restoration was the Original result.

| GUI action | Time | Result |
|---|---:|---|
| Turbo 4, first generation | 31.58 s | 1024 × 1024 |
| Turbo 4, same seed after engine restart | 32.42 s | Identical pixels |
| Original 10, after switching from Turbo | 65.57 s | Same pixels as HEAD baseline |
| Estimate Depth | 12.66 s | 1024 × 1024 |
| Restore and Upscale, budget mode | 34.10 s | 2048 × 2048, nine tiles |
| Turbo 4, after both image tasks | 26.54 s | Identical pixels |

The serial Turbo → Original → Depth → Restoration → Turbo lifecycle peaked at
10.13 GiB physical engine memory; the app peaked at 0.241 GiB. Health observations
reported one resident model at each stage. Cancel was invoked while the GUI showed
2/4 steps; the job ended cancelled and exposed no PNG. Export retained Turbo's
model identity, seed, schedule and checkpoint digest. Stopping the owned test
engine produced a visible failure banner; Try Again restored the ready state.

Also exercised: scrolling, prompt/seed search, zero-result search, project creation
and image moves, selection, arrow keys, Space/Quick Look, image details, prompt and
settings reuse, comparison slider and actual-size view, light/dark appearance,
761- and 1180-point window widths, and showing/hiding the inspector. A resize crash
and empty-gallery alignment defect found during this audit were fixed and their
GUI cases repeated successfully. The README screenshot is this real test session.

Validation: 405 Python tests passed with 19 GPU tests skipped in the CPU run;
the MPS suite then passed all 37 tests, including those 19. The final server/Turbo
focused suite passed 82 tests after the lazy-default correction. Swift passed
69 tests, including three opt-in HTTP tests using the CPU stand-in. Debug and
Release builds, Ruff lint/format, and offline wheel/sdist builds passed; package
contents include Turbo code and exclude weights, the virtualenv and scratch data.
Independent code review findings were addressed and rechecked.

Not verified: CUDA reference execution, multi-hour memory soak, other Apple chips
or RAM capacities, external 1× displays, full VoiceOver navigation, clean-machine
installation, signing/notarization or store submission. No general Original
speedup or release certification is claimed.

## Gallery motion follow-up — 2026-10-14

The pre-gallery liquid pointer trail and deformation were restored, with an iris
flower silhouette in the background and running tile. The composer now sits below
the gallery. Idle backgrounds pause; running tiles continue, and Reduce Motion or
the animation setting produces a still frame. Metal render tests cover pointer
response and still-frame invariance. Native hover events were also recorded during
the diagnostic run; the temporary logging was removed from the final build.

The same M5, fox prompt, seed 0 and 1024 × 1024 GUI workload produced Turbo in
47.01 s after a long idle, 16.88 s on repeat, and 28.61 s after switching back from
Original. Original 10 steps took 70.84 s after restoration. All four outputs match
their earlier reference pixels exactly. These individual GUI job times include
loading when needed; they are not a controlled animation performance comparison.
The engine's peak physical memory across depth, restoration, Original and Turbo
was 10.15 GiB. Earlier stage load measurements above were reused, not repeated.

Depth completed in 14.01 s and budget restoration in 33.34 s (2048 × 2048).
The first depth attempt exposed a closed stdout/stderr pipe in the old test-engine
process. Restarting it with a persistent log file and using the UI's Try Again
recovered successfully. Also checked: cancellation, model-specific controls,
search, settings reuse, light/dark appearance, restart, and a 761-point window.
The separate library retained all 19 inherited images and 14 conversation records;
both original library locations were unchanged by hash (10 and 23 files).

Validation: 406 Python tests passed, 19 GPU tests skipped; the earlier 37-test MPS
run applies to the unchanged engine. Swift passed 72 tests, including HTTP tests
and three Metal canvas tests; the latter were rerun after the final rendering fix.
Debug/Release builds and Ruff lint/format passed. Raw evidence is under
`.scratch/flower-ui-20261013/`. The limitations listed above still apply.
One accessibility-driver case remains unconfirmed with a physical pointer:
pressing the native search cancel button cleared its text but left the filtered
gallery until keyboard clearing. Keyboard search and clearing both succeeded.

## Reproduction commands

Use the pinned checkpoints and run GPU commands serially. The validation scripts
save environment information, paging counters, timings, digests and image evidence.
Only load fixture files you created or trust.

```bash
python scripts/validate_turbo.py prepare --out .scratch/turbo-fox --seed 0
python scripts/validate_turbo.py reference --out .scratch/turbo-fox
python scripts/validate_turbo.py upstream-bf16 --out .scratch/turbo-fox
python scripts/pair_turbo_storage.py --fixture .scratch/turbo-fox/fixture.pt --out .scratch/turbo-pairs.json --pairs 3
python scripts/profile_generation.py --checkpoint iris-3b/turbo --out .scratch/turbo-runtime --reload
python scripts/profile_generation.py --checkpoint iris-3b --out .scratch/original-runtime --steps 20
```

Local raw evidence is under `.scratch/turbo-20261013/`: `actual-*`, the three
native fixture folders, and `storage-fresh-pairs.json`. Model weights and large
fixtures are deliberately excluded from Git.
