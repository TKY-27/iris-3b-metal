# Iris Metal startup / inference validation — 2026-10-11

This is the optimization session's record, before the subsequent independent audit.

This report records local implementation and measurements, not a release. No commit, push, PR, checkpoint conversion, additional checkpoint, or optional task checkpoint was retained.

## Baseline and controls

- Local HEAD: `b62c759e5ed4ca64c87a23f329ee972fb4f36ef5`. The working tree already contained GUI, task, server, documentation and test changes. Their original diff was saved before optimization work; the baseline Python package includes those pre-existing changes.
- Actual hardware: Mac17,2, base Apple M5, 10 CPU cores (4 performance / 6 efficiency), 10 GPU cores, 24 GiB unified memory. macOS 27.0.1 (26A434), AC power, low-power mode 0.
- Python 3.14.8; PyTorch 2.14.1, source commit `5c4886908584029761b579af026dcfb627c84070`; Transformers 5.19.0. No MPS/PYTORCH environment overrides. MPS recommended working set: 19,069,665,280 bytes.
- Existing local `iris-3b/model.safetensors`, 11,950,112,680 bytes. Existing pinned Qwen snapshot `ebb281ec70b05090aa6165b016eac8ec08e71b17`. Final inference runs use the local DiT and offline Hub access.
- FP16 trunk/attention, BF16 text adapter and pixel linears, FP32 residuals/norms/modulation and designated checkpoint values. Original GPU checkpoint cast order remains intact. CFG, interval, solver, order, schedule, steps and model are unchanged.
- Saved actual CPU noise tensors are reused for comparisons. The raw positive/negative states and original adapted conditioning are captured once. Production requests preserve positive/negative ordering; fixed conditioning isolates engine comparisons.
- Only one full DiT is resident. No unrelated process was killed. No OS cache purge; these are downloaded-weight / unpurged OS-cache measurements, not proven cold-filesystem measurements.
- An initial resolver diagnostic selected the incomplete remote DiT cache and unintentionally started a download. Its own process was stopped and only the newly created incomplete file was removed; all subsequent inference uses the existing local checkpoint. No added weights remain.
- Detailed bounded fixtures, scripts and JSON measurements reside in `.scratch/optimization-20261011/`. Accepted request timing has no tracing, per-step global synchronization, or substituted solver. The existing completion-based StepReporter is unchanged.

## Retained implementation

1. Qwen transfers are nonblocking on MPS, with one decoder layer in flight. The layer completion event retires GPU weights and CPU backing storage before the next layer; reader/source cleanup also waits after an exception. All missing prompts still share the weight pass, only real tokens are evaluated, and embedding rows remain selective. No extra full weight clones or pinned copies.
2. Complete immutable Hub snapshots resolve locally. Required config, index, nonempty shards and tokenizer files are checked; partial/malformed snapshots and mutable revisions retain the existing Hub fallback. Existing local directories retain their original behavior.
3. The pipeline caches detached copies of full-length adapted conditioning for four recent ordered conditioning batches, including the recurring negative prompt. Raw `encode()` results retain their original contract. Model/config/precision changes, raw tensor versions and unload invalidate adapted entries. Combined raw and adapted bytes stay within the original raw-encoding cache budget. Batch shape, ordering and duplicate rows are part of the key; partial batch reuse cannot alter GEMM rounding. Returned concatenations isolate cache entries from caller mutation.
4. The benchmark separates import time from GPU sampling, resolution/config, tokenizer initialization, text encoding, model loading and request time. A genuinely uncached warm-engine prompt is separate from cache hits. Forward event diagnostics read after completion and are excluded from acceptance timing.

## Rejected experiments

- Fixed 64/128/256/full adapter buckets reduced the standalone B2 adapter median from 0.163 to 0.038 seconds (five pairs), but the native 1024 fixed-state velocity failed the existing `rtol=1e-2, atol=1e-2` gate at early, middle and late 10-step states. Relative errors were 0.00335–0.00383, maxima 0.046875–0.0625, with 0.3–0.8% mismatched values. No tolerance was changed. The complete adapter implementation was restored exactly. The initial five-pair 10/20-step and three-pair rectangular request trial therefore describes a rejected candidate, not the final retained cache implementation. Final comparisons were rerun.

- Bounded asynchronous DiT loading retained CPU→MPS FP32 copy then GPU cast. Per-tensor events increased the three-process median from 6.87 to 7.52 seconds. A byte-window completion event still measured 6.84 versus 7.10 seconds across three pairs. The loader was restored exactly.
- Shared PiT conditioning casts and direct FP16-attention→BF16 merge passed fixed-output equality, including nondefault attention dtype handling, but the encompassing request screen did not improve. Both changes were restored exactly.
- Removing the V layout copy did not produce a consistent copy+SDPA+merge win across trunk, PiT and non-square shapes (seven pairs). The intentional copy remains.
- Existing rejected `silu_mul` and sigmoid/merge fusions were reviewed and not repeated. Small modulation copies, dispatch tuning and compilation were not pursued: the observed submission gaps and pointwise costs did not justify additional complexity.
- GUI animation off did not show a reliable advantage over on. No GUI rendering change was retained.

An early discarded PiT screen used reversed CFG conditioning order. Its images and request-quality comparison are excluded from acceptance. Fixed-state equality and adapter-only timings remain separately identified diagnostics. The final production comparisons use the original pipeline ordering.

## Profiling limits and measured bottleneck

A representative 10-second Metal System Trace interval contains one full B2 model evaluation. Python GPU-encoder interval union was 3.6615 seconds over a 3.6936-second span (507 encoders); the 32.1 ms gaps are not automatically CPU submission stalls. Other WindowServer/Codex activity was present. Shader interval export was empty, so precise per-kernel hardware duration attribution was unavailable.

The installed-source dispatch conditions and trace function metadata include `prefill_attention_mpp_a10_float16_bq64_bk32_bd128_wm4_wn1_hm0_dc0_maskfloat16`, native GQA and Apple10/M5 mapping. This establishes an MPP-capable dispatched path without assuming unfused attention. Function metadata is not a hardware kernel timestamp.

A separate event-instrumented forward took 3.7615 seconds. Coarse complete-command intervals attribute approximately 2.480 seconds to 278 measured linears and 0.850 seconds to 28 SDPA calls; V copies total 5.4 ms and output merges 17.0 ms. Two linear and two normalization event intervals were unavailable, not counted as zero. These measurements are diagnostic and not additive CPU+GPU timings or acceptance runs. GEMM and SDPA remain the material model-evaluation costs.

PyTorch source for the installed commit: [MPS Copy.mm](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/Copy.mm), [MPS Attention.mm](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/Attention.mm). Nonblocking copies retain CPU storage through Metal buffer lifetime; explicit layer source retention also bounds the safetensors lifetime. Fusing CPU transfer and dtype conversion can convert on CPU, which is why the DiT GPU-cast order was preserved.

## Validation boundaries

Final retained-code validation: the original full adapter, all six native 1024 early/middle/late fixed-state velocities (10 and 20 steps), and both complete float32 10/20-step outputs are bitwise equal, finite, and pass the existing tolerances. Complete outputs have NRMSE 0, LPIPS 0 and SSIM 1. All 13 final paired production uint8 images are bitwise equal; representative final 10-step, 20-step and rectangular images were viewed directly. The existing face, hands/pottery and typography fixtures also showed input-specific equality in the earlier bucket trial, but that trial's later native velocity failure overruled those passes. The final implementation uses the unmodified adapter and denoiser. Existing short-step grid artifacts and the baseline sign spelling error are unchanged.

GUI Release build and Debug tests succeeded (32 tests, 3 skips, no failures). Actual generating canvas, progress text, stop and retry were exercised using a test bundle identifier, test preferences and isolated generated-image libraries. Cancellation stopped at completed step 2/10; retry completed. macOS accessibility traversal of a multi-result screen triggered two recursive SwiftUI/AppKit AX stack-overflow crashes. GUI source was unchanged by this optimization, and this AX case is not reported as passing. Subsequent measurements used new chats and avoided the problematic completed multi-result traversal. After one crash, the automation briefly relaunched the own test executable without its isolation arguments and displayed the default library. No generation or edit was performed there; the own process was closed and relaunched with isolation. Absence of framework/cache writes to that default library was not verified by hashes. This access limitation is recorded rather than claiming fully verified library isolation.

HTTP live tests reject invalid steps with 400, cancel at step 2/10, keep observed progress monotone, then finish two repeated requests and serve valid PNGs. Final phase tables, paired request results, memory lifecycle checks, and regression results follow below.

A diagnostic initially read the first forward end event twice and blocked in `MPSEventPool::elapsedTime` after GPU work had completed. The installed runtime consumes the end-event CPU notification on each read. Only the own diagnostic process was terminated; each timing event is now read as an end once. This did not affect uninstrumented acceptance runs. Source: [installed MPSEvent.mm](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/mps/MPSEvent.mm).

The original physical-footprint helper assumed reading `ri_interval_max_phys_footprint` reset it. XNU reads it with reset disabled. The helper now explicitly calls `proc_reset_footprint_interval` for its own process after each interval read; a 32 MiB mmap/unmap check confirmed reset to the current footprint. The paired-request process imported the old helper before this correction: its `peak_gib` values are cumulative process high-water marks, not independent request peaks. Fresh final stage and lifecycle runs use the corrected helper and separate interval/current/lifetime values. Source: [Apple XNU kern_resource.c](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_resource.c).

## Final production requests

One shared full DiT, actual saved noise, original full adapter and solver, CFG 3.0 over [0,1], order 2, flow shift 4.0, empty negative prompt. AB/BA pairs alternate; timings end after the CPU image is ready, with no tracing or per-step global synchronization. Positive and negative raw encodings are held fixed. Candidate conditioning uses the original full adapter and its cache. All 13 paired final images are bitwise equal.

| Workload | Pairs | Baseline median (IQR), s | Candidate median (IQR), s | Paired B−C median, s | Exploratory median interval, s |
|---|---:|---:|---:|---:|---:|
| 768×1344, 10 steps | 3 | 43.157 (0.071) | 42.950 (0.080) | 0.136 | [0.113, 0.248] |
| 1024×1024, 10 steps | 5 | 39.919 (3.455) | 40.176 (2.107) | -0.257 | [-0.675, 1.496] |
| 1024×1024, 20 steps | 5 | 87.786 (1.043) | 87.511 (0.549) | 0.745 | [-0.357, 1.406] |

The interval is an exact small-sample bootstrap summary of the observed paired differences, not an independent-run population confidence guarantee. Baseline A/A adjacent differences were 0.838, 1.034 and 1.307 seconds; its six runs rose from 37.77 to 42.89 seconds. The final 10-step block also drifted upward. The 10/20-step intervals include zero, and the rectangular 0.136-second paired median is below the broader A/A variability. **No complete-request or model-evaluation speedup is demonstrated.** Stage savings are reported separately. GPU clock/thermal pressure could not be collected without elevated powermetrics access; pmset recorded no thermal/performance warning. Other user processes remained running.

Paired-process cumulative physical high-water marks were about 10.01 GiB at 1024² and 10.13 GiB at 768×1344. Allocator/driver bytes were respectively 6,103,077,632 / 9,817,980,928 at 1024² and 6,142,235,392 / 9,933,717,504 at the rectangle. The shared process holds the candidate’s two conditioning entries even during baseline runs; these are not isolated-mode physical-memory comparisons. Corrected fresh-process and lifecycle measurements are below.

System Pageouts stayed zero. Swapouts increased by 1,357 pages during one candidate 10-step request (about 21.2 MiB with 16 KiB pages), while system swap used declined about 80 MiB and pressure level was normal (1). All other paired requests had zero Swapouts. This is global system activity, with no process attribution; it is not reported as candidate-caused swapping or as a completely swap-free session.

The saved initial tensors, rather than seed alone, define the controlled inputs:

- `.scratch/optimization-20261011/noise-1024x1024.pt`: `c7d6440eb22385d39b0232a9d26c212c1881ce9584872df0b90f1fcff7af001f`
- `.scratch/optimization-20261011/noise-768x1344.pt`: `c912100426ccc098035ed86546e0865d5ae40aacc99e7c3e07880036a4323acd`
- `iris-3b/config.yaml`: `077ce19156d9425a7e665a0059be02b3c5a1fcd7aea7607047b4f7486efa6cd5`

## Fresh-process stages and warm request diagnostics

Three fresh process pairs, downloaded weights and unpurged file cache, outer completion synchronization. The adapter row is a cold B2 adapter diagnostic; production calls preserve separate B1 positive/negative batches. Load/adapter/forward code is identical in final B/C: their differences measure noise, not retained speedups. Steady forward pools three fixed-state evaluations per process (nine samples/mode), with no solver or image conversion.

| Stage | Baseline median (IQR), s | Candidate median (IQR), s |
|---|---:|---:|
| Process start → import ready | 0.733784 (0.032098) | 0.715737 (0.009178) |
| Imports within script | 0.717207 (0.032860) | 0.692348 (0.006471) |
| Existing local DiT resolution | 0.000074 (0.000005) | 0.000059 (0.000027) |
| Qwen config/tokenizer initialization, offline | 0.796936 (0.013464) | 0.799740 (0.013351) |
| First raw Qwen encoding | 2.597737 (0.348303) | 2.211016 (0.047909) |
| DiT load, original GPU casts | 6.866370 (0.603326) | 7.051227 (0.244072) |
| First full adapter, B2 | 0.429312 (0.012689) | 0.470663 (0.024735) |
| First model completion, fixed conditioning | 4.156168 (0.247514) | 4.288153 (0.168179) |
| Steady complete model evaluation | 4.020081 (0.358321) | 4.087752 (0.296391) |
| Warm-engine genuinely uncached Qwen prompt | 3.613312 (0.186296) | 3.205259 (0.037116) |

First Qwen B−C pairs were 0.959, 0.387 and 0.358 seconds; fresh warm-engine uncached pairs were 0.687, 0.408 and 0.389 seconds. **These fresh-stage processes also experienced substantial system-wide paging during original full-model loading.** B Pageouts [106,117,90], Swapouts [172609,110784,98523]; C Pageouts [127,115,61], Swapouts [103788,118350,86372]. Global swap-used changes ranged from −1.53 to +1.46 GiB. These are not process-attributed and do not establish an overall cold-start advantage. Further isolated warm text pairs are below.

Seven complete pinned-cache resolution pairs measured the original Hub path versus complete-cache local-first lookup. All weight-file access was forced local-only; no download was possible. The old path's remote metadata request was retained. Results apply to this existing Qwen snapshot, not a fabricated complete remote DiT snapshot.
- baseline: median 0.984 ms, IQR 0.080 ms.
- candidate: median 0.286 ms, IQR 0.038 ms.

The benchmark's separate fresh-process default-path 1024²/10-step runs below are **one run/mode**, sequential, with different background/paging conditions and normal CPU-generated noise. They verify the four startup categories and corrected reporting; they are not acceptance evidence for a speedup. Only the paired saved-noise production table above establishes controlled request comparisons.

| Measurement | Baseline, s | Candidate, s |
|---|---:|---:|
| Imports | 0.469652 | 0.716240 |
| Local checkpoint resolution | 0.000069 | 0.000080 |
| Text encoder/tokenizer init | 1.026571 | 1.348299 |
| First text encoding | 0.915749 | 2.172735 |
| DiT load | 7.040042 | 7.079782 |
| First generation after preparation | 37.495483 | 38.917895 |
| First request including preparation | 46.506323 | 49.566220 |
| Imports + complete first request | 46.975975 | 50.282460 |
| Warm cached request | 38.497324 | 39.402682 |
| Warm genuinely uncached request | 42.412381 | 44.759319 |

A separate event-instrumented 10-step request per mode reads events after completion. Those original diagnostic JSON files timestamp denoising at StepReporter.finish, before the final solver tail / finite check; this boundary is explicitly not the complete final solver wall time. The benchmark now additionally timestamps through entry to image conversion, after the existing finite check, and reports the forward-completion boundary separately. No per-step global synchronization was added. Diagnostics have adapter misses because their engine wrapper changes cache identity.

| Diagnostic measurement | Baseline, s | Candidate, s |
|---|---:|---:|
| Original completed-forward-step boundary | 39.108977 | 42.569475 |
| CPU image conversion | 0.003936 | 0.004069 |
| First host model submission | 0.195404 | 0.190348 |
| First coarse model completion | 4.064167 | 4.347236 |
| Solver/CFG/submission/wait residual to old boundary | 0.133343 | 0.129648 |

The residual is wall time minus coarse forward intervals, not an isolated solver GPU-kernel measurement. Callback intervals include StepReporter’s bounded lookahead. Import initialization, CPU waits and concurrent GPU intervals must not be added as independent costs.

A loading diagnostic measured 6.600 s total, 14.3 ms CPU safetensors view access, 6.346 s blocking copy API-inclusive time and 142 ms cast API submission. Copy intervals include prior GPU cast waits; file faults can be deferred. There were 16 unavailable cast events. Original CPU→MPS FP32 then GPU cast is retained.

Text diagnostics separate tokenization (0.07–0.08 ms), selective embedding submission (18.7–41.4 ms), CPU weight-view access (15.4–43.6 ms), and transfer API-inclusive time (blocking 2.943 s, nonblocking 11.8 ms). The complete stage in those sequential instrumented runs was 3.155 versus 0.795 s. File-cache/background differences prevent interpreting that as a 4× stage speedup; the API result demonstrates removal of individual host waits only. `vmmap` and `lsof` after both encodings show no Qwen shard mapping/open file; reader/source counts are zero.

## Adapted-conditioning cache

Five interleaved/reversed pairs of the original separate B1 positive/negative adapter batches, from saved actual raw states. No bucket optimization remains. Miss/hit outputs are bitwise equal.

| Stage | Median (IQR), s |
|---|---:|
| Baseline full adapters | 0.192050 (0.006876) |
| Candidate cache misses | 0.192801 (0.001226) |
| Candidate cache hits | 0.000367 (0.000122) |

Cache-miss overhead is within observed adapter variability. Five additional paired saved-noise one-step diagnostics preserve the complete production solver and StepReporter; they isolate prompt-to-first-model boundaries, and do not replace the required 10/20-step comparisons.

| Boundary from request start | Baseline median (IQR), s | Candidate cache-hit median (IQR), s |
|---|---:|---:|
| First host model submission | 0.193578 (0.003845) | 0.001952 (0.000471) |
| First completed-model callback | 4.618351 (0.003169) | 4.409027 (0.020684) |

This demonstrates cached prompt-to-submission savings, not faster individual model evaluations. Two raw plus two adapted entries use 43,012,800 bytes in a 294,950,400-byte budget. The cache-stage interval had zero Pageouts, seven global Swapouts and a declining system swap-used total; pressure remained normal.

## Memory and lifecycle

Corrected fresh-process lifetime peaks: B 10.01377 GiB (IQR 0.00426), C 10.01516 GiB (IQR 0.00031), roughly 1.4 MiB median difference. First raw Qwen encoding phase peaks are about B 1.49444 / C 1.49604 GiB. No material candidate peak increase was observed. Global paging during the original full checkpoint loader is reported above, not hidden by these footprint numbers.

Nineteen distinct mixed prompts exercise raw/adapted eviction with actual small requests. Raw entries settle at 15, adapted entries at four, combined bytes 288,804,000 ≤ 294,950,400. Physical current plateaus around 7.828 GiB after the cache fills; allocator/driver plateau at 6,069,758,720 / 7,553,482,752 bytes. Qwen readers/sources are released after each request. Cancellation at completed step one, recovery, unload, weak-reference release and reload succeed. Unload clears adapted entries and returns allocator bytes to zero; the retained raw CPU cache follows the existing contract.

Whole lifecycle system deltas are 305 Pageouts and 181,063 Swapouts, with net swap-used +99 MiB. Most swapping occurs during original initial/reload checkpoint work, not steady cached requests; these are global counters, not process attribution. One immediate post-reload unload footprint snapshot was stale despite allocator=0 and driver≈1.7 MiB. A separate bounded unload observation after deleting all diagnostic references is recorded with the final warm-text pairs. Sustained engine retention through a cache was not observed.

## Final warm-engine Qwen pairs

The first five warm pairs were close to noise: paired median 0.0146 s, with two losses; small regression-test work also overlapped this block's warmup/A/A and beginning. Rather than declaring a win from its fastest run, timing was extended to ten total pairs. The additional five reversed/interleaved pairs ran after regression tests finished, with no own parallel inference or tracing. Background utilization tails were 9.5% before / 9.25% after; no pmset thermal warning. External user processes remained running.

| Warm uncached Qwen stage | Baseline median (IQR), s | Candidate median (IQR), s | Paired B−C median, s |
|---|---:|---:|---:|
| Additional five isolated pairs | 3.378766 (0.036220) | 3.093572 (0.053997) | 0.294977 |
| All ten, exploratory pooled | 3.341577 (0.363171) | 3.073026 (0.075091) | 0.290086 |

All five additional paired improvements are 0.248–0.379 s. Their small-sample median resampling interval spans [0.248,0.379] s. Baseline A/A IQR in that block is 0.0956 s; adjacent absolute changes are 0.1497, 0.2742 and 0.0884 s. The observed median improvement of 0.295 s exceeds those measured changes. Across all ten, the exploratory bootstrap interval is [0.0146,0.3790] s; this wider, block-dependent result is not a universal throughput guarantee. This supports retaining the one-layer path for the complete uncached encoding stage, with background/clock uncertainty, and does not establish faster full generation.

All ten raw hidden-state/mask digests match bitwise, reader/source counts return to zero, and allocator/driver are identical at 6,128,636,928 / 8,758,427,648 bytes. Pooled physical phase peaks B 9.02967 / C 9.03036 GiB differ by approximately 0.7 MiB. All twenty text encodings have zero global Swapouts; Pageouts total B 186 / C 179. System pressure remains normal (1), with swap-used totals declining. Both bounded unload observations give allocator=0, driver=720,896 bytes, no surviving engine weak reference, and physical current 0.827–0.829 GiB immediately and at 0.1/0.5/2-second observations. The earlier stale ledger snapshot is therefore not evidence of sustained cache/model retention. No two-layer window was pursued: the one-layer implementation gives a bounded stage benefit without justifying extra residency.

## Regression and interface results

Relevant final pytest: **109 passed**, four existing upstream MPS pin-memory warnings, no skipped or failed tests. Coverage includes CLI, engine, pipeline, HTTP server, task-model paths, cache completeness/fallback/order/duplicates/eviction/version/precision/unload, inference-mode raw inputs, streamed-copy exception/reuse cleanup, text adapter, solver and caption overflow. No optional task checkpoint was downloaded. Ruff and diff whitespace checks pass. GUI Release build and 32 Debug tests (three skips) are separate from the native real-model correctness and performance checks.

The environment did not expose a `.venv/bin/iris-metal` executable; the CLI smoke uses its actual entry implementation via `python -m iris_metal.cli`, without installing a second environment or framework. Final runtime outcomes and cleanup follow below.

The corrected full-denoising boundary was verified with the final benchmark at 64²/2 steps: completed-forward boundary 0.518953 s, complete denoising 0.519259 s, CPU image conversion 0.000323 s; all timing events are read once after the request. An additional actual saved-noise 1024²/10-step diagnostic pair verifies that the final solver tail and finite check are included:

| One-pair boundary diagnostic | Baseline, s | Candidate, s |
|---|---:|---:|
| Completed forward steps | 37.216512 | 38.998410 |
| Complete denoising | 37.223064 | 39.001462 |
| CPU image conversion | 0.006473 | 0.003970 |
| Request through CPU image | 37.321897 | 39.005585 |

Images are bitwise equal. This one sequential pair is a phase-boundary diagnostic with clock/background drift, not a performance winner. The five-pair 10/20-step production results remain the acceptance evidence.

Final-code CLI real-model generation saved a valid 64² PNG at two steps. Python API correctness, repeated mixed requests, error cleanup, cancellation and reload were exercised above. Final-code live HTTP returns 400 for steps=0, cancels at completed step 2/10, then serves two successful 64²/2-step PNG requests with monotone progress, no queued work and a healthy resident text-to-image model. Existing lightweight HTTP/task-model tests also pass. GUI actual canvas, progress, stop/retry and animation contention were exercised earlier in this session; GUI source is byte-identical to the pre-existing dirty baseline. The AX crashes and briefly non-isolated automation relaunch remain limitations; final GUI AX traversal was not repeated. Animation medians were on 45.49 s, off 44.82 s (three requests each); one occluded on request was 40.82 s. Order, thermal and recovery effects prevent attributing the difference to animation. No GUI rendering change was retained.

## Cleanup and final state

All own model/server/GUI benchmark processes have exited. Own trace recording, build products, pytest fixtures, isolated GUI libraries, invalid-order images and large temporary raw/adapted/velocity tensors were removed. Fresh test preferences were deleted; the secondary test domain did not exist. Existing user applications, default library data, unrelated scratch work and checkpoint files were not intentionally modified or removed. The earlier automation relaunch's framework/cache-write absence remains unverified, as described above.

Bounded retained evidence consists of two actual saved noise tensors, representative images, small JSON/log/script records and trace exports: approximately 53.2 MiB logical bytes. Some diagnostic scripts refer to removed temporary conditioning fixtures; their recorded metrics remain, and the public benchmark works directly from the existing checkpoint. No trajectory archive, duplicate checkpoint or converted weights remain.

Final preservation checks confirm the complete original tracked dirty diff is unchanged, HEAD is unchanged, checkpoint size/inode/mtime are unchanged, and `loading.py`, `kernels.py`, `model.py`, pre-existing `server.py` and `tasks.py` are byte-identical to the saved baseline. Only the retained text streaming/local resolver/cache, benchmark, focused tests and this report are additions to the user's original work. No commit, push, PR, visibility or release action was performed.

The demonstrated gains are bounded uncached-Qwen stage time and cached prompt-to-model submission. Complete model evaluation and 10/20-step generation gains are not demonstrated. Original loading remains roughly seven seconds, and GEMM/SDPA dominate the unchanged denoiser. Cold filesystem timing, GPU clock/power telemetry, precise per-shader hardware timestamps, and the GUI AX case remain unverified or inaccessible; no broader optimization work is implied by this report.
