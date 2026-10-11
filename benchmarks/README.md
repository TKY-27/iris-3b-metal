# Benchmarks

Release measurements of Iris-3B Metal 0.3.0, taken on 2026-10-11. The engine
code is the same in every run (commits `c649d8a` and `f4e83f2`, which differ
only in tests and CI). The raw reports and logs are in
[`results/2026-10-11/`](results/2026-10-11/); `scripts/plot_benchmarks.py`
turns them into `summary.json` and the charts in the README, and
`results/2026-10-11/commands.sh` records the commands.

## Machine and conditions

| | |
|---|---|
| Computer | MacBook Pro (Mac17,2), Apple M5: 10-core CPU (4P + 6E), 10-core GPU, 24 GB |
| System | macOS 27.0.1 (26A434), AC power, Low Power Mode off, no thermal or performance warning recorded by `pmset -g therm` |
| Software | Python 3.14.8, PyTorch 2.14.1 (MPS), Transformers 5.19.0; MLX 0.32.3 for iris3b-mlx-optimized |
| Weights | `speridlabs/iris-3b` @ `7445443`, `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281e`, already downloaded; file cache not purged |

**Background load.** No other compute workload ran and no process was
terminated, but the Claude desktop app, which drove this session, and the
window server used about 40% of the GPU's time between them throughout
(`results/2026-10-11/gpu-load.log`, per process every minute; the benchmark
process itself had about 55%). Every implementation ran under that same load,
interleaved, so the comparisons are like for like. On a quiet GPU the engine
is faster: the first runs of the day, with 3% background utilization, took
39 s for 10 steps, 3.9 s per step (`results/2026-10-11/early/`), against
4.6 s per step below. iris3b-mlx-optimized ran at the same speed as on
2026-10-10 (86–89 s for 10 steps then), so its numbers seem less affected.

## 1. Generation from a prompt

`scripts/benchmark.py --size 1024x1024 --steps {10,20} --repeats 3 --profile`,
three fresh processes per step count, alternating 10 and 20 steps. Seed 0,
guidance 3 over the whole schedule, DPM-Solver++ (2nd order), flow shift 4,
empty negative prompt, FP16 trunk and attention. Medians over the three
processes, range in brackets:

| Seconds | 10 steps | 20 steps |
|---|---:|---:|
| Importing PyTorch and the engine | 0.7 [0.7–0.7] | 0.7 [0.7–0.7] |
| Tokenizer and text-encoder setup | 1.4 [1.3–1.6] | 1.4 [1.3–1.4] |
| Text encoding, prompt and negative prompt | 2.1 [1.9–2.7] | 2.1 [2.0–2.1] |
| Model load | 6.1 [5.8–6.1] | 6.1 [5.8–6.2] |
| First image, kernels compiled on the way | 45.3 [43.0–46.1] | 89.8 [89.7–91.5] |
| **Launch to first image** | **55.5** [54.2–55.8] | **100.2** [99.6–101.9] |
| **Next image, prompt cached** | **44.9** [43.1–45.3] | **90.5** [90.1–91.3] |
| Next image, new prompt | 48.4 [47.3–48.8] | 94.9 [94.4–95.1] |
| Peak memory of the process, GiB | 10.0 | 10.0 |

- *Text encoding* streams Qwen3-VL-4B through the GPU one layer at a time.
- *Model load* reads the 12 GB FP32 checkpoint and converts it on the GPU.
- An image request runs from the prompt to a ready `PIL.Image` (a CPU copy,
  so the GPU work has finished). Each sampling step evaluates the model on a
  batch of two (guidance): 4.6 s per step in the median, 4.7–4.8 s for the
  first step. Turning the result into an image takes 4 ms.
- *Peak memory* is the peak physical footprint of the process
  (`proc_pid_rusage`; what Activity Monitor calls Memory), including Metal
  allocations; 6.5 GiB of it are the weights.

**100 steps**, one process: first image 454 s, next image 457 s, launch to
first image 465 s, peak 10.0 GiB.

## 2. Same inputs, other implementations

Every implementation denoises the same fixture: the Qwen3-VL text states that
iris3b-mlx-optimized computes for its prompt 0, and NumPy `default_rng(42)`
noise, at 1024 × 1024 with guidance 3 over the whole schedule, flow shift 4
and the second-order DPM-Solver++. Each step costs two model evaluations in
all of them. Times are warm requests (model loaded, kernels compiled),
including per-request preparation and excluding text encoding and model load;
memory is the peak physical footprint of the process from `/usr/bin/time -l`.
Two rounds, the second in reverse order; median, range in brackets, and the
number of warm requests:

| 1024 × 1024 | 10 steps, s | 20 steps, s | Peak memory, GiB |
|---|---:|---:|---:|
| **This engine** | **42.1** [38.6–45.2], 4 | **92.8** [91.2–94.4], 2 | **10.0** |
| iris3b-mlx-optimized, speed preset | 83.6 [83.6–83.9], 4 | 168.9 [167.4–170.4], 2 | 11.4 |
| iris3b-mlx-optimized, memory preset | 84.0 [83.6–84.3], 4 | 168.6 [168.3–168.9], 2 | 10.1 |
| Upstream PyTorch, BF16 autocast | 69.2 [66.4–72.0], 2 | 135.2 [135.1–135.3], 2 | 17.9 |

| Implementation | Version | How it ran |
|---|---|---|
| This engine | this repository | `scripts/denoise_fixture.py FIXTURE --steps N` |
| [iris3b-mlx-optimized](https://github.com/octaviusp/iris3b-mlx-optimized), speed preset | `b06bbe4` | `bin/iris-mlx --action sample --backend mlx-optimized --dtype bfloat16 --precision-policy mlp --execution fused --eval-every 4 --attention-dtype reference --tf32`: BF16 MLP weights, everything else FP32 |
| iris3b-mlx-optimized, memory preset | `b06bbe4` | the same with `--quantization q8-bf16-fit`: 8-bit MLP weights |
| [Upstream PyTorch](https://github.com/speridlabs/iris-3b), BF16 autocast | `a8d1523` | `scripts/denoise_fixture.py FIXTURE --reference autocast`: FP32 weights under the BF16 autocast of upstream's `scripts/sample.py` |

**Distance from FP32.** After 10 steps, against the upstream code run in
FP32 on the same fixture (relative RMS error and PSNR of the final pixels,
LPIPS with AlexNet):

| 10 steps vs upstream FP32 | NRMSE | PSNR | LPIPS |
|---|---:|---:|---:|
| This engine | 3.50% | 37.9 dB | 0.012 |
| iris3b-mlx-optimized, speed preset | 0.64% | 52.7 dB | 0.0006 |
| iris3b-mlx-optimized, memory preset | 0.79% | 50.9 dB | 0.0008 |
| Upstream PyTorch, BF16 autocast | 5.87% | 33.4 dB | 0.032 |

iris3b-mlx-optimized keeps attention, norms and most linear layers in FP32,
which is why it stays much closer to FP32 and why it is slower; this engine
runs the transformer trunk in FP16 and is closer to FP32 than the upstream
BF16 recipe. These are distances from one reference trajectory, not image
quality, and a diffusion trajectory amplifies them differently per prompt;
the README's [Image quality](../README.md#image-quality) section compares
complete 100-step generations over eight prompts. Every result above was
bitwise identical across rounds.

**Upstream in FP32**, the default when the upstream code runs on a Mac, was
not rerun: at 1024 × 1024 it needs about 20 GiB and swaps heavily on a 24 GB
Mac. Measured once each on 2026-10-10 with the same fixture, it took 346 s
for 10 steps and 2940 s for 100.

**Published numbers.** iris3b-mlx-optimized reports 38.4 s (speed preset)
and 38.8 s (memory preset) for 10 steps on an M5 Pro with 48 GB. That is a
different, larger GPU, and its numbers cannot be compared with these.

**Earlier 100-step comparison.** On 2026-10-11, before the release, the same
fixture was denoised at 100 steps, with uncontrolled background load: this
engine 387–414 s at 10.6 GiB; iris3b-mlx-optimized 844–852 s at 11.4 GiB
(speed) and 842–864 s at 10.0 GiB (memory), interleaved, two runs each;
upstream BF16 autocast 919 s at 18.4 GiB (one run).

## Reproducing

```bash
pip install -e ".[dev]"
python scripts/benchmark.py --size 1024x1024 --steps 20 --repeats 3 --profile --out run.json
python scripts/denoise_fixture.py fixture.npz --steps 20 --repeats 2
python scripts/denoise_fixture.py fixture.npz --steps 20 --repeats 2 --reference autocast
python scripts/plot_benchmarks.py benchmarks/results/2026-10-11 --assets assets/benchmarks
```

`fixture.npz` holds `y` (`[2, 300, 12, 2560]` Qwen3-VL states, unconditional
row first), `mask` (`[2, 300]`) and `noise` (`[1, 3, 1024, 1024]`). The one
used here (86 MB) is not in the repository; `commands.sh` shows how it is
built from iris3b-mlx-optimized's conditioning. Before timing anything, check
that the GPU is idle and the Mac is not swapping (see
[CONTRIBUTING.md](../CONTRIBUTING.md)).

[`startup-validation-2026-10-11.md`](startup-validation-2026-10-11.md)
records the text-encoder and conditioning-cache work, with its rejected
experiments.
