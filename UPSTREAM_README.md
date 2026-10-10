# Iris-3B

**Pixel-space generation & general vision learner**

Generative priors are a promising foundation for downstream vision tasks. In
this project we explore pixel-space generative models as an alternative to
vision foundation models such as DINOv2.

Iris-3B is a 3B-parameter diffusion transformer that generates directly in
pixel space: no VAE, no latent space, every pixel is produced by the model, so
nothing is lost to a lossy, texture-biased latent. It is not only an image
generator: the same pixel-space prior, fine-tuned, estimates monocular depth and
restores and upscales images — one backbone, three tasks. This repository contains the
model, the training and fine-tuning code, and inference.

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2610.09450-b31b1b)](https://arxiv.org/abs/2610.09450)
[![Project Page](https://img.shields.io/badge/Project_Page-1f6feb)](https://speridlabs.com/research/iris)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/speridlabs/iris-3b)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/speridlabs/iris-3b)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

</div>

https://github.com/user-attachments/assets/2345a853-c673-4ddb-83b9-7e05c83198ed

## Generations

Samples from the final checkpoint at native aspect ratios of about one megapixel
(CFG 3, 100 steps, one fixed seed, no prompt rewriting).

<table>
<tr>
<td width="33%" valign="top"><img src="assets/gallery/0020.jpg"><br><sub>Studio portrait of a Maasai elder wearing vibrant beaded jewelry, deep red cloth, dark backdrop, Rembrandt lighting, ultra detailed skin texture</sub></td>
<td width="33%" valign="top"><img src="assets/gallery/0021.jpg"><br><sub>Close-up portrait of a young woman with silver glitter freckles and iridescent makeup, soft pastel background, high fashion beauty photography</sub></td>
<td width="33%" valign="top"><img src="assets/gallery/1166.jpg"><br><sub>A glass sculpture of a heart filled with flowers, caustics and reflections, 3D render</sub></td>
</tr>
</table>

<table>
<tr>
<td width="33%" valign="top"><img src="assets/gallery/0029.jpg"><br><sub>Black and white portrait of a fisherman with a thick grey beard and deep wrinkles, piercing eyes, overcast light, fine grain film photograph</sub></td>
<td width="33%" valign="top"><img src="assets/gallery/0115.jpg"><br><sub>A Byzantine-style mosaic of a peacock made of tiny gold and turquoise tiles, shimmering texture</sub></td>
<td width="33%" valign="top"><img src="assets/gallery/1153.jpg"><br><sub>A bronze sculpture of a horse in motion, patina, dramatic museum spotlight</sub></td>
</tr>
</table>

<table>
<tr>
<td width="33%" valign="top"><img src="assets/gallery/1088.jpg"><br><sub>The ancient city of Petra with the Treasury carved into pink sandstone, morning light</sub></td>
<td width="33%" valign="top"><img src="assets/gallery/1196.jpg"><br><sub>A tree with lightbulbs instead of fruits glowing at dusk, surreal concept art</sub></td>
<td width="33%" valign="top"><img src="assets/gallery/1214.jpg"><br><sub>Thousands of sky lanterns rising into the night sky at Yi Peng festival in Chiang Mai</sub></td>
</tr>
</table>

<table>
<tr>
<td width="50%" valign="top"><img src="assets/gallery/0005.jpg"><br><sub>The aurora borealis swirling green and violet over a snowy Lofoten fishing village with red wooden cabins, reflections in a calm fjord, night photograph</sub></td>
<td width="50%" valign="top"><img src="assets/gallery/1001.jpg"><br><sub>Volcanic eruption at night in Iceland, rivers of glowing lava flowing across black fields, plumes of steam lit orange, long exposure</sub></td>
</tr>
</table>

More samples: [project page](https://speridlabs.com/research/iris#gallery).

## One prior, many tasks

Iris-3B is a general vision learner, not only a text-to-image model: the same
pixel-space backbone, fine-tuned with no architectural change, does dense
vision tasks where detail matters. Both fine-tunes ship with the release (see
[Depth, image restoration and upscaling](#depth-image-restoration-and-upscaling) for usage).

**Monocular depth** — photo → relative depth, one forward pass:

<table>
<tr>
<td width="25%" valign="top"><img src="assets/downstream/depth-1-photo.webp"><br><sub>Photo</sub></td>
<td width="25%" valign="top"><img src="assets/downstream/depth-1-depth.webp"><br><sub>Iris-3B depth</sub></td>
<td width="25%" valign="top"><img src="assets/downstream/depth-2-photo.webp"><br><sub>Photo</sub></td>
<td width="25%" valign="top"><img src="assets/downstream/depth-2-depth.webp"><br><sub>Iris-3B depth</sub></td>
</tr>
</table>

**Image restoration and upscaling** — degraded input → restored image:

<table>
<tr>
<td width="25%" valign="top"><img src="assets/downstream/restore-1-input.webp"><br><sub>Input</sub></td>
<td width="25%" valign="top"><img src="assets/downstream/restore-1-iris.webp"><br><sub>Iris-3B</sub></td>
<td width="25%" valign="top"><img src="assets/downstream/restore-2-input.webp"><br><sub>Input</sub></td>
<td width="25%" valign="top"><img src="assets/downstream/restore-2-iris.webp"><br><sub>Iris-3B</sub></td>
</tr>
</table>

Interactive depth point clouds and before/after sliders: [project page](https://speridlabs.com/research/iris#downstream)
· [demo](https://huggingface.co/spaces/speridlabs/iris-3b).

## Model

<div align="center"><img src="assets/arch.svg" width="720" alt="Iris-3B architecture"></div>

| | |
|---|---|
| Parameters | 3B (excluding the frozen text encoder) |
| Trunk | 24 blocks at width 2560: 8 dual-stream (separate image/text weights, joint attention) + 16 single-stream; GQA 20 query / 5 key-value heads, gated attention, sandwich RMSNorm, shared-bias adaLN |
| Patch size | 16 |
| Pixel head | 4 pixel-transformer (PiT) blocks: each patch is decoded back to pixels with per-pixel modulation, attending across patches |
| Text encoder | Qwen3-VL-4B-Instruct (frozen), 12 hidden layers aggregated by a learned layer-attention adapter |
| Objective | rectified flow, v-prediction, logit-normal timesteps |

## Install

```bash
pip install -e .            # Python >= 3.11, PyTorch >= 2.7.1
pip install -e ".[dev]"     # + pytest, ruff
```

## Inference

```bash
hf download speridlabs/iris-3b --local-dir iris-3b    # ~12 GB
python scripts/sample.py --checkpoint iris-3b \
    --prompt "a red fox sleeping in fresh snow, golden hour"
```

`--checkpoint` is a directory holding `model.safetensors` + `config.yaml`, or a
training `.pth` (its EMA weights and embedded config are used). Defaults:
1024×1024, 100 DPM-Solver++ (order 2) steps, CFG 3. Use `--txt-file` for a
prompt list, `--steps`, `--seed`, `--negative-prompt` to adjust.

Turn a training checkpoint into a release directory (EMA weights only, FP32 by
default, config reduced to model/text-encoder/flow settings):

```bash
python scripts/export_checkpoint.py output/run/checkpoints/latest.pth exported/iris-3b
```

### Depth, image restoration and upscaling

The release also ships Iris-3B fine-tuned for monocular depth estimation and
for image restoration and upscaling — the general-vision-learner side of
the model — in the `depth/` and `upscaler/` folders of the same Hugging Face
repo (~12 GB each). Both run the full model in
a single forward pass with the empty prompt, so no text encoder is loaded.

```bash
python scripts/depth.py photo.jpg --out depth_out      # <name>.npy + colorized <name>.png
python scripts/upscale.py photo.jpg --out upscaled     # <name>_x4.png
```

- **Depth** returns relative log depth at the input resolution (−1 near, +1
  far). It is affine-invariant, not metric: fit a scale and shift in log space
  to compare with ground truth. Inputs above 1024 px on the long side are
  downscaled for the model (`--max-side 0` keeps native resolution).
- **Restoration** is a one-step restorer trained on 1024×1024 crops (4× of
  256×256 inputs). The input is upsampled bicubically to 4×; outputs smaller
  than 1024 on the short side are run as one tile, larger ones as overlapping
  1024×1024 tiles (stride 512) blended with a Gaussian window, followed by a
  wavelet colour fix against the bicubic upsample. By default inputs are first
  downscaled to at most 512 px short side / 1024 px long side, which caps the
  output at 2048×4096 (21 tiles); `--no-budget` keeps the input size.

Python API: `iris3b.downstream.depth.DepthPredictor` and
`iris3b.downstream.restoration.Restorer`. Fine-tuning checkpoints are converted
with `scripts/export_downstream.py`.

### Demo

The Gradio app behind the [Hugging Face Space](https://huggingface.co/spaces/speridlabs/iris-3b)
(text-to-image, depth and upscaling tabs) is in `demo/`; it runs on ZeroGPU
there and on any local GPU. All three models together take ~45 GB of GPU memory;
with `IRIS_OFFLOAD=1` they stay on the CPU and each request moves only the model it
needs to the GPU (~20 GB peak), which is what the Space does:

```bash
pip install -e ".[ui]"
python demo/app.py          # IRIS_MODEL_REPO=<dir or repo id> to use other weights
```

## Training

### Data

Training reads tar shards indexed by a `wids-meta.json`. Each sample is an
image plus a JSON with its `caption`, `height` and `width`. Build them from a
folder of images with `.txt`/`.json` caption sidecars, or from a Hugging Face
dataset:

```bash
python scripts/prepare_wids.py folder /path/to/images --out /data/my-set
python scripts/prepare_wids.py hf user/dataset --split train \
    --image-column image --caption-column text --out /data/my-set
```

### Recipe

Iris-3B was trained in three resolution stages followed by supervised
fine-tuning, each resuming from the previous one. The configs list only what
differs from the defaults in `src/iris3b/config.py`:

| Config | Resolution | Steps | Global batch | Shift | Notes |
|---|---|---|---|---|---|
| `configs/iris3b/stage1_256.yaml` | 256² | 0 → 365K | 1024 | 2 | REPA (DINOv2-B/14, weight 0.5) |
| `configs/iris3b/stage2_512.yaml` | 512, multi-aspect | 365K → 530K | 1024 | 3 | |
| `configs/iris3b/stage3_1024.yaml` | 1024, multi-aspect | 530K → 625K | 512 | 4 | LR 5e-5 |
| `configs/iris3b/sft_1024.yaml` | 1024, multi-aspect | 625K → 665K | 512 | 4 | constant LR 4e-5, no warmup |

Common to all stages: hybrid Muon — Muon (momentum 0.95, Nesterov, RMS-matched
LR) for hidden weight matrices, AdamW (β₁ 0.9, β₂ 0.95) for embeddings, norms and
biases — with weight decay 0, gradient clipping at 0.5 and a constant LR of 1e-4
after a 2K-step linear warmup unless noted. EMA decay 0.9999, 10% caption dropout
for classifier-free guidance, logit-normal (0, 1) timesteps with the
resolution-dependent shift above. From the 512 stage on, RoPE uses isotropic
coordinates for non-square images. bf16, FSDP2 hybrid sharding, `torch.compile`.

Single node:

```bash
torchrun --nproc_per_node=8 scripts/train.py --config configs/iris3b/stage1_256.yaml \
    data.data_dirs=[/data/my-set] train.expected_global_batch=128
```

Multi-node (run on every node with its own `NODE_RANK`):

```bash
NNODES=8 NODE_RANK=0 MASTER_ADDR=node0 \
    scripts/launch_multinode.sh configs/iris3b/stage1_256.yaml data.data_dirs=[/data/my-set]
```

Any config key can be overridden as `key=value` after the config path.

### Tests

```bash
pytest -q
```

## Acknowledgements

The dual-level design (a patch-level transformer followed by a pixel-level
transformer head) follows the architecture described in
[PixelDiT](https://arxiv.org/abs/2511.20645). This is an independent
implementation; it contains no code from that project. We also build on
[DiT](https://arxiv.org/abs/2212.09748),
[REPA](https://arxiv.org/abs/2410.06940),
[Muon](https://kellerjordan.github.io/posts/muon/) through
[Dion](https://github.com/microsoft/dion),
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) and
[DINOv2](https://github.com/facebookresearch/dinov2).

## License

The code in this repository and the model weights on
[Hugging Face](https://huggingface.co/speridlabs/iris-3b) are licensed under the
[Apache License 2.0](LICENSE); see also [NOTICE](NOTICE).

Third-party components are not redistributed here and remain under their own
licenses:

| Component | Use | License |
|---|---|---|
| [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | frozen text encoder, downloaded at runtime | Apache 2.0 |
| [DINOv2](https://github.com/facebookresearch/dinov2) | REPA teacher (training only), downloaded at runtime | Apache 2.0 |
| [Dion](https://github.com/microsoft/dion) | Muon optimizer, pip dependency | MIT |

## Citation

```bibtex
@article{licai2026iris,
  title   = {Iris-3B: Going Beyond the Latent with Pixel-Space
             Diffusion Training, Conversion and Fine-Tuning},
  author  = {Li Cai, Hanqiu and Garabito, Chema},
  journal = {arXiv preprint arXiv:2610.09450},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.09450}
}
```

---

**Made with ❤️ by [speridlabs.com](https://speridlabs.com)**
