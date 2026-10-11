# Changelog

## 0.4.0 — 2026-10-11 (macOS app; Python engine 0.3.0)

- Validate Metal kernel offsets before launch, and report Python engine and HTTP API versions so the app can flag unsupported combinations.
- Make legacy History import and read-only library handling safer; prevent duplicate imports and avoid starting work that cannot be saved.
- Improve cancellation and result saving by checking cancellation around uploads and writes and waiting for the job task to finish.
- Tighten app job and engine lifecycle handling.
- Check GitHub for compatible stable macOS releases and show their release notes. Updates still require manual installation; automatic installation is not supported.
- This is a source-only release with no application binaries or model checkpoints. The Python engine remains independently versioned at 0.3.0.

## 0.3.0 — 2026-10-11

First public release. Earlier versions were never published; the app's
version history (0.1 and 0.2) only matters for the library migration below.

### Engine

- Runs the unmodified Iris-3B checkpoint (revision `7445443`) on Apple GPUs
  through PyTorch MPS, with fused Metal kernels for everything between the
  matrix multiplies and a forward pass restructured for inference.
- FP16 transformer trunk and attention; BF16 text adapter and pixel stage;
  FP32 residual stream, norms and modulation. Every result is checked for
  overflow.
- Qwen3-VL-4B text encoder streamed one layer at a time from its checkpoint
  files (peak about 1.5 GiB), for the real prompt tokens only.
- One resident diffusion model, cached prompt encodings and cached adapted
  conditioning for recent prompts, the empty negative prompt included.
- Depth estimation and 4× restoration/upscaling with the official
  fine-tunes, downloaded on first use.

### Interfaces

- `iris-metal` command for text-to-image, `depth` and `upscale`; it never
  replaces an existing file unless `-o` names it, and checks every input and
  output before loading a model.
- `IrisPipeline` Python API.
- `iris-metal serve`, a local HTTP API with a bounded job queue, progress,
  cancellation and checks against browser and DNS-rebinding requests.

### macOS app (Iris 0.3)

- Chat workspace: new chats, projects, pinned chats, recents by date, search
  and an All Images library; batches of one to four images.
- Quality presets of 4, 10, 20 (default), 30 and 50 steps, prompt adherence
  presets around the checkpoint's guidance, exact steps, guidance, seed and
  negative prompt.
- Restore and Upscale and depth estimation from any image, with a comparison
  window; per-step and per-tile progress and Stop.
- A library from Iris 0.1 (`History`) is imported once into a chat named
  “Earlier images”.

### Known limitations

- Apple Silicon only; tested on one Mac (M5, 24 GB). Macs with 16 GB are
  untested.
- The app is not distributed as a signed or notarized binary; build it from
  source.
- Fewer than 100 steps change fine detail such as hands; the command line and
  the API default to the model's evaluated 100 steps.
