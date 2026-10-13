# Changelog

## Unreleased (engine and macOS app)

- Add the official Iris-3B Turbo full-parameter checkpoint with its fixed four-step deterministic flow-Euler sampler. Download and retain Original and Turbo separately; preserve Original guidance, negative prompts and step controls.
- Save the selected model, sampler, precision, checkpoint provenance and generation settings in PNG metadata and library entries. Advertise engine model capabilities so older engines cannot silently handle Turbo as Original.
- Restore the pointer-reactive liquid field in the gallery background and running tile, with an iris-flower silhouette, reduced-motion support and idle/hidden-window pausing.
- Bound gallery thumbnail decoding and cache memory; use native progress, reduce-motion behavior, keyboard selection and Quick Look. Fix seed-field focus, a split-view resize crash and the prompt bar's position in empty galleries.
- Advance the library format to 2, preserving existing chat history while preventing older apps from dropping Turbo settings. Add same-seed regeneration.
- Replace the chat workspace with a gallery. Every request is generated on its own, so results are now listed by day, newest first, with each image's prompt, seed and steps, instead of as a conversation.
- Move the prompt bar below the gallery with a Generate button; the prompt stays after generating so it can be refined. ⌘N puts the cursor in it.
- Add a details panel (⌘I) with the full prompt, size, steps, guidance, seed (with Copy), creation date, generation time and project, and with Reuse Prompt and Settings and Generate with New Seed.
- Search images by prompt words or exact seed; move single images between projects from their menu, the details panel, or by dragging them onto a project.
- The sidebar lists All Images and projects. Chats of earlier versions keep their images, which appear in the gallery; chat titles and pins are kept in the library file but no longer shown.

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
