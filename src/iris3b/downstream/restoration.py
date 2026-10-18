"""Image restoration and upscaling with Iris-3B.

The restorer is a one-step model. It takes the low-quality image upsampled
bicubically to the target size, mapped to [-1, 1], and runs a single forward
pass at model time 500 (sigma 0.5) with the empty prompt; the restored image is
``x - 0.5 * v``. It was fine-tuned on 1024x1024 crops (4x of 256x256 inputs)
and only runs at that tile size:

- an output whose short side is below 1024 is enlarged to 1024 for the pass
  and resized back afterwards, so a 256x256 input is exactly one tile;
- a larger output is covered by overlapping 1024x1024 tiles at stride 512
  (50% overlap, the last tile flush with the far edge), fused with a Gaussian
  window, so every pixel is a weighted average of the tiles that cover it.

A final wavelet colour fix keeps the restored high frequencies and takes the
low frequencies (colour, illumination) from the bicubic upsample, which a
one-step restorer otherwise lets drift.

Compute grows with the output area: 4x of a 512x512 input is a 2048x2048
output, 9 tiles; 512x1024 gives 21 tiles. ``fit_budget`` caps the input so a
request stays within a fixed number of tiles.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from iris3b.downstream import load_export
from iris3b.models.dit import IrisDiT

# colour fix: a-trous binomial pyramid, effective radius 16 px
_DILATIONS = (1, 2, 4, 8, 16)
_KERNEL = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0


def _kernel(image: torch.Tensor) -> torch.Tensor:
    return _KERNEL.to(image)[None, None].repeat(image.shape[1], 1, 1, 1)


def _blur(image: torch.Tensor, dilation: int, kernel: torch.Tensor) -> torch.Tensor:
    padded = F.pad(image, (dilation,) * 4, mode="replicate")
    return F.conv2d(padded, kernel, groups=image.shape[1], dilation=dilation)


def _high(image: torch.Tensor) -> torch.Tensor:
    """High-frequency part; with ``_low`` it sums back to the input."""
    kernel = _kernel(image)
    high, low = torch.zeros_like(image), image
    for dilation in _DILATIONS:
        blurred = _blur(low, dilation, kernel)
        high, low = high + (low - blurred), blurred
    return high


def _low(image: torch.Tensor) -> torch.Tensor:
    kernel = _kernel(image)
    for dilation in _DILATIONS:
        image = _blur(image, dilation, kernel)
    return image


def wavelet_color_fix(content: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """``content``'s detail on ``reference``'s colour; both the same size."""
    return _high(content) + _low(reference)


def tile_positions(size: int, tile: int, stride: int) -> list[int]:
    """Start offsets along one axis; the last tile is flush with the far edge."""
    if size <= tile:
        return [0]
    return [*range(0, size - tile, stride), size - tile]


def gaussian_window(tile: int, device: torch.device) -> torch.Tensor:
    """[1, 1, tile, tile] fusion weights (variance 0.01 in tile-normalized coordinates).

    The vertical centre is ``tile / 2`` and the horizontal ``(tile - 1) / 2``;
    the reported results were produced with exactly these weights.
    """
    index = torch.arange(tile, device=device, dtype=torch.float32)
    cols = torch.exp(-((index - (tile - 1) / 2) ** 2) / (tile * tile) / 0.02)
    rows = torch.exp(-((index - tile / 2) ** 2) / (tile * tile) / 0.02)
    return (rows[:, None] * cols[None, :])[None, None]


def tiled(fn, x: torch.Tensor, tile: int, stride: int) -> torch.Tensor:
    """Apply ``fn`` to overlapping ``tile`` crops of [1, C, H, W] and fuse them."""
    height, width = x.shape[-2:]
    if height <= tile and width <= tile:
        return fn(x)
    window = gaussian_window(tile, x.device)
    out = torch.zeros_like(x)
    weight = torch.zeros_like(x[:, :1])
    for top in tile_positions(height, tile, stride):
        for left in tile_positions(width, tile, stride):
            crop = (..., slice(top, top + tile), slice(left, left + tile))
            out[crop] += fn(x[crop].contiguous()) * window
            weight[crop] += window
    return out / weight


def fit_budget(image: Image.Image, short_side: int = 512, long_side: int = 1024) -> Image.Image:
    """Downscale (never upscale) so the short side is at most ``short_side`` and
    the long side at most ``long_side``; at 4x this caps the output at
    2048 x 4096, i.e. at most 21 tiles."""
    width, height = image.size
    scale = min(1.0, short_side / min(width, height), long_side / max(width, height))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


class Restorer:
    def __init__(self, source: str, device: str | torch.device = "cuda"):
        cfg, settings, weights, prompt = load_export(source, "restoration")
        self.device = torch.device(device)
        with torch.device("meta"):
            model = IrisDiT(cfg.model)
        model.load_state_dict(weights, strict=True, assign=True)
        # FP32 weights; inference runs under BF16 autocast
        self.model = model.eval().requires_grad_(False).to(self.device, torch.float32)
        self.embeddings = prompt["embeddings"].to(self.device, torch.float32)
        self.mask = prompt["mask"].to(self.device, torch.bool)
        self.patch = cfg.model.patch_size
        self.tile = settings["tile"]
        self.sigma = settings["sigma"]
        self.time = settings["sigma"] * cfg.flow.num_train_timesteps
        if cfg.flow.prediction != "v":
            raise ValueError("the restorer expects a v-prediction parent")

    def to(self, device: str | torch.device) -> "Restorer":
        self.device = torch.device(device)
        self.model.to(self.device)
        self.embeddings, self.mask = self.embeddings.to(self.device), self.mask.to(self.device)
        return self

    def restore_tile(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, tile, tile] in [-1, 1] -> restored, same range and size."""
        t = torch.full((x.shape[0],), self.time, device=x.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            y = self.embeddings.expand(len(x), *self.embeddings.shape[1:])
            v = self.model(x, t, y, y_mask=self.mask.expand(len(x), -1)).x
        return x.float() - self.sigma * v.float()

    @torch.inference_mode()
    def __call__(self, image: Image.Image, scale: float = 4.0, color_fix: bool = True) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        lq = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].to(self.device, torch.float32) / 255
        reference = F.interpolate(lq, scale_factor=scale, mode="bicubic", align_corners=False)
        out_size = reference.shape[-2:]
        x = reference
        if min(out_size) <= self.tile:
            ratio = self.tile / min(out_size)
            work = tuple(max(self.tile, round(side * ratio)) for side in out_size)
            x = F.interpolate(x, size=work, mode="bicubic", align_corners=False, antialias=True)
        height, width = x.shape[-2:]
        x = F.pad(x * 2 - 1, (0, -width % self.patch, 0, -height % self.patch))
        y = tiled(self.restore_tile, x, self.tile, self.tile // 2)
        y = (y[..., :height, :width] + 1) / 2
        if (height, width) != tuple(out_size):
            y = F.interpolate(y, size=out_size, mode="bicubic", align_corners=False, antialias=True)
        if color_fix:
            y = wavelet_color_fix(y, reference)
        pixels = (y.clamp(0, 1)[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
        return Image.fromarray(pixels)
