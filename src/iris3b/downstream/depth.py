"""Monocular depth with Iris-3B: one forward pass, relative log depth.

The fine-tuned model sees a 4-channel input (RGB in [-1, 1] plus a zero
channel) at the final timestep with the empty prompt, and a 1x1 convolution
reduces its 3-channel output to one channel. Training regressed the per-image
normalized log depth: ``log(depth)`` with its 2nd/98th percentiles mapped to
-1 (near) and +1 (far). The output is therefore affine-invariant relative log
depth, not metric depth; to compare with ground truth, fit a scale and shift
in log space.

Training images were 480x640 and 352x1216; inference runs at the input's own
aspect ratio, resized to a multiple of the 16-pixel patch size.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn

from iris3b.config import ModelConfig
from iris3b.downstream import load_export
from iris3b.models.dit import IrisDiT


def _widen(linear: nn.Linear, in_features: int) -> nn.Linear:
    return nn.Linear(in_features, linear.out_features, bias=linear.bias is not None)


class IrisDepth(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pixel = IrisDiT(cfg)
        # the extra input channel is concatenated after RGB
        p = cfg.patch_size
        self.pixel.s_embedder.proj = _widen(self.pixel.s_embedder.proj, p * p * (cfg.in_channels + 1))
        self.pixel.pixel_embedder.proj = _widen(self.pixel.pixel_embedder.proj, cfg.in_channels + 1)
        self.depth_reducer = nn.Conv2d(cfg.in_channels, 1, 1)
        self.num_train_timesteps = 1000

    def forward(self, rgb: torch.Tensor, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] RGB in [-1, 1] -> [B, 1, H, W] relative log depth (-1 near, +1 far)."""
        batch, _, height, width = rgb.shape
        x = torch.cat([rgb, rgb.new_zeros(batch, 1, height, width)], dim=1)
        t = rgb.new_full((batch,), float(self.num_train_timesteps))
        out = self.pixel(
            x, t, embeddings.expand(batch, *embeddings.shape[1:]), y_mask=mask.expand(batch, -1)
        ).x
        return self.depth_reducer(out)


class DepthPredictor:
    def __init__(self, source: str, device: str | torch.device = "cuda"):
        cfg, _, weights, prompt = load_export(source, "depth")
        self.device = torch.device(device)
        with torch.device("meta"):
            model = IrisDepth(cfg.model)
        model.num_train_timesteps = cfg.flow.num_train_timesteps
        model.load_state_dict(weights, strict=True, assign=True)
        # FP32 weights; inference runs under BF16 autocast
        self.model = model.eval().requires_grad_(False).to(self.device, torch.float32)
        self.embeddings = prompt["embeddings"].to(self.device, torch.float32)
        self.mask = prompt["mask"].to(self.device, torch.bool)
        self.patch = cfg.model.patch_size

    def to(self, device: str | torch.device) -> "DepthPredictor":
        self.device = torch.device(device)
        self.model.to(self.device)
        self.embeddings, self.mask = self.embeddings.to(self.device), self.mask.to(self.device)
        return self

    @torch.inference_mode()
    def __call__(self, image: Image.Image, max_side: int = 1024) -> np.ndarray:
        """Relative log depth [H, W] at the input resolution (-1 near, +1 far).

        The image is downscaled so its long side is at most ``max_side`` (0 keeps
        it native), resized with Lanczos to a multiple of the patch size, and the
        prediction is resized back bilinearly.
        """
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        scale = min(1.0, max_side / max(width, height)) if max_side else 1.0
        size = tuple(
            max(self.patch, round(side * scale / self.patch) * self.patch) for side in (width, height)
        )
        if size != (width, height):
            image = image.resize(size, Image.Resampling.LANCZOS)
        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].to(self.device, torch.float32)
        rgb = rgb / 255 * 2 - 1
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            depth = self.model(rgb, self.embeddings, self.mask).float()
        if depth.shape[-2:] != (height, width):
            depth = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
        return depth[0, 0].cpu().numpy()


def colorize(depth: np.ndarray, cmap: str = "inferno") -> Image.Image:
    """Near = bright, scaled to the prediction's own 2nd-98th percentile range."""
    from matplotlib import colormaps

    low, high = np.percentile(depth, [2, 98])
    near = np.clip((high - depth) / max(high - low, 1e-6), 0, 1)
    return Image.fromarray((colormaps[cmap](near)[..., :3] * 255).round().astype(np.uint8))
