"""Load a released Iris-3B checkpoint directory for Apple GPU inference."""

from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from torch import nn

from iris3b.config import Config, inference_config
from iris3b.models.dit import IrisDiT


def load_config(checkpoint: str | Path, overrides: list[str] | None = None) -> Config:
    path = Path(checkpoint) / "config.yaml"
    return inference_config(OmegaConf.to_container(OmegaConf.load(path)), overrides)


def linear_dtypes(model: nn.Module, trunk: torch.dtype) -> dict[str, torch.dtype]:
    """Storage dtype of every linear weight and bias.

    The patch transformer uses ``trunk``. The text adapter and the pixel stage
    stay BF16 whenever the trunk is reduced precision: the pixel stream
    reaches about 1.4e4, too close to FP16's 6.5e4 limit, and neither part is
    sensitive to BF16 rounding.
    """
    bf16 = torch.float32 if trunk == torch.float32 else torch.bfloat16
    bf16_modules = set()
    for dit in (module for module in model.modules() if isinstance(module, IrisDiT)):
        for part in (dit.y_embedder, dit.pixel_embedder, dit.pixel_blocks, dit.final_layer):
            bf16_modules.update(id(sub) for sub in part.modules())
    return {
        f"{name}.{param}": bf16 if id(module) in bf16_modules else trunk
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        for param, _ in module.named_parameters(recurse=False)
    }


def load_weights(
    model: nn.Module, path: str | Path, device: torch.device, dtype: torch.dtype = torch.float16
) -> nn.Module:
    """Stream a safetensors file into ``model``, built on the meta device.

    Linear weights and biases are stored as ``linear_dtypes(model, dtype)``
    says; everything else (norm gains, adaLN biases, position tables) stays
    FP32. Tensors are converted one at a time, so the FP32 checkpoint is never
    resident.
    """
    linear = linear_dtypes(model, dtype)
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        keys = reader.keys()
        for key in keys:
            state[key] = reader.get_tensor(key).to(device).to(linear.get(key, torch.float32))
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False)


def load_dit(
    checkpoint: str | Path, cfg: Config, device: torch.device, dtype: torch.dtype = torch.float16
) -> IrisDiT:
    """The text-to-image model of a released checkpoint directory."""
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    return load_weights(model, Path(checkpoint) / "model.safetensors", device, dtype)
