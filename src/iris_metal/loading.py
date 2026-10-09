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


def linear_parameter_names(model: nn.Module) -> set[str]:
    return {
        f"{name}.{param}"
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        for param, _ in module.named_parameters(recurse=False)
    }


def load_dit(
    checkpoint: str | Path, cfg: Config, device: torch.device, dtype: torch.dtype = torch.bfloat16
) -> IrisDiT:
    """Stream ``model.safetensors`` into an ``IrisDiT`` built on the meta device.

    Linear weights and biases are stored in ``dtype``; for BF16 that is the
    value autocast would round the FP32 master weights to on every call.
    Norm gains, adaLN biases and position tables stay FP32. Tensors are
    converted one at a time, so the FP32 checkpoint is never resident.
    """
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    low_precision = linear_parameter_names(model)
    state = {}
    with safe_open(str(Path(checkpoint) / "model.safetensors"), framework="pt", device="cpu") as reader:
        keys = reader.keys()
        for key in keys:
            target = dtype if key in low_precision else torch.float32
            state[key] = reader.get_tensor(key).to(target).to(device)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False)
