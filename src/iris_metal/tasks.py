"""GPU fine-tunes using upstream image preprocessing."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file

from iris3b.config import inference_config
from iris3b.downstream.depth import DepthPredictor, IrisDepth
from iris3b.downstream.restoration import Restorer, tile_positions
from iris3b.models.dit import IrisDiT
from iris_metal.loading import load_weights
from iris_metal.model import MetalIris, require_finite

TASK_FOLDERS = {"depth": "depth", "restoration": "upscaler"}


def resolve_task(source: str | Path, task: str, revision: str | None) -> Path:
    """A local task export (or a local copy of the repo holding it), or its folder of
    a Hugging Face repo (only that folder is downloaded)."""
    folder = TASK_FOLDERS[task]
    if Path(source).is_dir():
        nested = Path(source) / folder
        return nested if (nested / "config.yaml").is_file() else Path(source)
    from huggingface_hub import snapshot_download

    root = snapshot_download(str(source), revision=revision, allow_patterns=[f"{folder}/*"])
    return Path(root) / folder


MAX_UPSCALE_ASPECT = 4  # the model's widest native size, 2048 x 512


def check_upscale_input(image: Image.Image) -> None:
    """Reject inputs the restorer cannot bound. It widens the short side of its
    4x output to a full tile, so a thin strip becomes an enormous work area
    (1 x 1024 pixels -> 1024 x 1048576)."""
    if max(image.size) > MAX_UPSCALE_ASPECT * min(image.size):
        raise ValueError(
            f"images to upscale may be at most {MAX_UPSCALE_ASPECT}:1, not {image.width} x {image.height}"
        )


def depth_image(depth: np.ndarray) -> Image.Image:
    """16-bit grayscale, near = bright, over the prediction's 2nd-98th percentile
    range (upstream ``colorize`` without its matplotlib colormap)."""
    low, high = np.percentile(depth, [2, 98])
    near = np.clip((high - depth) / max(high - low, 1e-6), 0, 1)
    return Image.fromarray((near * 65535).round().astype(np.uint16))


def _read_export(path: Path, task: str) -> tuple[object, dict, dict]:
    raw = OmegaConf.to_container(OmegaConf.load(path / "config.yaml"))
    settings = raw.pop("task", None)
    if settings is None:
        raise ValueError(f"{path} is not a depth or upscaling export (its config.yaml has no task)")
    if settings["name"] != task:
        raise ValueError(f"{path} holds a {settings['name']!r} export, not {task!r}")
    return inference_config(raw, []), settings, load_file(path / "empty_prompt.safetensors")


class MetalDepth(DepthPredictor):
    """Relative log depth (-1 near, +1 far); see ``DepthPredictor.__call__``."""

    def __init__(self, export: Path, device: str = "mps", attention_dtype: torch.dtype = torch.float16):
        cfg, _, prompt = _read_export(export, "depth")
        self.device = torch.device(device)
        with torch.device("meta"):
            model = IrisDepth(cfg.model)
        load_weights(model, export / "model.safetensors", self.device)
        engine = MetalIris(model.pixel, attention_dtype=attention_dtype)
        text = engine.encode_text(prompt["embeddings"], prompt["mask"])
        time = float(cfg.flow.num_train_timesteps)
        reducer = model.depth_reducer

        def predict(rgb: torch.Tensor, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            x = torch.cat([rgb, rgb.new_zeros(rgb.shape[0], 1, *rgb.shape[2:])], dim=1)
            t = rgb.new_full((rgb.shape[0],), time)
            return require_finite(reducer(engine(x, t, text.expand(rgb.shape[0], -1, -1))))

        # the upstream __call__ evaluates self.model(rgb, self.embeddings, self.mask)
        self.model = predict
        self.embeddings = self.mask = None
        self.patch = cfg.model.patch_size


class MetalRestorer(Restorer):
    """One-step restoration and 4x upscaling; see ``Restorer.__call__``.

    ``on_tile``, when set, runs before every tile; the server reports progress
    from it, and raising from it stops the image between tiles (cancellation).
    """

    on_tile: Callable[[], None] | None = None

    def __init__(self, export: Path, device: str = "mps", attention_dtype: torch.dtype = torch.float16):
        cfg, settings, prompt = _read_export(export, "restoration")
        if cfg.flow.prediction != "v":
            raise ValueError("the restorer expects a v-prediction parent")
        self.device = torch.device(device)
        with torch.device("meta"):
            model = IrisDiT(cfg.model)
        load_weights(model, export / "model.safetensors", self.device)
        self.engine = MetalIris(model, attention_dtype=attention_dtype)
        self.text = self.engine.encode_text(prompt["embeddings"], prompt["mask"])
        self.patch = cfg.model.patch_size
        self.tile = settings["tile"]
        self.sigma = settings["sigma"]
        self.time = settings["sigma"] * cfg.flow.num_train_timesteps

    def tile_count(self, size: tuple[int, int]) -> int:
        """How many tiles ``__call__`` runs at 4x for an input of ``size`` (width, height),
        for reporting progress; it repeats the upstream work-area arithmetic."""
        out = [side * 4 for side in reversed(size)]
        if min(out) <= self.tile:
            ratio = self.tile / min(out)
            out = [max(self.tile, round(side * ratio)) for side in out]
        height, width = (side + -side % self.patch for side in out)
        if height <= self.tile and width <= self.tile:
            return 1
        stride = self.tile // 2
        return len(tile_positions(height, self.tile, stride)) * len(tile_positions(width, self.tile, stride))

    @torch.no_grad()
    def restore_tile(self, x: torch.Tensor) -> torch.Tensor:
        if self.on_tile is not None:  # the previous tile has synced (require_finite)
            self.on_tile()
        t = torch.full((x.shape[0],), self.time, device=x.device)
        v = self.engine(x, t, self.text.expand(x.shape[0], -1, -1))
        return require_finite(x.float() - self.sigma * v)
