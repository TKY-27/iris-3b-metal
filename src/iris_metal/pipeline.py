"""Text-to-image generation with Iris-3B on an Apple GPU."""

import gc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import iris3b.text  # noqa: F401  (registers the Qwen3-VL encoder)
from iris3b.flow.solver import FlowDPMSolver
from iris3b.registry import TEXT_ENCODERS
from iris3b.text.base import TextEncoder
from iris_metal.loading import load_config, load_dit
from iris_metal.model import MetalIris


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def to_images(samples: torch.Tensor) -> list[Image.Image]:
    """``[B, 3, H, W]`` in ``[-1, 1]`` -> 8-bit RGB images (round to nearest)."""
    pixels = ((samples.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return [Image.fromarray(np.ascontiguousarray(x.permute(1, 2, 0).cpu().numpy())) for x in pixels]


class IrisPipeline:
    """Loads lazily: the text encoder when a prompt needs encoding, the
    diffusion model when sampling starts.

    The text encoder (about 8 GB) is released after encoding unless
    ``keep_text_encoder`` is set, so a first call never holds both models.
    The most recent prompt encodings are kept on the CPU, so repeating a
    prompt does not reload the encoder.
    """

    ENCODING_CACHE_SIZE = 8

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "mps",
        attention_dtype: torch.dtype = torch.float16,
        keep_text_encoder: bool = False,
    ):
        self.checkpoint = Path(checkpoint)
        self.cfg = load_config(self.checkpoint)
        self.device = torch.device(device)
        self.attention_dtype = attention_dtype
        self.keep_text_encoder = keep_text_encoder
        self._text_encoder: TextEncoder | None = None
        self._engine: MetalIris | None = None
        self._encodings: dict[tuple[str, ...], tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def engine(self) -> MetalIris:
        if self._engine is None:
            dit = load_dit(self.checkpoint, self.cfg, self.device)
            self._engine = MetalIris(dit, attention_dtype=self.attention_dtype)
        return self._engine

    def encode_prompts(
        self, prompts: list[str], negative_prompt: str = ""
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Raw encoder ``(states, mask)`` for the prompts and for the CFG negative."""
        positive_key, negative_key = tuple(prompts), ("negative", negative_prompt)
        missing = [key for key in (positive_key, negative_key) if key not in self._encodings]
        if missing:
            if self._text_encoder is None:
                self._text_encoder = TEXT_ENCODERS.build(
                    self.cfg.text_encoder.name, self.cfg.text_encoder, device=self.device
                )
            for key in missing:
                encoding = (
                    self._text_encoder.null(negative_prompt)
                    if key is negative_key
                    else self._text_encoder.encode(prompts)
                )
                self._encodings[key] = (encoding.embeddings.cpu(), encoding.mask.cpu())
            while len(self._encodings) > self.ENCODING_CACHE_SIZE:
                self._encodings.pop(next(iter(self._encodings)))
            if not self.keep_text_encoder:
                self._text_encoder = None
                release_memory()
        return self._encodings[positive_key], self._encodings[negative_key]

    @torch.no_grad()
    def __call__(
        self,
        prompts: list[str],
        height: int = 1024,
        width: int = 1024,
        steps: int | None = None,
        cfg_scale: float | None = None,
        seed: int = 0,
        negative_prompt: str | None = None,
        on_step: Callable[[int, int], None] | None = None,
    ) -> list[Image.Image]:
        sample = self.cfg.sample
        steps = sample.steps if steps is None else steps
        cfg_scale = sample.cfg_scale if cfg_scale is None else cfg_scale
        negative_prompt = sample.negative_prompt if negative_prompt is None else negative_prompt

        (states, mask), (null_states, null_mask) = self.encode_prompts(prompts, negative_prompt)
        engine = self.engine
        cond = engine.encode_text(states, mask)
        uncond = engine.encode_text(null_states, null_mask).expand_as(cond)

        step = 0

        def velocity(x: torch.Tensor, t: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
            nonlocal step
            out = engine(x, t, text)
            step += 1
            if on_step is not None:
                on_step(step, steps)
            return out

        # CPU noise keeps a seed's image identical across machines
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(len(prompts), 3, height, width, generator=generator)
        solver = FlowDPMSolver(
            velocity,
            num_timesteps=self.cfg.flow.num_train_timesteps,
            cfg_scale=cfg_scale,
            cfg_interval=tuple(sample.cfg_interval),
            prediction=self.cfg.flow.prediction,
        )
        x = solver.sample(
            noise.to(self.device),
            cond,
            uncond if cfg_scale != 1.0 else None,
            steps=steps,
            order=sample.order,
            shift=self.cfg.flow.shift,
        )
        return to_images(x)
