"""Text-to-image generation with Iris-3B on an Apple GPU."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from iris3b.flow.solver import FlowDPMSolver
from iris_metal.loading import load_config, load_dit
from iris_metal.model import MetalIris
from iris_metal.text import StreamingQwen3VLEncoder

Encoding = tuple[torch.Tensor, torch.Tensor]  # (states, mask)


def to_images(samples: torch.Tensor) -> list[Image.Image]:
    """``[B, 3, H, W]`` in ``[-1, 1]`` -> 8-bit RGB images (round to nearest)."""
    pixels = ((samples.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return [Image.fromarray(np.ascontiguousarray(x.permute(1, 2, 0).cpu().numpy())) for x in pixels]


class IrisPipeline:
    """Loads the diffusion model on first use and keeps it resident.

    Prompts are encoded by streaming the text encoder layer by layer, so it
    never holds more than one layer; the encodings of the most recent texts
    are kept on the CPU.
    """

    ENCODING_CACHE_SIZE = 16

    def __init__(
        self, checkpoint: str | Path, device: str = "mps", attention_dtype: torch.dtype = torch.float16
    ):
        self.checkpoint = Path(checkpoint)
        self.cfg = load_config(self.checkpoint)
        self.device = torch.device(device)
        self.attention_dtype = attention_dtype
        self._text_encoder: StreamingQwen3VLEncoder | None = None
        self._engine: MetalIris | None = None
        self._encodings: dict[str, Encoding] = {}

    @property
    def engine(self) -> MetalIris:
        if self._engine is None:
            dit = load_dit(self.checkpoint, self.cfg, self.device)
            self._engine = MetalIris(dit, attention_dtype=self.attention_dtype)
        return self._engine

    def encode(self, texts: list[str]) -> list[Encoding]:
        """Text-encoder ``(states, mask)`` of each text; missing ones are encoded in one pass."""
        for text in texts:
            if text in self._encodings:
                self._encodings[text] = self._encodings.pop(text)  # most recently used last
        missing = list(dict.fromkeys(text for text in texts if text not in self._encodings))
        if missing:
            if self._text_encoder is None:
                self._text_encoder = StreamingQwen3VLEncoder(self.cfg.text_encoder, self.device)
            encoding = self._text_encoder.encode(missing)
            for i, text in enumerate(missing):
                # clones, so an evicted entry does not keep its whole batch alive
                self._encodings[text] = (
                    encoding.embeddings[i : i + 1].clone(),
                    encoding.mask[i : i + 1].clone(),
                )
        result = [self._encodings[text] for text in texts]
        while len(self._encodings) > self.ENCODING_CACHE_SIZE:
            self._encodings.pop(next(iter(self._encodings)))
        return result

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

        guided = cfg_scale != 1.0
        encodings = self.encode([*prompts, negative_prompt] if guided else prompts)
        engine = self.engine
        states, mask = (torch.cat(parts) for parts in zip(*encodings[: len(prompts)], strict=True))
        cond = engine.encode_text(states, mask)
        uncond = engine.encode_text(*encodings[-1]).expand_as(cond) if guided else None

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
            uncond,
            steps=steps,
            order=sample.order,
            shift=self.cfg.flow.shift,
        )
        return to_images(x)
