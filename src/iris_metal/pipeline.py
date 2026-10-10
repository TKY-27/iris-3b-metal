"""Text-to-image generation with Iris-3B on an Apple GPU."""

import gc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from iris3b.flow.solver import FlowDPMSolver
from iris_metal.loading import load_config, load_dit
from iris_metal.model import MetalIris, require_finite
from iris_metal.text import StreamingQwen3VLEncoder

Encoding = tuple[torch.Tensor, torch.Tensor]  # (states, mask)


def release_memory() -> None:
    """Return freed GPU memory to the system, e.g. after dropping a model."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def to_images(samples: torch.Tensor) -> list[Image.Image]:
    """``[B, 3, H, W]`` in ``[-1, 1]`` -> 8-bit RGB images (round to nearest)."""
    pixels = ((samples.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return [Image.fromarray(np.ascontiguousarray(x.permute(1, 2, 0).cpu().numpy())) for x in pixels]


class StepReporter:
    """Reports each sampling step once the GPU has finished it.

    The host queues steps far ahead of the GPU, so a callback run when a step is
    queued would show progress that has not happened and would check for
    cancellation after the last step had already been queued. Each step records
    an event and the next step waits for it, which keeps one step queued and the
    GPU busy.
    """

    def __init__(self, total: int, on_step: Callable[[int, int], None] | None, device: torch.device):
        self.total = total
        self.on_step = on_step
        self.use_events = device.type == "mps"
        self.done = 0
        self._pending: torch.mps.Event | None = None

    def queued(self) -> None:
        """Call after queuing a step."""
        if not self.use_events:  # the CPU runs each step before returning
            self._report()
            return
        event = torch.mps.Event()
        event.record()
        if self._pending is not None:
            self._pending.synchronize()
            self._report()
        self._pending = event

    def finish(self) -> None:
        """Wait for the last queued step and report it."""
        if self._pending is not None:
            self._pending.synchronize()
            self._pending = None
            self._report()

    def _report(self) -> None:
        self.done += 1
        if self.on_step is not None:
            self.on_step(self.done, self.total)


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
    def loaded(self) -> bool:
        return self._engine is not None

    def unload(self) -> None:
        """Release the diffusion model; the next generation loads it again."""
        if self._engine is not None:
            self._engine = None
            release_memory()

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

        reporter = StepReporter(steps, on_step, self.device)

        def velocity(x: torch.Tensor, t: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
            out = engine(x, t, text)
            reporter.queued()
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
        reporter.finish()
        return to_images(require_finite(x))
