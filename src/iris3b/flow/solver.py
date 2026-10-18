"""FlowDPM-Solver++: multistep DPM-Solver++ on the rectified-flow schedule.

Data-prediction (x0) formulation with the flow noise schedule
``alpha(t) = 1 - t, sigma(t) = t, lambda(t) = log((1 - t) / t)``.
The model predicts velocity (``x0_hat = x - t * v_hat``) or, with
``prediction="x"``, the clean image directly (``x0_hat = out``).

Time grid: ``sigma = 1 - linspace(1.0, 0.001, steps + 1)`` remapped by the
training resolution shift, traversed descending and ending at exactly t = 0,
so the final (always first-order) update is an exact projection onto the
predicted clean image. NFE == steps.

CFG is applied to the raw model output (affine-equivalent to CFG on the
noise prediction in either parameterization) and is gated to
model-evaluation times strictly inside
``cfg_interval`` (t = 1 is noise).
"""

import math
from collections.abc import Callable

import torch

from iris3b.flow.schedule import shift_sigma

# model(x, t_model, y) -> velocity or clean image, where t_model = 1000 * t
ModelFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _lambda(t: float) -> float:
    if t <= 0.0:
        return math.inf
    return math.log((1.0 - t) / t)


class FlowDPMSolver:
    def __init__(
        self,
        model_fn: ModelFn,
        num_timesteps: int = 1000,
        cfg_scale: float = 1.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        prediction: str = "v",
        reuse_cfg_text: bool = False,
    ):
        """``reuse_cfg_text`` passes one CFG text batch to every guided call
        of a ``sample``; ``model_fn`` must then not modify its text input."""
        self.model_fn = model_fn
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.cfg_interval = cfg_interval
        if prediction not in ("v", "x"):
            raise ValueError(f"unknown prediction type '{prediction}'")
        self.prediction = prediction
        self.reuse_cfg_text = reuse_cfg_text
        self._cfg_text: torch.Tensor | None = None

    def _model_out(
        self, x: torch.Tensor, t: float, cond: torch.Tensor, uncond: torch.Tensor | None
    ) -> torch.Tensor:
        lo, hi = self.cfg_interval
        use_cfg = uncond is not None and self.cfg_scale != 1.0 and lo < t < hi
        rows = 2 * x.shape[0] if use_cfg else x.shape[0]
        t_model = torch.full((rows,), t * self.num_timesteps, device=x.device, dtype=torch.float32)
        if not use_cfg:
            return self.model_fn(x, t_model, cond)
        text = self._cfg_text
        if text is None:
            text = torch.cat([uncond, cond], dim=0)
            if self.reuse_cfg_text:
                self._cfg_text = text
        out = self.model_fn(torch.cat([x, x], dim=0), t_model, text)
        out_uncond, out_cond = out.chunk(2, dim=0)
        return out_uncond + self.cfg_scale * (out_cond - out_uncond)

    def _pred_x0(
        self, x: torch.Tensor, t: float, cond: torch.Tensor, uncond: torch.Tensor | None
    ) -> torch.Tensor:
        out = self._model_out(x, t, cond, uncond)
        if self.prediction == "x":
            return out
        return x - t * out

    @staticmethod
    def time_grid(steps: int, shift: float) -> list[float]:
        sigma = 1.0 - torch.linspace(1.0, 0.001, steps + 1, dtype=torch.float64)
        shifted = shift_sigma(sigma, shift)
        return shifted.flip(0).tolist()  # descending, last value exactly 0

    @staticmethod
    def _first_order(x: torch.Tensor, s: float, t: float, x0: torch.Tensor) -> torch.Tensor:
        h = _lambda(t) - _lambda(s)
        phi1 = math.expm1(-h)
        return (t / s) * x - (1.0 - t) * phi1 * x0

    @staticmethod
    def _second_order(
        x: torch.Tensor,
        prev1: tuple[float, torch.Tensor],
        prev0: tuple[float, torch.Tensor],
        t: float,
    ) -> torch.Tensor:
        (s1, x0_1), (s0, x0_0) = prev1, prev0
        lam_t, lam_0, lam_1 = _lambda(t), _lambda(s0), _lambda(s1)
        h, h0 = lam_t - lam_0, lam_0 - lam_1
        r0 = h0 / h
        d = (x0_0 - x0_1) / r0
        phi1 = math.expm1(-h)
        return (t / s0) * x - (1.0 - t) * phi1 * x0_0 - 0.5 * (1.0 - t) * phi1 * d

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor | None = None,
        steps: int = 50,
        order: int = 2,
        shift: float = 1.0,
    ) -> torch.Tensor:
        """Integrate from noise ``z`` to a sample. Multistep, lower-order final."""
        grid = self.time_grid(steps, shift)
        x = z
        self._cfg_text = None
        history: list[tuple[float, torch.Tensor]] = []
        for i in range(1, steps + 1):
            s, t = grid[i - 1], grid[i]
            x0 = self._pred_x0(x, s, cond, uncond)
            history.append((s, x0))
            # order ramps up over the first steps and back down at the end
            # (lower_order_final), so the terminal t=0 update is first-order:
            # an exact x <- x0 projection.
            step_order = min(i, order, steps + 1 - i)
            if step_order == 1:
                x = self._first_order(x, s, t, x0)
            else:
                x = self._second_order(x, history[-2], history[-1], t)
            if len(history) > 2:
                history.pop(0)
        self._cfg_text = None
        return x
