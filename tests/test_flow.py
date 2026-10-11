import torch

from iris3b.config import FlowConfig
from iris3b.flow import FlowDPMSolver, FlowSchedule, RectifiedFlow
from iris3b.flow.schedule import shift_sigma


def test_schedule_grid_endpoints():
    sched = FlowSchedule(1000, shift=3.0)
    assert sched.sigmas[0] == 0.0
    expected_last = shift_sigma(torch.tensor(0.999, dtype=torch.float64), 3.0).float()
    torch.testing.assert_close(sched.sigmas[-1], expected_last)
    assert sched.model_times.dtype == torch.int64
    assert sched.model_times[0] == 0 and sched.model_times[-1] <= 1000


def test_add_noise_interpolation():
    sched = FlowSchedule(1000, shift=1.0)
    x0 = torch.zeros(2, 3, 8, 8)
    noise = torch.ones_like(x0)
    idx = torch.tensor([0, 999])
    x_t, t_model = sched.add_noise(x0, noise, idx)
    torch.testing.assert_close(x_t[0], torch.zeros(3, 8, 8))  # sigma[0] = 0
    torch.testing.assert_close(x_t[1], torch.full((3, 8, 8), sched.sigmas[999].item()))
    assert t_model.shape == (2,)


def test_shift_map_monotonic():
    s = torch.linspace(0, 0.999, 100, dtype=torch.float64)
    for shift in (1.0, 3.0, 4.0):
        out = shift_sigma(s, shift)
        assert torch.all(out.diff() > 0)
        assert out[0] == 0
        assert torch.all(out >= s - 1e-9) if shift >= 1 else True


def test_training_loss_target():
    """With a model that predicts exactly eps - x0, the loss is zero."""

    class Oracle(torch.nn.Module):
        def __init__(self, x0, noise):
            super().__init__()
            self.v = noise - x0

        def forward(self, x, t, y, **kwargs):
            from iris3b.models.dit import IrisOutput

            return IrisOutput(x=self.v, features={})

    x0 = torch.randn(4, 3, 8, 8)
    noise = torch.randn_like(x0)
    rf = RectifiedFlow(FlowConfig(shift=3.0))
    out = rf.training_loss(Oracle(x0, noise), x0, y=torch.zeros(4, 1, 8), noise=noise)
    assert out.loss.item() < 1e-10


def test_solver_constant_velocity_exact():
    """dx/dt = c integrates to z - t0 * c; the solver must land there exactly."""
    c = torch.randn(1, 3, 8, 8)

    def velocity(x, t_model, y):
        return c.expand_as(x)

    solver = FlowDPMSolver(velocity, cfg_scale=1.0)
    z = torch.randn(1, 3, 8, 8)
    for steps in (1, 2, 5, 20):
        for shift in (1.0, 3.0):
            t0 = solver.time_grid(steps, shift)[0]
            out = solver.sample(z.clone(), cond=torch.zeros(1, 1, 8), steps=steps, shift=shift)
            torch.testing.assert_close(out, z - t0 * c, rtol=1e-4, atol=1e-5)


def test_solver_nfe_and_terminal_projection():
    calls = []

    def velocity(x, t_model, y):
        calls.append(t_model[0].item())
        return x / (t_model[0].item() / 1000.0)  # => x0_hat == 0 everywhere

    solver = FlowDPMSolver(velocity, cfg_scale=1.0)
    out = solver.sample(torch.randn(1, 3, 4, 4), cond=torch.zeros(1, 1, 8), steps=10, shift=3.0)
    assert len(calls) == 10  # NFE == steps, no eval at t=0
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-5, rtol=0)


def test_cfg_interval_gating():
    seen_batches = []

    def velocity(x, t_model, y):
        seen_batches.append(x.shape[0])
        return torch.zeros_like(x)

    solver = FlowDPMSolver(velocity, cfg_scale=5.0, cfg_interval=(0.0, 0.5))
    uncond = torch.zeros(1, 1, 8)
    solver.sample(torch.randn(1, 3, 4, 4), cond=torch.zeros(1, 1, 8), uncond=uncond, steps=8, shift=1.0)
    grid = solver.time_grid(8, 1.0)
    expected = [2 if 0.0 < t < 0.5 else 1 for t in grid[:-1]]
    assert seen_batches == expected


def test_x_prediction_loss_path():
    class OracleX(torch.nn.Module):
        def __init__(self, x0):
            super().__init__()
            self.x0 = x0

        def forward(self, x, t, y, **kwargs):
            from iris3b.models.dit import IrisOutput

            return IrisOutput(x=self.x0, features={})

    x0 = torch.randn(4, 3, 8, 8)
    rf = RectifiedFlow(FlowConfig(prediction="x", shift=1.0))
    # at high sigma the conversion (x_t - x0_hat)/sigma recovers v exactly
    idx = torch.full((4,), 900, dtype=torch.long)
    out = rf.training_loss(OracleX(x0), x0, y=torch.zeros(4, 1, 8), timestep_idx=idx)
    assert out.loss.item() < 1e-6


def test_training_loss_reduction_none_matches_mean():
    class Zero(torch.nn.Module):
        def forward(self, x, t, y, **kwargs):
            class Out:
                pass

            o = Out()
            o.x = torch.zeros_like(x)
            o.features = {}
            return o

    rf = RectifiedFlow(FlowConfig(shift=1.0))
    x0 = torch.randn(4, 3, 8, 8)
    y = torch.zeros(4, 2, 8)
    idx = torch.tensor([0, 250, 500, 999])
    noise = torch.randn_like(x0)
    per = rf.training_loss(Zero(), x0, y, timestep_idx=idx, noise=noise, reduction="none").loss
    mean = rf.training_loss(Zero(), x0, y, timestep_idx=idx, noise=noise).loss
    assert per.shape == (4,)
    torch.testing.assert_close(per.mean(), mean)


def test_released_cfg_intervals_preserve_n2_branch_order_and_nfe():
    for steps, upper, full_calls in (
        (10, 1.0, 10),
        (20, 1.0, 20),
        (10, 0.95, 8),
        (20, 0.95, 16),
        (10, 0.9, 6),
        (20, 0.9, 13),
    ):
        calls = []
        cond = torch.tensor([[10.0], [20.0]])
        uncond = torch.tensor([[-10.0], [-20.0]])

        def velocity(x, t, text, calls=calls, cond=cond, uncond=uncond):
            calls.append(x.shape[0])
            expected = torch.cat([uncond, cond]) if x.shape[0] == 4 else cond
            assert torch.equal(text, expected)
            return torch.zeros_like(x)

        solver = FlowDPMSolver(velocity, cfg_scale=3, cfg_interval=(0, upper))
        assert solver.time_grid(steps, 4)[0] < 1
        solver.sample(torch.ones(2, 3, 2, 2), cond, uncond, steps=steps, order=2, shift=4)
        assert len(calls) == steps
        assert calls.count(4) == full_calls
        assert calls.count(2) == steps - full_calls
