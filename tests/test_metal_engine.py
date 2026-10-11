import os

import pytest
import torch
from safetensors.torch import save_file

from iris3b.config import Config, ModelConfig, PixelStageConfig
from iris3b.models.dit import IrisDiT
from iris_metal.kernels import (
    gated_residual_rms_norm_,
    pixel_mlp_,
    pixel_rms_norm,
    qk_norm_rope,
    qk_norm_rope_reference,
    rms_norm_modulate,
)
from iris_metal.loading import linear_dtypes, load_dit
from iris_metal.model import MetalIris, require_finite

# GitHub's macOS runners have a virtual GPU on which some MPS kernels neither compile nor
# match an Apple Silicon GPU; CI sets IRIS_SKIP_GPU_TESTS.
requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available() or os.environ.get("IRIS_SKIP_GPU_TESTS") == "1",
    reason="needs an Apple Silicon GPU",
)

# the released Iris-3B topology at toy width; heads stay 128 wide like the release
RELEASED_TOPOLOGY = ModelConfig(
    block="single_stream",
    dual_depth=1,
    hidden_size=256,
    depth=3,
    num_heads=2,
    num_kv_heads=1,
    gated_attention=True,
    sandwich_norm=True,
    patch_size=4,
    modulation="shared_bias",
    adaln_zero_init=False,
    rope_aspect="isotropic",
    text_dim=64,
    text_len=16,
    text_adapter="lap_blocks2",
    text_lap_num_layers=3,
    text_lap_num_heads=4,
    repa_layer=1,
    pixel=PixelStageConfig(depth=2, hidden_size=16, attn_hidden_size=256, num_heads=2, modulation="post"),
)


def _randomized_model(seed: int = 0) -> IrisDiT:
    torch.manual_seed(seed)
    model = IrisDiT(RELEASED_TOPOLOGY).eval()
    with torch.no_grad():
        # default init leaves the head at zero; perturb everything so every path matters
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.05)
    return model


def _inputs(batch: int = 2, size: int = 24, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    cfg = RELEASED_TOPOLOGY
    x = torch.randn(batch, 3, size, size, generator=gen)
    t = torch.tensor([125.0, 850.0][:batch])
    states = torch.randn(batch, cfg.text_len, cfg.text_lap_num_layers, cfg.text_dim, generator=gen)
    mask = torch.zeros(batch, cfg.text_len, dtype=torch.long)
    mask[0, :5] = 1
    mask[-1, :11] = 1
    return x, t, states * mask[:, :, None, None], mask


def test_fp32_forward_matches_upstream():
    model = _randomized_model()
    x, t, states, mask = _inputs()
    with torch.no_grad():
        expected = model(x, t, states, y_mask=mask).x
        engine = MetalIris(model)
        actual = engine(x, t, engine.encode_text(states, mask))
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_non_square_grid_matches_upstream():
    model = _randomized_model()
    x, t, states, mask = _inputs()
    x = x[..., :16]  # 6 x 4 patch grid
    with torch.no_grad():
        expected = model(x, t, states, y_mask=mask).x
        engine = MetalIris(model)
        actual = engine(x, t, engine.encode_text(states, mask))
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


DTYPES = [torch.float32, torch.bfloat16, torch.float16]


@requires_mps
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("out_dtype", DTYPES)
def test_qk_norm_rope_kernel_matches_reference(dtype, out_dtype):
    # every pair is reachable: trunk or pixel-stage projections times --attention
    gen = torch.Generator().manual_seed(0)
    batch, tokens, heads = 2, 37, 3
    fused = torch.randn(batch, tokens, 2 * heads * 128, generator=gen).to("mps", dtype)
    weight = (torch.rand(128, generator=gen) + 0.5).to("mps")
    angles = torch.rand(tokens + 5, 64, generator=gen) * 6.0
    cos, sin = angles.cos().to("mps"), angles.sin().to("mps")

    out = torch.zeros(batch, heads, tokens + 5, 128, device="mps", dtype=out_dtype)
    qk_norm_rope(fused, weight, cos, sin, 1e-6, heads, col_offset=heads * 128, out=out, offset=5)
    expected = qk_norm_rope_reference(fused[..., heads * 128 :].float(), weight, cos[5:], sin[5:], 1e-6)
    tolerance = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 8e-3}[out_dtype]
    torch.testing.assert_close(out[:, :, 5:].float(), expected, rtol=tolerance, atol=1e-5)
    assert torch.all(out[:, :, :5] == 0)


@requires_mps
def test_mps_forward_matches_upstream():
    model = _randomized_model()
    x, t, states, mask = _inputs()
    with torch.no_grad():
        expected = model(x, t, states, y_mask=mask).x
        engine = MetalIris(model.to("mps"))
        text = engine.encode_text(states, mask)
        actual = engine(x.to("mps"), t.to("mps"), text).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)


@requires_mps
@pytest.mark.parametrize("branch_dtype", [torch.float32, torch.bfloat16])
def test_row_kernels_match_reference(branch_dtype):
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2, 7, 384, generator=gen)
    branch = torch.randn(2, 7, 384, generator=gen).to(branch_dtype)
    weight = torch.rand(384, generator=gen) + 0.5
    shift, scale, gate = (torch.randn(2, 1, 384, generator=gen) for _ in range(3))
    to_mps = lambda *ts: [t.to("mps") for t in ts]  # noqa: E731

    expected = rms_norm_modulate(x, weight, shift, scale, 1e-6, torch.bfloat16)
    actual = rms_norm_modulate(*to_mps(x, weight, shift, scale), 1e-6, torch.bfloat16)
    torch.testing.assert_close(actual.cpu(), expected, rtol=8e-3, atol=1e-5)

    expected = gated_residual_rms_norm_(x.clone(), branch, weight, gate, 1e-6)
    actual = gated_residual_rms_norm_(*to_mps(x.clone(), branch, weight, gate), 1e-6)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)


@requires_mps
def test_kernels_reject_non_fp32_gains():
    x = torch.randn(1, 4, 128, device="mps")
    gain = torch.ones(128, device="mps", dtype=torch.bfloat16)
    zeros = torch.zeros(1, 1, 128, device="mps")
    with pytest.raises(ValueError, match="FP32"):
        rms_norm_modulate(x, gain, zeros, zeros, 1e-6, torch.bfloat16)


@requires_mps
def test_pixel_kernels_match_reference():
    gen = torch.Generator().manual_seed(0)
    bf16 = torch.bfloat16
    pixels = torch.randn(5, 256, 16, generator=gen).to(bf16)
    branch = torch.randn(5, 256, 16, generator=gen).to(bf16)
    mods = (torch.randn(5, 256, 64, generator=gen) * 0.3).to(bf16)
    norm_weight, next_weight = (torch.rand(16, generator=gen) + 0.5 for _ in range(2))
    fc1, fc2 = torch.nn.Linear(16, 64).to(bf16), torch.nn.Linear(64, 16).to(bf16)

    expected_pixels = pixels.clone()
    expected = pixel_mlp_(expected_pixels, branch, mods, norm_weight, fc1, fc2, next_weight, 1e-6)
    actual_pixels = pixels.to("mps")
    actual = pixel_mlp_(
        actual_pixels,
        branch.to("mps"),
        mods.to("mps"),
        norm_weight.to("mps"),
        fc1.to("mps"),
        fc2.to("mps"),
        next_weight.to("mps"),
        1e-6,
    )
    # one BF16 ulp where rounding order differs
    torch.testing.assert_close(actual_pixels.cpu(), expected_pixels, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        pixel_rms_norm(pixels.to("mps"), norm_weight.to("mps"), 1e-6).cpu(),
        pixel_rms_norm(pixels, norm_weight, 1e-6),
        rtol=1e-2,
        atol=1e-2,
    )


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def _save_checkpoint(model: IrisDiT, directory) -> Config:
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, directory / "model.safetensors")
    return Config(model=RELEASED_TOPOLOGY)


def test_load_dit_precision_plan(tmp_path):
    model = _randomized_model()
    loaded = load_dit(tmp_path, _save_checkpoint(model, tmp_path), torch.device("cpu"))
    linear = linear_dtypes(loaded, torch.float16)
    for name, param in loaded.named_parameters():
        assert param.dtype == linear.get(name, torch.float32), name
        torch.testing.assert_close(param, model.get_parameter(name).to(param.dtype), rtol=0, atol=0)
    assert linear["blocks.1.q_proj.weight"] == torch.float16  # trunk
    assert linear["pixel_blocks.0.adaln.weight"] == torch.bfloat16  # pixel stage: range
    assert linear["y_embedder.refiner.proj.weight"] == torch.bfloat16  # text adapter
    engine = MetalIris(loaded)  # the plan is exactly what the engine accepts
    assert (engine.dtype, engine.pixel_dtype, engine.text_dtype) == (
        torch.float16,
        torch.bfloat16,
        torch.bfloat16,
    )


@requires_mps
def test_default_path_is_within_20_percent_of_upstream_autocast_error(tmp_path):
    """The default configuration (FP16 trunk, BF16 pixel stage, fused kernels, FP16
    attention) against upstream under BF16 autocast, both measured against FP32."""
    model = _randomized_model()
    x, t, states, mask = _inputs()
    with torch.no_grad():
        exact = model(x, t, states, y_mask=mask).x
        gpu = [x.to("mps"), t.to("mps"), states.to("mps")]
        upstream = model.to("mps")
        with torch.autocast("mps", dtype=torch.bfloat16):
            reference = upstream(*gpu, y_mask=mask.to("mps")).x.float().cpu()
        engine = MetalIris(
            load_dit(tmp_path, _save_checkpoint(model.cpu(), tmp_path), torch.device("mps")),
            attention_dtype=torch.float16,
        )
        actual = engine(gpu[0], gpu[1], engine.encode_text(states, mask)).cpu()
    reference_error = _relative_error(reference, exact)
    assert 0 < reference_error < 0.05
    assert _relative_error(actual, exact) < 1.2 * reference_error


def test_require_finite_rejects_overflow():
    ok = torch.zeros(2, 3)
    assert require_finite(ok) is ok
    for bad in (float("inf"), float("nan")):
        with pytest.raises(FloatingPointError):
            require_finite(torch.tensor([0.0, bad]))
