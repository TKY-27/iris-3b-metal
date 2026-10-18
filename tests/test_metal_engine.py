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


@pytest.mark.parametrize("bf16_linears", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=requires_mps)])
def test_prepared_conditioning_is_exact_and_request_local(batch, device, bf16_linears):
    model = _randomized_model()
    if bf16_linears:  # Turbo's production storage
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.to(torch.bfloat16)
    model = model.to(device)
    autocast = device == "mps" or bf16_linears
    x, t, states, mask = (value.to(device) for value in _inputs(batch=batch))
    x = x[..., :16]
    adapter_calls = []
    core_calls = {name: [] for name in model.modulation_cores}
    handles = [model.y_embedder.register_forward_hook(lambda *_: adapter_calls.append(1))]
    for name, core in model.modulation_cores.items():
        handles.append(core.register_forward_hook(lambda *_, name=name: core_calls[name].append(1)))
    try:
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
            prepared = model.embed_text(states, mask)
            original = prepared.clone()
            for timestep in (t, t + 63):
                adapter_calls.clear()
                for calls in core_calls.values():
                    calls.clear()
                actual = model(
                    x,
                    timestep,
                    states,
                    y_mask=mask,
                    prepared_text=prepared,
                    precompute_modulation=True,
                ).x
                assert not adapter_calls
                assert all(len(calls) == 1 for calls in core_calls.values())
                expected = model(x, timestep, states, y_mask=mask).x
                assert torch.isfinite(actual).all() and _bits(actual).equal(_bits(expected))
                assert _bits(prepared).equal(_bits(original))
            # Equal shapes must not alias different prompts.
            changed_states = states + 0.25
            changed = model.embed_text(changed_states, mask)
            assert not torch.equal(changed, prepared)
            actual = model(
                x, t, changed_states, y_mask=mask, prepared_text=changed, precompute_modulation=True
            ).x
            expected = model(x, t, changed_states, y_mask=mask).x
            assert torch.isfinite(actual).all() and _bits(actual).equal(_bits(expected))
    finally:
        for handle in handles:
            handle.remove()


def test_prepared_conditioning_rejects_training_and_gradients():
    model = _randomized_model()
    x, t, states, mask = _inputs()
    with pytest.raises(ValueError, match="no-grad inference"):
        model(x, t, states, y_mask=mask, precompute_modulation=True)
    with torch.no_grad(), pytest.raises(ValueError, match="no-grad inference"):
        model.train()(x, t, states, y_mask=mask, precompute_modulation=True)


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


def _modulation_layouts(batch: int, width: int, gen: torch.Generator) -> dict[str, list[torch.Tensor]]:
    """(shift, scale, gate) as contiguous rows, chunks of one [B, 1, 6D] tensor (strided,
    offset), a batch-broadcast expand (stride 0), and a misaligned view the wrapper must copy."""
    mods = torch.randn(batch, 1, 6 * width, generator=gen).to("mps")
    chunks = list(mods.chunk(6, dim=-1)[:3])
    one = torch.randn(1, 1, 3 * width, generator=gen).to("mps")
    padded = torch.randn(batch * 3 * width + 1, generator=gen).to("mps")
    misaligned = [padded[1 + i * batch * width :][: batch * width].view(batch, 1, width) for i in range(3)]
    return {
        "contiguous": [c.contiguous() for c in chunks],
        "chunks": chunks,
        "expanded": [c.expand(batch, 1, width) for c in one.chunk(3, dim=-1)],
        "misaligned": misaligned,
    }


@requires_mps
@pytest.mark.parametrize("batch", [1, 2, 3])
def test_row_kernels_read_strided_modulation_exactly(batch):
    from iris_metal.kernels import _per_batch

    gen = torch.Generator().manual_seed(batch)
    width = 384
    x = torch.randn(batch, 5, width, generator=gen).to("mps")
    branch = torch.randn(batch, 5, width, generator=gen).to("mps", torch.bfloat16)
    weight = (torch.rand(width, generator=gen) + 0.5).to("mps")
    for name, (shift, scale, gate) in _modulation_layouts(batch, width, gen).items():
        dense = [t.contiguous() for t in (shift, scale, gate)]
        expected = rms_norm_modulate(x, weight, dense[0], dense[1], 1e-6, torch.float16)
        actual = rms_norm_modulate(x, weight, shift, scale, 1e-6, torch.float16)
        assert actual.view(torch.int16).equal(expected.view(torch.int16)), name
        expected = gated_residual_rms_norm_(x.clone(), branch, weight, dense[2], 1e-6)
        actual = gated_residual_rms_norm_(x.clone(), branch, weight, gate, 1e-6)
        assert actual.view(torch.int32).equal(expected.view(torch.int32)), name
        rows, _ = _per_batch(gate, batch, width)
        in_place = rows.untyped_storage().data_ptr() == gate.untyped_storage().data_ptr()
        assert in_place == (name != "misaligned"), name


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


@requires_mps
@pytest.mark.parametrize("parameter", ["rms_gain", "norm_gain", "next_gain", "fc1_bias", "fc2_bias"])
def test_pixel_kernels_reject_short_parameter_buffers(parameter, monkeypatch):
    import iris_metal.kernels as kernels

    def forbidden():
        pytest.fail("invalid buffers reached Metal compilation")

    monkeypatch.setattr(kernels, "_library", forbidden)
    pixels = torch.zeros(1, 1, 16, device="mps", dtype=torch.bfloat16)
    gain = torch.ones(16, device="mps")
    if parameter == "rms_gain":
        with pytest.raises(ValueError, match="gain width"):
            pixel_rms_norm(pixels, gain[:1], 1e-6)
        return
    mods = torch.zeros(1, 1, 64, device="mps", dtype=torch.bfloat16)
    fc1 = torch.nn.Linear(16, 64, device="mps", dtype=torch.bfloat16)
    fc2 = torch.nn.Linear(64, 16, device="mps", dtype=torch.bfloat16)
    if parameter in ("fc1_bias", "fc2_bias"):
        layer = fc1 if parameter == "fc1_bias" else fc2
        layer.bias = torch.nn.Parameter(layer.bias[:1])
    with pytest.raises(ValueError, match="gain width|pixel-stage widths"):
        pixel_mlp_(
            pixels,
            pixels.clone(),
            mods,
            gain[:1] if parameter == "norm_gain" else gain,
            fc1,
            fc2,
            gain[:1] if parameter == "next_gain" else gain,
            1e-6,
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


@pytest.mark.parametrize("options", [{"col_offset": -1}, {"offset": -1}, {"heads": 0}])
@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=requires_mps)])
def test_qk_norm_rope_rejects_invalid_pointer_offsets(options, device):
    x = torch.ones(1, 2, 128, device=device)
    weight = torch.ones(128, device=device)
    cos = torch.ones(2, 64, device=device)
    sin = torch.zeros_like(cos)
    kwargs = {"heads": 1, **options}
    with pytest.raises(ValueError, match="positive.*nonnegative"):
        qk_norm_rope(x, weight, cos, sin, 1e-6, **kwargs)


def _old_rms_norm(norm, x):
    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + norm.eps)
    return norm.weight * normed.to(x.dtype)


def _bits(t):
    return t.detach().contiguous().cpu().view(torch.uint8)


def _count_copies(fn):
    from torch.utils._python_dispatch import TorchDispatchMode

    class Count(TorchDispatchMode):
        copies = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            Count.copies += func.overloadpacket is torch.ops.aten._to_copy
            return func(*args, **(kwargs or {}))

    with Count():
        fn()
    return Count.copies


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=requires_mps)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_rms_norm_inference_reuses_one_fp32_copy_bitwise(device, dtype):
    from iris3b.nn.norms import RMSNorm

    gen = torch.Generator().manual_seed(4)
    norm = RMSNorm(128).to(device)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(128, generator=gen) + 0.5)
    projection = torch.randn(2, 7, 4 * 128, generator=gen).to(device, dtype)
    projection[0, 0] = -0.0  # signed zeros survive normalization
    projection[0, 1] = 0.0
    heads = projection.reshape(2, 7, 4, 128)
    for x in (heads, heads[:, 3:], projection.reshape(14, 4, 128).transpose(0, 1)):
        with torch.no_grad():
            assert _bits(norm(x)).equal(_bits(_old_rms_norm(norm, x)))
            saved = _count_copies(lambda x=x: _old_rms_norm(norm, x)) - _count_copies(lambda x=x: norm(x))
        assert saved == (dtype != torch.float32)
    x = heads.detach().float().requires_grad_(True).to(dtype)  # grad-enabled path is untouched
    leaf = x.detach().requires_grad_(True)
    expected = torch.autograd.grad(_old_rms_norm(norm, leaf).square().sum(), (leaf, norm.weight))
    actual = torch.autograd.grad(norm(leaf).square().sum(), (leaf, norm.weight))
    assert all(_bits(a).equal(_bits(e)) for a, e in zip(actual, expected, strict=True))


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=requires_mps)])
def test_shared_autocast_operands_are_exact_and_cast_once(batch, device, monkeypatch):
    import iris3b.models.blocks.mmdit as mmdit
    import iris3b.models.blocks.single_stream as single_stream
    import iris3b.nn.mlp as mlp

    model = _randomized_model()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.to(torch.bfloat16)  # BF16 storage, as the Turbo loader keeps
    model = model.to(device).requires_grad_(False)
    x, t, states, mask = (v.to(device) for v in _inputs(batch=batch))

    def forward():
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            prepared = model.embed_text(states, mask)
            return model(x, t, states, y_mask=mask, prepared_text=prepared, precompute_modulation=True).x

    shared, shared_copies = forward(), _count_copies(forward)
    for module in (mlp, mmdit, single_stream):
        monkeypatch.setattr(module, "autocast_operand", lambda tensor: tensor)
    separate, separate_copies = forward(), _count_copies(forward)
    assert _bits(shared).equal(_bits(separate))
    assert shared_copies < separate_copies
    with torch.autocast(device, dtype=torch.bfloat16):  # autograd keeps upstream casts
        assert mlp.autocast_operand(x) is x


def test_autocast_operand_only_casts_under_no_grad_autocast():
    from iris3b.nn.mlp import autocast_operand

    x = torch.randn(3, 8)
    with torch.no_grad():
        assert autocast_operand(x) is x
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert autocast_operand(x).dtype == torch.bfloat16
            assert autocast_operand(x.double()).dtype == torch.float64
            linear = torch.nn.Linear(8, 4)  # FP32 master weights stay valid
            assert _bits(linear(autocast_operand(x))).equal(_bits(linear(x)))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert autocast_operand(x) is x


def test_solver_reuses_one_cfg_text_batch_only_when_asked():
    from iris3b.flow.solver import FlowDPMSolver

    texts, times = [], []

    def velocity(x, t, text):
        texts.append(text)
        times.append(t)
        return x * 0.5 + text.mean() + t[:, None, None, None] * 1e-4

    gen = torch.Generator().manual_seed(0)
    z = torch.randn(2, 3, 8, 8, generator=gen)
    cond, uncond = torch.randn(2, 4, generator=gen), torch.randn(2, 4, generator=gen)
    results = []
    for reuse in (False, True):
        texts.clear()
        solver = FlowDPMSolver(velocity, cfg_scale=3.0, cfg_interval=(0.2, 0.9), reuse_cfg_text=reuse)
        results.append(solver.sample(z, cond, uncond, steps=6, shift=4.0))
        guided = [text for text in texts if text is not cond]
        assert len(guided) < len(texts)  # the interval leaves unguided steps
        assert all(text.equal(torch.cat([uncond, cond])) for text in guided)
        assert len({id(text) for text in guided}) == (1 if reuse else len(guided))
        assert solver._cfg_text is None
    assert results[0].view(torch.int32).equal(results[1].view(torch.int32))
    for t in times:
        assert t.dtype == torch.float32 and t.unique().numel() == 1


def test_shape_caches_are_bounded_at_inference_and_rebuilt_exactly():
    from iris3b.nn.embeddings import SHAPE_CACHE_SIZE, PixelEmbedder

    embedder = PixelEmbedder(3, 16, 4).eval()
    first = embedder._pos(16, 32, torch.device("cpu"), torch.float32).clone()
    for size in range(1, SHAPE_CACHE_SIZE + 2):
        embedder._pos(16, 16 + 16 * size, torch.device("cpu"), torch.float32)
    assert len(embedder._pos_cache) == SHAPE_CACHE_SIZE
    assert _bits(embedder._pos(16, 32, torch.device("cpu"), torch.float32)).equal(_bits(first))
    embedder.train()
    for size in range(SHAPE_CACHE_SIZE + 2):
        embedder._pos(32, 16 + 16 * size, torch.device("cpu"), torch.float32)
    assert len(embedder._pos_cache) > SHAPE_CACHE_SIZE  # training shape buckets stay resident

    model = _randomized_model()
    engine = MetalIris(model)
    sizes = [(16, 16 + 4 * n) for n in range(SHAPE_CACHE_SIZE + 2)]
    for grid in sizes:
        model._fetch_rope_img(grid, torch.device("cpu"))
        model._fetch_rope_pix(grid, torch.device("cpu"))
        model._fetch_rope_txt(grid[1], torch.device("cpu"))
        engine._rope(grid, 7)
    for cache in (model._rope_img, model._rope_pix, model._rope_txt, engine._ropes):
        assert len(cache) == SHAPE_CACHE_SIZE
    rebuilt = engine._rope(sizes[0], 7)[0]
    assert _bits(rebuilt.cos).equal(_bits(MetalIris(model)._rope(sizes[0], 7)[0].cos))


def test_loading_stops_between_tensors_without_a_partial_model(tmp_path):
    cfg = _save_checkpoint(_randomized_model(), tmp_path)
    seen = []

    def stop(after=5):
        seen.append(1)
        if len(seen) > after:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        load_dit(tmp_path, cfg, torch.device("cpu"), raise_if_cancelled=stop)
    assert len(seen) == 6
    model = load_dit(tmp_path, cfg, torch.device("cpu"), raise_if_cancelled=lambda: None)
    assert not any(p.is_meta for p in model.parameters())


@requires_mps
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("batch", "tokens", "width"), [(1, 4396, 2560), (2, 300, 2560), (3, 7, 256)])
def test_fused_norm_tails_are_bitwise_eager(dtype, batch, tokens, width, monkeypatch):
    import iris3b.nn.modulation as modulation
    from iris3b.nn.norms import RMSNorm

    gen = torch.Generator().manual_seed(width + tokens)
    norm = RMSNorm(width).to("mps")
    with torch.no_grad():
        norm.weight.copy_(torch.rand(width, generator=gen) + 0.5)
    x = (torch.randn(batch, tokens, width, generator=gen) * 3).to("mps", dtype)
    x[0, 0, :4] = torch.tensor([-0.0, 0.0, -0.0, 1e-30])
    x[0, -1] = 0.0  # an all-zero row
    shift, scale, gate = torch.randn(batch, 1, 6 * width, generator=gen).to("mps").chunk(6, dim=-1)[:3]
    residual = torch.randn(batch, tokens, width, generator=gen).to("mps")
    branch = (torch.randn(batch, tokens, width, generator=gen) * 4).to("mps", torch.bfloat16)
    fused = []
    original = modulation._fused
    monkeypatch.setattr(modulation, "_fused", lambda *a: fused.append(a[0]) or original(*a))
    with torch.no_grad():
        actual = modulation.norm_modulate(norm, x, shift, scale)
        expected = modulation.modulate(norm(x), shift, scale)
        assert _bits(actual).equal(_bits(expected)) and torch.isfinite(actual).all()
        actual = modulation.gated_norm_residual(residual, gate, norm, branch)
        assert _bits(actual).equal(_bits(residual + gate * norm(branch)))
    assert len(fused) == 2
    with torch.no_grad():  # unsupported inputs take the eager expressions
        modulation.norm_modulate(norm, x.half(), shift, scale)
        modulation.norm_modulate(torch.nn.Identity(), x, shift, scale)
        misaligned = torch.randn(batch * width + 1, device="mps")[1:].view(batch, 1, width)
        modulation.norm_modulate(norm, x, misaligned, scale)
        modulation.gated_norm_residual(residual.bfloat16(), gate, norm, branch)
        modulation.norm_modulate(norm.cpu(), x.cpu(), shift.cpu(), scale.cpu())
    modulation.norm_modulate(norm.to("mps"), x, shift, scale)  # autograd on
    assert len(fused) == 2


@requires_mps
@pytest.mark.parametrize(("rows", "pixels", "width"), [(4096, 256, 16), (5, 16, 16), (3, 7, 256)])
def test_fused_pixel_tails_are_bitwise_eager(rows, pixels, width, monkeypatch):
    import iris3b.nn.modulation as modulation
    from iris3b.nn.norms import RMSNorm

    gen = torch.Generator().manual_seed(rows + width)
    norm = RMSNorm(width).to("mps")
    with torch.no_grad():
        norm.weight.copy_(torch.rand(width, generator=gen) + 0.5)
    x = (torch.randn(rows, pixels, width, generator=gen) * 3).to("mps", torch.bfloat16)
    x[0, 0, :4] = torch.tensor([-0.0, 0.0, -0.0, 1e-30])
    x[0, -1] = 0.0  # an all-zero row
    branch = (torch.randn(rows, pixels, width, generator=gen) * 4).to("mps", torch.bfloat16)
    mods = torch.randn(rows, pixels, 4 * width, generator=gen).to("mps", torch.bfloat16)
    fused = []
    original = modulation._elementwise_out
    monkeypatch.setattr(modulation, "_elementwise_out", lambda *a: fused.append(a[0]) or original(*a))
    with torch.no_grad():
        for scale, shift in (mods.chunk(4, dim=-1)[:2], mods.chunk(4, dim=-1)[2:]):
            actual = modulation.modulated_residual(x, branch, shift, scale)
            assert _bits(actual).equal(_bits(x + modulation.modulate(branch, shift, scale)))
        with torch.autocast("mps", dtype=torch.bfloat16):
            actual = modulation.norm_operand(norm, x)
            assert actual.dtype == torch.bfloat16 and _bits(actual).equal(_bits(norm(x).bfloat16()))
    assert fused == ["modulated_residual"] * 2 + ["norm_operand"]
    with torch.no_grad():  # unsupported inputs take the eager expressions
        scale, shift = mods.chunk(4, dim=-1)[:2]
        modulation.modulated_residual(x.float(), branch, shift, scale)
        modulation.modulated_residual(x, branch, shift, scale.contiguous())
        modulation.modulated_residual(x.cpu(), branch.cpu(), shift.cpu(), scale.cpu())
        assert modulation.norm_operand(norm, x).dtype == torch.float32  # no autocast
        with torch.autocast("mps", dtype=torch.float16):
            modulation.norm_operand(norm, x)
        with torch.autocast("mps", dtype=torch.bfloat16):
            modulation.norm_operand(torch.nn.Identity(), x)
            modulation.norm_operand(norm, x.float())
    with torch.autocast("mps", dtype=torch.bfloat16):  # autograd on
        modulation.norm_operand(norm, x)
        modulation.modulated_residual(x, branch, shift, scale)
    assert len(fused) == 3


@requires_mps
@pytest.mark.parametrize("batch", [1, 2])
def test_fused_norm_tails_keep_the_prepared_forward_bitwise(batch, monkeypatch):
    import iris3b.nn.modulation as modulation

    model = _randomized_model()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.to(torch.bfloat16)  # Turbo's production storage
    model = model.to("mps").requires_grad_(False)
    x, t, states, mask = (v.to("mps") for v in _inputs(batch=batch))

    def forward():
        with torch.no_grad(), torch.autocast("mps", dtype=torch.bfloat16):
            prepared = model.embed_text(states, mask)
            return model(x, t, states, y_mask=mask, prepared_text=prepared, precompute_modulation=True).x

    fused = forward()
    monkeypatch.setattr(modulation, "_fusable", lambda *args: False)
    monkeypatch.setattr(modulation, "_elementwise", lambda *args: False)
    eager = forward()
    assert torch.isfinite(fused).all() and _bits(fused).equal(_bits(eager))
