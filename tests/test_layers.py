import math
from contextlib import contextmanager

import pytest
import torch

import iris3b.nn.attention as attention_module
from iris3b.nn import RMSNorm, SwiGLU, TimestepEmbedder, rope_1d, rope_2d
from iris3b.nn.rope import apply_rope


def test_rmsnorm_fp32_math():
    norm = RMSNorm(8).to(torch.bfloat16)
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16) * 100
    out = norm(x)
    assert out.dtype == torch.bfloat16
    expected = x.float() / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(out.float(), expected.to(torch.bfloat16).float(), rtol=0, atol=0)
    # fp32 gain promotes the result to fp32 (autocast training path)
    assert RMSNorm(8)(x).dtype == torch.float32


def test_swiglu_two_thirds_width():
    ff = SwiGLU(1536, 4.0)
    assert ff.w1.out_features == 4096  # int(2 * int(1536 * 4) / 3)
    assert ff.w1.bias is None and ff.w2.bias is None and ff.w3.bias is None


def test_timestep_embedding_layout():
    emb = TimestepEmbedder(64, freq_dim=8, max_period=10.0)
    t = torch.tensor([3.0])
    feats = emb.timestep_embedding(t)
    freqs = torch.exp(-math.log(10.0) * torch.arange(4).float() / 4)
    torch.testing.assert_close(feats[0, :4], torch.cos(3.0 * freqs))
    torch.testing.assert_close(feats[0, 4:], torch.sin(3.0 * freqs))
    out = emb(torch.tensor([5.0, 7.0]))
    assert out.shape == (2, 1, 64)


def test_rope2d_resolution_independent_span():
    """Coordinate span is [0, scale] no matter the grid size."""
    a = rope_2d(64, 8, 8, scale=16.0)
    b = rope_2d(64, 32, 32, scale=16.0)
    assert a.shape == (64, 32) and b.shape == (32 * 32, 32)
    # last token of the grid sits at coordinate (16, 16) in both cases
    torch.testing.assert_close(a[-1], b[-1], rtol=1e-5, atol=1e-6)


def test_rope2d_xy_interleave():
    cis = rope_2d(8, 2, 3, theta=10000.0, scale=16.0)  # 2 freqs -> 4 complex per token
    # token (row 0, col j): x angle varies, y angle = 0 -> y slots are 1+0j
    torch.testing.assert_close(cis[1, 1], torch.complex(torch.tensor(1.0), torch.tensor(0.0)))
    assert cis[1, 0].angle().abs() > 0  # x slot rotated


def test_rope2d_default_matches_explicit_construction():
    """Default args reproduce the 2D construction bit for bit."""
    head_dim, h, w, theta, scale = 64, 4, 6, 10000.0, 16.0
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 4)[: head_dim // 4].float() / head_dim)
    x_ang = torch.outer(torch.linspace(0, scale, w), freqs)
    y_ang = torch.outer(torch.linspace(0, scale, h), freqs)
    x_cis = torch.polar(torch.ones_like(x_ang), x_ang)[None, :, :].expand(h, w, -1)
    y_cis = torch.polar(torch.ones_like(y_ang), y_ang)[:, None, :].expand(h, w, -1)
    ref = torch.stack([x_cis, y_cis], dim=-1).reshape(h * w, head_dim // 2)
    assert torch.equal(rope_2d(head_dim, h, w, theta=theta, scale=scale), ref)


def test_rope2d_frame_axis_reserves_the_inert_tail():
    """Frame slots take over pairs that carry no phase, leaving x/y untouched."""
    base = rope_2d(64, 4, 6)
    tagged = rope_2d(64, 4, 6, frame_pairs=2)
    assert tagged.shape == base.shape == (24, 32)
    torch.testing.assert_close(tagged[:, :28], base[:, :28], rtol=0, atol=0)
    freqs = 1.0 / 10000.0 ** (torch.arange(0, 64, 4).float() / 64)
    assert (freqs[-2:] * 16.0).max() < 1e-2  # the donated pairs were inert
    torch.testing.assert_close(tagged[:, 28:], torch.ones(24, 4, dtype=torch.complex64), rtol=0, atol=0)


def test_rope2d_frame_tag_is_live_at_index_one():
    tagged = rope_2d(64, 4, 6, frame_pairs=2, frame_theta=10.0, frame_index=1)
    assert tagged[:, 28:].angle().abs().min() > 0.1


def test_constant_frame_index_is_invisible_to_attention():
    """Pretrain-at-zero / finetune-later: a global frame offset changes no logit."""
    q, k = torch.randn(1, 24, 2, 64), torch.randn(1, 24, 2, 64)
    logits = []
    for frame_index in (0, 7):
        qr, kr = apply_rope(q, k, rope_2d(64, 4, 6, frame_pairs=2, frame_index=frame_index))
        logits.append(torch.einsum("bnhd,bmhd->bhnm", qr, kr))
    torch.testing.assert_close(logits[0], logits[1], rtol=1e-5, atol=1e-4)


def test_rope2d_isotropic_aspect_preserves_ratio():
    h, w, scale, j = 4, 8, 16.0, 3  # freq 3 keeps every angle below pi (no wrap)
    freq = (1.0 / 10000.0 ** (torch.arange(0, 64, 4).float() / 64))[j]
    step = scale / (w - 1)
    iso = rope_2d(64, h, w, scale=scale, aspect="isotropic")
    torch.testing.assert_close(iso[1, 2 * j].angle(), step * freq)
    torch.testing.assert_close(iso[w, 2 * j + 1].angle(), step * freq)
    ratio = iso[-1, 2 * j].angle() / iso[-1, 2 * j + 1].angle()
    torch.testing.assert_close(ratio, torch.tensor((w - 1) / (h - 1)), rtol=1e-5, atol=1e-5)
    sq = rope_2d(64, h, w, scale=scale)  # square mode erases the ratio
    torch.testing.assert_close(sq[-1, 2 * j].angle(), sq[-1, 2 * j + 1].angle())


def test_apply_rope_preserves_norm():
    q = torch.randn(2, 5, 4, 64)
    k = torch.randn(2, 5, 4, 64)
    cis = rope_1d(64, 5)
    q2, k2 = apply_rope(q, k, cis)
    torch.testing.assert_close(q2.norm(dim=-1), q.norm(dim=-1), rtol=1e-4, atol=1e-4)
    assert q2.dtype == q.dtype


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("torch_flash", attention_module.SDPBackend.FLASH_ATTENTION),
        ("torch_cudnn", attention_module.SDPBackend.CUDNN_ATTENTION),
    ],
)
def test_torch_attention_backend_is_forced(monkeypatch, backend, expected):
    selected = []
    gqa_flags = []

    @contextmanager
    def fake_kernel(value):
        selected.append(value)
        yield

    def fake_sdpa(q, k, v, attn_mask=None, enable_gqa=False):
        gqa_flags.append(enable_gqa)
        return q + k + v

    monkeypatch.setattr(attention_module, "sdpa_kernel", fake_kernel)
    monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", fake_sdpa)
    q = torch.randn(2, 4, 7, 16)
    out = attention_module.scaled_dot_product(q, q, q, backend=backend, enable_gqa=True)
    assert selected == [expected]
    assert gqa_flags == [True]
    torch.testing.assert_close(out, q * 3)


@pytest.mark.parametrize("backend", ["fa3", "fa4"])
def test_external_attention_backend_layout(monkeypatch, backend):
    qkv = torch.randn(2, 7, 3, 4, 16, requires_grad=True)
    q, k, v = (tensor.transpose(1, 2) for tensor in qkv.unbind(2))
    received = []

    def fake_flash(q_ext, k_ext, v_ext, **kwargs):
        assert kwargs == {}
        received.extend((q_ext, k_ext, v_ext))
        result = q_ext + 2 * k_ext + 3 * v_ext
        return (result, torch.empty(0)) if backend == "fa4" else result

    monkeypatch.setattr(attention_module, "_validate_external_attention", lambda *args: None)
    monkeypatch.setattr(attention_module, "_external_attention_func", lambda name: fake_flash)
    out = attention_module.scaled_dot_product(q, k, v, backend=backend)

    assert all(tensor.shape == (2, 7, 4, 16) for tensor in received)
    assert all(tensor.stride(-1) == 1 for tensor in received)
    assert received[0].data_ptr() == q.data_ptr()
    torch.testing.assert_close(out, q + 2 * k + 3 * v)
    out.sum().backward()
    expected_grad = torch.ones_like(qkv)
    expected_grad[:, :, 1].mul_(2)
    expected_grad[:, :, 2].mul_(3)
    torch.testing.assert_close(qkv.grad, expected_grad)


def test_external_attention_backend_is_strict():
    q = torch.randn(2, 4, 7, 16, dtype=torch.bfloat16)
    mask = torch.ones(2, 1, 7, 7, dtype=torch.bool)
    with pytest.raises(ValueError, match="does not support arbitrary attention masks"):
        attention_module.scaled_dot_product(q, q, q, backend="fa4", attn_mask=mask)
    with pytest.raises(ValueError, match="requires matching float16 or bfloat16"):
        attention_module.scaled_dot_product(q.float(), q.float(), q.float(), backend="fa3")
    with pytest.raises(ValueError, match="requires Q/K/V on one CUDA device"):
        attention_module.scaled_dot_product(q, q, q, backend="fa3")
    kv = q[:, :2]
    with pytest.raises(ValueError, match="unequal Q/KV heads require enable_gqa=True"):
        attention_module.scaled_dot_product(q, kv, kv, backend="fa3")
    with pytest.raises(ValueError, match="Q heads to be divisible"):
        attention_module.scaled_dot_product(q, q[:, :3], q[:, :3], backend="fa3", enable_gqa=True)
    with pytest.raises(ValueError, match="K and V must have the same number of heads"):
        attention_module.scaled_dot_product(q, kv, q[:, :1], backend="fa3", enable_gqa=True)


def test_external_attention_missing_install_has_actionable_error(monkeypatch):
    attention_module._external_attention_func.cache_clear()

    def missing_module(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(attention_module, "import_module", missing_module)
    with pytest.raises(
        RuntimeError, match=r"attention backend 'fa4' is unavailable: cannot import flash_attn\.cute"
    ):
        attention_module._external_attention_func("fa4")
    attention_module._external_attention_func.cache_clear()


def test_unknown_attention_backend_is_rejected():
    q = torch.randn(2, 4, 7, 16)
    with pytest.raises(ValueError, match="unknown attention backend 'typo'"):
        attention_module.scaled_dot_product(q, q, q, backend="typo")
