"""Fused Metal kernels for the transformer trunk, with PyTorch references.

- ``qk_norm_rope``: per-head RMSNorm + RoPE + ``[B, N, H, D] -> [B, H, N, D]``
  for one query or key projection.
- ``rms_norm_modulate``: RMSNorm + adaLN modulation + downcast, the input of
  every attention and MLP branch.
- ``gated_residual_rms_norm_``: sandwich post-norm + adaLN gate + residual add,
  in place on the FP32 residual stream.
- ``pixel_mlp_``: the per-pixel half of a pixel-transformer block (both
  modulated residual adds, RMSNorm and the 16 -> 64 -> 16 GELU MLP) plus the
  RMSNorm that feeds the next block, one thread per pixel.

Rounding is deliberately coarser-grained than the upstream autocast path,
which rounds after nearly every elementwise op: norms, gains, modulation and
rotation are computed in FP32 and rounded once on store. That is closer to an
FP32 forward, not bit-identical to the reference. Queries and keys are bounded
by their RMSNorm, so they may be stored in FP16, which holds them with 8x the
precision of BF16.

The ``float*`` kernel arguments (norm gains, rotary tables, modulation) must
be FP32; the wrappers check this, since a BF16 gain would be misread silently.
"""

from functools import cache

import torch

HEAD_DIM = 128

_SOURCE = r"""
#include <c10/metal/special_math.h>
#include <metal_stdlib>
using namespace metal;

// One SIMD group per (batch, token, head) row of 128 values; lane i owns the
// rotary pairs i and i + 32, i.e. elements [2i, 2i+1] and [64+2i, 64+2i+1].
template <typename T, typename U>
kernel void qk_norm_rope(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* cos_table [[buffer(3)]],
    device const float* sin_table [[buffer(4)]],
    constant long& n_src [[buffer(5)]],
    constant long& n_dst [[buffer(6)]],
    constant long& heads [[buffer(7)]],
    constant long& dst_offset [[buffer(8)]],
    constant long& row_stride [[buffer(9)]],
    constant long& col_offset [[buffer(10)]],
    constant float& eps [[buffer(11)]],
    uint3 pos [[thread_position_in_grid]]) {
  const uint lane = pos.x;
  const long row = pos.y;
  if (row >= n_src * heads) {
    return;
  }
  const long token = row / heads;
  const long head = row % heads;
  const long batch = pos.z;

  device const T* x = src + (batch * n_src + token) * row_stride + col_offset + head * 128;
  const uint e0 = 2 * lane;
  const uint e1 = 64 + 2 * lane;
  float a0 = float(x[e0]), a1 = float(x[e0 + 1]);
  float b0 = float(x[e1]), b1 = float(x[e1 + 1]);

  const float ss = simd_sum(a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1);
  const float r = rsqrt(ss / 128.0f + eps);
  a0 *= r * weight[e0];
  a1 *= r * weight[e0 + 1];
  b0 *= r * weight[e1];
  b1 *= r * weight[e1 + 1];

  const long out_token = dst_offset + token;
  const long t = out_token * 64;
  const float ca = cos_table[t + lane], sa = sin_table[t + lane];
  const float cb = cos_table[t + 32 + lane], sb = sin_table[t + 32 + lane];

  device U* y = dst + ((batch * heads + head) * n_dst + out_token) * 128;
  y[e0] = U(a0 * ca - a1 * sa);
  y[e0 + 1] = U(a0 * sa + a1 * ca);
  y[e1] = U(b0 * cb - b1 * sb);
  y[e1 + 1] = U(b0 * sb + b1 * cb);
}

#define INSTANTIATE(T, U)                                                           \
  template [[host_name("qk_norm_rope_" #T "_" #U)]] kernel void qk_norm_rope<T, U>( \
      device const T*, device U*, device const float*, device const float*,         \
      device const float*, constant long&, constant long&, constant long&,          \
      constant long&, constant long&, constant long&, constant float&, uint3);

INSTANTIATE(float, float);
INSTANTIATE(bfloat, bfloat);
INSTANTIATE(bfloat, half);
INSTANTIATE(bfloat, float);

// One SIMD group per token row; ``width`` is a multiple of 128 and lane i
// owns the 4-vectors i, i + 32, ... so loads coalesce across the group.
template <typename T>
inline float inv_rms(device const T* x, uint lane, long width, float eps) {
  device const vec<T, 4>* x4 = reinterpret_cast<device const vec<T, 4>*>(x);
  float ss = 0.0f;
  for (long j = lane; j < width / 4; j += 32) {
    const float4 v = float4(x4[j]);
    ss += dot(v, v);
  }
  return rsqrt(simd_sum(ss) / float(width) + eps);
}

template <typename T, typename U>
kernel void rms_norm_modulate(
    device const T* x [[buffer(0)]],
    device U* out [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* shift [[buffer(3)]],
    device const float* scale [[buffer(4)]],
    constant long& rows [[buffer(5)]],
    constant long& rows_per_batch [[buffer(6)]],
    constant long& width [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint2 pos [[thread_position_in_grid]]) {
  const uint lane = pos.x;
  const long row = pos.y;
  if (row >= rows) {
    return;
  }
  device const T* xr = x + row * width;
  const float r = inv_rms(xr, lane, width, eps);
  const long batch = row / rows_per_batch;
  device const vec<T, 4>* x4 = reinterpret_cast<device const vec<T, 4>*>(xr);
  device vec<U, 4>* o4 = reinterpret_cast<device vec<U, 4>*>(out + row * width);
  device const float4* w4 = reinterpret_cast<device const float4*>(weight);
  device const float4* sh4 = reinterpret_cast<device const float4*>(shift + batch * width);
  device const float4* sc4 = reinterpret_cast<device const float4*>(scale + batch * width);
  for (long j = lane; j < width / 4; j += 32) {
    const float4 h = float4(x4[j]) * r * w4[j];
    o4[j] = vec<U, 4>(h * (1.0f + sc4[j]) + sh4[j]);
  }
}

template <typename T>
kernel void gated_residual_rms_norm(
    device float* residual [[buffer(0)]],
    device const T* branch [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    constant long& rows [[buffer(4)]],
    constant long& rows_per_batch [[buffer(5)]],
    constant long& width [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint2 pos [[thread_position_in_grid]]) {
  const uint lane = pos.x;
  const long row = pos.y;
  if (row >= rows) {
    return;
  }
  device const T* br = branch + row * width;
  const float r = inv_rms(br, lane, width, eps);
  const long batch = row / rows_per_batch;
  device const vec<T, 4>* b4 = reinterpret_cast<device const vec<T, 4>*>(br);
  device float4* x4 = reinterpret_cast<device float4*>(residual + row * width);
  device const float4* w4 = reinterpret_cast<device const float4*>(weight);
  device const float4* g4 = reinterpret_cast<device const float4*>(gate + batch * width);
  for (long j = lane; j < width / 4; j += 32) {
    x4[j] += g4[j] * (float4(b4[j]) * r * w4[j]);
  }
}

#define INSTANTIATE_ROW(T, U)                                                                  \
  template [[host_name("rms_norm_modulate_" #T "_" #U)]] kernel void rms_norm_modulate<T, U>(   \
      device const T*, device U*, device const float*, device const float*,                    \
      device const float*, constant long&, constant long&, constant long&, constant float&,    \
      uint2);

INSTANTIATE_ROW(float, bfloat);
INSTANTIATE_ROW(float, float);

#define INSTANTIATE_RESIDUAL(T)                                                                 \
  template [[host_name("gated_residual_rms_norm_" #T)]] kernel void gated_residual_rms_norm<T>( \
      device float*, device const T*, device const float*, device const float*,                 \
      constant long&, constant long&, constant long&, constant float&, uint2);

INSTANTIATE_RESIDUAL(bfloat);
INSTANTIATE_RESIDUAL(float);

// Pixel stage, one thread per pixel with its 16 channels in registers. Like
// the upstream autocast path, the stream and the MLP's matmul inputs and
// outputs are stored in BF16; modulation and norms run in FP32 and round once.
constant constexpr int PIX = 16;
constant constexpr int PIX_HIDDEN = 64;

inline float round_bf16(float x) {
  return float(bfloat(x));
}

inline void store_normed(device bfloat* out, thread const float* x, device const float* weight, float eps) {
  float ss = 0.0f;
  for (int c = 0; c < PIX; ++c) {
    ss += x[c] * x[c];
  }
  const float r = rsqrt(ss / PIX + eps);
  for (int c = 0; c < PIX; ++c) {
    out[c] = bfloat(x[c] * r * weight[c]);
  }
}

kernel void pixel_rms_norm(
    device const bfloat* pixels [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    constant long& n_pixels [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= n_pixels) {
    return;
  }
  float x[PIX];
  for (int c = 0; c < PIX; ++c) {
    x[c] = float(pixels[idx * PIX + c]);
  }
  store_normed(out + idx * PIX, x, weight, eps);
}

// mods per pixel: scale1, shift1, scale2, shift2 (16 each), the "post" PiT layout.
kernel void pixel_mlp(
    device bfloat* pixels [[buffer(0)]],
    device const bfloat* branch [[buffer(1)]],
    device const bfloat* mods [[buffer(2)]],
    device const float* norm_weight [[buffer(3)]],
    device const bfloat* fc1_weight [[buffer(4)]],
    device const bfloat* fc1_bias [[buffer(5)]],
    device const bfloat* fc2_weight [[buffer(6)]],
    device const bfloat* fc2_bias [[buffer(7)]],
    device const float* next_norm_weight [[buffer(8)]],
    device bfloat* next_input [[buffer(9)]],
    constant long& n_pixels [[buffer(10)]],
    constant float& eps [[buffer(11)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= n_pixels) {
    return;
  }
  device const bfloat* mod = mods + idx * 4 * PIX;
  float x[PIX];
  for (int c = 0; c < PIX; ++c) {
    const float attn = float(branch[idx * PIX + c]);
    const float modulated = round_bf16(attn * (1.0f + float(mod[c])) + float(mod[PIX + c]));
    x[c] = round_bf16(float(pixels[idx * PIX + c]) + modulated);
  }

  float ss = 0.0f;
  for (int c = 0; c < PIX; ++c) {
    ss += x[c] * x[c];
  }
  const float r = rsqrt(ss / PIX + eps);
  float h[PIX];
  for (int c = 0; c < PIX; ++c) {
    h[c] = round_bf16(x[c] * r * norm_weight[c]);
  }

  float out[PIX];
  for (int c = 0; c < PIX; ++c) {
    out[c] = float(fc2_bias[c]);
  }
  for (int j = 0; j < PIX_HIDDEN; ++j) {
    float a = float(fc1_bias[j]);
    for (int c = 0; c < PIX; ++c) {
      a += h[c] * float(fc1_weight[j * PIX + c]);
    }
    a = round_bf16(a);
    const float g = round_bf16(0.5f * a * (1.0f + c10::metal::erf(a * M_SQRT1_2_F)));
    for (int c = 0; c < PIX; ++c) {
      out[c] += g * float(fc2_weight[c * PIX_HIDDEN + j]);
    }
  }

  for (int c = 0; c < PIX; ++c) {
    const float mlp = round_bf16(out[c]);
    const float modulated = round_bf16(mlp * (1.0f + float(mod[2 * PIX + c])) + float(mod[3 * PIX + c]));
    x[c] = round_bf16(x[c] + modulated);
    pixels[idx * PIX + c] = bfloat(x[c]);
  }
  store_normed(next_input + idx * PIX, x, next_norm_weight, eps);
}
"""

_TYPE_NAMES = {torch.float32: "float", torch.bfloat16: "bfloat", torch.float16: "half"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_fp32(**tensors: torch.Tensor) -> None:
    for name, tensor in tensors.items():
        _require(tensor.dtype == torch.float32 and tensor.is_contiguous(), f"{name} must be contiguous FP32")


@cache
def _library():
    return torch.mps.compile_shader(_SOURCE)


def qk_norm_rope_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``[B, N, H*128]`` projection -> ``[B, H, N, 128]`` normalized, rotated heads."""
    batch, tokens, width = x.shape
    h = x.reshape(batch, tokens, width // HEAD_DIM, HEAD_DIM).float()
    normed = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    pairs = normed.reshape(*normed.shape[:-1], HEAD_DIM // 2, 2)
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    rotated = torch.stack(
        (pairs[..., 0] * c - pairs[..., 1] * s, pairs[..., 0] * s + pairs[..., 1] * c), dim=-1
    )
    return rotated.flatten(-2).to(out_dtype or x.dtype).transpose(1, 2).contiguous()


def qk_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    heads: int,
    col_offset: int = 0,
    out: torch.Tensor | None = None,
    offset: int = 0,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fused per-head RMSNorm + RoPE + ``[B, N, H, D] -> [B, H, N, D]`` layout.

    ``x``: contiguous ``[B, N, W]`` projection whose columns
    ``[col_offset, col_offset + heads*128)`` hold the heads, so a fused QKV
    output is read in place. ``cos``/``sin``: FP32 ``[N_out, 64]`` tables for
    the output token positions. When ``out`` (``[B, heads, N_out, 128]``) is
    given, rows are written from token ``offset`` on, so that text and image
    projections land in one attention input without a concatenation. The
    result has ``out_dtype`` (default: the input dtype).
    """
    batch, tokens, width = x.shape
    if out is None:
        out = x.new_empty(batch, heads, tokens, HEAD_DIM, dtype=out_dtype or x.dtype)
    if x.device.type != "mps":
        cols = x[..., col_offset : col_offset + heads * HEAD_DIM]
        rows = slice(offset, offset + tokens)
        out[:, :, rows] = qk_norm_rope_reference(cols, weight, cos[rows], sin[rows], eps, out.dtype)
        return out
    _require(
        col_offset + heads * HEAD_DIM <= width and x.is_contiguous(),
        "qk_norm_rope needs a contiguous [B, N, W] input holding the heads",
    )
    _require(
        out.shape[:2] == (batch, heads) and out.shape[3] == HEAD_DIM and out.is_contiguous(),
        f"qk_norm_rope output must be a contiguous [{batch}, {heads}, N, {HEAD_DIM}] tensor",
    )
    _require(
        cos.shape == sin.shape == (cos.shape[0], HEAD_DIM // 2)
        and offset + tokens <= min(cos.shape[0], out.shape[2]),
        "qk_norm_rope rotary tables or output do not cover the written rows",
    )
    _require_fp32(weight=weight, cos=cos, sin=sin)
    _require(weight.numel() == HEAD_DIM, f"qk_norm_rope gain must have {HEAD_DIM} entries")
    name = f"qk_norm_rope_{_TYPE_NAMES[x.dtype]}_{_TYPE_NAMES[out.dtype]}"
    getattr(_library(), name)(
        x,
        out,
        weight,
        cos,
        sin,
        tokens,
        out.shape[2],
        heads,
        offset,
        width,
        col_offset,
        float(eps),
        threads=[32, tokens * heads, batch],
        group_size=[32, 8, 1],
    )
    return out


def _row_launch(rows: int) -> dict:
    return {"threads": [32, rows, 1], "group_size": [32, 8, 1]}


def _per_batch(vector: torch.Tensor, batch: int, width: int) -> torch.Tensor:
    """``[B, 1, D]`` modulation slice -> contiguous FP32 ``[B, D]``."""
    return vector.reshape(batch, width).float().contiguous()


def rms_norm_modulate(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``(rms_norm(x) * weight * (1 + scale) + shift).to(out_dtype)`` for ``x``: ``[B, N, D]``
    and per-sample ``shift``/``scale``: ``[B, 1, D]``."""
    batch, tokens, width = x.shape
    if x.device.type != "mps":
        normed = torch.nn.functional.rms_norm(x.float(), (width,), weight, eps)
        return torch.addcmul(shift, normed, 1 + scale).to(out_dtype)
    _require(
        width % 128 == 0 and x.is_contiguous(), "rms_norm_modulate needs a contiguous input, width % 128 == 0"
    )
    _require_fp32(weight=weight)
    _require(weight.numel() == width, "rms_norm_modulate gain width mismatch")
    out = torch.empty(x.shape, device=x.device, dtype=out_dtype)
    name = f"rms_norm_modulate_{_TYPE_NAMES[x.dtype]}_{_TYPE_NAMES[out_dtype]}"
    getattr(_library(), name)(
        x,
        out,
        weight,
        _per_batch(shift, batch, width),
        _per_batch(scale, batch, width),
        batch * tokens,
        tokens,
        width,
        float(eps),
        **_row_launch(batch * tokens),
    )
    return out


def gated_residual_rms_norm_(
    residual: torch.Tensor, branch: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float
) -> torch.Tensor:
    """``residual += gate * rms_norm(branch) * weight`` in place; ``residual``: FP32 ``[B, N, D]``."""
    batch, tokens, width = residual.shape
    if residual.device.type != "mps":
        normed = torch.nn.functional.rms_norm(branch.float(), (width,), weight, eps)
        return residual.addcmul_(gate, normed)
    _require(width % 128 == 0 and branch.shape == residual.shape, "gated_residual_rms_norm_ shape mismatch")
    _require(branch.is_contiguous(), "gated_residual_rms_norm_ needs a contiguous branch")
    _require_fp32(residual=residual, weight=weight)
    _require(weight.numel() == width, "gated_residual_rms_norm_ gain width mismatch")
    getattr(_library(), f"gated_residual_rms_norm_{_TYPE_NAMES[branch.dtype]}")(
        residual,
        branch,
        weight,
        _per_batch(gate, batch, width),
        batch * tokens,
        tokens,
        width,
        float(eps),
        **_row_launch(batch * tokens),
    )
    return residual


PIXEL_WIDTH, PIXEL_HIDDEN = 16, 64


def pixel_rms_norm(pixels: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``rms_norm(pixels) * weight`` over the 16 channels of each pixel, in the pixel dtype."""
    if pixels.device.type != "mps" or pixels.dtype != torch.bfloat16:
        return torch.nn.functional.rms_norm(pixels.float(), pixels.shape[-1:], weight, eps).to(pixels.dtype)
    _require(
        pixels.shape[-1] == PIXEL_WIDTH and pixels.is_contiguous(), "pixel_rms_norm needs 16-wide pixels"
    )
    _require_fp32(weight=weight)
    out = torch.empty_like(pixels)
    n_pixels = pixels.numel() // PIXEL_WIDTH
    _library().pixel_rms_norm(pixels, out, weight, n_pixels, float(eps), threads=n_pixels, group_size=256)
    return out


def pixel_mlp_reference(
    pixels: torch.Tensor,
    branch: torch.Tensor,
    mods: torch.Tensor,
    norm_weight: torch.Tensor,
    fc1: torch.nn.Linear,
    fc2: torch.nn.Linear,
    next_norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """The upstream post-modulated PiT update after attention; values the autocast
    path materializes are rounded to the pixel dtype."""
    f = torch.nn.functional
    dtype = pixels.dtype
    scale1, shift1, scale2, shift2 = mods.float().chunk(4, dim=-1)
    modulated = (branch.float() * (1 + scale1) + shift1).to(dtype)
    x = (pixels.float() + modulated.float()).to(dtype)
    h = f.rms_norm(x.float(), x.shape[-1:], norm_weight, eps).to(dtype)
    a = f.linear(h.float(), fc1.weight.float(), fc1.bias.float()).to(dtype)
    g = f.gelu(a.float()).to(dtype)
    mlp = f.linear(g.float(), fc2.weight.float(), fc2.bias.float()).to(dtype)
    modulated = (mlp.float() * (1 + scale2) + shift2).to(dtype)
    pixels.copy_((x.float() + modulated.float()).to(dtype))
    return f.rms_norm(pixels.float(), pixels.shape[-1:], next_norm_weight, eps).to(dtype)


def pixel_mlp_(
    pixels: torch.Tensor,
    branch: torch.Tensor,
    mods: torch.Tensor,
    norm_weight: torch.Tensor,
    fc1: torch.nn.Linear,
    fc2: torch.nn.Linear,
    next_norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Finish a post-modulated PiT block in place; return the next block's normed input.

    ``pixels``/``branch``: BF16 ``[..., 16]`` (stream, attention branch);
    ``mods``: BF16 ``[..., 64]`` per pixel (scale1, shift1, scale2, shift2).
    """
    if pixels.device.type != "mps" or pixels.dtype != torch.bfloat16:
        return pixel_mlp_reference(pixels, branch, mods, norm_weight, fc1, fc2, next_norm_weight, eps)
    _require(
        pixels.shape[-1] == PIXEL_WIDTH
        and branch.shape == pixels.shape
        and mods.shape == (*pixels.shape[:-1], 4 * PIXEL_WIDTH)
        and fc1.weight.shape == (PIXEL_HIDDEN, PIXEL_WIDTH)
        and fc2.weight.shape == (PIXEL_WIDTH, PIXEL_HIDDEN),
        "pixel_mlp_ needs the released pixel-stage widths",
    )
    tensors = (pixels, branch, mods, fc1.weight, fc1.bias, fc2.weight, fc2.bias)
    _require(
        all(t.dtype == torch.bfloat16 and t.is_contiguous() for t in tensors),
        "pixel_mlp_ needs contiguous BF16 activations and MLP parameters",
    )
    _require_fp32(norm_weight=norm_weight, next_norm_weight=next_norm_weight)
    next_input = torch.empty_like(pixels)
    n_pixels = pixels.numel() // PIXEL_WIDTH
    _library().pixel_mlp(
        pixels,
        branch,
        mods,
        norm_weight,
        fc1.weight,
        fc1.bias,
        fc2.weight,
        fc2.bias,
        next_norm_weight,
        next_input,
        n_pixels,
        float(eps),
        threads=n_pixels,
        group_size=256,
    )
    return next_input
