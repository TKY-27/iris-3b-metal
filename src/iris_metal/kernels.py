"""Fused Metal kernel for the attention input path, with a PyTorch reference.

``qk_norm_rope`` replaces the upstream sequence ``RMSNorm -> apply_rope ->
transpose`` for one query or key projection. The row is normalized, scaled by
the FP32 gain and rotated entirely in FP32, then rounded once on store. The
store dtype may differ from the input dtype: queries and keys are bounded by
the RMSNorm, so FP16 holds them with 8x the precision of BF16.
"""

from functools import cache

import torch

HEAD_DIM = 128

_SOURCE = r"""
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
INSTANTIATE(half, half);
INSTANTIATE(float, half);
"""

_TYPE_NAMES = {torch.float32: "float", torch.bfloat16: "bfloat", torch.float16: "half"}


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
    if col_offset + heads * HEAD_DIM > width or not x.is_contiguous() or not out.is_contiguous():
        raise ValueError("qk_norm_rope needs a contiguous [B, N, W] input holding the heads")
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
