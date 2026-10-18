"""AdaLN modulation helpers and their exact fused MPS tails."""

from functools import cache

import torch
from torch import nn

from iris3b.nn.norms import RMSNorm


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def norm_modulate(norm: nn.Module, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``modulate(norm(x), shift, scale)``, fused on MPS without autograd."""
    if _fusable(norm, x, shift, scale):
        name = "norm_modulate_float" if x.dtype == torch.float32 else "norm_modulate_bfloat"
        return _fused(name, norm, x, _rows(scale, x), _rows(shift, x))
    return modulate(norm(x), shift, scale)


def gated_norm_residual(
    residual: torch.Tensor, gate: torch.Tensor, norm: nn.Module, branch: torch.Tensor
) -> torch.Tensor:
    """``residual + gate * norm(branch)``, fused on MPS without autograd."""
    if (
        residual.dtype == torch.float32
        and branch.dtype == torch.bfloat16
        and residual.shape == branch.shape
        and _fusable(norm, residual, gate)
        and _fusable(norm, branch, gate)
    ):
        return _fused("gated_norm_residual", norm, branch, _rows(gate, branch), residual)
    return residual + gate * norm(branch)


def modulated_residual(
    residual: torch.Tensor, branch: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``residual + modulate(branch, shift, scale)``, fused for BF16 inference."""
    if (
        _elementwise(residual, branch)
        and shift.stride() == scale.stride()
        and _row_views(residual, shift, scale)
    ):
        width4, stride4 = residual.shape[-1] // 4, scale.stride(-2) // 4
        return _elementwise_out("modulated_residual", residual, branch, scale, shift, width4, stride4)
    return residual + modulate(branch, shift, scale)


def norm_operand(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``norm(x)`` as a BF16-autocast Linear operand."""
    if (
        isinstance(norm, RMSNorm)
        and torch.is_autocast_enabled("mps")
        and torch.get_autocast_dtype("mps") == torch.bfloat16
        and _elementwise(x)
        and norm.weight.dtype == torch.float32
        and norm.weight.shape == (x.shape[-1],)
        and _aligned(norm.weight)
    ):
        return _elementwise_out("norm_operand", x, _inv_rms(norm, x), norm.weight, x.shape[-1] // 4)
    return norm(x)


# eager rounding points; PyTorch keeps the reduction
_TAILS = r"""
#include <metal_stdlib>
using namespace metal;

template <typename T>
inline float4 normed(device const vec<T, 4>* x, device const float* r, device const float4* w, uint i, uint2 p) {
  return w[p.x] * float4(vec<T, 4>(float4(x[i]) * r[p.y]));
}

// mod_a: scale, mod_b: shift
template <typename T>
kernel void norm_modulate(
    device const vec<T, 4>* x [[buffer(0)]], device const float* r [[buffer(1)]],
    device const float4* w [[buffer(2)]], device const float* mod_a [[buffer(3)]],
    device const float* mod_b [[buffer(4)]], device float4* out [[buffer(5)]],
    constant uint& width4 [[buffer(6)]], constant uint& rows_per_batch [[buffer(7)]],
    constant uint& mod_stride [[buffer(8)]], uint2 p [[thread_position_in_grid]]) {
  const uint i = p.y * width4 + p.x, m = p.y / rows_per_batch * mod_stride / 4 + p.x;
  const float4 one_plus = 1.0f + reinterpret_cast<device const float4*>(mod_a)[m];
  const float4 scaled = normed(x, r, w, i, p) * one_plus;
  out[i] = scaled + reinterpret_cast<device const float4*>(mod_b)[m];
}

// mod_a: gate, mod_b: the FP32 residual
kernel void gated_norm_residual(
    device const bfloat4* x [[buffer(0)]], device const float* r [[buffer(1)]],
    device const float4* w [[buffer(2)]], device const float* mod_a [[buffer(3)]],
    device const float* mod_b [[buffer(4)]], device float4* out [[buffer(5)]],
    constant uint& width4 [[buffer(6)]], constant uint& rows_per_batch [[buffer(7)]],
    constant uint& mod_stride [[buffer(8)]], uint2 p [[thread_position_in_grid]]) {
  const uint i = p.y * width4 + p.x, m = p.y / rows_per_batch * mod_stride / 4 + p.x;
  const float4 gated = reinterpret_cast<device const float4*>(mod_a)[m] * normed(x, r, w, i, p);
  out[i] = reinterpret_cast<device const float4*>(mod_b)[i] + gated;
}

// BF16 residual + modulate(branch, shift, scale)
kernel void modulated_residual(
    device bfloat4* out [[buffer(0)]], device const bfloat4* x [[buffer(1)]],
    device const bfloat4* branch [[buffer(2)]], device const bfloat* scale [[buffer(3)]],
    device const bfloat* shift [[buffer(4)]], constant uint& width4 [[buffer(5)]],
    constant uint& mod_stride4 [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  const uint row = i / width4, m = row * mod_stride4 + (i - row * width4);
  const float4 one_plus = float4(bfloat4(1.0f + float4(reinterpret_cast<device const bfloat4*>(scale)[m])));
  const float4 scaled = float4(bfloat4(float4(branch[i]) * one_plus));
  const float4 shifted = float4(bfloat4(scaled + float4(reinterpret_cast<device const bfloat4*>(shift)[m])));
  out[i] = bfloat4(float4(x[i]) + shifted);
}

// autocast's BF16 cast of RMSNorm(x)
kernel void norm_operand(
    device bfloat4* out [[buffer(0)]], device const bfloat4* x [[buffer(1)]],
    device const float* r [[buffer(2)]], device const float4* w [[buffer(3)]],
    constant uint& width4 [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  const uint row = i / width4;
  out[i] = bfloat4(normed(x, r, w, i, uint2(i - row * width4, row)));
}

template [[host_name("norm_modulate_float")]] kernel void norm_modulate<float>(
    device const float4*, device const float*, device const float4*, device const float*,
    device const float*, device float4*, constant uint&, constant uint&, constant uint&, uint2);
template [[host_name("norm_modulate_bfloat")]] kernel void norm_modulate<bfloat>(
    device const bfloat4*, device const float*, device const float4*, device const float*,
    device const float*, device float4*, constant uint&, constant uint&, constant uint&, uint2);
"""


@cache
def _library():
    return torch.mps.compile_shader(_TAILS)


def _rows(mod: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return mod.reshape(x.shape[0], x.shape[-1])


def _aligned(t: torch.Tensor) -> bool:
    """Rows the kernels can read as aligned four-element vectors."""
    return t.stride(-1) == 1 and t.storage_offset() % 4 == 0 and (t.dim() < 2 or t.stride(-2) % 4 == 0)


def _fusable(norm: nn.Module, x: torch.Tensor, *mods: torch.Tensor) -> bool:
    if torch.is_grad_enabled() or x.device.type != "mps" or not isinstance(norm, RMSNorm):
        return False
    batch, width = x.shape[0], x.shape[-1]
    return (
        x.dim() == 3
        and x.dtype in (torch.float32, torch.bfloat16)
        and x.is_contiguous()
        and _aligned(x)
        and width % 4 == 0
        and x.numel() < 2**31
        and norm.weight.dtype == torch.float32
        and norm.weight.shape == (width,)
        and _aligned(norm.weight)
        and all(
            m.dtype == torch.float32 and m.shape == (batch, 1, width) and _aligned(_rows(m, x)) for m in mods
        )
        and (batch == 1 or len({_rows(m, x).stride(0) for m in mods}) == 1)
    )


def _elementwise(x: torch.Tensor, *others: torch.Tensor) -> bool:
    """Contiguous BF16 MPS inference tensors shaped like ``x``."""
    return (
        not torch.is_grad_enabled()
        and x.device.type == "mps"
        and x.dtype == torch.bfloat16
        and x.shape[-1] % 4 == 0
        and 0 < x.numel() < 2**31
        and all(
            t.dtype == x.dtype and t.shape == x.shape and t.is_contiguous() and _aligned(t)
            for t in (x, *others)
        )
    )


def _row_views(x: torch.Tensor, *mods: torch.Tensor) -> bool:
    """Rows of 3-D ``x`` at one aligned stride."""
    return x.dim() == 3 and all(
        m.dtype == x.dtype
        and m.shape == x.shape
        and _aligned(m)
        and m.stride(0) == x.shape[1] * m.stride(1)
        and x.shape[0] * m.stride(0) < 2**31
        for m in mods
    )


def _elementwise_out(name: str, x: torch.Tensor, *args) -> torch.Tensor:
    out = torch.empty_like(x)
    n4 = x.numel() // 4
    getattr(_library(), name)(out, x, *args, threads=[n4], group_size=[min(256, n4)])
    return out


def _inv_rms(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
    x32 = x.float()  # as RMSNorm.forward computes it
    return torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + norm.eps)


def _fused(
    name: str, norm: RMSNorm, x: torch.Tensor, mod_a: torch.Tensor, mod_b: torch.Tensor
) -> torch.Tensor:
    batch, tokens, width = x.shape
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    rows = batch * tokens
    getattr(_library(), name)(
        x, _inv_rms(norm, x), norm.weight, mod_a, mod_b, out, width // 4, tokens,
        mod_a.stride(0) if batch > 1 else 0,
        threads=[width // 4, rows], group_size=[min(64, width // 4), min(4, rows)],
    )  # fmt: skip
    return out


class SharedCoreModulation(nn.Module):
    """One shared modulation core plus a per-block low-rank residual.

    ``m_i(c) = m_shared(c) + U_i V_i c`` with ``V_i: D -> r`` and
    ``U_i: r -> 6D``, both bias-free. The core is one ``Linear(D, 6D)`` per
    stream, owned by the model and shared by every patch block; it is
    deliberately NOT registered as a submodule here, so it appears exactly once
    in ``parameters()`` and once in ``state_dict()`` instead of once per block.

    Drop-in for the ``nn.Linear(dim, 6 * dim)`` it replaces: identical
    ``[..., 6*D]`` output layout (shift/scale/gate for attention, then for the
    MLP).

    Zero-init (``model.adaln_zero_init``): the core and ``adaln_up`` (U) are
    zeroed by the model's leaf-name scan, which leaves the block exactly as
    identity-at-init as the per-block projection. ``down`` (V) deliberately
    keeps its default init -- with both factors zero the residual's gradient is
    identically zero and the branch could never train.
    """

    def __init__(self, dim: int, rank: int, core: nn.Linear):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"model.modulation_rank must be positive, got {rank}")
        self.down = nn.Linear(dim, rank, bias=False)
        self.adaln_up = nn.Linear(rank, 6 * dim, bias=False)
        self._core = (core,)  # tuple hides the shared core from nn.Module registration

    @property
    def core(self) -> nn.Linear:
        return self._core[0]

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.core(cond) + self.adaln_up(self.down(cond))


class SharedCoreBias(nn.Module):
    """One shared modulation core plus a per-block learned bias (adaLN-single).

    ``m_i(c) = m_shared(c) + b_i``. The block-specific term is unconditional, so
    every block reads the same timestep response and only shifts it by a learned
    constant, the most aggressive sharing that still lets blocks differ at all.

    Same contract as :class:`SharedCoreModulation`: the core is owned by the
    model and hidden from registration here, and the ``[..., 6*D]`` output layout
    is unchanged.

    The bias starts at zero, which makes the block identity-at-init whenever the
    core is (``model.adaln_zero_init``) without needing the zero-init scan to
    know about it. Unlike the low-rank residual there is no dead-branch hazard:
    a bias always receives gradient regardless of the core's value.
    """

    def __init__(self, dim: int, core: nn.Linear):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(6 * dim))
        self._core = (core,)  # tuple hides the shared core from nn.Module registration

    @property
    def core(self) -> nn.Linear:
        return self._core[0]

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.core(cond) + self.bias


class ModulationBuilder:
    """Builds a block's per-stream adaLN projection for the configured mode.

    ``per_block`` returns a plain ``nn.Linear(dim, 6 * dim, bias=True)`` per
    block. ``shared_lowrank`` returns a
    :class:`SharedCoreModulation` bound to a per-stream core created on first
    request inside ``cores``, a ``ModuleDict`` owned by the model.
    ``shared_bias`` returns a :class:`SharedCoreBias` bound to that same core.

    Streams are named by the caller: the dual-stream block asks for ``img`` and
    ``txt``, the single-stream block for one ``shared`` projection.
    ``stream_aliases`` redirects one stream name onto another's core, which is
    how a hybrid trunk keeps its single-stream half on the dual half's ``img``
    core instead of allocating a third one.
    """

    MODES = ("per_block", "shared_lowrank", "shared_bias")
    SHARED = ("shared_lowrank", "shared_bias")

    def __init__(
        self,
        dim: int,
        mode: str = "per_block",
        rank: int = 64,
        cores: nn.ModuleDict | None = None,
        stream_aliases: dict[str, str] | None = None,
    ):
        if mode not in self.MODES:
            raise ValueError(f"model.modulation must be one of {self.MODES}, got {mode!r}")
        if mode in self.SHARED and cores is None:
            raise ValueError(f"{mode} modulation needs a core container owned by the model")
        self.dim = dim
        self.mode = mode
        self.rank = rank
        self.cores = cores
        self.stream_aliases = stream_aliases or {}

    def __call__(self, stream: str) -> nn.Module:
        if self.mode == "per_block":
            return nn.Linear(self.dim, 6 * self.dim, bias=True)
        key = f"adaln_{self.stream_aliases.get(stream, stream)}"
        if key not in self.cores:
            self.cores[key] = nn.Linear(self.dim, 6 * self.dim, bias=True)
        core = self.cores[key]
        if self.mode == "shared_bias":
            return SharedCoreBias(self.dim, core)
        return SharedCoreModulation(self.dim, self.rank, core)
