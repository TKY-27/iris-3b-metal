"""Inference-only forward pass over the weights of an upstream ``IrisDiT``.

The upstream model is evaluated in its released configuration: FP32 master
weights under BF16 autocast. This module follows that recipe with BF16 linear
weights and biases (autocast rounds them the same way on every call) and FP32
norm gains, adaLN biases and residual streams. Elementwise work is fused and
rounded less often than under autocast, and attention inputs default to FP16
(see ``kernels``), so results are closer to an FP32 forward rather than
bit-identical to the reference. The computation is restructured for Apple GPUs:

- the text adapter depends only on the prompt and runs once per prompt, not
  once per sampling step;
- the shared adaLN cores run once per step instead of once per block;
- QK-RMSNorm, RoPE and the head transpose are one Metal kernel, and so are
  each branch's input norm + modulation and its post-norm + gated residual;
- the per-pixel half of every pixel block is one Metal kernel;
- grouped-query attention uses the native SDPA path, without expanding K/V;
- the single-stream tail keeps one ``[text | image]`` token tensor instead of
  re-concatenating it in every block, and the last block skips text rows that
  no output depends on.
"""

import torch
import torch.nn.functional as F
from torch import nn

from iris3b.models.blocks.mmdit import MMDiTBlock
from iris3b.models.blocks.pit import PiTBlock
from iris3b.models.blocks.single_stream import SingleStreamBlock
from iris3b.models.dit import IrisDiT
from iris3b.nn.norms import RMSNorm
from iris3b.nn.rope import rope_1d, rope_2d
from iris_metal.kernels import (
    HEAD_DIM,
    gated_residual_rms_norm_,
    pixel_mlp_,
    pixel_rms_norm,
    qk_norm_rope,
    rms_norm_modulate,
)


def linear(x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
    """Linear in the weight dtype, i.e. autocast's matmul precision."""
    return F.linear(x.to(layer.weight.dtype), layer.weight, layer.bias)


def swiglu(x: torch.Tensor, mlp: nn.Module) -> torch.Tensor:
    return linear(F.silu(linear(x, mlp.w1)) * linear(x, mlp.w3), mlp.w2)


class RopeTable:
    """FP32 cos/sin tables for ``[text | image]`` token order."""

    def __init__(self, cos: torch.Tensor, sin: torch.Tensor):
        self.cos = cos.contiguous()
        self.sin = sin.contiguous()

    @classmethod
    def from_complex(cls, *tables: torch.Tensor, device: torch.device) -> "RopeTable":
        cis = torch.view_as_real(torch.cat(tables)).to(device)
        return cls(cis[..., 0], cis[..., 1])


def check_supported(dit: IrisDiT) -> None:
    """Reject topologies and parameter dtypes the fused forward does not handle."""
    cfg = dit.cfg
    requirements = {
        "a pixel stage with post-modulated blocks": cfg.pixel.enabled and cfg.pixel.modulation == "post",
        "shared_bias modulation with a dual-stream prefix": cfg.modulation == "shared_bias"
        and cfg.dual_depth > 0,
        "a single-stream final block": isinstance(dit.blocks[-1], SingleStreamBlock),
        "grouped-query, gated, sandwich-normed, QK-normed attention": (
            cfg.num_kv_heads is not None
            and cfg.num_kv_heads < cfg.num_heads
            and cfg.gated_attention
            and cfg.sandwich_norm
            and cfg.qk_norm
        ),
        f"{HEAD_DIM}-wide attention heads": (
            cfg.hidden_size // cfg.num_heads == HEAD_DIM
            and cfg.pixel.attn_hidden_size // cfg.pixel.num_heads == HEAD_DIM
        ),
        "a masked text adapter with text RoPE and position table": (
            cfg.text_adapter != "linear" and cfg.text_rope and cfg.text_abs_pos_embed
        ),
    }
    missing = [name for name, ok in requirements.items() if not ok]
    if missing:
        raise ValueError(f"MetalIris supports the released Iris-3B topology; this model lacks {missing}")

    # the Metal kernels read norm gains, adaLN biases and tables as FP32
    linear_params = {id(p) for m in dit.modules() if isinstance(m, nn.Linear) for p in m.parameters()}
    compute = dit.s_embedder.proj.weight.dtype
    for name, param in dit.named_parameters():
        expected = compute if id(param) in linear_params else torch.float32
        if param.dtype != expected:
            raise ValueError(f"{name} is {param.dtype}, expected {expected}; load weights with load_dit")


class MetalIris:
    """Velocity predictor ``(x, t, text) -> v`` for a loaded ``IrisDiT``.

    ``text`` is the output of :meth:`encode_text`, not raw encoder states.
    """

    def __init__(self, dit: IrisDiT, attention_dtype: torch.dtype | None = None):
        check_supported(dit)
        cfg = dit.cfg
        self.dit = dit
        self.cfg = cfg
        self.device = dit.s_embedder.proj.weight.device
        self.dtype = dit.s_embedder.proj.weight.dtype
        self.attention_dtype = attention_dtype or self.dtype
        self._ropes: dict[tuple[int, int], tuple[RopeTable, RopeTable]] = {}

    # -- conditioning ------------------------------------------------------
    @torch.no_grad()
    def encode_text(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encoder states ``[B, T, layers, dim]`` -> trunk text tokens ``[B, T, D]`` (FP32).

        The adapter is evaluated with the upstream modules under autocast, the
        same arithmetic as the reference sampler, once per prompt.
        """
        dit, n = self.dit, self.cfg.text_len
        low_precision = self.dtype != torch.float32
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=low_precision):
            # FP32 states, as the reference sampler casts them to the weight dtype
            tokens = dit.y_embedder(states[:, :n].to(self.device, torch.float32), mask[:, :n].to(self.device))
        return tokens.float() + dit.y_pos_embedding[:, : tokens.shape[1]].float()

    def _rope(self, grid: tuple[int, int], n_text: int) -> tuple[RopeTable, RopeTable]:
        """(trunk ``[text | image]`` table, pixel-stage image table) for a patch grid."""
        key = (grid, n_text)
        if key not in self._ropes:
            cfg = self.cfg
            image = rope_2d(
                HEAD_DIM,
                *grid,
                theta=cfg.rope_theta,
                scale=cfg.rope_scale,
                aspect=cfg.rope_aspect,
                frame_pairs=cfg.rope_frame_pairs,
                frame_theta=cfg.rope_frame_theta,
            )
            pixel = rope_2d(
                cfg.pixel.attn_hidden_size // cfg.pixel.num_heads,
                *grid,
                theta=cfg.rope_theta,
                scale=cfg.rope_scale,
                aspect=cfg.rope_aspect,
            )
            text = rope_1d(HEAD_DIM, n_text, theta=cfg.text_rope_theta)
            self._ropes[key] = (
                RopeTable.from_complex(text, image, device=self.device),
                RopeTable.from_complex(pixel, device=self.device),
            )
        return self._ropes[key]

    # -- forward -----------------------------------------------------------
    @torch.no_grad()
    def __call__(self, x: torch.Tensor, t: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, 3, H, W]``; ``t``: model time ``[B]``; returns FP32 velocity."""
        dit, p = self.dit, self.cfg.patch_size
        batch, channels, height, width = x.shape
        if height % p or width % p:
            raise ValueError(f"input {height}x{width} is not divisible by patch_size {p}")
        grid = (height // p, width // p)
        n_text = text.shape[1]
        rope, pixel_rope = self._rope(grid, n_text)

        patches = x.reshape(batch, channels, grid[0], p, grid[1], p).permute(0, 2, 4, 1, 3, 5)
        image = linear(patches.reshape(batch, grid[0] * grid[1], -1), dit.s_embedder.proj)
        t_emb = dit.t_embedder(t)  # [B, 1, D], BF16
        cond = F.silu(t_emb)
        cores = {name: linear(cond, core) for name, core in dit.modulation_cores.items()}

        # residual streams are FP32 and updated in place
        image, text = image.float(), text.float().clone()
        tokens = None
        for i, block in enumerate(dit.blocks):
            if isinstance(block, MMDiTBlock):
                self._dual_block(block, image, text, cores, rope)
                continue
            if tokens is None:
                tokens = torch.cat([text, image], dim=1)
            last = i == len(dit.blocks) - 1
            tokens = self._single_block(block, tokens, n_text, cores["adaln_img"], rope, last)
        image = tokens  # the final block returns image rows only

        s_cond = F.silu(t_emb + image).reshape(batch * grid[0] * grid[1], -1)
        pixels = dit.pixel_embedder(x.to(self.dtype))
        blocks = dit.pixel_blocks
        h = pixel_rms_norm(pixels, blocks[0].norm1.weight, blocks[0].norm1.eps)
        for i, block in enumerate(blocks):
            next_norm = blocks[i + 1].norm1 if i + 1 < len(blocks) else dit.final_layer.norm
            h = self._pixel_block(block, pixels, h, s_cond, grid, pixel_rope, next_norm)
        out = linear(h, dit.final_layer.linear)
        out_channels = out.shape[-1]  # 3, also for task models with an extra input channel
        out = out.reshape(batch, grid[0], grid[1], p, p, out_channels).permute(0, 5, 1, 3, 2, 4)
        return out.reshape(batch, out_channels, height, width).float()

    # -- blocks ------------------------------------------------------------
    def _norm_modulate(
        self, x: torch.Tensor, norm: RMSNorm, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return rms_norm_modulate(x, norm.weight, shift, scale, norm.eps, self.dtype)

    @staticmethod
    def _residual(residual: torch.Tensor, gate: torch.Tensor, branch: torch.Tensor, norm: RMSNorm) -> None:
        gated_residual_rms_norm_(residual, branch, norm.weight, gate, norm.eps)

    def _heads(
        self, projection: torch.Tensor, norm: RMSNorm, rope: RopeTable, heads: int, **kwargs
    ) -> torch.Tensor:
        return qk_norm_rope(
            projection,
            norm.weight,
            rope.cos,
            rope.sin,
            norm.eps,
            heads,
            out_dtype=self.attention_dtype,
            **kwargs,
        )

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """``q``, ``k``: ``[B, H, N, 128]`` (K possibly grouped); ``v``: ``[B, N, H_kv*128]``.
        Returns ``[B, N, H*128]`` in the compute dtype."""
        batch, heads, tokens, _ = q.shape
        v_heads = k.new_empty(k.shape).copy_(v.unflatten(2, (k.shape[1], HEAD_DIM)).transpose(1, 2))
        out = F.scaled_dot_product_attention(q, k, v_heads, enable_gqa=k.shape[1] != heads)
        merged = q.new_empty(batch, tokens, heads, HEAD_DIM, dtype=self.dtype).copy_(out.transpose(1, 2))
        return merged.flatten(2)

    def _dual_block(
        self,
        block: MMDiTBlock,
        x: torch.Tensor,
        y: torch.Tensor,
        cores: dict[str, torch.Tensor],
        rope: RopeTable,
    ) -> None:
        """Updates the FP32 image ``x`` and text ``y`` streams in place."""
        mods_x = cores["adaln_img"] + block.adaln_img.bias
        mods_y = cores["adaln_txt"] + block.adaln_txt.bias
        xs1, xc1, xg1, xs2, xc2, xg2 = mods_x.chunk(6, dim=-1)
        ys1, yc1, yg1, ys2, yc2, yg2 = mods_y.chunk(6, dim=-1)
        attn = block.attn
        n_text, n_tokens = y.shape[1], y.shape[1] + x.shape[1]
        hx = self._norm_modulate(x, block.norm_x1, xs1, xc1)
        hy = self._norm_modulate(y, block.norm_y1, ys1, yc1)

        batch = x.shape[0]
        q = x.new_empty(batch, attn.num_heads, n_tokens, HEAD_DIM, dtype=self.attention_dtype)
        k = x.new_empty(batch, attn.num_kv_heads, n_tokens, HEAD_DIM, dtype=self.attention_dtype)
        for h, proj_q, proj_k, norm_q, norm_k, offset in (
            (hy, attn.q_proj_y, attn.k_proj_y, attn.q_norm_y, attn.k_norm_y, 0),
            (hx, attn.q_proj_x, attn.k_proj_x, attn.q_norm_x, attn.k_norm_x, n_text),
        ):
            self._heads(linear(h, proj_q), norm_q, rope, attn.num_heads, out=q, offset=offset)
            self._heads(linear(h, proj_k), norm_k, rope, attn.num_kv_heads, out=k, offset=offset)
        v = torch.cat([linear(hy, attn.v_proj_y), linear(hx, attn.v_proj_x)], dim=1)
        out = self._attention(q, k, v)

        out_x = out[:, n_text:] * torch.sigmoid(linear(hx, block.attn_gate_x))
        out_y = out[:, :n_text] * torch.sigmoid(linear(hy, block.attn_gate_y))
        self._residual(x, xg1, linear(out_x, attn.proj_x), block.attn_post_norm_x)
        self._residual(y, yg1, linear(out_y, attn.proj_y), block.attn_post_norm_y)
        mlp_x = swiglu(self._norm_modulate(x, block.norm_x2, xs2, xc2), block.mlp_x)
        mlp_y = swiglu(self._norm_modulate(y, block.norm_y2, ys2, yc2), block.mlp_y)
        self._residual(x, xg2, mlp_x, block.mlp_post_norm_x)
        self._residual(y, yg2, mlp_y, block.mlp_post_norm_y)

    def _single_block(
        self,
        block: SingleStreamBlock,
        tokens: torch.Tensor,
        n_text: int,
        core: torch.Tensor,
        rope: RopeTable,
        image_only: bool,
    ) -> torch.Tensor:
        """``tokens``: FP32 ``[B, T + L, D]``, updated in place. With ``image_only``
        the text rows still feed attention but are dropped from the output."""
        s1, c1, g1, s2, c2, g2 = (core + block.adaln.bias).chunk(6, dim=-1)
        h = self._norm_modulate(tokens, block.norm1, s1, c1)
        q = self._heads(linear(h, block.q_proj), block.q_norm, rope, block.num_heads)
        k = self._heads(linear(h, block.k_proj), block.k_norm, rope, block.num_kv_heads)
        out = self._attention(q, k, linear(h, block.v_proj))
        if image_only:
            out, h, tokens = out[:, n_text:], h[:, n_text:], tokens[:, n_text:].contiguous()
        out = out * torch.sigmoid(linear(h, block.attn_gate))
        self._residual(tokens, g1, linear(out, block.attn_proj), block.attn_post_norm)
        mlp = swiglu(self._norm_modulate(tokens, block.norm2, s2, c2), block.mlp)
        self._residual(tokens, g2, mlp, block.mlp_post_norm)
        return tokens

    def _pixel_block(
        self,
        block: PiTBlock,
        pixels: torch.Tensor,
        h: torch.Tensor,
        cond: torch.Tensor,
        grid: tuple[int, int],
        rope: RopeTable,
        next_norm: RMSNorm,
    ) -> torch.Tensor:
        """Updates ``pixels`` in place and returns ``next_norm(pixels)``.

        ``pixels``: ``[B*L, p*p, d_pix]`` in the compute dtype, like the upstream
        autocast pixel stream; ``h``: ``block.norm1(pixels)``; ``cond``: FP32 ``[B*L, D]``.
        """
        if block.modulation != "post":
            raise ValueError("MetalIris supports post-modulated PiT blocks only")
        mods = linear(cond, block.adaln).unflatten(1, (block.pixels_per_patch, -1))
        n_patches = grid[0] * grid[1]
        h = h.reshape(pixels.shape[0] // n_patches, n_patches, -1)
        attn = block.attn
        qkv = linear(linear(h, block.compress), attn.qkv)
        width = attn.num_heads * HEAD_DIM
        q = self._heads(qkv, attn.q_norm, rope, attn.num_heads)
        k = self._heads(qkv, attn.k_norm, rope, attn.num_heads, col_offset=width)
        out = linear(self._attention(q, k, qkv[..., 2 * width :]), attn.proj)
        out = linear(out, block.expand).reshape_as(pixels)
        mlp = block.mlp
        return pixel_mlp_(
            pixels, out, mods, block.norm2.weight, mlp.fc1, mlp.fc2, next_norm.weight, next_norm.eps
        )
