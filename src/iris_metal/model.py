"""Inference-only forward pass over the weights of an upstream ``IrisDiT``.

The upstream model is evaluated in its released configuration: FP32 master
weights under BF16 autocast. This module reproduces that arithmetic with BF16
linear weights (autocast would round them identically on every call) and FP32
norm gains, biases and residual streams, while restructuring the computation
for Apple GPUs:

- the text adapter depends only on the prompt and runs once per prompt, not
  once per sampling step;
- the shared adaLN cores run once per step instead of once per block;
- QK-RMSNorm, RoPE and the head transpose are one Metal kernel;
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
from iris_metal.kernels import HEAD_DIM, qk_norm_rope


def rms_norm(x: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
    """Upstream ``RMSNorm`` with an FP32 gain: normalize in FP32, round to the
    input dtype, scale in FP32. FP32 output."""
    if x.dtype == torch.float32:
        return F.rms_norm(x, (x.shape[-1],), norm.weight, norm.eps)
    return F.rms_norm(x, (x.shape[-1],), eps=norm.eps) * norm.weight


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.addcmul(shift, x, 1 + scale)


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


class MetalIris:
    """Velocity predictor ``(x, t, text) -> v`` for a loaded ``IrisDiT``.

    ``text`` is the output of :meth:`encode_text`, not raw encoder states.
    """

    def __init__(self, dit: IrisDiT, attention_dtype: torch.dtype | None = None):
        cfg = dit.cfg
        if not cfg.pixel.enabled or cfg.modulation != "shared_bias" or not cfg.text_rope:
            raise ValueError("MetalIris supports the released Iris-3B topology only")
        if not isinstance(dit.blocks[-1], SingleStreamBlock):
            raise ValueError("MetalIris expects a single-stream final block")
        if cfg.hidden_size // cfg.num_heads != HEAD_DIM:
            raise ValueError(f"MetalIris needs {HEAD_DIM}-wide attention heads")
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

        tokens = None
        for i, block in enumerate(dit.blocks):
            if isinstance(block, MMDiTBlock):
                image, text = self._dual_block(block, image, text, cores, rope)
                continue
            if tokens is None:
                tokens = torch.cat([text, image.float()], dim=1)
            last = i == len(dit.blocks) - 1
            tokens = self._single_block(block, tokens, n_text, cores["adaln_img"], rope, last)
        image = tokens  # the final block returns image rows only

        s_cond = F.silu(t_emb + image).reshape(batch * grid[0] * grid[1], -1)
        pixels = dit.pixel_embedder(x.to(self.dtype))
        for block in dit.pixel_blocks:
            pixels = self._pixel_block(block, pixels, s_cond, grid, pixel_rope)
        out = linear(rms_norm(pixels, dit.final_layer.norm), dit.final_layer.linear)
        out = out.reshape(batch, grid[0], grid[1], p, p, channels).permute(0, 5, 1, 3, 2, 4)
        return out.reshape(batch, channels, height, width).float()

    # -- blocks ------------------------------------------------------------
    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """``[B, H, N, 128]`` heads (K/V possibly grouped) -> ``[B, N, H*128]``."""
        v = v.to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=k.shape[1] != q.shape[1])
        return out.transpose(1, 2).to(self.dtype).flatten(2)

    def _dual_block(
        self,
        block: MMDiTBlock,
        x: torch.Tensor,
        y: torch.Tensor,
        cores: dict[str, torch.Tensor],
        rope: RopeTable,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mods_x = cores["adaln_img"] + block.adaln_img.bias
        mods_y = cores["adaln_txt"] + block.adaln_txt.bias
        xs1, xc1, xg1, xs2, xc2, xg2 = mods_x.chunk(6, dim=-1)
        ys1, yc1, yg1, ys2, yc2, yg2 = mods_y.chunk(6, dim=-1)
        attn = block.attn
        n_text, n_tokens = y.shape[1], y.shape[1] + x.shape[1]
        hx = modulate(rms_norm(x, block.norm_x1), xs1, xc1).to(self.dtype)
        hy = modulate(rms_norm(y, block.norm_y1), ys1, yc1).to(self.dtype)

        batch, eps = x.shape[0], attn.q_norm_x.eps
        q = x.new_empty(batch, attn.num_heads, n_tokens, HEAD_DIM, dtype=self.attention_dtype)
        k = x.new_empty(batch, attn.num_kv_heads, n_tokens, HEAD_DIM, dtype=self.attention_dtype)
        for h, proj_q, proj_k, norm_q, norm_k, offset in (
            (hy, attn.q_proj_y, attn.k_proj_y, attn.q_norm_y, attn.k_norm_y, 0),
            (hx, attn.q_proj_x, attn.k_proj_x, attn.q_norm_x, attn.k_norm_x, n_text),
        ):
            qk_norm_rope(
                linear(h, proj_q),
                norm_q.weight,
                rope.cos,
                rope.sin,
                eps,
                attn.num_heads,
                out=q,
                offset=offset,
            )
            qk_norm_rope(
                linear(h, proj_k),
                norm_k.weight,
                rope.cos,
                rope.sin,
                eps,
                attn.num_kv_heads,
                out=k,
                offset=offset,
            )
        v = torch.cat([linear(hy, attn.v_proj_y), linear(hx, attn.v_proj_x)], dim=1)
        v = v.unflatten(2, (attn.num_kv_heads, HEAD_DIM)).transpose(1, 2)
        out = self._attention(q, k, v)

        out_x = out[:, n_text:] * torch.sigmoid(linear(hx, block.attn_gate_x))
        out_y = out[:, :n_text] * torch.sigmoid(linear(hy, block.attn_gate_y))
        x = torch.addcmul(x, xg1, rms_norm(linear(out_x, attn.proj_x), block.attn_post_norm_x))
        y = torch.addcmul(y, yg1, rms_norm(linear(out_y, attn.proj_y), block.attn_post_norm_y))
        mlp_x = swiglu(modulate(rms_norm(x, block.norm_x2), xs2, xc2), block.mlp_x)
        mlp_y = swiglu(modulate(rms_norm(y, block.norm_y2), ys2, yc2), block.mlp_y)
        x = torch.addcmul(x, xg2, rms_norm(mlp_x, block.mlp_post_norm_x))
        y = torch.addcmul(y, yg2, rms_norm(mlp_y, block.mlp_post_norm_y))
        return x, y

    def _single_block(
        self,
        block: SingleStreamBlock,
        tokens: torch.Tensor,
        n_text: int,
        core: torch.Tensor,
        rope: RopeTable,
        image_only: bool,
    ) -> torch.Tensor:
        """``tokens``: FP32 ``[B, T + L, D]``. With ``image_only`` the text rows
        still feed attention but are dropped from the output."""
        s1, c1, g1, s2, c2, g2 = (core + block.adaln.bias).chunk(6, dim=-1)
        h = modulate(rms_norm(tokens, block.norm1), s1, c1).to(self.dtype)
        eps = block.q_norm.eps
        q = qk_norm_rope(
            linear(h, block.q_proj),
            block.q_norm.weight,
            rope.cos,
            rope.sin,
            eps,
            block.num_heads,
            out_dtype=self.attention_dtype,
        )
        k = qk_norm_rope(
            linear(h, block.k_proj),
            block.k_norm.weight,
            rope.cos,
            rope.sin,
            eps,
            block.num_kv_heads,
            out_dtype=self.attention_dtype,
        )
        v = linear(h, block.v_proj).unflatten(2, (block.num_kv_heads, HEAD_DIM)).transpose(1, 2)
        out = self._attention(q, k, v)
        if image_only:
            out, h, tokens = out[:, n_text:], h[:, n_text:], tokens[:, n_text:]
        out = out * torch.sigmoid(linear(h, block.attn_gate))
        tokens = torch.addcmul(tokens, g1, rms_norm(linear(out, block.attn_proj), block.attn_post_norm))
        mlp = swiglu(modulate(rms_norm(tokens, block.norm2), s2, c2), block.mlp)
        return torch.addcmul(tokens, g2, rms_norm(mlp, block.mlp_post_norm))

    def _pixel_block(
        self,
        block: PiTBlock,
        pixels: torch.Tensor,
        cond: torch.Tensor,
        grid: tuple[int, int],
        rope: RopeTable,
    ) -> torch.Tensor:
        """``pixels``: BF16 ``[B*L, p*p, d_pix]``; ``cond``: FP32 ``[B*L, D]``.

        The upstream pixel stream stays in BF16 under autocast; so does this one.
        """
        if block.modulation != "post":
            raise ValueError("MetalIris supports post-modulated PiT blocks only")
        mods = linear(cond, block.adaln).unflatten(1, (block.pixels_per_patch, -1))
        scale1, shift1, scale2, shift2 = mods.chunk(4, dim=-1)

        n_patches = grid[0] * grid[1]
        batch = pixels.shape[0] // n_patches
        h = rms_norm(pixels, block.norm1).to(self.dtype).reshape(batch, n_patches, -1)
        compact = linear(h, block.compress)
        attn = block.attn
        qkv = linear(compact, attn.qkv)
        width, eps = attn.num_heads * HEAD_DIM, attn.q_norm.eps
        q = qk_norm_rope(
            qkv, attn.q_norm.weight, rope.cos, rope.sin, eps, attn.num_heads, out_dtype=self.attention_dtype
        )
        k = qk_norm_rope(
            qkv,
            attn.k_norm.weight,
            rope.cos,
            rope.sin,
            eps,
            attn.num_heads,
            col_offset=width,
            out_dtype=self.attention_dtype,
        )
        v = qkv[..., 2 * width :].unflatten(2, (attn.num_heads, HEAD_DIM)).transpose(1, 2)
        out = linear(self._attention(q, k, v), attn.proj)
        out = linear(out, block.expand).reshape_as(pixels)
        pixels = pixels + modulate(out, shift1, scale1)
        mlp = linear(F.gelu(linear(rms_norm(pixels, block.norm2), block.mlp.fc1)), block.mlp.fc2)
        return pixels + modulate(mlp, shift2, scale2)
