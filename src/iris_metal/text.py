"""Prompt encoding with the Qwen3-VL language stack, one decoder layer at a time.

Produces the same states as the upstream ``Qwen3VLTextEncoder`` (selected
hidden layers over the caption and template suffix, zero-padded to
``max_length``) without holding the 8 GB model:

- the template is right-padded and attention is causal, so real tokens never
  see the padding; only the real tokens are run;
- layers past the deepest selected hidden layer, the final norm and the vision
  tower are never loaded;
- each layer's weights are read from the safetensors shards, used for every
  prompt, and released before the next layer is read.

The arithmetic follows the transformers implementation in BF16: FP32
RMSNorm statistics, BF16 linears, residuals and rotary products.
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

from iris3b.config import TextEncoderConfig
from iris3b.text.base import TextEncoding
from iris3b.text.qwen3_vl import _PROMPT_PREFIX, _PROMPT_SUFFIX

logger = logging.getLogger(__name__)

_FILES = ["config.json", "tokenizer*", "vocab.json", "merges.txt", "model*.safetensors*"]
# the Qwen3-VL-4B-Instruct snapshot the engine was validated against
VALIDATED_REVISIONS = {"Qwen/Qwen3-VL-4B-Instruct": "ebb281ec70b05090aa6165b016eac8ec08e71b17"}
_PREFIX = "model.language_model."


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    return weight * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class StreamingQwen3VLEncoder:
    def __init__(self, cfg: TextEncoderConfig, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoTokenizer

        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.hidden_layers = tuple(cfg.hidden_layers)
        if cfg.on_caption_overflow not in ("warn", "error", "silent"):
            raise ValueError(f"unknown text_encoder.on_caption_overflow {cfg.on_caption_overflow!r}")
        root = Path(cfg.pretrained)
        if not root.is_dir():
            revision = VALIDATED_REVISIONS.get(cfg.pretrained)
            root = Path(snapshot_download(cfg.pretrained, revision=revision, allow_patterns=_FILES))
        config = json.loads((root / "config.json").read_text())["text_config"]
        if config["hidden_size"] != cfg.dim:
            raise ValueError(f"text encoder width {config['hidden_size']} != text_encoder.dim {cfg.dim}")
        # the deepest hidden state transformers reports is normed; only earlier layers are raw
        last = config["num_hidden_layers"] - 1
        if not self.hidden_layers or list(self.hidden_layers) != sorted(set(self.hidden_layers)):
            raise ValueError("text_encoder.hidden_layers must be sorted, unique and non-empty")
        if self.hidden_layers[0] < 1 or self.hidden_layers[-1] > last:
            raise ValueError(f"text_encoder.hidden_layers must lie in [1, {last}]")
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.eps = config["rms_norm_eps"]
        theta = config.get("rope_theta") or config["rope_parameters"]["rope_theta"]
        self.inv_freq = 1.0 / theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)

        weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self._weight_map = {key: root / shard for key, shard in weight_map.items()}
        self._readers: dict[Path, safe_open] = {}

        self.tokenizer = AutoTokenizer.from_pretrained(root)
        self._prefix = self.tokenizer.encode(_PROMPT_PREFIX, add_special_tokens=False)
        self._suffix = self.tokenizer.encode(_PROMPT_SUFFIX, add_special_tokens=False)
        self.caption_budget = cfg.max_length - len(self._suffix)
        if self.caption_budget < 1:
            raise ValueError(f"text_encoder.max_length={cfg.max_length} leaves no room for a caption")

    def _reader(self, name: str) -> safe_open:
        path = self._weight_map[_PREFIX + name]
        if path not in self._readers:
            self._readers[path] = safe_open(str(path), framework="pt", device="cpu")
        return self._readers[path]

    def _tensor(self, name: str) -> torch.Tensor:
        return self._reader(name).get_tensor(_PREFIX + name).to(self.device, self.dtype)

    def _token_ids(self, texts: list[str]) -> list[list[int]]:
        captions = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        longest = max(len(ids) for ids in captions)
        if longest > self.caption_budget:
            message = f"prompt is {longest} tokens; only the first {self.caption_budget} are used"
            if self.cfg.on_caption_overflow == "error":
                raise ValueError(message)
            if self.cfg.on_caption_overflow == "warn":
                logger.warning(message)
        return [self._prefix + ids[: self.caption_budget] + self._suffix for ids in captions]

    def _embed(self, ids: list[int]) -> torch.Tensor:
        """Reads only the embedding rows in use, not the 0.8 GB table."""
        name = "embed_tokens.weight"
        table = self._reader(name).get_slice(_PREFIX + name)
        return torch.cat([table[i : i + 1] for i in ids]).to(self.device, self.dtype)[None]

    def _rope(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        angles = torch.arange(length).float()[:, None] * self.inv_freq
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(self.device, self.dtype), angles.sin().to(self.device, self.dtype)

    def _layer(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, w: dict) -> torch.Tensor:
        batch, length, _ = h.shape
        x = rms_norm(h, w["input_layernorm"], self.eps)
        q = F.linear(x, w["q_proj"]).view(batch, length, self.heads, self.head_dim)
        k = F.linear(x, w["k_proj"]).view(batch, length, self.kv_heads, self.head_dim)
        v = F.linear(x, w["v_proj"]).view(batch, length, self.kv_heads, self.head_dim)
        q = rms_norm(q, w["q_norm"], self.eps).transpose(1, 2)
        k = rms_norm(k, w["k_norm"], self.eps).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        attn = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), is_causal=True, enable_gqa=True)
        h = h + F.linear(attn.transpose(1, 2).reshape(batch, length, -1), w["o_proj"])
        x = rms_norm(h, w["post_attention_layernorm"], self.eps)
        return h + F.linear(F.silu(F.linear(x, w["gate_proj"])) * F.linear(x, w["up_proj"]), w["down_proj"])

    def encode(self, texts: list[str]) -> TextEncoding:
        """``[B, max_length, layers, dim]`` states and ``[B, max_length]`` mask for ``texts``."""
        try:
            return self._encode(texts)
        finally:
            self._readers.clear()  # closes the shard files (8 GB mapped) between requests

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> TextEncoding:
        sequences = self._token_ids(texts)
        states = [self._embed(ids) for ids in sequences]
        ropes = [self._rope(len(ids)) for ids in sequences]
        captured: list[list[torch.Tensor]] = [[] for _ in texts]
        names = {
            "input_layernorm": "input_layernorm",
            "post_attention_layernorm": "post_attention_layernorm",
            **{p: f"self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")},
            **{p: f"mlp.{p}" for p in ("gate_proj", "up_proj", "down_proj")},
        }
        for index in range(max(self.hidden_layers)):
            weights = {key: self._tensor(f"layers.{index}.{name}.weight") for key, name in names.items()}
            for i, (cos, sin) in enumerate(ropes):
                states[i] = self._layer(states[i], cos, sin, weights)
                if index + 1 in self.hidden_layers:
                    captured[i].append(states[i][0, len(self._prefix) :])
            del weights

        length = self.cfg.max_length
        embeddings = torch.zeros(len(texts), length, len(self.hidden_layers), self.cfg.dim, dtype=self.dtype)
        mask = torch.zeros(len(texts), length, dtype=torch.int64)
        for i, layers in enumerate(captured):
            stacked = torch.stack(layers, dim=1).cpu()
            embeddings[i, : stacked.shape[0]] = stacked
            mask[i, : stacked.shape[0]] = 1
        return TextEncoding(embeddings=embeddings, mask=mask)
