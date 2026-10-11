import re
import sys
import weakref
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from iris3b.config import TextEncoderConfig
from iris3b.text.qwen3_vl import Qwen3VLTextEncoder


class _FakeTokenizer:
    """Whitespace/special-token BPE stub with no padding or truncation of its own."""

    last_instance: "_FakeTokenizer | None" = None
    pad_token_id = 0

    def __init__(self):
        self.calls: list[list[str]] = []
        self._vocab: dict[str, int] = {}
        type(self).last_instance = self

    @classmethod
    def from_pretrained(cls, _checkpoint: str) -> "_FakeTokenizer":
        return cls()

    def _ids(self, text: str) -> list[int]:
        pieces = re.findall(r"<\|[^>]+\|>|\w+|[^\s\w]", text, flags=re.UNICODE)
        return [self._vocab.setdefault(piece, len(self._vocab) + 1) for piece in pieces]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self._ids(text)

    def __call__(self, texts: list[str], *, add_special_tokens: bool) -> dict[str, list[list[int]]]:
        assert add_special_tokens is False
        self.calls.append(list(texts))
        return {"input_ids": [self._ids(text) for text in texts]}


class _FakeDecoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, num_hidden_layers=6)
        self.anchor = nn.Parameter(torch.ones(()))
        self.forward_calls = 0
        self.last_kwargs: dict[str, object] = {}

    def forward(self, input_ids, attention_mask, use_cache, return_dict, output_hidden_states=False):
        self.forward_calls += 1
        self.last_kwargs = {
            "input_ids": input_ids.detach().clone(),
            "attention_mask": attention_mask.detach().clone(),
            "use_cache": use_cache,
            "return_dict": return_dict,
            "output_hidden_states": output_hidden_states,
        }
        hidden = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        layer_states = tuple(hidden + 100 * layer for layer in range(7)) if output_hidden_states else None
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=layer_states)


class _BombVision:
    def __call__(self, *args, **kwargs):
        _FakeQwenModel.vision_calls += 1
        raise AssertionError("the Qwen vision tower must not run for T2I text encoding")


class _FakeQwenModel:
    hidden_size = 4
    last_decoder: _FakeDecoder | None = None
    last_load_kwargs: dict[str, object] = {}
    last_wrapper_ref: weakref.ReferenceType | None = None
    vision_calls = 0

    def __init__(self):
        self.visual = _BombVision()
        self.decoder = _FakeDecoder(self.hidden_size)
        type(self).last_decoder = self.decoder

    @classmethod
    def from_pretrained(cls, _checkpoint: str, **kwargs) -> "_FakeQwenModel":
        cls.last_load_kwargs = dict(kwargs)
        wrapper = cls()
        cls.last_wrapper_ref = weakref.ref(wrapper)
        return wrapper

    def get_decoder(self) -> _FakeDecoder:
        return self.decoder


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch, hidden_size: int = 4) -> None:
    module = ModuleType("transformers")
    module.AutoTokenizer = _FakeTokenizer
    module.Qwen3VLForConditionalGeneration = _FakeQwenModel
    monkeypatch.setitem(sys.modules, "transformers", module)
    _FakeQwenModel.hidden_size = hidden_size
    _FakeQwenModel.last_decoder = None
    _FakeQwenModel.last_load_kwargs = {}
    _FakeQwenModel.last_wrapper_ref = None
    _FakeQwenModel.vision_calls = 0
    _FakeTokenizer.last_instance = None


def _config(tmp_path, **kwargs) -> TextEncoderConfig:
    values = {
        "name": "qwen3_vl",
        "pretrained": "Qwen/Qwen3-VL-test",
        "dim": 4,
        "max_length": 8,
        "hidden_layers": [],
        "null_embed_dir": str(tmp_path),
    }
    values.update(kwargs)
    return TextEncoderConfig(**values)


def test_text_only_forward_strips_template_and_returns_fixed_masked_states(monkeypatch, tmp_path):
    _install_fake_transformers(monkeypatch)
    encoder = Qwen3VLTextEncoder(_config(tmp_path))
    result = encoder.encode(["red panda", "小熊猫"])

    assert result.embeddings.shape == (2, 8, 4)
    assert result.mask.shape == (2, 8)
    assert result.mask.dtype == torch.int64

    tokenizer = _FakeTokenizer.last_instance
    decoder = _FakeQwenModel.last_decoder
    assert tokenizer is not None and decoder is not None
    assert tokenizer.calls[-1] == ["red panda", "小熊猫"]
    start = encoder._prefix_len
    fed_ids = decoder.last_kwargs["input_ids"]
    fed_mask = decoder.last_kwargs["attention_mask"]
    assert fed_ids.shape == (2, start + encoder.max_length)
    assert torch.equal(fed_ids[:, :start], encoder._prefix_ids.expand(2, -1))
    expected_ids = fed_ids[:, start : start + encoder.max_length]
    expected_mask = fed_mask[:, start : start + encoder.max_length]
    torch.testing.assert_close(
        result.embeddings[..., 0], (expected_ids * expected_mask).to(result.embeddings.dtype)
    )
    assert torch.equal(result.mask, expected_mask)
    assert torch.count_nonzero(result.embeddings[result.mask == 0]) == 0

    assert decoder.last_kwargs["use_cache"] is False
    assert decoder.last_kwargs["return_dict"] is True
    assert decoder.training is False
    assert all(not parameter.requires_grad for parameter in decoder.parameters())
    assert _FakeQwenModel.vision_calls == 0
    assert _FakeQwenModel.last_wrapper_ref is not None
    assert _FakeQwenModel.last_wrapper_ref() is None
    assert _FakeQwenModel.last_load_kwargs["attn_implementation"] == "sdpa"
    assert _FakeQwenModel.last_load_kwargs["dtype"] is torch.bfloat16


def test_selected_post_block_layers_are_stacked_and_pad_masked(monkeypatch, tmp_path):
    _install_fake_transformers(monkeypatch)
    encoder = Qwen3VLTextEncoder(_config(tmp_path, hidden_layers=[1, 3, 6]))
    result = encoder.encode(["red panda", "小熊猫"])

    assert result.embeddings.shape == (2, 8, 3, 4)
    decoder = _FakeQwenModel.last_decoder
    tokenizer = _FakeTokenizer.last_instance
    assert decoder is not None and tokenizer is not None
    assert decoder.last_kwargs["output_hidden_states"] is True
    start = encoder._prefix_len
    ids = decoder.last_kwargs["input_ids"][:, start : start + encoder.max_length]
    mask = decoder.last_kwargs["attention_mask"][:, start : start + encoder.max_length]
    for slot, layer in enumerate((1, 3, 6)):
        expected = (ids + 100 * layer) * mask
        torch.testing.assert_close(result.embeddings[:, :, slot, 0], expected.to(torch.float32))
    assert torch.count_nonzero(result.embeddings[result.mask == 0]) == 0
    assert "_layers-1-3-6.pth" in encoder._null_cache_path().name


@pytest.mark.parametrize("layers", [[2, 1], [1, 1], [0], [7]])
def test_hidden_layer_indices_must_be_sorted_unique_and_in_range(monkeypatch, tmp_path, layers):
    _install_fake_transformers(monkeypatch)
    with pytest.raises(ValueError, match="hidden_layers"):
        Qwen3VLTextEncoder(_config(tmp_path, hidden_layers=layers))


def test_null_and_negative_prompts_use_template_and_empty_null_is_cached(monkeypatch, tmp_path):
    _install_fake_transformers(monkeypatch)
    encoder = Qwen3VLTextEncoder(_config(tmp_path))
    decoder = _FakeQwenModel.last_decoder
    tokenizer = _FakeTokenizer.last_instance
    assert decoder is not None and tokenizer is not None

    start = encoder._prefix_len
    suffix_len = encoder._suffix_len

    first = encoder.null("")
    assert decoder.forward_calls == 1
    assert tokenizer.calls[-1] == [""]
    # empty caption: the template prefix is followed straight by the assistant turn
    ids = decoder.last_kwargs["input_ids"]
    assert torch.equal(ids[0, :start], encoder._prefix_ids)
    assert torch.equal(ids[0, start : start + suffix_len], encoder._suffix_ids)

    second = encoder.null("")
    assert decoder.forward_calls == 1
    assert torch.equal(first.embeddings, second.embeddings)
    assert torch.equal(first.mask, second.mask)

    encoder.null("low quality")
    assert decoder.forward_calls == 2
    assert tokenizer.calls[-1] == ["low quality"]
    caption = torch.tensor(encoder.tokenizer.encode("low quality"))
    ids = decoder.last_kwargs["input_ids"]
    assert torch.equal(ids[0, start : start + len(caption)], caption)
    assert torch.equal(ids[0, start + len(caption) : start + len(caption) + suffix_len], encoder._suffix_ids)


def test_checkpoint_width_must_match_config(monkeypatch, tmp_path):
    _install_fake_transformers(monkeypatch, hidden_size=5)
    with pytest.raises(ValueError, match="hidden size is 5.*text_encoder.dim=4"):
        Qwen3VLTextEncoder(_config(tmp_path))


def test_missing_native_transformers_support_has_clear_error(monkeypatch, tmp_path):
    module = ModuleType("transformers")
    module.AutoTokenizer = _FakeTokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)
    with pytest.raises(RuntimeError, match="transformers>=4.57.0"):
        Qwen3VLTextEncoder(_config(tmp_path))
