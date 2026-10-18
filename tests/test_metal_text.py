import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from iris3b.config import TextEncoderConfig
from iris3b.text.qwen3_vl import _PROMPT_PREFIX, _PROMPT_SUFFIX
from iris_metal import text as text_module
from iris_metal.text import StreamingQwen3VLEncoder

transformers = pytest.importorskip("transformers")
QWEN = "Qwen/Qwen3-VL-4B-Instruct"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]


@pytest.fixture(scope="module")
def tokenizer_dir():
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(QWEN, allow_patterns=TOKENIZER_FILES, local_files_only=True)
    except Exception:
        pytest.skip("the Qwen3-VL tokenizer is not in the local Hugging Face cache")


def _tiny_text_model(tmp_path, tokenizer_dir):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    text_config = {
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "rope_scaling": {"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True},
        "rope_theta": 5e6,
    }
    torch.manual_seed(0)
    model = Qwen3VLTextModel(Qwen3VLTextConfig(**text_config)).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.02)  # norms start at exactly 1
    weights = {f"model.language_model.{k}": v.contiguous() for k, v in model.state_dict().items()}
    save_file(weights, tmp_path / "model.safetensors")
    weight_map = dict.fromkeys(weights, "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (tmp_path / "config.json").write_text(json.dumps({"text_config": text_config}))
    for name in TOKENIZER_FILES:
        shutil.copy(Path(tokenizer_dir) / name, tmp_path / name)
    return model


@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize(
    "shard", ["../other.safetensors", "/other.safetensors", "..\\other.safetensors", ".", ".."]
)
def test_streaming_rejects_shard_traversal_before_opening_weights(tmp_path, monkeypatch, local, shard):
    config = {
        "hidden_size": 64,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5e6,
    }
    (tmp_path / "config.json").write_text(json.dumps({"text_config": config}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.language_model.embed_tokens.weight": shard}})
    )
    if not local:
        monkeypatch.setattr(text_module, "cached_snapshot", lambda *args, **kwargs: None)
        monkeypatch.setattr(text_module, "snapshot_download", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(text_module, "safe_open", lambda *args, **kwargs: pytest.fail("opened weight"))
    cfg = TextEncoderConfig(
        pretrained=str(tmp_path) if local else QWEN, dim=64, max_length=24, hidden_layers=[1]
    )
    with pytest.raises(ValueError, match="flat shard files"):
        StreamingQwen3VLEncoder(cfg, torch.device("cpu"))


def test_streaming_encoder_matches_padded_transformers_forward(tmp_path, tokenizer_dir):
    model = _tiny_text_model(tmp_path, tokenizer_dir)
    cfg = TextEncoderConfig(pretrained=str(tmp_path), dim=64, max_length=24, hidden_layers=[1, 3, 4])
    encoder = StreamingQwen3VLEncoder(cfg, torch.device("cpu"), dtype=torch.float32)
    texts = ["a red fox in fresh snow", ""]
    actual = encoder.encode(texts)

    # the upstream recipe: template, right padding to max_length, attention mask
    tok = encoder.tokenizer
    prefix = tok.encode(_PROMPT_PREFIX, add_special_tokens=False)
    suffix = tok.encode(_PROMPT_SUFFIX, add_special_tokens=False)
    width = len(prefix) + cfg.max_length
    ids = torch.full((len(texts), width), tok.pad_token_id or 0)
    attention = torch.zeros(len(texts), width, dtype=torch.long)
    for row, text in enumerate(texts):
        seq = prefix + tok.encode(text, add_special_tokens=False) + suffix
        ids[row, : len(seq)] = torch.tensor(seq)
        attention[row, : len(seq)] = 1
    with torch.no_grad():
        hidden = model(input_ids=ids, attention_mask=attention, output_hidden_states=True).hidden_states
    mask = attention[:, len(prefix) :]
    expected = torch.stack([hidden[layer][:, len(prefix) :] for layer in cfg.hidden_layers], dim=2)
    expected = expected * mask[:, :, None, None]

    assert torch.equal(actual.mask, mask)
    torch.testing.assert_close(actual.embeddings, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or os.environ.get("IRIS_SKIP_GPU_TESTS") == "1",
    reason="needs an Apple Silicon GPU",
)
def test_streaming_error_waits_and_releases_shard_storage(tmp_path, tokenizer_dir, monkeypatch):
    _tiny_text_model(tmp_path, tokenizer_dir)
    cfg = TextEncoderConfig(pretrained=str(tmp_path), dim=64, max_length=24, hidden_layers=[1, 3, 4])
    encoder = StreamingQwen3VLEncoder(cfg, torch.device("mps"))

    def fail(*args):
        assert encoder._sources and encoder._readers
        raise RuntimeError("interrupted layer")

    original = encoder._layer
    monkeypatch.setattr(encoder, "_layer", fail)
    with pytest.raises(RuntimeError, match="interrupted layer"):
        encoder.encode(["a fox"])
    assert not encoder._sources and not encoder._readers
    monkeypatch.setattr(encoder, "_layer", original)
    result = encoder.encode(["a fox"])
    assert torch.isfinite(result.embeddings).all()
    assert not encoder._sources and not encoder._readers


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available() or os.environ.get("IRIS_SKIP_GPU_TESTS") == "1",
                reason="needs an Apple Silicon GPU",
            ),
        ),
    ],
)
def test_streaming_encoder_stops_between_layers_and_stays_exact(tmp_path, tokenizer_dir, device):
    _tiny_text_model(tmp_path, tokenizer_dir)
    cfg = TextEncoderConfig(pretrained=str(tmp_path), dim=64, max_length=24, hidden_layers=[1, 3, 4])
    encoder = StreamingQwen3VLEncoder(cfg, torch.device(device))
    expected = encoder.encode(["a fox", ""])
    reads = []

    def stop_at_third_layer():
        reads.append(1)
        if len(reads) == 3:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        encoder.encode(["a fox", ""], raise_if_cancelled=stop_at_third_layer)
    assert len(reads) == 3 and not encoder._sources and not encoder._readers
    actual = encoder.encode(["a fox", ""], raise_if_cancelled=lambda: None)
    assert actual.embeddings.view(torch.int16).equal(expected.embeddings.view(torch.int16))
    assert actual.mask.equal(expected.mask)
