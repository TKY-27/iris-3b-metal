import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from iris3b.config import TextEncoderConfig
from iris3b.text.qwen3_vl import _PROMPT_PREFIX, _PROMPT_SUFFIX
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
