from types import SimpleNamespace

import torch

from iris_metal import pipeline
from iris_metal.pipeline import IrisPipeline


class FakeEncoder:
    """Encodes a text as its length, so results are easy to check."""

    batches: list[list[str]] = []

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts):
        FakeEncoder.batches.append(list(texts))
        states = torch.tensor([[float(len(text))] for text in texts])
        return SimpleNamespace(embeddings=states, mask=torch.ones(len(texts), 1))


def _pipeline(tmp_path, monkeypatch, cache_size=16):
    (tmp_path / "config.yaml").write_text("model: {}\n")
    FakeEncoder.batches = []
    monkeypatch.setattr(pipeline, "StreamingQwen3VLEncoder", FakeEncoder)
    monkeypatch.setattr(IrisPipeline, "ENCODING_CACHE_SIZE", cache_size)
    return IrisPipeline(tmp_path, device="cpu")


def _values(encodings):
    return [states.item() for states, _ in encodings]


def test_missing_texts_are_encoded_once_in_one_batch(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path, monkeypatch)
    assert _values(pipe.encode(["aa", "b", "aa", ""])) == [2, 1, 2, 0]
    assert _values(pipe.encode(["b", ""])) == [1, 0]
    assert FakeEncoder.batches == [["aa", "b", ""]]


def test_requested_texts_survive_eviction(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path, monkeypatch, cache_size=2)
    for prompt in ("a", "bb", "ccc", "a"):
        assert _values(pipe.encode([prompt, ""])) == [len(prompt), 0]
    # "a" was evicted by the size-2 cache and re-encoded; the negative never was
    assert FakeEncoder.batches == [["a", ""], ["bb"], ["ccc"], ["a"]]
