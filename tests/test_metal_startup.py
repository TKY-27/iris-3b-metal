import pytest
import torch

from iris_metal import cache
from test_metal_pipeline import _pipeline


@pytest.mark.parametrize(
    "partial", [None, "config.json", "tokenizer.json", "model.safetensors.index.json", "shard.safetensors"]
)
def test_complete_and_partial_language_cache(tmp_path, monkeypatch, partial):
    files = {
        "config.json": "{}",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "model.safetensors.index.json": '{"weight_map":{"model.language_model.x":"shard.safetensors"}}',
        "shard.safetensors": "fixture",
    }
    for name, data in files.items():
        if name != partial:
            (tmp_path / name).write_text(data)
    monkeypatch.setattr(
        cache,
        "try_to_load_from_cache",
        lambda repo, name, revision: str(tmp_path / name) if (tmp_path / name).exists() else None,
    )
    actual = cache.cached_snapshot("repo", "a" * 40, language=True)
    assert actual == (tmp_path if partial is None else None)


def test_mutable_revision_does_not_use_local_shortcut(monkeypatch):
    monkeypatch.setattr(
        cache, "try_to_load_from_cache", lambda *a, **k: pytest.fail("mutable revision must use hub")
    )
    assert cache.cached_snapshot("repo", "main") is None
    assert cache.cached_snapshot("repo", None) is None


@pytest.mark.parametrize("index", ['{"weight_map":[]}', '{"weight_map":{}}', "invalid"])
def test_invalid_language_index_falls_back(tmp_path, monkeypatch, index):
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        (tmp_path / name).write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(index)
    monkeypatch.setattr(cache, "try_to_load_from_cache", lambda repo, name, revision: str(tmp_path / name))
    assert cache.cached_snapshot("repo", "a" * 40, language=True) is None


def test_cache_rejects_mixed_snapshots_and_shard_traversal(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root in (first, second):
        (root / "config.json").write_text("{}")
        (root / "tokenizer.json").write_text("{}")
        (root / "tokenizer_config.json").write_text("{}")
        (root / "model.safetensors.index.json").write_text(
            '{"weight_map":{"model.language_model.x":"shard.safetensors"}}'
        )
        (root / "shard.safetensors").write_text("fixture")
    monkeypatch.setattr(
        cache,
        "try_to_load_from_cache",
        lambda repo, name, revision: str((second if name == "tokenizer.json" else first) / name),
    )
    assert cache.cached_snapshot("repo", "a" * 40, language=True) is None
    monkeypatch.setattr(cache, "try_to_load_from_cache", lambda repo, name, revision: str(first / name))
    (first / "model.safetensors.index.json").write_text(
        '{"weight_map":{"model.language_model.x":"../second/shard.safetensors"}}'
    )
    assert cache.cached_snapshot("repo", "a" * 40, language=True) is None


def test_checkpoint_cache_and_existing_directory(tmp_path, monkeypatch):
    from iris_metal.cli import resolve_checkpoint

    (tmp_path / "config.yaml").write_text("model: {}")
    (tmp_path / "model.safetensors").write_text("fixture")
    monkeypatch.setattr(cache, "try_to_load_from_cache", lambda repo, name, revision: str(tmp_path / name))
    assert cache.cached_snapshot("repo", "a" * 40) == tmp_path
    (tmp_path / "model.safetensors").unlink()
    assert cache.cached_snapshot("repo", "a" * 40) is None
    assert resolve_checkpoint(str(tmp_path), None) == tmp_path


class AdaptEngine:
    dtype = text_dtype = attention_dtype = torch.float32

    def __init__(self):
        self.dit = object()
        self.calls = []

    def encode_text(self, states, mask):
        self.calls.append(states.clone())
        return states + 2


def test_adapted_cache_order_duplicates_versions_and_unload(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path, monkeypatch)
    engine = AdaptEngine()
    pipe._engine = engine
    texts = ["aa", "b", "aa", ""]
    raw = pipe.encode(texts)
    first = pipe._conditioning(texts, raw, engine)
    assert first.flatten().tolist() == [4, 3, 4, 2]
    assert len(engine.calls) == 1 and len(engine.calls[0]) == 4
    assert torch.equal(pipe._conditioning(texts, raw, engine), first)
    assert len(engine.calls) == 1
    first.add_(100)
    assert pipe._conditioning(texts, raw, engine).flatten().tolist() == [4, 3, 4, 2]
    assert len(engine.calls) == 1
    raw[0][0].add_(1)
    assert pipe._conditioning(texts, raw, engine).flatten().tolist() == [5, 3, 5, 2]
    assert len(engine.calls) == 2
    engine.text_dtype = torch.bfloat16
    pipe._conditioning(texts, raw, engine)
    assert len(engine.calls) == 3
    other = AdaptEngine()
    pipe._conditioning(texts, raw, other)
    assert len(other.calls) == 1
    pipe.unload()
    assert not pipe._adapted and pipe._conditioning_identity is None


def test_adapted_cache_invalidates_after_adapter_or_position_mutation(tmp_path, monkeypatch):
    class MutableEngine(AdaptEngine):
        def __init__(self):
            super().__init__()
            self.dit = torch.nn.Module()
            self.dit.y_embedder = torch.nn.Linear(1, 1, bias=False)
            self.dit.y_pos_embedding = torch.nn.Parameter(torch.zeros(1, 1))

        def encode_text(self, states, mask):
            self.calls.append(states.clone())
            return self.dit.y_embedder(states) + self.dit.y_pos_embedding

    pipe = _pipeline(tmp_path, monkeypatch)
    engine = MutableEngine()
    raw = pipe.encode(["a"])
    first = pipe._conditioning(["a"], raw, engine)
    assert len(engine.calls) == 1
    assert torch.equal(pipe._conditioning(["a"], raw, engine), first)
    assert len(engine.calls) == 1
    with torch.no_grad():
        engine.dit.y_embedder.weight.add_(1)
    second = pipe._conditioning(["a"], raw, engine)
    assert len(engine.calls) == 2
    assert not torch.equal(second, first)
    with torch.no_grad():
        engine.dit.y_pos_embedding.add_(1)
    third = pipe._conditioning(["a"], raw, engine)
    assert len(engine.calls) == 3
    assert not torch.equal(third, second)


def test_combined_cache_budget_and_large_batch(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path, monkeypatch, cache_size=2)
    engine = AdaptEngine()
    for texts in [["a", ""], ["bb", ""], ["ccc", ""], ["a", "bb", "ccc", ""]]:
        raw = pipe.encode(texts)
        result = pipe._conditioning(texts, raw, engine)
        assert result.flatten().tolist() == [len(t) + 2 for t in texts]
        used = sum(pipe._encoding_bytes(e) for e in pipe._encodings.values()) + sum(
            pipe._encoding_bytes((e[1],)) for e in pipe._adapted.values()
        )
        assert used <= pipe._cache_budget


def test_inference_mode_inputs_preserve_public_api(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path, monkeypatch)
    engine = AdaptEngine()
    with torch.inference_mode():
        raw = pipe.encode(["a", ""])
        result = pipe._conditioning(["a", ""], raw, engine)
    assert result.flatten().tolist() == [3, 2]
    assert not pipe._adapted


def test_adapted_cache_keeps_recent_prompts_and_recurring_negative(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path, monkeypatch, cache_size=64)
    engine = AdaptEngine()
    for text in ["a", "b", "c", "d", "e"]:
        raw = pipe.encode([text, ""])
        pipe._conditioning([text], raw[:1], engine)
        pipe._conditioning([""], raw[-1:], engine)
    assert list(pipe._adapted) == [("c",), ("d",), ("e",), ("",)]


def test_cache_preserves_batch_dependent_arithmetic(tmp_path, monkeypatch):
    class BatchSensitiveEngine(AdaptEngine):
        def encode_text(self, states, mask):
            self.calls.append(states.clone())
            return states + 100 * len(states) + torch.arange(len(states))[:, None]

    pipe = _pipeline(tmp_path, monkeypatch, cache_size=64)
    engine = BatchSensitiveEngine()
    for texts in [["a"], ["a", "bb", "a"], ["bb", "a", "a"]]:
        raw = pipe.encode(texts)
        result = pipe._conditioning(texts, raw, engine)
        expected = [len(text) + 100 * len(texts) + i for i, text in enumerate(texts)]
        assert result.flatten().tolist() == expected
        calls = len(engine.calls)
        assert torch.equal(pipe._conditioning(texts, raw, engine), result)
        assert len(engine.calls) == calls


def test_partial_checkpoint_snapshot_uses_existing_download_fallback(tmp_path, monkeypatch):
    import huggingface_hub

    from iris_metal.cli import REPO_ID, REVISION, resolve_checkpoint

    (tmp_path / "config.yaml").write_text("model: {}")
    monkeypatch.setattr(cache, "try_to_load_from_cache", lambda repo, name, revision: str(tmp_path / name))
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert resolve_checkpoint(REPO_ID, None) == tmp_path
    assert calls == [
        (REPO_ID, {"revision": REVISION, "allow_patterns": ["config.yaml", "model.safetensors"]})
    ]


def test_task_exports_resolve_from_a_complete_immutable_cache(tmp_path, monkeypatch):
    import huggingface_hub

    from iris_metal.tasks import resolve_task

    folder = tmp_path / "snapshot/upscaler"
    folder.mkdir(parents=True)
    for name in ("config.yaml", "model.safetensors", "empty_prompt.safetensors"):
        (folder / name).write_text("fixture")
    monkeypatch.setattr(
        cache, "try_to_load_from_cache", lambda repo, name, revision: str(tmp_path / "snapshot" / name)
    )
    downloads = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: downloads.append(k) or str(tmp_path / "hub")
    )
    assert resolve_task("org/repo", "restoration", "a" * 40) == folder
    assert downloads == []
    assert resolve_task("org/repo", "restoration", "main") == tmp_path / "hub/upscaler"  # mutable
    (folder / "empty_prompt.safetensors").write_text("")  # partial download
    assert resolve_task("org/repo", "restoration", "a" * 40) == tmp_path / "hub/upscaler"
    assert [k["allow_patterns"] for k in downloads] == [["upscaler/*"], ["upscaler/*"]]
    assert resolve_task(str(tmp_path / "snapshot"), "restoration", None) == folder  # local export
