"""Turbo's fixed official recipe, independent trajectory and API lifecycle checks."""

import io
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from iris3b.sampling import generate_few_step
from iris_metal.pipeline import IrisPipeline
from iris_metal.server import Job, Models, png, text_to_image_job, validate
from iris_metal.turbo import TURBO_SIGMAS, attach_metadata, load_sigmas
from test_metal_pipeline import _pipeline


def write_distill(root):
    (root / "distill.yaml").write_text(
        "dmd:\n  sigmas: [1.0, 0.938, 0.833, 0.625]\n  deterministic: true\n  method: tdm\n"
    )


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.cfg = SimpleNamespace(in_channels=3)
        self.calls = []

    def forward(self, x, t, y, y_mask):
        self.calls.append((x.clone(), t.clone(), y.clone(), y_mask.clone()))
        return SimpleNamespace(x=x * self.weight + y.reshape(-1, 1, 1, 1) * 0.01)


class ToyEncoder:
    def encode(self, prompts):
        return SimpleNamespace(
            embeddings=torch.tensor([len(p) for p in prompts], dtype=torch.float32),
            mask=torch.ones(len(prompts), 1),
        )


def test_official_sampler_matches_independent_flow_euler_and_calls_once_per_sigma():
    noise = torch.randn(2, 3, 16, 32, generator=torch.Generator().manual_seed(17))
    model = ToyModel()
    actual = generate_few_step(
        model, ToyEncoder(), ["abc", "d"], 16, 32, list(TURBO_SIGMAS), noise=noise, device="cpu"
    )
    # v-prediction Euler update is x_next = x + (s_next - s) * v.
    expected = noise.clone()
    for sigma, next_sigma in zip(TURBO_SIGMAS, (*TURBO_SIGMAS[1:], 0.0), strict=True):
        velocity = expected * 0.25 + torch.tensor([3.0, 1.0])[:, None, None, None] * 0.01
        expected += (next_sigma - sigma) * velocity
    torch.testing.assert_close(actual, expected.clamp(-1, 1), atol=2e-7, rtol=1e-6)
    assert len(model.calls) == 4
    assert [call[1][0].item() for call in model.calls] == [1000.0, 938.0, 833.0, 625.0]
    assert all(call[0].shape[0] == 2 for call in model.calls)
    assert torch.equal(noise, model.calls[0][0])


def test_distill_recipe_rejects_modified_schedule_or_stochastic_mode(tmp_path):
    assert load_sigmas(tmp_path) is None
    write_distill(tmp_path)
    assert load_sigmas(tmp_path) == TURBO_SIGMAS
    original = (tmp_path / "distill.yaml").read_text()
    for modified in (
        original.replace("0.938", "0.9"),
        original.replace("true", "false"),
        original.replace("tdm", "dmd2"),
        original.replace("1.0", "true"),
    ):
        (tmp_path / "distill.yaml").write_text(modified)
        with pytest.raises(ValueError, match="unsupported Turbo"):
            load_sigmas(tmp_path)


def test_turbo_pipeline_uses_only_positive_prompt_and_reports_four_steps(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path, monkeypatch)
    write_distill(tmp_path)
    pipe = IrisPipeline(tmp_path, device="cpu")
    pipe._engine = ToyModel()
    steps = []
    image = pipe(["cat"], height=16, width=32, seed=9, on_step=lambda *s: steps.append(s))[0]
    assert steps == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert image.size == (32, 16)
    assert list(pipe._encodings) == ["cat"]
    assert pipe.last_metadata["sigmas"] == list(TURBO_SIGMAS)
    assert pipe.last_metadata["hf_revision"] is None
    with pytest.raises(ValueError, match="fixed four-step"):
        pipe(["cat"], steps=4)
    pipe.unload()
    assert not pipe.loaded


def test_turbo_prepares_text_once_per_request_and_recovers_after_cancel(tmp_path, monkeypatch):
    from test_metal_engine import _inputs, _randomized_model

    _pipeline(tmp_path, monkeypatch)
    write_distill(tmp_path)
    pipe = IrisPipeline(tmp_path, device="cpu")
    model = pipe._engine = _randomized_model()
    _, _, states, mask = _inputs(batch=1)
    pipe._encodings["cat"] = (states, mask)
    calls = []
    handle = model.y_embedder.register_forward_hook(lambda *_: calls.append(1))
    try:
        expected = pipe(["cat"], height=16, width=32, seed=5)[0].tobytes()
        assert len(calls) == 1

        def cancel(*_):
            raise RuntimeError("cancelled")

        with pytest.raises(RuntimeError, match="cancelled"):
            pipe(["cat"], height=16, width=32, seed=5, on_step=cancel)
        assert len(calls) == 2
        actual = pipe(["cat"], height=16, width=32, seed=5)[0].tobytes()
        assert len(calls) == 3
        assert actual == expected
    finally:
        handle.remove()


@pytest.mark.parametrize("control", ["steps", "cfg_scale", "negative_prompt", "order", "shift"])
def test_turbo_api_rejects_controls_instead_of_silently_ignoring(control):
    defaults = SimpleNamespace(steps=100, cfg_scale=3.0, negative_prompt="")
    with pytest.raises(ValueError, match="does not apply"):
        validate({"prompt": "cat", "model": "turbo", control: 4}, defaults)
    params = validate({"prompt": "cat", "model": "turbo"}, defaults)
    assert (params["steps"], params["cfg_scale"], params["negative_prompt"]) == (4, 1.0, "")


def test_model_switches_unload_before_load_and_preserve_tasks():
    log = []

    class Pipe:
        cfg = SimpleNamespace(sample=SimpleNamespace(steps=100, cfg_scale=3.0, negative_prompt=""))

        def __init__(self, model):
            self.model, self.loaded = model, True

        def unload(self):
            log.append(("unload", self.model))
            self.loaded = False

    original = Pipe("original")

    def load(model):
        assert not original.loaded
        log.append(("load", model))
        return Pipe(model)

    models = Models(original, lambda task: object(), load)
    turbo = models.text_to_image("turbo")
    assert models.loaded_model == "turbo"
    models.task("depth")
    assert not turbo.loaded and models.loaded_model is None
    assert models.resident == "depth"
    assert models.text_to_image() is original
    assert log == [("unload", "original"), ("load", "turbo"), ("unload", "turbo"), ("unload", "turbo")]
    assert models.text_to_image("turbo") is turbo
    assert not original.loaded and not turbo.loaded
    assert sum(event == ("load", "turbo") for event in log) == 1


def test_png_and_job_preserve_reproduction_metadata():
    metadata = {"model": "turbo", "sigmas": list(TURBO_SIGMAS), "seed": 3}
    image = Image.new("RGB", (16, 16))
    attach_metadata([image], metadata)
    assert json.loads(Image.open(io.BytesIO(png(image))).info["iris"]) == metadata

    class Pipe:
        model = "turbo"
        loaded = True
        cfg = SimpleNamespace(sample=SimpleNamespace(steps=100, cfg_scale=3.0, negative_prompt=""))
        last_metadata = metadata

        def __call__(self, prompts, on_step, **kwargs):
            assert "steps" not in kwargs and "cfg_scale" not in kwargs and "negative_prompt" not in kwargs
            for step in range(1, 5):
                on_step(step, 4)
            return [image]

    models = Models(Pipe(), lambda task: None)
    params = {"prompt": "cat", "model": "turbo", "steps": 4, "cfg_scale": 1.0, "negative_prompt": ""}
    job = Job("1", "image", params, None)
    text_to_image_job(models, params)(job)
    assert job.summary()["metadata"] == metadata


def test_turbo_cancelled_step_does_not_poison_next_request(tmp_path, monkeypatch):
    _pipeline(tmp_path, monkeypatch)
    write_distill(tmp_path)
    pipe = IrisPipeline(tmp_path, device="cpu")
    pipe._engine = ToyModel()

    def cancel(step, total):
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        pipe(["cat"], height=16, width=16, seed=5, on_step=cancel)
    assert pipe.last_metadata is None
    first = pipe(["cat"], height=16, width=16, seed=5)[0]
    second = pipe(["cat"], height=16, width=16, seed=5)[0]
    assert first.tobytes() == second.tobytes()


def test_turbo_loader_preserves_bf16_linear_storage(tmp_path, monkeypatch):
    import iris_metal.pipeline as module

    _pipeline(tmp_path, monkeypatch)
    write_distill(tmp_path)
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs["dtype"])
        return ToyModel()

    monkeypatch.setattr(module, "load_dit", load)
    pipe = IrisPipeline(tmp_path, device="cpu")
    assert pipe.engine is pipe.engine
    assert calls == [torch.bfloat16]


def test_turbo_download_is_selective_and_pinned(tmp_path, monkeypatch):
    import huggingface_hub

    from iris_metal import cache
    from iris_metal.cli import REPO_ID, resolve_checkpoint
    from iris_metal.turbo import TURBO_REVISION

    monkeypatch.setattr(cache, "cached_snapshot", lambda *a, **k: None)
    monkeypatch.setattr("iris_metal.cli.checkpoint_revision", lambda path: None)
    calls = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert resolve_checkpoint(REPO_ID, None, "turbo") == tmp_path / "turbo"
    assert calls == [
        (
            (REPO_ID,),
            {
                "revision": TURBO_REVISION,
                "allow_patterns": ["turbo/config.yaml", "turbo/distill.yaml", "turbo/model.safetensors"],
            },
        )
    ]
    (tmp_path / "turbo").mkdir()
    write_distill(tmp_path / "turbo")
    assert resolve_checkpoint(str(tmp_path), None, "turbo") == tmp_path / "turbo"
    assert resolve_checkpoint(str(tmp_path / "turbo"), None, "turbo") == tmp_path / "turbo"


def test_turbo_local_cache_requires_distillation_metadata(tmp_path, monkeypatch):
    from iris_metal import cache

    root = tmp_path / "turbo"
    root.mkdir()
    for name in ("config.yaml", "model.safetensors"):
        (root / name).write_text("fixture")
    monkeypatch.setattr(cache, "try_to_load_from_cache", lambda repo, name, revision: str(tmp_path / name))
    assert cache.cached_snapshot("repo", "a" * 40, turbo=True) is None
    write_distill(root)
    assert cache.cached_snapshot("repo", "a" * 40, turbo=True) == root


def test_bf16_linear_storage_is_exactly_upcast_then_autocast(tmp_path):
    from safetensors.torch import save_file

    from iris3b.nn.norms import RMSNorm
    from iris_metal.loading import load_weights

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 16)
            self.norm = RMSNorm(16)

        def forward(self, x):
            return self.norm(self.linear(x))

    torch.manual_seed(44)
    source = Net()
    weights = {name: t.to(torch.bfloat16) for name, t in source.state_dict().items()}
    file = tmp_path / "model.safetensors"
    save_file(weights, str(file))
    with torch.device("meta"):
        reference, reduced = Net(), Net()
    load_weights(reference, file, torch.device("cpu"), torch.float32)
    load_weights(reduced, file, torch.device("cpu"), torch.bfloat16)
    x = torch.randn(7, 16)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(reference(x), reduced(x))
    assert reference.linear.weight.dtype == torch.float32
    assert reduced.linear.weight.dtype == torch.bfloat16
    assert reduced.norm.weight.dtype == torch.float32


def test_original_selection_refuses_a_turbo_checkpoint(tmp_path):
    from iris_metal.cli import resolve_checkpoint

    write_distill(tmp_path)
    with pytest.raises(ValueError, match="Turbo checkpoint"):
        resolve_checkpoint(str(tmp_path), None, "original")


def test_turbo_selection_refuses_missing_recipe(tmp_path):
    from iris_metal.cli import resolve_checkpoint

    (tmp_path / "turbo").mkdir()
    (tmp_path / "turbo/config.yaml").write_text("model: {}")
    with pytest.raises(ValueError, match="incomplete"):
        resolve_checkpoint(str(tmp_path), None, "turbo")


def test_official_turbo_reuses_complete_local_download_before_hub(tmp_path, monkeypatch):
    import huggingface_hub

    from iris_metal import cli
    from iris_metal.turbo import TURBO_REVISION

    monkeypatch.setattr(cli, "__file__", str(tmp_path / "src/iris_metal/cli.py"))
    root = tmp_path / "iris-3b/turbo"
    root.mkdir(parents=True)
    write_distill(root)
    (root / "config.yaml").write_text("model: {}")
    (root / "model.safetensors").write_bytes(b"fixture")
    meta = tmp_path / "iris-3b/.cache/huggingface/download/turbo"
    meta.mkdir(parents=True)
    (meta / "model.safetensors.metadata").write_text(TURBO_REVISION + "\n" + "a" * 64 + "\n1\n")
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("no duplicate download")
    )
    assert cli.resolve_checkpoint(cli.REPO_ID, None, "turbo") == root


def test_http_startup_and_turbo_first_request_never_resolve_original(tmp_path, monkeypatch):
    import iris_metal.server as server_module
    from iris_metal import cli

    monkeypatch.setattr(cli.torch.backends.mps, "is_available", lambda: True)
    calls = []

    def resolve(name, revision, model="original"):
        calls.append(model)
        return tmp_path

    monkeypatch.setattr(cli, "resolve_checkpoint", resolve)

    class Pipe:
        model = "turbo"
        loaded = False
        cfg = SimpleNamespace(sample=SimpleNamespace(steps=100, cfg_scale=3.0, negative_prompt=""))

    monkeypatch.setattr(cli, "IrisPipeline", lambda checkpoint: Pipe())
    captured = []
    monkeypatch.setattr(server_module, "serve", lambda models, *args: captured.append(models))
    cli.serve_main([])
    assert calls == []
    models = captured[0]
    assert models.resident is None and models.loaded_model is None
    assert models.defaults.steps == 100
    assert models.text_to_image("turbo").model == "turbo"
    assert calls == ["turbo"]
    assert "original" not in models._generation
