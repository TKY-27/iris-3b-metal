import pytest
import torch
from scripts.compare_runs import grid_score


def test_flat_images_have_neutral_grid_score():
    assert grid_score(torch.zeros(3, 32, 32)) == 1.0
    assert grid_score(torch.zeros(3, 8, 8)) == 1.0


def test_grid_score_uses_axes_with_boundaries():
    image = torch.arange(32).float()[None, None, :].expand(3, 8, 32)
    assert grid_score(image) == pytest.approx(1.0)


def test_grid_score_for_pure_patch_seam_is_finite():
    image = torch.zeros(3, 32, 32)
    image[:, :, 16:] = 1
    assert grid_score(image) > 1


def _run(path, ids, seed=0, size=(16, 16), unrecorded=(), *, checkpoint="model", prompts="set", **config):
    import json

    from PIL import Image

    path.mkdir()
    for image_id in (*ids, *unrecorded):
        Image.new("RGB", size).save(path / f"{image_id}.png")
    images = {i: {"prompt": i, "size": list(size), "seconds": 1.0} for i in ids}
    expected = {i: {"prompt": i, "size": list(size)} for i in ids}
    settings = {"split": "dev", "seed": seed, "scale": 1.0, "out": str(path), **config}
    provenance = {
        "checkpoint": {"config_sha256": checkpoint, "weights_sha256": checkpoint},
        "prompts_sha256": prompts,
        "source_sha256": {"src/a.py": "source"},
    }
    (path / "meta.json").write_text(
        json.dumps({"config": settings, "provenance": provenance, "expected": expected, "images": images})
    )
    return path


def test_whole_run_requires_the_complete_comparable_fixture_set(tmp_path):
    from scripts.compare_runs import mismatches

    reference = _run(tmp_path / "ref", ["a", "b"])
    assert mismatches(reference, _run(tmp_path / "same", ["a", "b"])) == []
    assert any("missing ['b']" in issue for issue in mismatches(reference, _run(tmp_path / "partial", ["a"])))
    assert any(
        "unexpected ['c']" in issue
        for issue in mismatches(reference, _run(tmp_path / "extra", ["a", "b", "c"]))
    )
    assert any(
        "seed" in issue for issue in mismatches(reference, _run(tmp_path / "seed", ["a", "b"], seed=1))
    )
    assert any(
        "size differs" in issue
        for issue in mismatches(reference, _run(tmp_path / "size", ["a", "b"], size=(32, 16)))
    )
    assert any(
        "alone" in issue
        for issue in mismatches(reference, _run(tmp_path / "loose", ["a", "b"], unrecorded=["z"]))
    )


def test_comparison_checks_identity_actual_pixels_and_declared_variants(tmp_path, monkeypatch):
    import json
    import sys

    from PIL import Image
    from scripts import compare_runs
    from scripts.compare_runs import mismatches

    reference = _run(tmp_path / "ref", ["a", "b"], steps=100)
    other = _run(tmp_path / "other", ["a", "b"], steps=50)
    assert any("steps" in issue for issue in mismatches(reference, other))
    assert mismatches(reference, other, {"steps"}) == []
    with pytest.raises(ValueError, match="unsupported experiment variables"):
        mismatches(reference, other, {"seed"})
    meta_path = other / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["provenance"]["source_sha256"]["src/a.py"] = "candidate-source"
    meta_path.write_text(json.dumps(meta))
    assert mismatches(reference, other, {"steps"}) == []
    report = tmp_path / "report.json"
    monkeypatch.setitem(sys.modules, "lpips", None)
    monkeypatch.setattr(
        sys, "argv", ["compare_runs", str(reference), str(other), "--vary", "steps", "--json", str(report)]
    )
    compare_runs.main()
    result = json.loads(report.read_text())
    assert result["complete_comparison"] is True
    assert result["variants"]["steps"] == {str(reference): 100, str(other): 50}
    assert result["provenance"][str(reference)]["source_sha256"]
    assert result["provenance"][str(other)]["source_sha256"] == {"src/a.py": "candidate-source"}
    meta["provenance"]["checkpoint"]["weights_sha256"] = "different"
    meta_path.write_text(json.dumps(meta))
    assert any("checkpoint provenance" in issue for issue in mismatches(reference, other, {"steps"}))
    meta["provenance"]["checkpoint"]["weights_sha256"] = "model"
    meta["provenance"]["prompts_sha256"] = "different"
    meta_path.write_text(json.dumps(meta))
    assert any("prompts_sha256 provenance" in issue for issue in mismatches(reference, other, {"steps"}))
    meta["provenance"]["prompts_sha256"] = "set"
    meta_path.write_text(json.dumps(meta))
    Image.new("RGB", (32, 16)).save(other / "a.png")
    assert any("a PNG dimensions differ" in issue for issue in mismatches(reference, other, {"steps"}))


def test_missing_identity_and_incomplete_reference_cannot_compare(tmp_path):
    import json

    from scripts.compare_runs import mismatches

    reference = _run(tmp_path / "ref", ["a", "b"])
    other = _run(tmp_path / "other", ["a", "b"])
    meta_path = reference / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["expected"]["c"] = {"prompt": "c", "size": [16, 16]}
    meta_path.write_text(json.dumps(meta))
    assert any("incomplete prompt set" in issue for issue in mismatches(reference, other))
    meta["expected"].pop("c")
    meta.pop("provenance")
    meta_path.write_text(json.dumps(meta))
    assert any(
        "missing checkpoint, prompt or source provenance" in issue for issue in mismatches(reference, other)
    )


def test_evaluate_refuses_changed_checkpoint_prompts_and_source_on_resume(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    from PIL import Image
    from scripts import evaluate

    prompts = tmp_path / "prompts.json"
    source = {"dev": [{"id": "a", "prompt": "alpha", "size": [16, 16]}], "holdout": []}
    prompts.write_text(json.dumps(source))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.yaml").write_text("model: fixture")
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"initial weights")
    out = tmp_path / "run"
    installed = tmp_path / "installed" / "iris3b"
    installed.mkdir(parents=True)
    installed_init = installed / "__init__.py"
    installed_init.write_text("# installed source")

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            self.cfg = SimpleNamespace(sample=SimpleNamespace(negative_prompt=""))

        def encode(self, texts):
            return texts

        @property
        def engine(self):
            return self

        def __call__(self, prompts, **kwargs):
            return [Image.new("RGB", (kwargs["width"], kwargs["height"]), "red")]

    monkeypatch.setattr(evaluate, "PROMPTS", prompts)
    monkeypatch.setattr(evaluate, "iris3b", SimpleNamespace(__file__=str(installed_init)))
    monkeypatch.setattr(evaluate, "resolve_checkpoint", lambda *_: checkpoint)
    monkeypatch.setattr(evaluate, "IrisPipeline", FakePipeline)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--out", str(out)])
    evaluate.main()
    saved = (out / "a.png").read_bytes()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["images"]["a"]["prompt"] == "alpha"
    assert meta["expected"] == {"a": {"prompt": "alpha", "size": [16, 16]}}
    assert meta["provenance"]["checkpoint"]["weights_sha256"] == evaluate.sha256(weights)
    assert meta["provenance"]["source_sha256"]["scripts/evaluate.py"]
    assert meta["provenance"]["source_sha256"][str(installed_init)] == evaluate.sha256(installed_init)

    weights.write_bytes(b"changed weights")
    with pytest.raises(SystemExit, match="different configuration or provenance"):
        evaluate.main()
    weights.write_bytes(b"initial weights")
    source["dev"][0]["prompt"] = "beta"
    prompts.write_text(json.dumps(source))
    with pytest.raises(SystemExit, match="different configuration or provenance"):
        evaluate.main()
    source["dev"][0]["prompt"] = "alpha"
    prompts.write_text(json.dumps(source))
    original_sha256 = evaluate.sha256
    monkeypatch.setattr(
        evaluate, "sha256", lambda path: "changed" if path.suffix == ".py" else original_sha256(path)
    )
    with pytest.raises(SystemExit, match="different configuration or provenance"):
        evaluate.main()
    meta.pop("provenance")
    (out / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="different configuration or provenance"):
        evaluate.main()
    assert (out / "a.png").read_bytes() == saved


def test_subset_comparison_is_reported_as_exploratory(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from PIL import Image
    from scripts import compare_runs

    monkeypatch.setitem(sys.modules, "lpips", None)  # fidelity metrics are not under test
    reference = _run(tmp_path / "ref", ["a", "b"])
    partial = _run(tmp_path / "partial", ["a"])
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["compare_runs", str(reference), str(partial), "--json", str(report)])
    with pytest.raises(SystemExit, match="missing"):
        compare_runs.main()
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--subset"])
    compare_runs.main()
    assert "EXPLORATORY" in capsys.readouterr().out
    result = json.loads(report.read_text())
    assert result["complete_comparison"] is False and result["prompts"] == ["a"]
    for run in (reference, partial):
        path = run / "meta.json"
        meta = json.loads(path.read_text())
        meta.pop("provenance")
        meta.pop("expected")
        path.write_text(json.dumps(meta))
    monkeypatch.setattr(sys, "argv", ["compare_runs", str(reference), str(partial)])
    with pytest.raises(SystemExit, match="missing checkpoint, prompt or source provenance"):
        compare_runs.main()
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--subset", "--json", str(report)])
    compare_runs.main()
    assert json.loads(report.read_text())["complete_comparison"] is False
    Image.new("RGB", (32, 16)).save(partial / "a.png")
    with pytest.raises(SystemExit, match="PNG dimensions differ from metadata"):
        compare_runs.main()


def test_precision_identity_hashes_weight_contents_and_loaded_sources(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace

    from scripts import precision_study

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.yaml").write_text("model: tiny")
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"first")
    installed = tmp_path / "installed" / "iris3b"
    installed.mkdir(parents=True)
    source = installed / "__init__.py"
    source.write_text("# installed")
    monkeypatch.setattr(precision_study, "iris3b", SimpleNamespace(__file__=str(source)))
    monkeypatch.setattr(precision_study.importlib.metadata, "version", lambda _: "test-version")
    first = precision_study.inputs_identity(checkpoint)
    assert first["source_sha256"][str(source)] == precision_study.sha256(source)
    stat = weights.stat()
    weights.write_bytes(b"other")
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert weights.stat().st_size == stat.st_size
    assert precision_study.inputs_identity(checkpoint)["checkpoint"] != first["checkpoint"]


def _precision_set(out):
    import json

    import numpy as np
    from scripts import precision_study

    out.mkdir()
    plan = {
        "inputs": {
            "checkpoint": {"config_sha256": "config", "weights_sha256": "weights"},
            "dependencies": {"torch": "test-version"},
            "source_sha256": {"src/fixture.py": "f" * 64},
        },
        "prompts_sha256": "prompts",
        "split": "dev",
        "fixtures": {
            "a": {"prompt": "alpha", "seed": 0, "size": 2},
            "b": {"prompt": "beta", "seed": 1, "size": 2},
        },
    }
    for key in plan["fixtures"]:
        np.savez(
            out / f"fixture-{key}.npz",
            noise=np.zeros((1, 3, 2, 2), dtype=np.float32),
            y=np.zeros((2, 1, 1), dtype=np.float32),
            mask=np.ones((2, 1), dtype=bool),
        )
    manifest = {
        "plan": plan,
        "files": {key: precision_study.sha256(out / f"fixture-{key}.npz") for key in plan["fixtures"]},
    }
    (out / "fixtures.json").write_text(json.dumps(manifest))
    for mode in ("fp32", "autocast"):
        source = {f"src/{mode}.py": ("a" if mode == "fp32" else "b") * 64}
        for key in plan["fixtures"]:
            np.savez(
                out / f"{mode}-{key}.npz",
                final=np.ones((1, 3, 2, 2), dtype=np.float32),
                provenance=json.dumps(precision_study.result_provenance(manifest, key, mode, 5, source)),
            )
    return manifest


def test_precision_score_rejects_missing_and_mismatched_results(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    import numpy as np
    import torch
    from scripts import precision_study

    out = tmp_path / "precision"
    manifest = _precision_set(out)
    args = SimpleNamespace(out=out, modes=["autocast"])
    candidate = out / "autocast-a.npz"

    class FakeLPIPS:
        def __init__(self, **_):
            pass

        def __call__(self, *_):
            return torch.tensor(0.0)

    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=FakeLPIPS))
    precision_study.score(args)
    with np.load(out / "fp32-a.npz") as reference, np.load(candidate) as result:
        assert (
            json.loads(str(reference["provenance"]))["source_sha256"]
            != json.loads(str(result["provenance"]))["source_sha256"]
        )

    fixture = out / "fixture-b.npz"
    saved_fixture = fixture.read_bytes()
    fixture.unlink()
    with pytest.raises(SystemExit, match="incomplete fixture set"):
        precision_study.score(args)
    fixture.write_bytes(saved_fixture)

    candidate.unlink()
    with pytest.raises(SystemExit, match="incomplete result set"):
        precision_study.score(args)

    source = {"src/autocast.py": "b" * 64}
    original = precision_study.result_provenance(manifest, "a", "autocast", 5, source)
    for field in ("config_sha256", "weights_sha256"):
        changed_manifest = json.loads(json.dumps(manifest))
        changed_manifest["plan"]["inputs"]["checkpoint"][field] = "other"
        assert precision_study.result_provenance(changed_manifest, "a", "autocast", 5, source) != original
    for field, value in (("plan_sha256", "other checkpoint"), ("mode", "engine"), ("steps", 4)):
        changed = {**original, field: value}
        np.savez(candidate, final=np.ones((1, 3, 2, 2)), provenance=json.dumps(changed))
        with pytest.raises(SystemExit, match="different or missing provenance"):
            precision_study.score(args)

    np.savez(candidate, final=np.full((1, 3, 2, 2), np.inf), provenance=json.dumps(original))
    with pytest.raises(SystemExit, match="nonfinite final values"):
        precision_study.score(args)

    np.savez(candidate, final=np.ones((1, 3, 1, 2)), provenance=json.dumps(original))
    with pytest.raises(SystemExit, match="unexpected final dimensions"):
        precision_study.score(args)

    np.savez(candidate, final=np.ones((1, 3, 2, 2)))
    with pytest.raises(SystemExit, match="missing steps provenance"):
        precision_study.score(args)

    np.savez(candidate, final=np.ones((1, 3, 2, 2)), provenance=json.dumps({**original, "source_sha256": {}}))
    with pytest.raises(SystemExit, match="missing source provenance"):
        precision_study.score(args)


def test_precision_run_refuses_changed_inputs_before_loading_model(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import precision_study

    out = tmp_path / "precision"
    manifest = _precision_set(out)
    monkeypatch.setattr(precision_study, "inputs_identity", lambda _: {"checkpoint": "other"})
    args = SimpleNamespace(out=out, checkpoint=tmp_path, mode="fp32", steps=5)
    with pytest.raises(SystemExit, match="another checkpoint or dependencies"):
        precision_study.denoise(args, None)
    current = {**manifest["plan"]["inputs"], "source_sha256": {"src/changed.py": "c" * 64}}
    monkeypatch.setattr(precision_study, "inputs_identity", lambda _: current)
    with pytest.raises(SystemExit, match="different or missing provenance"):
        precision_study.denoise(args, None)


def test_precision_run_records_new_mode_source_on_shared_fixtures(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    import numpy as np
    import torch
    from scripts import precision_study

    out = tmp_path / "precision"
    manifest = _precision_set(out)
    new_source = {"src/engine.py": "c" * 64}
    current = {**manifest["plan"]["inputs"], "source_sha256": new_source}
    monkeypatch.setattr(precision_study, "inputs_identity", lambda _: current)
    original_device = torch.device
    monkeypatch.setattr(precision_study.torch, "device", lambda _: original_device("cpu"))
    monkeypatch.setattr(precision_study.torch.mps, "synchronize", lambda: None)
    monkeypatch.setattr(precision_study, "load_dit", lambda *_: object())

    class FakeEngine:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode_text(self, *_):
            return torch.zeros((2, 1))

    class FakeSolver:
        def __init__(self, *_args, **_kwargs):
            pass

        def sample(self, noise, *_args, **_kwargs):
            return torch.ones_like(noise)

    class FakeLPIPS:
        def __init__(self, **_):
            pass

        def __call__(self, *_):
            return torch.tensor(0.0)

    monkeypatch.setattr(precision_study, "MetalIris", FakeEngine)
    monkeypatch.setattr(precision_study, "FlowDPMSolver", FakeSolver)
    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(LPIPS=FakeLPIPS))
    cfg = SimpleNamespace(
        sample=SimpleNamespace(cfg_scale=1, cfg_interval=[0, 1], order=1),
        flow=SimpleNamespace(num_train_timesteps=10, prediction="velocity", shift=1),
    )
    args = SimpleNamespace(out=out, checkpoint=tmp_path, mode="engine", steps=5)
    precision_study.denoise(args, cfg)
    with np.load(out / "engine-a.npz") as data:
        assert json.loads(str(data["provenance"]))["source_sha256"] == new_source
    precision_study.score(SimpleNamespace(out=out, modes=["engine"]))
    monkeypatch.setattr(
        precision_study, "inputs_identity", lambda _: {**current, "source_sha256": {"src/other.py": "d" * 64}}
    )
    with pytest.raises(SystemExit, match="different or missing provenance"):
        precision_study.denoise(args, cfg)
