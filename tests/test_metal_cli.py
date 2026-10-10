import argparse
from pathlib import Path

import pytest
from PIL import Image

from iris_metal import cli


def test_task_outputs_never_share_a_name():
    inputs = [
        Path("a/photo.jpg"),
        Path("b/photo.jpg"),
        Path("photo.png"),
        Path("c/Photo.HEIC"),
        Path("x.jpg"),
    ]
    targets = cli.task_outputs(inputs, Path("depth"), "_depth")
    assert [t.name for t in targets] == [
        "photo_depth.png",
        "photo-2_depth.png",
        "photo-3_depth.png",
        "Photo-4_depth.png",  # APFS compares names case-insensitively
        "x_depth.png",
    ]
    assert all(t.parent == Path("depth") for t in targets)


def test_task_outputs_do_not_collide_with_a_suffixed_input_stem():
    targets = cli.task_outputs(
        [Path("photo-2.jpg"), Path("photo.jpg"), Path("b/photo.jpg")], Path("o"), "_x4"
    )
    assert len({t.name.casefold() for t in targets}) == 3


def test_unreadable_inputs_are_reported_before_any_model_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(cli, "load_task", lambda *a, **k: pytest.fail("the model must not load"))
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    good = tmp_path / "good.png"
    Image.new("RGB", (32, 16)).save(good)
    for command, path in (("depth", broken), ("restoration", broken), ("depth", tmp_path / "missing.png")):
        with pytest.raises(SystemExit):
            cli.task_main(command, [str(good), str(path)])
    thin = tmp_path / "thin.png"
    Image.new("RGB", (1, 64)).save(thin)
    with pytest.raises(SystemExit):
        cli.task_main("restoration", [str(thin)])


def test_bad_outputs_and_arguments_fail_before_generating(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(cli, "IrisPipeline", lambda *a, **k: pytest.fail("the model must not load"))
    for argv in (
        ["fox", "-o", str(tmp_path)],  # a directory
        ["fox", "-o", str(tmp_path / "out")],  # no image extension
        ["fox", "--steps", "0"],
        ["fox", "--cfg-scale", "nan"],
        ["fox", "--seed", str(2**64)],
    ):
        with pytest.raises(SystemExit):
            cli.main(argv)


def test_int_range():
    assert cli.int_range(1)("5") == 5
    for text in ("0", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.int_range(1)(text)
    with pytest.raises(argparse.ArgumentTypeError):
        cli.int_range(0, 10)("10")
