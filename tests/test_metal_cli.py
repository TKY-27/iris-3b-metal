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


def test_default_outputs_never_replace_an_earlier_image(tmp_path):
    (tmp_path / "iris.png").write_bytes(b"earlier")
    (tmp_path / "IRIS-3.png").write_bytes(b"earlier")
    names = [p.name for p in cli.output_paths(tmp_path / "iris.png", 3, overwrite=False)]
    assert names == ["iris-2.png", "iris-4.png", "iris-5.png"]


def test_an_explicit_output_is_used_as_given(tmp_path):
    (tmp_path / "fox.png").write_bytes(b"earlier")
    assert cli.output_paths(tmp_path / "fox.png", 1, overwrite=True) == [tmp_path / "fox.png"]
    names = [p.name for p in cli.output_paths(tmp_path / "fox.png", 3, overwrite=True)]
    assert names == ["fox.png", "fox-2.png", "fox-3.png"]


def test_a_name_taken_while_generating_moves_to_the_next(tmp_path):
    planned = cli.output_paths(tmp_path / "iris.png", 1, overwrite=False)[0]
    planned.write_bytes(b"another run")  # took the name during the minutes of generation
    saved = cli.save_new(planned, lambda f: f.write(b"this run"), cli.numbered(tmp_path / "iris.png"))
    assert saved.name == "iris-2.png"
    assert planned.read_bytes() == b"another run"
    assert saved.read_bytes() == b"this run"


def test_a_dangling_symlink_takes_its_name(tmp_path):
    (tmp_path / "iris.png").symlink_to(tmp_path / "missing.png")
    saved = cli.save_new(tmp_path / "iris.png", lambda f: f.write(b"x"), cli.numbered(tmp_path / "iris.png"))
    assert saved.name == "iris-2.png"
    assert cli.output_paths(tmp_path / "iris.png", 1, overwrite=False)[0].name == "iris-3.png"


def test_task_outputs_skip_earlier_results_and_their_siblings(tmp_path):
    (tmp_path / "photo_depth.npy").write_bytes(b"earlier")  # its PNG was deleted; the array is kept
    (tmp_path / "photo-2_depth.png").write_bytes(b"earlier")
    targets = cli.task_outputs([Path("photo.jpg"), Path("b/photo.jpg")], tmp_path, "_depth", (".npy",))
    assert [t.name for t in targets] == ["photo-3_depth.png", "photo-4_depth.png"]


def test_outputs_in_formats_pillow_cannot_write_are_refused(tmp_path):
    parser = argparse.ArgumentParser()
    cli.check_output(parser, tmp_path / "fox.webp")
    for name in ("fox.psd", "fox.xbm", "fox"):
        with pytest.raises(SystemExit):
            cli.check_output(parser, tmp_path / name)


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
