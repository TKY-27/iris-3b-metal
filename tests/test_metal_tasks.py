import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import save_file

from iris3b.config import Config
from iris3b.downstream.depth import IrisDepth
from iris3b.models.dit import IrisDiT
from iris_metal.tasks import MetalDepth, MetalRestorer, check_upscale_input
from test_metal_engine import RELEASED_TOPOLOGY


def _export(tmp_path, model, task: dict):
    torch.manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.05)
    config = OmegaConf.to_container(OmegaConf.structured(Config(model=RELEASED_TOPOLOGY)))
    (tmp_path / "config.yaml").write_text(OmegaConf.to_yaml({**config, "task": task}))
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, tmp_path / "model.safetensors")
    cfg = RELEASED_TOPOLOGY
    prompt = {
        "embeddings": torch.randn(1, cfg.text_len, cfg.text_lap_num_layers, cfg.text_dim),
        "mask": torch.ones(1, cfg.text_len, dtype=torch.bool),
    }
    save_file(prompt, tmp_path / "empty_prompt.safetensors")
    return prompt


def _relative_error(actual, expected):
    return ((actual - expected).norm() / expected.norm()).item()


def test_depth_matches_upstream_forward(tmp_path):
    model = IrisDepth(RELEASED_TOPOLOGY).eval()
    prompt = _export(tmp_path, model, {"name": "depth"})
    rgb = torch.rand(2, 3, 24, 32) * 2 - 1
    with torch.no_grad():
        expected = model(rgb, prompt["embeddings"], prompt["mask"])
        actual = MetalDepth(tmp_path, device="cpu").model(rgb, None, None)
    assert actual.shape == expected.shape == (2, 1, 24, 32)
    assert _relative_error(actual, expected) < 0.03  # 16-bit linears against the FP32 reference


def test_restorer_matches_upstream_forward(tmp_path):
    model = IrisDiT(RELEASED_TOPOLOGY).eval()
    prompt = _export(tmp_path, model, {"name": "restoration", "tile": 32, "sigma": 0.5})
    x = torch.rand(1, 3, 32, 32) * 2 - 1
    t = torch.full((1,), 500.0)
    with torch.no_grad():
        expected = x - 0.5 * model(x, t, prompt["embeddings"], y_mask=prompt["mask"]).x
        actual = MetalRestorer(tmp_path, device="cpu").restore_tile(x)
    assert _relative_error(actual, expected) < 0.01


def test_restorer_calls_on_tile_and_stops_when_it_raises(tmp_path):
    model = IrisDiT(RELEASED_TOPOLOGY).eval()
    _export(tmp_path, model, {"name": "restoration", "tile": 32, "sigma": 0.5})
    restorer = MetalRestorer(tmp_path, device="cpu")
    image = Image.new("RGB", (16, 12))  # 4x: 64 x 48, several overlapping 32-pixel tiles
    tiles = []
    restorer.on_tile = lambda: tiles.append(1)
    assert restorer(image, color_fix=False).size == (64, 48)
    assert len(tiles) > 1

    def stop():
        raise InterruptedError

    restorer.on_tile = stop
    with pytest.raises(InterruptedError):
        restorer(image)


def test_tile_count_matches_the_tiles_run(tmp_path):
    _export(tmp_path, IrisDiT(RELEASED_TOPOLOGY).eval(), {"name": "restoration", "tile": 32, "sigma": 0.5})
    restorer = MetalRestorer(tmp_path, device="cpu")
    for size in ((8, 8), (16, 12), (12, 16), (20, 9), (33, 17), (40, 40), (64, 16)):
        tiles = []
        restorer.restore_tile = lambda x, tiles=tiles: tiles.append(1) or x  # only the tiling is under test
        restorer(Image.new("RGB", size), color_fix=False)
        assert len(tiles) == restorer.tile_count(size), size


def test_upscale_inputs_beyond_four_to_one_are_rejected():
    check_upscale_input(Image.new("RGB", (1024, 256)))
    for size in ((1, 1024), (1025, 256)):
        with pytest.raises(ValueError, match="4:1"):
            check_upscale_input(Image.new("RGB", size))
