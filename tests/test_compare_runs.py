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
