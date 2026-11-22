import pytest
import torch

from iris3b.nn.embeddings import _sincos_1d, sincos_pos_embed_2d


@pytest.mark.parametrize("embed_dim", [16, 64, 1152])
@pytest.mark.parametrize("height,width", [(3, 5), (5, 3), (1, 1), (0, 3), (3, 0)])
def test_sincos_pos_embed_matches_full_grid(embed_dim, height, width):
    grid = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing="xy",
    )
    expected = torch.cat([_sincos_1d(embed_dim // 2, axis) for axis in grid], dim=1).float()
    actual = sincos_pos_embed_2d(embed_dim, height, width)
    assert actual.shape == (height * width, embed_dim)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_sincos_pos_embed_rejects_negative_size():
    with pytest.raises(RuntimeError):
        sincos_pos_embed_2d(16, -1, 3)
