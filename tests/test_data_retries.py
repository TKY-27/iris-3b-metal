from types import SimpleNamespace

import pytest

from iris3b.data.datasets import PixelDataset


class _Reader:
    def __init__(self, sizes):
        self.sizes = sizes

    def __len__(self):
        return len(self.sizes)

    def info(self, index):
        height, width = self.sizes[index]
        return {"height": height, "width": width}


def test_corrupt_bucket_sample_retries_with_same_shape():
    dataset = PixelDataset.__new__(PixelDataset)
    dataset.reader = _Reader([(32, 32), (32, 64), (32, 32)])
    dataset.policy = SimpleNamespace(uniform=False, key=lambda h, w: f"{h}x{w}")

    def get(index):
        if index == 0:
            raise ValueError("corrupt image")
        return index

    dataset._get = get
    assert dataset[0] == 2


def test_corrupt_bucket_with_no_same_shape_fails():
    dataset = PixelDataset.__new__(PixelDataset)
    dataset.reader = _Reader([(32, 32), (32, 64)])
    dataset.policy = SimpleNamespace(uniform=False, key=lambda h, w: f"{h}x{w}")
    with pytest.raises(RuntimeError, match="no replacement sample for shape 32x32"):
        dataset._retry_index(0)
