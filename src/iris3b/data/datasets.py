"""Pixel-space image/caption datasets over indexed tar shards.

Every sample is a dict ``{image, caption, index, img_hw, aspect_ratio}`` where
image is float32 CHW in ``[-1, 1]``, index is the absolute dataset index (the
identity the frozen validation grid keys on, invariant to any batching policy),
img_hw is the original (h, w) as int64, and aspect_ratio is the realized
height/width of the training shape.
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from iris3b.config import DataConfig
from iris3b.data.shapes import Shape, build_shape_policy
from iris3b.data.wids import IndexedTarDataset, MissingShardError
from iris3b.registry import DATASETS
from iris3b.seeding import mix_seed

_RETRIES = 10


def _to_tensor(img) -> torch.Tensor:
    """PIL RGB image to float32 CHW in [-1, 1]."""
    arr = np.array(img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
    return t.div_(127.5).sub_(1.0)


def _center_crop(img, ch: int, cw: int):
    w, h = img.size
    top = int(round((h - ch) / 2.0))
    left = int(round((w - cw) / 2.0))
    return img.crop((left, top, left + cw, top + ch))


def _resize_shortest(img, target: int):
    """Resize so the shortest side equals target, bicubic, aspect preserved."""
    from PIL import Image

    w, h = img.size
    if w <= h:
        size = (target, int(target * h / w))
    else:
        size = (int(target * w / h), target)
    return img.resize(size, Image.Resampling.BICUBIC)


def _resize_covering(img, oh: int, ow: int, bh: int, bw: int):
    """Aspect-preserving bicubic resize sized so a (bh, bw) center crop is covered."""
    from PIL import Image

    if bh / oh > bw / ow:
        rh, rw = bh, round(ow * bh / oh)
    else:
        rh, rw = round(oh * bw / ow), bw
    return img.resize((rw, rh), Image.Resampling.BICUBIC)


def select_caption(info: dict, cfg: DataConfig, rng: random.Random) -> str:
    """Pick a caption for one sample.

    With ``cfg.caption_fields`` set, sample uniformly among those keys the
    sample carries (``cfg.caption_field`` when it carries none). Otherwise the
    caption is ``info[cfg.caption_field]``.

    Draws come from ``rng`` so the choice is a pure function of the caller's
    (seed, epoch, index) key: resume-invariant, resampled every epoch.
    """
    if cfg.caption_fields:
        present = [f for f in cfg.caption_fields if info.get(f)]
        if present:
            return str(info[rng.choice(present)])
        return str(info.get(cfg.caption_field) or "")
    text = info[cfg.caption_field]
    return "" if text is None else str(text)


class _TarImageDataset(Dataset):
    """Shared tar-backed decoding, caption selection, and retry plumbing."""

    def __init__(self, cfg: DataConfig, patch_size: int = 16):
        cfg.validate(patch_size)
        self.cfg = cfg
        self.seed = 0  # overwritten by build_dataloader with train.seed
        self.epoch = 0  # overwritten by the trainer each epoch
        self.reader = IndexedTarDataset(cfg.data_dirs)
        self.policy = build_shape_policy(
            cfg.resolved_shape_policy(),
            cfg.image_size,
            patch_size,
            cfg.aspect_ratio_bucket,
            cfg.shape_align,
            cfg.shape_max_ratio,
        )

    def __len__(self) -> int:
        return len(self.reader)

    def _image(self, sample: dict):
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            if ext in sample:
                return sample[ext].convert("RGB")
        raise KeyError(f"sample {sample.get('__key__')} has no image entry")

    def _caption_rng(self, idx: int) -> random.Random:
        return random.Random(mix_seed(self.seed, self.epoch, idx))

    def _get(self, idx: int) -> dict:
        raise NotImplementedError

    def _retry_index(self, idx: int) -> int:
        return (idx + 1) % len(self)

    def __getitem__(self, idx: int) -> dict:
        err: Exception | None = None
        for _ in range(_RETRIES):
            try:
                return self._get(idx)
            except MissingShardError:
                raise
            except Exception as e:  # noqa: BLE001 - any bad sample triggers a retry
                err = e
                idx = self._retry_index(idx)
        raise RuntimeError(f"too many undecodable samples near index {idx}") from err


@DATASETS.register("pixel")
@DATASETS.register("pixel_multiscale")
class PixelDataset(_TarImageDataset):
    """Resolution is whatever ``data.shape_policy`` resolves for the sample.

    A uniform policy resizes the shortest side to the target (BICUBIC) and
    center-crops the square. Any shape-varying policy resizes so the target
    rectangle is covered (BICUBIC) and center-crops it; a sample already at its
    target size is used as is.

    Bucketed batching keys samples by their ``.json`` ``height`` and ``width``,
    so every record needs both fields.
    """

    def key_of(self, idx: int) -> str:
        """Grouping key from the sample's metadata; the image is not decoded."""
        if self.policy.uniform:
            return self.policy.key(0, 0)
        info = self.reader.info(idx)
        return self.policy.key(int(info["height"]), int(info["width"]))

    def _retry_index(self, idx: int) -> int:
        """Substitute a sample of the SAME key, so a batch stays homogeneous."""
        if self.policy.uniform:
            return (idx + 1) % len(self)
        key = self.key_of(idx)
        for offset in range(1, len(self)):
            candidate = (idx + offset) % len(self)
            try:
                if self.key_of(candidate) == key:
                    return candidate
            except (KeyError, TypeError, ValueError):
                continue
        raise RuntimeError(f"no replacement sample for shape {key} near index {idx}")

    def _get(self, idx: int) -> dict:
        sample = self.reader[idx]
        info = sample[".json"]
        img = self._image(sample)
        oh = int(info.get("height", img.height))
        ow = int(info.get("width", img.width))
        caption = select_caption(info, self.cfg, self._caption_rng(idx))
        target = self.policy.target(oh, ow)
        if self.policy.uniform:
            size = target.height
            img = _center_crop(_resize_shortest(img, size), size, size)
        elif (oh, ow) != (target.height, target.width):
            img = _center_crop(
                _resize_covering(img, oh, ow, target.height, target.width),
                target.height,
                target.width,
            )
        return {
            "image": _to_tensor(img),
            "caption": caption,
            "index": idx,
            "img_hw": torch.tensor([oh, ow], dtype=torch.int64),
            "aspect_ratio": target.height / target.width,
        }


_WORDS = ("amber", "canyon", "drift", "ember", "harbor", "lattice", "meadow", "orbit", "quartz", "willow")


class SyntheticDataset(Dataset):
    """Deterministic random images and captions for smoke tests; data_dirs is ignored."""

    LENGTH = 25_600  # ~100 steps/epoch at global batch 256; keeps epoch churn out of s/it

    #: native sizes cycled through so a shape-varying policy sees real spread
    _NATIVE = ((512, 512), (384, 768), (768, 384), (640, 480), (480, 640), (256, 1024))

    def __init__(self, cfg: DataConfig, patch_size: int = 16):
        self.cfg = cfg
        self.policy = build_shape_policy(
            cfg.resolved_shape_policy(),
            cfg.image_size,
            patch_size,
            cfg.aspect_ratio_bucket,
            cfg.shape_align,
            cfg.shape_max_ratio,
        )

    def __len__(self) -> int:
        return self.LENGTH

    def _native(self, idx: int) -> tuple[int, int]:
        return self._NATIVE[idx % len(self._NATIVE)]

    def _target(self, idx: int) -> Shape:
        return self.policy.target(*self._native(idx))

    def key_of(self, idx: int) -> str:
        return self._target(idx).key

    def __getitem__(self, idx: int) -> dict:
        oh, ow = self._native(idx)
        target = self._target(idx)
        gen = torch.Generator().manual_seed(idx)
        image = torch.rand((3, target.height, target.width), generator=gen) * 2.0 - 1.0
        rng = random.Random(idx)
        caption = f"a {rng.choice(_WORDS)} {rng.choice(_WORDS)} scene {idx}"
        return {
            "image": image,
            "caption": caption,
            "index": idx,
            "img_hw": torch.tensor([oh, ow], dtype=torch.int64),
            "aspect_ratio": target.height / target.width,
        }


@DATASETS.register("synthetic")
@DATASETS.register("synthetic_multiscale")
def _synthetic(cfg: DataConfig, patch_size: int = 16) -> SyntheticDataset:
    return SyntheticDataset(cfg, patch_size)
