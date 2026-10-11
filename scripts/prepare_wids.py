"""Pack an image/caption corpus into tar shards indexed by ``wids-meta.json``.

Every sample becomes ``<key>.<ext>`` (the source image bytes, re-encoded as PNG
only when the format is not JPEG, PNG or WebP) plus ``<key>.json`` holding the
caption and the image ``height`` and ``width``; bucketed training reads the size
from the JSON without decoding the image. Keys are the zero-padded corpus
index, so shard contents are a pure function of the source order.

Sources:
  folder DIR   .jpg/.jpeg/.png/.webp files under DIR (recursive, sorted by
               path), each with a sidecar ``<stem>.json`` (copied into the
               sample's JSON) or ``<stem>.txt`` (stored as ``caption``); images
               without a sidecar are skipped
  hf REPO_ID   a Hugging Face datasets repo (requires ``pip install datasets``);
               --image-column and --caption-column name the columns, and the
               caption is stored as ``caption``

Usage:
    python scripts/prepare_wids.py folder /path/to/images --out /path/to/wids/dataset
    python scripts/prepare_wids.py hf user/dataset --split train \\
        --image-column image --caption-column text --out /path/to/wids/dataset

Train on the result with ``data.data_dirs=[/path/to/wids/dataset]``.
"""

import argparse
import io
import json
import os
import tarfile
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
FORMAT_EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}

_hf_source = None  # per-worker (dataset, image column, caption column), set by _init_hf


def _image_member(data: bytes) -> tuple[bytes, str, int, int]:
    """``(bytes, extension, height, width)`` of one image, re-encoded only if unsupported."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        ext = FORMAT_EXT.get(image.format)
        if ext is None:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue(), ".png", height, width
    return data, ext, height, width


def _add(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _write_shard(path: Path, samples: Iterable[tuple[str, bytes, dict]]) -> int:
    """Write ``(key, image bytes, info)`` samples to ``path``; returns the sample count."""
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with tarfile.open(tmp, "w") as tar:
        for key, data, info in samples:
            payload, ext, height, width = _image_member(data)
            _add(tar, key + ext, payload)
            meta = {**info, "height": height, "width": width}
            _add(tar, key + ".json", json.dumps(meta, ensure_ascii=False).encode())
            count += 1
    tmp.replace(path)
    return count


def _folder_items(root: Path) -> list[tuple[Path, Path]]:
    """``(image, sidecar)`` pairs under ``root`` in path order."""
    items = []
    for image in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()):
        sidecar = next(
            (s for s in (image.with_suffix(".json"), image.with_suffix(".txt")) if s.is_file()), None
        )
        if sidecar is not None:
            items.append((image, sidecar))
    return items


def _pack_folder_shard(task: tuple[str, int, list[tuple[Path, Path]]]) -> int:
    path, first, items = task

    def samples():
        for index, (image, sidecar) in enumerate(items, start=first):
            text = sidecar.read_text(encoding="utf-8")
            if sidecar.suffix == ".json":
                info = json.loads(text)
                if not isinstance(info, dict):
                    raise ValueError(f"{sidecar} must hold a JSON object")
            else:
                info = {"caption": text.strip()}
            yield f"{index:09d}", image.read_bytes(), info

    return _write_shard(Path(path), samples())


def _init_hf(repo: str, split: str, image_column: str, caption_column: str) -> None:
    global _hf_source
    from datasets import Image, load_dataset

    dataset = load_dataset(repo, split=split).cast_column(image_column, Image(decode=False))
    _hf_source = (dataset, image_column, caption_column)


def _pack_hf_shard(task: tuple[str, int, int]) -> int:
    path, start, stop = task
    dataset, image_column, caption_column = _hf_source

    def samples():
        for index in range(start, stop):
            row = dataset[index]
            image = row[image_column]
            data = image["bytes"] if image["bytes"] is not None else Path(image["path"]).read_bytes()
            yield f"{index:09d}", data, {"caption": str(row[caption_column] or "")}

    return _write_shard(Path(path), samples())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", choices=("folder", "hf"))
    parser.add_argument("location", help="image directory, or a Hugging Face datasets repo id")
    parser.add_argument("--out", required=True, help="output dataset directory")
    parser.add_argument("--split", default="train", help="hf: dataset split")
    parser.add_argument("--image-column", default="image", help="hf: image column")
    parser.add_argument("--caption-column", default="caption", help="hf: caption column")
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="one process per shard")
    args = parser.parse_args(argv)
    if args.samples_per_shard <= 0:
        parser.error("--samples-per-shard must be positive")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_shard = args.samples_per_shard

    def shard_path(shard: int) -> str:
        return str(out / f"shard-{shard:06d}.tar")

    if args.source == "folder":
        items = _folder_items(Path(args.location))
        if not items:
            parser.error(f"no image with a .json or .txt sidecar under {args.location}")
        tasks = [
            (shard_path(shard), first, items[first : first + per_shard])
            for shard, first in enumerate(range(0, len(items), per_shard))
        ]
        worker, initializer, initargs = _pack_folder_shard, None, ()
    else:
        from datasets import load_dataset

        # downloads and prepares the split once, before the workers load it from the cache
        total = len(load_dataset(args.location, split=args.split))
        tasks = [
            (shard_path(shard), first, min(first + per_shard, total))
            for shard, first in enumerate(range(0, total, per_shard))
        ]
        worker, initializer = _pack_hf_shard, _init_hf
        initargs = (args.location, args.split, args.image_column, args.caption_column)

    with ProcessPoolExecutor(args.workers, initializer=initializer, initargs=initargs) as pool:
        counts = list(pool.map(worker, tasks))
    shardlist = [{"url": Path(task[0]).name, "nsamples": n} for task, n in zip(tasks, counts, strict=True)]
    (out / "wids-meta.json").write_text(json.dumps({"name": out.name, "shardlist": shardlist}, indent=1))
    print(f"wrote {sum(counts)} samples in {len(shardlist)} shards to {out}")


if __name__ == "__main__":
    main()
