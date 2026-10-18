"""Local-first lookup for complete, immutable Hugging Face snapshots."""

import json
import re
from pathlib import Path

from huggingface_hub import try_to_load_from_cache


def cached_snapshot(
    repo: str, revision: str | None, *, language: bool = False, turbo: bool = False, task: str | None = None
) -> Path | None:
    # Branches and tags retain snapshot_download's freshness checks.
    if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        return None
    required = (
        ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"]
        if language
        else ["config.yaml", "model.safetensors"]
    )
    if turbo:
        required = ["turbo/config.yaml", "turbo/distill.yaml", "turbo/model.safetensors"]
    if task is not None:  # a depth or upscaler export
        required = [f"{task}/config.yaml", f"{task}/model.safetensors", f"{task}/empty_prompt.safetensors"]
    paths = [try_to_load_from_cache(repo, name, revision=revision) for name in required]
    try:
        if not all(
            isinstance(path, str) and Path(path).is_file() and Path(path).stat().st_size for path in paths
        ):
            return None
        root = Path(paths[0]).parent
        # A cache lookup for each file must identify the same snapshot.
        if any(Path(path).parent != root for path in paths[1:]):
            return None
    except OSError:
        return None
    if language:
        try:
            shards = set(json.loads((root / required[1]).read_text())["weight_map"].values())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None
        try:
            if not shards or not all(
                isinstance(shard, str)
                and shard == Path(shard).name
                and shard not in (".", "..")
                and "\\" not in shard
                and (root / shard).is_file()
                and (root / shard).stat().st_size
                for shard in shards
            ):
                return None
        except OSError:
            return None
    return root
