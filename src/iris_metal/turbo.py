"""Pinned official Turbo recipe and deterministic flow-Euler sampling."""

import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
from PIL.PngImagePlugin import PngInfo

TURBO_REVISION = "adc05b909fb05d4276d26aae5a52b36406b2ef26"
UPSTREAM_REVISION = "cee0c130a62a9c288ad8918f1f7cdf23cc228d69"
TURBO_SIGMAS = (1.0, 0.938, 0.833, 0.625)
PNG_METADATA_KEY = "iris"


def load_sigmas(checkpoint: Path) -> tuple[float, ...] | None:
    path = checkpoint / "distill.yaml"
    if not path.exists():
        return None
    recipe = OmegaConf.to_container(OmegaConf.load(path))
    dmd = recipe.get("dmd", {}) if isinstance(recipe, dict) else {}
    if not isinstance(dmd, dict):
        raise ValueError("unsupported Turbo distill.yaml: dmd must describe the official recipe")
    sigmas = dmd.get("sigmas")
    if (
        not isinstance(sigmas, list)
        or any(isinstance(s, bool) or not isinstance(s, (int, float)) for s in sigmas)
        or tuple(sigmas) != TURBO_SIGMAS
        or dmd.get("deterministic") is not True
        or dmd.get("method") != "tdm"
    ):
        raise ValueError(
            "unsupported Turbo distill.yaml: expected official deterministic four-step TDM recipe"
        )
    return tuple(float(s) for s in sigmas)


def recipe_hash(checkpoint: Path, name: str) -> str:
    return hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()


def png_info(image) -> PngInfo:
    info = PngInfo()
    if PNG_METADATA_KEY in image.info:
        info.add_text(PNG_METADATA_KEY, image.info[PNG_METADATA_KEY])
    return info


def attach_metadata(images, metadata: dict) -> None:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    for image in images:
        image.info[PNG_METADATA_KEY] = encoded


def checkpoint_revision(checkpoint: Path) -> str | None:
    # Hub cache paths and local-dir download metadata both record an immutable
    # revision. Do not attribute arbitrary user exports to the official Hub pin.
    import re

    root = checkpoint.parent if checkpoint.name == "turbo" else checkpoint
    if root.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", root.name):
        return root.name
    name = "turbo/model.safetensors.metadata" if checkpoint.name == "turbo" else "model.safetensors.metadata"
    path = root / ".cache/huggingface/download" / name
    if path.is_file():
        lines = path.read_text().splitlines()
        revision = lines[0] if lines else ""
        if re.fullmatch(r"[0-9a-f]{40}", revision):
            return revision
    return None


def checkpoint_sha256(checkpoint: Path) -> str | None:
    """Immutable Hub blob digest, when recorded by its downloader."""
    import re

    root = checkpoint.parent if checkpoint.name == "turbo" else checkpoint
    if root.parent.name == "snapshots":
        blob = (checkpoint / "model.safetensors").resolve().name
        return blob if re.fullmatch(r"[0-9a-f]{64}", blob) else None
    name = "turbo/model.safetensors.metadata" if checkpoint.name == "turbo" else "model.safetensors.metadata"
    path = root / ".cache/huggingface/download" / name
    if path.is_file():
        lines = path.read_text().splitlines()
        if len(lines) > 1 and re.fullmatch(r"[0-9a-f]{64}", lines[1]):
            return lines[1]
    return None
