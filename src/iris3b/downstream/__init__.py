"""Iris-3B fine-tuned for downstream tasks: monocular depth and image restoration.

Both tasks run the full Iris-3B transformer in a single forward pass, conditioned
on the embedding of the empty prompt, so no text encoder is loaded. A task
export is a directory holding the files below; the release ships them in the
``depth/`` and ``upscaler/`` folders of the ``speridlabs/iris-3b`` Hub repo.

    config.yaml               model/flow sections of the parent plus a ``task`` section
    model.safetensors         FP32 weights of the fine-tuned model
    empty_prompt.safetensors  ``embeddings`` [1, T, L, D] FP32 and ``mask`` [1, T] bool
"""

import os
from pathlib import Path

import torch

from iris3b.config import inference_config


def load_export(source: str | Path, task: str) -> tuple[object, dict, dict[str, torch.Tensor], dict]:
    """``(config, task settings, weights, empty prompt)`` from a local directory or a
    Hub folder ``owner/repo/subfolder`` (e.g. ``speridlabs/iris-3b/depth``)."""
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    path = Path(source)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        owner, repo, sub = str(source).split("/", 2)
        root = snapshot_download(
            f"{owner}/{repo}", allow_patterns=f"{sub}/*", token=os.environ.get("HF_TOKEN")
        )
        path = Path(root) / sub
    raw = OmegaConf.to_container(OmegaConf.load(path / "config.yaml"))
    settings = raw.pop("task")
    if settings["name"] != task:
        raise ValueError(f"{path} holds a {settings['name']!r} export, not {task!r}")
    prompt = load_file(path / "empty_prompt.safetensors")
    return inference_config(raw, []), settings, load_file(path / "model.safetensors"), prompt
