"""Export a fine-tuned depth or restoration checkpoint as a task release directory.

Usage:
    python scripts/export_downstream.py depth step-010000.pth exported/iris-3b/depth \
        --parent-config exported/iris-3b/config.yaml
    python scripts/export_downstream.py restoration inference.pth exported/iris-3b/upscaler \
        --parent-config exported/iris-3b/config.yaml --conditioning empty-conditioning.pt

``--parent-config`` is the ``config.yaml`` of the Iris-3B release; it fixes the
architecture, and the weights are checked against it with a strict load. The
depth checkpoint carries its own empty-prompt embedding; the restoration
export takes it from ``--conditioning``. Writes ``model.safetensors`` (FP32),
``empty_prompt.safetensors`` and ``config.yaml``, loadable by
``iris3b.downstream.depth.DepthPredictor`` / ``iris3b.downstream.restoration.Restorer``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from iris3b.config import inference_config
from iris3b.downstream.depth import IrisDepth
from iris3b.models.dit import IrisDiT


def depth(payload: dict) -> tuple[dict, dict, dict]:
    if payload["config"]["marigold"]["stage"] != 0:
        raise ValueError("expected the direct-regression (stage 0) depth checkpoint")
    weights = {k: v for k, v in payload["model"].items() if k.startswith(("pixel.", "depth_reducer."))}
    return weights, payload["conditioning"], {"name": "depth"}


def restoration(payload: dict, conditioning: str) -> tuple[dict, dict, dict]:
    generator = payload["config"]["generator"]
    if generator["mode"] != "full" or generator["model_t"] != generator["coeff_t"]:
        raise ValueError(f"unsupported restorer settings {generator}")
    weights = {k.removeprefix("model."): v for k, v in payload["generator"].items()}
    prompt = torch.load(conditioning, map_location="cpu", weights_only=True)
    crop = payload["config"]["data"]["crop_size"]
    return weights, prompt, {"name": "restoration", "sigma": generator["model_t"] / 1000, "tile": crop}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task", choices=["depth", "restoration"])
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument("--parent-config", required=True)
    parser.add_argument("--conditioning", help="empty-prompt cache (restoration only)")
    args = parser.parse_args(argv)

    raw = OmegaConf.to_container(OmegaConf.load(args.parent_config))
    cfg = inference_config(raw, [])
    payload = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if args.task == "depth":
        weights, prompt, task = depth(payload)
    else:
        if not args.conditioning:
            parser.error("restoration needs --conditioning")
        weights, prompt, task = restoration(payload, args.conditioning)

    with torch.device("meta"):
        model = IrisDepth(cfg.model) if args.task == "depth" else IrisDiT(cfg.model)
    model.load_state_dict(weights, strict=True, assign=True)
    embeddings, mask = prompt["embeddings"], prompt["mask"]
    if embeddings.shape[0] != 1 or mask.shape != embeddings.shape[:2]:
        raise ValueError(f"unexpected empty-prompt shapes {tuple(embeddings.shape)} / {tuple(mask.shape)}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.float().contiguous() for k, v in weights.items()},
        str(out / "model.safetensors"),
        metadata={"format": "pt"},
    )
    save_file(
        {"embeddings": embeddings.float().contiguous(), "mask": mask.bool().contiguous()},
        str(out / "empty_prompt.safetensors"),
    )
    OmegaConf.save(
        OmegaConf.create({**{k: raw[k] for k in ("model", "text_encoder", "flow")}, "task": task}),
        str(out / "config.yaml"),
    )
    print(f"wrote {out}: {len(weights)} tensors, task {task}")


if __name__ == "__main__":
    main()
