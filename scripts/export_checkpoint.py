"""Export a training checkpoint as inference weights plus their config.

Usage:
    python scripts/export_checkpoint.py output/run/checkpoints/latest.pth exported/iris-3b
    python scripts/export_checkpoint.py CKPT.pth OUT_DIR --dtype bf16

Writes ``OUT_DIR/model.safetensors`` (the EMA weights, or the raw weights of a
run trained without EMA) and ``OUT_DIR/config.yaml`` holding only the model,
text-encoder and flow sections; optimizer state, data paths and run settings
are dropped. ``scripts/sample.py --checkpoint OUT_DIR`` loads the result.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from iris3b.config import INFERENCE_SECTIONS, inference_config
from iris3b.models.dit import IrisDiT
from iris3b.sampling import load_for_inference

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", help="training .pth written by scripts/train.py")
    parser.add_argument("out_dir", help="output directory")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    args = parser.parse_args(argv)

    raw, weights = load_for_inference(args.checkpoint)
    cfg = inference_config(raw)
    # key and shape check against the configured architecture, without allocating it
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    model.load_state_dict(weights, strict=True, assign=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = DTYPES[args.dtype]
    save_file(
        {name: tensor.to(dtype).contiguous() for name, tensor in weights.items()},
        str(out / "model.safetensors"),
        metadata={"format": "pt"},
    )
    resolved = OmegaConf.to_container(OmegaConf.structured(cfg))
    sections = {key: resolved[key] for key in INFERENCE_SECTIONS}
    del sections["text_encoder"]["null_embed_dir"]
    OmegaConf.save(OmegaConf.create(sections), str(out / "config.yaml"))
    print(
        f"wrote {out / 'model.safetensors'} ({len(weights)} tensors, {args.dtype}) and {out / 'config.yaml'}"
    )


if __name__ == "__main__":
    main()
