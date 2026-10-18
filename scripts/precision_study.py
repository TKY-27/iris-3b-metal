"""Compare complete generations with the upstream code run in full FP32.

Usage:
    python scripts/precision_study.py fixtures --out runs/precision --ids portrait-fisherman sign-bread
    python scripts/precision_study.py run fp32 --out runs/precision          # the reference
    python scripts/precision_study.py run autocast --out runs/precision      # upstream's BF16 recipe
    python scripts/precision_study.py run engine --out runs/precision        # this engine
    python scripts/precision_study.py score --out runs/precision autocast engine

``fixtures`` saves each prompt's text-encoder states (with the negative
prompt) and its initial noise, so every implementation denoises exactly the
same inputs with the upstream sampler. ``run`` denoises every fixture in
``--out`` and reuses a result only if it records the same source, checkpoint,
config, fixture, step count and mode; any other existing result stops the run
instead of being replaced. The FP32 reference takes about 8 minutes per
512 x 512 image on an M5. ``score`` reports NRMSE (RMS
error over RMS of the reference) and LPIPS against the FP32 results. Over a
whole trajectory small differences are amplified, so use several prompts.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch

import iris3b
import iris_metal
from iris3b.flow.solver import FlowDPMSolver
from iris_metal.cli import REPO_ID, resolve_checkpoint
from iris_metal.loading import load_config, load_dit
from iris_metal.model import MetalIris

MODES = ("fp32", "autocast", "engine", "engine-bf16")
ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ("torch", "numpy", "safetensors", "transformers", "tokenizers", "huggingface-hub", "omegaconf")


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def inputs_identity(checkpoint: Path) -> dict:
    sources = [Path(__file__).resolve()]
    for package in (iris3b, iris_metal):
        sources.extend(sorted(Path(package.__file__).resolve().parent.rglob("*.py")))
    return {
        "checkpoint": {
            "config_sha256": sha256(checkpoint / "config.yaml"),
            "weights_sha256": sha256(checkpoint / "model.safetensors"),
        },
        "source_sha256": {
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): sha256(path)
            for path in sources
        },
        "dependencies": {name: importlib.metadata.version(name) for name in DEPENDENCIES},
    }


def plan_digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def load_manifest(out: Path) -> dict:
    path = out / "fixtures.json"
    if not path.is_file():
        raise SystemExit(f"{out} has no complete fixture manifest; use fixtures first")
    manifest = json.loads(path.read_text())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit(f"{path} has no complete fixture set")
    actual = {p.stem.removeprefix("fixture-") for p in out.glob("fixture-*.npz")}
    if set(files) != actual or set(files) != set(manifest["plan"]["fixtures"]):
        raise SystemExit(f"{out} has an incomplete fixture set; use a new --out")
    for key, digest in files.items():
        if sha256(out / f"fixture-{key}.npz") != digest:
            raise SystemExit(f"{out} has a changed fixture {key}; use a new --out")
    return manifest


def result_provenance(manifest: dict, key: str, mode: str, steps: int, source_sha256: dict) -> dict:
    return {
        "plan_sha256": plan_digest(manifest["plan"]),
        "fixture_sha256": manifest["files"][key],
        "source_sha256": source_sha256,
        "mode": mode,
        "steps": steps,
    }


def read_result(path: Path, expected: dict, size: int) -> np.ndarray:
    with np.load(path) as data:
        kept = json.loads(str(data.get("provenance", "null")))
        if kept != expected or "final" not in data:
            raise SystemExit(f"{path} has different or missing provenance; use a new --out")
        final = data["final"]
    if final.shape != (1, 3, size, size):
        raise SystemExit(f"{path} has unexpected final dimensions")
    if not np.issubdtype(final.dtype, np.number) or not np.isfinite(final).all():
        raise SystemExit(f"{path} has nonfinite final values")
    return final


def make_fixtures(args, cfg) -> None:
    from iris_metal.pipeline import IrisPipeline

    listed = json.loads(Path(args.prompts).read_text())[args.split]
    prompts = {p["id"]: p["prompt"] for p in listed}
    if len(prompts) != len(listed) or len(set(args.ids)) != len(args.ids) or set(args.ids) - prompts.keys():
        raise SystemExit("prompt ids must be unique and present in the selected split")
    args.out.mkdir(parents=True, exist_ok=True)
    plan = {
        "inputs": inputs_identity(Path(args.checkpoint)),
        "prompts_sha256": sha256(Path(args.prompts)),
        "split": args.split,
        "fixtures": {
            key: {"prompt": prompts[key], "seed": seed, "size": args.size}
            for seed, key in enumerate(args.ids)
        },
    }
    manifest_path = args.out / "fixtures.json"
    if manifest_path.exists():
        if load_manifest(args.out)["plan"] != plan:
            raise SystemExit(f"{args.out} holds another fixture plan; use a new --out")
        return
    if any(p for p in args.out.glob("*.npz") if not p.name.startswith("fixture-")):
        raise SystemExit(f"{args.out} has results without a fixture manifest; use a new --out")
    existing = {p.stem.removeprefix("fixture-") for p in args.out.glob("fixture-*.npz")}
    if existing - plan["fixtures"].keys():
        raise SystemExit(f"{args.out} has other fixtures; use a new --out")
    for key in existing:
        path = args.out / f"fixture-{key}.npz"
        identity = {"plan_sha256": plan_digest(plan), **plan["fixtures"][key]}
        with np.load(path) as data:
            kept = json.loads(str(data.get("identity", "null")))
            valid = (
                data["noise"].shape == (1, 3, args.size, args.size)
                and data["y"].shape[0] == data["mask"].shape[0] == 2
                and np.isfinite(data["noise"]).all()
                and np.isfinite(data["y"]).all()
            )
        if kept != identity or not valid:
            raise SystemExit(f"{path} holds other or invalid inputs; use a new --out")
    pipe = IrisPipeline(args.checkpoint) if len(existing) < len(plan["fixtures"]) else None
    for key, details in plan["fixtures"].items():
        path = args.out / f"fixture-{key}.npz"
        if key in existing:
            print(f"fixture {key} exists, kept")
            continue
        (cond, cond_mask), (neg, neg_mask) = pipe.encode([details["prompt"], cfg.sample.negative_prompt])
        noise = torch.randn(
            1, 3, args.size, args.size, generator=torch.Generator().manual_seed(details["seed"])
        )
        with open(path, "xb") as file:
            np.savez(
                file,
                y=torch.cat([neg, cond]).float().numpy(),  # unconditional row first
                mask=torch.cat([neg_mask, cond_mask]).numpy(),
                noise=noise.numpy(),
                identity=json.dumps({"plan_sha256": plan_digest(plan), **details}),
            )
        print(f"fixture {key} (seed {details['seed']})")
    files = {key: sha256(args.out / f"fixture-{key}.npz") for key in plan["fixtures"]}
    with open(manifest_path, "xb") as file:
        file.write((json.dumps({"plan": plan, "files": files}, indent=2) + "\n").encode())


def upstream_velocity(model, states: torch.Tensor, mask: torch.Tensor, autocast: bool):
    """The upstream model as the solver's velocity; ``rows`` index the (unconditional, conditional) states."""

    def velocity(x: torch.Tensor, t: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        with torch.autocast("mps", dtype=torch.bfloat16, enabled=autocast):
            return model(x, t, states[rows], y_mask=mask[rows]).x.float()

    return velocity


def denoise(args, cfg) -> None:
    manifest = load_manifest(args.out)
    current = inputs_identity(Path(args.checkpoint))
    prepared = manifest["plan"]["inputs"]
    if any(current[field] != prepared[field] for field in ("checkpoint", "dependencies")):
        raise SystemExit(f"{args.out} has another checkpoint or dependencies; use a new --out")
    if args.steps <= 0:
        raise SystemExit("steps must be positive")
    pending = []
    for key in sorted(manifest["files"]):
        target = args.out / f"{args.mode}-{key}.npz"
        made_from = result_provenance(manifest, key, args.mode, args.steps, current["source_sha256"])
        if target.exists():
            read_result(target, made_from, manifest["plan"]["fixtures"][key]["size"])
        else:
            pending.append((key, target, made_from))
    if not pending:
        return
    device = torch.device("mps")
    sample = cfg.sample
    if args.mode in ("fp32", "autocast"):
        model = load_dit(args.checkpoint, cfg, device, torch.float32)
    else:
        dtype = torch.bfloat16 if args.mode == "engine-bf16" else torch.float16
        engine = MetalIris(load_dit(args.checkpoint, cfg, device, dtype), attention_dtype=torch.float16)

    for key, target, made_from in pending:
        with np.load(args.out / f"fixture-{key}.npz") as data:
            states, mask = torch.from_numpy(data["y"]), torch.from_numpy(data["mask"])
            noise = torch.from_numpy(data["noise"]).to(device)
        if args.mode in ("fp32", "autocast"):
            model_fn = upstream_velocity(model, states.to(device), mask.to(device), args.mode == "autocast")
            rows = torch.tensor([0, 1], device=device)
            cond, uncond = rows[1:], rows[:1]
        else:
            text = engine.encode_text(states, mask)
            model_fn, cond, uncond = engine, text[1:], text[:1]
        solver = FlowDPMSolver(
            model_fn,
            num_timesteps=cfg.flow.num_train_timesteps,
            cfg_scale=sample.cfg_scale,
            cfg_interval=tuple(sample.cfg_interval),
            prediction=cfg.flow.prediction,
        )
        start = time.perf_counter()
        final = solver.sample(noise, cond, uncond, steps=args.steps, order=sample.order, shift=cfg.flow.shift)
        if not torch.isfinite(final).all().item():
            raise SystemExit(f"{args.mode} {key} produced nonfinite final values")
        torch.mps.synchronize()
        with open(target, "xb") as file:
            np.savez(file, final=final.cpu().numpy(), provenance=json.dumps(made_from))
        print(f"{args.mode} {key}: {time.perf_counter() - start:.0f}s", flush=True)


def score(args) -> None:
    manifest = load_manifest(args.out)
    keys = sorted(manifest["files"])
    results = {}
    steps = None
    for mode in ("fp32", *args.modes):
        actual = {p.stem.removeprefix(f"{mode}-") for p in args.out.glob(f"{mode}-*.npz")}
        if actual != set(keys):
            raise SystemExit(f"{mode} has an incomplete result set; use a new --out")
        mode_source = None
        for key in keys:
            path = args.out / f"{mode}-{key}.npz"
            with np.load(path) as data:
                kept = json.loads(str(data.get("provenance", "null")))
            if not isinstance(kept, dict) or not isinstance(kept.get("steps"), int) or kept["steps"] <= 0:
                raise SystemExit(f"{path} has missing steps provenance")
            source = kept.get("source_sha256")
            if (
                not isinstance(source, dict)
                or not source
                or any(
                    not isinstance(name, str)
                    or not name
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                    for name, digest in source.items()
                )
            ):
                raise SystemExit(f"{path} has missing source provenance")
            if mode_source is None:
                mode_source = source
            elif source != mode_source:
                raise SystemExit(f"{mode} has inconsistent source provenance")
            if steps is None:
                steps = kept["steps"]
            results[mode, key] = read_result(
                path,
                result_provenance(manifest, key, mode, steps, source),
                manifest["plan"]["fixtures"][key]["size"],
            )
    import lpips

    net = lpips.LPIPS(net="alex", verbose=False)
    for mode in args.modes:
        errors, distances = [], []
        for key in keys:
            reference, result = (np.clip(results[name, key], -1, 1) for name in ("fp32", mode))
            errors.append(np.sqrt(((result - reference) ** 2).mean() / (reference**2).mean()))
            with torch.no_grad():
                distances.append(net(torch.from_numpy(result), torch.from_numpy(reference)).item())
            print(f"{mode:12s} {key:24s} NRMSE {errors[-1]:7.2%}  LPIPS {distances[-1]:.4f}")
        print(
            f"{mode:12s} {'mean of ' + str(len(keys)):24s} NRMSE {np.mean(errors):7.2%}  "
            f"LPIPS {np.mean(distances):.4f}  (median NRMSE {np.median(errors):.2%})"
        )


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, required=True, help="directory of fixtures and results")
    common.add_argument("--checkpoint", default=REPO_ID, help="local directory or Hugging Face repo")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fixtures = commands.add_parser("fixtures", parents=[common])
    fixtures.add_argument("--prompts", default="benchmarks/prompts.json")
    fixtures.add_argument("--split", choices=("dev", "holdout"), default="dev")
    fixtures.add_argument("--ids", nargs="+", required=True)
    fixtures.add_argument("--size", type=int, default=512)
    run = commands.add_parser("run", parents=[common])
    run.add_argument("mode", choices=MODES)
    run.add_argument("--steps", type=int, default=100)
    scoring = commands.add_parser("score", parents=[common])
    scoring.add_argument("modes", nargs="+", choices=MODES[1:])
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    if args.command == "score":
        score(args)
        return
    args.checkpoint = resolve_checkpoint(args.checkpoint, None)
    if args.command == "fixtures":
        make_fixtures(args, load_config(args.checkpoint))
    else:
        denoise(args, load_config(args.checkpoint))


if __name__ == "__main__":
    main()
