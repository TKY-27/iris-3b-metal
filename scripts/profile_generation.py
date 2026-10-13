"""Actual pipeline cold/warm generation, image evidence and physical memory."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

import iris_metal.pipeline as pipeline_module
from benchmark import (
    _CURRENT_FOOTPRINT,
    _INTERVAL_MAX_FOOTPRINT,
    environment,
    paging_counters,
    peak_footprint_gib,
    swap_used_mb,
)
from iris_metal.pipeline import IrisPipeline
from iris_metal.turbo import png_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", default="A red fox sleeping in fresh snow, golden hour")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--compare-head", type=Path)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = []
    original_loader = pipeline_module.load_dit
    original_encoder = pipeline_module.StreamingQwen3VLEncoder

    def timed_load(*args, **kwargs):
        start = time.perf_counter()
        model = original_loader(*args, **kwargs)
        torch.mps.synchronize()
        stages.append({"phase": "dit_load", "seconds": time.perf_counter() - start})
        return model

    class TimedEncoder(original_encoder):
        def __init__(self, *args, **kwargs):
            start = time.perf_counter()
            super().__init__(*args, **kwargs)
            stages.append({"phase": "qwen_tokenizer_setup", "seconds": time.perf_counter() - start})

        def encode(self, texts):
            start = time.perf_counter()
            result = super().encode(texts)
            torch.mps.synchronize()
            stages.append(
                {
                    "phase": "qwen_encode_including_streamed_layer_transfer",
                    "seconds": time.perf_counter() - start,
                }
            )
            return result

    pipeline_module.load_dit = timed_load
    pipeline_module.StreamingQwen3VLEncoder = TimedEncoder
    results = {
        "environment": environment(),
        "requests": [],
        "paging_before": paging_counters(),
        "swap_before_mb": swap_used_mb(),
    }
    start = time.perf_counter()
    pipe = IrisPipeline(args.checkpoint)
    results["init_seconds"] = time.perf_counter() - start
    peak_footprint_gib(_INTERVAL_MAX_FOOTPRINT)
    for i in range(args.repeats):
        callbacks = []
        start = time.perf_counter()
        image = pipe(
            [args.prompt],
            height=args.size,
            width=args.size,
            steps=args.steps,
            on_step=lambda step, total, callbacks=callbacks, start=start: callbacks.append(
                (step, total, time.perf_counter() - start)
            ),
        )[0]
        seconds = time.perf_counter() - start
        target = args.out / f"request-{i}.png"
        image.save(target, pnginfo=png_info(image))
        results["requests"].append(
            {
                "seconds": seconds,
                "steps": callbacks,
                "physical_peak_gib": peak_footprint_gib(_INTERVAL_MAX_FOOTPRINT),
                "physical_current_gib": peak_footprint_gib(_CURRENT_FOOTPRINT),
                "allocated_bytes": torch.mps.current_allocated_memory(),
                "driver_bytes": torch.mps.driver_allocated_memory(),
                "png_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            }
        )
    if args.compare_head:
        import importlib.util

        spec = importlib.util.spec_from_file_location("iris_head_pipeline", args.compare_head)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        head = module.IrisPipeline(args.checkpoint)
        head._engine = pipe._engine
        head._encodings = pipe._encodings
        start = time.perf_counter()
        baseline = head([args.prompt], height=args.size, width=args.size, steps=args.steps)[0]
        results["head_comparison"] = {
            "seconds": time.perf_counter() - start,
            "pixels_bitwise_equal": baseline.tobytes() == image.tobytes(),
            "head_pixel_sha256": hashlib.sha256(baseline.tobytes()).hexdigest(),
        }
        baseline.save(args.out / "head-original.png")
        head.unload()
    pipe.unload()
    results["unloaded"] = {
        "loaded": pipe.loaded,
        "allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_bytes": torch.mps.driver_allocated_memory(),
    }
    if args.reload:
        start = time.perf_counter()
        reloaded = pipe([args.prompt], height=args.size, width=args.size, steps=args.steps)[0]
        results["reload"] = {
            "seconds": time.perf_counter() - start,
            "pixels_bitwise_equal": reloaded.tobytes() == image.tobytes(),
        }
        reloaded.save(args.out / "reloaded.png", pnginfo=png_info(reloaded))
        pipe.unload()
        results["unloaded_after_reload"] = {
            "loaded": pipe.loaded,
            "allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_bytes": torch.mps.driver_allocated_memory(),
        }
    results["stages"] = stages
    results["metadata"] = pipe.last_metadata
    results["paging_after"] = paging_counters()
    results["swap_after_mb"] = swap_used_mb()
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
