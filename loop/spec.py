"""What the gate measures; ``prepare`` freezes it."""

import json
from pathlib import Path

FOX = "A red fox sleeping in fresh snow, golden hour"
NEW_PROMPTS = (
    "A ceramic teapot on a wooden table, morning light",
    "A lighthouse on a rocky coast at dusk, oil painting",
    "A bowl of lemons on a marble counter, soft shadows",
    "A snowy mountain cabin with smoke from the chimney",
    "A hummingbird hovering near a red flower, macro photograph",
    "A neon-lit street market in the rain at night",
)
TURBO_SIZES = ((512, 512), (1280, 720), (720, 1280), (2048, 512), (768, 1024))
TRAJECTORY_IDS = {
    "dev": (
        "portrait-fisherman",
        "hands-pottery",
        "texture-wool",
        "sign-bread",
        "city-windows",
        "cat-window",
    ),
    "holdout": (
        "portrait-freckles",
        "hands-piano",
        "texture-bark",
        "chalkboard-menu",
        "fence-brick",
        "red-panda",
    ),
}
TRAJECTORY_SIZE, TRAJECTORY_STEPS = 512, 20
VELOCITY_IDS, VELOCITY_STEPS, VELOCITY_PROBES = ("portrait-fisherman", "sign-bread"), 10, 3
OCR = {"dev": ("sign-bread", ("FRESH", "BREAD")), "holdout": ("chalkboard-menu", ("TOMATO", "SOUP"))}
OCR_STEPS = 20
WORKLOADS = {
    "turbo-repeat": {"model": "turbo", "prompts": [FOX] * 7},
    "turbo-new": {"model": "turbo", "prompts": [FOX, *NEW_PROMPTS]},
    "original-10": {"model": "original", "steps": 10, "prompts": [FOX] * 4},
}
AUDIT_WORKLOADS = {"original-100": {"model": "original", "steps": 100, "prompts": [FOX] * 2}}
DEPTH_INPUT, RESTORE_INPUT = "demo/examples/1.webp", "demo/examples/lowres/1.png"


def build(repo: Path) -> dict:
    listed = json.loads((repo / "benchmarks/prompts.json").read_text())
    entries = {split: {p["id"]: p for p in listed[split]} for split in listed}

    def fixture(split, key, seed, size=None, steps=None):
        entry = entries[split][key]
        width, height = size or entry["size"]
        return {
            "id": key,
            "prompt": entry["prompt"],
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
        }

    turbo = {}
    for split, keys in (("dev", list(entries["dev"])), ("holdout", list(entries["holdout"]))):
        jobs = [fixture(split, key, 0) for key in keys]
        repeats = [{**job, "id": f"{job['id']}#2"} for job in jobs[:3]]  # second visits use the caches
        turbo[split] = jobs + repeats
    turbo["dev"] += [
        {"id": f"fox-{w}x{h}", "prompt": FOX, "width": w, "height": h, "seed": 0, "steps": None}
        for w, h in TURBO_SIZES
    ]
    square = (TRAJECTORY_SIZE, TRAJECTORY_SIZE)
    return {
        "turbo": turbo,
        "trajectories": {
            split: [fixture(split, key, seed, square, TRAJECTORY_STEPS) for seed, key in enumerate(keys)]
            for split, keys in TRAJECTORY_IDS.items()
        },
        "velocity": [
            fixture("dev", key, seed, None, VELOCITY_STEPS) for seed, key in enumerate(VELOCITY_IDS)
        ],
        "velocity_probes": VELOCITY_PROBES,
        "ocr": {
            split: {**fixture(split, key, 0, None, OCR_STEPS), "words": list(words)}
            for split, (key, words) in OCR.items()
        },
        "workloads": {**WORKLOADS, **AUDIT_WORKLOADS},
        "workload_size": [1024, 1024],
        "tasks": {"depth": DEPTH_INPUT, "restoration": RESTORE_INPUT},
    }
