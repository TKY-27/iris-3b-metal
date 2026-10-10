"""The real `iris-metal serve` HTTP layer over stand-in models, for working on
the app without a GPU or model downloads. CPU only; images are synthetic.

    python apps/macos/fake_engine.py [--port 8765] [--step-seconds 0.05] [--load-seconds 0]
"""

import argparse
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageFilter

from iris_metal.server import JobRunner, LocalServer, Models, make_handler


class FakePipeline:
    cfg = SimpleNamespace(sample=SimpleNamespace(steps=100, cfg_scale=3.0, negative_prompt=""))
    loaded = True

    def __init__(self, step_seconds: float, load_seconds: float):
        self.step_seconds = step_seconds
        self.load_seconds = load_seconds

    def __call__(self, prompts, height, width, steps, on_step, seed=0, **kwargs):
        time.sleep(self.load_seconds)  # the app shows this as preparing
        for step in range(1, steps + 1):
            time.sleep(self.step_seconds)
            on_step(step, steps)
        rng = np.random.default_rng(seed)
        y, x = np.mgrid[0:height, 0:width]
        base = rng.uniform(0, 255, 3)
        rgb = np.stack([(base[i] + 80 * np.sin((x + y * (i + 1)) / 97 + i)) % 255 for i in range(3)], -1)
        return [Image.fromarray(rgb.astype(np.uint8))]

    def unload(self):
        self.loaded = False


def fake_depth(image, max_side):
    gray = np.asarray(image.convert("L").filter(ImageFilter.GaussianBlur(8)), dtype=np.float32)
    return gray / 127.5 - 1


class FakeRestorer:
    on_tile = None

    def __call__(self, image, color_fix=True):
        if self.on_tile is not None:
            self.on_tile()
        return image.convert("RGB").resize((image.width * 4, image.height * 4), Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--step-seconds", type=float, default=0.05)
    parser.add_argument("--load-seconds", type=float, default=0.0, help="delay before the first step")
    args = parser.parse_args()
    models = Models(
        FakePipeline(args.step_seconds, args.load_seconds),
        lambda task: fake_depth if task == "depth" else FakeRestorer(),
    )
    server = LocalServer(("127.0.0.1", args.port), make_handler(JobRunner(models, 8)))
    print(f"fake engine on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
