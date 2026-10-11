# Contributing

Bug reports, measurements on other Macs and focused pull requests are welcome.
For a larger change, open an issue first so the approach can be agreed before
you spend time on it.

## Setup

```bash
git clone https://github.com/TKY-27/iris-3b-metal && cd iris-3b-metal
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # use ".[train,dev]" for upstream training tests
```

## Checks

CI runs Ruff, Python tests and macOS app checks on every pull request. With the
`dev` extra, run Ruff and the engine and interface suite:

```bash
ruff check . && ruff format --check .
pytest tests/test_metal_*.py tests/test_compare_runs.py
```

The full Python suite includes the upstream training tests and needs the
`train` extra:

```bash
pip install -e ".[train,dev]"
pytest
```

The app checks used in CI are:

```bash
cd apps/macos
xcodebuild test -project Iris.xcodeproj -scheme Iris -destination platform=macOS CODE_SIGNING_ALLOWED=NO
xcodebuild build -project Iris.xcodeproj -scheme Iris -configuration Release CODE_SIGNING_ALLOWED=NO
```

The tests use tiny randomly initialized models and need neither the
checkpoints nor a network connection. The tests that run kernels on the GPU
need an Apple Silicon GPU; CI skips them with `IRIS_SKIP_GPU_TESTS=1`, because
GitHub's macOS runners only have a virtual GPU, so run them locally.

For the macOS app, `python apps/macos/fake_engine.py` serves the real HTTP API
over CPU stand-ins, so the app can be worked on without a GPU or model
downloads. Launch development builds with
`--args -libraryRoot /tmp/iris-dev -enginePort 8790` to keep them away from
your own library. The Xcode project is generated from `apps/macos/project.yml`
with [XcodeGen](https://github.com/yonaskolb/XcodeGen) and is checked into the
repository. If project settings change, update both files or regenerate the
project.

## Changes to the engine

- **Image quality comes first.** A change to precision, kernels or the
  sampler needs evidence: per-step velocity error against the FP32 forward
  and complete generations compared with `scripts/compare_runs.py` or
  `scripts/precision_study.py`. One 100-step image is not enough; trajectories
  amplify small differences differently per prompt.
- **Speed claims need controlled measurements.** Check that the GPU is idle
  (`ioreg -r -d 1 -w 0 -c IOAccelerator | grep "Device Utilization"`) and
  that the Mac is not swapping, alternate the order of the variants, and
  report medians over several runs with `scripts/benchmark.py` or
  `scripts/denoise_fixture.py`. A laptop's speed drifts by about 10% as it
  warms up.
- Keep `src/iris3b/` (the upstream code) close to upstream; changes there
  should fix a real problem and be listed in `NOTICE`.

## Pull requests

Keep each pull request to one change, with tests where behavior changes and
documentation that matches. Do not commit checkpoints, generated images,
benchmark runs or local paths. By contributing you agree that your
contribution is licensed under the Apache License 2.0.
