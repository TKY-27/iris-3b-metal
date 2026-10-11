# Contributing

Bug reports, measurements on other Macs and focused pull requests are welcome.
For a larger change, open an issue first so the approach can be agreed before
you spend time on it.

## Setup

```bash
git clone https://github.com/TKY-27/iris-3b-metal && cd iris-3b-metal
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ",train" for the upstream training tests
```

## Checks

CI runs these on every pull request; run them before you push:

```bash
ruff check . && ruff format --check .
pytest tests/test_metal_*.py                     # engine, CLI, HTTP API (".[dev]")
pytest                                           # everything (".[train,dev]")
cd apps/macos && xcodebuild test -scheme Iris -destination platform=macOS
```

The tests use tiny randomly initialized models and need neither the
checkpoints nor a network connection. Tests marked "needs an Apple GPU" skip
without one.

For the macOS app, `python apps/macos/fake_engine.py` serves the real HTTP API
over CPU stand-ins, so the app can be worked on without a GPU or model
downloads. Launch development builds with
`--args -libraryRoot /tmp/iris-dev -enginePort 8790` to keep them away from
your own library. The Xcode project is generated from `apps/macos/project.yml`
with [XcodeGen](https://github.com/yonaskolb/XcodeGen); edit both, or
regenerate.

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
