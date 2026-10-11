#!/bin/zsh
# The commands behind the 2026-10-11 results, run from the repository root with the
# checkpoint in ./iris-3b and HF_HUB_OFFLINE=1. Kept as a record; adjust paths to rerun.
set -euo pipefail
export PYTHONPATH=src HF_HUB_OFFLINE=1

# --- pipeline/: three fresh processes per step count, alternating 10 and 20 steps
for steps in 10 20 20 10 10 20; do
  python scripts/benchmark.py --checkpoint iris-3b --size 1024x1024 --steps $steps --repeats 3 --profile --out run.json
done
python scripts/benchmark.py --checkpoint iris-3b --size 1024x1024 --steps 100 --repeats 2 --out r1-s100.json

# --- fixture/: identical inputs for every implementation
# iris3b-mlx-optimized at b06bbe4, prepared with its scripts/prepare.py; COMP is its checkout.
COMP=/path/to/iris3b-mlx-optimized
# The fixture: that project's Qwen conditioning of its prompt 0, and NumPy default_rng(42) noise.
(cd $COMP && .venv/bin/python -c "
import numpy as np
c = np.load('out/conditioning-0.npz')
noise = np.random.default_rng(42).normal(size=(1, 3, 1024, 1024)).astype(np.float32)
np.savez('fixture.npz', y=c['y'], mask=c['mask'], noise=noise)")
FIX=$COMP/fixture.npz

COMMON=(--height 1024 --width 1024 --prompt 0 --seed 42)
SPEED=(--backend mlx-optimized --dtype bfloat16 --precision-policy mlp --execution fused --eval-every 4
       --cache-mib 256 --attention-dtype reference --tf32)
MEMORY=($SPEED --quantization q8-bf16-fit)
(cd $COMP && bin/iris-mlx --name export-speed $SPEED $COMMON --steps 10 --export)
(cd $COMP && bin/iris-mlx --name export-memory $MEMORY $COMMON --steps 10 --export)

# Two rounds, the second in reverse order; three repeats at 10 steps, two at 20.
for steps in 10 20; do
  r=$([[ $steps == 10 ]] && echo 3 || echo 2)
  /usr/bin/time -l python scripts/denoise_fixture.py $FIX --checkpoint iris-3b --steps $steps --repeats $r
  (cd $COMP && /usr/bin/time -l bin/iris-mlx --name speed $SPEED $COMMON --steps $steps --repeats $r --action sample)
  (cd $COMP && /usr/bin/time -l bin/iris-mlx --name memory $MEMORY $COMMON --steps $steps --repeats $r --action sample)
  /usr/bin/time -l python scripts/denoise_fixture.py $FIX --checkpoint iris-3b --reference autocast --steps $steps --repeats 2
done

# --- summary.json and the README charts
python scripts/plot_benchmarks.py benchmarks/results/2026-10-11 --assets assets/benchmarks
