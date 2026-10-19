"""Run one command under the gate's locks.

python loop/gpu.py run [--wait SECONDS] -- python scripts/flops.py
python loop/gpu.py cpu [--wait SECONDS] -- python -m pytest -q tests/test_flow.py

``run`` holds the GPU; ``cpu`` runs CPU-only work (GPU tests skipped) outside timing.
"""

import argparse
import os
import subprocess
import sys

from common import gpu_lock, quiet_lock


def main() -> int:
    if "--" not in sys.argv:
        raise SystemExit(__doc__)
    split = sys.argv.index("--")
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("mode", choices=("run", "cpu"))
    parser.add_argument("--wait", type=float, default=60.0)
    args, command = parser.parse_args(sys.argv[1:split]), sys.argv[split + 1 :]
    if not command:
        raise SystemExit(__doc__)
    label = " ".join(command)[:200]
    if args.mode == "cpu":
        with quiet_lock(exclusive=False, wait=args.wait):
            return subprocess.run(command, env={**os.environ, "IRIS_SKIP_GPU_TESTS": "1"}).returncode
    with gpu_lock(label, wait=args.wait):
        return subprocess.run(command).returncode


if __name__ == "__main__":
    sys.exit(main())
