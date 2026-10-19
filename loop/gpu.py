"""Run one command while holding the shared GPU lock.

python loop/gpu.py run -- python -m pytest -q tests/test_metal_engine.py
"""

import subprocess
import sys

from common import gpu_lock


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 3 or args[:2] != ["run", "--"]:
        raise SystemExit(__doc__)
    with gpu_lock(" ".join(args[2:])[:200]):
        return subprocess.run(args[2:]).returncode


if __name__ == "__main__":
    sys.exit(main())
