"""Run one command while holding the shared GPU lock.

python loop/gpu.py run [--wait SECONDS] -- python scripts/flops.py
"""

import subprocess
import sys

from common import gpu_lock


def main() -> int:
    args = sys.argv[1:]
    if args[:1] != ["run"] or "--" not in args:
        raise SystemExit(__doc__)
    options, command = args[1 : args.index("--")], args[args.index("--") + 1 :]
    wait = float(options[1]) if options[:1] == ["--wait"] else 60.0
    if not command:
        raise SystemExit(__doc__)
    with gpu_lock(" ".join(command)[:200], wait=wait):
        return subprocess.run(command).returncode


if __name__ == "__main__":
    sys.exit(main())
