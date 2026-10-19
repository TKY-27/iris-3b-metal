import os
import subprocess
import sys

import pytest

import common


def test_run_child_kills_the_process_group_on_timeout(tmp_path):
    script = "import subprocess, time; subprocess.Popen(['sleep', '30']); time.sleep(30)"
    with pytest.raises(RuntimeError, match="timed out"):
        common.run_child([sys.executable, "-c", script], tmp_path / "log", timeout=1)


def test_forged_holder_does_not_skip_the_lock(monkeypatch):
    monkeypatch.setenv(common.LOCKED_ENV, "999999")
    assert not common.holds_lock()
    monkeypatch.setenv(common.LOCKED_ENV, str(os.getppid()))
    assert common.holds_lock()


def test_lock_gives_up_after_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "STATE", tmp_path)
    with common.gpu_lock("first"):
        os.environ.pop(common.LOCKED_ENV)
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]); import common, pathlib; "
            "common.STATE = pathlib.Path(sys.argv[2]); "
            "common.gpu_lock('second', wait=0).__enter__()"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(common.KIT), str(tmp_path)], capture_output=True, text=True
        )
        os.environ[common.LOCKED_ENV] = str(os.getpid())
    assert "GPU busy (first" in result.stderr
