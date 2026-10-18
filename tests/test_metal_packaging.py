import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "apps/macos/package_release.sh"


def package(tmp_path, mode, **overrides):
    env = {key: value for key, value in os.environ.items() if not key.startswith("IRIS_")}
    env.update(
        IRIS_RELEASE_VERSION="0.6.0",
        IRIS_RELEASE_BUILD="1",
        IRIS_RELEASE_OUTPUT=str(tmp_path / "new release"),
        **overrides,
    )
    return subprocess.run(
        ["bash", str(SCRIPT), "--mode", mode, "--dry-run"], env=env, capture_output=True, text=True
    )


def test_ad_hoc_dry_run_needs_no_apple_credentials_or_outputs(tmp_path):
    result = package(tmp_path, "ad-hoc")
    assert result.returncode == 0, result.stderr
    assert "ad-hoc" in result.stdout
    assert not (tmp_path / "new release").exists()


def test_developer_id_still_requires_its_credentials(tmp_path):
    result = package(tmp_path, "developer-id")
    assert result.returncode != 0
    assert "IRIS_DEVELOPER_ID" in result.stderr


@pytest.mark.parametrize("mode", ["ad-hoc", "invalid"])
def test_packager_refuses_existing_destinations_and_unknown_modes(tmp_path, mode):
    destination = tmp_path / "new release"
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_text("unchanged")
    result = package(tmp_path, mode)
    assert result.returncode != 0
    assert sentinel.read_text() == "unchanged"
