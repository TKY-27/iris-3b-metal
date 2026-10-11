"""Iris inference on Apple Silicon."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("iris-3b-metal")
except PackageNotFoundError:
    import tomllib

    try:
        metadata = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
        __version__ = metadata["project"]["version"]
    except (OSError, ValueError, KeyError):
        __version__ = "0+unknown"
