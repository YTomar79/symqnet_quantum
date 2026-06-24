"""Helpers for loading on-disk artifacts safely."""

from __future__ import annotations

from pathlib import Path

import torch

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: str | Path) -> bool:
    """Return True if the file is an unmaterialized Git LFS pointer."""
    try:
        with open(path, "rb") as handle:
            return handle.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def require_materialized(path: str | Path) -> Path:
    """Ensure a tracked file holds real content rather than an LFS pointer."""
    resolved = Path(path)
    if is_lfs_pointer(resolved):
        raise RuntimeError(
            f"{resolved} is a Git LFS pointer, not the actual file. "
            "Install Git LFS and fetch the tracked files with "
            "'git lfs install && git lfs pull', then retry."
        )
    return resolved


def load_torch(path: str | Path, map_location=None, *, weights_only: bool = False):
    """Load a checkpoint, failing clearly on unmaterialized LFS pointers."""
    resolved = require_materialized(path)
    try:
        return torch.load(resolved, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(resolved, map_location=map_location)
