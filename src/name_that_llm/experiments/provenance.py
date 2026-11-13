"""Reproducibility helpers shared by experiment entry points."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def git_revision() -> str:
    """Current commit hash, with a `-dirty` suffix if the tree has changes."""
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
