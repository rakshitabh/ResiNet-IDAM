"""Model-artifact loading with explicit availability diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ArtifactError(RuntimeError):
    """Raised when a required deployment artifact is unavailable or invalid."""


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactError(f"Required artifact not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactError(f"Invalid JSON artifact: {path}") from error


def artifact_status(paths: dict[str, Path]) -> dict[str, bool]:
    return {name: path.exists() for name, path in paths.items()}
