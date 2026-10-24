"""Detector registry and availability reporting."""

from __future__ import annotations

from pathlib import Path

from src.detectors.hai_autoencoder import HaiAutoencoderDetector
from src.detectors.ids2025 import Ids2025Detector


def build_registry(root: Path) -> dict[str, object]:
    return {
        "OT": HaiAutoencoderDetector(root / "models"),
        "IT": Ids2025Detector(root),
    }
