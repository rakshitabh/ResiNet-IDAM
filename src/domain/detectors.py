"""Detector interfaces used by the application orchestration layer."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from src.domain.events import DetectionResult, Source


class ArtifactError(RuntimeError):
    """Raised when a model or preprocessing artifact cannot be loaded."""
    pass


class Detector(Protocol):
    source: Source
    name: str

    def available(self) -> bool:
        ...

    def feature_schema(self) -> list[str]:
        ...

    def detect(
        self,
        values: dict[str, float],
        *,
        timestamp: datetime,
        asset_id: str,
        device_id: str,
    ) -> DetectionResult:
        ...
