"""Causal Sequence Window Anomaly Detector for Streaming Telemetry.

Maintains a sliding historical FIFO buffer of length L to aggregate anomaly scores
strictly backward-looking over time:
  Score_window(t) = (1 / L) * sum_{i=0}^{L-1} error(t - i)

Ensures zero future-event leakage and measures detection delay (time to first alert).
"""

from __future__ import annotations

from collections import deque
from typing import Sequence
import numpy as np


class CausalSequenceWindowScorer:
    """Aggregates streaming anomaly scores over a causal sliding window."""

    def __init__(self, window_size: int = 5, aggregation: str = "mean") -> None:
        if window_size < 1:
            raise ValueError("Window size must be >= 1.")
        self.window_size = window_size
        self.aggregation = aggregation
        self._buffer: deque[float] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._buffer.clear()

    def update(self, score: float) -> float:
        self._buffer.append(float(score))
        if self.aggregation == "mean":
            return float(np.mean(self._buffer))
        elif self.aggregation == "max":
            return float(np.max(self._buffer))
        elif self.aggregation == "exp_weighted":
            weights = np.exp(np.linspace(-1.0, 0.0, len(self._buffer)))
            weights /= weights.sum()
            return float(np.sum(np.array(self._buffer) * weights))
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")

    def score_sequence(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        self.reset()
        aggregated = np.zeros(len(scores), dtype=np.float32)
        for i, s in enumerate(scores):
            aggregated[i] = self.update(s)
        return aggregated


def compute_detection_delay(y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> dict[str, float | int]:
    """Computes detection delay (in samples) from start of attack burst to first true detection."""
    attack_indices = np.where(y_true_binary == 1)[0]
    if len(attack_indices) == 0:
        return {"first_attack_index": -1, "first_detected_index": -1, "detection_delay_samples": 0}

    # Group contiguous attack bursts
    diffs = np.diff(attack_indices)
    split_points = np.where(diffs > 1)[0]
    starts = [attack_indices[0]] + [attack_indices[i + 1] for i in split_points]
    ends = [attack_indices[i] for i in split_points] + [attack_indices[-1]]

    delays = []
    detected_bursts = 0

    for s, e in zip(starts, ends):
        burst_preds = y_pred_binary[s : e + 1]
        det_pos = np.where(burst_preds == 1)[0]
        if len(det_pos) > 0:
            delay = int(det_pos[0])
            delays.append(delay)
            detected_bursts += 1
        else:
            delays.append(int(e - s + 1))  # Entire burst missed

    return {
        "total_attack_bursts": len(starts),
        "detected_attack_bursts": detected_bursts,
        "burst_detection_rate": float(detected_bursts / len(starts)) if starts else 0.0,
        "median_detection_delay_samples": float(np.median(delays)) if delays else 0.0,
        "p95_detection_delay_samples": float(np.percentile(delays, 95)) if delays else 0.0,
    }

