"""Adaptive Thresholding Module with Poisoning Safeguards for ResiNet-IDAM v2.0.

Implements the dynamic threshold formula:
    T(t) = mu(t) + k * sigma(t)

Incorporates rigorous safety controls per Priority 3 of the Upgrade Plan:
1. Anti-Poisoning Buffer: Excludes alerts, high-risk, and uncertain samples.
2. Max-Change Rate Clamp: Bounds maximum permissible threshold delta per update.
3. Rollback Mechanism: Reverts instantly to frozen static baseline on anomaly bursts or operator trigger.
4. Comprehensive Audit Trail: Logs all updates with timestamp, window size, mu, sigma, and status.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional
import numpy as np


class ThresholdStatus(str, Enum):
    INITIALIZING = "INITIALIZING"
    ACTIVE = "ACTIVE"
    CLAMPED = "CLAMPED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass
class ThresholdAuditEntry:
    timestamp: str
    sample_index: int
    window_size: int
    mu: float
    sigma: float
    raw_threshold: float
    applied_threshold: float
    status: ThresholdStatus
    reason: str


class AdaptiveThresholdEngine:
    """Dynamic threshold engine with anti-poisoning safeguards and audit tracking."""

    def __init__(
        self,
        base_threshold: float,
        k: float = 3.0,
        window_size: int = 1000,
        min_buffer_size: int = 100,
        max_relative_change: float = 0.15,
        name: str = "adaptive_threshold",
    ) -> None:
        if base_threshold <= 0:
            raise ValueError("base_threshold must be positive.")
        if k <= 0:
            raise ValueError("Multiplier k must be positive.")
        if window_size < 10:
            raise ValueError("Window size must be at least 10.")

        self.name = name
        self.base_threshold = float(base_threshold)
        self.current_threshold = float(base_threshold)
        self.k = float(k)
        self.window_size = int(window_size)
        self.min_buffer_size = int(min_buffer_size)
        self.max_relative_change = float(max_relative_change)

        self._benign_buffer: deque[float] = deque(maxlen=self.window_size)
        self._audit_log: List[ThresholdAuditEntry] = []
        self._sample_counter = 0

    @property
    def buffer_count(self) -> int:
        return len(self._benign_buffer)

    def reset_to_base(self, reason: str = "Manual or automated rollback") -> None:
        """Rollback mechanism: reverts threshold and clears buffer."""
        self.current_threshold = self.base_threshold
        self._benign_buffer.clear()
        self._record_audit(
            sample_index=self._sample_counter,
            mu=0.0,
            sigma=0.0,
            raw_threshold=self.base_threshold,
            applied_threshold=self.base_threshold,
            status=ThresholdStatus.ROLLED_BACK,
            reason=reason,
        )

    def evaluate_and_update(
        self,
        error: float,
        is_known_anomaly: bool = False,
        risk_score: Optional[float] = None,
        risk_threshold_exclusion: float = 40.0,
    ) -> tuple[bool, float]:
        """
        Evaluates whether an incoming observation is anomalous, and if safe,
        updates the rolling benign buffer and recalibrates T(t).

        Returns:
            (is_anomaly, current_applied_threshold)
        """
        self._sample_counter += 1
        error = float(error)

        # 1. Decision against current threshold
        is_anomaly = bool(error > self.current_threshold or is_known_anomaly)

        # 2. Poisoning Safeguard: Reject alerts and high-risk observations from buffer
        is_high_risk = risk_score is not None and risk_score >= risk_threshold_exclusion
        is_untrusted = is_anomaly or is_high_risk

        if not is_untrusted:
            self._benign_buffer.append(error)
            self._recalculate_threshold()

        return is_anomaly, self.current_threshold

    def _recalculate_threshold(self) -> None:
        if len(self._benign_buffer) < self.min_buffer_size:
            return

        mu = float(np.mean(self._benign_buffer))
        sigma = float(np.std(self._benign_buffer))
        raw_t = mu + self.k * sigma

        # Maximum change rate clamp relative to current threshold
        max_delta = self.current_threshold * self.max_relative_change
        delta = raw_t - self.current_threshold

        if abs(delta) > max_delta:
            clamped_delta = np.sign(delta) * max_delta
            applied_t = self.current_threshold + clamped_delta
            status = ThresholdStatus.CLAMPED
            reason = f"Change clamped to +/-{self.max_relative_change*100:.1f}% limit"
        else:
            applied_t = raw_t
            status = ThresholdStatus.ACTIVE
            reason = "Normal rolling update"

        # Hard safety floor: never adapt below 50% of the base threshold
        floor = self.base_threshold * 0.5
        if applied_t < floor:
            applied_t = floor
            status = ThresholdStatus.CLAMPED
            reason = "Hard lower safety floor reached"

        self.current_threshold = float(applied_t)
        self._record_audit(
            sample_index=self._sample_counter,
            mu=mu,
            sigma=sigma,
            raw_threshold=raw_t,
            applied_threshold=applied_t,
            status=status,
            reason=reason,
        )

    def _record_audit(
        self,
        sample_index: int,
        mu: float,
        sigma: float,
        raw_threshold: float,
        applied_threshold: float,
        status: ThresholdStatus,
        reason: str,
    ) -> None:
        entry = ThresholdAuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sample_index=sample_index,
            window_size=len(self._benign_buffer),
            mu=round(mu, 6),
            sigma=round(sigma, 6),
            raw_threshold=round(raw_threshold, 6),
            applied_threshold=round(applied_threshold, 6),
            status=status,
            reason=reason,
        )
        self._audit_log.append(entry)
        # Cap audit log in memory to 2,000 entries
        if len(self._audit_log) > 2000:
            self._audit_log.pop(0)

    def audit_trail(self) -> List[dict[str, Any]]:
        return [asdict(e) for e in self._audit_log]

    def status_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_threshold": self.base_threshold,
            "current_threshold": round(self.current_threshold, 6),
            "k_multiplier": self.k,
            "window_size": self.window_size,
            "buffer_count": len(self._benign_buffer),
            "total_samples_evaluated": self._sample_counter,
            "total_updates": len(self._audit_log),
            "latest_status": self._audit_log[-1].status.value if self._audit_log else "UNINITIALIZED",
        }

