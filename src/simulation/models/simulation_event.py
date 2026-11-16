"""Data models for simulated IT/OT events in ResiNet-IDAM."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class EventSource(str, Enum):
    IT = "IT"
    OT = "OT"


@dataclass
class SimulationEvent:
    event_id: str
    offset_seconds: float
    source: EventSource
    event_type: str
    user_id: str
    device_id: str
    asset_id: str
    ip_address: str = "192.168.1.100"
    mac_address: str = "00:00:00:00:00:00"
    telemetry_record_id: str | int | None = None
    identity_trust: float = 0.80
    device_trust: float = 0.80
    asset_criticality: float = 0.50
    behavior_deviation: float = 0.10
    expected_outcome: str | None = None
    row_number: int = 1
    state: EventState = EventState.PENDING
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "offset_seconds": self.offset_seconds,
            "source": self.source.value,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "asset_id": self.asset_id,
            "ip_address": self.ip_address,
            "mac_address": self.mac_address,
            "telemetry_record_id": self.telemetry_record_id,
            "identity_trust": self.identity_trust,
            "device_trust": self.device_trust,
            "asset_criticality": self.asset_criticality,
            "behavior_deviation": self.behavior_deviation,
            "expected_outcome": self.expected_outcome,
            "row_number": self.row_number,
            "state": self.state.value,
            "context": self.context,
        }

