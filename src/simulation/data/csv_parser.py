"""Canonical CSV Parser and Ingestion Engine for ResiNet-IDAM Simulation."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Any

from src.simulation.data.dataset_validator import (
    ValidationError,
    validate_event_row,
    validate_file_size,
    validate_row_count,
    validate_schema_columns,
)
from src.simulation.models.simulation_event import EventSource, EventState, SimulationEvent


class SimulationCsvParser:
    """Parses and validates CSV event scenarios conforming to the 15-column canonical contract."""

    @classmethod
    def parse(cls, content: str | bytes | Path) -> tuple[list[SimulationEvent], str]:
        """Parse, validate, and return sorted simulation events with dataset SHA-256 hash."""
        if isinstance(content, Path):
            with open(content, "rb") as f:
                raw_bytes = f.read()
            text = raw_bytes.decode("utf-8", errors="replace")
        elif isinstance(content, bytes):
            raw_bytes = content
            text = raw_bytes.decode("utf-8", errors="replace")
        else:
            text = content
            raw_bytes = content.encode("utf-8")

        validate_file_size(raw_bytes)
        dataset_hash = hashlib.sha256(raw_bytes).hexdigest()

        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            raise ValidationError("CSV file is empty or missing header line")

        # Normalize column names: strip whitespace
        fieldnames = [c.strip() for c in reader.fieldnames]
        validate_schema_columns(fieldnames)

        raw_rows = list(reader)
        validate_row_count(len(raw_rows))
        if not raw_rows:
            raise ValidationError("CSV dataset contains zero event rows")

        seen_ids: set[str] = set()
        validated_events: list[SimulationEvent] = []

        for idx, row in enumerate(raw_rows, start=1):
            cleaned_row = {k.strip(): v.strip() if isinstance(v, str) else v for k, v in row.items()}
            validate_event_row(cleaned_row, idx, seen_ids)

            source = EventSource(cleaned_row["source"].upper())
            offset = float(cleaned_row["offset_seconds"])
            
            # Context factors with defaults
            id_trust = float(cleaned_row.get("identity_trust") or 0.80)
            dev_trust = float(cleaned_row.get("device_trust") or 0.80)
            asset_crit = float(cleaned_row.get("asset_criticality") or 0.50)
            beh_dev = float(cleaned_row.get("behavior_deviation") or 0.10)
            
            exp_outcome = cleaned_row.get("expected_outcome")
            if exp_outcome:
                exp_outcome = exp_outcome.strip().upper().replace(" ", "_")

            # Ground-truth attack label isolated strictly for post-prediction evaluation (Phase 4.1)
            raw_gt = cleaned_row.get("ground_truth_label") or cleaned_row.get("label")
            if raw_gt is not None and str(raw_gt).strip():
                gt_label = str(raw_gt).strip()
                is_atk = gt_label.lower() not in ("normal", "benign", "0", "false")
            else:
                gt_label = None
                is_atk = None

            event = SimulationEvent(
                event_id=cleaned_row["event_id"],
                offset_seconds=offset,
                source=source,
                event_type=cleaned_row["event_type"],
                user_id=cleaned_row["user_id"],
                device_id=cleaned_row["device_id"],
                asset_id=cleaned_row["asset_id"],
                ip_address=cleaned_row.get("ip_address") or "192.168.1.100",
                mac_address=cleaned_row.get("mac_address") or "00:00:00:00:00:00",
                telemetry_record_id=cleaned_row.get("telemetry_record_id") or None,
                identity_trust=id_trust,
                device_trust=dev_trust,
                asset_criticality=asset_crit,
                behavior_deviation=beh_dev,
                expected_outcome=exp_outcome,
                ground_truth_label=gt_label,
                is_attack_ground_truth=is_atk,
                row_number=idx,
                state=EventState.PENDING,
                context=cleaned_row.get("context", ""),
            )
            validated_events.append(event)

        # Sort events by offset_seconds ascending; preserve original row_number as tie-breaker
        sorted_events = sorted(validated_events, key=lambda e: (e.offset_seconds, e.row_number))
        return sorted_events, dataset_hash

