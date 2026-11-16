"""Dataset validation rules and upload limit enforcement for simulation CSVs."""

from __future__ import annotations

import ipaddress
import re
from typing import Any

MAX_CSV_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_EVENT_ROWS = 10_000
MAX_OFFSET_SECONDS = 86_400.0  # 24 hours

MANDATORY_COLUMNS = {
    "event_id",
    "offset_seconds",
    "source",
    "event_type",
    "user_id",
    "device_id",
    "asset_id",
}

VALID_SOURCES = {"IT", "OT"}
VALID_EXPECTED_OUTCOMES = {"ALLOW", "REQUIRE_VERIFICATION", "REQUIRE VERIFICATION", "DENY"}

MAC_REGEX = re.compile(r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$")


class ValidationError(ValueError):
    """Raised when CSV validation fails with specific row-level context."""

    def __init__(self, message: str, row: int | None = None, column: str | None = None) -> None:
        self.row = row
        self.column = column
        prefix = f"Row {row}" if row is not None else "Dataset"
        if column:
            prefix += f", Column '{column}'"
        super().__init__(f"{prefix}: {message}")


def validate_file_size(file_bytes: bytes | int) -> None:
    size = len(file_bytes) if isinstance(file_bytes, bytes) else file_bytes
    if size > MAX_CSV_BYTES:
        raise ValidationError(
            f"File size {size / (1024 * 1024):.2f} MB exceeds maximum permitted limit of {MAX_CSV_BYTES / (1024 * 1024):.0f} MB"
        )


def validate_row_count(row_count: int) -> None:
    if row_count > MAX_EVENT_ROWS:
        raise ValidationError(
            f"Row count {row_count} exceeds maximum allowed limit of {MAX_EVENT_ROWS} events"
        )


def validate_schema_columns(columns: list[str]) -> None:
    col_set = set(columns)
    missing = MANDATORY_COLUMNS - col_set
    if missing:
        raise ValidationError(f"Missing mandatory columns: {', '.join(sorted(missing))}")


def validate_event_row(row: dict[str, Any], row_idx: int, seen_ids: set[str]) -> dict[str, Any]:
    event_id = str(row.get("event_id", "")).strip()
    if not event_id:
        raise ValidationError("event_id must not be empty", row=row_idx, column="event_id")
    if event_id in seen_ids:
        raise ValidationError(f"Duplicate event_id '{event_id}' detected", row=row_idx, column="event_id")
    seen_ids.add(event_id)

    raw_offset = row.get("offset_seconds")
    try:
        offset = float(raw_offset)
    except (ValueError, TypeError):
        raise ValidationError(f"offset_seconds must be numeric, got '{raw_offset}'", row=row_idx, column="offset_seconds")
    if offset < 0.0:
        raise ValidationError("offset_seconds must be non-negative", row=row_idx, column="offset_seconds")
    if offset > MAX_OFFSET_SECONDS:
        raise ValidationError(f"offset_seconds {offset} exceeds maximum permitted 86400s", row=row_idx, column="offset_seconds")

    source = str(row.get("source", "")).strip().upper()
    if source not in VALID_SOURCES:
        raise ValidationError(f"source must be 'IT' or 'OT', got '{source}'", row=row_idx, column="source")

    for field_name in ("event_type", "user_id", "device_id", "asset_id"):
        val = str(row.get(field_name, "")).strip()
        if not val:
            raise ValidationError(f"{field_name} must not be empty", row=row_idx, column=field_name)

    ip_addr = str(row.get("ip_address", "")).strip()
    if ip_addr:
        try:
            ipaddress.ip_address(ip_addr)
        except ValueError:
            raise ValidationError(f"Invalid IP address format '{ip_addr}'", row=row_idx, column="ip_address")

    mac_addr = str(row.get("mac_address", "")).strip()
    if mac_addr and not MAC_REGEX.match(mac_addr):
        raise ValidationError(f"Invalid MAC address format '{mac_addr}'", row=row_idx, column="mac_address")

    for trust_field in ("identity_trust", "device_trust", "asset_criticality", "behavior_deviation"):
        val = row.get(trust_field)
        if val is not None and str(val).strip() != "":
            try:
                f_val = float(val)
                if not (0.0 <= f_val <= 1.0):
                    raise ValidationError(f"{trust_field} must be in [0.0, 1.0], got {f_val}", row=row_idx, column=trust_field)
            except (ValueError, TypeError):
                raise ValidationError(f"{trust_field} must be numeric, got '{val}'", row=row_idx, column=trust_field)

    expected = row.get("expected_outcome")
    if expected is not None and str(expected).strip() != "":
        exp_str = str(expected).strip().upper().replace(" ", "_")
        if exp_str not in {"ALLOW", "REQUIRE_VERIFICATION", "DENY"}:
            # Check if it was without underscore
            if str(expected).strip().upper() not in VALID_EXPECTED_OUTCOMES:
                raise ValidationError(
                    f"expected_outcome must be 'ALLOW', 'REQUIRE_VERIFICATION', or 'DENY', got '{expected}'",
                    row=row_idx,
                    column="expected_outcome",
                )

    return row

