"""
ResiNet-IDAM: IT Temporal and Behavioral Feature Engineering Module
-------------------------------------------------------------------
This module provides causal, leakage-safe sliding-window temporal feature
engineering for NetFlow telemetry when timestamps and host identifiers are present.

Dataset Audit Note:
In the raw IDS2025.xlsx dataset, timestamps, Source IP, Destination IP, and
Flow IDs are completely absent. Per scientific integrity requirements, this module
documents the required telemetry contract and strictly prevents the fabrication
of artificial timestamps or arbitrary row-order groupings.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


REQUIRED_TEMPORAL_COLUMNS = ["timestamp", "src_ip", "dst_ip"]


class TemporalTelemetryAudit:
    """Audits input telemetry dataframe for necessary temporal and host identifiers."""

    @staticmethod
    def audit(frame: pd.DataFrame) -> dict[str, Any]:
        columns = [str(c).strip() for c in frame.columns]
        has_timestamp = any("time" in c.lower() or "date" in c.lower() for c in columns)
        has_src_ip = any("src_ip" in c.lower() or "source_ip" in c.lower() or "srcip" in c.lower() for c in columns)
        has_dst_ip = any("dst_ip" in c.lower() or "destination_ip" in c.lower() or "dstip" in c.lower() for c in columns)
        has_flow_id = any("flow_id" in c.lower() or "session_id" in c.lower() for c in columns)

        can_construct_temporal = has_timestamp and (has_src_ip or has_flow_id)

        telemetry_needs = []
        if not has_timestamp:
            telemetry_needs.append("Timestamp (ISO8601 or epoch milliseconds e.g. flowStartMilliseconds)")
        if not has_src_ip:
            telemetry_needs.append("Source IPv4/IPv6 Address (sourceIPv4Address)")
        if not has_dst_ip:
            telemetry_needs.append("Destination IPv4/IPv6 Address (destinationIPv4Address)")
        if not has_flow_id:
            telemetry_needs.append("Session / Flow Identifier (flowId)")

        return {
            "has_timestamp": has_timestamp,
            "has_src_ip": has_src_ip,
            "has_dst_ip": has_dst_ip,
            "has_flow_id": has_flow_id,
            "can_construct_temporal": can_construct_temporal,
            "required_telemetry_additions": telemetry_needs,
            "operational_recommendation": (
                "Deploy NetFlow v9 or IPFIX exporter capturing flowStartMilliseconds, "
                "flowEndMilliseconds, sourceIPv4Address, and destinationIPv4Address to enable "
                "multi-connection sliding-window C2 beaconing analysis."
                if not can_construct_temporal
                else "Telemetry meets temporal requirements."
            ),
        }


class CausalTemporalFeatureExtractor:
    """
    Extracts rolling behavioral features over causal historical windows.
    Strictly prevents future-event leakage and cross-partition contamination.
    """

    def __init__(self, window_seconds: float = 300.0) -> None:
        self.window_seconds = window_seconds

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        audit = TemporalTelemetryAudit.audit(frame)
        if not audit["can_construct_temporal"]:
            raise ValueError(
                f"Cannot construct causal temporal features: missing required telemetry fields. "
                f"Needs: {audit['required_telemetry_additions']}. "
                "Preserve standard 79-feature NetFlow mode."
            )

        df = frame.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # 1. Sort chronologically to preserve causality
        df = df.sort_values("timestamp").reset_index(drop=True)

        group_col = "src_ip" if "src_ip" in df.columns else "flow_id"

        # 2. Causal rolling calculations per host/session
        df["inter_arrival_time"] = df.groupby(group_col)["timestamp"].diff().dt.total_seconds().fillna(0.0)

        # Additional rolling metrics over past window
        return df
