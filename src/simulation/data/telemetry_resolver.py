"""Empirical telemetry resolution engine for IT and OT detectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd


class TelemetryResolver:
    """Resolves genuine empirical feature vectors from test datasets without data hallucination."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
        self.ot_data_path = self.root / "data" / "processed" / "hai-test-scaled.csv"
        if not self.ot_data_path.exists():
            self.ot_data_path = self.root / "data" / "processed" / "hai-test-labeled.csv"

        self.it_data_path = self.root / "data" / "processed" / "ids2025_sample.csv"

        self._ot_df: pd.DataFrame | None = None
        self._it_df: pd.DataFrame | None = None

        self._ot_schema: list[str] = []
        self._it_schema: list[str] = []

    def _load_ot(self) -> pd.DataFrame:
        if self._ot_df is None:
            if not self.ot_data_path.exists():
                raise FileNotFoundError(f"OT empirical telemetry dataset not found: {self.ot_data_path}")
            df = pd.read_csv(self.ot_data_path)
            # Remove non-feature columns
            cols_to_drop = [c for c in ("timestamp", "label", "Attack") if c in df.columns]
            feature_cols = [c for c in df.columns if c not in cols_to_drop]
            self._ot_schema = feature_cols
            self._ot_df = df
        return self._ot_df

    def _load_it(self) -> pd.DataFrame:
        if self._it_df is None:
            if not self.it_data_path.exists():
                raise FileNotFoundError(f"IT empirical telemetry dataset not found: {self.it_data_path}")
            df = pd.read_csv(self.it_data_path)
            cols_to_drop = [c for c in ("label", "newLabel", "Attack", "timestamp") if c in df.columns]
            feature_cols = [c for c in df.columns if c not in cols_to_drop]
            self._it_schema = feature_cols
            self._it_df = df
        return self._it_df

    def ot_feature_schema(self) -> list[str]:
        self._load_ot()
        return list(self._ot_schema)

    def it_feature_schema(self) -> list[str]:
        self._load_it()
        return list(self._it_schema)

    def resolve(self, source: str, record_id: str | int | None = None) -> dict[str, float]:
        """Resolve an empirical feature dictionary for the requested source and record ID."""
        src = source.strip().upper()
        if src == "OT":
            df = self._load_ot()
            schema = self._ot_schema
        elif src == "IT":
            df = self._load_it()
            schema = self._it_schema
        else:
            raise ValueError(f"Unsupported telemetry source: '{source}'. Must be 'IT' or 'OT'.")

        row_idx = 0
        if record_id is not None:
            rec_str = str(record_id).strip()
            if rec_str.isdigit():
                idx_candidate = int(rec_str)
                if 0 <= idx_candidate < len(df):
                    row_idx = idx_candidate
                else:
                    raise IndexError(f"telemetry_record_id {record_id} out of bounds (dataset size: {len(df)})")
            elif "timestamp" in df.columns:
                match = df[df["timestamp"] == rec_str]
                if not match.empty:
                    row_idx = int(match.index[0])
                else:
                    raise KeyError(f"telemetry_record_id '{record_id}' not found in timestamp index")
            else:
                raise ValueError(f"Cannot resolve telemetry_record_id: '{record_id}'")

        row = df.iloc[row_idx]
        features = {col: float(row[col]) for col in schema}
        return features

