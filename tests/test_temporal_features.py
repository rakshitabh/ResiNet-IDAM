import unittest
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.features.it_temporal_features import (
    TemporalTelemetryAudit,
    CausalTemporalFeatureExtractor,
)


class TestTemporalFeatures(unittest.TestCase):

    def test_audit_flags_missing_fields_in_raw_netflow(self):
        # Emulate 79-feature NetFlow flow without timestamps or IPs
        df_netflow = pd.DataFrame({
            "Flow Duration": [100, 200, 300],
            "Total Fwd Packets": [10, 20, 30],
        })
        audit = TemporalTelemetryAudit.audit(df_netflow)
        self.assertFalse(audit["can_construct_temporal"])
        self.assertFalse(audit["has_timestamp"])
        self.assertFalse(audit["has_src_ip"])
        self.assertGreater(len(audit["required_telemetry_additions"]), 0)

    def test_extractor_rejects_missing_temporal_telemetry(self):
        df_netflow = pd.DataFrame({"Flow Duration": [100, 200]})
        extractor = CausalTemporalFeatureExtractor()
        with self.assertRaises(ValueError) as ctx:
            extractor.transform(df_netflow)
        self.assertIn("missing required telemetry fields", str(ctx.exception))

    def test_extractor_computes_causal_inter_arrival_time(self):
        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        df_valid = pd.DataFrame({
            "timestamp": [
                base_time + timedelta(seconds=10),
                base_time,  # Out of order to test chronological sort
                base_time + timedelta(seconds=25),
            ],
            "src_ip": ["192.168.1.10", "192.168.1.10", "192.168.1.10"],
            "dst_ip": ["10.0.0.1", "10.0.0.1", "10.0.0.1"],
            "flow_bytes": [500, 200, 800],
        })

        extractor = CausalTemporalFeatureExtractor(window_seconds=300.0)
        transformed = extractor.transform(df_valid)

        # Check sorted order
        self.assertEqual(transformed["timestamp"].iloc[0], base_time)
        self.assertEqual(transformed["timestamp"].iloc[1], base_time + timedelta(seconds=10))
        self.assertEqual(transformed["timestamp"].iloc[2], base_time + timedelta(seconds=25))

        # Check causal inter-arrival times
        self.assertEqual(transformed["inter_arrival_time"].iloc[0], 0.0)
        self.assertEqual(transformed["inter_arrival_time"].iloc[1], 10.0)
        self.assertEqual(transformed["inter_arrival_time"].iloc[2], 15.0)


if __name__ == "__main__":
    unittest.main()
