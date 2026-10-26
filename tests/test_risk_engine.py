import unittest

from src.zero_trust.risk_engine import RiskInput, calculate_risk


def context(**overrides):
    values = {
        "identity_trust": 0.8,
        "device_trust": 0.8,
        "asset_criticality": 0.2,
        "anomaly_severity": 0.1,
        "behavior_deviation": 0.1,
    }
    values.update(overrides)
    return RiskInput(**values)


class TestRiskEngine(unittest.TestCase):

    def test_formula_and_breakdown(self):
        result = calculate_risk(context())
        self.assertEqual(result["risk_score"], 14.0)
        self.assertEqual(result["risk_level"], "LOW")
        self.assertEqual(result["recommended_action"], "RECORD_AND_MONITOR")
        self.assertEqual(result["risk_components"]["anomaly_severity"], 4.0)

    def test_all_severity_bands(self):
        self.assertEqual(calculate_risk(context())["risk_level"], "LOW")
        self.assertEqual(
            calculate_risk(context(anomaly_severity=0.8))["risk_level"], "MEDIUM"
        )
        self.assertEqual(
            calculate_risk(
                context(anomaly_severity=1.0, behavior_deviation=1.0)
            )["risk_level"],
            "HIGH",
        )
        self.assertEqual(
            calculate_risk(
                context(
                    anomaly_severity=1.0,
                    behavior_deviation=1.0,
                    asset_criticality=1.0,
                    identity_trust=0.0,
                    device_trust=0.0,
                )
            )["risk_level"],
            "CRITICAL",
        )

    def test_monotonicity(self):
        baseline = calculate_risk(context())["risk_score"]
        self.assertGreater(
            calculate_risk(context(anomaly_severity=0.9))["risk_score"],
            baseline,
        )
        self.assertGreater(
            calculate_risk(context(identity_trust=0.1))["risk_score"],
            baseline,
        )
        self.assertGreater(
            calculate_risk(context(device_trust=0.1))["risk_score"],
            baseline,
        )

    def test_score_is_bounded_and_deterministic(self):
        data = context(
            anomaly_severity=1.0,
            behavior_deviation=1.0,
            asset_criticality=1.0,
            identity_trust=0.0,
            device_trust=0.0,
        )
        first = calculate_risk(data)
        self.assertEqual(first, calculate_risk(data))
        self.assertGreaterEqual(first["risk_score"], 0)
        self.assertLessEqual(first["risk_score"], 100)

    def test_invalid_numeric_values_are_rejected(self):
        for field, value in (
            ("anomaly_severity", 1.2),
            ("device_trust", -0.1),
            ("asset_criticality", float("nan")),
            ("identity_trust", True),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    calculate_risk(context(**{field: value}))

    def test_zero_trust_decision_rules(self):
        from src.zero_trust.risk_engine import evaluate_zero_trust_decision

        # Allow case
        allow_event = {
            "context": {"identity_trust": 0.9, "device_trust": 0.9, "asset_criticality": 0.2, "anomaly_severity": 0.1, "behavior_deviation": 0.1},
            "risk_score": 15.0,
        }
        dec_allow = evaluate_zero_trust_decision(allow_event)
        self.assertEqual(dec_allow["verdict"], "ALLOW")
        self.assertFalse(dec_allow["is_automatic_action"])

        # Require verification case (moderate risk)
        verify_event = {
            "context": {"identity_trust": 0.5, "device_trust": 0.8, "asset_criticality": 0.5, "anomaly_severity": 0.4, "behavior_deviation": 0.3},
            "risk_score": 55.0,
        }
        dec_verify = evaluate_zero_trust_decision(verify_event)
        self.assertEqual(dec_verify["verdict"], "REQUIRE VERIFICATION")

        # Deny case (coordinated attack: high anomaly, behavior, criticality)
        deny_event = {
            "context": {"identity_trust": 0.3, "device_trust": 0.3, "asset_criticality": 0.8, "anomaly_severity": 0.9, "behavior_deviation": 0.8},
            "risk_score": 85.0,
        }
        dec_deny = evaluate_zero_trust_decision(deny_event)
        self.assertEqual(dec_deny["verdict"], "DENY")
        self.assertGreater(len(dec_deny["rule_traces"]), 0)
        self.assertFalse(dec_deny["is_automatic_action"])


if __name__ == "__main__":
    unittest.main()
