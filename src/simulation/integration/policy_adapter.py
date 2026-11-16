"""Authoritative Zero-Trust policy evaluation adapter with explicit safety boundaries."""

from __future__ import annotations

from typing import Any
from src.zero_trust.risk_engine import evaluate_zero_trust_decision


class PolicyAdapter:
    """Invokes authoritative Zero-Trust policy rules with human-in-the-loop decision-support labeling."""

    @classmethod
    def evaluate_policy(cls, risk_result: dict[str, Any]) -> dict[str, Any]:
        decision_payload = {
            "risk_score": risk_result["risk_score"],
            "context": risk_result["inputs"],
        }
        policy_eval = evaluate_zero_trust_decision(decision_payload)

        # Guarantee strict safety boundary: Rule 10
        verdict = policy_eval["verdict"]
        raw_action = policy_eval["action"]
        recommended_action = f"{raw_action} (Simulated Advisory)"

        return {
            "verdict": verdict,
            "recommended_action": recommended_action,
            "raw_action": raw_action,
            "reason": policy_eval["reason"],
            "rule_traces": policy_eval.get("rule_traces", []),
            "is_automatic_action": False,
        }

