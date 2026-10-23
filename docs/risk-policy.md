# Simulated Zero-Trust Risk Policy

The risk engine combines the Autoencoder result with explicitly simulated identity, device, and asset context. These
values are not authentication evidence and must not be interpreted as a real identity-provider or device-management
decision.

## Input contract

Every numeric input is normalized to `[0, 1]`:

- `anomaly_severity`: severity derived from the anomaly detector.
- `behavior_deviation`: contextual behavioral deviation.
- `identity_trust`: simulated confidence in the identity.
- `device_trust`: simulated confidence in the device.
- `asset_criticality`: simulated importance of the asset.

Every event also carries `identity_id`, `device_id`, and `asset_id`.

## Formula

```text
Risk = 100 × (
    0.40 × anomaly_severity
  + 0.20 × behavior_deviation
  + 0.20 × (1 − identity_trust)
  + 0.10 × (1 − device_trust)
  + 0.10 × asset_criticality
)
```

The implementation returns each weighted component, the normalized inputs, the final score, and the policy action.
The formula and weights are an engineering baseline, not a validated security standard.

## Risk bands and policy actions

| Score | Level | Action |
|---:|---|---|
| 0–29.99 | LOW | Record and monitor |
| 30–59.99 | MEDIUM | Alert for review |
| 60–79.99 | HIGH | Prioritize investigation |
| 80–100 | CRITICAL | Recommend containment for human approval |

The prototype never automatically controls physical industrial equipment.
