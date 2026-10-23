# Baseline Autoencoder Evaluation

**Model:** `dense-autoencoder-baseline-v1`  
**Dataset:** HAI prepared test data  
**Evaluation date:** 23 October 2026

## Reproduction

The baseline uses a dense Autoencoder with the architecture:

```text
66 -> 32 -> 16 -> 32 -> 66
```

Training uses the first 80% of the prepared training timeline and validation uses the final 20%. Training is
chronological and does not shuffle rows. The threshold is the 99th percentile of normal validation reconstruction
errors. Test labels are used only after inference to calculate evaluation metrics.

Use Python 3.13 with TensorFlow 2.21.0:

```powershell
py -3.13 -m venv .venv-ae
.\\.venv-ae\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python src/train_autoencoder.py
python src/evaluate_autoencoder.py
```

The evaluation command loads `models/autoencoder.keras` and `models/anomaly_threshold.json`, then writes:

- `reports/test_anomaly_scores.csv`
- `reports/baseline_metrics.json`

The inference feature matrix excludes both `timestamp` and `label`.

## Measured result

| Metric | Value |
|---|---:|
| Test rows | 54,000 |
| Normal rows | 51,019 |
| Attack rows | 2,981 |
| Threshold | 0.8220725656 |
| Precision | 0.348978 |
| Recall | 0.463938 |
| F1 | 0.398329 |
| False-positive rate | 0.050569 |
| ROC-AUC | 0.763834 |
| PR-AUC | 0.423460 |

Confusion matrix:

|  | Predicted normal | Predicted anomaly |
|---|---:|---:|
| Actual normal | 48,439 | 2,580 |
| Actual attack | 1,598 | 1,383 |

## Interpretation and limitations

These are benchmark results for the prepared HAI split, not a guarantee of zero-day detection or production
performance. The measured false-positive rate is approximately 5.06% on the test normal records. The test set is
imbalanced, so precision, recall, F1, false-positive rate, ROC-AUC and PR-AUC should be considered together.

The threshold was selected before test evaluation from chronological validation errors. The normal-only interpretation
of the `hai-train1` source is supported by the official HAI dataset documentation, but the raw source checksum and
release-version provenance are not present in this checkout.
