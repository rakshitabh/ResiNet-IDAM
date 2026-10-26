# ResiNet-IDAM Dataset Validation Report

**Dataset:** HAI industrial telemetry  
**Inspection date:** 23 October 2026  
**Status:** Phase 1 audit complete; training semantics and scaler provenance remain open

## Source and prepared files

The project is designed to use the HAI dataset from the [official HAI repository](https://github.com/icsdataset/hai).
Raw source files are not committed to this repository. The audit inspected the prepared files currently available under
`data/processed/`.

The official HAI README places `hai-train1` in the **Normal Dataset** group and `hai-test1` in the
**Attack Dataset** group. The repository preprocessing scripts use those exact source names. This supports the
normal-only interpretation of the prepared training file, but the downloaded raw source and release version are not
present in this checkout, so the source-to-processed-file provenance is not independently hash-verifiable.

| File | Rows | Columns | Purpose |
|---|---:|---:|---|
| `hai-train-scaled.csv` | 280,800 | 67 | Timestamp plus 66 model features |
| `hai-test-scaled.csv` | 54,000 | 68 | Timestamp, 66 model features and `label` |
| `hai-test-labeled.csv` | 54,000 | 88 | Unscaled/labeled evaluation data |
| `feature_distribution_report.csv` | 66 features | 13 fields | Train/test distribution comparison |

The machine-readable outputs are [dataset-inspection.json](../reports/dataset-inspection.json) and
[dataset-validation.json](../reports/dataset-validation.json).

## Feature contract

The model input contains exactly 66 numeric process features. `timestamp` is metadata and `label` is reserved for
offline evaluation; neither belongs in an inference feature matrix. The training and scaled test files currently use
the same feature names and ordering.

The prepared files contain no missing cells, invalid timestamps, or duplicate rows. Timestamps are sampled at one
second intervals:

- Training: `2022-08-04 18:00:01` through `2022-08-08 00:00:00`
- Test: `2022-08-12 16:00:01` through `2022-08-13 07:00:00`

The test label distribution is:

| Label | Meaning used by the prepared data | Rows |
|---:|---|---:|
| 0 | Normal | 51,019 |
| 1 | Attack | 2,981 |

## Constant and shifted features

No feature is constant in the prepared training file. Seven features are constant in the prepared test file despite
varying in training:

- `P1_PP04SP`
- `P2_ATSW_Lamp`
- `P2_AutoGO`
- `P2_MASW`
- `P2_MASW_Lamp`
- `P2_ManualGO`
- `P2_ManualSD`

These features must not be removed automatically based only on this observation. Their process meaning and the
reason for the train/test shift must be investigated and recorded before final model selection. The distribution
report also shows substantial mean shifts for some features, so test-time behavior should be treated as a dataset
limitation rather than silently corrected.

## Preprocessing verification

The feature names and ordering are consistent across the available scaled training and test files. However,
`models/standard_scaler.joblib` is not currently available, so the audit cannot independently verify that:

- the scaler was fitted only on training data;
- the saved scaler matches the current feature ordering; or
- the scaled files can be reproduced from the raw inputs.

The scaler must be regenerated or restored from the approved preprocessing run before model artifacts are treated as
reproducible.

## Training-data semantics

The prepared training file has no label column. The official HAI documentation identifies the corresponding
`hai-train1` source as a normal dataset, which supports using it for normal-behavior training. The current checkout
does not contain the raw source or a preparation manifest, so the exact source release and file lineage remain
unverified.

Before Autoencoder training, the team must either:

1. preserve the raw `hai-train1` source and record its HAI release/version and checksum; or
2. reconstruct a trusted normal-only subset from the original labeled/source data if the prepared file cannot be
   linked to `hai-train1`.

No final threshold or test-set performance claim should be made until this assumption is resolved.

## Validation result

The structural validation passed for the currently available prepared files:

- 66 features are present and ordered consistently.
- Model input excludes `timestamp` and `label`.
- Numeric values are finite.
- Test labels contain only `0` and `1`.
- No missing cells or invalid timestamps were found.

The overall dataset readiness gate remains **blocked for fully reproducible model training** by the missing scaler
artifact and missing source-file provenance. The normal-only interpretation is supported by the official HAI dataset
documentation but is not independently hash-verifiable from this checkout.

## Reproduction commands

From the repository root:

```powershell
python src/data/inspect_dataset.py
python src/data/validate_dataset.py
```

The validator exits with a non-zero status for structural errors. Warnings are retained in the JSON report for
conditions that require human review, such as missing preprocessing artifacts or unresolved training semantics.
