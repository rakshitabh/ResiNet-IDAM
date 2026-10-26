
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent.parent

train_path = root / "data" / "processed" / "hai-train-features.csv"
test_path = root / "data" / "processed" / "hai-test-labeled.csv"

output_dir = root / "data" / "processed"
model_dir = root / "models"

print("Loading datasets...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

if "timestamp" not in train.columns or "timestamp" not in test.columns:
    raise ValueError("Both datasets must contain a timestamp column.")

feature_cols = [col for col in train.columns if col not in {"timestamp", "label"}]
test_feature_cols = [col for col in test.columns if col not in {"timestamp", "label"}]

if feature_cols != test_feature_cols:
    raise ValueError(
        "Training and test feature columns must have identical names and ordering."
    )

if "label" not in test.columns:
    raise ValueError("The test dataset must contain a label column.")

# Select exactly the same features in the same order.
X_train = train[feature_cols].astype("float32")
X_test = test[feature_cols].astype("float32")

if not X_train.map(pd.api.types.is_number).all().all():
    raise ValueError("Training features must be numeric.")
if not X_test.map(pd.api.types.is_number).all().all():
    raise ValueError("Test features must be numeric.")
if not X_train.notna().all().all() or not X_test.notna().all().all():
    raise ValueError("Training and test features must not contain missing values.")

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nFitting scaler on training data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Scaling test data...")
X_test_scaled = scaler.transform(X_test)

output_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

# Preserve timestamps and labels alongside scaled features.
train_scaled = pd.DataFrame(
    X_train_scaled, columns=feature_cols
)
train_scaled.insert(0, "timestamp", train["timestamp"])

test_scaled = pd.DataFrame(
    X_test_scaled, columns=feature_cols
)
test_scaled.insert(0, "timestamp", test["timestamp"])
test_scaled["label"] = test["label"].astype(int)

train_scaled.to_csv(
    output_dir / "hai-train-scaled.csv", index=False
)
test_scaled.to_csv(
    output_dir / "hai-test-scaled.csv", index=False
)

joblib.dump(scaler, model_dir / "standard_scaler.joblib")
(model_dir / "feature_columns.json").write_text(
    json.dumps(feature_cols, indent=2),
    encoding="utf-8",
)
(model_dir / "preprocessing_metadata.json").write_text(
    json.dumps(
        {
            "source_training_file": str(train_path),
            "source_test_file": str(test_path),
            "scaler": "StandardScaler",
            "scaler_fitted_on": "training features only",
            "feature_columns": feature_cols,
            "feature_count": len(feature_cols),
            "training_rows": len(train),
            "test_rows": len(test),
            "timestamp_excluded": True,
            "label_excluded_from_model_input": True,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("\nScaling completed.")
print("Training output:", train_scaled.shape)
print("Test output:", test_scaled.shape)
print("Scaler saved:", model_dir / "standard_scaler.joblib")
print("Feature contract saved:", model_dir / "feature_columns.json")
print("Preprocessing metadata saved:", model_dir / "preprocessing_metadata.json")
