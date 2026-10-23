
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent

raw_train_path = root / "data" / "raw" / "hai-train1.txt"
test_path = root / "data" / "processed" / "hai-test-labeled.csv"
train_path = root / "data" / "processed" / "hai-train-features.csv"

# Read training data to reproduce its feature selection.
raw_train = pd.read_csv(raw_train_path, nrows=1000)
constant_cols = [
    col for col in raw_train.columns
    if col != "timestamp" and raw_train[col].nunique(dropna=False) <= 1
]

# Load processed training features and test headers.
train = pd.read_csv(train_path, nrows=5)
test = pd.read_csv(test_path, nrows=5)

train_features = [
    col for col in train.columns if col != "timestamp"
]

# Select only the training features from test data.
test_features = [
    col for col in test.columns
    if col in train_features
]

print("Training feature count:", len(train_features))
print("Matching test feature count:", len(test_features))
print("Same feature names:", set(train_features) == set(test_features))
print("Same feature order:", train_features == test_features)

missing_in_test = sorted(set(train_features) - set(test.columns))
print("Missing training features in test:", missing_in_test)

print("\nFeature selection is consistent."
      if train_features == test_features
      else "\nFeature mismatch remains. Do not train yet.")
