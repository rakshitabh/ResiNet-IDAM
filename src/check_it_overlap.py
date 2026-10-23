
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "it"

train = np.load(DATA_DIR / "it_train.npz")
test = np.load(DATA_DIR / "it_test.npz")

X_train = train["X"]
X_test = test["X"]
y_train = train["y"]
y_test = test["y"]

# Compare exact feature vectors across the two splits.
train_rows = {row.tobytes() for row in X_train}
test_rows = {row.tobytes() for row in X_test}

overlap = train_rows.intersection(test_rows)

print("Unique training feature rows:", len(train_rows))
print("Unique test feature rows:", len(test_rows))
print("Exact feature-vector overlap:", len(overlap))

# Check whether identical vectors carry different labels across splits.
train_labels = {}
for row, label in zip(X_train, y_train):
    train_labels.setdefault(row.tobytes(), set()).add(int(label))

conflicts = 0
for row, label in zip(X_test, y_test):
    key = row.tobytes()
    if key in train_labels and int(label) not in train_labels[key]:
        conflicts += 1

print("Test rows with conflicting labels among exact matches:", conflicts)
