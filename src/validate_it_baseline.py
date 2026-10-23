
from pathlib import Path
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "it"

train = np.load(DATA_DIR / "it_train.npz")
X_train, y_train = train["X"], train["y"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=2,
    class_weight="balanced",
)

print("Running 5-fold cross-validation on training data...")

results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    },
    n_jobs=1,
    return_train_score=False,
)

for metric in ["accuracy", "macro_f1", "weighted_f1"]:
    scores = results[f"test_{metric}"]
    print(
        f"{metric}: "
        f"{scores.mean():.4f} ± {scores.std():.4f}"
    )
