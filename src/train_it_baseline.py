
from pathlib import Path
import json

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "it"
MODEL_DIR = ROOT / "models" / "it"
REPORT_DIR = ROOT / "data" / "reports"

def main():
    train = np.load(DATA_DIR / "it_train.npz")
    test = np.load(DATA_DIR / "it_test.npz")

    X_train, y_train = train["X"], train["y"]
    X_test, y_test = test["X"], test["y"]

    with open(DATA_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    class_names = metadata["class_names"]

    print("Training Random Forest baseline...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=2,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nClassification report:")
    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    print(report)

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "random_forest.joblib")
    (REPORT_DIR / "it_baseline_report.txt").write_text(
        "Accuracy: "
        + str(accuracy_score(y_test, predictions))
        + "\n\n"
        + report,
        encoding="utf-8",
    )

    print("\nModel saved:", MODEL_DIR / "random_forest.joblib")
    print("Report saved:", REPORT_DIR / "it_baseline_report.txt")

if __name__ == "__main__":
    main()
