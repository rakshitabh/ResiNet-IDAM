import pandas as pd

from src.data.validate_dataset import validate


def _write_dataset(path, columns, rows):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def test_validate_accepts_matching_feature_contract(tmp_path):
    columns = ["timestamp", "feature_a", "feature_b", "label"]
    rows = [
        ["2022-01-01 00:00:00", 0.1, 0.2, 0],
        ["2022-01-01 00:00:01", 0.2, 0.3, 1],
    ]
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    _write_dataset(train, columns[:3], [row[:-1] for row in rows])
    _write_dataset(test, columns, rows)

    report = validate(
        train,
        test,
        tmp_path / "missing-scaler.joblib",
        tmp_path / "missing-features.json",
        tmp_path / "missing-metadata.json",
        tmp_path / "missing-distribution.csv",
    )

    assert report["valid"] is True
    assert report["feature_columns"] == ["feature_a", "feature_b"]
    assert report["test_label_distribution"] == {"0": 1, "1": 1}


def test_validate_rejects_feature_order_mismatch(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    _write_dataset(
        train,
        ["timestamp", "feature_a", "feature_b"],
        [["2022-01-01 00:00:00", 0.1, 0.2]],
    )
    _write_dataset(
        test,
        ["timestamp", "feature_b", "feature_a", "label"],
        [["2022-01-01 00:00:00", 0.2, 0.1, 0]],
    )

    report = validate(
        train,
        test,
        tmp_path / "missing-scaler.joblib",
        tmp_path / "missing-features.json",
        tmp_path / "missing-metadata.json",
        tmp_path / "missing-distribution.csv",
    )

    assert report["valid"] is False
    assert "feature names/order do not match" in report["errors"][0]
