import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ids2025_source_contract():
    frame_path = ROOT / "data" / "processed" / "IDS2025.csv"
    if not frame_path.exists():
        return

    import pandas as pd

    frame = pd.read_csv(frame_path, nrows=2)
    assert frame.columns[-1] == "newLabel"
    assert len(frame.columns) - 1 == 79
    assert all(pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns[:-1])


def test_ids2025_deployment_metadata_contract():
    metadata_path = ROOT / "data" / "processed" / "it_realistic" / "metadata.json"
    if not metadata_path.exists():
        return

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    configuration = metadata["configurations"]["all_features"]
    assert metadata["label_column"] == "newLabel"
    assert configuration["feature_count"] == 79
    assert len(configuration["features"]) == 79
