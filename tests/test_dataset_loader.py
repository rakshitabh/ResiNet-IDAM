import io

from app import load_dataset, resolve_dataset_path


def test_load_dataset_reads_uploaded_csv():
    uploaded = io.StringIO("timestamp,value,label\n2024-01-01 00:00:00,1.5,0\n2024-01-01 00:00:01,2.0,1\n")
    uploaded.name = "sample.csv"

    frame = load_dataset(uploaded, nrows=2)

    assert list(frame.columns) == ["timestamp", "value", "label"]
    assert len(frame) == 2
    assert frame.iloc[0]["value"] == 1.5


def test_resolve_dataset_path_finds_processed_dataset(tmp_path):
    dataset = tmp_path / "data" / "processed" / "hai-test-scaled.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("timestamp,value\n2024-01-01 00:00:00,1.0\n", encoding="utf-8")

    resolved = resolve_dataset_path(tmp_path)

    assert resolved == dataset
