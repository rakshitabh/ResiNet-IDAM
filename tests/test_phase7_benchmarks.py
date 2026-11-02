"""Unit and Integration Tests for Phase 7: Resource Benchmarks, Ablation, and Packaging.

Tests:
1. test_resource_benchmark_report_schema: Verifies resource efficiency benchmark schema and metrics.
2. test_full_ablation_report_schema: Validates E0->E5 ablation JSON and CSV consistency.
3. test_deployment_artifacts_present: Confirms Dockerfile, docker-compose.yml, and deployment docs exist.
4. test_ablation_fusion_superiority: Asserts that OT model fusion (E3) achieves superior F1 and reduced FPR over baseline E0.
5. test_memory_footprint_within_edge_budget: Validates that measured peak RAM is under the 4 GB gateway ceiling.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"


def test_resource_benchmark_report_schema():
    json_path = REPORTS_DIR / "resource_efficiency_benchmark.json"
    csv_path = REPORTS_DIR / "resource_efficiency_benchmark.csv"

    assert json_path.exists(), "resource_efficiency_benchmark.json is missing"
    assert csv_path.exists(), "resource_efficiency_benchmark.csv is missing"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "hardware_environment" in data
    assert "benchmark_records" in data
    records = data["benchmark_records"]
    assert len(records) >= 8

    # Check key columns in CSV
    df = pd.read_csv(csv_path)
    required_cols = {"Domain", "Model / Component", "Batch Size", "Median Latency (ms)", "Throughput (samples/s)", "Peak RAM (MB)"}
    assert required_cols.issubset(df.columns)


def test_full_ablation_report_schema():
    json_path = REPORTS_DIR / "full_ablation_e0_to_e5.json"
    csv_path = REPORTS_DIR / "full_ablation_e0_to_e5.csv"

    assert json_path.exists(), "full_ablation_e0_to_e5.json is missing"
    assert csv_path.exists(), "full_ablation_e0_to_e5.csv is missing"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "results" in data
    results = data["results"]
    assert len(results) >= 6

    # Verify presence of all experiments E0 to E5
    exp_ids = {r["Experiment"] for r in results}
    for e in ("E0", "E1", "E2", "E3", "E4", "E5"):
        assert e in exp_ids, f"Experiment {e} missing from ablation report"


def test_deployment_artifacts_present():
    dockerfile = ROOT / "Dockerfile"
    compose = ROOT / "docker-compose.yml"
    deploy_doc = ROOT / "docs" / "deployment.md"

    assert dockerfile.exists(), "Dockerfile is missing"
    assert compose.exists(), "docker-compose.yml is missing"
    assert deploy_doc.exists(), "docs/deployment.md is missing"

    # Verify Dockerfile contains non-root user
    df_content = dockerfile.read_text(encoding="utf-8")
    assert "useradd" in df_content or "USER" in df_content

    # Verify compose specifies port 8501 and 8000
    compose_content = compose.read_text(encoding="utf-8")
    assert "8501:8501" in compose_content
    assert "8000:8000" in compose_content


def test_ablation_fusion_superiority():
    csv_path = REPORTS_DIR / "full_ablation_e0_to_e5.csv"
    df = pd.read_csv(csv_path)

    # In OT, E3 (Model Fusion) must have higher F1 than E0 (Baseline)
    ot_df = df[df["Domain"] == "OT"].set_index("Experiment")
    e0_f1 = float(ot_df.loc["E0", "F1-Score"].rstrip("%"))
    e3_f1 = float(ot_df.loc["E3", "F1-Score"].rstrip("%"))
    assert e3_f1 > e0_f1, f"Expected E3 F1 ({e3_f1}%) > E0 F1 ({e0_f1}%)"

    e0_fpr = float(ot_df.loc["E0", "FPR"].rstrip("%"))
    e3_fpr = float(ot_df.loc["E3", "FPR"].rstrip("%"))
    assert e3_fpr < e0_fpr, f"Expected E3 FPR ({e3_fpr}%) < E0 FPR ({e0_fpr}%)"


def test_memory_footprint_within_edge_budget():
    csv_path = REPORTS_DIR / "resource_efficiency_benchmark.csv"
    df = pd.read_csv(csv_path)

    # All components must consume < 500 MB RAM (well below 4 GB Raspberry Pi ceiling)
    for _, row in df.iterrows():
        assert row["Peak RAM (MB)"] < 500.0, f"Component {row['Model / Component']} exceeded 500 MB memory limit"

