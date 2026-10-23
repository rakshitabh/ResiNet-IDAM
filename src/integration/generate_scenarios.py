"""
ResiNet-IDAM: Synthetic Cyber-Physical Scenario Generator
Explicitly creates benchmark evaluation scenarios by aligning IT network flows
with continuous 1 Hz OT sensor telemetry from the HAI testbed.

Scenarios:
- Scenario A: Correlated Attack (IT Precursor + OT Physical Impact)
- Scenario B: Stealth Infiltration (IT Infiltration + OT Physical Drift)
- Scenario C: Isolated IT Threat (IT DoS/Web Attack + Normal OT)
- Scenario D: Isolated OT Fault (Normal IT + OT Physical Anomaly)
- Normal: Normal IT + Normal OT

Validation and Test sets are split chronologically based on OT attack blocks
to guarantee strict out-of-sample isolation.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IT_DIR = ROOT / "data" / "processed" / "it"
OT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "processed" / "integration"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    print("Loading IT test data and metadata...")
    it_test = np.load(IT_DIR / "it_test.npz")
    X_it = it_test["X"]
    y_it = it_test["y"]

    with open(IT_DIR / "metadata.json", "r", encoding="utf-8") as f:
        it_meta = json.load(f)
    class_to_int = it_meta["class_to_integer"]
    int_to_class = {v: k for k, v in class_to_int.items()}

    # Index IT flows by class for scenario sampling
    class_indices = {c: np.where(y_it == c)[0] for c in range(len(class_to_int))}
    for c_int, c_name in int_to_class.items():
        print(f"  IT Class '{c_name}': {len(class_indices[c_int])} test flows available")

    print("\nLoading OT test telemetry...")
    ot_df = pd.read_csv(OT_DIR / "hai-test-scaled.csv")
    ot_df["timestamp"] = pd.to_datetime(ot_df["timestamp"])
    total_ot_records = len(ot_df)

    # Identify continuous attack blocks in OT
    ot_df["block"] = (ot_df["label"] != ot_df["label"].shift(1)).cumsum()
    attack_blocks = ot_df[ot_df["label"] == 1]["block"].unique().tolist()
    print(f"Total OT attack blocks identified: {len(attack_blocks)} blocks across {ot_df['label'].sum()} attack seconds.")

    # Chronological partition of attack blocks:
    # First 7 blocks -> Validation split (approx Aug 12 16:00 to Aug 12 23:00)
    # Last 7 blocks -> Test split (approx Aug 12 23:00 to Aug 13 07:00)
    val_blocks = set(attack_blocks[:7])
    test_blocks = set(attack_blocks[7:])
    split_cutoff_time = ot_df[ot_df["block"] == attack_blocks[7]]["timestamp"].min()
    print(f"Split cutoff timestamp: {split_cutoff_time}")
    print(f"  Validation OT attack blocks: {sorted(list(val_blocks))}")
    print(f"  Test OT attack blocks: {sorted(list(test_blocks))}")

    # Assign Scenario Types across the attack blocks
    # Blocks 0, 1 -> Scenario A (PortScan/BruteForce + OT)
    # Block 2 -> Scenario B (Infiltration + OT)
    # Block 3 -> Scenario D (Isolated OT, Normal IT)
    # Blocks 4, 5, 6 -> Scenario A
    # Blocks 7, 8 -> Scenario A
    # Block 9 -> Scenario B
    # Block 10 -> Scenario D
    # Blocks 11, 12, 13 -> Scenario A

    scenario_labels = []
    it_flow_indices = []
    it_assigned_classes = []
    zt_boundary = []
    zt_asset_crit = []
    zt_trust_score = []

    # Map each OT second to an aligned scenario
    for idx, row in ot_df.iterrows():
        is_ot_attack = (row["label"] == 1)
        blk = row["block"]

        if is_ot_attack:
            if blk in [attack_blocks[2], attack_blocks[9]]:
                # Scenario B: Stealth Infiltration
                scen = "Scenario_B_Stealth_Infil"
                chosen_it_class = class_to_int["Infiltration"]
                it_idx = np.random.choice(class_indices[chosen_it_class])
                bnd = np.random.choice([0, 1], p=[0.2, 0.8])
                crit = 0.95
                trust = 0.3
            elif blk in [attack_blocks[3], attack_blocks[10]]:
                # Scenario D: Isolated OT Fault (No IT precursor, normal IT network)
                scen = "Scenario_D_Isolated_OT"
                chosen_it_class = class_to_int["Normal"]
                it_idx = np.random.choice(class_indices[chosen_it_class])
                bnd = 0
                crit = 0.8
                trust = 0.9
            else:
                # Scenario A: Correlated Cyber-Physical Attack
                scen = "Scenario_A_Correlated"
                chosen_it_class = np.random.choice([
                    class_to_int["Brute Force"],
                    class_to_int["PortScan"],
                    class_to_int["Botnet ARES"],
                ])
                it_idx = np.random.choice(class_indices[chosen_it_class])
                bnd = 1
                crit = 0.9
                trust = 0.2
        else:
            # During normal OT intervals, randomly inject Scenario C (Isolated IT) in ~5% of rows
            # Otherwise Normal
            r = np.random.rand()
            if r < 0.04:
                # Scenario C: Isolated IT Threat (DoS or Web Attack, but physical plant normal)
                scen = "Scenario_C_Isolated_IT"
                chosen_it_class = np.random.choice([
                    class_to_int["Dos/DDos"],
                    class_to_int["Web Attack"]
                ])
                it_idx = np.random.choice(class_indices[chosen_it_class])
                bnd = np.random.choice([0, 1], p=[0.7, 0.3])
                crit = 0.4
                trust = 0.4
            else:
                scen = "Normal"
                chosen_it_class = class_to_int["Normal"]
                it_idx = np.random.choice(class_indices[chosen_it_class])
                bnd = 0
                crit = 0.3
                trust = 0.95

        scenario_labels.append(scen)
        it_flow_indices.append(it_idx)
        it_assigned_classes.append(chosen_it_class)
        zt_boundary.append(bnd)
        zt_asset_crit.append(crit)
        zt_trust_score.append(trust)

    # Compute Zero Trust Context metric: ZT = 0.4*boundary + 0.3*crit + 0.3*(1 - trust)
    zt_metric = (
        0.4 * np.array(zt_boundary) +
        0.3 * np.array(zt_asset_crit) +
        0.3 * (1.0 - np.array(zt_trust_score))
    )

    scenario_df = pd.DataFrame({
        "timestamp": ot_df["timestamp"],
        "ot_record_idx": ot_df.index,
        "ot_label": ot_df["label"],
        "it_flow_idx": it_flow_indices,
        "it_true_class": it_assigned_classes,
        "it_true_label_name": [int_to_class[c] for c in it_assigned_classes],
        "scenario": scenario_labels,
        "is_cyber_physical": [1 if s in ["Scenario_A_Correlated", "Scenario_B_Stealth_Infil"] else 0 for s in scenario_labels],
        "zt_boundary_crossing": zt_boundary,
        "zt_asset_criticality": zt_asset_crit,
        "zt_trust_score": zt_trust_score,
        "zt_context": zt_metric.round(4),
    })

    # Partition chronologically by cutoff
    val_df = scenario_df[scenario_df["timestamp"] < split_cutoff_time].copy()
    test_df = scenario_df[scenario_df["timestamp"] >= split_cutoff_time].copy()

    val_path = OUTPUT_DIR / "scenarios_val.csv"
    test_path = OUTPUT_DIR / "scenarios_test.csv"
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n[OK] Validation scenario partition saved: {len(val_df)} rows to {val_path.name}")
    print(val_df["scenario"].value_counts().to_string())

    print(f"\n[OK] Test scenario partition saved: {len(test_df)} rows to {test_path.name}")
    print(test_df["scenario"].value_counts().to_string())

    metadata = {
        "dataset_type": "Synthetic Cyber-Physical Evaluation Benchmark (Version 1.0)",
        "ot_source": "hai-test-scaled.csv",
        "it_source": "it_test.npz",
        "cutoff_timestamp": str(split_cutoff_time),
        "total_records": len(scenario_df),
        "validation_records": len(val_df),
        "test_records": len(test_df),
        "scenarios": {
            "Scenario_A_Correlated": "IT network precursor followed by OT setpoint manipulation",
            "Scenario_B_Stealth_Infil": "IT Infiltration flow paired with physical process drift",
            "Scenario_C_Isolated_IT": "IT DoS/Web Attack with normal physical plant operation",
            "Scenario_D_Isolated_OT": "Physical sensor/actuator anomaly with normal IT traffic",
            "Normal": "Normal IT network flow and normal OT process",
        }
    }
    with open(OUTPUT_DIR / "synthetic_scenarios_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n[OK] Scenario metadata saved to synthetic_scenarios_metadata.json")

if __name__ == "__main__":
    main()
