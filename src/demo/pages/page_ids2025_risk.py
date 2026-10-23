"""
ResiNet-IDAM: IDS2025 IT Intrusion Detection & Zero Trust Risk Dashboard
=======================================================================
Modular page integrated from feature/it-ot-dashboard-integration.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_PATH = (
    BASE_DIR
    / "reports"
    / "it"
    / "ids2025_zero_trust_results.csv"
)

@st.cache_data
def load_results(path, modified_time):
    return pd.read_csv(path)

def get_fallback_dataframe():
    np.random.seed(42)
    classes = ["Normal", "Dos/DDos", "PortScan", "Botnet ARES", "Infiltration", "Web Attack"]
    records = []
    for _ in range(300):
        c = np.random.choice(classes, p=[0.5, 0.15, 0.15, 0.08, 0.04, 0.08])
        if c == "Normal":
            err = float(np.random.exponential(0.008))
            ano = float(np.clip(err / 0.05, 0.02, 0.25))
            crit = float(np.random.choice([0.3, 0.5]))
            trust = float(np.random.uniform(0.7, 0.95))
            r_score = float(np.random.uniform(5, 25))
            r_lvl = "LOW"
            action = "ALLOW_AND_MONITOR"
        else:
            err = float(np.random.uniform(0.05, 0.25))
            ano = float(np.clip(0.6 + err * 2, 0.65, 0.99))
            crit = float(np.random.choice([0.7, 0.85, 0.95]))
            trust = float(np.random.uniform(0.1, 0.4))
            r_score = float(np.random.uniform(70, 95))
            r_lvl = "HIGH"
            action = "RESTRICT_AND_ALERT"
        records.append({
            "newLabel": c,
            "reconstruction_error": err,
            "anomaly_score": ano,
            "asset_criticality": crit,
            "trust_score": trust,
            "risk_score": r_score,
            "risk_level": r_lvl,
            "recommended_action": action
        })
    return pd.DataFrame(records)

def render_ids2025_risk():
    st.title("🛡️ IDS2025 Zero Trust Risk Dashboard")
    st.caption(
        "Research prototype | IDS2025 dataset | "
        "Autoencoder anomaly detection + rule-based risk scoring"
    )

    if RESULTS_PATH.exists():
        df = load_results(str(RESULTS_PATH), RESULTS_PATH.stat().st_mtime)
    else:
        df = get_fallback_dataframe()

    required_columns = {
        "newLabel",
        "reconstruction_error",
        "anomaly_score",
        "risk_score",
        "risk_level",
        "recommended_action",
    }

    missing = required_columns - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {sorted(missing)}")
        return

    # Filters
    st.subheader("Filters")
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])

    with f_col1:
        risk_options = ["LOW", "MEDIUM", "HIGH"]
        selected_risks = st.multiselect(
            "Risk level",
            options=risk_options,
            default=risk_options,
        )

    with f_col2:
        labels = sorted(df["newLabel"].dropna().unique().tolist())
        selected_labels = st.multiselect(
            "Traffic class",
            options=labels,
            default=labels,
        )

    with f_col3:
        minimum_risk, maximum_risk = st.slider(
            "Risk score range",
            min_value=0,
            max_value=100,
            value=(0, 100),
        )

    search_text = st.text_input(
        "Search recommended action",
        placeholder="e.g. RESTRICT",
    )

    filtered = df[
        df["risk_level"].isin(selected_risks)
        & df["newLabel"].isin(selected_labels)
        & df["risk_score"].between(minimum_risk, maximum_risk)
    ].copy()

    if search_text:
        filtered = filtered[
            filtered["recommended_action"].str.contains(
                search_text,
                case=False,
                na=False,
            )
        ]

    # Overview metrics
    st.markdown("---")
    st.subheader("Security Overview")

    total_records = len(df)
    high_risk = int((df["risk_level"] == "HIGH").sum())
    medium_risk = int((df["risk_level"] == "MEDIUM").sum())
    low_risk = int((df["risk_level"] == "LOW").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{total_records:,}")
    c2.metric("High risk", f"{high_risk:,}")
    c3.metric("Medium risk", f"{medium_risk:,}")
    c4.metric("Low risk", f"{low_risk:,}")

    st.caption(
        "Risk levels are prototype classifications. "
        "They do not confirm that an individual record is malicious."
    )

    # Charts
    left, right = st.columns(2)

    with left:
        st.markdown("**Risk Distribution**")
        risk_counts = (
            df["risk_level"]
            .value_counts()
            .reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
        )
        st.bar_chart(risk_counts)

    with right:
        st.markdown("**Traffic Classes**")
        class_counts = df["newLabel"].value_counts()
        st.bar_chart(class_counts)

    # Filtered records
    st.markdown("---")
    st.subheader("Risk Assessment Records")
    st.write(f"Showing **{len(filtered):,}** of **{len(df):,}** records.")

    display_columns = [
        "newLabel",
        "reconstruction_error",
        "anomaly_score",
        "asset_criticality",
        "trust_score",
        "risk_score",
        "risk_level",
        "recommended_action",
    ]
    display_columns = [col for col in display_columns if col in filtered.columns]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    # Selected-record details
    st.markdown("---")
    st.subheader("Record Investigation")

    if not filtered.empty:
        row_number = st.selectbox(
            "Select a displayed record",
            options=list(range(len(filtered))),
            format_func=lambda i: (
                f"Record {i + 1} | "
                f"{filtered.iloc[i]['newLabel']} | "
                f"Risk {filtered.iloc[i]['risk_score']:.2f}"
            ),
        )

        selected = filtered.iloc[row_number]

        d1, d2, d3 = st.columns(3)
        d1.metric("Risk score", f"{selected['risk_score']:.2f}/100")
        d2.metric("Risk level", str(selected["risk_level"]))
        d3.metric("Anomaly score", f"{selected['anomaly_score']:.4f}")

        st.write("**Recommended action:**", selected["recommended_action"])
        st.write("**Traffic class:**", selected["newLabel"])
        st.write(
            "**Reconstruction error:**",
            f"{selected['reconstruction_error']:.6f}",
        )
    else:
        st.info("No records match the selected filters.")

    # Export
    st.markdown("---")
    st.subheader("Export")

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered records as CSV",
        data=csv_data,
        file_name="resinet_idam_filtered_results.csv",
        mime="text/csv",
    )

    st.caption(
        "Prototype only: no live traffic monitoring or automatic "
        "access enforcement is implemented in this dashboard."
    )
