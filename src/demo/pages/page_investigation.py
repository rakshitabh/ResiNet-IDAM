"""
ResiNet-IDAM: 06 Alert Investigation Page
----------------------------------------
Inspect individual predictions across IT, OT, and Cyber-Physical domains.
Supports reviewer diagnostic drill-downs and SQLite investigation history logging.
"""

from pathlib import Path
import json
import streamlit as st
import numpy as np
import pandas as pd
from src.demo.db import add_investigation, get_investigations

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
INTEG_DIR = DATA_DIR / "integration"
IT_DIR = DATA_DIR / "it_realistic"

@st.cache_data
def load_investigation_data():
    df_scen = pd.read_csv(INTEG_DIR / "scenarios_test.csv")
    scores = np.load(INTEG_DIR / "scores_test.npz")
    
    df_scen["p_attack"] = scores["p_attack"]
    df_scen["S_ot"] = scores["S_ot"]
    df_scen["ot_mse"] = scores["ot_mse"]
    df_scen["ot_label"] = scores["ot_label"]
    df_scen["is_cyber_physical"] = scores["is_cyber_physical"]

    return df_scen

def render_investigation():
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; margin-bottom: 4px;">06 Alert Investigation: Inspect Individual Predictions</h2>
            <p style="color: #94a3b8; font-size: 14px;">Drill down into individual telemetry flows, diagnose false alarms or missed attacks, and log reviewer notes into SQLite.</p>
        </div>
    """, unsafe_allow_html=True)

    df_scen = load_investigation_data()

    tab_inspect, tab_history = st.tabs(["🔍 Record Investigation & Diagnostics", "📜 Reviewer Investigation History (SQLite)"])

    with tab_inspect:
        col_f1, col_f2, col_f3 = st.columns([1.2, 1.2, 1.5])

        with col_f1:
            domain_filter = st.selectbox("Filter Domain:", ["Cross-Domain Scenarios", "IT Network Flows", "OT Physical Faults"])

        with col_f2:
            outcome_filter = st.selectbox(
                "Filter Prediction Status:",
                ["All Records", "Missed Attacks (False Negatives)", "False Alarms (False Positives)", "Confirmed Detections (True Positives)"]
            )

        # Apply filtering
        filtered_df = df_scen.copy()

        if domain_filter == "IT Network Flows":
            filtered_df = filtered_df[filtered_df["it_true_label_name"] != "Normal"]
        elif domain_filter == "OT Physical Faults":
            filtered_df = filtered_df[filtered_df["ot_label"] == 1]

        if outcome_filter == "Missed Attacks (False Negatives)":
            # True attack but low risk
            filtered_df = filtered_df[(filtered_df["ot_label"] == 1) & (filtered_df["S_ot"] < 0.50)]
        elif outcome_filter == "False Alarms (False Positives)":
            # Normal but high score
            filtered_df = filtered_df[(filtered_df["scenario"] == "Normal") & ((filtered_df["p_attack"] >= 0.50) | (filtered_df["S_ot"] >= 0.50))]
        elif outcome_filter == "Confirmed Detections (True Positives)":
            filtered_df = filtered_df[(filtered_df["scenario"] != "Normal") & ((filtered_df["p_attack"] >= 0.50) | (filtered_df["S_ot"] >= 0.50))]

        with col_f3:
            st.markdown(f"**Found {len(filtered_df):,} matching records**")
            selected_idx = st.selectbox(
                "Select Record to Deep-Dive:",
                filtered_df.index[:150],
                format_func=lambda idx: f"Record #{idx} | {df_scen.loc[idx, 'scenario']} | IT: {df_scen.loc[idx, 'it_true_label_name']}"
            )

        # Deep-Dive Record Inspection
        rec = df_scen.loc[selected_idx]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Detailed Telemetry Record Profile")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Record Index", f"#{selected_idx}")
        with r2:
            st.metric("Scenario Category", rec["scenario"])
        with r3:
            st.metric("IT Attack Flow", rec["it_true_label_name"])
        with r4:
            ot_status = "PHYSICAL ATTACK" if rec["ot_label"] == 1 else "NORMAL PROCESS"
            st.metric("OT Sensor Ground Truth", ot_status)

        st.markdown("<br>", unsafe_allow_html=True)

        col_scores, col_diag = st.columns([1, 1.2])

        with col_scores:
            st.markdown("""
                <div style="background: #111a24; border: 1px solid #1e293b; border-radius: 8px; padding: 18px;">
                    <h4 style="color: #00e5ff; margin-top: 0;">Multi-Channel Confidence Scores</h4>
            """, unsafe_allow_html=True)
            st.write(f"• **IT Network Attack Probability (p_attack):** `{rec['p_attack']:.4f}`")
            st.write(f"• **OT Autoencoder Error (MSE):** `{rec['ot_mse']:.6f}`")
            st.write(f"• **OT Normalized Anomaly Score (S_ot):** `{rec['S_ot']:.4f}`")
            
            # Reconstruction threshold indicator
            if rec["ot_mse"] >= 0.821954:
                st.markdown("<span style='color: #10b981; font-weight: 600;'>[OK] OT Anomaly Threshold (θ = 0.822) Tripped</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: #ef4444; font-weight: 600;'>[!] OT Anomaly Below Threshold (θ = 0.822)</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_diag:
            st.markdown("""
                <div style="background: #111a24; border: 1px solid #1e293b; border-radius: 8px; padding: 18px;">
                    <h4 style="color: #f59e0b; margin-top: 0;">Automated Diagnostic Assessment</h4>
            """, unsafe_allow_html=True)
            
            if rec["scenario"] == "Scenario_D_Isolated_OT" and rec["ot_mse"] < 0.821954:
                st.write("🔍 **Diagnostic:** This record represents one of the **122 missed attack seconds** in Scenario D. Sensor tampering occurred, but process deviations stayed within normal baseline tolerances.")
            elif rec["it_true_label_name"] == "Botnet ARES" and rec["p_attack"] < 0.50:
                st.write("🔍 **Diagnostic:** Botnet ARES C2 beaconing flow. Low packet count and standard TCP flags mimic legitimate HTTP/DNS keepalives.")
            elif rec["scenario"] == "Scenario_A_Correlated":
                st.write("🔍 **Diagnostic:** Coordinated attack detected across both network intrusion signals and physical process fluctuations.")
            elif rec["scenario"] == "Normal" and (rec["p_attack"] >= 0.50 or rec["S_ot"] >= 0.50):
                st.write("🔍 **Diagnostic:** False positive alarm. A transient network burst or physical hydraulic wave exceeded baseline thresholds during benign operation.")
            else:
                st.write("🔍 **Diagnostic:** Standard classified transmission in compliance with model operating bands.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Reviewer SQLite Audit Log Form
        st.markdown("#### 📝 Record Reviewer Audit Note (SQLite)")
        with st.form("audit_form"):
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                verdict = st.selectbox(
                    "Investigation Verdict:",
                    ["Confirmed True Attack", "Benign False Positive", "Known Model Limitation (Missed Attack)", "Requires Testbed Validation"]
                )
            with col_v2:
                notes = st.text_input("Reviewer Notes / Action Item:", placeholder="e.g., DNS keepalive overlap; setpoint deviation below 0.822 threshold.")

            submit_btn = st.form_submit_button("💾 Save Investigation Audit Note")
            if submit_btn:
                add_investigation(
                    domain=domain_filter,
                    record_id=selected_idx,
                    true_label=f"{rec['scenario']} | {rec['it_true_label_name']}",
                    pred_label=f"p_att={rec['p_attack']:.2f}, S_ot={rec['S_ot']:.2f}",
                    verdict=verdict,
                    notes=notes,
                    confidence=rec["p_attack"]
                )
                st.success(f"Audit note for Record #{selected_idx} saved to SQLite investigation database!")

    with tab_history:
        st.markdown("#### Saved Investigation History (SQLite: `data/investigation_history.db`)")
        inv_rows = get_investigations()
        if len(inv_rows) == 0:
            st.info("No investigation notes logged yet. Use the 'Record Investigation' tab above to log your findings.")
        else:
            df_hist = pd.DataFrame(inv_rows, columns=[
                "ID", "Timestamp", "Domain", "Record ID", "Ground Truth", "Model Scores", "Verdict", "Reviewer Notes", "Confidence"
            ])
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
