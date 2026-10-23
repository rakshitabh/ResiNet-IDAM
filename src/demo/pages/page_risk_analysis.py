"""
ResiNet-IDAM: 04 Risk Analysis Page
-----------------------------------
Multi-channel cyber-physical threat correlation and triage across benchmark scenarios.
Demonstrates non-shortcut fusion and isolated OT dispatch in Scenario D.
"""

from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib

ROOT = Path(__file__).resolve().parents[3]
INTEG_DIR = ROOT / "data" / "processed" / "integration"
MODEL_PATH = ROOT / "models" / "fusion_model.joblib"

@st.cache_data
def load_integration_data():
    df_scen = pd.read_csv(INTEG_DIR / "scenarios_test.csv")
    scores = np.load(INTEG_DIR / "scores_test.npz")
    
    p_att = scores["p_attack"]
    S_ot = scores["S_ot"]
    
    # Predict R_CP using non-shortcut logistic regression model
    fusion_model = joblib.load(MODEL_PATH)
    X = np.column_stack([p_att, S_ot, p_att * S_ot])
    R_cp = fusion_model.predict_proba(X)[:, 1]

    # Assign multi-channel triage tier
    tiers = []
    for p, s, r in zip(p_att, S_ot, R_cp):
        if r >= 0.50:
            tiers.append("Tier 4: Critical CP Alert")
        elif s >= 0.50:
            tiers.append("Tier 2: Plant Engineering")
        elif p >= 0.50:
            tiers.append("Tier 2: Enterprise SOC")
        else:
            tiers.append("Normal / Filtered")

    df_scen["p_attack"] = p_att
    df_scen["S_ot"] = S_ot
    df_scen["R_cp"] = R_cp
    df_scen["triage_tier"] = tiers

    return df_scen

def render_risk_analysis():
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; margin-bottom: 4px;">04 Risk Analysis: IT-OT Correlation & Multi-Channel Triage</h2>
            <p style="color: #94a3b8; font-size: 14px;">Replays cyber-physical threat scenarios through the verified non-shortcut correlation model and multi-channel triage router.</p>
        </div>
    """, unsafe_allow_html=True)

    df_scen = load_integration_data()

    # Scenario Selection
    col_sel, col_desc = st.columns([1.5, 2])
    with col_sel:
        scenarios = [
            "All Scenarios",
            "Scenario_A_Correlated",
            "Scenario_B_Stealth_Infil",
            "Scenario_C_Isolated_IT",
            "Scenario_D_Isolated_OT",
            "Normal"
        ]
        scenario_choice = st.selectbox("Select Scenario to Inspect:", scenarios, index=1)

    with col_desc:
        if scenario_choice == "Scenario_A_Correlated":
            st.info("🔴 **Scenario A (1,069 records):** Multi-stage coordinated attack. Network intrusion coincides with physical valve manipulation.")
        elif scenario_choice == "Scenario_B_Stealth_Infil":
            st.info("🟣 **Scenario B (56 records):** Stealth infiltration threat. Low-volume privilege escalation with subtle physical sensor drift.")
        elif scenario_choice == "Scenario_C_Isolated_IT":
            st.info("🔵 **Scenario C (984 records):** Isolated IT network flood (DoS/PortScan) with zero physical OT impact.")
        elif scenario_choice == "Scenario_D_Isolated_OT":
            st.warning("🟡 **Scenario D (132 records):** Direct physical sensor tampering without any IT network precursor.")
        elif scenario_choice == "Normal":
            st.success("🟢 **Normal Baseline (24,445 records):** Standard background operational traffic.")

    # Filter data for display
    if scenario_choice == "All Scenarios":
        sub_df = df_scen.iloc[:2000]
    else:
        sub_df = df_scen[df_scen["scenario"] == scenario_choice]

    # Metric Row for Selected Scenario
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", f"{len(sub_df):,}")
    with c2:
        t4_count = (sub_df["triage_tier"] == "Tier 4: Critical CP Alert").sum()
        st.metric("Tier 4 Critical CP", f"{t4_count:,}", f"{(t4_count/len(sub_df)*100):.1f}%" if len(sub_df)>0 else "")
    with c3:
        t2_ot_count = (sub_df["triage_tier"] == "Tier 2: Plant Engineering").sum()
        st.metric("Tier 2 Plant Eng", f"{t2_ot_count:,}", f"{(t2_ot_count/len(sub_df)*100):.1f}%" if len(sub_df)>0 else "")
    with c4:
        t2_it_count = (sub_df["triage_tier"] == "Tier 2: Enterprise SOC").sum()
        st.metric("Tier 2 SOC", f"{t2_it_count:,}", f"{(t2_it_count/len(sub_df)*100):.1f}%" if len(sub_df)>0 else "")

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Risk Curve Timeline (Plotly)
    st.markdown("#### 📈 Multi-Channel Risk Timeline")
    st.caption("Displays the three independent signals: IT attack probability p_attack, OT reconstruction score S_ot, and cyber-physical correlation R_cp:")

    plot_window = min(len(sub_df), 400)
    window_df = sub_df.iloc[:plot_window]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(window_df)),
        y=window_df["p_attack"],
        name="IT Network Risk (p_attack)",
        line=dict(color="#00e5ff", width=1.8)
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(window_df)),
        y=window_df["S_ot"],
        name="OT Anomaly Score (S_ot)",
        line=dict(color="#38bdf8", width=1.8)
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(window_df)),
        y=window_df["R_cp"],
        name="Cyber-Physical Risk (R_cp)",
        line=dict(color="#c084fc", width=2.5)
    ))
    # Threshold Line at 0.50
    fig.add_hline(y=0.50, line_dash="dash", line_color="#ef4444", annotation_text="Triage Threshold (0.50)", annotation_position="top left")

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        yaxis_range=[-0.05, 1.05],
        xaxis_title="Replay Sequence Index (Seconds)",
        yaxis_title="Normalized Score &isin; [0, 1]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Scenario D Audit Callout
    if scenario_choice in ["Scenario_D_Isolated_OT", "All Scenarios"]:
        st.markdown("### 🛠️ Scenario D Detailed Engineering Audit")
        col_s1, col_s2 = st.columns([1.5, 1])
        with col_s1:
            st.markdown("""
                <div style="background: #111a24; border: 1px solid #1e293b; border-radius: 8px; padding: 18px;">
                    <h4 style="color: #f59e0b; margin-top: 0;">Multi-Channel Resolution for Isolated OT Physical Faults</h4>
                    <p style="color: #cbd5e1; font-size: 13px; line-height: 1.6;">
                        In Scenario D, an adversary tampers with physical OT sensor setpoints on the plant floor without generating IT network traffic.
                        <br><br>
                        • <b>Previous Prototype Flaw:</b> The previous single-model architecture enforced joint IT-OT corroboration, resulting in <b>100% false negatives (0 / 132 caught)</b>.
                        <br><br>
                        • <b>Multi-Channel Routing:</b> In ResiNet-IDAM, Tier 2 Plant Engineering alerts are decoupled from IT signals. When an OT reconstruction error spikes, it dispatches directly to plant engineering.
                        <br><br>
                        • <b>Missed Attack Ground Truth:</b> Out of 132 attack seconds, 10 exceeded the Autoencoder threshold (7.6%) and 122 remained below threshold. All 10 detected instances were dispatched to plant operators without suppression.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        with col_s2:
            scen_d_triage = df_scen[df_scen["scenario"] == "Scenario_D_Isolated_OT"]["triage_tier"].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=scen_d_triage.index,
                values=scen_d_triage.values,
                hole=0.45,
                marker=dict(colors=["#1e293b", "#f59e0b", "#3b82f6"])
            )])
            fig_pie.update_layout(
                title="Scenario D Triage Distribution",
                height=260,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8")
            )
            st.plotly_chart(fig_pie, use_container_width=True)
