"""
ResiNet-IDAM: 03 OT Monitoring Page
---------------------------------
Industrial physical telemetry monitoring, Autoencoder reconstruction error analysis,
and honest diagnostic of the 1,598 missed physical attacks.
"""

from pathlib import Path
import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
INTEG_DIR = DATA_DIR / "integration"
THRESHOLD_FILE = ROOT / "models" / "autoencoder_threshold.json"

@st.cache_data
def load_ot_sample_data():
    # Load threshold metadata
    threshold_meta = json.load(open(THRESHOLD_FILE))
    theta = float(threshold_meta["threshold"])

    # Load 5,000 consecutive rows from hai-test-scaled around attack block 2 & 3 for smooth interactive rendering
    df_raw = pd.read_csv(DATA_DIR / "hai-test-scaled.csv", nrows=6000)
    
    # Load calibrated test scores for integration
    scores_data = np.load(INTEG_DIR / "scores_test.npz")
    ot_mse_seq = scores_data["ot_mse"][:6000]
    S_ot_seq = scores_data["S_ot"][:6000]
    ot_labels = scores_data["ot_label"][:6000]

    return df_raw, theta, ot_mse_seq, S_ot_seq, ot_labels

def render_ot_monitoring():
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; margin-bottom: 4px;">03 OT Monitoring: Industrial Telemetry Anomalies</h2>
            <p style="color: #94a3b8; font-size: 14px;">Replays physical process dynamics from the HAI 21.03 water treatment testbed (66 channels @ 1 Hz) and tracks Autoencoder reconstruction deviations.</p>
        </div>
    """, unsafe_allow_html=True)

    df_raw, theta, ot_mse, S_ot, ot_labels = load_ot_sample_data()

    # High-level OT Status Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Autoencoder Threshold (θ)", f"{theta:.4f}", help="Calibrated at 99th percentile of validation reconstruction error")
    with c2:
        st.metric("Attack Recall", "46.39%", delta="-53.61% missed", delta_color="inverse")
    with c3:
        st.metric("Attack Precision", "34.92%", help="34.92% of threshold alarms correspond to true attacks")
    with c4:
        st.metric("Missed Attack Seconds", "1,598 s", delta="Silent Physical Deviations", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # Time Window & Sensor Channel Controls
    col_sensor, col_window = st.columns([1.5, 1])
    
    available_channels = [c for c in df_raw.columns if c not in ["timestamp", "label"]]
    
    # Common informative channels in HAI
    preferred_channels = ["P1_FT01", "P1_FT02", "P1_LCV01D", "P1_LIT01", "P1_PCV01D", "P2_VTR01"]
    default_sensor = "P1_LIT01" if "P1_LIT01" in available_channels else available_channels[0]

    with col_sensor:
        sensor_choice = st.selectbox(
            "Select Physical Sensor / Actuator Channel:",
            available_channels,
            index=available_channels.index(default_sensor) if default_sensor in available_channels else 0
        )

    with col_window:
        window_range = st.slider(
            "Time Window (Seconds Replay):",
            min_value=0,
            max_value=min(len(df_raw), len(ot_mse)),
            value=(500, 3500),
            step=100
        )

    start_idx, end_idx = window_range
    sub_df = df_raw.iloc[start_idx:end_idx]
    sub_mse = ot_mse[start_idx:end_idx]
    sub_labels = df_raw["label"].iloc[start_idx:end_idx].values

    # Interactive Dual-Axis Plotly Chart: Telemetry vs Reconstruction Error
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[
            f"Physical Process Telemetry: {sensor_choice} (Normalized z-score)",
            f"Deep Autoencoder Reconstruction Error (MSE) vs Threshold (θ = {theta:.4f})"
        ]
    )

    time_axis = np.arange(start_idx, end_idx)

    # 1. Telemetry curve
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=sub_df[sensor_choice],
            name=f"{sensor_choice} Value",
            line=dict(color="#38bdf8", width=1.5)
        ),
        row=1, col=1
    )

    # 2. Reconstruction MSE curve
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=sub_mse,
            name="Reconstruction MSE",
            line=dict(color="#c084fc", width=1.5)
        ),
        row=2, col=1
    )

    # Threshold horizontal line
    fig.add_trace(
        go.Scatter(
            x=[start_idx, end_idx],
            y=[theta, theta],
            mode="lines",
            name=f"Threshold (θ = {theta:.4f})",
            line=dict(color="#ef4444", width=2, dash="dash")
        ),
        row=2, col=1
    )

    # Highlight Attack Intervals in Red
    attack_starts = []
    attack_ends = []
    in_attack = False

    for i, val in enumerate(sub_labels):
        t = start_idx + i
        if val == 1 and not in_attack:
            attack_starts.append(t)
            in_attack = True
        elif val == 0 and in_attack:
            attack_ends.append(t)
            in_attack = False
    if in_attack:
        attack_ends.append(end_idx)

    for a_start, a_end in zip(attack_starts, attack_ends):
        for r in [1, 2]:
            fig.add_vrect(
                x0=a_start, x1=a_end,
                fillcolor="#ef4444", opacity=0.22,
                layer="below", line_width=1, line_color="#ef4444",
                annotation_text="ATTACK" if r == 1 else "",
                annotation_position="top left",
                annotation_font=dict(color="#ef4444", size=10),
                row=r, col=1
            )

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Replay Time Step (Seconds)", row=2, col=1, gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")

    st.plotly_chart(fig, use_container_width=True)

    # Core Scientific Takeaways & Diagnostics
    st.markdown("### 🔬 Diagnostic Analysis: The 1,598 Missed Physical Attacks")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("""
            <div style="background: #111a24; border: 1px solid #1e293b; border-radius: 8px; padding: 18px;">
                <h4 style="color: #f59e0b; margin-top: 0;">Why do Physical Attacks Slip Past Autoencoders?</h4>
                <ul style="color: #cbd5e1; font-size: 13px; line-height: 1.6; padding-left: 20px;">
                    <li><b>Operational Tolerance Bands:</b> In industrial control systems (ICS), actuators and water pumps operate over wide allowable dynamic ranges. Stealthy attacks intentionally spoof sensor setpoints slowly, staying within the normal distribution.</li>
                    <li><b>Threshold Trade-off:</b> Setting θ = 0.822 (99th percentile) suppresses false alarms to 5.0%, but causes the Autoencoder to miss <b>53.61% of attack seconds</b>. Lowering θ would increase recall, but overwhelm plant engineers with false alarms.</li>
                    <li><b>Temporal Inertia:</b> Large fluid systems have significant thermal and hydraulic inertia. Physical anomalies may take minutes to reflect in sensor deviations.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown("""
            <div style="background: #111a24; border: 1px solid #1e293b; border-radius: 8px; padding: 18px;">
                <h4 style="color: #00e5ff; margin-top: 0;">Scenario D Resolution: Multi-Channel Triage</h4>
                <ul style="color: #cbd5e1; font-size: 13px; line-height: 1.6; padding-left: 20px;">
                    <li><b>The Single-Model Flaw:</b> Previous single-decision fusion models required simultaneous IT network alerts, resulting in <b>0% detection for isolated OT sensor tampering</b> (132/132 attacks missed).</li>
                    <li><b>Multi-Channel Architecture:</b> ResiNet-IDAM separates IT and OT alarms. When an OT sensor deviates without an IT precursor, it dispatches directly as a <b>Tier 2 Plant Engineering Alert</b>.</li>
                    <li><b>Ground Truth Realism:</b> Of the 132 seconds of Scenario D attack, 10 were caught by the threshold (7.6%) and 122 were missed due to low reconstruction error. All 10 caught attacks were successfully triaged without suppression.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
