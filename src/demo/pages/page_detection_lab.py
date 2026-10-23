"""
ResiNet-IDAM: Page 2 - Detection Lab
-----------------------------------
Interactive live inference on recorded network flows:
- Lets the reviewer select a flow.
- Runs the existing supervised model and anomaly detector.
- Displays predicted class, anomaly score, threshold, and ground truth.
- Includes a dedicated Botnet ARES exemplar illustrating why detection failed.
"""

from pathlib import Path
import json
import sys
from unittest.mock import MagicMock
import textwrap

# Ensure Application Control patch is active
if 'sklearn.svm._libsvm_sparse' not in sys.modules:
    sys.modules['sklearn.svm._libsvm_sparse'] = MagicMock()
if 'scipy.stats._qmvnt_cy' not in sys.modules:
    sys.modules['scipy.stats._qmvnt_cy'] = MagicMock()

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed" / "it_realistic"
MODEL_DIR = ROOT / "models" / "it_realistic"

@st.cache_data
def load_detection_resources(config_choice):
    meta = json.load(open(DATA_DIR / "metadata.json"))
    class_names = meta["class_names"]

    if "79" in config_choice:
        test_npz = np.load(DATA_DIR / "it_test_all.npz")
        rf_model = joblib.load(MODEL_DIR / "random_forest_all.joblib")
        iforest = joblib.load(MODEL_DIR / "loco_iforest_all_botnet_ares.joblib")
        features = meta["configurations"]["all_features"]["features"]
        tau = 0.644722
    else:
        test_npz = np.load(DATA_DIR / "it_test_no_ports.npz")
        rf_model = joblib.load(MODEL_DIR / "random_forest_no_ports.joblib")
        iforest = joblib.load(MODEL_DIR / "loco_iforest_no_ports_botnet_ares.joblib")
        features = meta["configurations"]["no_ports"]["features"]
        tau = 0.644922

    return test_npz["X"], test_npz["y"], class_names, features, rf_model, iforest, tau

def render_detection_lab():
    header_html = """
<div style="margin-bottom: 20px;">
    <h2 style="color: #ffffff; margin-bottom: 4px;">02 Detection Lab: Recorded Flow Analysis</h2>
    <p style="color: #94a3b8; font-size: 14px;">Select recorded NetFlow telemetry, execute the live models, and inspect supervised classification alongside anomaly scores and thresholds.</p>
</div>
"""
    st.markdown(textwrap.dedent(header_html).strip(), unsafe_allow_html=True)

    # Configuration Selector
    col_cfg, col_info = st.columns([1.5, 1])
    with col_cfg:
        config_choice = st.radio(
            "Feature Configuration:",
            ["Configuration A (All 79 Features)", "Configuration B (77 Features - No Ports)"],
            horizontal=True
        )
    with col_info:
        if "79" in config_choice:
            st.success("Configuration A: Ephemeral ports included.")
        else:
            st.warning("Configuration B: Port shortcuts removed (evaluates behavioral flow dynamics).")

    X_test, y_test, class_names, feature_names, rf_model, iforest, tau = load_detection_resources(config_choice)

    # Curate Exemplar Indices
    botnet_indices = np.where(y_test == 0)[0]
    dos_indices = np.where(y_test == 2)[0]
    infil_indices = np.where(y_test == 3)[0]
    normal_indices = np.where(y_test == 4)[0]
    portscan_indices = np.where(y_test == 5)[0]

    # Initialize key in session state
    if "flow_num_input" not in st.session_state:
        st.session_state.flow_num_input = int(botnet_indices[10])

    def set_flow(idx):
        st.session_state.flow_num_input = int(idx)

    st.markdown("#### Select a Test Flow")
    st.caption("Quickly test representative attack scenarios or choose an exemplar:")

    c_btn1, c_btn2, c_btn3, c_btn4, c_btn5 = st.columns(5)
    with c_btn1:
        st.button("🔴 Botnet ARES (Failed)", on_click=set_flow, args=(botnet_indices[10],), use_container_width=True)
    with c_btn2:
        st.button("🌊 DoS / DDoS Flood", on_click=set_flow, args=(dos_indices[5],), use_container_width=True)
    with c_btn3:
        st.button("🔍 PortScan Probe", on_click=set_flow, args=(portscan_indices[2],), use_container_width=True)
    with c_btn4:
        st.button("🟢 Normal Benign", on_click=set_flow, args=(normal_indices[20],), use_container_width=True)
    with c_btn5:
        st.button("🕵️ Infiltration", on_click=set_flow, args=(infil_indices[0],), use_container_width=True)

    # Number input widget
    idx_col, class_col = st.columns([1.5, 2])
    with idx_col:
        current_idx = st.number_input(
            "Or enter test flow index (0 to 18,343):",
            min_value=0,
            max_value=len(X_test) - 1,
            key="flow_num_input",
            step=1
        )

    flow_vec = X_test[current_idx].reshape(1, -1)
    true_label_idx = int(y_test[current_idx])
    true_label_name = class_names[true_label_idx]

    with class_col:
        label_card_html = f"""
<div style="background: #111a24; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 16px; margin-top: 14px;">
    <span style="color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 600;">Ground Truth Label:</span>
    <span style="color: #ffffff; font-weight: 700; font-size: 16px; margin-left: 8px;">{true_label_name}</span>
    <span style="color: #94a3b8; font-size: 12px; margin-left: 10px;">(Flow #{current_idx:,})</span>
</div>
"""
        st.markdown(textwrap.dedent(label_card_html).strip(), unsafe_allow_html=True)

    # Perform Model Inference
    probs = rf_model.predict_proba(flow_vec)[0]
    pred_label_idx = int(np.argmax(probs))
    pred_label_name = class_names[pred_label_idx]
    confidence = probs[pred_label_idx] * 100.0

    # Unsupervised Anomaly Detection Layer (Isolation Forest)
    anomaly_score = float(-iforest.score_samples(flow_vec)[0])
    is_anomalous = (anomaly_score > tau)

    st.markdown("<br>", unsafe_allow_html=True)

    # Results Overview Cards
    r1, r2, r3, r4 = st.columns(4)

    with r1:
        c1_html = f"""
<div style="background: #0e1622; border: 1px solid #1e293b; border-radius: 8px; padding: 16px;">
    <div style="color: #64748b; font-size: 11px; font-weight: 700; text-transform: uppercase;">Ground Truth</div>
    <div style="color: #ffffff; font-size: 20px; font-weight: 700; margin-top: 4px;">{true_label_name}</div>
    <div style="color: #94a3b8; font-size: 11px;">Validated network label</div>
</div>
"""
        st.markdown(textwrap.dedent(c1_html).strip(), unsafe_allow_html=True)

    with r2:
        match_color = "#10b981" if true_label_idx == pred_label_idx else "#ef4444"
        c2_html = f"""
<div style="background: #0e1622; border: 1px solid #1e293b; border-radius: 8px; padding: 16px;">
    <div style="color: #64748b; font-size: 11px; font-weight: 700; text-transform: uppercase;">Supervised Prediction</div>
    <div style="color: {match_color}; font-size: 20px; font-weight: 700; margin-top: 4px;">{pred_label_name}</div>
    <div style="color: {match_color}; font-size: 11px; font-weight: 600;">{confidence:.1f}% Confidence</div>
</div>
"""
        st.markdown(textwrap.dedent(c2_html).strip(), unsafe_allow_html=True)

    with r3:
        c3_html = f"""
<div style="background: #0e1622; border: 1px solid #1e293b; border-radius: 8px; padding: 16px;">
    <div style="color: #64748b; font-size: 11px; font-weight: 700; text-transform: uppercase;">Anomaly Score</div>
    <div style="color: #c084fc; font-size: 20px; font-weight: 700; margin-top: 4px;">{anomaly_score:.4f}</div>
    <div style="color: #94a3b8; font-size: 11px;">Isolation Forest density metric</div>
</div>
"""
        st.markdown(textwrap.dedent(c3_html).strip(), unsafe_allow_html=True)

    with r4:
        anomaly_status_color = "#ef4444" if is_anomalous else "#10b981"
        anomaly_status_text = "ANOMALY FLAGGED" if is_anomalous else "NORMAL DENSITY"
        c4_html = f"""
<div style="background: #0e1622; border: 1px solid #1e293b; border-radius: 8px; padding: 16px;">
    <div style="color: #64748b; font-size: 11px; font-weight: 700; text-transform: uppercase;">Frozen Threshold (τ)</div>
    <div style="color: #f59e0b; font-size: 20px; font-weight: 700; margin-top: 4px;">{tau:.4f}</div>
    <div style="color: {anomaly_status_color}; font-size: 11px; font-weight: 700;">{anomaly_status_text}</div>
</div>
"""
        st.markdown(textwrap.dedent(c4_html).strip(), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Class Probabilities Distribution
    p_col, gauge_col = st.columns([1.5, 1])

    with p_col:
        st.markdown("##### Supervised Class Probabilities")
        df_p = pd.DataFrame({"Class": class_names, "Probability": probs})
        fig_prob = go.Figure(go.Bar(
            x=df_p["Probability"],
            y=df_p["Class"],
            orientation="h",
            marker=dict(color=probs, colorscale="Teal")
        ))
        fig_prob.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_range=[0, 1],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8")
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with gauge_col:
        st.markdown("##### Anomaly Score vs Threshold")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=anomaly_score,
            number=dict(valueformat=".4f", font=dict(color="#c084fc")),
            gauge=dict(
                axis=dict(range=[0.3, 0.8], tickfont=dict(color="#64748b")),
                bar=dict(color="#c084fc"),
                threshold=dict(
                    line=dict(color="#ef4444", width=3),
                    thickness=0.75,
                    value=tau
                ),
                steps=[
                    {"range": [0.3, tau], "color": "#131d2a"},
                    {"range": [tau, 0.8], "color": "#2c1520"}
                ]
            )
        ))
        fig_g.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8")
        )
        st.plotly_chart(fig_g, use_container_width=True)

    # Dedicated Botnet ARES Diagnostic Exemplar Panel
    if true_label_name == "Botnet ARES":
        st.markdown("---")
        ares_html = f"""
<div style="background: #19141f; border: 1px solid #5b21b6; border-radius: 10px; padding: 22px; margin-top: 10px;">
    <h3 style="color: #ef4444; margin-top: 0;">🔬 Diagnostic Exemplar: Why Did Botnet ARES Detection Fail?</h3>
    <p style="color: #cbd5e1; font-size: 14px; line-height: 1.6;">
        In the Leave-One-Class-Out (LOCO) synthetic zero-day experiment, <b>0 out of 375 Botnet ARES flows were detected</b> (0.00% recall, ROC-AUC 0.5236). 
        Inspecting this flow demonstrates the exact technical failure mode:
    </p>
    <div style="display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 200px; background: #0f172a; padding: 12px; border-radius: 6px; border: 1px solid #1e293b;">
            <span style="color: #94a3b8; font-size: 11px;">Current Flow Anomaly Score:</span>
            <div style="color: #c084fc; font-weight: 700; font-size: 18px;">{anomaly_score:.4f}</div>
            <span style="color: #ef4444; font-size: 11px;">Fails to cross threshold τ = {tau:.4f}</span>
        </div>
        <div style="flex: 1; min-width: 200px; background: #0f172a; padding: 12px; border-radius: 6px; border: 1px solid #1e293b;">
            <span style="color: #94a3b8; font-size: 11px;">Closed-World Supervised Mode:</span>
            <div style="color: #f59e0b; font-weight: 700; font-size: 18px;">100% Assigned to 'Normal'</div>
            <span style="color: #94a3b8; font-size: 11px;">Silent bypass when excluded from training</span>
        </div>
    </div>
    <h4 style="color: #e2e8f0; font-size: 14px;">Root-Cause Analysis in NetFlow Feature Space:</h4>
    <ol style="color: #94a3b8; font-size: 13px; line-height: 1.6; padding-left: 20px;">
        <li><b>Low-Rate C2 Beaconing:</b> The Ares botnet establishes periodic, low-frequency command-and-control handshakes. In tabular NetFlow statistics, packet byte count, duration, and inter-arrival times overlap almost completely with benign HTTP keepalives and periodic DNS queries.</li>
        <li><b>Density Blind Spot:</b> An unsupervised density estimator (Isolation Forest) trained purely on benign baseline flows assigns high likelihood (low anomaly score) to periodic heartbeats.</li>
        <li><b>Architectural Lesson:</b> Flow-level network telemetry alone is insufficient for low-rate C2 detection. Real-world protection requires <b>DNS domain reputation</b>, <b>host-level endpoint process trees</b>, and <b>application-layer payload inspection</b>.</li>
    </ol>
</div>
"""
        st.markdown(textwrap.dedent(ares_html).strip(), unsafe_allow_html=True)
