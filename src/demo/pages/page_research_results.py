"""
ResiNet-IDAM: 05 Research Results Page
-------------------------------------
Comprehensive presentation of the 7-class benchmark, diagnostic overlap analysis,
synthetic LOCO zero-day evaluation, failed Botnet ARES root cause, and core limitations.
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed" / "it_realistic"
REPORT_DIR = ROOT / "data" / "reports"

@st.cache_data
def load_results_data():
    loco_data = json.load(open(DATA_DIR / "loco_predictions.json"))
    return loco_data

def render_research_results():
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; margin-bottom: 4px;">05 Research Results: Benchmarks, LOCO & Limitations</h2>
            <p style="color: #94a3b8; font-size: 14px;">Evaluates model generalization across out-of-burst contiguous partitions and synthetic unseen zero-day attacks.</p>
        </div>
    """, unsafe_allow_html=True)

    loco_data = load_results_data()

    tab_bench, tab_loco, tab_overlap, tab_limits = st.tabs([
        "📊 7-Class Benchmark Comparison",
        "🛡️ Synthetic LOCO (Zero-Day) Evaluation",
        "🔍 Diagnostic 824 Overlap Vectors",
        "⚠️ Core Limitations & Scope"
    ])

    with tab_bench:
        st.markdown("#### Realistic Out-of-Burst 7-Class Benchmark vs Frozen Reference")
        st.caption("Contiguous sequential 80/20 block split (row-order temporal proxy) evaluated across both feature configurations:")

        df_bench = pd.DataFrame({
            "Metric": [
                "Number of Features",
                "Split Protocol",
                "Overall Test Accuracy",
                "Macro F1-Score",
                "Weighted F1-Score",
                "Benign FPR (Normal=Neg)",
                "Attack FNR (Missed=Norm)",
                "Exact Overlap Vectors"
            ],
            "Frozen Baseline Reference (V1.0)": [
                "79",
                "Random Stratified (Point-wise)",
                "99.56%",
                "96.76%",
                "99.56%",
                "0.15%",
                "0.44%",
                "0 rows"
            ],
            "Configuration A (All 79 Features)": [
                "79",
                "Contiguous Sequential (Out-of-Burst)",
                "99.58%",
                "96.82%",
                "99.58%",
                "1.30% (68 / 5,237)",
                "0.06% (8 / 13,107)",
                "1 row"
            ],
            "Configuration B (77 Feats - No Ports)": [
                "77",
                "Contiguous Sequential (Out-of-Burst)",
                "98.66%",
                "95.63%",
                "98.66%",
                "3.67% (192 / 5,237)",
                "0.40% (52 / 13,107)",
                "824 unique vectors (7,080 rows)"
            ]
        })
        st.dataframe(df_bench, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly Comparison Bar Chart
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_acc = go.Figure(data=[
                go.Bar(name="Accuracy", x=["Frozen Baseline", "Config A (79 Feats)", "Config B (No Ports)"], y=[99.56, 99.58, 98.66], marker_color="#00e5ff"),
                go.Bar(name="Macro F1", x=["Frozen Baseline", "Config A (79 Feats)", "Config B (No Ports)"], y=[96.76, 96.82, 95.63], marker_color="#a855f7")
            ])
            fig_acc.update_layout(
                title="Classification Performance (%)",
                barmode="group",
                yaxis_range=[90, 100],
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8")
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_c2:
            fig_err = go.Figure(data=[
                go.Bar(name="False Positive Rate (FPR)", x=["Frozen Baseline", "Config A (79 Feats)", "Config B (No Ports)"], y=[0.15, 1.30, 3.67], marker_color="#f59e0b"),
                go.Bar(name="False Negative Rate (FNR)", x=["Frozen Baseline", "Config A (79 Feats)", "Config B (No Ports)"], y=[0.44, 0.06, 0.40], marker_color="#ef4444")
            ])
            fig_err.update_layout(
                title="Error Rates on Out-of-Sample Test Partition (%)",
                barmode="group",
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8")
            )
            st.plotly_chart(fig_err, use_container_width=True)

    with tab_loco:
        st.markdown("#### Synthetic Leave-One-Class-Out (LOCO) Zero-Day Evaluation")
        st.caption("Measures detection capabilities when an attack class is completely excluded from training:")

        loco_rows = []
        for exp_name, data in loco_data.items():
            cfg, cls = exp_name.split(" | ")
            loco_rows.append({
                "Feature Configuration": cfg,
                "Withheld Attack Class": cls,
                "Test Support (N)": 6 if cls == "Infiltration" else 375,
                "Supervised -> Normal (Silent Bypass)": "100.00%",
                "Anomaly Recall (Frozen τ)": f"{data['anomaly_recall']:.2f}%",
                "Validation FPR": f"{data['val_fpr']:.2f}%",
                "Test FPR (Normal)": f"{data['test_fpr']:.2f}%",
                "ROC-AUC": f"{data['roc_auc']:.4f}"
            })
        df_loco = pd.DataFrame(loco_rows)
        st.dataframe(df_loco, use_container_width=True, hide_index=True)

        # In-Depth Focus on Key Limitation: Botnet ARES Failure
        st.error(
            "**Key Research Limitation — Unseen Botnet ARES Detection Failure (0.00% Recall):**\n\n"
            "Across both 79-feature and 77-feature configurations, the anomaly detector completely failed to detect "
            "withheld Botnet ARES flows (**0 / 375 detected at the frozen 1% threshold**, ROC-AUC **0.5236 to 0.5587**).\n\n"
            "**Root Cause:** Low-rate botnet C2 beaconing packets match benign background keepalives and periodic polling in NetFlow statistics "
            "(byte counts, duration, and TCP flags). Flow-level density estimators trained solely on benign traffic cannot separate them without "
            "domain reputation, host process telemetry, or deep packet payload inspection."
        )

        st.warning(
            "**Statistical Uncertainty Warning on Infiltration:**\n\n"
            "`Infiltration` contains only 29 total instances in `IDS2025` (23 training / 6 test). "
            "A test recall of **33.33% (2 / 6 detected)** carries an asymptotic 95% Wilson confidence interval of **[9.7%, 70.0%]**. "
            "This finding demonstrates statistical divergence, but cannot be claimed as reliable zero-day defense."
        )

    with tab_overlap:
        st.markdown("#### Diagnostic Quantification of the 824 Overlapping Vectors (Config B)")
        st.caption("When ephemeral port numbers are removed, 824 unique feature vectors occur across train and test:")

        col_ov1, col_ov2 = st.columns([1.2, 1])
        with col_ov1:
            st.markdown("""
                <div style="background: #111a24; border: 1px solid #1e293b; border-radius: 8px; padding: 18px;">
                    <h4 style="color: #00e5ff; margin-top: 0;">Repeated Patterns vs Ambiguous Collisions</h4>
                    <p style="color: #cbd5e1; font-size: 13px;">
                        • <b>Repeated Patterns (97.69%, 805 / 824 vectors):</b> Share the exact same label in both train and test. 
                        These represent automated attack scripts that generate thousands of identical frames (e.g., 5,014 identical PortScan probes, 748 Brute Force attempts, 937 DoS packets).
                        <br><br>
                        • <b>Ambiguous Collisions (2.31%, 19 / 824 vectors):</b> Feature collisions where identical packet length and flag signatures belong to different classes:
                        <br>&nbsp;&nbsp;&nbsp;&nbsp;- <b>17 vectors:</b> Normal &harr; PortScan (empty TCP reset/SYN frames)
                        <br>&nbsp;&nbsp;&nbsp;&nbsp;- <b>1 vector:</b> Normal &harr; Dos/DDos
                        <br>&nbsp;&nbsp;&nbsp;&nbsp;- <b>1 vector:</b> Normal &harr; Botnet ARES
                        <br><br>
                        • <b>Diagnostic Impact:</b> These 19 ambiguous collisions directly drove the benign FPR increase from <b>1.30% to 3.67%</b>.
                    </p>
                </div>
            """, unsafe_allow_html=True)

        with col_ov2:
            fig_ov = go.Figure(data=[go.Pie(
                labels=["Identical Repeated Patterns (805)", "Ambiguous Collisions (19)"],
                values=[805, 19],
                hole=0.45,
                marker=dict(colors=["#00e5ff", "#ef4444"])
            )])
            fig_ov.update_layout(
                title="Unique Overlap Vectors Breakdown",
                height=280,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8")
            )
            st.plotly_chart(fig_ov, use_container_width=True)

    with tab_limits:
        st.markdown("#### ResiNet-IDAM: Core Research Limitations Matrix")
        st.caption("Transparent documentation of experimental constraints as stated in the project report:")

        df_limits = pd.DataFrame({
            "Dimension": [
                "IT Temporal Splitting",
                "Zero-Day Generalization",
                "Statistical Sample Size",
                "Flow Feature Overlap",
                "Physical Telemetry Coverage",
                "Cross-Domain Coupling"
            ],
            "Limitation Description": [
                "IDS2025 contains zero timestamps, flow IDs, or IP addresses; contiguous splitting is a row-order proxy, not a verified campaign-independent test.",
                "LOCO evaluates withheld classes from the same simulated testbed; does not establish real-world campaign-independent zero-day detection.",
                "Infiltration has only 6 test examples, creating very wide confidence intervals (+/-35%). Results are strictly anecdotal.",
                "Unsupervised density estimation fails on low-rate Botnet ARES (0.00% recall) due to overlapping NetFlow distributions with normal keepalives.",
                "OT Autoencoder achieves 46.39% recall, missing 1,598 attack seconds where sensor spoofing remained within physical tolerance bands.",
                "IDS2025 and HAI were captured on separate testbeds and are synthetically paired. Validating cross-domain correlation requires synchronized hybrid testbeds."
            ],
            "Impact on Research Claims": [
                "Prevents intra-burst point-wise interpolation, but out-of-distribution limits remain untested.",
                "Claims must be limited to synthetic held-out-class evaluation.",
                "Infiltration detection cannot be claimed as generalized zero-day sensitivity.",
                "Intrusion detection cannot claim zero-day C2 detection on tabular flow features alone.",
                "Simple reconstruction thresholds cannot detect subtle in-band sensor tampering.",
                "Multi-stage cyber-physical triage is a methodological framework, not an empirically validated production deployment."
            ]
        })
        st.dataframe(df_limits, use_container_width=True, hide_index=True)
