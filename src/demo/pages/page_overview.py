"""
ResiNet-IDAM: Page 1 - Overview
-------------------------------
Provides:
- One-line explanation of ResiNet-IDAM.
- IT detection, OT anomaly detection, and risk-decision engine status.
- Four meaningful core research metrics.
- Clean system architecture diagram rendered via components.html to prevent markdown code block artifacts.
"""

import streamlit as st
import streamlit.components.v1 as components

def render_overview():
    # Hero Title & One-Line Explanation
    st.markdown("""<div style="background: linear-gradient(135deg, #0d1520, #152232); border: 1px solid #1f2d3d; border-radius: 12px; padding: 26px 30px; margin-bottom: 24px;">
<div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
<div>
<span style="background: rgba(0, 229, 255, 0.12); color: #00e5ff; border: 1px solid rgba(0, 229, 255, 0.25); padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">
Adaptive IT-OT Cybersecurity Framework
</span>
<h1 style="color: #ffffff; margin: 10px 0 6px 0; font-size: 30px; font-weight: 800; letter-spacing: -0.5px;">
ResiNet-IDAM
</h1>
<p style="color: #94a3b8; font-size: 16px; margin: 0; line-height: 1.5; max-width: 850px;">
An adaptive cyber-physical intrusion detection and risk-aware Zero-Trust decision framework designed to protect hybrid IT-OT critical infrastructures.
</p>
</div>
<div style="background: #0b1119; border: 1px solid #1e293b; border-radius: 8px; padding: 10px 16px; text-align: right;">
<div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
<span style="display: inline-block; width: 9px; height: 9px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981;"></span>
<span style="color: #f1f5f9; font-weight: 600; font-size: 13px;">Framework Operational</span>
</div>
<div style="color: #64748b; font-size: 11px; margin-top: 2px;">Research Prototype Replay Mode</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # Component Status Row
    st.markdown("### 📡 Subsystem Operational Status")
    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown("""<div style="background: #0e1622; border: 1px solid #1e293b; border-left: 4px solid #00e5ff; border-radius: 8px; padding: 16px;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="color: #e2e8f0; font-weight: 700; font-size: 14px;">IT Network Detection</span>
<span style="color: #10b981; font-size: 12px; font-weight: 600;">ACTIVE</span>
</div>
<div style="color: #94a3b8; font-size: 12px; margin-top: 6px;">
Random Forest classifier evaluated across 79 and 77 NetFlow features on contiguous out-of-burst partitions.
</div>
</div>""", unsafe_allow_html=True)

    with s2:
        st.markdown("""<div style="background: #0e1622; border: 1px solid #1e293b; border-left: 4px solid #38bdf8; border-radius: 8px; padding: 16px;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="color: #e2e8f0; font-weight: 700; font-size: 14px;">OT Anomaly Detection</span>
<span style="color: #10b981; font-size: 12px; font-weight: 600;">ACTIVE</span>
</div>
<div style="color: #94a3b8; font-size: 12px; margin-top: 6px;">
Deep Autoencoder monitoring 66 continuous physical process channels at 1 Hz on HAI water treatment telemetry.
</div>
</div>""", unsafe_allow_html=True)

    with s3:
        st.markdown("""<div style="background: #0e1622; border: 1px solid #1e293b; border-left: 4px solid #c084fc; border-radius: 8px; padding: 16px;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="color: #e2e8f0; font-weight: 700; font-size: 14px;">Risk-Aware Zero-Trust Engine</span>
<span style="color: #10b981; font-size: 12px; font-weight: 600;">ACTIVE</span>
</div>
<div style="color: #94a3b8; font-size: 12px; margin-top: 6px;">
Multi-channel cyber-physical triage coupled with transparent contextual Zero-Trust policy evaluation.
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Meaningful Metrics Row
    st.markdown("### 📊 Core Research Benchmark Metrics")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown("""<div style="background: #0e1622; border: 1px solid #1e293b; border-radius: 10px; padding: 18px; text-align: center;">
<div style="color: #64748b; font-size: 11px; font-weight: 700; text-transform: uppercase;">IT Out-of-Burst Accuracy</div>
<div style="color: #00e5ff; font-size: 28px; font-weight: 800; margin: 4px 0;">99.58%</div>
<div style="color: #94a3b8; font-size: 12px;">All 79 Features</div>
<div style="color: #f59e0b; font-size: 11px; margin-top: 4px;">No Ports (77 Feats): <b>98.66%</b></div>
</div>""", unsafe_allow_html=True)

    with m2:
        st.markdown("""<div style="background: #0e1622; border: 1px solid #1e293b; border-radius: 10px; padding: 18px; text-align: center;">
<div style="color: #64748b; font-size: 11px; font-weight: 700; text-transform: uppercase;">OT Process Telemetry Accuracy</div>
<div style="color: #38bdf8; font-size: 28px; font-weight: 800; margin: 4px 0;">92.27%</div>
<div style="color: #94a3b8; font-size: 12px;">54,000s Test Records</div>
<div style="color: #94a3b8; font-size: 11px; margin-top: 4px;">Attack Recall: <b>46.39%</b> (14 blocks)</div>
</div>""", unsafe_allow_html=True)

    with m3:
        st.markdown("""<div style="background: #0e1622; border: 1px solid #1e293b; border-radius: 10px; padding: 18px; text-align: center;">
<div style="color: #64748b; font-size: 11px; font-weight: 700; text-transform: uppercase;">Cyber-Physical Fusion F1</div>
<div style="color: #c084fc; font-size: 28px; font-weight: 800; margin: 4px 0;">28.95%</div>
<div style="color: #94a3b8; font-size: 12px;">Precision: 56.44% | Recall: 19.47%</div>
<div style="color: #10b981; font-size: 11px; margin-top: 4px;">Zero Synthetic Shortcut Features</div>
</div>""", unsafe_allow_html=True)

    with m4:
        st.markdown("""<div style="background: #0e1622; border: 1px solid #1e293b; border-radius: 10px; padding: 18px; text-align: center;">
<div style="color: #64748b; font-size: 11px; font-weight: 700; text-transform: uppercase;">Normal Baseline False Alarm Rate</div>
<div style="color: #10b981; font-size: 28px; font-weight: 800; margin: 4px 0;">1.05% - 1.30%</div>
<div style="color: #94a3b8; font-size: 12px;">Validated on Normal Flows</div>
<div style="color: #94a3b8; font-size: 11px; margin-top: 4px;">Low operational alert fatigue</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Clean System Architecture Diagram rendered via components.html
    st.markdown("### 🏛️ System Architecture")
    
    arch_svg_html = """<!DOCTYPE html>
<html>
<head>
<style>
body { margin: 0; padding: 0; background-color: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.card { background: #0a0f16; border: 1px solid #1e293b; border-radius: 12px; padding: 16px 20px; box-sizing: border-box; }
</style>
</head>
<body>
<div class="card">
<svg width="100%" height="220" viewBox="0 0 960 220" xmlns="http://www.w3.org/2000/svg">
<!-- Domain Ingestion Blocks -->
<rect x="20" y="20" width="180" height="70" rx="8" fill="#131e2c" stroke="#00e5ff" stroke-width="1.5"/>
<text x="110" y="48" fill="#00e5ff" font-size="14" font-weight="700" text-anchor="middle">IT NetFlow Telemetry</text>
<text x="110" y="70" fill="#94a3b8" font-size="11" text-anchor="middle">79 Flow Features (IDS2025)</text>

<rect x="20" y="125" width="180" height="70" rx="8" fill="#131e2c" stroke="#38bdf8" stroke-width="1.5"/>
<text x="110" y="153" fill="#38bdf8" font-size="14" font-weight="700" text-anchor="middle">OT Physical Telemetry</text>
<text x="110" y="175" fill="#94a3b8" font-size="11" text-anchor="middle">66 Process Channels @ 1 Hz</text>

<!-- Connecting Lines -->
<line x1="200" y1="55" x2="260" y2="55" stroke="#475569" stroke-width="2"/>
<line x1="200" y1="160" x2="260" y2="160" stroke="#475569" stroke-width="2"/>

<!-- Detection Models -->
<rect x="260" y="20" width="200" height="70" rx="8" fill="#15202e" stroke="#334155" stroke-width="1.5"/>
<text x="360" y="48" fill="#f1f5f9" font-size="13" font-weight="700" text-anchor="middle">IT Random Forest</text>
<text x="360" y="70" fill="#00e5ff" font-size="11" font-weight="600" text-anchor="middle">p_attack &isin; [0, 1]</text>

<rect x="260" y="125" width="200" height="70" rx="8" fill="#15202e" stroke="#334155" stroke-width="1.5"/>
<text x="360" y="153" fill="#f1f5f9" font-size="13" font-weight="700" text-anchor="middle">Deep Autoencoder</text>
<text x="360" y="175" fill="#38bdf8" font-size="11" font-weight="600" text-anchor="middle">Reconstruction Error S_ot</text>

<!-- Connect to Fusion -->
<line x1="460" y1="55" x2="520" y2="95" stroke="#475569" stroke-width="2"/>
<line x1="460" y1="160" x2="520" y2="120" stroke="#475569" stroke-width="2"/>

<!-- Fusion & Zero-Trust -->
<rect x="520" y="45" width="210" height="120" rx="10" fill="#1e1833" stroke="#a855f7" stroke-width="2"/>
<text x="625" y="77" fill="#c084fc" font-size="14" font-weight="800" text-anchor="middle">Cyber-Physical Fusion</text>
<text x="625" y="97" fill="#e2e8f0" font-size="12" text-anchor="middle">& Zero-Trust Policy Engine</text>
<text x="625" y="120" fill="#94a3b8" font-size="10" text-anchor="middle">R_CP = &sigma;(6.52 p_att + 4.71 S_ot ...)</text>
<text x="625" y="142" fill="#a855f7" font-size="10" font-weight="600" text-anchor="middle">Context: User, Device, Resource, Risk</text>

<!-- Connect to Decision -->
<line x1="730" y1="105" x2="780" y2="105" stroke="#a855f7" stroke-width="2"/>

<!-- Decision Outputs -->
<rect x="780" y="15" width="160" height="52" rx="6" fill="#1f1519" stroke="#ef4444" stroke-width="1.5"/>
<text x="860" y="36" fill="#ef4444" font-size="11" font-weight="700" text-anchor="middle">Tier 4 Critical Alert</text>
<text x="860" y="53" fill="#fca5a5" font-size="10" text-anchor="middle">Plant/Network Isolation</text>

<rect x="780" y="80" width="160" height="52" rx="6" fill="#1b1c18" stroke="#f59e0b" stroke-width="1.5"/>
<text x="860" y="101" fill="#f59e0b" font-size="11" font-weight="700" text-anchor="middle">Zero-Trust Decision</text>
<text x="860" y="118" fill="#fde68a" font-size="10" text-anchor="middle">Allow / Verify / Deny</text>

<rect x="780" y="145" width="160" height="52" rx="6" fill="#131c26" stroke="#38bdf8" stroke-width="1.5"/>
<text x="860" y="166" fill="#38bdf8" font-size="11" font-weight="700" text-anchor="middle">Tier 2 Triage Route</text>
<text x="860" y="183" fill="#bae6fd" font-size="10" text-anchor="middle">Plant Eng vs Enterprise SOC</text>
</svg>
</div>
</body>
</html>"""
    components.html(arch_svg_html, height=250)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Use the left navigation sidebar to explore the Detection Lab, Zero-Trust Decision Simulator, and Research Evaluation benchmarks.")
