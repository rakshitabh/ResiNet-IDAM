"""
ResiNet-IDAM: Adaptive IT-OT Intrusion Detection Research Demo
=============================================================
Main Streamlit application entry point.
"""

import sys
from unittest.mock import MagicMock

# Proactively patch binary modules blocked by Windows Application Control
# (These modules are not used by Random Forest or Isolation Forest).
if 'sklearn.svm._libsvm_sparse' not in sys.modules:
    sys.modules['sklearn.svm._libsvm_sparse'] = MagicMock()
if 'scipy.stats._qmvnt_cy' not in sys.modules:
    sys.modules['scipy.stats._qmvnt_cy'] = MagicMock()

import streamlit as st

# Configure wide layout and page metadata
st.set_page_config(
    page_title="ResiNet-IDAM: Adaptive IT-OT Intrusion Detection Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    .main {
        background-color: #0b0f17;
    }
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        background-color: #0e141f;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stMetricValue"] {
        font-size: 22px;
        font-weight: 700;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        border-color: #00e5ff;
        color: #00e5ff;
    }
</style>
""", unsafe_allow_html=True)

# Import Pages
from src.demo.pages.page_overview import render_overview
from src.demo.pages.page_detection_lab import render_detection_lab
from src.demo.pages.page_zero_trust_simulator import render_zero_trust_simulator
from src.demo.pages.page_evaluation import render_evaluation
from src.demo.pages.page_ids2025_risk import render_ids2025_risk

def main():
    # Sidebar Brand & Navigation
    with st.sidebar:
        st.markdown("""
        <div style="padding: 10px 0 20px 0;">
            <h2 style="color: #00e5ff; margin:0; font-size: 22px; font-weight: 800; letter-spacing: -0.5px;">
                🛡️ ResiNet-IDAM
            </h2>
            <div style="color: #94a3b8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">
                IT-OT Intrusion Detection & Zero-Trust
            </div>
        </div>
        """, unsafe_allow_html=True)

        pages = [
            "Page 1: Overview",
            "Page 2: Detection Lab",
            "Page 3: Zero-Trust Decision Simulator",
            "Page 4: Evaluation",
            "Page 5: IDS2025 Risk Dashboard",
        ]

        page_choice = st.radio(
            "Navigation",
            pages,
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border-color: #1e293b;'>", unsafe_allow_html=True)

        # System Health Card
        st.markdown("""
        <div style="background-color: #131d2e; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
            <div style="font-size: 12px; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">Runtime Pipeline</div>
            <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;">
                <span style="color: #94a3b8;">IT Flow RF:</span>
                <span style="color: #10b981; font-weight: 600;">ACTIVE (7-Class)</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;">
                <span style="color: #94a3b8;">OT HAI Detector:</span>
                <span style="color: #10b981; font-weight: 600;">ACTIVE (98.6%)</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 11px;">
                <span style="color: #94a3b8;">Zero-Trust Engine:</span>
                <span style="color: #38bdf8; font-weight: 600;">ENFORCING</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color: #1e293b;'>", unsafe_allow_html=True)
        st.markdown("""
<div style="color: #64748b; font-size: 11px; text-align: center;">
    Adaptive IT-OT Intrusion Detection<br>
    &copy; 2026 ResiNet-IDAM Project
</div>
""", unsafe_allow_html=True)

    # Route to Selected Page
    if page_choice == "Page 1: Overview":
        render_overview()
    elif page_choice == "Page 2: Detection Lab":
        render_detection_lab()
    elif page_choice == "Page 3: Zero-Trust Decision Simulator":
        render_zero_trust_simulator()
    elif page_choice == "Page 4: Evaluation":
        render_evaluation()
    elif page_choice == "Page 5: IDS2025 Risk Dashboard":
        render_ids2025_risk()

if __name__ == "__main__":
    main()
