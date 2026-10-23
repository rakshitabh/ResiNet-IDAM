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

# Import 4 Specific Pages
from src.demo.pages.page_overview import render_overview
from src.demo.pages.page_detection_lab import render_detection_lab
from src.demo.pages.page_zero_trust_simulator import render_zero_trust_simulator
from src.demo.pages.page_evaluation import render_evaluation

def main():
    # Sidebar Header & Navigation
    with st.sidebar:
        st.markdown("""
<div style="padding: 10px 0 16px 0; border-bottom: 1px solid #1e293b; margin-bottom: 16px;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 24px;">🛡️</span>
        <div>
            <div style="color: #ffffff; font-weight: 800; font-size: 18px; letter-spacing: 0.5px;">RESINET-IDAM</div>
            <div style="color: #00e5ff; font-size: 11px; font-weight: 600; text-transform: uppercase;">Reviewer Demo v2.0</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        page_choice = st.radio(
            "Navigation Modules:",
            [
                "Page 1: Overview",
                "Page 2: Detection Lab",
                "Page 3: Zero-Trust Decision Simulator",
                "Page 4: Evaluation"
            ],
            index=0
        )

        st.markdown("<br><hr style='border-color: #1e293b;'>", unsafe_allow_html=True)

        st.markdown("""
<div style="color: #94a3b8; font-size: 12px; line-height: 1.5;">
    <b style="color: #cbd5e1;">Evaluation Mode:</b> Replay of recorded telemetry<br>
    <b style="color: #cbd5e1;">IT Dataset:</b> IDS2025 (79 feats)<br>
    <b style="color: #cbd5e1;">OT Dataset:</b> HAI 21.03 (66 feats)<br>
    <b style="color: #cbd5e1;">Branch:</b> <code>feature/realistic-evaluation</code>
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

if __name__ == "__main__":
    main()
