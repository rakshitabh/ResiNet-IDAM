"""Unified ResiNet-IDAM Streamlit Security Operations Application."""

from __future__ import annotations

from pathlib import Path
import streamlit as st

from src.application.detection_service import DetectionService
from src.infrastructure.config import AppConfig
from src.ui.simulation_view import render_simulation_view
from src.ui.views import (
    load_dataset,
    load_metrics,
    load_profile,
    render_admin,
    render_alert_prioritization,
    render_alerts,
    render_detection,
    render_health,
    render_investigation,
    render_overview,
    render_risk,
    resolve_dataset_path,
)

ROOT = Path(__file__).resolve().parent
config = AppConfig.from_environment(ROOT)


@st.cache_resource
def get_service() -> DetectionService:
    return DetectionService(config.root, config.database_path)


def main() -> None:
    st.set_page_config(
        page_title="ResiNet-IDAM Security Operations",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container {max-width: 1500px; padding-top: 1.5rem;}
        [data-testid="stMetricValue"] {font-size: 1.6rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    service = get_service()

    with st.sidebar:
        st.title("ResiNet-IDAM")
        st.caption("Security operations console")

        page = st.radio(
            "Workspace",
            [
                "Operations overview",
                "Detection workspace",
                "Zero-Trust simulation",
                "Alert management",
                "Risk assessment",
                "Model and data health",
                "Alert prioritization",
                "Administration",
            ],
        )

        st.divider()
        st.caption(f"Environment: {config.environment}")
        availability = service.availability()
        st.caption(
            f"IT detector: {'available' if availability['IT'] else 'not configured'}"
        )
        st.caption(
            f"OT detector: {'available' if availability['OT'] else 'not configured'}"
        )
        st.divider()
        st.caption("Research prototype & decision-support only. Does not authorize automatic industrial actions.")

    if page == "Operations overview":
        render_overview(service)
    elif page == "Detection workspace":
        render_detection(service)
    elif page == "Zero-Trust simulation":
        render_simulation_view(service)
    elif page == "Alert management":
        render_alerts(service)
    elif page == "Risk assessment":
        render_risk(service)
    elif page == "Model and data health":
        render_health(service)
    elif page == "Alert prioritization":
        render_alert_prioritization(service)
    else:
        render_admin(service)


if __name__ == "__main__":
    main()