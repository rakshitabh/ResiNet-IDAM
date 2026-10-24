"""Unified ResiNet-IDAM Streamlit application."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.application.detection_service import DetectionService
from src.infrastructure.config import AppConfig


ROOT = Path(__file__).resolve().parent
config = AppConfig.from_environment(ROOT)


@st.cache_resource
def get_service() -> DetectionService:
    return DetectionService(config.root, config.database_path)


@st.cache_data
def load_profile() -> dict[str, Any]:
    path = config.reports_path / "dataset-profile.json"
    return __import__("json").loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_metrics() -> dict[str, Any]:
    path = config.reports_path / "baseline_metrics.json"
    return __import__("json").loads(path.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(
        page_title="ResiNet-IDAM Security Operations",
        page_icon="R",
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
                "Alert management",
                "Risk assessment",
                "Model and data health",
                "Administration",
            ],
        )
        st.divider()
        st.caption(f"Environment: {config.environment}")
        st.caption(f"IT detector: {'available' if service.availability()['IT'] else 'not configured'}")
        st.caption(f"OT detector: {'available' if service.availability()['OT'] else 'not configured'}")

    if page == "Operations overview":
        render_overview(service)
    elif page == "Detection workspace":
        render_detection(service)
    elif page == "Alert management":
        render_alerts(service)
    elif page == "Risk assessment":
        render_risk(service)
    elif page == "Model and data health":
        render_health(service)
    else:
        render_admin(service)


def render_overview(service: DetectionService) -> None:
    st.title("Operations overview")
    summary = service.store.summary()
    distribution = summary["risk_distribution"]
    high = distribution.get("HIGH", 0) + distribution.get("CRITICAL", 0)
    cards = st.columns(4)
    cards[0].metric("Processed events", summary["total_events"])
    cards[1].metric("Anomalies", summary["anomalies_detected"])
    cards[2].metric("High / critical", high)
    cards[3].metric("Detectors online", sum(service.availability().values()))
    st.subheader("Risk posture")
    risk_frame = pd.DataFrame(
        {"Events": [distribution.get(level, 0) for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")]},
        index=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )
    st.bar_chart(risk_frame)
    events = service.store.list(limit=10)
    st.subheader("Recent alerts")
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
    else:
        st.info("No events have been scored.")


def render_detection(service: DetectionService) -> None:
    st.title("Detection workspace")
    available = service.availability()
    source = st.selectbox("Telemetry source", ["OT", "IT"], disabled=not any(available.values()))
    if not available[source]:
        st.warning(f"{source} detector artifacts are not configured in this deployment.")
        st.info("Add the detector model and metadata artifacts, then restart the application.")
        return
    detector = service.detector(source)
    schema = detector.feature_schema()
    st.caption(f"{detector.name} · {len(schema)} ordered features")
    if source == "OT":
        data_path = config.data_path / "processed" / "hai-test-scaled.csv"
        if not data_path.exists():
            st.error(f"Observation file not found: {data_path}")
            return
        frame = pd.read_csv(data_path)
        index = st.number_input("Observation row", 0, len(frame) - 1, 0)
        row = frame.iloc[int(index)]
        values = {name: float(row[name]) for name in schema}
        timestamp = pd.to_datetime(row["timestamp"], utc=True).to_pydatetime()
    else:
        st.info("IT upload/sample selection will use the configured IDS2025 feature contract.")
        uploaded = st.file_uploader("Upload one-row IT CSV", type="csv")
        if uploaded is None:
            return
        frame = pd.read_csv(uploaded)
        missing = [name for name in schema if name not in frame.columns]
        if missing:
            st.error(f"Missing IT features: {missing[:5]}")
            return
        row = frame.iloc[0]
        values = {name: float(row[name]) for name in schema}
        timestamp = datetime.now(timezone.utc)
    with st.form("detection-form"):
        asset_id = st.text_input("Asset", "asset-01")
        device_id = st.text_input("Device", "device-01")
        identity_id = st.text_input("Identity", "identity-01")
        left, right = st.columns(2)
        identity_trust = left.slider("Identity trust", 0.0, 1.0, 0.8)
        device_trust = right.slider("Device trust", 0.0, 1.0, 0.8)
        asset_criticality = left.slider("Asset criticality", 0.0, 1.0, 0.7)
        behavior_deviation = right.slider("Behavior deviation", 0.0, 1.0, 0.2)
        submit = st.form_submit_button("Run detection and create alert")
    if submit:
        try:
            event = service.detect(
                source,
                values,
                timestamp=timestamp,
                identity_id=identity_id,
                device_id=device_id,
                asset_id=asset_id,
                identity_trust=identity_trust,
                device_trust=device_trust,
                asset_criticality=asset_criticality,
                behavior_deviation=behavior_deviation,
            )
        except (RuntimeError, ValueError) as error:
            st.error(str(error))
        else:
            st.success(f"Alert created: {event['event_id']}")
            st.json(event)


def render_alerts(service: DetectionService) -> None:
    st.title("Alert management")
    level = st.selectbox("Risk level", ["ALL", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
    status = st.selectbox("Review status", ["ALL", "NEW", "IN_REVIEW", "RESOLVED"])
    events = service.store.list(
        None if level == "ALL" else level,
        review_status=None if status == "ALL" else status,
        limit=500,
    )
    if not events:
        st.info("No alerts match the selected filters.")
        return
    table = pd.DataFrame(events)
    st.dataframe(
        table[["event_id", "source", "timestamp", "asset_id", "prediction", "risk_score", "risk_level", "review_status"]],
        use_container_width=True,
        hide_index=True,
    )
    event_id = st.selectbox("Open alert", [event["event_id"] for event in events])
    event = next(item for item in events if item["event_id"] == event_id)
    st.json(event)
    new_status = st.selectbox("Update review status", ["NEW", "IN_REVIEW", "RESOLVED"])
    if st.button("Save review status"):
        if service.store.update_review(event_id, new_status):
            st.success("Review status updated.")
            st.rerun()


def render_risk(service: DetectionService) -> None:
    st.title("Risk assessment")
    events = service.store.list(limit=500)
    if not events:
        st.info("Score an observation to inspect contextual risk.")
        return
    event_id = st.selectbox("Select event", [event["event_id"] for event in events])
    event = next(item for item in events if item["event_id"] == event_id)
    cards = st.columns(3)
    cards[0].metric("Risk score", f"{event['risk_score']:.1f}/100")
    cards[1].metric("Risk level", event["risk_level"])
    cards[2].metric("Recommended action", event["recommended_action"])
    st.subheader("Risk components")
    st.bar_chart(pd.Series(event["risk_components"]))
    st.json({"context": event["context"], "components": event["risk_components"]})


def render_health(service: DetectionService) -> None:
    st.title("Model and data health")
    availability = service.availability()
    st.subheader("Detector availability")
    st.dataframe(
        pd.DataFrame({"Detector": list(availability), "Available": list(availability.values())}),
        use_container_width=True,
        hide_index=True,
    )
    try:
        profile = load_profile()
        metrics = load_metrics()
    except FileNotFoundError as error:
        st.error(str(error))
        return
    cards = st.columns(4)
    cards[0].metric("Model features", profile["model_input"]["feature_count"])
    cards[1].metric("Training rows", f"{profile['training']['rows']:,}")
    cards[2].metric("Test rows", f"{profile['test']['rows']:,}")
    cards[3].metric("F1", f"{metrics.get('f1', 0):.4f}")
    with st.expander("Evaluation diagnostics"):
        st.json(metrics)
    with st.expander("Dataset feature dictionary"):
        st.dataframe(pd.DataFrame(profile["features"]), use_container_width=True, hide_index=True)


def render_admin(service: DetectionService) -> None:
    st.title("Administration")
    st.subheader("Runtime configuration")
    st.json(
        {
            "environment": config.environment,
            "database": str(config.database_path),
            "models": str(config.models_path),
            "data": str(config.data_path),
            "reports": str(config.reports_path),
            "detectors": service.availability(),
        }
    )
    st.info("Configuration is read from RESINET_* environment variables. Restart after changing artifacts or paths.")


if __name__ == "__main__":
    main()
