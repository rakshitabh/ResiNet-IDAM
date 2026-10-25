"""Unified ResiNet-IDAM Streamlit application."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, IO

import pandas as pd
import plotly.express as px
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


def resolve_dataset_path(root: str | Path | None = None) -> Path:
    project_root = Path(root) if root is not None else ROOT
    candidate_dirs = [
        project_root / "data" / "processed",
        project_root / "data" / "raw",
        project_root / "data",
    ]

    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for name in (
            "hai-test-scaled.csv",
            "hai-train-scaled.csv",
            "ids2025_test_scaled.csv",
            "ids2025_train_scaled.csv",
            "train.csv",
            "test.csv",
        ):
            path = directory / name
            if path.exists():
                return path

        files = sorted(directory.glob("*.csv"))
        if files:
            return files[0]

    raise FileNotFoundError(
        "No CSV dataset was found in the configured project data folders. "
        "Upload a CSV file or place the processed dataset under data/processed/."
    )


def load_dataset(source: str | Path | IO[str] | IO[bytes] | None = None, *, nrows: int | None = None) -> pd.DataFrame:
    if source is None:
        source = resolve_dataset_path()

    if hasattr(source, "read"):
        payload = source.read()
        if isinstance(payload, bytes):
            text = payload.decode("utf-8-sig")
            buffer = StringIO(text)
            if getattr(source, "name", "").lower().endswith(".csv"):
                return pd.read_csv(buffer, nrows=nrows)
            return pd.read_csv(StringIO(text), nrows=nrows)
        if isinstance(payload, str):
            return pd.read_csv(StringIO(payload), nrows=nrows)
        raise TypeError("Uploaded dataset payload must be text or bytes.")

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, nrows=nrows)


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
                "Investigation workflow",
                "Alert prioritization",
                "Administration",
            ],
        )

        st.divider()
        st.caption(f"Environment: {config.environment}")
        st.caption(
            f"IT detector: {'available' if service.availability()['IT'] else 'not configured'}"
        )
        st.caption(
            f"OT detector: {'available' if service.availability()['OT'] else 'not configured'}"
        )

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
    elif page == "Investigation workflow":
        render_investigation(service)
    elif page == "Alert prioritization":
        render_alert_prioritization(service)
    else:
        render_admin(service)


def render_overview(service: DetectionService) -> None:
    st.title("Operations overview")

    st.subheader("Dataset")
    dataset_mode = st.radio("Source", ["Project data", "Upload CSV"], horizontal=True)
    dataset_frame: pd.DataFrame | None = None

    if dataset_mode == "Project data":
        try:
            dataset_path = resolve_dataset_path()
        except FileNotFoundError as error:
            st.warning(str(error))
            dataset_frame = pd.DataFrame()
        else:
            st.caption(f"Loaded from {dataset_path}")
            dataset_frame = load_dataset(dataset_path, nrows=5000)
    else:
        uploaded = st.file_uploader("Load dataset CSV", type=["csv"])
        if uploaded is not None:
            try:
                dataset_frame = load_dataset(uploaded, nrows=5000)
            except Exception as error:
                st.error(f"Could not read uploaded dataset: {error}")
                dataset_frame = pd.DataFrame()
        else:
            dataset_frame = pd.DataFrame()

    if not dataset_frame.empty:
        preview_columns = [
            column for column in dataset_frame.columns if column not in {"timestamp", "label"}
        ][:8]
        st.dataframe(dataset_frame.head(10)[preview_columns or dataset_frame.columns], use_container_width=True, hide_index=True)
        st.caption(f"Rows: {len(dataset_frame)}, columns: {len(dataset_frame.columns)}")

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
        {
            "Events": [
                distribution.get(level, 0)
                for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
            ]
        },
        index=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )
    st.plotly_chart(
        px.bar(
            risk_frame.reset_index().rename(columns={"index": "Level", "Events": "Events"}),
            x="Level",
            y="Events",
        ),
        use_container_width=True,
    )

    st.subheader("Stale alert summary")
    events = service.store.list(limit=500)
    if events:
        frame = pd.DataFrame(events)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            frame["is_stale"] = (
                frame["review_status"].isin(["NEW", "IN_REVIEW"])
                & (
                    pd.Timestamp.utcnow().tz_localize(None) - frame["timestamp"]
                    > pd.Timedelta(hours=24)
                )
            )
            stale = frame[frame["is_stale"]]
        else:
            stale = pd.DataFrame()

        if stale.empty:
            st.info("No stale unresolved alerts.")
        else:
            st.metric("Stale unresolved alerts", len(stale))
            st.dataframe(
                stale[
                    [
                        "event_id",
                        "asset_id",
                        "source",
                        "risk_level",
                        "review_status",
                        "timestamp",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No alerts available yet.")

    st.subheader("Recent alerts")
    recent = service.store.list(limit=10)
    if recent:
        st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)
    else:
        st.info("No events have been scored.")


def render_detection(service: DetectionService) -> None:
    st.title("Detection workspace")
    available = service.availability()
    source = st.selectbox(
        "Telemetry source",
        ["OT", "IT"],
        disabled=not any(available.values()),
    )

    if not available[source]:
        st.warning(f"{source} detector artifacts are not configured in this deployment.")
        st.info("Add the detector model and metadata artifacts, then restart the application.")
        return

    detector = service.detector(source)
    config_name = None

    if source == "OT":
        schema = detector.feature_schema()
        st.caption(f"{detector.name} · {len(schema)} ordered features")

        data_path = config.data_path / "processed" / "hai-test-scaled.csv"
        if not data_path.exists():
            st.error(f"Observation file not found: {data_path}")
            return

        frame = pd.read_csv(data_path)
        index = st.number_input("Observation row", 0, len(frame) - 1, 0)
        row = frame.iloc[int(index)]
        values = {name: float(row[name]) for name in schema}
        timestamp = pd.to_datetime(row["timestamp"], utc=True).to_pydatetime()

        ot_summary = detector.score(values)
        st.subheader("OT anomaly diagnostics")
        cols = st.columns(4)
        cols[0].metric("Reconstruction error", f"{ot_summary['reconstruction_error']:.4f}")
        cols[1].metric("Threshold", f"{ot_summary['threshold']:.4f}")
        cols[2].metric("Decision", ot_summary["prediction"])
        cols[3].metric("Confidence", f"{ot_summary['confidence']:.2%}")

        st.json(
            {
                "source": "OT",
                "prediction": ot_summary["prediction"],
                "reconstruction_error": ot_summary["reconstruction_error"],
                "threshold": ot_summary["threshold"],
                "anomaly_decision": ot_summary["anomaly_decision"],
                "confidence": ot_summary["confidence"],
            }
        )

    else:
        detector = service.detector("IT")
        available_configs = detector.configurations()
        if not available_configs:
            st.error("No IDS2025 feature configurations are available.")
            return

        config_labels = {
            "all_features": "79-feature mode",
            "no_ports": "77-feature mode (no ports)",
        }
        config_name = st.selectbox(
            "IT feature configuration",
            list(available_configs.keys()),
            format_func=lambda key: config_labels.get(key, key),
        )

        schema = detector.feature_schema(config_name)
        st.caption(f"{detector.name} · {config_name} · {len(schema)} ordered features")

        uploaded = st.file_uploader("Upload one-row IT CSV", type="csv")
        if uploaded is None:
            return

        frame = pd.read_csv(uploaded)
        missing = [name for name in schema if name not in frame.columns]
        if missing:
            st.error(f"Missing IT features for {config_name}: {missing[:5]}")
            return

        row = frame.iloc[0]
        values = {name: float(row[name]) for name in schema}
        timestamp = datetime.now(timezone.utc)

        try:
            probabilities = detector.predict_proba(values, config_name=config_name)
            if probabilities:
                prob_frame = pd.DataFrame(
                    {
                        "Class": list(probabilities.keys()),
                        "Probability": list(probabilities.values()),
                    }
                ).sort_values("Probability", ascending=False)
                st.subheader("Prediction confidence")
                st.plotly_chart(
                    px.bar(
                        prob_frame,
                        x="Class",
                        y="Probability",
                        color="Class",
                    ),
                    use_container_width=True,
                )
        except Exception:
            pass

        try:
            ranked = detector.feature_importance(config_name=config_name, top_n=10)
            if ranked:
                st.subheader("Top feature importance")
                st.dataframe(
                    pd.DataFrame(ranked, columns=["Feature", "Importance"]),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception:
            pass

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
                config_name=config_name if source == "IT" else None,
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

    if "timestamp" in table.columns:
        table["timestamp"] = pd.to_datetime(table["timestamp"], errors="coerce")
        table["is_stale"] = (
            table["review_status"].isin(["NEW", "IN_REVIEW"])
            & (
                pd.Timestamp.utcnow().tz_localize(None) - table["timestamp"]
                > pd.Timedelta(hours=24)
            )
        )

    st.subheader("Stale alerts")
    stale = table.get("is_stale", pd.Series(False, index=table.index))
    stale = table[stale]
    if not stale.empty:
        st.dataframe(
            stale[["event_id", "asset_id", "risk_level", "review_status", "timestamp"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No stale unresolved alerts.")

    st.subheader("Review queue")
    cols = st.columns(3)
    cols[0].metric("New", int((table["review_status"] == "NEW").sum()))
    cols[1].metric("In review", int((table["review_status"] == "IN_REVIEW").sum()))
    cols[2].metric("Resolved", int((table["review_status"] == "RESOLVED").sum()))

    display = table[
        [
            "event_id",
            "source",
            "timestamp",
            "asset_id",
            "risk_score",
            "risk_level",
            "review_status",
        ]
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)

    event_id = st.selectbox("Open alert", [event["event_id"] for event in events])
    event = next(item for item in events if item["event_id"] == event_id)

    st.json(event)

    new_status = st.selectbox("Update review status", ["NEW", "IN_REVIEW", "RESOLVED"])
    if st.button("Save review status"):
        if service.store.update_review(event_id, new_status):
            st.success("Review status updated.")
            st.rerun()


def render_alert_prioritization(service: DetectionService) -> None:
    st.title("Alert prioritization")
    events = service.store.list(limit=500)
    if not events:
        st.info("No alert history available for prioritization.")
        return

    frame = pd.DataFrame(events)
    if frame.empty:
        st.info("No events exist yet.")
        return

    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)

    frame["asset_bucket"] = frame["asset_id"].fillna("unknown")
    frame["source_bucket"] = frame["source"].fillna("unknown")
    frame["asset_id"] = frame["asset_id"].fillna("unknown")
    frame["identity_id"] = frame["identity_id"].fillna("unknown")

    st.subheader("Related alerts by asset")
    asset_groups = (
        frame.groupby(["asset_bucket", "source_bucket"], dropna=False)
        .agg(
            alerts=("event_id", "count"),
            max_risk=("risk_score", "max"),
            latest=("timestamp", "max"),
        )
        .reset_index()
        .sort_values(["alerts", "max_risk"], ascending=[False, False])
    )
    st.dataframe(asset_groups, use_container_width=True, hide_index=True)

    st.subheader("Potential duplicate clusters")
    if "timestamp" in frame.columns:
        frame_sorted = frame.sort_values("timestamp").reset_index(drop=True)
        frame_sorted["time_gap_seconds"] = (
            frame_sorted["timestamp"].diff().dt.total_seconds().fillna(0)
        )

        duplicate_candidates = frame_sorted[
            (frame_sorted["asset_id"].notna())
            & (frame_sorted["time_gap_seconds"] <= 300)
            & (frame_sorted["risk_level"].isin(["HIGH", "CRITICAL"]))
        ]

        if duplicate_candidates.empty:
            st.info("No likely alert clusters found in the current time window.")
        else:
            st.dataframe(
                duplicate_candidates[
                    [
                        "event_id",
                        "source",
                        "asset_id",
                        "timestamp",
                        "risk_score",
                        "risk_level",
                        "recommended_action",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Timestamp data is unavailable for correlation.")

    st.subheader("Highest-risk assets")
    assets = (
        frame.groupby("asset_id")
        .agg(
            events=("event_id", "count"),
            highest_risk=("risk_score", "max"),
            latest_event=("timestamp", "max"),
            affected_identities=("identity_id", lambda s: s.nunique()),
        )
        .reset_index()
        .sort_values(["highest_risk", "events"], ascending=[False, False])
    )
    st.dataframe(assets, use_container_width=True, hide_index=True)

    st.subheader("Most affected identities")
    identities = (
        frame.groupby("identity_id")
        .agg(
            events=("event_id", "count"),
            highest_risk=("risk_score", "max"),
            asset_count=("asset_id", "nunique"),
        )
        .reset_index()
        .sort_values(["highest_risk", "events"], ascending=[False, False])
    )
    st.dataframe(identities, use_container_width=True, hide_index=True)

    st.subheader("Suggested incident grouping")
    if "asset_id" in frame.columns:
        suggestions = (
            frame.groupby("asset_id")
            .agg(
                total_alerts=("event_id", "count"),
                highest_risk=("risk_score", "max"),
                latest_alert=("timestamp", "max"),
            )
            .reset_index()
            .sort_values(["total_alerts", "highest_risk"], ascending=[False, False])
        )
        st.dataframe(suggestions, use_container_width=True, hide_index=True)

    st.subheader("Recommended action by asset")
    action_summary = (
        frame.groupby(["asset_id", "recommended_action"])
        .size()
        .reset_index(name="count")
        .sort_values(["asset_id", "count"], ascending=[True, False])
    )
    st.dataframe(action_summary, use_container_width=True, hide_index=True)


def evaluate_zero_trust_decision(event: dict[str, Any]) -> dict[str, Any]:
    context = event.get("context", {})
    identity_trust = float(context.get("identity_trust", 1.0))
    device_trust = float(context.get("device_trust", 1.0))
    asset_criticality = float(context.get("asset_criticality", 0.0))
    anomaly_severity = float(context.get("anomaly_severity", 0.0))
    behavior_deviation = float(context.get("behavior_deviation", 0.0))
    risk_score = float(event.get("risk_score", 0.0))

    rule_traces: list[dict[str, str]] = []

    if anomaly_severity >= 0.80 and behavior_deviation >= 0.65 and asset_criticality >= 0.60:
        rule_traces.append(
            {
                "Rule": "R1: Coordinated attack posture",
                "Condition": "anomaly_severity >= 0.80 and behavior_deviation >= 0.65 and asset_criticality >= 0.60",
                "Evaluated": (
                    f"anomaly={anomaly_severity:.2f}, "
                    f"behavior={behavior_deviation:.2f}, "
                    f"asset={asset_criticality:.2f}"
                ),
                "Result": "TRIGGERED",
            }
        )

    if identity_trust < 0.40 and device_trust < 0.40 and risk_score >= 60:
        rule_traces.append(
            {
                "Rule": "R2: Compromised identity and device trust",
                "Condition": "identity_trust < 0.40 and device_trust < 0.40 and risk_score >= 60",
                "Evaluated": (
                    f"identity={identity_trust:.2f}, "
                    f"device={device_trust:.2f}, "
                    f"risk={risk_score:.1f}"
                ),
                "Result": "TRIGGERED",
            }
        )

    if risk_score >= 50 or identity_trust < 0.60 or device_trust < 0.60:
        rule_traces.append(
            {
                "Rule": "R3: Elevated access risk",
                "Condition": "risk_score >= 50 or identity_trust < 0.60 or device_trust < 0.60",
                "Evaluated": (
                    f"risk={risk_score:.1f}, "
                    f"identity={identity_trust:.2f}, "
                    f"device={device_trust:.2f}"
                ),
                "Result": "TRIGGERED",
            }
        )

    if (
        anomaly_severity >= 0.80
        and behavior_deviation >= 0.65
        and asset_criticality >= 0.60
    ) or (
        identity_trust < 0.40 and device_trust < 0.40 and risk_score >= 60
    ) or risk_score >= 90:
        verdict = "DENY"
        reason = (
            "Critical attack posture detected. The event combines abnormal behavior, "
            "high asset criticality, and weakened identity/device trust. Access must be blocked."
        )
    elif risk_score >= 50 or identity_trust < 0.60 or device_trust < 0.60:
        verdict = "REQUIRE VERIFICATION"
        reason = (
            "Potential security risk detected. Re-authentication, identity validation, "
            "and additional behavioral review are required before access continues."
        )
    else:
        verdict = "ALLOW"
        reason = (
            "Contextual risk is within acceptable bounds. Identity, device posture, "
            "and behavior indicators are consistent with trusted operations."
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "risk_score": risk_score,
        "rule_traces": rule_traces,
    }


def render_investigation(service: DetectionService) -> None:
    st.title("Investigation workflow")
    events = service.store.list(limit=500)
    if not events:
        st.info("No events are available for analysis yet.")
        return

    event_id = st.selectbox(
        "Select alert for investigation",
        [event["event_id"] for event in events],
    )
    event = next(item for item in events if item["event_id"] == event_id)

    decision = evaluate_zero_trust_decision(event)

    st.subheader("Alert summary")
    cols = st.columns(5)
    cols[0].metric("Source", event["source"])
    cols[1].metric("Prediction", event["prediction"])
    cols[2].metric("Risk score", f"{event['risk_score']:.1f}")
    cols[3].metric("Risk level", event["risk_level"])
    cols[4].metric("Zero-Trust verdict", decision["verdict"])

    threat_tab, evidence_tab, analyst_tab = st.tabs(
        ["Threat review", "Evidence", "Analyst notes"]
    )

    with threat_tab:
        st.write(f"Recommended action: {event['recommended_action']}")
        st.write(f"Decision reason: {decision['reason']}")

        if decision["rule_traces"]:
            st.caption("Decision trace")
            st.dataframe(
                pd.DataFrame(decision["rule_traces"]),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Contribution breakdown")
        components = event.get("risk_components", {})
        if components:
            st.plotly_chart(
                px.bar(
                    pd.Series(components)
                    .reset_index()
                    .rename(columns={"index": "Factor", 0: "Contribution"}),
                    x="Factor",
                    y="Contribution",
                ),
                use_container_width=True,
            )
        else:
            st.info("No risk component data is attached to this event.")

    with evidence_tab:
        context = event.get("context", {})
        context_df = pd.DataFrame(
            {
                "Signal": [
                    "Identity trust",
                    "Device trust",
                    "Asset criticality",
                    "Anomaly severity",
                    "Behavior deviation",
                ],
                "Value": [
                    context.get("identity_trust", 0.0),
                    context.get("device_trust", 0.0),
                    context.get("asset_criticality", 0.0),
                    context.get("anomaly_severity", 0.0),
                    context.get("behavior_deviation", 0.0),
                ],
            }
        )
        st.dataframe(context_df, use_container_width=True, hide_index=True)

        st.subheader("Operational payload")
        st.json(
            {
                "event_id": event["event_id"],
                "source": event["source"],
                "detector": event["detector"],
                "timestamp": event["timestamp"],
                "asset_id": event["asset_id"],
                "device_id": event["device_id"],
                "identity_id": event.get("identity_id"),
                "prediction": event["prediction"],
                "anomaly_score": event["anomaly_score"],
                "threshold": event["threshold"],
                "confidence": event["confidence"],
                "risk_score": event["risk_score"],
                "risk_level": event["risk_level"],
                "recommended_action": event["recommended_action"],
                "context": context,
                "risk_components": event.get("risk_components", {}),
                "review_status": event["review_status"],
            }
        )

    with analyst_tab:
        st.caption("Analyst triage")
        triage_verdict = st.selectbox(
            "Investigation verdict",
            ["Confirmed attack", "Benign false positive", "Requires validation", "Investigate later"],
        )

        note_key = f"investigation_note_{event_id}"
        note_value = st.session_state.get(note_key, "")

        analyst_note = st.text_area(
            "Investigation notes",
            value=note_value,
            placeholder="Document the reason, related asset, and recommended response.",
        )

        if st.button("Save analyst note"):
            st.session_state[note_key] = analyst_note
            if triage_verdict:
                st.session_state[f"investigation_verdict_{event_id}"] = triage_verdict
            st.success("Investigation note saved for this alert.")

        saved_verdict = st.session_state.get(f"investigation_verdict_{event_id}")
        if saved_verdict:
            st.info(f"Saved verdict: {saved_verdict}")

        if event.get("source") == "OT":
            st.markdown(
                """
                <div style='padding: 12px; border-left: 4px solid #f59e0b; background: #1f2937; border-radius: 8px;'>
                    <strong>OT triage guidance:</strong>
                    Check whether the anomaly is a sensor deviation, a process fault, or an active cyber-physical attack.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style='padding: 12px; border-left: 4px solid #38bdf8; background: #0f172a; border-radius: 8px;'>
                    <strong>IT triage guidance:</strong>
                    Confirm whether the event is a real intrusion, a model-induced false alarm, or a partially trusted traffic anomaly.
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_risk(service: DetectionService) -> None:
    st.title("Risk assessment")
    events = service.store.list(limit=500)
    if not events:
        st.info("Score an observation to inspect contextual risk.")
        return

    event_id = st.selectbox("Select event", [event["event_id"] for event in events])
    event = next(item for item in events if item["event_id"] == event_id)

    decision = evaluate_zero_trust_decision(event)

    cards = st.columns(4)
    cards[0].metric("Risk score", f"{event['risk_score']:.1f}/100")
    cards[1].metric("Risk level", event["risk_level"])
    cards[2].metric("Recommended action", event["recommended_action"])
    cards[3].metric("Zero-Trust verdict", decision["verdict"])

    st.subheader("Zero-Trust decision")
    st.info(f"{decision['verdict']}: {decision['reason']}")
    if decision["rule_traces"]:
        st.caption("Decision trace")
        st.dataframe(
            pd.DataFrame(decision["rule_traces"]),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Risk components")
    components = pd.DataFrame(
        {
            "Factor": list(event.get("risk_components", {}).keys()),
            "Contribution": list(event.get("risk_components", {}).values()),
        }
    )
    if not components.empty:
        st.plotly_chart(
            px.bar(
                components,
                x="Factor",
                y="Contribution",
                color="Factor",
            ),
            use_container_width=True,
        )

    st.subheader("Context signals")
    context = event.get("context", {})
    context_df = pd.DataFrame(
        {
            "Signal": [
                "Identity trust",
                "Device trust",
                "Asset criticality",
                "Anomaly severity",
                "Behavior deviation",
            ],
            "Value": [
                context.get("identity_trust", 0.0),
                context.get("device_trust", 0.0),
                context.get("asset_criticality", 0.0),
                context.get("anomaly_severity", 0.0),
                context.get("behavior_deviation", 0.0),
            ],
        }
    )
    st.dataframe(context_df, use_container_width=True, hide_index=True)

    st.subheader("Full event payload")
    st.json(
        {
            "event_id": event["event_id"],
            "source": event["source"],
            "detector": event["detector"],
            "prediction": event["prediction"],
            "risk_score": event["risk_score"],
            "risk_level": event["risk_level"],
            "recommended_action": event["recommended_action"],
            "context": event.get("context", {}),
            "risk_components": event.get("risk_components", {}),
            "zero_trust_decision": decision,
        }
    )


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

    model_input = profile.get("model_input", {})
    training = profile.get("training", {})
    test = profile.get("test", {})

    cards = st.columns(4)
    cards[0].metric("Model features", model_input.get("feature_count", 0))
    cards[1].metric("Training rows", f"{training.get('rows', 0):,}")
    cards[2].metric("Test rows", f"{test.get('rows', 0):,}")
    cards[3].metric("F1", f"{float(metrics.get('f1', 0.0)):.4f}")

    st.subheader("Operational evaluation summary")
    summary_rows = [
        ("Threshold", metrics.get("threshold", 0.0)),
        ("Precision", metrics.get("precision", 0.0)),
        ("Recall", metrics.get("recall", 0.0)),
        ("F1", metrics.get("f1", 0.0)),
        ("False positive rate", metrics.get("false_positive_rate", 0.0)),
        ("ROC AUC", metrics.get("roc_auc", 0.0)),
        ("PR AUC", metrics.get("pr_auc", 0.0)),
    ]
    summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Confusion matrix")
    confusion = metrics.get("confusion_matrix", {})
    confusion_df = pd.DataFrame(
        {
            "Outcome": ["True negative", "False positive", "False negative", "True positive"],
            "Count": [
                confusion.get("tn", 0),
                confusion.get("fp", 0),
                confusion.get("fn", 0),
                confusion.get("tp", 0),
            ],
        }
    )
    st.dataframe(confusion_df, use_container_width=True, hide_index=True)

    with st.expander("Evaluation diagnostics"):
        st.json(metrics)

    with st.expander("Dataset feature dictionary"):
        feature_rows = profile.get("features", [])
        if feature_rows:
            feature_table = pd.DataFrame(feature_rows)
            st.dataframe(feature_table, use_container_width=True, hide_index=True)
        else:
            st.info("No feature dictionary is available for this dataset profile.")

    with st.expander("Feature groups"):
        groups = profile.get("feature_groups", [])
        if groups:
            st.dataframe(pd.DataFrame(groups), use_container_width=True, hide_index=True)
        else:
            st.info("No feature group metadata is available.")

    with st.expander("Operational notes"):
        st.write(profile.get("label_usage", "No dataset label usage note available."))
        st.write(profile.get("dataset", "Unknown dataset"))
        st.write(profile.get("source", "Unknown source"))


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
    st.info(
        "Configuration is read from RESINET_* environment variables. Restart after changing artifacts or paths."
    )


if __name__ == "__main__":
    main()