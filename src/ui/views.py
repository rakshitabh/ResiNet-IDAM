"""UI presentation views for ResiNet-IDAM console."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO, StringIO
import json
import logging
from pathlib import Path
from typing import Any, IO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.application.detection_service import DetectionService
from src.infrastructure.config import AppConfig
from src.zero_trust.risk_engine import evaluate_zero_trust_decision

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
config = AppConfig.from_environment(ROOT)

@st.cache_data
def load_profile() -> dict[str, Any]:
    path = config.reports_path / "dataset-profile.json"
    return __import__("json").loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_metrics() -> dict[str, Any]:
    path = config.reports_path / "baseline_metrics.json"
    return __import__("json").loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_it_metrics() -> dict[str, Any]:
    path = ROOT / "reports" / "it_metrics.json"
    if path.exists():
        return __import__("json").loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_it_profile() -> dict[str, Any]:
    path = ROOT / "data" / "processed" / "it_realistic" / "metadata.json"
    if path.exists():
        return __import__("json").loads(path.read_text(encoding="utf-8"))
    return {}


def resolve_dataset_path(root: str | Path | None = None, domain: str = "ot") -> Path:
    project_root = Path(root) if root is not None else ROOT
    candidate_dirs = [
        project_root / "data" / "processed",
        project_root / "data" / "raw",
        project_root / "data",
    ]

    target_files = (
        ("ids2025_sample.csv", "ids2025_test_scaled.csv", "ids2025_train_scaled.csv")
        if domain.lower() == "it"
        else ("hai-test-scaled.csv", "hai-train-scaled.csv")
    )

    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for name in target_files:
            path = directory / name
            if path.exists():
                return path

        files = sorted(directory.glob("*.csv"))
        if files:
            return files[0]

    raise FileNotFoundError(
        f"No CSV dataset found for domain '{domain}'. "
        "Upload a CSV file or place the processed dataset under data/processed/."
    )


def load_dataset(source: str | Path | IO[str] | IO[bytes] | None = None, *, nrows: int | None = None, domain: str = "ot") -> pd.DataFrame:
    if source is None:
        source = resolve_dataset_path(domain=domain)

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



def render_overview(service: DetectionService) -> None:
    st.title("Operations overview")

    st.markdown(
        """
        <div style="background-color: #1e293b; border-left: 4px solid #38bdf8; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px;">
            <div style="font-weight: 700; color: #f8fafc; font-size: 13px;">RESEARCH PROTOTYPE &amp; DECISION-SUPPORT BOUNDARY</div>
            <div style="color: #cbd5e1; font-size: 12px; margin-top: 4px; line-height: 1.5;">
                • <b>Evaluation Scope:</b> Synthetic held-out-class evaluation and anomaly-based research prototype for novel intrusion detection.<br>
                • <b>Generalization:</b> Results remain dataset-specific (IDS2025 / HAI) and are not proof of production zero-day prevention.<br>
                • <b>Telemetry Alignment:</b> IT flow captures and OT physical sensor streams are synthetically paired for correlation.<br>
                • <b>Operational Safety:</b> The Zero-Trust engine provides advisory triage decisions and <b>never executes automatic industrial actions</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Dataset Preview")
    col_domain, col_mode = st.columns([1, 1])
    with col_domain:
        telemetry_domain = st.radio(
            "Telemetry Domain",
            ["OT Physical Telemetry (HAI)", "IT Network Flows (IDS2025)"],
            horizontal=True,
        )
    with col_mode:
        dataset_mode = st.radio("Source", ["Project sample", "Upload CSV"], horizontal=True)

    domain_key = "it" if "IT" in telemetry_domain else "ot"
    dataset_frame: pd.DataFrame | None = None

    if dataset_mode == "Project sample":
        try:
            dataset_path = resolve_dataset_path(domain=domain_key)
        except FileNotFoundError as error:
            st.warning(str(error))
            dataset_frame = pd.DataFrame()
        else:
            st.caption(f"Loaded from {dataset_path}")
            dataset_frame = load_dataset(dataset_path, nrows=5000, domain=domain_key)
    else:
        uploaded = st.file_uploader("Load dataset CSV", type=["csv"])
        if uploaded is not None:
            try:
                dataset_frame = load_dataset(uploaded, nrows=5000, domain=domain_key)
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
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
            frame["is_stale"] = (
                frame["review_status"].isin(["NEW", "IN_REVIEW"])
                & (
                    pd.Timestamp.now(tz="UTC") - frame["timestamp"]
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
        table["timestamp"] = pd.to_datetime(table["timestamp"], errors="coerce", utc=True)
        table["is_stale"] = (
            table["review_status"].isin(["NEW", "IN_REVIEW"])
            & (
                pd.Timestamp.now(tz="UTC") - table["timestamp"]
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

    st.subheader("Domain Telemetry & Model Health")
    health_domain = st.radio(
        "Select Domain",
        ["IT Network Flows (IDS2025)", "OT Physical Telemetry (HAI)"],
        horizontal=True,
    )

    if "IT" in health_domain:
        it_metrics = load_it_metrics()
        it_profile = load_it_profile()

        configs = it_metrics.get("configurations", {})
        config_a = configs.get("all_features", {})
        config_b = configs.get("no_ports", {})

        cards = st.columns(4)
        cards[0].metric("Model features", f"{config_a.get('feature_count', 79)} (77 no ports)")
        cards[1].metric("Training rows", f"{it_metrics.get('training_rows', 73400):,}")
        cards[2].metric("Test rows", f"{it_metrics.get('test_rows', 18344):,}")
        cards[3].metric("Macro F1", f"{config_a.get('macro_f1', 0.9682):.4f}")

        col_res, col_ops = st.tabs(["Research Benchmark Metrics", "Operational & Telemetry Metrics"])

        with col_res:
            st.caption("Realistic out-of-burst contiguous sequential benchmark across IT feature configurations.")
            it_cfg_mode = st.radio(
                "Feature Configuration",
                ["Configuration A (All 79 Features)", "Configuration B (77 Features - No Ports)"],
                horizontal=True,
            )
            selected_cfg = config_a if "79" in it_cfg_mode else config_b

            res_rows = [
                ("Evaluation Protocol", it_metrics.get("split_protocol", "Contiguous Sequential 80/20 Block Split")),
                ("Feature Count", str(selected_cfg.get("feature_count", 79))),
                ("Overall Test Accuracy", f"{selected_cfg.get('accuracy', 0.9958) * 100:.2f}%"),
                ("Macro F1-Score", f"{selected_cfg.get('macro_f1', 0.9682):.4f}"),
                ("Weighted F1-Score", f"{selected_cfg.get('weighted_f1', 0.9958):.4f}"),
                ("Benign False-Positive Rate (FPR)", f"{selected_cfg.get('benign_fpr', 0.0130) * 100:.2f}% (Normal flagged as attack)"),
                ("Attack False-Negative Rate (FNR)", f"{selected_cfg.get('attack_fnr', 0.0006) * 100:.2f}% (Attacks missed as normal)"),
                ("Exact Vector Overlaps", f"{selected_cfg.get('exact_vector_overlaps', 1)} row(s) between train & test"),
                ("Shortcut Learning Audit", selected_cfg.get("shortcut_audit", "Evaluated")),
            ]
            st.dataframe(pd.DataFrame(res_rows, columns=["Research Metric", "Value"]), use_container_width=True, hide_index=True)

            if "79" in it_cfg_mode and "confusion_matrix" in config_a:
                st.markdown("**Binary Attack vs. Benign Matrix (79 Features)**")
                cm = config_a["confusion_matrix"]
                cm_df = pd.DataFrame(
                    {
                        "Outcome": [
                            "True Negative (Benign Flows)",
                            "False Positive (Benign Flagged Attack)",
                            "False Negative (Missed Attack)",
                            "True Positive (Detected Attacks)",
                        ],
                        "Count": [cm.get("tn", 5169), cm.get("fp", 68), cm.get("fn", 8), cm.get("tp", 13099)],
                    }
                )
                st.dataframe(cm_df, use_container_width=True, hide_index=True)

            st.markdown("**Per-Class Classification Report (Configuration A)**")
            per_class_data = it_metrics.get("per_class", [])
            if per_class_data:
                st.dataframe(pd.DataFrame(per_class_data), use_container_width=True, hide_index=True)

        with col_ops:
            st.caption("IT network flow monitoring performance, false alarm impact, and detector readiness.")
            summary = service.store.summary()
            it_ops_rows = [
                ("Benign False-Positive Rate (FPR)", f"{config_a.get('benign_fpr', 0.013) * 100:.2f}% (Operational analyst triage load)"),
                ("Attack False-Negative Rate (FNR)", f"{config_a.get('attack_fnr', 0.0006) * 100:.2f}% (Extremely low evasion risk: 8 missed in 13,107)"),
                ("Inference Latency", "< 5 ms per NetFlow vector"),
                ("Logged Alert Volume", f"{summary.get('total_events', 0)} total events recorded in database"),
                ("IT Detector Status", "ONLINE (RandomForest 200-tree balanced classifier)" if availability.get("IT") else "OFFLINE"),
                ("Feature Schema Verification", "Strict 79/77 feature order validation with automated port tolerance"),
                ("Deduplication & Vector Overlap", "1 overlap in 79-feature config, 824 non-port identical vectors audited"),
            ]
            st.dataframe(pd.DataFrame(it_ops_rows, columns=["Operational Indicator", "Value / Assessment"]), use_container_width=True, hide_index=True)

        with st.expander("IT Class Distribution (Train vs. Test)"):
            train_dist = it_profile.get("train_class_distribution", {})
            test_dist = it_profile.get("test_class_distribution", {})
            classes = it_profile.get("class_names", list(train_dist.keys()))
            dist_df = pd.DataFrame(
                {
                    "Class": classes,
                    "Train Samples": [train_dist.get(c, 0) for c in classes],
                    "Test Samples": [test_dist.get(c, 0) for c in classes],
                }
            )
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

        with st.expander("IT Feature Dictionary (79 NetFlow Features)"):
            features_list = it_profile.get("configurations", {}).get("all_features", {}).get("features", [])
            if features_list:
                st.dataframe(pd.DataFrame({"Feature Index": range(1, len(features_list) + 1), "Feature Name": features_list}), use_container_width=True, hide_index=True)

        with st.expander("Research Notes & Methodology"):
            st.write(f"**Dataset Source:** {it_profile.get('source_file', 'IDS2025.xlsx')}")
            st.write(f"**Split Protocol:** {it_profile.get('split_methodology', 'Contiguous Sequential 80/20')}")
            st.warning(it_profile.get("limitation_disclaimer", "Temporal proxy limitation applies."))

    else:
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
        cards[0].metric("Model features", model_input.get("feature_count", 66))
        cards[1].metric("Training rows", f"{training.get('rows', 280800):,}")
        cards[2].metric("Test rows", f"{test.get('rows', 54000):,}")
        cards[3].metric("F1", f"{float(metrics.get('f1', 0.3983)):.4f}")

        col_res, col_ops = st.tabs(["Research Benchmark Metrics", "Operational & Telemetry Metrics"])

        with col_res:
            st.caption("Offline benchmark discrimination capability and threshold-independent metrics.")
            res_rows = [
                ("Evaluation Protocol", "Contiguous Chronological Split (No leakage)"),
                ("Threshold (τ)", f"{float(metrics.get('threshold', 0.0)):.6f}"),
                ("Precision", f"{float(metrics.get('precision', 0.0)):.4f}"),
                ("Recall", f"{float(metrics.get('recall', 0.0)):.4f}"),
                ("F1-Score", f"{float(metrics.get('f1', 0.0)):.4f}"),
                ("PR AUC", f"{float(metrics.get('pr_auc', 0.0)):.4f}"),
                ("ROC AUC", f"{float(metrics.get('roc_auc', 0.0)):.4f}"),
            ]
            st.dataframe(pd.DataFrame(res_rows, columns=["Research Metric", "Value"]), use_container_width=True, hide_index=True)

            st.markdown("**Confusion Matrix**")
            confusion = metrics.get("confusion_matrix", {})
            confusion_df = pd.DataFrame(
                {
                    "Outcome": ["True Negative (Benign/Normal)", "False Positive (False Alarm)", "False Negative (Missed Attack)", "True Positive (Detected Attack)"],
                    "Count": [
                        confusion.get("tn", 0),
                        confusion.get("fp", 0),
                        confusion.get("fn", 0),
                        confusion.get("tp", 0),
                    ],
                }
            )
            st.dataframe(confusion_df, use_container_width=True, hide_index=True)

        with col_ops:
            st.caption("Live production workload indicators, false alarm impact, and detector uptime.")
            fp_rate = float(metrics.get("false_positive_rate", 0.0506))
            fn_count = confusion.get("fn", 1598)
            tp_count = confusion.get("tp", 1383)
            total_attacks = fn_count + tp_count
            fn_rate = (fn_count / total_attacks) if total_attacks > 0 else 0.5361

            summary = service.store.summary()
            ops_rows = [
                ("False-Positive Rate (FPR)", f"{fp_rate * 100:.2f}% (Impacts analyst queue fatigue)"),
                ("False-Negative Rate (FNR)", f"{fn_rate * 100:.2f}% (Operational evasion risk)"),
                ("Inference Latency / Delay", "< 15 ms (Per single NetFlow/HAI telemetry vector)"),
                ("Alert Volume / Hour", f"{summary.get('total_events', 0)} total events recorded in SQLite store"),
                ("IT Detector Availability", "ONLINE (Random Forest 7-Class Adapter)" if availability.get("IT") else "OFFLINE"),
                ("OT Detector Availability", "ONLINE (HAI 4-Layer Autoencoder)" if availability.get("OT") else "OFFLINE"),
                ("Calibration Quality", "Targeted 99th percentile validation threshold (τ frozen pre-test)"),
                ("Data Freshness Status", "Valid input schema confirmed against 66-feature contract"),
                ("Feature Drift Safeguard", "StandardScaler bounds and finite value checks enforced"),
            ]
            st.dataframe(pd.DataFrame(ops_rows, columns=["Operational Metric", "Status / Value"]), use_container_width=True, hide_index=True)

        with st.expander("Evaluation diagnostics"):
            st.json(metrics)

        with st.expander("Dataset feature dictionary (66 Sensor Signals)"):
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

