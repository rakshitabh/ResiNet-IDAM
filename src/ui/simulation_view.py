"""Interactive Zero-Trust Simulation Console view for Streamlit."""

from __future__ import annotations

import io
from pathlib import Path
import time
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.application.detection_service import DetectionService
from src.simulation.data.csv_parser import SimulationCsvParser
from src.simulation.engine.controller import SimulationController
from src.simulation.models.simulation_event import EventState
from src.simulation.models.simulation_run import RunState
from src.simulation.reporting.csv_report import CsvReportGenerator
from src.simulation.reporting.json_report import JsonReportGenerator


def get_controller(service: DetectionService) -> SimulationController:
    if "sim_controller" not in st.session_state:
        st.session_state["sim_controller"] = SimulationController(
            root=service.root,
            store=None,
        )
    return st.session_state["sim_controller"]


def render_simulation_view(service: DetectionService) -> None:
    st.markdown(
        """
        <style>
        .sim-header { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.2rem; }
        .sim-sub { color: #8892b0; font-size: 0.95rem; margin-bottom: 1.2rem; }
        .metric-card {
            background-color: #112240;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
            border: 1px solid #233554;
        }
        .metric-title { font-size: 0.8rem; color: #8892b0; text-transform: uppercase; }
        .metric-value { font-size: 1.8rem; font-weight: 700; margin-top: 4px; }
        .badge-allow { background-color: #0f5132; color: #75b798; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
        .badge-require { background-color: #664d03; color: #ffda6a; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
        .badge-deny { background-color: #842029; color: #ea868f; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
        .badge-review { background-color: #3d0066; color: #c299ff; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
        .safety-note {
            background-color: rgba(13, 110, 253, 0.1);
            border-left: 4px solid #0d6efd;
            padding: 8px 12px;
            margin-top: 10px;
            font-size: 0.85rem;
            color: #90caf9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sim-header">▶ Zero-Trust Simulation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sim-sub">CSV-Driven Zero-Trust | Industrial Incident Progression | 5-Factor Risk Assessment</div>',
        unsafe_allow_html=True,
    )

    controller = get_controller(service)
    scenarios_dir = service.root / "data" / "simulation" / "scenarios"
    available_scenarios = sorted([p.name for p in scenarios_dir.glob("*.csv")])
    if not available_scenarios and (service.root / "data" / "simulation" / "zero_trust_simulation_scenarios.csv").exists():
        available_scenarios = ["zero_trust_simulation_scenarios.csv"]

    col_dataset, col_status, col_time = st.columns([2.5, 1.2, 2.3])

    with col_dataset:
        selected_file = st.selectbox(
            "Simulation Dataset",
            available_scenarios + ["Upload custom CSV..."],
            index=0 if available_scenarios else 0,
        )

    # Handle scenario load
    if selected_file == "Upload custom CSV...":
        uploaded = st.file_uploader("Upload CSV scenario file (Max 10MB)", type=["csv"])
        if uploaded is not None:
            raw_bytes = uploaded.read()
            try:
                events, d_hash = SimulationCsvParser.parse(raw_bytes)
                if controller.current_run is None or controller.current_run.dataset_name != uploaded.name:
                    controller.load_scenario(events, uploaded.name, d_hash)
                    st.success(f"Loaded {len(events)} events from {uploaded.name}")
            except Exception as ex:
                st.error(f"Validation Error: {ex}")
    else:
        file_path = scenarios_dir / selected_file if (scenarios_dir / selected_file).exists() else service.root / "data" / "simulation" / selected_file
        if file_path.exists():
            if controller.current_run is None or controller.current_run.dataset_name != selected_file:
                with open(file_path, "rb") as f:
                    content = f.read()
                try:
                    events, d_hash = SimulationCsvParser.parse(content)
                    controller.load_scenario(events, selected_file, d_hash)
                except Exception as ex:
                    st.error(f"Dataset parse error: {ex}")

    run = controller.current_run
    current_state = run.state.value if run else "READY"
    current_elapsed = controller.clock.elapsed_sim_seconds()
    max_duration = max((e.offset_seconds for e in controller._raw_events), default=60.0)
    progress_pct = min(1.0, current_elapsed / max_duration) if max_duration > 0 else 0.0

    with col_status:
        st.write("")
        st.write("")
        state_color = "#198754" if current_state == "RUNNING" else ("#ffc107" if current_state == "PAUSED" else "#0dcaf0")
        st.markdown(
            f'<div style="text-align:center; padding: 6px; background-color: {state_color}; color: #000; border-radius: 20px; font-weight: 700;">{current_state}</div>',
            unsafe_allow_html=True,
        )

    with col_time:
        time_str = f"Elapsed Time: {controller.clock.format_time(current_elapsed)} / {controller.clock.format_time(max_duration)}"
        st.write("")
        st.markdown(f"**{time_str}**")
        st.progress(progress_pct)

    # Controls Row
    c_start, c_pause, c_resume, c_stop, c_restart, c_speed = st.columns([1, 1, 1, 1, 1, 2])
    with c_start:
        if st.button("▶ Start", use_container_width=True, disabled=(current_state in ("RUNNING", "COMPLETED"))):
            controller.start()
            st.rerun()
    with c_pause:
        if st.button("⏸ Pause", use_container_width=True, disabled=(current_state != "RUNNING")):
            controller.pause()
            st.rerun()
    with c_resume:
        if st.button("⏵ Resume", use_container_width=True, disabled=(current_state != "PAUSED")):
            controller.resume()
            st.rerun()
    with c_stop:
        if st.button("⏹ Stop", use_container_width=True, disabled=(current_state not in ("RUNNING", "PAUSED"))):
            controller.stop()
            st.rerun()
    with c_restart:
        if st.button("🔄 Restart", use_container_width=True):
            controller.restart()
            st.rerun()
    with c_speed:
        speed_val = st.radio("Playback Speed:", [1.0, 2.0, 5.0, 10.0], horizontal=True, format_func=lambda x: f"{int(x)}x")
        controller.set_speed(speed_val)

    # Process ticks if running
    if current_state == "RUNNING":
        controller.tick()

    st.divider()

    # 6 Metric Cards
    total_ev = run.total_events if run else 0
    proc_ev = len(run.results) if run else 0
    allow_cnt = run.allow_count if run else 0
    req_cnt = run.require_verification_count if run else 0
    deny_cnt = run.deny_count if run else 0
    alert_cnt = run.alert_investigation_count if run else 0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Events</div><div class="metric-value" style="color:#64ffda;">{total_ev}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Processed</div><div class="metric-value" style="color:#00e5ff;">{proc_ev}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">ALLOW</div><div class="metric-value" style="color:#00e676;">{allow_cnt}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">REQUIRE VERIFICATION</div><div class="metric-value" style="color:#ffd600;">{req_cnt}</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="metric-card"><div class="metric-title">DENY / REVOKE</div><div class="metric-value" style="color:#ff1744;">{deny_cnt}</div></div>', unsafe_allow_html=True)
    with m6:
        st.markdown(f'<div class="metric-card"><div class="metric-title">ALERT / INVESTIGATION</div><div class="metric-value" style="color:#d500f9;">{alert_cnt}</div></div>', unsafe_allow_html=True)

    st.write("")

    # Event Timeline Table
    st.subheader("Event Timeline (Full Industrial Incident Progression)")
    if controller.scheduler:
        table_rows = []
        for e in controller.scheduler.events:
            status_symbol = "⚪"
            if e.state == EventState.COMPLETED:
                status_symbol = "✅"
            elif e.state == EventState.RUNNING:
                status_symbol = "▶"
            elif e.state == EventState.FAILED:
                status_symbol = "❌"
            elif e.state == EventState.CANCELLED:
                status_symbol = "⏹"

            exp_badge = e.expected_outcome or "N/A"
            table_rows.append({
                "Status": status_symbol,
                "Event ID": e.event_id,
                "Offset": controller.clock.format_time(e.offset_seconds),
                "Source": e.source.value,
                "Event Type": e.event_type,
                "User ID": e.user_id,
                "Device ID": e.device_id,
                "Resource": e.asset_id,
                "Expected ZT Outcome": exp_badge,
                "Scenario Context": e.context or f"Execution of {e.event_type}",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Bottom 3 Panels
    b1, b2, b3 = st.columns([1.5, 1.2, 1.3])

    with b1:
        st.markdown("##### 📈 Simulation Progress & Risk Trend")
        if run and run.results:
            trend_df = pd.DataFrame([
                {"Time": r.scheduled_offset, "Risk Score": r.risk_score}
                for r in run.results
            ])
            fig_trend = px.area(
                trend_df,
                x="Time",
                y="Risk Score",
                range_y=[0, 100],
                markers=True,
                color_discrete_sequence=["#ff1744"],
            )
            fig_trend.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8892b0"),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Risk trend will plot as events execute.")

    with b2:
        st.markdown(f"##### 🍩 Zero-Trust Decisions (Processed: {proc_ev})")
        if proc_ev > 0:
            dist_data = {
                "ALLOW": allow_cnt,
                "REQUIRE VERIFICATION": req_cnt,
                "DENY / REVOKE": deny_cnt,
                "ALERT / INVESTIGATION": alert_cnt,
            }
            labels = [k for k, v in dist_data.items() if v > 0]
            values = [v for k, v in dist_data.items() if v > 0]
            colors = {
                "ALLOW": "#00e676",
                "REQUIRE VERIFICATION": "#ffd600",
                "DENY / REVOKE": "#ff1744",
                "ALERT / INVESTIGATION": "#d500f9",
            }
            color_seq = [colors[l] for l in labels]

            fig_donut = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=color_seq),
            )])
            fig_donut.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(font=dict(size=10, color="#8892b0")),
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("Decision distribution will render once events are processed.")

    with b3:
        st.markdown("##### ℹ️ Current Event Insight")
        if run and run.results:
            latest = run.results[-1]
            st.markdown(
                f"""
                <div style="background-color: #112240; padding: 12px; border-radius: 8px; border: 1px solid #233554;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="background-color:#00e5ff; color:#000; padding:2px 8px; border-radius:4px; font-weight:700;">{latest.event_id}</span>
                        <span style="font-weight:600; color:#fff;">{latest.event_type} @ {latest.scheduled_time_iso}</span>
                    </div>
                    <div style="margin-top:8px; font-size:0.85rem; color:#ccd6f6;">
                        <div><b>User:</b> {latest.user_id} &nbsp;|&nbsp; <b>Device:</b> {latest.device_id}</div>
                        <div><b>Resource:</b> {latest.asset_id}</div>
                        <div><b>Risk Score:</b> <span style="color:#ff1744; font-weight:bold;">{latest.risk_score}</span> ({latest.risk_level})</div>
                        <div><b>Verdict:</b> {latest.policy_verdict}</div>
                        <div><b>Action:</b> {latest.recommended_action}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if latest.top_contributing_features:
                st.caption(f"Top Deviating Sensor: {latest.top_contributing_features[0].get('feature_name', '')}")
        else:
            st.info("No event processed yet.")

    # Safety Notice Banner (Rule 10)
    st.markdown(
        """
        <div class="safety-note">
            <b>SIMULATED DECISION SUPPORT ONLY</b>: All Zero-Trust verdicts and recommended containment actions are advisory.
            No network firewall rules, PLC commands, or physical actuator disconnections are executed (Rule 10 Safety Guarantee).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Download Reports Section
    if run and run.results:
        st.write("")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_content = CsvReportGenerator.generate(run)
            st.download_button(
                "📥 Download CSV Execution Report",
                data=csv_content,
                file_name=f"{run.run_id}_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_d2:
            json_content = JsonReportGenerator.generate(run)
            st.download_button(
                "📥 Download JSON Audit Report",
                data=json_content,
                file_name=f"{run.run_id}_report.json",
                mime="application/json",
                use_container_width=True,
            )

    # Continuous rerun if running to refresh clock
    if current_state == "RUNNING":
        time.sleep(0.5)
        st.rerun()

