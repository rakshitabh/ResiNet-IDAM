"""
ResiNet-IDAM: Page 3 - Zero-Trust Decision Simulator
---------------------------------------------------
A transparent Zero-Trust Policy Decision Engine simulator.
Evaluates subject identity, device posture, target resource criticality,
network boundary, and observed cyber-physical risk to render:
- ALLOW
- REQUIRE VERIFICATION (Step-Up MFA / Operational Approval)
- DENY
Along with explicit, step-by-step policy rule reasons.
"""

import textwrap
import streamlit as st
import numpy as np
import pandas as pd

def evaluate_zero_trust_policy(user, device, resource, network, p_attack, S_ot, R_cp):
    """
    Evaluates transparent contextual Zero-Trust policy rules.
    Returns: verdict ("ALLOW", "REQUIRE VERIFICATION", "DENY"), primary_reason, rule_traces
    """
    rule_traces = []

    # Rule 1: Coordinated Cyber-Physical Attack Lockout
    r1_triggered = (R_cp >= 0.50 and "Level 1" in resource)
    rule_traces.append({
        "Rule": "R1: Cyber-Physical Threat Lockout",
        "Condition": "R_cp >= 0.50 AND Target = Level 1 PLC Control",
        "Evaluated": f"R_cp = {R_cp:.2f}, Resource = {resource.split(':')[0]}",
        "Result": "TRIGGERED (DENY)" if r1_triggered else "PASS"
    })
    if r1_triggered:
        return "DENY", "Critical coordinated cyber-physical attack detected (R_cp >= 0.50). Physical actuators locked out to prevent catastrophic damage.", rule_traces

    # Rule 2: Active Physical Process Anomaly Lockout
    r2_triggered = (S_ot >= 0.50 and "Level 1" in resource and network != "Internal OT Control Room (Air-Gapped VLAN)")
    rule_traces.append({
        "Rule": "R2: Remote PLC Lockout on Physical Deviation",
        "Condition": "S_ot >= 0.50 AND Target = Level 1 AND Location != OT Control Room",
        "Evaluated": f"S_ot = {S_ot:.2f}, Network = {network}",
        "Result": "TRIGGERED (DENY)" if r2_triggered else "PASS"
    })
    if r2_triggered:
        return "DENY", "Active physical process anomaly detected in OT telemetry (S_ot >= 0.50). Remote setpoint changes prohibited; local hands-on control required.", rule_traces

    # Rule 3: Device Posture & Compliance Gate
    r3_triggered = ("Unmanaged" in device and ("Level 1" in resource or "Level 2" in resource or "Level 3" in resource))
    rule_traces.append({
        "Rule": "R3: Unmanaged Device Posture Gate",
        "Condition": "Device = Unmanaged BYOD AND Target in [Level 1, 2, 3]",
        "Evaluated": f"Device = {device}, Resource = {resource.split(':')[0]}",
        "Result": "TRIGGERED (DENY)" if r3_triggered else "PASS"
    })
    if r3_triggered:
        return "DENY", "Device posture failed compliance checks. Unmanaged and non-compliant hardware is strictly forbidden from accessing OT or core IT infrastructure.", rule_traces

    # Rule 4: Air-Gap & Perimeter Boundary Rule
    r4_triggered = ("Level 1" in resource and network != "Internal OT Control Room (Air-Gapped VLAN)")
    rule_traces.append({
        "Rule": "R4: Level 1 Boundary Air-Gap Rule",
        "Condition": "Target = Level 1 PLC Control AND Location != Internal OT Control Room",
        "Evaluated": f"Location = {network}",
        "Result": "TRIGGERED (DENY)" if r4_triggered else "PASS"
    })
    if r4_triggered:
        return "DENY", "Physical actuator and PLC setpoint modifications are air-gapped and strictly restricted to the physical OT Control Room.", rule_traces

    # Rule 5: Least Privilege / Role Authorization
    r5_triggered = (
        ("Level 1" in resource and user not in ["Plant Operator", "Lead Automation Engineer"]) or
        ("Level 2" in resource and user in ["Corporate Employee"])
    )
    rule_traces.append({
        "Rule": "R5: Least-Privilege Authorization",
        "Condition": "User Role authorized for target resource domain",
        "Evaluated": f"User = {user}, Target = {resource.split(':')[0]}",
        "Result": "TRIGGERED (DENY)" if r5_triggered else "PASS"
    })
    if r5_triggered:
        return "DENY", f"User role '{user}' lacks requisite clearance for target resource '{resource}'. Access denied under least-privilege enforcement.", rule_traces

    # Rule 6: Remote VPN Step-Up Verification
    r6_triggered = (network == "Remote Access VPN / External" and "Level 2" in resource)
    rule_traces.append({
        "Rule": "R6: Remote SCADA Verification Gate",
        "Condition": "Location = Remote VPN AND Target = Level 2 SCADA/Historian",
        "Evaluated": f"Location = {network}, Target = Level 2",
        "Result": "TRIGGERED (VERIFY)" if r6_triggered else "PASS"
    })
    if r6_triggered:
        return "REQUIRE VERIFICATION", "Remote access to SCADA telemetry historian requires Step-Up Hardware Token MFA and Plant Operator out-of-band authorization.", rule_traces

    # Rule 7: Elevated Network Intrusion Suspicion Gate
    r7_triggered = (p_attack >= 0.25 and user != "Enterprise Security Analyst")
    rule_traces.append({
        "Rule": "R7: Threat-Elevated Verification",
        "Condition": "IT Intrusion Risk (p_attack) >= 0.25",
        "Evaluated": f"p_attack = {p_attack:.2f}",
        "Result": "TRIGGERED (VERIFY)" if r7_triggered else "PASS"
    })
    if r7_triggered:
        return "REQUIRE VERIFICATION", f"Network telemetry exhibits anomalous or attack-like activity (p_attack = {p_attack:.2f}). Session requires re-authentication and behavioral telemetry logging.", rule_traces

    # Rule 8: Third-Party Contractor Scrutiny
    r8_triggered = (user == "Third-Party Maintenance Contractor" and ("Level 1" in resource or "Level 2" in resource))
    rule_traces.append({
        "Rule": "R8: Contractor Dual-Authorization Gate",
        "Condition": "User = Third-Party Contractor AND Target in [Level 1, 2]",
        "Evaluated": f"User = Contractor, Resource = {resource.split(':')[0]}",
        "Result": "TRIGGERED (VERIFY)" if r8_triggered else "PASS"
    })
    if r8_triggered:
        return "REQUIRE VERIFICATION", "Third-party vendor access requires active sponsor confirmation and temporary cryptographic session key authorization.", rule_traces

    # Default Rule: Allow Least-Privilege Access
    rule_traces.append({
        "Rule": "R9: Baseline Compliance Validation",
        "Condition": "All posture, role, boundary, and risk evaluations satisfied",
        "Evaluated": "Low risk, authorized role, compliant device",
        "Result": "PASSED (ALLOW)"
    })
    return "ALLOW", "All contextual identity, device posture, location perimeter, and cyber-physical risk conditions satisfied. Least-privilege access granted.", rule_traces

def render_zero_trust_simulator():
    header_html = """
<div style="margin-bottom: 20px;">
    <h2 style="color: #ffffff; margin-bottom: 4px;">03 Zero-Trust Decision Simulator</h2>
    <p style="color: #94a3b8; font-size: 14px;">Interactive demonstration of the contextual Zero-Trust policy engine. Evaluates identity, device posture, resource criticality, and live cyber-physical risk to render transparent access verdicts.</p>
</div>
"""
    st.markdown(textwrap.dedent(header_html).strip(), unsafe_allow_html=True)

    st.info(
        "💡 **Policy Engine Simulator:** This module provides a transparent simulation of contextual Zero-Trust policy evaluation "
        "designed for critical infrastructure. It explicitly details the reasoning behind each access decision based on multi-channel risk telemetry."
    )

    # State initialization
    if "zt_user" not in st.session_state:
        st.session_state.zt_user = "Plant Operator"
    if "zt_device" not in st.session_state:
        st.session_state.zt_device = "Hardened Engineering Workstation"
    if "zt_resource" not in st.session_state:
        st.session_state.zt_resource = "Level 1: PLC Actuator Control & Setpoints (P1_FCV01D)"
    if "zt_network" not in st.session_state:
        st.session_state.zt_network = "Internal OT Control Room (Air-Gapped VLAN)"
    if "zt_p_att" not in st.session_state:
        st.session_state.zt_p_att = 0.02
    if "zt_s_ot" not in st.session_state:
        st.session_state.zt_s_ot = 0.05

    def apply_preset(user, device, resource, network, p_att, s_ot):
        st.session_state.zt_user = user
        st.session_state.zt_device = device
        st.session_state.zt_resource = resource
        st.session_state.zt_network = network
        st.session_state.zt_p_att = p_att
        st.session_state.zt_s_ot = s_ot

    # Preset Testing Scenarios
    st.markdown("#### Preset Test Scenarios")
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.button(
            "🟢 Normal Plant Operator",
            on_click=apply_preset,
            args=(
                "Plant Operator",
                "Hardened Engineering Workstation",
                "Level 1: PLC Actuator Control & Setpoints (P1_FCV01D)",
                "Internal OT Control Room (Air-Gapped VLAN)",
                0.02,
                0.05
            ),
            use_container_width=True
        )

    with p2:
        st.button(
            "🟡 Remote Vendor Maintenance",
            on_click=apply_preset,
            args=(
                "Third-Party Maintenance Contractor",
                "Corporate Managed Laptop",
                "Level 2: SCADA Telemetry Historian & HMI",
                "Remote Access VPN / External",
                0.08,
                0.05
            ),
            use_container_width=True
        )

    with p3:
        st.button(
            "🔴 Physical Anomaly (Scenario D)",
            on_click=apply_preset,
            args=(
                "Lead Automation Engineer",
                "Corporate Managed Laptop",
                "Level 1: PLC Actuator Control & Setpoints (P1_FCV01D)",
                "Corporate Enterprise LAN",
                0.05,
                0.85
            ),
            use_container_width=True
        )

    with p4:
        st.button(
            "🚫 Coordinated Attack (Scenario A)",
            on_click=apply_preset,
            args=(
                "Plant Operator",
                "Hardened Engineering Workstation",
                "Level 1: PLC Actuator Control & Setpoints (P1_FCV01D)",
                "Internal OT Control Room (Air-Gapped VLAN)",
                0.95,
                0.82
            ),
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Simulator Input Controls
    st.markdown("#### Configure Access Request Parameters")
    col_sub, col_res = st.columns(2)

    with col_sub:
        st.markdown("##### 👤 Subject & Device Context")
        st.selectbox(
            "User Identity & Role:",
            [
                "Plant Operator",
                "Lead Automation Engineer",
                "Enterprise Security Analyst",
                "Third-Party Maintenance Contractor",
                "Corporate Employee"
            ],
            key="zt_user"
        )
        st.selectbox(
            "Device Posture & Trust State:",
            [
                "Hardened Engineering Workstation",
                "Plant Floor HMI Terminal",
                "Corporate Managed Laptop",
                "Unmanaged BYOD / Unknown Device"
            ],
            key="zt_device"
        )

    with col_res:
        st.markdown("##### 🎯 Target Resource & Network Boundary")
        st.selectbox(
            "Target Resource & Criticality:",
            [
                "Level 1: PLC Actuator Control & Setpoints (P1_FCV01D)",
                "Level 2: SCADA Telemetry Historian & HMI",
                "Level 3: IT Enterprise Active Directory & Domain Controller",
                "Level 4: Internal Engineering Documentation & File Share"
            ],
            key="zt_resource"
        )
        st.selectbox(
            "Access Location & Boundary:",
            [
                "Internal OT Control Room (Air-Gapped VLAN)",
                "Corporate Enterprise LAN",
                "Remote Access VPN / External"
            ],
            key="zt_network"
        )

    # Live Cyber-Physical Threat Inputs
    st.markdown("##### ⚡ Observed Cyber-Physical Risk Telemetry")
    col_it_risk, col_ot_risk = st.columns(2)

    with col_it_risk:
        st.slider(
            "IT Network Attack Probability (p_attack):",
            min_value=0.0,
            max_value=1.0,
            key="zt_p_att",
            step=0.01
        )
    with col_ot_risk:
        st.slider(
            "OT Physical Process Deviation (S_ot):",
            min_value=0.0,
            max_value=1.0,
            key="zt_s_ot",
            step=0.01
        )

    p_att_val = float(st.session_state.zt_p_att)
    s_ot_val = float(st.session_state.zt_s_ot)

    # Calculate R_cp using the non-shortcut logistic formula
    logit = 6.5199 * p_att_val + 4.7116 * s_ot_val - 0.0324 * (p_att_val * s_ot_val) - 7.9211
    R_cp_val = 1.0 / (1.0 + np.exp(-logit))

    rcp_badge_html = f"""
<div style="background: #111a24; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 16px; margin-bottom: 20px;">
    <span style="color: #64748b; font-size: 12px; font-weight: 600; text-transform: uppercase;">Derived Cyber-Physical Risk Score (R_CP):</span>
    <span style="color: #c084fc; font-weight: 700; font-size: 16px; margin-left: 8px;">{R_cp_val:.4f}</span>
    <span style="color: #94a3b8; font-size: 12px; margin-left: 12px;">(Triage Threshold = 0.50)</span>
</div>
"""
    st.markdown(textwrap.dedent(rcp_badge_html).strip(), unsafe_allow_html=True)

    # Run Policy Evaluation
    verdict, reason, rule_traces = evaluate_zero_trust_policy(
        user=st.session_state.zt_user,
        device=st.session_state.zt_device,
        resource=st.session_state.zt_resource,
        network=st.session_state.zt_network,
        p_attack=p_att_val,
        S_ot=s_ot_val,
        R_cp=R_cp_val
    )

    # Display Verdict Card
    if verdict == "ALLOW":
        v_bg = "#0c281e"
        v_border = "#10b981"
        v_color = "#10b981"
        v_icon = "🟢"
    elif verdict == "REQUIRE VERIFICATION":
        v_bg = "#29210c"
        v_border = "#f59e0b"
        v_color = "#f59e0b"
        v_icon = "🟡"
    else:
        v_bg = "#2b1016"
        v_border = "#ef4444"
        v_color = "#ef4444"
        v_icon = "🔴"

    verdict_card_html = f"""
<div style="background: {v_bg}; border: 2px solid {v_border}; border-radius: 12px; padding: 22px 28px; margin: 10px 0 24px 0;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 32px;">{v_icon}</span>
        <div>
            <div style="color: #94a3b8; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Policy Decision Verdict</div>
            <div style="color: {v_color}; font-size: 26px; font-weight: 800; letter-spacing: 0.5px;">{verdict}</div>
        </div>
    </div>
    <div style="margin-top: 14px; color: #f1f5f9; font-size: 14px; line-height: 1.6; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 12px;">
        <b>Decision Rationale:</b> {reason}
    </div>
</div>
"""
    st.markdown(textwrap.dedent(verdict_card_html).strip(), unsafe_allow_html=True)

    # Policy Evaluation Trace Table
    st.markdown("#### 🔍 Step-by-Step Policy Evaluation Audit Trace")
    st.caption("Demonstrates transparent rule processing without opaque shortcuts:")

    df_trace = pd.DataFrame(rule_traces)
    st.dataframe(df_trace, use_container_width=True, hide_index=True)
