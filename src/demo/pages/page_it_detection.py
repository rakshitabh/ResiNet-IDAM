"""
ResiNet-IDAM: 02 IT Detection Page
---------------------------------
Network flow classification, interactive inference on realistic test flows,
and comparison between Configuration A (79 features) and Configuration B (77 features without ports).
"""

from pathlib import Path
import json
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed" / "it_realistic"
MODEL_DIR = ROOT / "models" / "it_realistic"

@st.cache_data
def load_it_data(config_choice):
    meta = json.load(open(DATA_DIR / "metadata.json"))
    class_names = meta["class_names"]
    
    if config_choice == "Configuration A (All 79 Features)":
        test_npz = np.load(DATA_DIR / "it_test_all.npz")
        model = joblib.load(MODEL_DIR / "random_forest_all.joblib")
        features = meta["configurations"]["all_features"]["feature_names"]
    else:
        test_npz = np.load(DATA_DIR / "it_test_no_ports.npz")
        model = joblib.load(MODEL_DIR / "random_forest_no_ports.joblib")
        features = meta["configurations"]["no_ports"]["feature_names"]

    return test_npz["X"], test_npz["y"], class_names, features, model

def render_it_detection():
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; margin-bottom: 4px;">02 IT Detection: Network Flow Classification</h2>
            <p style="color: #94a3b8; font-size: 14px;">Evaluates Random Forest multi-class intrusion detection on the contiguous out-of-burst test partition.</p>
        </div>
    """, unsafe_allow_html=True)

    # Configuration Selector
    col_cfg, col_stat = st.columns([1.5, 1])
    with col_cfg:
        config_choice = st.radio(
            "Select Feature Configuration:",
            ["Configuration A (All 79 Features)", "Configuration B (77 Features - No Ports)"],
            horizontal=True
        )

    with col_stat:
        if "79" in config_choice:
            st.success("🟢 **Configuration A:** All 79 numeric NetFlow features included (Accuracy: 99.58% | FPR: 1.30%)")
        else:
            st.warning("🟠 **Configuration B:** Ports removed to test behavioral traffic dynamics (Accuracy: 98.66% | FPR: 3.67%)")

    X_test, y_test, class_names, feature_names, model = load_it_data(config_choice)

    tab_inspect, tab_matrix = st.tabs(["🔍 Interactive Flow Inspector & Prediction", "📊 Confusion Matrix & Benchmark Breakdown"])

    with tab_inspect:
        st.markdown("#### Test Flow Selector")
        c_filter, c_idx, c_btn = st.columns([1.5, 1.5, 1])

        with c_filter:
            selected_class = st.selectbox("Filter flows by True Class:", ["All Classes"] + class_names)

        # Get matching indices
        if selected_class == "All Classes":
            valid_indices = np.arange(len(y_test))
        else:
            class_idx = class_names.index(selected_class)
            valid_indices = np.where(y_test == class_idx)[0]

        with c_idx:
            selected_test_idx = st.selectbox(
                f"Select Flow Index ({len(valid_indices)} available):",
                valid_indices[:200],  # show first 200 for clean UI
                index=0
            )

        with c_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🎲 Random Sample", use_container_width=True):
                selected_test_idx = int(np.random.choice(valid_indices))

        # Perform Inference
        flow_vector = X_test[selected_test_idx].reshape(1, -1)
        true_label_idx = int(y_test[selected_test_idx])
        true_label_name = class_names[true_label_idx]

        probs = model.predict_proba(flow_vector)[0]
        pred_label_idx = int(np.argmax(probs))
        pred_label_name = class_names[pred_label_idx]
        confidence = probs[pred_label_idx] * 100.0

        is_correct = (true_label_idx == pred_label_idx)

        # Display Prediction Result Card
        st.markdown("<br>", unsafe_allow_html=True)
        col_res1, col_res2, col_res3 = st.columns([1, 1, 1.2])

        with col_res1:
            st.markdown(f"""
                <div style="background: #111a24; border: 1px solid #1e293b; border-radius: 8px; padding: 16px;">
                    <div style="color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 600;">Ground Truth Class</div>
                    <div style="color: #ffffff; font-size: 20px; font-weight: 700; margin-top: 4px;">{true_label_name}</div>
                    <div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">Flow #{selected_test_idx:,} in contiguous test partition</div>
                </div>
            """, unsafe_allow_html=True)

        with col_res2:
            status_color = "#10b981" if is_correct else "#ef4444"
            status_text = "CORRECT DETECTION" if is_correct else "MISCLASSIFIED"
            st.markdown(f"""
                <div style="background: #111a24; border: 1px solid #1e293b; border-radius: 8px; padding: 16px;">
                    <div style="color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 600;">Model Prediction</div>
                    <div style="color: {status_color}; font-size: 20px; font-weight: 700; margin-top: 4px;">{pred_label_name}</div>
                    <div style="color: {status_color}; font-size: 12px; margin-top: 4px; font-weight: 600;">{status_text} ({confidence:.2f}% confidence)</div>
                </div>
            """, unsafe_allow_html=True)

        with col_res3:
            # Probability Bar Chart
            df_probs = pd.DataFrame({
                "Class": class_names,
                "Probability": probs
            })
            fig_p = px.bar(
                df_probs,
                x="Probability",
                y="Class",
                orientation="h",
                color="Probability",
                color_continuous_scale="Teal",
                range_x=[0, 1]
            )
            fig_p.update_layout(
                height=160,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=10),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_p, use_container_width=True)

        # Inspect Flow Feature Attributes
        st.markdown("#### 🔬 Network Flow Feature Inspector")
        st.caption("Standardized NetFlow statistical values for this specific transmission:")

        # Show table of top features
        df_feat = pd.DataFrame({
            "Feature Name": feature_names,
            "Standardized Value (z-score)": flow_vector[0]
        })
        
        # Display top 12 non-zero / highest magnitude features
        df_feat["Magnitude"] = df_feat["Standardized Value (z-score)"].abs()
        df_top = df_feat.sort_values(by="Magnitude", ascending=False).drop(columns=["Magnitude"]).head(14)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.dataframe(df_top.iloc[:7], use_container_width=True, hide_index=True)
        with col_t2:
            st.dataframe(df_top.iloc[7:], use_container_width=True, hide_index=True)

    with tab_matrix:
        st.markdown("#### 7x7 Multi-Class Confusion Matrix (Contiguous Test Partition: 18,344 Flows)")

        # Compute full confusion matrix
        y_preds = model.predict(X_test)
        cm = confusion_matrix(y_test, y_preds, labels=list(range(len(class_names))))

        # Plotly Heatmap
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Flow Count"),
            x=class_names,
            y=class_names,
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Per-class summary table
        report_dict = classification_report(y_test, y_preds, target_names=class_names, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()
        df_report = df_report.loc[class_names + ["macro avg", "weighted avg"]][["precision", "recall", "f1-score", "support"]]
        df_report["precision"] = df_report["precision"].map("{:.2%}".format)
        df_report["recall"] = df_report["recall"].map("{:.2%}".format)
        df_report["f1-score"] = df_report["f1-score"].map("{:.2%}".format)
        df_report["support"] = df_report["support"].astype(int)

        st.markdown("#### Per-Class Precision, Recall, and F1-Score")
        st.dataframe(df_report, use_container_width=True)

        if "77" in config_choice:
            st.info(
                "**Diagnostic Observation on Configuration B:** "
                "Notice that removing `Source Port` and `Destination Port` causes **38 Normal flows to be classified as Botnet ARES** "
                "and **141 Normal flows to be classified as Dos/DDos**. This directly stems from the **19 ambiguous feature collisions** "
                "where empty TCP reset packets in benign traffic match attack flows when port numbers are unavailable."
            )
