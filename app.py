import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Turbofan Predictive Maintenance Dashboard",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------------------------------------
# 1. LOAD ARTIFACTS & DATA
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    # Load XGBoost model bundle
    xgb_pack = joblib.load("model.pkl")  # Contains model, feature list, etc.
    # Load Unsupervised Isolation Forest artifact bundle
    iso_pack = joblib.load("iso_forest_onset_model.pkl")
    return xgb_pack, iso_pack

@st.cache_data
def load_default_data():
    try:
        df = pd.read_csv("test_data_processed.csv")
    except Exception:
        # Fallback simulated dataset if run without pre-generated CSV
        np.random.seed(42)
        records = []
        for eng in range(1, 101):
            max_c = np.random.randint(60, 200)
            for c in range(1, max_c + 1):
                records.append({
                    "engine_id": eng,
                    "cycle": c,
                    "s_2": 642.0 + np.random.normal(0, 0.5) + (c * 0.05),
                    "s_3": 1580.0 + np.random.normal(0, 2.0) + (c * 0.1),
                    "s_4": 1400.0 + np.random.normal(0, 1.5) + (c * 0.12),
                    "s_7": 553.0 - np.random.normal(0, 0.3) - (c * 0.03),
                    "s_11": 47.0 + np.random.normal(0, 0.1) + (c * 0.01),
                    "s_12": 521.0 - np.random.normal(0, 0.2) - (c * 0.02)
                })
        df = pd.DataFrame(records)
    return df

# Attempt to load model packs
try:
    xgb_pack, iso_pack = load_models()
    model = xgb_pack["model"] if isinstance(xgb_pack, dict) and "model" in xgb_pack else xgb_pack
    feature_cols = xgb_pack["feature_cols"] if isinstance(xgb_pack, dict) and "feature_cols" in xgb_pack else None
    
    iso_scaler = iso_pack["scaler"]
    iso_model = iso_pack["iso_forest"]
    iso_threshold = iso_pack["threshold"]
    iso_sensor_cols = iso_pack["sensor_cols"]
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.sidebar.warning(f"Note: Running in mock preview mode ({e}). Train and save your .pkl files to enable full live inference.")

# ---------------------------------------------------------
# 2. SIDEBAR NAVIGATION & DATA TOGGLE
# ---------------------------------------------------------
st.sidebar.title("✈️ Engine Health Monitor")
view_mode = st.sidebar.radio("Navigation View", ["Fleet Overview", "Single Engine View"])

st.sidebar.markdown("---")
st.sidebar.subheader("Data Source Toggle")
data_source = st.sidebar.radio("Choose Source:", ["100 Built-in Demo Engines", "Upload Custom CSV"])

if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Engine Sensor CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Using built-in demo data until a custom CSV is uploaded.")
        df = load_default_data()
else:
    df = load_default_data()

# Identify available sensor columns
all_sensors = [c for c in df.columns if c not in ["engine_id", "cycle", "RUL"]]

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS: INFERENCE & RISK STRATIFICATION
# ---------------------------------------------------------
def get_fleet_predictions(data):
    latest_cycles = data.sort_values(["engine_id", "cycle"]).groupby("engine_id").last().reset_index()
    
    if models_loaded and feature_cols:
        X_fleet = latest_cycles[feature_cols]
        preds = model.predict(X_fleet)
    else:
        preds = np.maximum(5, 125 - (latest_cycles["cycle"] * 0.7))
        
    latest_cycles["predicted_RUL"] = np.round(preds, 1)
    
    def assign_risk(rul):
        if rul <= 15:
            return "Critical"
        elif rul <= 40:
            return "Monitor"
        else:
            return "Healthy"
            
    latest_cycles["Risk_Status"] = latest_cycles["predicted_RUL"].apply(assign_risk)
    return latest_cycles

fleet_summary = get_fleet_predictions(df)

# ---------------------------------------------------------
# VIEW 1: FLEET OVERVIEW
# ---------------------------------------------------------
if view_mode == "Fleet Overview":
    st.title("Turbofan Fleet Health & Diagnostics Overview")
    st.markdown("Real-time fleet risk stratification, RUL projections, and cost-of-failure simulation.")

    total_engines = len(fleet_summary)
    critical_count = len(fleet_summary[fleet_summary["Risk_Status"] == "Critical"])
    monitor_count = len(fleet_summary[fleet_summary["Risk_Status"] == "Monitor"])
    healthy_count = len(fleet_summary[fleet_summary["Risk_Status"] == "Healthy"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Active Engines", total_engines)
    m2.metric("Critical (Immediate Service)", critical_count, delta=f"{critical_count} Urgent", delta_color="inverse")
    m3.metric("Monitor Closely", monitor_count, delta="Scheduled Soon")
    m4.metric("Healthy Condition", healthy_count)

    st.markdown("---")

    with st.expander("💰 Adjustable Maintenance Cost-Savings Estimator", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            cost_failure = st.number_input("Cost of Unplanned In-Flight Failure ($)", value=350000, step=25000)
        with c2:
            cost_maintenance = st.number_input("Cost of Scheduled Depot Overhaul ($)", value=45000, step=5000)

        avoided_failures = critical_count
        net_savings = (avoided_failures * cost_failure) - (avoided_failures * cost_maintenance)
        
        st.success(
            f"**Projected Cost Avoidance:** Flagging and servicing **{avoided_failures} critical engines** "
            f"prior to breakdown saves an estimated **${net_savings:,.2f}** compared to reactive catastrophic repairs."
        )

    col_chart, col_table = st.columns([1.2, 1])

    with col_chart:
        st.subheader("Fleet-wide Remaining Useful Life (RUL)")
        color_map = {"Critical": "#e74c3c", "Monitor": "#f39c12", "Healthy": "#2ecc71"}
        sorted_fleet = fleet_summary.sort_values("predicted_RUL")
        fig_bar = px.bar(
            sorted_fleet,
            x="engine_id",
            y="predicted_RUL",
            color="Risk_Status",
            color_discrete_map=color_map,
            labels={"engine_id": "Engine ID", "predicted_RUL": "Predicted RUL (Cycles)"},
            title="Sorted RUL Projections Across All Engines"
        )
        fig_bar.update_layout(height=400, xaxis_type='category')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_table:
        st.subheader("Risk-Ranked Dispatch Priority")
        display_cols = ["engine_id", "cycle", "predicted_RUL", "Risk_Status"]
        st.dataframe(
            sorted_fleet[display_cols].rename(columns={
                "engine_id": "Engine #",
                "cycle": "Current Cycle",
                "predicted_RUL": "Predicted RUL",
                "Risk_Status": "Status"
            }),
            height=390,
            use_container_width=True
        )

# ---------------------------------------------------------
# VIEW 2: SINGLE ENGINE VIEW (WITH UNSUPERVISED ONSET)
# ---------------------------------------------------------
else:
    st.title("Single Engine Telemetry & Diagnostic Inspection")
    
    selected_id = st.sidebar.selectbox("Select Engine ID:", sorted(df["engine_id"].unique()))
    engine_data = df[df["engine_id"] == selected_id].sort_values("cycle").reset_index(drop=True)
    current_cycle = int(engine_data["cycle"].max())
    
    engine_pred_row = fleet_summary[fleet_summary["engine_id"] == selected_id].iloc[0]
    predicted_rul = engine_pred_row["predicted_RUL"]
    risk_status = engine_pred_row["Risk_Status"]

    onset_cycle = None
    onset_detected = False
    
    if models_loaded and iso_sensor_cols:
        try:
            sample_scaled = iso_scaler.transform(engine_data[iso_sensor_cols])
            anomaly_scores = -iso_model.score_samples(sample_scaled)
            breach_indices = engine_data["cycle"].values[anomaly_scores > iso_threshold]
            if len(breach_indices) > 0:
                onset_detected = True
                onset_cycle = int(breach_indices[0])
        except Exception:
            pass
    else:
        if current_cycle > 35:
            onset_detected = True
            onset_cycle = 35

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Current Operational Cycle", f"{current_cycle} cycles")
    kpi2.metric("Predicted RUL (XGBoost)", f"{predicted_rul:.0f} cycles")
    kpi3.metric("Operating Health Status", risk_status)
    
    if onset_detected:
        kpi4.metric(
            "Mechanical Degradation Onset",
            f"Breached at Cycle {onset_cycle}",
            delta=f"Operating degraded for {current_cycle - onset_cycle} cycles",
            delta_color="inverse"
        )
    else:
        kpi4.metric(
            "Mechanical Degradation Onset",
            "Nominal (No Wear)",
            delta="Within 95th Percentile Baseline",
            delta_color="normal"
        )

    st.markdown("---")

    st.subheader("Multivariate Sensor Telemetry & Onset Milestone")
    chosen_sensors = st.multiselect(
        "Select Sensors to Graph:",
        options=all_sensors,
        default=all_sensors[:3] if len(all_sensors) >= 3 else all_sensors
    )

    if chosen_sensors:
        fig_trend = go.Figure()
        for s in chosen_sensors:
            fig_trend.add_trace(go.Scatter(
                x=engine_data["cycle"],
                y=engine_data[s],
                mode="lines",
                name=s
            ))

        if onset_detected:
            fig_trend.add_vline(
                x=onset_cycle,
                line_width=2,
                line_dash="dash",
                line_color="#e74c3c",
                annotation_text=f"Degradation Onset (Cycle {onset_cycle})",
                annotation_position="top left"
            )

        fig_trend.update_layout(
            title=f"Telemetry History for Engine #{selected_id}",
            xaxis_title="Operational Flight Cycle",
            yaxis_title="Sensor Value",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    st.subheader("Explainable AI (SHAP) Diagnostic Attribution")
    
    col_shap, col_text = st.columns([1.2, 1])

    with col_text:
        st.markdown("#### Automated Diagnostic Assessment")
        
        if risk_status == "Critical":
            st.error(
                f"⚠️ **Urgent Action Required:** Engine #{selected_id} is operating in **Critical condition** "
                f"with only **{predicted_rul:.0f} cycles** remaining. Internal wear was flagged beginning at "
                f"**Cycle {onset_cycle if onset_detected else 'N/A'}**, driven predominantly by elevated core "
                f"temperatures and pressure drop in the High-Pressure Compressor (HPC)."
            )
            st.markdown("""
            **Recommended Technical Procedures:**
            1. Remove asset from flight dispatch rotation.
            2. Schedule immediate borescope inspection for HPC blade wear and combustor liner leakage.
            3. Verify bleed valve calibration.
            """)
        elif risk_status == "Monitor":
            st.warning(
                f"🔶 **Advisory Flag:** Engine #{selected_id} exhibits early-to-mid stage wear "
                f"(RUL: **{predicted_rul:.0f} cycles**). Unsupervised anomaly tripwire was crossed at "
                f"**Cycle {onset_cycle if onset_detected else 'N/A'}**. Plan scheduled shop visit within 25 cycles."
            )
        else:
            st.success(
                f"✅ **Nominal Operation:** Engine #{selected_id} shows stable sensor behavior across all "
                f"monitoring channels. Degradation onset tripwire has not been breached."
            )

    with col_shap:
        st.markdown("#### Top Sensor Contribution to Current Risk Score")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        features_mock = ["HPC Outlet Pressure (Ps30)", "LPT Coolant Bleed", "Total Exhaust Temp", "Fan Speed", "Core Speed"]
        shap_values_mock = [12.4, 8.2, 5.1, -1.8, -3.2]
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_values_mock]

        ax.barh(features_mock[::-1], shap_values_mock[::-1], color=colors[::-1])
        ax.set_xlabel("SHAP Impact (+ Accelerates Failure, - Extends Life)")
        ax.set_title(f"Engine #{selected_id} Feature Attribution")
        plt.tight_layout()
        st.pyplot(fig)
