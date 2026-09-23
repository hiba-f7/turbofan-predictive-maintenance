"""
Turbofan Engine Predictive Maintenance Dashboard
-------------------------------------------------
Run locally with:  streamlit run app.py
Deploy for free on Streamlit Community Cloud by connecting this repo.

Requires these files (produced by predictive_maintenance.ipynb) to sit in
the same folder as this script:
    - model.pkl              (Step 13)
    - feature_cols.json      (Step 13)
    - sensor_cols.json       (Step 13)
    - dashboard_data.csv     (Step 13)
    - safety_margin.json     (Step 17, optional — falls back to a default
                               margin if missing, with a warning shown)
"""

import json
import re

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from scipy.stats import linregress

st.set_page_config(page_title="Turbofan RUL Predictor & Copilot", layout="wide")

# ---------- Sensor -> physical component reference ----------
# Sensor descriptions are published in Saxena & Goebel (2008), "Damage
# Propagation Modeling for Aircraft Engine Run-to-Failure Simulation" —
# the original C-MAPSS paper. FD001 has a single documented fault mode:
# High-Pressure Compressor (HPC) degradation, which is why several of the
# entries below point toward the HPC / core.

SENSOR_INFO = {
    "sensor_2": {
        "name": "T24 - LPC outlet temperature",
        "subsystem": "Low-Pressure Compressor (LPC)",
        "action": "Inspect LPC stages for fouling or erosion",
    },
    "sensor_3": {
        "name": "T30 - HPC outlet temperature",
        "subsystem": "High-Pressure Compressor (HPC)",
        "action": "Perform borescope inspection of HPC blades for erosion or fouling",
    },
    "sensor_4": {
        "name": "T50 - LPT outlet temperature",
        "subsystem": "Low-Pressure Turbine (LPT)",
        "action": "Inspect turbine blades and nozzle guide vanes for thermal degradation",
    },
    "sensor_6": {
        "name": "P15 - Bypass-duct pressure",
        "subsystem": "Fan Bypass Duct",
        "action": "Check bypass duct seals and fan bypass flow path",
    },
    "sensor_7": {
        "name": "P30 - HPC outlet pressure",
        "subsystem": "High-Pressure Compressor (HPC)",
        "action": "Perform borescope inspection of HPC; check blade tip clearances",
    },
    "sensor_8": {
        "name": "Nf - Physical fan speed",
        "subsystem": "Fan Module",
        "action": "Inspect fan blades and verify fan speed sensor calibration",
    },
    "sensor_9": {
        "name": "Nc - Physical core speed",
        "subsystem": "Core (HPC/HPT)",
        "action": "Inspect core rotating assembly for wear",
    },
    "sensor_11": {
        "name": "Ps30 - HPC outlet static pressure",
        "subsystem": "High-Pressure Compressor (HPC)",
        "action": "Perform borescope inspection on HPC stages; check for efficiency loss",
    },
    "sensor_12": {
        "name": "phi - Fuel flow / Ps30 ratio",
        "subsystem": "Fuel System / Combustor",
        "action": "Inspect fuel nozzles and combustor liner",
    },
    "sensor_13": {
        "name": "NRf - Corrected fan speed",
        "subsystem": "Fan Module",
        "action": "Check fan speed sensor and fan module balance",
    },
    "sensor_14": {
        "name": "NRc - Corrected core speed",
        "subsystem": "Core (HPC/HPT)",
        "action": "Inspect core rotating assembly for wear",
    },
    "sensor_15": {
        "name": "BPR - Bypass ratio",
        "subsystem": "Fan Bypass Duct",
        "action": "Check fan bypass flow path and duct integrity",
    },
    "sensor_17": {
        "name": "htBleed - Bleed enthalpy",
        "subsystem": "Bleed Air System",
        "action": "Inspect bleed air valves and ducting for leaks",
    },
    "sensor_20": {
        "name": "W31 - HPT coolant bleed",
        "subsystem": "High-Pressure Turbine (HPT)",
        "action": "Inspect HPT cooling passages and blades for thermal damage",
    },
    "sensor_21": {
        "name": "W32 - LPT coolant bleed",
        "subsystem": "Low-Pressure Turbine (LPT)",
        "action": "Inspect LPT cooling passages and blades for thermal damage",
    },
}

# Simulated Fault Isolation Manual knowledge base for grounded retrieval
FAULT_ISOLATION_MANUAL = [
    {
        "code": "FIM-72-001",
        "subsystem": "High-Pressure Compressor (HPC)",
        "sensors": ["sensor_3", "sensor_7", "sensor_11", "sensor_9", "sensor_14"],
        "keywords": ["compressor", "ps30", "p30", "t30", "core", "pressure", "hpc"],
        "ata_chapter": "ATA 72-30",
        "probable_cause": "Rotor blade tip rub, severe aerodynamic seal wear, or foreign object damage (FOD).",
        "actions": [
            "Ground engine from extended-range flight operations if RUL < 25 cycles.",
            "Perform flexible borescope inspection on HPC stages 3 through 6 through inspection ports B1 and B2.",
            "Check lower gearbox magnetic chip detectors for metallic debris.",
        ],
        "tools": "6mm articulating video-borescope (Part No. P-6021), 3/8-inch calibrated torque wrench set.",
    },
    {
        "code": "FIM-72-002",
        "subsystem": "Low-Pressure Turbine (LPT) & Exhaust",
        "sensors": ["sensor_4", "sensor_21"],
        "keywords": ["turbine", "t50", "lpt", "w32", "exhaust", "thermal"],
        "ata_chapter": "ATA 72-50",
        "probable_cause": "Thermal barrier coating spallation, cooling vane passage blockage, or nozzle guide vane erosion.",
        "actions": [
            "Log hot-section thermal exposure margins.",
            "Execute borescope inspection of LPT nozzle guide vanes and blade leading edges.",
            "If thermal distress exceeds 15% surface area, schedule module replacement within 15 operating cycles.",
        ],
        "tools": "High-intensity UV borescope probe, thermal clearance feeler gauge set.",
    },
    {
        "code": "FIM-72-003",
        "subsystem": "High-Pressure Turbine (HPT)",
        "sensors": ["sensor_20"],
        "keywords": ["hpt", "w31", "coolant", "turbine", "bleed"],
        "ata_chapter": "ATA 72-40",
        "probable_cause": "HPT rotor seal leakage or coolant bleed restriction.",
        "actions": [
            "Verify high-pressure turbine cooling airflow paths.",
            "Inspect stage 1 HPT blade shroud segments for thermal cracking.",
        ],
        "tools": "Optical thermal imaging borescope, depth micrometer.",
    },
    {
        "code": "FIM-72-004",
        "subsystem": "Low-Pressure Compressor (LPC) & Fan Module",
        "sensors": ["sensor_2", "sensor_8", "sensor_13"],
        "keywords": ["fan", "lpc", "t24", "nf", "nrf", "speed"],
        "ata_chapter": "ATA 72-20",
        "probable_cause": "Fan blade leading edge erosion, acoustic liner disbond, or unbalance due to ingestion.",
        "actions": [
            "Conduct tactile run-out check and visual inspection of all 24 wide-chord titanium fan blades.",
            "Perform vibration spectrum survey during ground run-up at 70% and 85% N1.",
        ],
        "tools": "Dial indicator with magnetic mount, portable optical tachometer.",
    },
    {
        "code": "FIM-72-005",
        "subsystem": "Combustor & Fuel Delivery System",
        "sensors": ["sensor_12"],
        "keywords": ["fuel", "combustor", "phi", "ratio", "flow"],
        "ata_chapter": "ATA 73-10",
        "probable_cause": "Fuel nozzle coking, fuel flow divider drift, or combustor liner hot spot.",
        "actions": [
            "Inspect fuel manifold pressure differential.",
            "Perform borescope check of combustor liner swirler nozzles for carbon buildup.",
        ],
        "tools": "Fuel nozzle spray calibration test rig, borescope with wide-angle lens.",
    },
]

DEFAULT_SAFETY_MARGIN_FRACTION = 0.3  # fallback if safety_margin.json is missing
DEMO_SNAPSHOT_FRACTION = 0.7  # default view point

# ---------- Load model + data (cached so it only loads once) ----------


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_data.csv")
    with open("feature_cols.json") as f:
        feature_cols = json.load(f)
    with open("sensor_cols.json") as f:
        sensor_cols = json.load(f)
    return df, feature_cols, sensor_cols


@st.cache_data
def load_safety_margin():
    try:
        with open("safety_margin.json") as f:
            data = json.load(f)
        return data["margin_cycles"], data["confidence_level"], True
    except FileNotFoundError:
        return None, 0.95, False


def slope(x):
    if len(x) < 2:
        return 0.0
    return linregress(range(len(x)), x)[0]


def add_features(raw_df, sensor_cols, roll_window=5, slope_window=10):
    """Same feature engineering as the training notebook."""
    out = raw_df.copy()
    for col in sensor_cols:
        out[f"{col}_rollmean"] = out.groupby("engine_id")[col].transform(
            lambda x: x.rolling(roll_window, min_periods=1).mean()
        )
        out[f"{col}_rollstd"] = out.groupby("engine_id")[col].transform(
            lambda x: x.rolling(roll_window, min_periods=1).std().fillna(0)
        )
        out[f"{col}_slope"] = out.groupby("engine_id")[col].transform(
            lambda x: x.rolling(slope_window, min_periods=2).apply(slope, raw=True).fillna(0)
        )
    return out


def readable_sensor_name(feature_name):
    """Turn 'sensor_4_rollmean' into 'sensor 4 (recent average)', etc."""
    suffix_labels = {
        "_rollmean": " (recent average)",
        "_rollstd": " (recent fluctuation)",
        "_slope": " (recent trend)",
    }
    for suffix, label in suffix_labels.items():
        if feature_name.endswith(suffix):
            base = feature_name[: -len(suffix)].replace("_", " ")
            return f"{base}{label}"
    return feature_name.replace("_", " ")


def extract_sensor_id(feature_name):
    """Turn 'sensor_4_rollmean' into 'sensor_4' so it can be looked up in SENSOR_INFO."""
    for suffix in ("_rollmean", "_rollstd", "_slope"):
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]
    return feature_name


def build_plain_english_summary(top_features_df, pred_rul):
    """Turn the top SHAP contributors into a one-line, non-technical sentence."""
    worsening = top_features_df[top_features_df["shap_value"] < 0].head(2)
    improving = top_features_df[top_features_df["shap_value"] > 0].head(1)

    worsening_names = [readable_sensor_name(f) for f in worsening["feature"]]
    improving_names = [readable_sensor_name(f) for f in improving["feature"]]

    if pred_rul < 20:
        lead = "This engine is flagged as high-risk"
    elif pred_rul < 50:
        lead = "This engine is flagged for monitoring"
    else:
        lead = "This engine looks healthy"

    parts = []
    if worsening_names:
        parts.append(f"mainly due to {', and '.join(worsening_names)} trending in a concerning direction")
    if improving_names and pred_rul >= 50:
        parts.append(f"supported by a stable reading from {improving_names[0]}")

    if not parts:
        return f"{lead}. No single sensor stands out as the dominant driver of this prediction."

    return f"{lead}, {'; '.join(parts)}."


def get_component_diagnosis(top_features_df, max_items=3):
    """Map the top *worsening* SHAP contributors to physical subsystems and
    a plausible inspection action."""
    worsening = top_features_df[top_features_df["shap_value"] < 0].copy()
    if worsening.empty:
        return []
    worsening["sensor_id"] = worsening["feature"].apply(extract_sensor_id)
    worsening["abs_value"] = worsening["shap_value"].abs()

    diagnosis = []
    seen_subsystems = set()
    for _, row in worsening.sort_values("abs_value", ascending=False).iterrows():
        info = SENSOR_INFO.get(row["sensor_id"])
        if info is None or info["subsystem"] in seen_subsystems:
            continue
        seen_subsystems.add(info["subsystem"])
        diagnosis.append(
            {
                "subsystem": info["subsystem"],
                "sensor_id": row["sensor_id"],
                "sensor_name": info["name"],
                "action": info["action"],
                "shap_value": row["shap_value"],
            }
        )
        if len(diagnosis) >= max_items:
            break
    return diagnosis


def rag_retrieve_manual_entry(diagnosis_list):
    """Retrieves relevant fault isolation directive grounded in the manual."""
    if not diagnosis_list:
        return FAULT_ISOLATION_MANUAL[0]

    flagged_sensors = {d.get("sensor_id", "") for d in diagnosis_list}
    flagged_subsystems = {d.get("subsystem", "").lower() for d in diagnosis_list}

    # Best match based on sensor overlap
    for entry in FAULT_ISOLATION_MANUAL:
        if any(s in entry["sensors"] for s in flagged_sensors):
            return entry

    # Fallback to subsystem keyword match
    for entry in FAULT_ISOLATION_MANUAL:
        if any(kw in " ".join(flagged_subsystems) for kw in entry["keywords"]):
            return entry

    return FAULT_ISOLATION_MANUAL[0]


def synthesize_maintenance_work_order(engine_id, cycle, pred_rul, safe_floor, diagnosis_list, manual_entry):
    """Generates an actionable engineering work order from the retrieved manual."""
    priority = "HIGH PRIORITY / GROUND ASSET" if pred_rul < 20 else ("MEDIUM PRIORITY / MONITOR" if pred_rul < 50 else "ROUTINE")
    
    actions_formatted = "\n".join([f"  {idx+1}. {act}" for idx, act in enumerate(manual_entry["actions"])])
    
    diagnostic_drivers = []
    for d in diagnosis_list:
        diagnostic_drivers.append(f"{d['subsystem']} ({d['sensor_name']})")
    drivers_str = ", ".join(diagnostic_drivers) if diagnostic_drivers else "Nominal operating telemetry baseline"

    work_order_text = f"""================================================================
ENGINEERING WORK ORDER: AIRFRAME & POWERPLANT DISPATCH
Directive Reference : {manual_entry['ata_chapter']} | Code: {manual_entry['code']}
================================================================
Asset Identifier    : Engine Unit #{engine_id}
Operational Cycle   : {cycle}
Predicted RUL       : {pred_rul:.1f} Cycles (Conservative Safe Floor: {safe_floor:.1f} Cycles)
Dispatch Priority   : {priority}

TELEMETRY & SHAP DIAGNOSIS:
Dominant Driver(s)  : {drivers_str}
Suspected Mechanism : {manual_entry['probable_cause']}

RETRIEVED TECHNICAL DIRECTIVES ({manual_entry['ata_chapter']}):
{actions_formatted}

REQUIRED TOOLING & CLEARANCES:
  - Tools: {manual_entry['tools']}
  - Compliance Sign-off Required prior to next flight cycle dispatch.
================================================================"""
    return work_order_text


def generate_pdf_report(engine_id, cycle, pred_rul, safe_floor, status_text, diagnosis, margin_calibrated, work_order_text=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def line(text, size=11, style="", gap_after=0):
        pdf.set_font("Helvetica", style, size)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if gap_after:
            pdf.ln(gap_after)

    def wrapped(text, size=10, style=""):
        pdf.set_font("Helvetica", style, size)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    line(f"Engine {engine_id} - Maintenance & Copilot Summary", size=16, style="B")
    line("Generated by Turbofan RUL Predictor & AI Decision Copilot", size=10, gap_after=4)

    line(f"Current cycle: {cycle}")
    line(f"Predicted Remaining Useful Life: {pred_rul:.0f} cycles")
    margin_note = "" if margin_calibrated else " (default, uncalibrated margin)"
    line(f"Safe operating floor (95% confidence){margin_note}: {safe_floor:.0f} cycles")
    line(f"Status: {status_text}")

    pdf.ln(4)
    line("Suggested inspection points:", size=12, style="B")

    if diagnosis:
        for d in diagnosis:
            wrapped(f"- {d['subsystem']}: {d['action']} (indicator: {d['sensor_name']})")
    else:
        line("No dominant risk driver identified at this cycle.")

    if work_order_text:
        pdf.ln(4)
        line("AI Copilot Work Order Directive:", size=12, style="B")
        for wo_line in work_order_text.splitlines():
            wrapped(wo_line, size=8.5)

    pdf.ln(6)
    wrapped(
        "Educational project output, not a certified maintenance record. "
        "Generated from a model trained on the NASA C-MAPSS FD001 simulated "
        "turbofan degradation dataset.",
        size=8.5,
        style="I",
    )

    return bytes(pdf.output())


model = load_model()
df, feature_cols, sensor_cols = load_data()
safety_margin, margin_confidence, margin_calibrated = load_safety_margin()

explainer = shap.TreeExplainer(model)


def compute_safe_floor(pred_rul):
    if margin_calibrated:
        return max(pred_rul - safety_margin, 0)
    return max(pred_rul * (1 - DEFAULT_SAFETY_MARGIN_FRACTION), 0)


# ---------- Sidebar ----------

st.sidebar.title("Turbofan RUL Predictor")
st.sidebar.write(
    "Predicts Remaining Useful Life (RUL) of a jet engine from sensor "
    "readings with XGBoost and synthesizes maintenance work orders via an AI Copilot."
)

data_source = st.sidebar.radio(
    "Data source",
    ["Browse fleet (built-in demo engines)", "Upload your own engine data"],
)

uploaded_engine_df = None

if data_source == "Upload your own engine data":
    st.sidebar.caption(
        "CSV must have the same columns as the raw C-MAPSS files: "
        "`engine_id, cycle, op_setting1, op_setting2, op_setting3, "
        "sensor_1 ... sensor_21`. One or more engines, one row per cycle."
    )
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        try:
            raw_upload = pd.read_csv(uploaded_file)
            missing_cols = [c for c in sensor_cols if c not in raw_upload.columns]
            if missing_cols:
                st.sidebar.error(
                    f"Missing expected columns: {missing_cols}. "
                    "This model expects the same sensor schema it was trained on."
                )
            else:
                uploaded_engine_df = add_features(raw_upload, sensor_cols)
                st.sidebar.success(f"Loaded {uploaded_engine_df['engine_id'].nunique()} engine(s).")
        except Exception as e:
            st.sidebar.error(f"Couldn't read that file: {e}")

active_df = uploaded_engine_df if uploaded_engine_df is not None else df

view_mode = st.sidebar.radio("View", ["Fleet overview", "Single engine"])

engine_ids = sorted(active_df["engine_id"].unique())

if view_mode == "Single engine":
    selected_engine = st.sidebar.selectbox("Select engine", engine_ids)

    cycle_options = sorted(active_df[active_df["engine_id"] == selected_engine]["cycle"].unique())
    default_idx = min(int(len(cycle_options) * DEMO_SNAPSHOT_FRACTION), len(cycle_options) - 1)
    default_cycle = int(cycle_options[default_idx])

    selected_cycle = st.sidebar.slider(
        "Cycle (point in the engine's life)",
        min_value=int(min(cycle_options)),
        max_value=int(max(cycle_options)),
        value=default_cycle,
        key=f"cycle_slider_{selected_engine}",
    )
    st.sidebar.caption(
        f"Defaults to {int(DEMO_SNAPSHOT_FRACTION*100)}% through this engine's "
        "recorded life. Drag to the far right to see it at its actual "
        "failure point (these are training engines run to failure)."
    )

    st.sidebar.divider()
    st.sidebar.subheader("Schedule compatibility")
    cycles_needed = st.sidebar.number_input(
        "Cycles until next scheduled rotation", min_value=0, value=3, step=1
    )
else:
    st.sidebar.divider()
    st.sidebar.subheader("Cost assumptions")
    st.sidebar.caption(
        "Illustrative figures — adjust to match a real fleet's actual costs."
    )
    cost_unplanned = st.sidebar.number_input(
        "Cost of an unplanned failure ($)", min_value=0, value=50000, step=5000
    )
    cost_scheduled = st.sidebar.number_input(
        "Cost of scheduled maintenance ($)", min_value=0, value=8000, step=1000
    )

    st.sidebar.divider()
    st.sidebar.subheader("Spare parts logistics")
    st.sidebar.caption(
        "Assumed lead time to receive a replacement part/module."
    )
    lead_time_cycles = st.sidebar.number_input(
        "Assumed parts lead time (cycles)", min_value=0, value=15, step=1
    )

# ---------- Main panel ----------

st.title("Engine Health Monitor & AI Decision Copilot")

if uploaded_engine_df is not None:
    st.info("Showing predictions on your uploaded data.")

if not margin_calibrated:
    st.sidebar.warning(
        "safety_margin.json not found — using a default 30% conservative "
        "reduction instead of a calibrated one. Run Step 17 in the notebook "
        "for a real, data-derived safety margin."
    )

if view_mode == "Fleet overview":

    @st.cache_data
    def compute_fleet_predictions(_df, _feature_cols, fraction=DEMO_SNAPSHOT_FRACTION):
        d = _df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
        rank = d.groupby("engine_id").cumcount()
        size = d.groupby("engine_id")["cycle"].transform("size")
        target_idx = (size * fraction).astype(int).clip(upper=size - 1)
        latest = d[rank == target_idx].reset_index(drop=True)

        preds = model.predict(latest[_feature_cols])
        latest["predicted_RUL"] = preds
        latest["safe_floor"] = latest["predicted_RUL"].apply(compute_safe_floor)

        def risk_label(rul):
            if rul < 20:
                return "🔴 Maintenance needed soon"
            elif rul < 50:
                return "🟡 Monitor closely"
            return "🟢 Healthy"

        latest["risk"] = latest["predicted_RUL"].apply(risk_label)
        return latest[["engine_id", "cycle", "predicted_RUL", "safe_floor", "risk"]].sort_values("predicted_RUL")

    fleet_summary = compute_fleet_predictions(active_df, feature_cols)

    st.caption(
        f"Demo note: since this dataset's engines were run to failure for "
        f"training, each engine's fleet-overview snapshot below is taken at "
        f"{int(DEMO_SNAPSHOT_FRACTION*100)}% of its recorded life rather "
        f"than its literal last cycle — otherwise every engine would show as "
        f"critical by construction. Uploaded fleet data is treated the same way."
    )

    n_engines = len(fleet_summary)
    n_critical = (fleet_summary["risk"] == "🔴 Maintenance needed soon").sum()
    n_watch = (fleet_summary["risk"] == "🟡 Monitor closely").sum()
    n_healthy = (fleet_summary["risk"] == "🟢 Healthy").sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total engines", n_engines)
    m2.metric("🔴 Needs maintenance", n_critical)
    m3.metric("🟡 Monitor closely", n_watch)
    m4.metric("🟢 Healthy", n_healthy)

    st.subheader("Estimated cost impact")
    savings_per_engine = max(cost_unplanned - cost_scheduled, 0)
    estimated_savings = n_critical * savings_per_engine

    c1, c2, c3 = st.columns(3)
    c1.metric("Engines flagged for proactive maintenance", n_critical)
    c2.metric("Est. savings per avoided failure", f"${savings_per_engine:,.0f}")
    c3.metric("Est. total savings this cycle", f"${estimated_savings:,.0f}")

    st.caption(
        "Illustrative only: assumes each engine flagged 🔴 would otherwise "
        "run to an unplanned failure, and that scheduling maintenance now "
        "instead costs the 'scheduled maintenance' amount rather than the "
        "'unplanned failure' amount. Adjust the cost assumptions in the "
        "sidebar to reflect a real fleet's numbers."
    )

    st.subheader("Spare parts / logistics alert")
    needs_order = fleet_summary[fleet_summary["safe_floor"] < lead_time_cycles]
    st.metric("Engines needing parts ordered now", len(needs_order))
    if len(needs_order) > 0:
        st.dataframe(
            needs_order[["engine_id", "safe_floor"]].rename(
                columns={"engine_id": "Engine", "safe_floor": "Safe floor (cycles)"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Fleet sorted by predicted risk (lowest RUL first)")
    st.dataframe(
        fleet_summary.rename(
            columns={
                "engine_id": "Engine",
                "cycle": "Current cycle",
                "predicted_RUL": "Predicted RUL",
                "safe_floor": "Safe floor (95%)",
                "risk": "Status",
            }
        ).style.format({"Predicted RUL": "{:.0f}", "Safe floor (95%)": "{:.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Predicted RUL across the fleet")
    chart_data = fleet_summary.set_index("engine_id")["predicted_RUL"]
    st.bar_chart(chart_data)

else:
    engine_df = active_df[active_df["engine_id"] == selected_engine].sort_values("cycle")
    current_row = engine_df[engine_df["cycle"] == selected_cycle]
    is_last_cycle = selected_cycle == engine_df["cycle"].max()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Sensor trends — Engine {selected_engine}")
        sensor_display_cols = [c for c in engine_df.columns if c.startswith("sensor_") and "_" not in c[7:]]
        chosen_sensors = st.multiselect(
            "Sensors to plot",
            options=sensor_display_cols,
            default=sensor_display_cols[:3],
        )
        if chosen_sensors:
            st.line_chart(engine_df.set_index("cycle")[chosen_sensors])

    pred_rul = None
    safe_floor = None
    status_text = ""
    diagnosis = []

    with col2:
        st.subheader("Prediction")

        if is_last_cycle:
            st.warning("This is the engine's last recorded cycle — it reached its failure point here in the data.")

        if not current_row.empty:
            X_row = current_row[feature_cols]
            pred_rul = model.predict(X_row)[0]
            safe_floor = compute_safe_floor(pred_rul)

            st.metric("Predicted RUL (cycles)", f"{pred_rul:.0f}")
            st.metric(
                f"Safe operating floor ({margin_confidence*100:.0f}% confidence)",
                f"{safe_floor:.0f} cycles",
            )

            if pred_rul < 20:
                status_text = "Maintenance recommended soon"
                st.error(status_text)
            elif pred_rul < 50:
                status_text = "Monitor closely"
                st.warning(status_text)
            else:
                status_text = "Engine healthy"
                st.success(status_text)

            if "RUL" in current_row.columns:
                st.caption(f"Actual RUL at this cycle (training label): {current_row['RUL'].values[0]:.0f}")

    st.divider()

    # ---------- Schedule compatibility ----------

    if pred_rul is not None:
        st.subheader("Schedule compatibility")
        if safe_floor >= cycles_needed:
            st.success(
                f"✅ This engine can safely complete {cycles_needed} more cycle(s) "
                f"before its conservative safety floor is reached "
                f"({safe_floor:.0f} cycles remaining at {margin_confidence*100:.0f}% confidence)."
            )
        else:
            st.error(
                f"⚠️ This engine may NOT safely complete {cycles_needed} more cycle(s) — "
                f"its conservative safety floor is only {safe_floor:.0f} cycles."
            )

    st.divider()

    # ---------- SHAP explanation for this specific prediction ----------

    st.subheader("Why this prediction? (SHAP feature contributions)")

    top_features = None
    if not current_row.empty:
        X_row = current_row[feature_cols]
        shap_values_row = explainer.shap_values(X_row)

        shap_df = pd.DataFrame(
            {"feature": feature_cols, "shap_value": shap_values_row[0]}
        )
        shap_df["abs_value"] = shap_df["shap_value"].abs()
        top_features = shap_df.sort_values("abs_value", ascending=False).head(10)

        st.info(build_plain_english_summary(top_features, pred_rul))

        st.bar_chart(top_features.set_index("feature")["shap_value"])
        st.caption(
            "Positive values push the predicted RUL up (healthier); negative "
            "values push it down (closer to failure)."
        )

    st.divider()

    # ---------- Component diagnosis & AI Maintenance Copilot ----------

    st.subheader("Component Diagnosis & AI Maintenance Copilot")
    st.caption(
        "Integrates SHAP feature attributions with a Retrieval-Augmented Generation (RAG) "
        "decision-support layer to synthesize grounded maintenance work orders from technical directives."
    )

    work_order_text = None
    if top_features is not None:
        diagnosis = get_component_diagnosis(top_features)
        
        diag_col, copilot_col = st.columns([1, 1])
        
        with diag_col:
            st.markdown("#### Physical Subsystem Diagnosis")
            if diagnosis:
                for d in diagnosis:
                    st.warning(f"**{d['subsystem']}** — {d['action']}  \n*Indicator: {d['sensor_name']}*")
            else:
                st.success("No single sensor is driving this prediction toward risk strongly enough to flag a component.")

        with copilot_col:
            st.markdown("#### RAG Work Order Synthesis")
            manual_entry = rag_retrieve_manual_entry(diagnosis)
            
            st.caption(f"Matched Manual Directive: **{manual_entry['ata_chapter']} ({manual_entry['code']})**")
            if st.button("🛠️ Synthesize Maintenance Work Order", key="gen_work_order"):
                work_order_text = synthesize_maintenance_work_order(
                    selected_engine, selected_cycle, pred_rul, safe_floor, diagnosis, manual_entry
                )
                st.session_state[f"wo_{selected_engine}_{selected_cycle}"] = work_order_text

            saved_wo = st.session_state.get(f"wo_{selected_engine}_{selected_cycle}")
            if saved_wo:
                st.code(saved_wo, language="text")
                work_order_text = saved_wo

    st.divider()

    # ---------- PDF export ----------

    if pred_rul is not None:
        try:
            current_wo = st.session_state.get(f"wo_{selected_engine}_{selected_cycle}")
            pdf_bytes = generate_pdf_report(
                selected_engine, selected_cycle, pred_rul, safe_floor, status_text, diagnosis, margin_calibrated, current_wo
            )
            st.download_button(
                "📄 Download PDF maintenance & copilot report",
                data=pdf_bytes,
                file_name=f"engine_{selected_engine}_cycle_{selected_cycle}_report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.warning(f"Couldn't generate the PDF report right now ({e}). The rest of the dashboard is unaffected.")

st.divider()
st.caption(
    "Educational project — not for real-world maintenance decisions. "
    "Trained on the NASA C-MAPSS FD001 turbofan degradation dataset."
)
