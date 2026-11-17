from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from driftwatch.detectors import calculate_psi, ks_test, adwin_test
from driftwatch.simulation import make_dataset, inject_drift, evaluate_model

st.set_page_config(page_title="DriftWatch", page_icon="📡", layout="wide")
st.title("📡 DriftWatch")
st.caption("Monitor changing data distributions and understand their impact on model performance.")

with st.sidebar:
    st.header("Experiment settings")
    n_samples = st.slider("Number of samples", 500, 10000, 3000, step=500)
    drift_type = st.selectbox("Drift scenario", ["No drift", "Feature drift", "Covariate drift", "Concept drift"])
    drift_strength = st.slider("Drift strength", 0.1, 3.0, 1.0, 0.1)
    drift_start = st.slider("Drift starts at (%)", 40, 90, 65, 5)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42)

if st.button("Run drift experiment", type="primary"):
    with st.spinner("Generating data and running detectors..."):
        X, y = make_dataset(n_samples=int(n_samples), random_state=int(random_state))
        split = int(len(X) * drift_start / 100)
        X_changed, y_changed = inject_drift(
            X, y, start=split, drift_type=drift_type, strength=float(drift_strength),
            random_state=int(random_state)
        )
        results = evaluate_model(X_changed, y_changed, split)
        st.session_state["experiment"] = {
            "X": X_changed, "y": y_changed, "split": split, "results": results,
            "drift_type": drift_type, "drift_strength": drift_strength
        }

if "experiment" not in st.session_state:
    st.info("Choose an experiment in the sidebar, then click **Run drift experiment**.")
    st.markdown("""
    **What this prototype does**
    - Creates a labelled synthetic dataset.
    - Injects feature, covariate, or concept drift after a selected point.
    - Compares PSI, Kolmogorov–Smirnov (KS), and ADWIN where installed.
    - Compares model performance before and after the drift point.
    """)
    st.stop()

exp = st.session_state["experiment"]
X, y, split, results = exp["X"], exp["y"], exp["split"], exp["results"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Scenario", exp["drift_type"])
m2.metric("Drift starts", f"{split:,} samples")
m3.metric("Accuracy before", f"{results['accuracy_before']:.3f}")
m4.metric("Accuracy after", f"{results['accuracy_after']:.3f}",
          delta=f"{results['accuracy_after'] - results['accuracy_before']:+.3f}")

st.subheader("1. Data distribution")
feature = st.selectbox("Feature to inspect", list(X.columns))
plot_df = pd.DataFrame({
    feature: X[feature],
    "Period": np.where(np.arange(len(X)) < split, "Reference", "Current")
})
fig = px.histogram(plot_df, x=feature, color="Period", barmode="overlay",
                   histnorm="probability density", opacity=0.55,
                   title=f"Reference vs current distribution: {feature}")
st.plotly_chart(fig, use_container_width=True)

st.subheader("2. Drift detection results")
reference = X.iloc[:split][feature].to_numpy()
current = X.iloc[split:][feature].to_numpy()
psi = calculate_psi(reference, current)
ks_stat, ks_p, ks_drift = ks_test(reference, current)
adwin = adwin_test(np.r_[reference, current], split)

detector_df = pd.DataFrame([
    {"Detector": "PSI", "Score": psi, "Decision": "Drift" if psi >= 0.2 else ("Possible drift" if psi >= 0.1 else "No drift"),
     "Interpretation": "PSI < 0.1: stable; 0.1–0.2: moderate; ≥ 0.2: significant"},
    {"Detector": "KS test", "Score": ks_stat, "Decision": "Drift" if ks_drift else "No drift",
     "Interpretation": f"p-value = {ks_p:.4g}; significance level = 0.05"},
    {"Detector": "ADWIN", "Score": adwin["score"], "Decision": adwin["decision"],
     "Interpretation": adwin["details"]},
])
st.dataframe(detector_df, use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)
with c1:
    st.metric("PSI", f"{psi:.4f}")
    st.caption("Population Stability Index. Thresholds are practical heuristics, not universal guarantees.")
with c2:
    st.metric("KS p-value", f"{ks_p:.4g}")
    st.caption("A small p-value indicates evidence that the two samples differ.")

st.subheader("3. Model performance")
perf_df = pd.DataFrame({
    "Period": ["Before drift", "After drift"],
    "Accuracy": [results["accuracy_before"], results["accuracy_after"]],
    "F1 score": [results["f1_before"], results["f1_after"]],
})
st.plotly_chart(px.bar(perf_df.melt("Period", var_name="Metric", value_name="Score"),
                       x="Metric", y="Score", color="Period", barmode="group",
                       range_y=[0, 1], title="Predictive performance comparison"),
                use_container_width=True)
st.dataframe(perf_df, use_container_width=True, hide_index=True)

st.subheader("4. Experiment notes")
st.write(
    "The reference period is used as the baseline distribution. The current period is compared "
    "against it. A drift alert indicates a distributional change; it does not automatically prove "
    "that model quality has degraded. Confirm impact using labelled outcomes and performance metrics."
)
