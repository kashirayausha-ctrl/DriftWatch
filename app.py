from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from driftwatch.detectors import analyze_feature_drift, adwin_test
from driftwatch.simulation import make_dataset, inject_drift, evaluate_model

st.set_page_config(page_title="DriftWatch", page_icon="📡", layout="wide")
st.title("📡 DriftWatch")
st.caption("Feature-wise distribution monitoring and model-performance monitoring.")

with st.sidebar:
    st.header("Experiment settings")
    n_samples = st.slider("Number of samples", 500, 10000, 3000, step=500)
    drift_type = st.selectbox(
        "Drift scenario",
        ["No drift", "Feature drift", "Covariate drift", "Concept drift"]
    )
    drift_strength = st.slider("Drift strength", 0.1, 3.0, 1.0, 0.1)
    drift_start = st.slider("Drift starts at (%)", 40, 90, 65, 5)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    psi_threshold = st.slider("PSI alert threshold", 0.05, 0.50, 0.20, 0.01)
    alpha = st.slider("Adjusted KS significance", 0.001, 0.10, 0.05, 0.001)

if st.button("Run drift experiment", type="primary"):
    with st.spinner("Generating data and running detectors..."):
        X, y = make_dataset(n_samples=int(n_samples), random_state=int(random_state))
        split = int(len(X) * drift_start / 100)
        X_changed, y_changed = inject_drift(
            X, y, start=split, drift_type=drift_type,
            strength=float(drift_strength), random_state=int(random_state)
        )
        results = evaluate_model(X_changed, y_changed, split)
        feature_results = analyze_feature_drift(
            X_changed.iloc[:split], X_changed.iloc[split:],
            psi_threshold=float(psi_threshold), alpha=float(alpha)
        )
        error_stream = results["error_stream"]
        error_adwin = adwin_test(error_stream, results["error_split"])
        st.session_state["experiment"] = {
            "X": X_changed, "y": y_changed, "split": split,
            "results": results, "feature_results": feature_results,
            "error_adwin": error_adwin, "drift_type": drift_type,
            "drift_strength": drift_strength, "psi_threshold": psi_threshold,
            "alpha": alpha
        }

if "experiment" not in st.session_state:
    st.info("Choose an experiment in the sidebar, then click **Run drift experiment**.")
    st.markdown("""
    **Monitoring workflow**
    - Generate a labelled synthetic classification dataset.
    - Inject a controlled distribution or target-relationship change.
    - Compare every numeric feature using PSI and KS tests.
    - Apply Benjamini-Hochberg correction across feature-wise KS tests.
    - Monitor model errors with ADWIN as a separate streaming signal.
    """)
    st.stop()

exp = st.session_state["experiment"]
X, y, split, results = exp["X"], exp["y"], exp["split"], exp["results"]
feature_results = exp["feature_results"]

drifted_count = int((feature_results["Decision"] == "Drift").sum())
k1, k2, k3, k4 = st.columns(4)
k1.metric("Scenario", exp["drift_type"])
k2.metric("Features flagged", f"{drifted_count}/{len(feature_results)}")
k3.metric("Accuracy before", f"{results['accuracy_before']:.3f}")
k4.metric("Accuracy after", f"{results['accuracy_after']:.3f}",
          delta=f"{results['accuracy_after'] - results['accuracy_before']:+.3f}")

st.subheader("1. Feature-wise drift overview")
st.caption(
    "A feature is flagged when PSI meets the practical threshold or its KS p-value "
    "remains significant after Benjamini-Hochberg correction."
)
st.dataframe(
    feature_results.style.format({
        "PSI": "{:.4f}", "KS statistic": "{:.4f}",
        "p-value": "{:.4g}", "Adjusted p-value": "{:.4g}"
    }),
    use_container_width=True, hide_index=True
)
if drifted_count:
    st.warning(f"{drifted_count} feature(s) show evidence of distributional drift.")
else:
    st.success("No feature-wise distributional drift was detected at the configured thresholds.")

st.subheader("2. Inspect a feature")
feature = st.selectbox("Feature", list(X.select_dtypes(include=np.number).columns))
plot_df = pd.DataFrame({
    feature: X[feature].to_numpy(),
    "Period": np.where(np.arange(len(X)) < split, "Reference", "Current")
})
fig = px.histogram(
    plot_df, x=feature, color="Period", barmode="overlay",
    histnorm="probability density", opacity=0.55,
    title=f"Reference vs current distribution: {feature}"
)
st.plotly_chart(fig, use_container_width=True)

selected = feature_results.loc[feature_results["Feature"] == feature]
if not selected.empty:
    st.dataframe(selected, use_container_width=True, hide_index=True)

st.subheader("3. Model-error monitoring")
error_adwin = exp["error_adwin"]
a1, a2, a3 = st.columns(3)
a1.metric("Error rate before", f"{results['error_before']:.3f}")
a2.metric("Error rate after", f"{results['error_after']:.3f}")
a3.metric("ADWIN on error stream", error_adwin["decision"])
st.caption(error_adwin["details"])
st.caption(
    "ADWIN monitors the ordered 0/1 prediction-error stream. It is distinct from "
    "feature-distribution tests and is more relevant to detecting changes in predictive behaviour."
)

st.subheader("4. Predictive performance")
perf_df = pd.DataFrame({
    "Period": ["Before drift", "After drift"],
    "Accuracy": [results["accuracy_before"], results["accuracy_after"]],
    "F1 score": [results["f1_before"], results["f1_after"]],
})
st.plotly_chart(
    px.bar(
        perf_df.melt("Period", var_name="Metric", value_name="Score"),
        x="Metric", y="Score", color="Period", barmode="group",
        range_y=[0, 1], title="Predictive performance comparison"
    ),
    use_container_width=True
)
st.dataframe(perf_df, use_container_width=True, hide_index=True)

st.subheader("5. Interpretation")
st.write(
    "Feature drift describes changes in input distributions. Concept drift describes "
    "changes in the relationship between inputs and the target and may occur without "
    "any feature-wise distribution shift. Therefore, a stable feature table does not "
    "rule out concept drift; inspect the model-error stream and labelled performance."
)
st.caption(
    f"Experiment settings: {exp['drift_type']} | strength {exp['drift_strength']} | "
    f"change point {split:,}/{len(X):,} | PSI threshold {exp['psi_threshold']:.2f} | "
    f"adjusted KS alpha {exp['alpha']:.3f}"
)
