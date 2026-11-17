# DriftWatch

**DriftWatch** is a Streamlit research prototype for monitoring changes in machine-learning input data and predictive behaviour. It generates controlled synthetic experiments and compares reference and current data windows.

## Current capabilities

- Synthetic binary-classification data with configurable change point.
- No-drift, feature-drift, covariate-drift and concept-drift scenarios.
- Feature-wise Population Stability Index (PSI).
- Feature-wise two-sample Kolmogorov–Smirnov (KS) tests.
- Benjamini–Hochberg false-discovery-rate adjustment across feature-wise KS tests.
- ADWIN monitoring on an ordered held-out prediction-error stream.
- Held-out model accuracy and F1 comparison before and after the change point.
- Streamlit dashboard with Plotly distribution and performance charts.

## Architecture

```text
app.py
  ├── Experiment controls and Streamlit dashboard
  ├── driftwatch/simulation.py
  │     ├── synthetic dataset generation
  │     ├── controlled drift injection
  │     └── held-out model evaluation
  └── driftwatch/detectors.py
        ├── PSI
        ├── KS test
        ├── Benjamini–Hochberg correction
        └── ADWIN on prediction errors
```

## Setup (Windows / VS Code)

Use Python 3.11–3.13 for broad compatibility with scientific packages.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If Python 3.12 is unavailable, create the environment with another supported installed Python version. Open the local URL printed by Streamlit, usually http://localhost:8501.

## How to run an experiment

1. Select a scenario in the sidebar.
2. Set sample count, drift strength, change-point percentage and random seed.
3. Set the PSI heuristic threshold and adjusted KS significance level.
4. Click **Run drift experiment**.
5. Review the feature-wise table, selected feature distribution, ADWIN error-stream result and held-out performance metrics.
6. Repeat with the **No drift** scenario and multiple random seeds as a negative control.

## Interpreting the detectors

- **PSI** measures the change in binned proportions. The dashboard default of 0.20 is a configurable operational heuristic, not a universal statistical boundary.
- **KS** tests whether two one-dimensional samples could plausibly come from the same continuous distribution. The dashboard adjusts the family of feature-wise p-values using Benjamini–Hochberg.
- **ADWIN** monitors the sequence of held-out 0/1 prediction errors. It is a separate signal from input-feature drift and may respond to changes in predictive behaviour.
- **Concept drift** concerns a change in the conditional relationship between features and target, P(y|X). It can occur even when P(X) remains stable; therefore a clean feature table does not rule it out.

A detected distribution shift does not by itself prove that model performance has degraded. Conversely, model error may change without a large marginal feature shift.

## Research evaluation protocol

For defensible results:

1. Run each scenario across multiple predeclared random seeds.
2. Include no-drift negative controls and several drift strengths.
3. Record per-feature PSI, raw KS p-value, adjusted p-value and decisions.
4. Report detection rate, false-positive rate, detection delay (where measurable), and held-out accuracy/F1 change.
5. Report mean and variability or confidence intervals across repeated runs.
6. Do not tune thresholds on the same runs used for final evaluation.

**Important:** The repository does not claim benchmark accuracy or detection performance until repeated experiments have been run and recorded. Populate the results section of the research report only with measured outputs.

## Limitations

- Current data are synthetic; real-world datasets and domain-specific validation are required.
- PSI depends on binning and its thresholds are heuristic.
- KS is univariate and does not detect all multivariate changes.
- The concept-drift simulation uses controlled label flips as a simplified change to P(y|X), not a complete causal model.
- ADWIN monitors model error only when timely labels are available.
- The current experiment is a prototype, not a production model-monitoring service.

## Project structure

```text
DriftWatch/
├── app.py
├── driftwatch/
│   ├── __init__.py
│   ├── detectors.py
│   └── simulation.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── METHODOLOGY.md
│   └── PRESENTATION_OUTLINE.md
├── requirements.txt
└── README.md
```

## Dependencies

See [requirements.txt](requirements.txt). ADWIN requires River, which is included in the dependency list.
