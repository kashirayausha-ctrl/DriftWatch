# DriftWatch

A Streamlit prototype for experimenting with machine-learning data drift.

## Features
- Synthetic binary-classification dataset generation.
- Controlled no-drift, feature-drift, covariate-drift, and concept-drift scenarios.
- PSI and two-sample Kolmogorov–Smirnov tests.
- ADWIN change detection using River.
- Random Forest performance comparison before and after the change point.
- Interactive Plotly distribution and metric charts.

## Setup (Windows / VS Code)

Use Python 3.11–3.13 for the smoothest compatibility with scientific packages.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `py -3.12` is not installed, use an installed compatible Python version:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in the terminal, usually http://localhost:8501.

## Important research notes
- PSI thresholds (0.1 and 0.2) are common heuristics, not universal statistical rules.
- KS tests compare one-dimensional empirical distributions. Run feature-wise tests and consider multiple-testing correction for formal experiments.
- ADWIN is a streaming change detector; this prototype feeds it the selected feature values as a stream. For a stronger experiment, feed prediction errors or model residuals to ADWIN as a separate monitoring stream.
- Concept drift means the relationship between features and labels changes. This synthetic implementation approximates it by flipping labels after the change point.
- The model is trained on the reference period and evaluated on the reference and current periods. For rigorous evaluation, use multiple random seeds, repeated runs, confidence intervals, and a held-out test design.
