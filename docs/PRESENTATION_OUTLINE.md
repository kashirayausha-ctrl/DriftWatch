# DriftWatch — Research Presentation Outline

Suggested 10-slide technical presentation. Replace all bracketed result placeholders with measured results from repeated experiments; do not invent values.

## Slide 1 — Title
**DriftWatch: Monitoring Data and Concept Drift in Machine Learning Systems**
- Research project title
- Team/member details
- Department, institution and academic term

## Slide 2 — Problem statement
- ML models are deployed in environments where incoming data can change.
- Distribution changes can reduce reliability and predictive performance.
- Monitoring should distinguish input-distribution changes from changes in predictive behaviour.

## Slide 3 — Objectives
- Simulate controlled drift scenarios.
- Compare feature distributions with PSI and KS.
- Control multiple feature-wise statistical tests with Benjamini–Hochberg.
- Monitor held-out prediction errors with ADWIN.
- Compare held-out model performance before and after change.

## Slide 4 — Types of drift
- Feature/data drift: changes in one or more input feature distributions.
- Covariate drift: change in P(X).
- Concept drift: change in P(y|X).
- Explain that concept drift can occur without visible marginal feature changes.

## Slide 5 — System architecture
Show the architecture diagram from docs/ARCHITECTURE.md.
Explain the flow: controls → synthetic data → drift injection → detectors/model evaluation → dashboard.

## Slide 6 — Detection algorithms
- PSI: binned distribution divergence; threshold is heuristic.
- KS: two-sample univariate distribution test.
- Benjamini–Hochberg: false-discovery-rate adjustment across features.
- ADWIN: adaptive streaming detector on ordered prediction errors.

## Slide 7 — Implementation
- Python, Streamlit, Pandas, NumPy, SciPy, scikit-learn, Plotly and River.
- Explain the role of app.py, simulation.py and detectors.py.
- Include genuine screenshots from the running dashboard.

## Slide 8 — Experimental setup
Report actual settings:
- Sample size: [measured setting]
- Drift start: [percentage]
- Drift strength: [setting]
- Random seeds: [list/count]
- PSI threshold and adjusted KS alpha: [values]
- Include no-drift control and all injected scenarios.

## Slide 9 — Results and discussion
Insert measured tables/charts:
- Detection rate by scenario: [results]
- False-positive rate on no-drift runs: [results]
- Per-feature PSI and adjusted KS results: [results]
- Held-out accuracy/F1 before vs after: [results]
- ADWIN detection delay: [results]
Discuss cases where feature drift and model degradation do not coincide.

## Slide 10 — Conclusion and future work
- Summarize measured findings only.
- Limitations: synthetic data, univariate tests, simplified concept-drift injection, label delay for error monitoring.
- Future work: real datasets, multivariate detectors, delayed-label monitoring, repeated statistical evaluation and production deployment.

## Likely viva questions

**Q: Is data drift the same as concept drift?**
No. Data/covariate drift concerns the input distribution; concept drift concerns the conditional relationship between inputs and target.

**Q: Why use both PSI and KS?**
PSI provides a practical magnitude-style divergence score based on bins, while KS provides a statistical two-sample test for a univariate feature.

**Q: Why adjust KS p-values?**
Testing many features increases the chance of false positives. Benjamini–Hochberg controls the expected false discovery rate under its assumptions.

**Q: Does drift always reduce accuracy?**
No. A distribution can change without harming the model; performance must be measured separately.

**Q: Why use ADWIN on prediction errors?**
It provides a streaming signal about changes in observed predictive error. It requires labels to arrive and is distinct from input-feature monitoring.

**Q: Why use synthetic data?**
It provides known change points and controlled scenarios for initial testing. Real datasets are needed to assess external validity.
