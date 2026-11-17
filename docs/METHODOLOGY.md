# DriftWatch — Research Methodology

## 1. Research objective

The objective is to study how changes in input distributions and feature–target relationships can be detected and how such changes relate to predictive model performance.

## 2. Experimental design

A labelled synthetic binary-classification dataset is generated with six numeric features. The ordered dataset is divided at a configurable change point into a reference window and a current window. A selected drift scenario is injected only into the current window.

### Scenarios

- **No drift:** no modification; serves as a negative control.
- **Feature drift:** selected feature marginals are shifted or rescaled.
- **Covariate drift:** feature-space distributions are perturbed while labels are not deliberately flipped.
- **Concept drift:** a controlled fraction of current-window labels is flipped, changing the observed feature–target relationship while leaving feature values unchanged.

The scenarios are controlled simulations and should not be interpreted as exhaustive representations of real-world drift.

## 3. Detection methods

### 3.1 Population Stability Index

Reference-derived quantile bins are used to estimate proportions in the reference and current windows. With smoothed proportions p_i and q_i:

PSI = Σ_i (q_i − p_i) ln(q_i / p_i)

The dashboard default threshold (0.20) is a practical heuristic and is exposed as a setting. Results should include the raw PSI value, not only a binary alert.

### 3.2 Kolmogorov–Smirnov test

For each numeric feature, the two-sample, two-sided KS test compares the empirical cumulative distribution functions. The KS statistic is the maximum absolute difference between the two empirical CDFs. The p-value is adjusted across tested features using the Benjamini–Hochberg procedure to control the false discovery rate.

The KS test is univariate. It can miss multivariate changes that do not appear in individual feature marginals.

### 3.3 ADWIN

ADWIN is applied to an ordered stream of held-out prediction errors (0 for correct, 1 for incorrect). It tests for changes in the stream mean and provides a complementary predictive-behaviour signal. It is not interchangeable with input-distribution tests. In practical deployment, labels must become available before error monitoring can operate.

## 4. Predictive performance evaluation

A Random Forest is trained on a training subset of the reference window. Performance is evaluated on a held-out reference subset and on the current window. Accuracy and F1 are reported for both periods. This avoids reporting in-sample training accuracy as the baseline.

The current window is used as the evaluation period in this prototype; a larger study should use repeated independent seeds and, where possible, an additional untouched test design.

## 5. Evaluation measures

For each scenario and seed, record:

- Feature-wise PSI and alert decision.
- KS statistic, raw p-value and adjusted p-value.
- ADWIN alert decision and delay relative to the held-out error-stream boundary, when detected.
- Held-out accuracy and F1 before and after the change point.
- False-positive rate on no-drift runs.
- Detection rate on injected-drift runs.

Report aggregate mean and standard deviation or confidence intervals across repeated seeds. Thresholds must be fixed before final evaluation or tuned on a separate validation set.

## 6. Threats to validity and limitations

- Synthetic data may not represent the complexity of production data.
- Label flipping approximates concept drift and may not reflect realistic changes in the target-generating process.
- PSI depends on binning and sample size.
- KS tests are univariate and multiple-testing correction does not address all forms of dependence.
- ADWIN performance depends on stream ordering, parameter settings and label availability.
- A detected data shift is not necessarily harmful; model impact must be evaluated separately.

## 7. Reproducibility

Record the random seed, sample count, scenario, drift strength, change-point percentage, PSI threshold, adjusted KS alpha, package versions and all raw per-run outputs. Run multiple seeds and retain the complete results table.
