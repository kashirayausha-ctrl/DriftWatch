import numpy as np
import pandas as pd
import pytest

from driftwatch.detectors import (
    calculate_psi, ks_test, wasserstein_test, jensen_shannon_test,
    page_hinkley_test,
)
from driftwatch.simulation import make_dataset, inject_drift, evaluate_model


def test_detectors_report_no_difference_for_identical_samples():
    sample = np.random.default_rng(7).normal(size=500)
    assert calculate_psi(sample, sample) < 1e-10
    statistic, p_value, drift = ks_test(sample, sample)
    assert statistic == 0
    assert p_value == 1
    assert drift is False
    assert wasserstein_test(sample, sample) == pytest.approx(0)
    assert jensen_shannon_test(sample, sample) == pytest.approx(0)


def test_detectors_detect_shifted_distribution():
    rng = np.random.default_rng(8)
    reference = rng.normal(0, 1, 1000)
    current = rng.normal(2, 1, 1000)
    assert calculate_psi(reference, current) > 0.1
    assert ks_test(reference, current)[2] is True
    assert wasserstein_test(reference, current) > 1.5
    assert jensen_shannon_test(reference, current) > 0


@pytest.mark.parametrize("drift_type", [
    "No drift", "Feature drift", "Covariate drift", "Concept drift",
    "Gradual concept drift",
])
def test_drift_injection_preserves_shape_and_does_not_mutate_inputs(drift_type):
    X, y = make_dataset(600, random_state=12)
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    X_new, y_new = inject_drift(X, y, 400, drift_type, random_state=13)
    assert X_new.shape == X.shape
    assert y_new.shape == y.shape
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_evaluation_uses_held_out_reference_samples():
    X, y = make_dataset(800, random_state=21)
    X_changed, y_changed = inject_drift(X, y, 520, "Feature drift")
    results = evaluate_model(X_changed, y_changed, split=520, random_state=21)
    assert results["reference_train_size"] + results["reference_validation_size"] == 520
    assert results["current_evaluation_size"] == 280
    for key in ("accuracy_before", "accuracy_after", "precision_before",
                "precision_after", "recall_before", "recall_after",
                "f1_before", "f1_after"):
        assert 0.0 <= results[key] <= 1.0


def test_invalid_detector_inputs_raise_clear_errors():
    with pytest.raises(ValueError):
        calculate_psi([], [1, 2])
    with pytest.raises(ValueError):
        ks_test([1, 2], [3, 4], alpha=1.0)


def test_page_hinkley_handles_insufficient_data():
    result = page_hinkley_test([0.0, 0.1, 0.0], split_index=1, min_instances=10)
    assert result["decision"] == "Insufficient data"
