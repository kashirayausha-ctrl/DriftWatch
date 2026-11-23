"""Statistical and streaming drift detectors used by DriftWatch."""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


def _clean(values: Any) -> np.ndarray:
    """Convert input to a finite, one-dimensional float array."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _validate_samples(reference: Any, current: Any) -> tuple[np.ndarray, np.ndarray]:
    ref, cur = _clean(reference), _clean(current)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("Reference and current samples must each contain at least one finite value.")
    return ref, cur


def calculate_psi(reference: Any, current: Any, bins: int = 10, epsilon: float = 1e-6) -> float:
    """Calculate PSI using quantile bin edges learned only from reference."""
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5.")
    ref, cur = _validate_samples(reference, current)
    if np.all(ref == ref[0]):
        ref_pct = np.array([1.0, 0.0])
        cur_pct = np.array([np.mean(cur == ref[0]), np.mean(cur != ref[0])])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
        if edges.size < 2:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)
        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_test(reference: Any, current: Any, alpha: float = 0.05) -> tuple[float, float, bool]:
    """Two-sided two-sample KS test for continuous, one-dimensional samples."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    ref, cur = _validate_samples(reference, current)
    result = ks_2samp(ref, cur, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue), bool(result.pvalue < alpha)


def wasserstein_test(reference: Any, current: Any) -> float:
    """Return the first Wasserstein distance (Earth Mover's Distance)."""
    ref, cur = _validate_samples(reference, current)
    return float(wasserstein_distance(ref, cur))


def jensen_shannon_test(reference: Any, current: Any, bins: int = 20, epsilon: float = 1e-10) -> float:
    """Estimate Jensen-Shannon distance from histograms on shared bin edges."""
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    ref, cur = _validate_samples(reference, current)
    low, high = min(float(ref.min()), float(cur.min())), max(float(ref.max()), float(cur.max()))
    if low == high:
        return 0.0
    edges = np.linspace(low, high, bins + 1)
    p, _ = np.histogram(ref, bins=edges)
    q, _ = np.histogram(cur, bins=edges)
    p, q = p.astype(float) / p.sum(), q.astype(float) / q.sum()
    p, q = np.clip(p, epsilon, None), np.clip(q, epsilon, None)
    p, q = p / p.sum(), q / q.sum()
    return float(jensenshannon(p, q, base=np.e))


def adwin_test(values: Any, split_index: int, delta: float = 0.002) -> dict[str, Any]:
    """Run River ADWIN on one numeric stream and report first post-split alert."""
    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1.")
    stream = _clean(values)
    if split_index < 0 or split_index > stream.size:
        raise ValueError("split_index must be between 0 and the stream length.")
    try:
        from river.drift import ADWIN
    except ImportError:
        return {"score": float("nan"), "decision": "Unavailable",
                "details": "Install river to enable ADWIN (pip install river)."}
    detector, detected_at = ADWIN(delta=delta), None
    for index, value in enumerate(stream):
        detector.update(float(value))
        if detector.drift_detected and index >= split_index:
            detected_at = index
            break
    if detected_at is None:
        return {"score": float("nan"), "decision": "No drift",
                "details": "No change point flagged in this stream."}
    delay = detected_at - split_index
    return {"score": float(delay), "decision": "Drift",
            "details": f"First alert {delay} samples after configured change point."}


def page_hinkley_test(values: Any, split_index: int, min_instances: int = 30,
                      delta: float = 0.005, threshold: float = 50.0,
                      alpha: float = 0.9999) -> dict[str, Any]:
    """Page-Hinkley mean-shift detector for an ordered numeric stream."""
    stream = _clean(values)
    if split_index < 0 or split_index > stream.size:
        raise ValueError("split_index must be between 0 and the stream length.")
    if min_instances < 1 or delta < 0 or threshold <= 0 or not 0 < alpha <= 1:
        raise ValueError("Invalid Page-Hinkley parameters.")
    if stream.size < min_instances:
        return {"score": float("nan"), "decision": "Insufficient data",
                "details": f"Need at least {min_instances} finite observations."}
    mean, cumulative, minimum, detected_at = 0.0, 0.0, 0.0, None
    for index, value in enumerate(stream, start=1):
        mean += (float(value) - mean) / index
        cumulative = alpha * cumulative + (float(value) - mean - delta)
        minimum = min(minimum, cumulative)
        if index >= max(min_instances, split_index + 1) and cumulative - minimum > threshold:
            detected_at = index - 1
            break
    if detected_at is None or detected_at < split_index:
        return {"score": float("nan"), "decision": "No drift",
                "details": "No post-change mean shift crossed the configured threshold."}
    delay = detected_at - split_index
    return {"score": float(delay), "decision": "Drift",
            "details": f"First alert {delay} samples after configured change point."}
