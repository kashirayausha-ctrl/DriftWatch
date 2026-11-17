from __future__ import annotations
import numpy as np
from scipy.stats import ks_2samp


def _clean(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def calculate_psi(reference, current, bins=10, epsilon=1e-6):
    """Calculate PSI using reference quantile bins."""
    ref, cur = _clean(reference), _clean(current)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), epsilon, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), epsilon, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_test(reference, current, alpha=0.05):
    ref, cur = _clean(reference), _clean(current)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan"), float("nan"), False
    result = ks_2samp(ref, cur, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue), bool(result.pvalue < alpha)


def adwin_test(values, split_index):
    """Run River ADWIN on a numeric stream; gracefully report if River is absent."""
    try:
        from river.drift import ADWIN
    except ImportError:
        return {
            "score": float("nan"),
            "decision": "Unavailable",
            "details": "Install river to enable ADWIN (pip install river)."
        }

    detector = ADWIN()
    detected_at = None
    stream = np.asarray(values, dtype=float)
    for index, value in enumerate(stream):
        detector.update(float(value))
        if detector.drift_detected and index >= split_index:
            detected_at = index
            break
    if detected_at is None:
        return {"score": float("nan"), "decision": "No drift", "details": "No change point flagged in this stream."}
    delay = detected_at - split_index
    return {"score": float(delay), "decision": "Drift", "details": f"First alert {delay} samples after configured change point."}
