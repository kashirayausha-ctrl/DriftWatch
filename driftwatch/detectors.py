from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _clean(values):
    """Convert a one-dimensional input to finite floating-point values."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def calculate_psi(reference, current, bins=10, epsilon=1e-6):
    """Population Stability Index using bin edges learned only from reference.

    Handles constant reference features explicitly instead of silently returning
    zero when the current sample moves away from that constant value.
    """
    ref, cur = _clean(reference), _clean(current)
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    if bins < 2 or epsilon <= 0:
        raise ValueError("bins must be >= 2 and epsilon must be > 0")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))

    if edges.size < 2:
        # A constant baseline still needs a boundary that can separate a
        # changed current value from the reference observations.
        value = float(ref[0])
        margin = max(abs(value) * 1e-6, 1e-6)
        edges = np.array([-np.inf, value + margin, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / ref.size, epsilon, None)
    cur_pct = np.clip(cur_counts / cur.size, epsilon, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_test(reference, current, alpha=0.05):
    """Two-sided, two-sample Kolmogorov-Smirnov test."""
    ref, cur = _clean(reference), _clean(current)
    if ref.size == 0 or cur.size == 0:
        return float("nan"), float("nan"), False
    result = ks_2samp(ref, cur, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue), bool(result.pvalue < alpha)


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted p-values; NaNs remain NaN."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return adjusted
    values = p[valid]
    order = np.argsort(values)
    ranked = values[order]
    m = ranked.size
    corrected = ranked * m / np.arange(1, m + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)
    restored = np.empty_like(corrected)
    restored[order] = corrected
    adjusted[valid] = restored
    return adjusted


def analyze_feature_drift(reference, current, psi_threshold=0.20, alpha=0.05):
    """Return a row for every numeric feature with multiplicity-adjusted KS tests.

    A feature is flagged when either its PSI reaches the configured practical
    threshold or its Benjamini-Hochberg adjusted KS p-value is below alpha.
    PSI and statistical evidence are kept as separate columns for review.
    """
    if not isinstance(reference, pd.DataFrame) or not isinstance(current, pd.DataFrame):
        raise TypeError("reference and current must be pandas DataFrames")

    columns = [c for c in reference.columns if c in current.columns]
    rows = []
    for column in columns:
        if not (pd.api.types.is_numeric_dtype(reference[column]) and
                pd.api.types.is_numeric_dtype(current[column])):
            continue
        ref = _clean(reference[column])
        cur = _clean(current[column])
        if ref.size == 0 or cur.size == 0:
            rows.append({
                "Feature": column, "PSI": np.nan, "KS statistic": np.nan,
                "p-value": np.nan, "Adjusted p-value": np.nan,
                "PSI signal": False, "KS signal": False, "Decision": "Insufficient data"
            })
            continue
        psi = calculate_psi(ref, cur)
        statistic, p_value, _ = ks_test(ref, cur, alpha=alpha)
        rows.append({
            "Feature": column, "PSI": psi, "KS statistic": statistic,
            "p-value": p_value, "Adjusted p-value": np.nan,
            "PSI signal": bool(np.isfinite(psi) and psi >= psi_threshold),
            "KS signal": False, "Decision": "No drift"
        })

    adjusted = benjamini_hochberg([row["p-value"] for row in rows])
    for row, q_value in zip(rows, adjusted):
        row["Adjusted p-value"] = float(q_value) if np.isfinite(q_value) else np.nan
        row["KS signal"] = bool(np.isfinite(q_value) and q_value < alpha)
        if row["PSI signal"] or row["KS signal"]:
            row["Decision"] = "Drift"
        elif row["Decision"] != "Insufficient data":
            row["Decision"] = "No drift"

    return pd.DataFrame(rows)


def adwin_test(values, split_index, delta=0.002):
    """Run River ADWIN on an ordered numeric stream."""
    try:
        from river.drift import ADWIN
    except ImportError:
        return {
            "score": float("nan"),
            "decision": "Unavailable",
            "details": "Install river to enable ADWIN (pip install river)."
        }

    stream = _clean(values)
    detector = ADWIN(delta=delta)
    detected_at = None
    for index, value in enumerate(stream):
        detector.update(float(value))
        if detector.drift_detected and index >= split_index:
            detected_at = index
            break

    if detected_at is None:
        return {
            "score": float("nan"), "decision": "No drift",
            "details": "No change point flagged in this stream."
        }
    delay = detected_at - split_index
    return {
        "score": float(delay), "decision": "Drift",
        "details": f"First alert {delay} samples after configured change point."
    }
