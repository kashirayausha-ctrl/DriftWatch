from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def make_dataset(n_samples=3000, random_state=42):
    X, y = make_classification(
        n_samples=n_samples, n_features=6, n_informative=4, n_redundant=1,
        n_clusters_per_class=2, class_sep=1.2, flip_y=0.02,
        random_state=random_state
    )
    columns = [f"feature_{i+1}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), pd.Series(y, name="target")


def inject_drift(X, y, start, drift_type, strength=1.0, random_state=42):
    """Return a copy with drift applied only after the change point."""
    X_new = X.copy()
    y_new = y.copy()
    if drift_type == "No drift":
        return X_new, y_new

    rng = np.random.default_rng(random_state)
    tail = X_new.index >= start
    if drift_type in ("Feature drift", "Covariate drift"):
        # Shift and rescale several feature distributions.
        X_new.loc[tail, "feature_1"] += strength * 1.5
        X_new.loc[tail, "feature_2"] *= (1.0 + 0.45 * strength)
        if drift_type == "Covariate drift":
            # Add a further feature-space perturbation without changing labels.
            X_new.loc[tail, "feature_3"] += rng.normal(
                loc=0.0, scale=0.35 * strength, size=int(tail.sum())
            )
    elif drift_type == "Concept drift":
        # Change the target-generating relationship by flipping a controlled
        # fraction of labels after the change point.
        flip_probability = min(0.45, 0.15 * strength)
        indices = np.flatnonzero(tail.to_numpy())
        flip_count = int(len(indices) * flip_probability)
        if flip_count:
            selected = rng.choice(indices, size=flip_count, replace=False)
            y_new.iloc[selected] = 1 - y_new.iloc[selected]
    return X_new, y_new


def evaluate_model(X, y, split):
    """Train on reference data and score separately on both time periods."""
    model = RandomForestClassifier(
        n_estimators=160, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X.iloc[:split], y.iloc[:split])
    pred_before = model.predict(X.iloc[:split])
    pred_after = model.predict(X.iloc[split:])
    return {
        "accuracy_before": accuracy_score(y.iloc[:split], pred_before),
        "accuracy_after": accuracy_score(y.iloc[split:], pred_after),
        "f1_before": f1_score(y.iloc[:split], pred_before, zero_division=0),
        "f1_after": f1_score(y.iloc[split:], pred_after, zero_division=0),
    }
