from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def make_dataset(n_samples=3000, random_state=42):
    X, y = make_classification(
        n_samples=n_samples, n_features=6, n_informative=4, n_redundant=1,
        n_clusters_per_class=2, class_sep=1.2, flip_y=0.02,
        random_state=random_state
    )
    columns = [f"feature_{i+1}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), pd.Series(y, name="target")


def inject_drift(X, y, start, drift_type, strength=1.0, random_state=42):
    """Return copies with a controlled change applied after the change point."""
    X_new, y_new = X.copy(), y.copy()
    if drift_type == "No drift":
        return X_new, y_new

    rng = np.random.default_rng(random_state)
    tail = X_new.index >= start
    if drift_type in ("Feature drift", "Covariate drift"):
        X_new.loc[tail, "feature_1"] += strength * 1.5
        X_new.loc[tail, "feature_2"] *= (1.0 + 0.45 * strength)
        if drift_type == "Covariate drift":
            X_new.loc[tail, "feature_3"] += rng.normal(
                loc=0.0, scale=0.35 * strength, size=int(tail.sum())
            )
    elif drift_type == "Concept drift":
        # Flip a controlled fraction of labels to alter P(y|x), while keeping
        # the feature distributions unchanged.
        flip_probability = min(0.45, 0.15 * strength)
        indices = np.flatnonzero(tail.to_numpy())
        flip_count = int(len(indices) * flip_probability)
        if flip_count:
            selected = rng.choice(indices, size=flip_count, replace=False)
            y_new.iloc[selected] = 1 - y_new.iloc[selected]
    return X_new, y_new


def evaluate_model(X, y, split, random_state=42):
    """Evaluate on held-out samples from each period, never on training rows."""
    X_before, y_before = X.iloc[:split], y.iloc[:split]
    X_after, y_after = X.iloc[split:], y.iloc[split:]
    if min(len(X_before), len(X_after)) < 20:
        raise ValueError("Each period needs at least 20 samples for evaluation.")

    X_train, X_test_before, y_train, y_test_before = train_test_split(
        X_before, y_before, test_size=0.25, random_state=random_state,
        stratify=y_before
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred_before = model.predict(X_test_before)
    pred_after = model.predict(X_after)
    errors_before = (pred_before != y_test_before.to_numpy()).astype(float)
    errors_after = (pred_after != y_after.to_numpy()).astype(float)
    error_stream = np.concatenate([errors_before, errors_after])

    return {
        "accuracy_before": accuracy_score(y_test_before, pred_before),
        "accuracy_after": accuracy_score(y_after, pred_after),
        "f1_before": f1_score(y_test_before, pred_before, zero_division=0),
        "f1_after": f1_score(y_after, pred_after, zero_division=0),
        "error_before": float(errors_before.mean()),
        "error_after": float(errors_after.mean()),
        "error_stream": error_stream,
        "error_split": len(errors_before),
    }
