"""Synthetic data generation, controlled drift injection and model evaluation."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [f"feature_{i}" for i in range(1, 7)]


def make_dataset(n_samples: int = 3000, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Create a reproducible binary-classification reference dataset."""
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100 for reliable evaluation.")
    X, y = make_classification(
        n_samples=n_samples, n_features=6, n_informative=4, n_redundant=1,
        n_clusters_per_class=2, class_sep=1.2, flip_y=0.02,
        random_state=random_state,
    )
    return pd.DataFrame(X, columns=FEATURE_COLUMNS), pd.Series(y, name="target")


def _concept_labels(X: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Generate labels from a changed decision function using features 3 and 4."""
    score = 1.25 * X["feature_3"].to_numpy() - 0.9 * X["feature_4"].to_numpy()
    probability = 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))
    return (rng.random(len(X)) < probability).astype(int)


def inject_drift(X: pd.DataFrame, y: pd.Series, start: int, drift_type: str,
                 strength: float = 1.0, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Apply controlled drift to rows at and after the selected change point.

    Feature drift changes selected marginal feature distributions. Covariate
    drift applies a joint feature-space transformation. Concept drift changes
    P(Y|X) by replacing the post-change label rule. Gradual concept drift
    blends old and new label rules across the post-change period.
    """
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if not 0 <= start <= len(X):
        raise ValueError("start must be between 0 and the number of samples.")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be a finite non-negative number.")
    if not set(FEATURE_COLUMNS).issubset(X.columns):
        raise ValueError(f"X must contain these columns: {sorted(FEATURE_COLUMNS)}")
    allowed = {"No drift", "Feature drift", "Covariate drift", "Concept drift", "Gradual concept drift"}
    if drift_type not in allowed:
        raise ValueError(f"drift_type must be one of: {sorted(allowed)}")

    X_new, y_new = X.copy(deep=True), y.copy(deep=True)
    tail_positions = np.arange(len(X_new)) >= start
    tail_index = X_new.index[tail_positions]
    rng = np.random.default_rng(random_state)
    if drift_type == "No drift" or start == len(X_new):
        return X_new, y_new

    if drift_type == "Feature drift":
        X_new.loc[tail_index, "feature_1"] += 1.5 * strength
        X_new.loc[tail_index, "feature_2"] *= 1.0 + 0.45 * strength
    elif drift_type == "Covariate drift":
        cols = ["feature_1", "feature_2", "feature_3"]
        block = X_new.loc[tail_index, cols].to_numpy(dtype=float)
        angle = min(0.6, 0.15 * strength)
        transform = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                              [np.sin(angle), np.cos(angle), 0.0],
                              [0.0, 0.0, 1.0 + 0.2 * strength]])
        transformed = block @ transform.T
        transformed[:, 0] += 0.5 * strength
        X_new.loc[tail_index, cols] = transformed
    elif drift_type in {"Concept drift", "Gradual concept drift"}:
        new_labels = _concept_labels(X_new.loc[tail_index], rng)
        old_labels = y_new.loc[tail_index].to_numpy(dtype=int)
        if drift_type == "Concept drift":
            y_new.loc[tail_index] = new_labels
        else:
            progress = np.linspace(0.0, 1.0, len(tail_index), endpoint=True)
            probability = np.clip(progress * min(1.0, strength), 0.0, 1.0)
            use_new_rule = rng.random(len(tail_index)) < probability
            y_new.loc[tail_index] = np.where(use_new_rule, new_labels, old_labels)
    return X_new, y_new


def _classification_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def evaluate_model(X: pd.DataFrame, y: pd.Series, split: int, random_state: int = 42,
                   validation_size: float = 0.25) -> dict[str, Any]:
    """Train on part of reference data; score held-out reference and current data.

    The reference period is stratified into training and validation subsets.
    The model sees only the training subset. Reference and current metrics
    therefore use independent evaluation observations.
    """
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if not 0 < split < len(X):
        raise ValueError("split must leave at least one reference and current sample.")
    if not 0 < validation_size < 0.5:
        raise ValueError("validation_size must be between 0 and 0.5.")
    X_ref, X_current = X.iloc[:split], X.iloc[split:]
    y_ref, y_current = y.iloc[:split], y.iloc[split:]
    if y_ref.nunique() < 2 or y_ref.value_counts().min() < 2:
        raise ValueError("Reference period must contain at least two samples of each target class.")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_ref, y_ref, test_size=validation_size, random_state=random_state, stratify=y_ref
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=random_state, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred_reference, pred_current = model.predict(X_valid), model.predict(X_current)
    before = _classification_metrics(y_valid, pred_reference)
    after = _classification_metrics(y_current, pred_current)
    return {
        "accuracy_before": before["accuracy"], "accuracy_after": after["accuracy"],
        "precision_before": before["precision"], "precision_after": after["precision"],
        "recall_before": before["recall"], "recall_after": after["recall"],
        "f1_before": before["f1"], "f1_after": after["f1"],
        "reference_train_size": int(len(X_train)),
        "reference_validation_size": int(len(X_valid)),
        "current_evaluation_size": int(len(X_current)),
        "model": model,
    }
