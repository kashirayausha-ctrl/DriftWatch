"""Stateful, feature-wise streaming drift monitoring.

The first reference_window_size observations establish a fixed baseline.
Subsequent observations enter a rolling current window. At configured
intervals, each feature is compared with its baseline using PSI and a
two-sample KS test. An event is emitted only when a feature transitions
from stable to drifted; persistent drift does not generate duplicate events.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from driftwatch.detectors import calculate_psi, ks_test, wasserstein_test


class StreamingDriftMonitor:
    """Incremental numeric-feature drift monitor with a fixed reference window.

    Parameters
    ----------
    reference_window_size:
        Number of initial observations used to establish the baseline.
    current_window_size:
        Number of most recent post-baseline observations used for comparison.
    check_interval:
        Evaluate the current window after it fills, then every N new samples.
    psi_threshold:
        PSI threshold that contributes to a drift decision.
    ks_alpha:
        Significance level for the two-sample KS test.
    max_events:
        Maximum number of recent drift events retained in memory.
    features:
        Optional expected feature names. If omitted, the first sample defines
        the feature schema.
    """

    def __init__(
        self,
        reference_window_size: int = 200,
        current_window_size: int = 100,
        check_interval: int = 25,
        psi_threshold: float = 0.2,
        ks_alpha: float = 0.05,
        max_events: int = 500,
        features: Sequence[str] | None = None,
    ) -> None:
        if reference_window_size < 2:
            raise ValueError("reference_window_size must be at least 2.")
        if current_window_size < 2:
            raise ValueError("current_window_size must be at least 2.")
        if check_interval < 1:
            raise ValueError("check_interval must be at least 1.")
        if not np.isfinite(psi_threshold) or psi_threshold < 0:
            raise ValueError("psi_threshold must be finite and non-negative.")
        if not 0 < ks_alpha < 1:
            raise ValueError("ks_alpha must be between 0 and 1.")
        if max_events < 1:
            raise ValueError("max_events must be at least 1.")

        configured_features = tuple(features) if features is not None else None
        if configured_features is not None:
            if not configured_features or len(set(configured_features)) != len(configured_features):
                raise ValueError("features must be a non-empty sequence of unique names.")
            if any(not isinstance(name, str) or not name for name in configured_features):
                raise ValueError("feature names must be non-empty strings.")

        self.reference_window_size = int(reference_window_size)
        self.current_window_size = int(current_window_size)
        self.check_interval = int(check_interval)
        self.psi_threshold = float(psi_threshold)
        self.ks_alpha = float(ks_alpha)
        self.max_events = int(max_events)
        self._configured_features = configured_features
        self.reset()

    @staticmethod
    def _timestamp(value: datetime | str | None) -> str:
        if value is None:
            stamp = datetime.now(timezone.utc)
        elif isinstance(value, datetime):
            stamp = value
        elif isinstance(value, str):
            try:
                stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("timestamp must be a valid ISO-8601 string.") from exc
        else:
            raise TypeError("timestamp must be a datetime, ISO-8601 string, or None.")

        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone.utc).isoformat()

    def _validate_sample(self, sample: Mapping[str, Any]) -> dict[str, float]:
        if not isinstance(sample, Mapping) or not sample:
            raise ValueError("sample must be a non-empty mapping of feature names to numbers.")

        keys = tuple(sample.keys())
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("feature names must be non-empty strings.")

        if self._features is None:
            if self._configured_features is not None:
                missing = set(self._configured_features) - set(keys)
                extra = set(keys) - set(self._configured_features)
                if missing or extra:
                    raise ValueError(
                        f"sample feature schema mismatch; missing={sorted(missing)}, "
                        f"extra={sorted(extra)}."
                    )
                self._features = self._configured_features
            else:
                self._features = keys
        elif set(keys) != set(self._features):
            missing = set(self._features) - set(keys)
            extra = set(keys) - set(self._features)
            raise ValueError(
                f"sample feature schema mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}."
            )

        converted: dict[str, float] = {}
        for feature in self._features:
            value = sample[feature]
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"feature '{feature}' must be numeric, not boolean.")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"feature '{feature}' must be numeric.") from exc
            if not np.isfinite(number):
                raise ValueError(f"feature '{feature}' must be finite.")
            converted[feature] = number
        return converted

    def update(
        self,
        sample: Mapping[str, Any],
        timestamp: datetime | str | None = None,
    ) -> list[dict[str, Any]]:
        """Consume one observation and return any newly raised drift events."""
        validated = self._validate_sample(sample)
        stamp = self._timestamp(timestamp)
        sample_index = self._samples_seen

        if len(self._reference) < self.reference_window_size:
            self._reference.append(validated)
            self._samples_seen += 1
            self._last_timestamp = stamp
            return []

        self._current.append(validated)
        self._current_samples_seen += 1
        self._samples_seen += 1
        self._last_timestamp = stamp

        if len(self._current) < self.current_window_size:
            return []
        if (self._current_samples_seen - self.current_window_size) % self.check_interval:
            return []

        return self._evaluate(sample_index, stamp)

    def _evaluate(self, sample_index: int, timestamp: str) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        for feature in self._features or ():
            reference = np.fromiter(
                (row[feature] for row in self._reference), dtype=float,
                count=self.reference_window_size,
            )
            current = np.fromiter(
                (row[feature] for row in self._current), dtype=float,
                count=self.current_window_size,
            )
            psi = calculate_psi(reference, current)
            ks_statistic, ks_p_value, ks_drift = ks_test(
                reference, current, alpha=self.ks_alpha
            )
            wasserstein = wasserstein_test(reference, current)
            drift_detected = psi >= self.psi_threshold or ks_drift
            was_drifting = self._drift_state.get(feature, False)
            self._drift_state[feature] = drift_detected

            if drift_detected and not was_drifting:
                event = {
                    "event_id": self._next_event_id,
                    "timestamp": timestamp,
                    "sample_index": sample_index,
                    "feature": feature,
                    "decision": "Drift",
                    "psi": float(psi),
                    "ks_statistic": float(ks_statistic),
                    "ks_p_value": float(ks_p_value),
                    "wasserstein_distance": float(wasserstein),
                    "detectors": [
                        name for name, detected in (
                            ("PSI", psi >= self.psi_threshold),
                            ("KS", ks_drift),
                        ) if detected
                    ],
                }
                self._next_event_id += 1
                self._events.append(event)
                if len(self._events) > self.max_events:
                    self._events.popleft()
                emitted.append(dict(event))
        return emitted

    @property
    def events(self) -> list[dict[str, Any]]:
        """Return a copy of the retained event history."""
        return [dict(event) for event in self._events]

    @property
    def is_ready(self) -> bool:
        """Whether both the reference and current windows are populated."""
        return (
            len(self._reference) == self.reference_window_size
            and len(self._current) == self.current_window_size
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the monitor's current state."""
        return {
            "samples_seen": self._samples_seen,
            "reference_samples": len(self._reference),
            "current_samples": len(self._current),
            "reference_window_size": self.reference_window_size,
            "current_window_size": self.current_window_size,
            "is_ready": self.is_ready,
            "features": list(self._features or ()),
            "drifting_features": sorted(
                feature for feature, drifting in self._drift_state.items() if drifting
            ),
            "last_timestamp": self._last_timestamp,
            "events": self.events,
        }

    def reset(self) -> None:
        """Clear all observations, alert state, and retained events."""
        self._reference: list[dict[str, float]] = []
        self._current: deque[dict[str, float]] = deque(
            maxlen=self.current_window_size
        )
        self._features: tuple[str, ...] | None = self._configured_features
        self._samples_seen = 0
        self._current_samples_seen = 0
        self._drift_state: dict[str, bool] = {}
        self._events: deque[dict[str, Any]] = deque()
        self._next_event_id = 1
        self._last_timestamp: str | None = None
