"""Stateful streaming drift monitoring with statistical and online detectors."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from driftwatch.detectors import calculate_psi, ks_test, wasserstein_test

try:
    from river.drift import ADWIN
except ImportError:  # pragma: no cover - exercised in environments without River
    ADWIN = None


class StreamingDriftMonitor:
    """Incremental feature-wise drift monitor.

    The initial reference window establishes a fixed baseline. Subsequent
    observations are checked online by ADWIN and a two-sided Page-Hinkley
    detector, and periodically by PSI, KS, and Wasserstein comparisons.
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
        enable_adwin: bool = False,
        adwin_delta: float = 0.002,
        enable_page_hinkley: bool = False,
        page_hinkley_delta: float = 0.005,
        page_hinkley_threshold: float = 5.0,
    ) -> None:
        if reference_window_size < 2 or current_window_size < 2:
            raise ValueError("reference_window_size and current_window_size must be at least 2.")
        if check_interval < 1 or max_events < 1:
            raise ValueError("check_interval and max_events must be at least 1.")
        if not np.isfinite(psi_threshold) or psi_threshold < 0:
            raise ValueError("psi_threshold must be finite and non-negative.")
        if not 0 < ks_alpha < 1 or not 0 < adwin_delta < 1:
            raise ValueError("ks_alpha and adwin_delta must be between 0 and 1.")
        if not np.isfinite(page_hinkley_delta) or page_hinkley_delta < 0:
            raise ValueError("page_hinkley_delta must be finite and non-negative.")
        if not np.isfinite(page_hinkley_threshold) or page_hinkley_threshold <= 0:
            raise ValueError("page_hinkley_threshold must be finite and positive.")

        configured_features = tuple(features) if features is not None else None
        if configured_features is not None:
            if (
                not configured_features
                or len(set(configured_features)) != len(configured_features)
                or any(not isinstance(name, str) or not name for name in configured_features)
            ):
                raise ValueError("features must contain unique, non-empty string names.")

        self.reference_window_size = int(reference_window_size)
        self.current_window_size = int(current_window_size)
        self.check_interval = int(check_interval)
        self.psi_threshold = float(psi_threshold)
        self.ks_alpha = float(ks_alpha)
        self.max_events = int(max_events)
        self._configured_features = configured_features
        self.enable_adwin = bool(enable_adwin)
        self.adwin_delta = float(adwin_delta)
        self.enable_page_hinkley = bool(enable_page_hinkley)
        self.page_hinkley_delta = float(page_hinkley_delta)
        self.page_hinkley_threshold = float(page_hinkley_threshold)
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

    def _initialize_online_detectors(self) -> None:
        self._adwin: dict[str, Any] = {}
        self._ph_state: dict[str, dict[str, float]] = {}
        for feature in self._features or ():
            baseline = np.asarray([row[feature] for row in self._reference], dtype=float)
            mean = float(np.mean(baseline))
            scale = float(np.std(baseline))
            if not np.isfinite(scale) or scale < 1e-12:
                scale = 1.0
            self._ph_state[feature] = {
                "mean": mean,
                "scale": scale,
                "positive": 0.0,
                "negative": 0.0,
                "positive_min": 0.0,
                "negative_min": 0.0,
            }
            if self.enable_adwin and ADWIN is not None:
                detector = ADWIN(delta=self.adwin_delta)
                for value in baseline:
                    detector.update(float(value))
                self._adwin[feature] = detector

    def _online_signals(self, feature: str, value: float) -> list[str]:
        signals: list[str] = []
        if self.enable_adwin and feature in self._adwin:
            detector = self._adwin[feature]
            detector.update(float(value))
            if detector.drift_detected:
                signals.append("ADWIN")

        if self.enable_page_hinkley:
            state = self._ph_state[feature]
            z_value = (value - state["mean"]) / state["scale"]
            positive_increment = z_value - self.page_hinkley_delta
            negative_increment = -z_value - self.page_hinkley_delta
            state["positive"] += positive_increment
            state["negative"] += negative_increment
            state["positive_min"] = min(state["positive_min"], state["positive"])
            state["negative_min"] = min(state["negative_min"], state["negative"])
            if (
                state["positive"] - state["positive_min"] > self.page_hinkley_threshold
                or state["negative"] - state["negative_min"] > self.page_hinkley_threshold
            ):
                signals.append("Page-Hinkley")
        return signals

    def _record_event(
        self,
        feature: str,
        sample_index: int,
        timestamp: str,
        detectors: list[str],
        metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "event_id": self._next_event_id,
            "timestamp": timestamp,
            "sample_index": sample_index,
            "feature": feature,
            "decision": "Drift",
            "detectors": list(dict.fromkeys(detectors)),
        }
        if metrics:
            event.update(metrics)
        self._next_event_id += 1
        self._events.append(event)
        if len(self._events) > self.max_events:
            self._events.popleft()
        return event

    def update(
        self,
        sample: Mapping[str, Any],
        timestamp: datetime | str | None = None,
    ) -> list[dict[str, Any]]:
        """Consume one observation and return newly raised drift events."""
        validated = self._validate_sample(sample)
        stamp = self._timestamp(timestamp)
        sample_index = self._samples_seen

        if len(self._reference) < self.reference_window_size:
            self._reference.append(validated)
            self._samples_seen += 1
            self._last_timestamp = stamp
            if len(self._reference) == self.reference_window_size:
                self._initialize_online_detectors()
            return []

        self._current.append(validated)
        self._current_samples_seen += 1
        self._samples_seen += 1
        self._last_timestamp = stamp

        emitted: list[dict[str, Any]] = []
        for feature, value in validated.items():
            signals = self._online_signals(feature, value)
            if signals and not self._drift_state.get(feature, False):
                emitted.append(self._record_event(feature, sample_index, stamp, signals))
                self._drift_state[feature] = True

        if len(self._current) == self.current_window_size and (
            self._current_samples_seen - self.current_window_size
        ) % self.check_interval == 0:
            emitted.extend(self._evaluate(sample_index, stamp))

        # Return detached snapshots only after all same-update enrichment is
        # complete, so an online alert can include later window metrics.
        return [dict(event) for event in emitted]

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
            psi_drift = psi >= self.psi_threshold
            drift_detected = psi_drift or ks_drift
            was_drifting = self._drift_state.get(feature, False)
            self._drift_state[feature] = drift_detected

            if drift_detected:
                detectors = [
                    name for name, detected in (("PSI", psi_drift), ("KS", ks_drift))
                    if detected
                ]
                metrics = {
                    "psi": float(psi),
                    "ks_statistic": float(ks_statistic),
                    "ks_p_value": float(ks_p_value),
                    "wasserstein_distance": float(wasserstein),
                }
                if not was_drifting:
                    emitted.append(self._record_event(
                        feature, sample_index, timestamp, detectors, metrics
                    ))
                else:
                    # An online detector may raise the alert earlier in this
                    # same update. Enrich that event with window-based metrics
                    # rather than emitting a duplicate alert.
                    for event in reversed(self._events):
                        if event["feature"] == feature and event["decision"] == "Drift":
                            event.update(metrics)
                            event["metrics_timestamp"] = timestamp
                            event["metrics_sample_index"] = sample_index
                            event["detectors"] = list(dict.fromkeys(
                                event["detectors"] + detectors
                            ))
                            break
        return emitted

    @property
    def events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events]

    @property
    def is_ready(self) -> bool:
        return (
            len(self._reference) == self.reference_window_size
            and len(self._current) == self.current_window_size
        )

    def snapshot(self) -> dict[str, Any]:
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
            "online_detectors": {
                "adwin": self.enable_adwin and ADWIN is not None,
                "page_hinkley": self.enable_page_hinkley,
            },
            "last_timestamp": self._last_timestamp,
            "events": self.events,
        }

    def reset(self) -> None:
        self._reference: list[dict[str, float]] = []
        self._current: deque[dict[str, float]] = deque(maxlen=self.current_window_size)
        self._features: tuple[str, ...] | None = self._configured_features
        self._samples_seen = 0
        self._current_samples_seen = 0
        self._drift_state: dict[str, bool] = {}
        self._adwin: dict[str, Any] = {}
        self._ph_state: dict[str, dict[str, float]] = {}
        self._events: deque[dict[str, Any]] = deque()
        self._next_event_id = 1
        self._last_timestamp: str | None = None
