import numpy as np
import pytest

from driftwatch.streaming import StreamingDriftMonitor


def _feed(monitor, values, start_index=0, timestamp_prefix="2026-11-23T10:00"):
    events = []
    for offset, value in enumerate(values):
        events.extend(
            monitor.update(
                {"signal": float(value)},
                timestamp=f"{timestamp_prefix}:{(start_index + offset) % 60:02d}+00:00",
            )
        )
    return events


def test_monitor_waits_for_reference_and_current_windows():
    monitor = StreamingDriftMonitor(
        reference_window_size=20,
        current_window_size=10,
        check_interval=5,
        psi_threshold=0.2,
    )
    assert monitor.update({"signal": 1.0}) == []
    _feed(monitor, np.zeros(19))
    assert monitor.snapshot()["reference_samples"] == 20
    assert monitor.snapshot()["current_samples"] == 0
    _feed(monitor, np.zeros(9))
    assert monitor.is_ready is False
    _feed(monitor, np.zeros(1))
    assert monitor.is_ready is True
    assert monitor.events == []


def test_monitor_emits_event_for_shifted_stream():
    rng = np.random.default_rng(44)
    reference = rng.normal(0.0, 1.0, 300)
    current = rng.normal(3.0, 1.0, 100)
    monitor = StreamingDriftMonitor(
        reference_window_size=300,
        current_window_size=100,
        check_interval=25,
        psi_threshold=0.2,
    )

    assert _feed(monitor, reference) == []
    events = _feed(monitor, current, start_index=300)
    assert len(events) == 1
    event = events[0]
    assert event["feature"] == "signal"
    assert event["decision"] == "Drift"
    assert event["psi"] >= 0.2
    assert "PSI" in event["detectors"]
    assert event["sample_index"] == 399


def test_persistent_drift_does_not_emit_duplicate_events():
    rng = np.random.default_rng(45)
    monitor = StreamingDriftMonitor(
        reference_window_size=250,
        current_window_size=80,
        check_interval=20,
        psi_threshold=0.2,
    )
    _feed(monitor, rng.normal(0, 1, 250))
    events = _feed(monitor, rng.normal(3, 1, 160), start_index=250)
    assert len(events) == 1
    assert len(monitor.events) == 1


def test_monitor_rejects_schema_and_non_finite_values():
    monitor = StreamingDriftMonitor(reference_window_size=5, current_window_size=3)
    monitor.update({"a": 1.0, "b": 2.0})
    with pytest.raises(ValueError, match="schema mismatch"):
        monitor.update({"a": 1.0})
    with pytest.raises(ValueError, match="finite"):
        monitor.update({"a": np.nan, "b": 2.0})


def test_monitor_reset_clears_state_and_events():
    rng = np.random.default_rng(46)
    monitor = StreamingDriftMonitor(
        reference_window_size=100,
        current_window_size=40,
        check_interval=10,
        psi_threshold=0.2,
    )
    _feed(monitor, rng.normal(0, 1, 100))
    _feed(monitor, rng.normal(4, 1, 40), start_index=100)
    assert monitor.events

    monitor.reset()
    state = monitor.snapshot()
    assert state["samples_seen"] == 0
    assert state["reference_samples"] == 0
    assert state["current_samples"] == 0
    assert state["events"] == []
    assert state["is_ready"] is False


def test_monitor_validates_configuration():
    with pytest.raises(ValueError):
        StreamingDriftMonitor(reference_window_size=1)
    with pytest.raises(ValueError):
        StreamingDriftMonitor(current_window_size=1)
    with pytest.raises(ValueError):
        StreamingDriftMonitor(check_interval=0)
