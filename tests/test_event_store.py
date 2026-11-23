import csv
import json

import pytest

from driftwatch.event_store import DriftEventStore


def _event(event_id, *, feature="temperature", severity="high", timestamp=None):
    return {
        "event_id": event_id,
        "timestamp": timestamp or f"2026-11-23T10:00:{event_id:02d}+00:00",
        "sample_index": event_id * 10,
        "feature": feature,
        "decision": "Drift",
        "severity": severity,
        "detectors": ["PSI", "KS"],
        "psi": 0.42,
        "ks_statistic": 0.31,
        "ks_p_value": 0.001,
        "wasserstein_distance": 0.8,
    }


def test_append_and_query_events(tmp_path):
    store = DriftEventStore(tmp_path / "history" / "events.db")

    first_id = store.append(_event(1))
    second_id = store.append(_event(2, feature="pressure"))

    assert first_id == 1
    assert second_id == 2
    assert store.count() == 2
    results = store.query(feature="temperature")
    assert len(results) == 1
    assert results[0]["event_id"] == 1
    assert results[0]["storage_id"] == first_id


def test_history_survives_store_recreation(tmp_path):
    database = tmp_path / "events.db"
    DriftEventStore(database).append(_event(1))

    reopened = DriftEventStore(database)
    assert reopened.count() == 1
    assert reopened.query()[0]["feature"] == "temperature"


def test_query_filters_and_pagination(tmp_path):
    store = DriftEventStore(tmp_path / "events.db")
    store.append_many([
        _event(1, feature="temperature", severity="high"),
        _event(2, feature="pressure", severity="low"),
        _event(3, feature="temperature", severity="high"),
    ])

    filtered = store.query(
        feature="temperature",
        decision="Drift",
        severity="high",
        start_time="2026-11-23T10:00:01+00:00",
        end_time="2026-11-23T10:00:03+00:00",
    )
    assert [event["event_id"] for event in filtered] == [3, 1]
    assert [event["event_id"] for event in store.query(limit=1, offset=1)] == [2]


def test_batch_append_is_atomic_when_event_is_invalid(tmp_path):
    store = DriftEventStore(tmp_path / "events.db")
    with pytest.raises(ValueError, match="missing required fields"):
        store.append_many([
            _event(1),
            {"feature": "temperature", "decision": "Drift"},
        ])
    assert store.count() == 0


def test_rejects_non_json_and_non_finite_event_values(tmp_path):
    store = DriftEventStore(tmp_path / "events.db")
    invalid = _event(1)
    invalid["psi"] = float("nan")
    with pytest.raises(ValueError, match="JSON-serializable"):
        store.append(invalid)
    assert store.count() == 0


def test_export_json_and_csv(tmp_path):
    store = DriftEventStore(tmp_path / "events.db")
    store.append(_event(1))

    payload = json.loads(store.export_json(feature="temperature"))
    assert payload[0]["event_id"] == 1
    assert payload[0]["storage_id"] == 1

    rows = list(csv.DictReader(store.export_csv().splitlines()))
    assert len(rows) == 1
    assert rows[0]["feature"] == "temperature"
    assert rows[0]["psi"] == "0.42"
    assert rows[0]["detectors"] == '["PSI","KS"]'


def test_empty_exports_and_invalid_pagination(tmp_path):
    store = DriftEventStore(tmp_path / "events.db")
    assert json.loads(store.export_json()) == []
    assert len(store.export_csv().splitlines()) == 1
    with pytest.raises(ValueError, match="limit"):
        store.query(limit=0)
    with pytest.raises(ValueError, match="offset"):
        store.query(offset=-1)
