"""SQLite-backed persistence and querying for drift events."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping


class DriftEventStore:
    """Persist drift events and provide filtered history/export operations.

    A connection is opened per operation, keeping instances safe to use from
    separate threads/processes that share the same SQLite database file.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        if str(database_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS drift_events (
                    storage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT,
                    timestamp TEXT NOT NULL,
                    feature TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    severity TEXT,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_drift_events_timestamp "
                "ON drift_events(timestamp)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_drift_events_feature "
                "ON drift_events(feature)"
            )

    @staticmethod
    def _validate_event(event: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(event, Mapping):
            raise TypeError("event must be a mapping.")
        required = ("timestamp", "feature", "decision")
        missing = [key for key in required if key not in event]
        if missing:
            raise ValueError(f"event is missing required fields: {', '.join(missing)}")
        if not all(isinstance(event[key], str) and event[key].strip() for key in required):
            raise ValueError("timestamp, feature, and decision must be non-empty strings.")
        normalized = dict(event)
        try:
            # Reject non-JSON values and NaN/Infinity before opening a transaction.
            json.dumps(normalized, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("event must contain finite, JSON-serializable values.") from exc
        return normalized

    def append(self, event: Mapping[str, Any]) -> int:
        """Persist one event and return its database storage ID."""
        return self.append_many([event])[0]

    def append_many(self, events: Iterable[Mapping[str, Any]]) -> list[int]:
        """Persist a batch atomically; invalid input writes nothing."""
        normalized = [self._validate_event(event) for event in events]
        if not normalized:
            return []
        rows = [
            (
                str(event["event_id"]) if event.get("event_id") is not None else None,
                event["timestamp"],
                event["feature"],
                event["decision"],
                str(event["severity"]) if event.get("severity") is not None else None,
                json.dumps(event, allow_nan=False, separators=(",", ":")),
            )
            for event in normalized
        ]
        with self._connect() as connection:
            storage_ids = []
            for row in rows:
                cursor = connection.execute(
                    """
                    INSERT INTO drift_events
                        (event_id, timestamp, feature, decision, severity, payload)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )
                storage_ids.append(int(cursor.lastrowid))
            return storage_ids

    def query(
        self,
        *,
        feature: str | None = None,
        decision: str | None = None,
        severity: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return newest matching events, with optional inclusive time bounds."""
        if limit < 1 or offset < 0:
            raise ValueError("limit must be >= 1 and offset must be >= 0.")
        filters: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("feature", feature),
            ("decision", decision),
            ("severity", severity),
        ):
            if value is not None:
                filters.append(f"{column} = ?")
                parameters.append(value)
        if start_time is not None:
            filters.append("timestamp >= ?")
            parameters.append(start_time)
        if end_time is not None:
            filters.append("timestamp <= ?")
            parameters.append(end_time)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        parameters.extend((limit, offset))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT storage_id, payload FROM drift_events
                {where}
                ORDER BY timestamp DESC, storage_id DESC
                LIMIT ? OFFSET ?
                """,
                parameters,
            ).fetchall()
        results = []
        for row in rows:
            event = json.loads(row["payload"])
            event["storage_id"] = row["storage_id"]
            results.append(event)
        return results

    def export_json(self, **filters: Any) -> str:
        """Serialize filtered history as a JSON array."""
        return json.dumps(self.query(**filters), indent=2, allow_nan=False)

    def export_csv(self, **filters: Any) -> str:
        """Serialize filtered history as CSV with stable, useful columns."""
        events = self.query(**filters)
        fields = (
            "storage_id",
            "event_id",
            "timestamp",
            "sample_index",
            "feature",
            "decision",
            "severity",
            "detectors",
            "psi",
            "ks_statistic",
            "ks_p_value",
            "wasserstein_distance",
        )
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            row = dict(event)
            if isinstance(row.get("detectors"), (list, dict)):
                row["detectors"] = json.dumps(row["detectors"], separators=(",", ":"))
            writer.writerow(row)
        return output.getvalue()

    def count(self, *, feature: str | None = None) -> int:
        """Return the number of stored events, optionally for one feature."""
        with self._connect() as connection:
            if feature is None:
                row = connection.execute("SELECT COUNT(*) FROM drift_events").fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) FROM drift_events WHERE feature = ?", (feature,)
                ).fetchone()
        return int(row[0])
