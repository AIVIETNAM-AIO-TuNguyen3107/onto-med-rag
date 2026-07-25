from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class LLMResponseCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_responses (
                    cache_key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def get(self, cache_key: str) -> tuple[Any, dict[str, Any]] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT response_json, metadata_json
                FROM llm_responses
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0]), json.loads(row[1])

    def put(
        self,
        cache_key: str,
        response: Any,
        metadata: dict[str, Any],
    ) -> None:
        response_json = json.dumps(
            response,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        metadata_json = json.dumps(
            metadata,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO llm_responses(cache_key, response_json, metadata_json)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_json = excluded.response_json,
                    metadata_json = excluded.metadata_json
                """,
                (cache_key, response_json, metadata_json),
            )
