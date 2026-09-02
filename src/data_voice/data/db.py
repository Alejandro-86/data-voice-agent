"""DuckDB in-process data store for the voice agent.

DuckDB runs entirely in-process — no server required.  The agent
queries it like a SQL database; results are returned as list-of-dicts
for easy serialisation and LLM formatting.
"""

import logging
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


class DataStore:
    """In-process DuckDB data store.

    Args:
        db_path: DuckDB path — use ':memory:' for in-memory (tests/demos).
        read_only: If True, reject any DML/DDL statements after seed.
    """

    def __init__(self, db_path: str = ":memory:", read_only: bool = False) -> None:
        self._conn = duckdb.connect(db_path)
        self._read_only = read_only
        self._seeded = False

    def seed_sample_data(self) -> None:
        """Create and populate a sample SaaS events table for demo purposes."""
        if self._seeded:
            return

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id     VARCHAR PRIMARY KEY,
                user_id      VARCHAR NOT NULL,
                model_id     VARCHAR NOT NULL,
                event_type   VARCHAR NOT NULL,
                characters   INTEGER NOT NULL,
                latency_ms   DOUBLE  NOT NULL,
                created_at   TIMESTAMP NOT NULL
            )
        """)

        self._conn.execute("""
            INSERT INTO events SELECT
                gen_random_uuid()::VARCHAR,
                'user-' || (i % 50)::VARCHAR,
                CASE (i % 3)
                    WHEN 0 THEN 'eleven_v3'
                    WHEN 1 THEN 'eleven_flash_v2_5'
                    ELSE 'eleven_multilingual_v2'
                END,
                CASE (i % 4)
                    WHEN 0 THEN 'tts_convert'
                    WHEN 1 THEN 'tts_stream'
                    WHEN 2 THEN 'voice_clone'
                    ELSE 'dubbing'
                END,
                (random() * 5000 + 100)::INTEGER,
                (random() * 500 + 50)::DOUBLE,
                TIMESTAMP '2024-01-01' + INTERVAL (i) HOUR
            FROM range(500) t(i)
        """)

        self._seeded = True
        logger.info("seeded 500 sample events")

    def execute(self, sql: str) -> list[dict[str, Any]]:
        """Execute a SQL query and return results as a list of dicts.

        Args:
            sql: SQL query string.

        Returns:
            List of row dicts.

        Raises:
            Exception: On SQL errors or write attempts when read_only=True.
        """
        if self._read_only:
            sql_upper = sql.strip().upper()
            write_keywords = ("DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "TRUNCATE")
            if any(sql_upper.startswith(k) for k in write_keywords):
                raise PermissionError("write operations are not allowed on a read-only store")

        rel = self._conn.execute(sql)
        columns = [desc[0] for desc in rel.description]
        rows = rel.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def get_schema(self) -> dict[str, list[dict[str, str]]]:
        """Return the schema of all tables as a dict.

        Returns:
            Dict mapping table name to list of column dicts (name, type).
        """
        tables = self.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        )
        schema: dict[str, list[dict[str, str]]] = {}
        for t in tables:
            name = t["table_name"]
            cols = self.execute(
                f"SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE table_name='{name}'"
            )
            schema[name] = [{"name": c["column_name"], "type": c["data_type"]} for c in cols]
        return schema

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()
