"""DuckDB query tool — executes SQL and returns a structured result."""

import logging
from dataclasses import dataclass
from typing import Any

from data_voice.data.db import DataStore

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """The result of running a SQL query through the tool.

    Attributes:
        success: True if the query executed without error.
        sql: The SQL that was executed.
        rows: Result rows as list of dicts, or None on failure.
        summary: A human-readable one-line summary of the result.
        error: Error message if success is False.
    """

    success: bool
    sql: str
    rows: list[dict[str, Any]] | None = None
    summary: str = ""
    error: str | None = None


class QueryTool:
    """Agent tool that executes SQL against the in-process DuckDB store.

    The LLM generates SQL; this tool executes it safely and returns
    both the raw rows and a formatted text summary for the LLM to use
    in its final answer.

    Args:
        store: DataStore instance.
    """

    name = "query_data"
    description = (
        "Execute a SQL query against the SaaS events dataset. "
        "Use standard SQL. Available tables and columns are provided in the system prompt. "
        "Returns rows as JSON and a text summary."
    )

    def __init__(self, store: DataStore) -> None:
        self._store = store

    def run(self, sql: str) -> QueryResult:
        """Execute a SQL query and return structured results.

        Args:
            sql: SQL SELECT statement to execute.

        Returns:
            QueryResult with rows and a text summary.
        """
        try:
            rows = self._store.execute(sql)
            summary = self._summarise(rows)
            logger.info("query ok, %d rows", len(rows))
            return QueryResult(success=True, sql=sql, rows=rows, summary=summary)
        except Exception as exc:
            logger.warning("query failed: %s", exc)
            return QueryResult(success=False, sql=sql, error=str(exc))

    @staticmethod
    def _summarise(rows: list[dict[str, Any]]) -> str:
        """Generate a brief human-readable summary of the query result."""
        if not rows:
            return "The query returned no rows."
        if len(rows) == 1 and len(rows[0]) == 1:
            key, val = next(iter(rows[0].items()))
            return f"{key}: {val}"
        return f"{len(rows)} rows returned."
