"""Unit tests for DuckDB data layer and agent tool functions."""

import pytest

from data_voice.data.db import DataStore
from data_voice.tools.query import QueryTool


class TestDataStore:
    def test_in_memory_store_initialises(self) -> None:
        store = DataStore(db_path=":memory:")
        assert store is not None

    def test_execute_simple_select(self) -> None:
        store = DataStore(db_path=":memory:")
        store.seed_sample_data()
        result = store.execute("SELECT COUNT(*) AS n FROM events")
        assert len(result) == 1
        assert result[0]["n"] > 0

    def test_execute_returns_list_of_dicts(self) -> None:
        store = DataStore(db_path=":memory:")
        store.seed_sample_data()
        result = store.execute("SELECT * FROM events LIMIT 3")
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_execute_raises_on_invalid_sql(self) -> None:
        store = DataStore(db_path=":memory:")
        with pytest.raises((RuntimeError, Exception)):  # duckdb raises RuntimeError
            store.execute("NOT VALID SQL $$$$")

    def test_schema_returns_table_info(self) -> None:
        store = DataStore(db_path=":memory:")
        store.seed_sample_data()
        schema = store.get_schema()
        assert "events" in schema
        assert len(schema["events"]) > 0

    def test_prevents_write_operations(self) -> None:
        store = DataStore(db_path=":memory:", read_only=True)
        store.seed_sample_data()
        with pytest.raises((PermissionError, RuntimeError)):
            store.execute("DROP TABLE events")


class TestQueryTool:
    def _make_store(self) -> DataStore:
        store = DataStore(db_path=":memory:")
        store.seed_sample_data()
        return store

    def test_query_returns_result(self) -> None:
        tool = QueryTool(store=self._make_store())
        result = tool.run(sql="SELECT COUNT(*) AS total FROM events")
        assert result.success is True
        assert result.rows is not None
        assert len(result.rows) == 1

    def test_query_failed_result_on_bad_sql(self) -> None:
        tool = QueryTool(store=self._make_store())
        result = tool.run(sql="SELECT * FROM nonexistent_table_xyz")
        assert result.success is False
        assert result.error is not None

    def test_query_result_has_summary(self) -> None:
        tool = QueryTool(store=self._make_store())
        result = tool.run(sql="SELECT model_id, COUNT(*) AS calls FROM events GROUP BY model_id")
        assert result.success is True
        assert result.summary != ""

    def test_query_tool_description(self) -> None:
        tool = QueryTool(store=self._make_store())
        assert "SQL" in tool.description or "sql" in tool.description.lower()
