"""LLM agent that answers natural language questions about data.

The agent receives a question, injects the database schema into the system
prompt, asks the LLM to generate SQL, runs it via the QueryTool, and
formats the result as a natural language answer.
"""

import logging
from dataclasses import dataclass
from typing import Any

from data_voice.tools.query import QueryTool

logger = logging.getLogger(__name__)

_SYSTEM_TEMPLATE = """You are a helpful data analyst assistant.
You answer questions about SaaS usage data by generating SQL queries.

Database schema:
{schema}

Rules:
- Generate a single SELECT SQL query to answer the question
- Return ONLY the SQL query, nothing else, no markdown
- Use only the tables and columns listed in the schema above
- If the question cannot be answered with SQL, say so directly
"""


@dataclass
class AgentResponse:
    """The agent's answer to a user question.

    Attributes:
        question: The original question.
        answer: Natural language answer.
        sql: The SQL that was executed, if any.
        rows_returned: Number of result rows.
    """

    question: str
    answer: str
    sql: str | None = None
    rows_returned: int = 0


class DataAgent:
    """Agentic data Q&A — generates SQL from a question, runs it, formats the answer.

    Args:
        llm_client: Anthropic (or stub) client with a `messages_create` method.
        query_tool: QueryTool wrapping the DuckDB store.
        schema: Database schema dict from `DataStore.get_schema()`.
        model: LLM model identifier.
    """

    def __init__(
        self,
        llm_client: Any,
        query_tool: QueryTool,
        schema: dict[str, Any],
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self._llm = llm_client
        self._tool = query_tool
        self._schema = schema
        self._model = model

    def ask(self, question: str) -> AgentResponse:
        """Answer a natural language question about the data.

        Args:
            question: The user's question.

        Returns:
            AgentResponse with the answer and the SQL used.

        Raises:
            ValueError: If the question is empty.
        """
        if not question.strip():
            raise ValueError("question cannot be empty")

        schema_text = self._format_schema()
        system = _SYSTEM_TEMPLATE.format(schema=schema_text)

        msg = self._llm.messages_create(
            model=self._model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": question}],
        )

        sql_or_text = msg.content[0].text.strip()

        # If the LLM returned SQL, run it
        if sql_or_text.upper().startswith("SELECT"):
            result = self._tool.run(sql=sql_or_text)
            if result.success:
                answer = self._format_answer(question, result.summary, result.rows or [])
                return AgentResponse(
                    question=question,
                    answer=answer,
                    sql=sql_or_text,
                    rows_returned=len(result.rows or []),
                )
            else:
                return AgentResponse(
                    question=question,
                    answer=f"I ran the query but encountered an error: {result.error}",
                    sql=sql_or_text,
                )

        # LLM responded with a direct text answer (e.g. for non-data questions)
        return AgentResponse(question=question, answer=sql_or_text)

    def _format_schema(self) -> str:
        """Format the schema dict as readable text for the system prompt."""
        lines: list[str] = []
        for table, cols in self._schema.items():
            col_str = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
            lines.append(f"  {table}: {col_str}")
        return "\n".join(lines)

    @staticmethod
    def _format_answer(
        question: str,
        summary: str,
        rows: list[dict[str, Any]],
    ) -> str:
        """Format the query result as a natural language answer."""
        if not rows:
            return f"The query returned no results for: {question}"
        if len(rows) == 1 and len(rows[0]) == 1:
            return summary
        return f"{summary} ({len(rows)} rows)"
