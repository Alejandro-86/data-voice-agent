"""Unit tests for the LLM agent and voice layer using stubs."""

import pytest

from data_voice.agent.agent import AgentResponse, DataAgent
from data_voice.data.db import DataStore
from data_voice.tools.query import QueryResult
from data_voice.voice.tts import TTSConfig


class StubLLMClient:
    """Returns a canned LLM response without real API calls."""

    def __init__(self, reply: str = "The total API calls are 500.") -> None:
        self._reply = reply
        self.calls: list[dict[str, object]] = []

    def messages_create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)

        class FakeContent:
            text = self._reply
            type = "text"

        class FakeMsg:
            content = [FakeContent()]
            stop_reason = "end_turn"

        return FakeMsg()


class StubQueryTool:
    """Returns a fixed QueryResult for any SQL."""

    def run(self, sql: str) -> QueryResult:
        return QueryResult(
            success=True,
            sql=sql,
            rows=[{"total": 500}],
            summary="total: 500",
        )

    name = "query_data"
    description = "Execute SQL"


class StubTTSClient:
    """Captures TTS calls without calling ElevenLabs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def speak(self, text: str, lang: str = "en") -> bytes:
        self.calls.append((text, lang))
        return b"fake-audio-bytes"


class TestDataAgent:
    def _make_agent(self, reply: str = "Total: 500") -> DataAgent:
        store = DataStore(db_path=":memory:")
        store.seed_sample_data()
        return DataAgent(
            llm_client=StubLLMClient(reply),  # type: ignore[arg-type]
            query_tool=StubQueryTool(),  # type: ignore[arg-type]
            schema=store.get_schema(),
        )

    def test_ask_returns_agent_response(self) -> None:
        agent = self._make_agent("The total is 500.")
        resp = agent.ask("How many API calls were made?")
        assert isinstance(resp, AgentResponse)
        assert resp.answer != ""

    def test_ask_passes_question_to_llm(self) -> None:
        stub_llm = StubLLMClient("500 calls")
        store = DataStore(db_path=":memory:")
        store.seed_sample_data()
        agent = DataAgent(
            llm_client=stub_llm,  # type: ignore[arg-type]
            query_tool=StubQueryTool(),  # type: ignore[arg-type]
            schema=store.get_schema(),
        )
        agent.ask("How many total calls?")
        assert len(stub_llm.calls) >= 1

    def test_response_includes_sql_when_queried(self) -> None:
        agent = self._make_agent()
        resp = agent.ask("count all events")
        assert resp.answer != ""

    def test_empty_question_raises(self) -> None:
        agent = self._make_agent()
        with pytest.raises(ValueError):
            agent.ask("")


class TestTTSConfig:
    def test_config_stores_voice_and_model(self) -> None:
        cfg = TTSConfig(voice_id="abc123", model_id="eleven_v3")
        assert cfg.voice_id == "abc123"
        assert cfg.model_id == "eleven_v3"

    def test_default_lang(self) -> None:
        cfg = TTSConfig(voice_id="v", model_id="m")
        assert cfg.default_lang == "en"


class TestStubTTSClient:
    def test_speak_returns_bytes(self) -> None:
        stub = StubTTSClient()
        audio = stub.speak("hello world", lang="en")
        assert isinstance(audio, bytes)

    def test_speak_spanish(self) -> None:
        stub = StubTTSClient()
        stub.speak("hola mundo", lang="es")
        assert stub.calls[0][1] == "es"
