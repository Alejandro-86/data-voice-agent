"""FastAPI application — /ask (text) and /ask/speak (audio) endpoints."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from data_voice.agent.agent import DataAgent
from data_voice.config import settings
from data_voice.data.db import DataStore
from data_voice.tools.query import QueryTool
from data_voice.voice.tts import TTSClient, TTSConfig

_agent: DataAgent | None = None
_tts: TTSClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise agent and TTS client on startup."""
    global _agent, _tts

    store = DataStore(db_path=settings.data_path)
    store.seed_sample_data()

    llm_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    query_tool = QueryTool(store=store)
    _agent = DataAgent(
        llm_client=llm_client,
        query_tool=query_tool,
        schema=store.get_schema(),
        model=settings.llm_model,
    )

    if settings.elevenlabs_api_key:
        _tts = TTSClient(
            api_key=settings.elevenlabs_api_key,
            config=TTSConfig(
                voice_id=settings.elevenlabs_voice_id,
                model_id=settings.elevenlabs_model_id,
            ),
        )
    yield


app = FastAPI(
    title="data-voice-agent",
    description="Ask natural language questions about data; get answers spoken back.",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    """Request body for /ask and /ask/speak."""

    question: str
    lang: str = "en"


class AskResponse(BaseModel):
    """Response body for /ask."""

    question: str
    answer: str
    sql: str | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    """Service health check."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Answer a natural language question as text."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="agent not initialised")
    try:
        response = _agent.ask(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AskResponse(
        question=response.question,
        answer=response.answer,
        sql=response.sql,
    )


@app.post("/ask/speak")
async def ask_and_speak(request: AskRequest) -> StreamingResponse:
    """Answer a natural language question as MP3 audio."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="agent not initialised")
    if _tts is None:
        raise HTTPException(
            status_code=503,
            detail="TTS not configured — set ELEVENLABS_API_KEY"
        )
    try:
        response = _agent.ask(request.question)
        audio = _tts.speak(response.answer, lang=request.lang)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    import io
    return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")
