# data-voice-agent

An agentic data Q&A system — ask natural language questions about datasets,
get answers spoken back in English or Spanish via the ElevenLabs API.

## Architecture

```
User question (text)
        │
        ▼
┌───────────────┐
│  LLM Agent    │  Anthropic Claude — reasons about the question
│  (Claude)     │  selects and calls the right tool
└──────┬────────┘
       │ tool call
       ▼
┌───────────────┐
│  DuckDB Tool  │  Translates question to SQL, runs against in-process DB
│               │  Returns structured result + summary text
└──────┬────────┘
       │ answer text
       ▼
┌───────────────┐
│  ElevenLabs   │  eleven_v3 model, language_code="en" or "es"
│  TTS          │  Returns audio/mpeg stream
└──────┬────────┘
       │
       ▼
    MP3 audio
```

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env   # add ANTHROPIC_API_KEY + ELEVENLABS_API_KEY
uvicorn data_voice.api:app --reload
# → http://localhost:8000/docs
```

## API

```
POST /ask         → { "question": "...", "lang": "en" }   → JSON answer
POST /ask/speak   → { "question": "...", "lang": "es" }   → audio/mpeg
GET  /health      → { "status": "ok" }
```

## Dataset

Ships with a sample SaaS metrics dataset (DuckDB in-process).
Load your own parquet/CSV via the `DATA_PATH` env var.
