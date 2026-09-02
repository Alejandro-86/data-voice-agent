"""ElevenLabs TTS client — converts text to audio bytes.

Supports multilingual output via the eleven_v3 model's language_code parameter.
The client is injectable so tests can stub it without real API calls.
"""

import io
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """Configuration for ElevenLabs TTS.

    Args:
        voice_id: ElevenLabs voice identifier.
        model_id: Model to use (eleven_v3 supports 70+ languages).
        default_lang: ISO 639-1 language code used when no lang is specified.
        stability: Voice stability (0=expressive, 1=consistent).
        similarity_boost: Voice similarity boost.
    """

    voice_id: str
    model_id: str = "eleven_v3"
    default_lang: str = "en"
    stability: float = 0.7
    similarity_boost: float = 0.75


class TTSClient:
    """ElevenLabs text-to-speech client.

    Args:
        api_key: ElevenLabs API key.
        config: TTS configuration.
    """

    def __init__(self, api_key: str, config: TTSConfig) -> None:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings

        self._client = ElevenLabs(api_key=api_key)
        self._config = config
        self._voice_settings_cls = VoiceSettings

    def speak(self, text: str, lang: str | None = None) -> bytes:
        """Convert text to audio bytes.

        Args:
            text: Text to synthesise.
            lang: ISO 639-1 language code (e.g. 'en', 'es').
                  Falls back to config.default_lang if not provided.

        Returns:
            MP3 audio bytes.
        """
        language = lang or self._config.default_lang

        audio_stream = self._client.text_to_speech.stream(
            text=text,
            voice_id=self._config.voice_id,
            model_id=self._config.model_id,
            language_code=language,
            voice_settings=self._voice_settings_cls(
                stability=self._config.stability,
                similarity_boost=self._config.similarity_boost,
            ),
        )

        buf = io.BytesIO()
        for chunk in audio_stream:
            buf.write(chunk)

        logger.info("tts: synthesised %d chars in %s", len(text), language)
        return buf.getvalue()
