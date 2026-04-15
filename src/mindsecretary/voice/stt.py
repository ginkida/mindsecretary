from __future__ import annotations

import asyncio
import logging
from io import BytesIO

from groq import AsyncGroq

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds


class GroqSTT:
    def __init__(self, api_key: str, model: str = "whisper-large-v3",
                 language: str = "ru"):
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.language = language

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes (ogg/opus from Telegram) to text.

        Retries up to 3 times with exponential backoff on transient errors.
        """
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            audio_file = BytesIO(audio_bytes)
            audio_file.name = "voice.ogg"
            try:
                response = await self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=self.language,
                    response_format="text",
                )
                if isinstance(response, str):
                    return response.strip()
                return str(response).strip()
            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Groq STT attempt %d/%d failed: %s, retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, e, delay,
                    )
                    await asyncio.sleep(delay)

        logger.error("Groq STT failed after %d attempts: %s", _MAX_RETRIES, last_error)
        raise last_error  # type: ignore[misc]
