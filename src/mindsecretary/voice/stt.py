from __future__ import annotations

import logging
from io import BytesIO

from groq import AsyncGroq

logger = logging.getLogger(__name__)


class GroqSTT:
    def __init__(self, api_key: str, model: str = "whisper-large-v3",
                 language: str = "ru"):
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.language = language

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes (ogg/opus from Telegram) to text."""
        # Groq SDK accepts file-like objects with a .name attribute
        audio_file = BytesIO(audio_bytes)
        audio_file.name = "voice.ogg"

        try:
            response = await self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=self.language,
                response_format="text",
            )
        except Exception as e:
            logger.error("Groq STT failed: %s", e)
            raise

        # response_format="text" returns a plain string
        if isinstance(response, str):
            return response.strip()
        # Some SDK versions may wrap it
        return str(response).strip()
