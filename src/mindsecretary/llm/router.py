from __future__ import annotations

import logging

from .client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class ModelRouter:
    def __init__(self, client: LLMClient):
        self.client = client

    async def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        return await self.client.chat(system, messages, tools, max_tokens)
