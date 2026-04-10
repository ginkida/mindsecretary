from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class LLMClient(ABC):
    @abstractmethod
    async def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


class AnthropicClient(LLMClient):
    """Claude Sonnet via Anthropic SDK — primary (and only) LLM client."""

    def __init__(self, api_key: str, model: str):
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    def _convert_user_content(self, content) -> str | list[dict]:
        """Convert user message content — handles text, multimodal (text+image)."""
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return str(content)

        # Multimodal: list of content blocks
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")

            if block_type == "text":
                blocks.append({"type": "text", "text": block.get("text", "")})

            elif block_type == "image_url":
                # Convert OpenAI image_url format → Anthropic image format
                url = block.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # Parse data URI: "data:image/jpeg;base64,DATA"
                    header, _, b64data = url.partition(",")
                    media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64data,
                        },
                    })

        return blocks if blocks else ""

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        """Convert OpenAI-style messages to Anthropic format."""
        converted = []
        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg["role"] == "user":
                converted.append({
                    "role": "user",
                    "content": self._convert_user_content(msg["content"]),
                })

            elif msg["role"] == "assistant":
                content_blocks: list[dict] = []
                text = msg.get("content")
                if text:
                    content_blocks.append({"type": "text", "text": text})
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", tc)
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", f"call_{fn.get('name', 'unknown')}"),
                        "name": fn.get("name", tc.get("name", "unknown")),
                        "input": args,
                    })
                if content_blocks:
                    converted.append({"role": "assistant", "content": content_blocks})

            elif msg["role"] == "tool":
                tool_results: list[dict] = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": messages[i].get("tool_call_id", ""),
                        "content": messages[i].get("content", ""),
                    })
                    i += 1
                converted.append({"role": "user", "content": tool_results})
                continue

            i += 1
        return converted

    async def chat(self, system, messages, tools=None, max_tokens=1024):
        kwargs: dict = {
            "model": self.model,
            "system": system,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools  # Our format matches Anthropic's native format

        resp = await self.client.messages.create(**kwargs)

        text = None
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        )
