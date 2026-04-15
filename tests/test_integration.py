"""Integration tests for Brain.process() with mocked LLM and Voyage API."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mindsecretary.core.brain import Brain, BrainResponse
from mindsecretary.core.config import Profile, Settings
from mindsecretary.core.database import Database
from mindsecretary.core.memory import Memory
from mindsecretary.llm.client import LLMResponse
from mindsecretary.llm.router import ModelRouter


def _make_profile() -> Profile:
    return Profile(
        name="Test",
        city="Moscow",
        timezone="Europe/Moscow",
        home_coords=[55.75, 37.62],
        work_coords=[55.75, 37.62],
        wake_up="07:00",
        work_start="09:00",
        work_end="18:00",
        sleep="23:00",
        commute_method="metro",
        commute_minutes=30,
        style="кратко",
        language="ru",
        notification_limit=10,
        quiet_hours=["23:00", "07:00"],
        priorities=["work"],
        dislikes=["spam"],
    )


def _make_settings() -> Settings:
    return Settings(
        model="claude-sonnet",
        max_tokens=1000,
        max_tool_rounds=5,
        stt_model="whisper",
        stt_language="ru",
        embedding_model="voyage-3",
        memory_top_k=5,
        relevance_weight=0.6,
        importance_weight=0.4,
    )


@pytest.fixture
def brain_env(tmp_path: Path):
    """Set up a full Brain with mocked LLM and Voyage."""
    db = Database(tmp_path / "test.db")
    # Create memories table (normally done by Memory)
    db.db.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            category TEXT NOT NULL,
            importance INTEGER DEFAULT 5,
            related_person TEXT,
            related_date TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT (datetime('now')),
            last_accessed TEXT
        )
    """)
    db.db.commit()

    # Mock Voyage client
    fake_emb = np.random.randn(1024).astype(np.float32)
    mock_voyage = MagicMock()
    mock_voyage.embed.return_value = MagicMock(embeddings=[fake_emb.tolist()])

    memory = Memory.__new__(Memory)
    memory.db = db.db
    memory.voyage = mock_voyage
    memory.model = "voyage-3"
    memory.relevance_w = 0.6
    memory.importance_w = 0.4

    # Mock LLM
    mock_client = AsyncMock()
    router = ModelRouter(client=mock_client)

    profile = _make_profile()
    settings = _make_settings()

    brain = Brain(
        router=router,
        memory=memory,
        db=db,
        profile=profile,
        settings=settings,
    )

    return brain, mock_client, db


class TestBrainProcess:
    @pytest.mark.asyncio
    async def test_simple_text_response(self, brain_env):
        """Brain returns LLM text when no tool calls."""
        brain, mock_client, db = brain_env

        mock_client.chat.return_value = LLMResponse(
            text="Привет! Чем могу помочь?",
            tool_calls=[],
            usage={"input_tokens": 100, "output_tokens": 20},
        )

        result = await brain.process("Привет")
        assert isinstance(result, BrainResponse)
        assert result.text == "Привет! Чем могу помочь?"
        assert result.tool_calls_made == 0
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_tool_call_save_memory(self, brain_env):
        """Brain executes save_memory tool call from LLM."""
        brain, mock_client, db = brain_env

        # First LLM call returns a tool call
        mock_client.chat.side_effect = [
            LLMResponse(
                text="Запомню!",
                tool_calls=[{
                    "id": "call_1",
                    "name": "save_memory",
                    "arguments": {
                        "content": "User likes tea",
                        "category": "preference",
                        "importance": 7,
                    },
                }],
                usage={"input_tokens": 100, "output_tokens": 30},
            ),
            # Second call: LLM sees tool result, responds
            LLMResponse(
                text="Запомнил, что ты любишь чай!",
                tool_calls=[],
                usage={"input_tokens": 150, "output_tokens": 20},
            ),
        ]

        result = await brain.process("Я люблю чай")
        assert result.tool_calls_made == 1
        assert "чай" in result.text

    @pytest.mark.asyncio
    async def test_tool_call_create_event(self, brain_env):
        """Brain creates an event via tool call."""
        brain, mock_client, db = brain_env

        mock_client.chat.side_effect = [
            LLMResponse(
                text="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "create_event",
                    "arguments": {
                        "title": "Dentist",
                        "start_at": "2026-04-20 10:00:00",
                    },
                }],
                usage={"input_tokens": 100, "output_tokens": 30},
            ),
            LLMResponse(
                text="Записал к стоматологу на 20 апреля!",
                tool_calls=[],
                usage={"input_tokens": 150, "output_tokens": 20},
            ),
        ]

        result = await brain.process("Запиши к стоматологу на 20 апреля в 10")
        assert result.tool_calls_made == 1

        events = db.get_events("2026-04-20")
        assert len(events) == 1
        assert events[0]["title"] == "Dentist"

    @pytest.mark.asyncio
    async def test_llm_failure_returns_error(self, brain_env):
        """Brain returns error message when LLM fails."""
        brain, mock_client, db = brain_env

        mock_client.chat.side_effect = Exception("API down")

        result = await brain.process("Привет")
        assert "Ошибка" in result.text

    @pytest.mark.asyncio
    async def test_max_tool_rounds(self, brain_env):
        """Brain stops after max_tool_rounds."""
        brain, mock_client, db = brain_env
        brain.settings.max_tool_rounds = 2

        # LLM always returns tool calls — should stop after 2 rounds
        mock_client.chat.return_value = LLMResponse(
            text="",
            tool_calls=[{
                "id": "call_loop",
                "name": "get_events",
                "arguments": {"date_from": "2026-04-15"},
            }],
            usage={"input_tokens": 100, "output_tokens": 20},
        )

        result = await brain.process("Что у меня сегодня?")
        assert result.tool_calls_made == 2
        assert "лимит" in result.text.lower() or result.text  # Fallback text

    @pytest.mark.asyncio
    async def test_interaction_logged(self, brain_env):
        """Both input and output are logged as interactions."""
        brain, mock_client, db = brain_env

        mock_client.chat.return_value = LLMResponse(
            text="OK",
            tool_calls=[],
            usage={"input_tokens": 50, "output_tokens": 10},
        )

        await brain.process("Test message", message_type="text")

        interactions = db.get_interactions(limit=10)
        assert len(interactions) == 2
        assert interactions[0]["direction"] == "out"  # most recent first
        assert interactions[1]["direction"] == "in"

    @pytest.mark.asyncio
    async def test_cost_logged(self, brain_env):
        """API cost is logged after each LLM call."""
        brain, mock_client, db = brain_env

        mock_client.chat.return_value = LLMResponse(
            text="Done",
            tool_calls=[],
            usage={"input_tokens": 1000, "output_tokens": 500},
        )

        await brain.process("Hello")

        stats = db.get_stats()
        assert stats["today_tokens"] == 1500
        assert stats["today_cost"] > 0
