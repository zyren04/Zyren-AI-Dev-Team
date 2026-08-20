"""
Tests for Session Resumption Manager
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.liaison.session import SessionResumptionManager, ConversationContext, ConversationTurn
from src.agents.liaison.config import VoiceSessionConfig
from src.agents.liaison.exceptions import SessionResumptionError
from google.genai.types import Part


class TestSessionResumptionManager:
    @pytest.fixture
    def config(self):
        return VoiceSessionConfig()

    @pytest.fixture
    def manager(self, config):
        return SessionResumptionManager(config)

    def test_initial_context(self, manager):
        assert isinstance(manager.context, ConversationContext)
        assert len(manager.context.turns) == 0
        assert manager.context.created_at > 0

    def test_record_turn(self, manager):
        parts = [Part(text="Hello")]
        manager.record_turn("user", parts)
        assert len(manager.context.turns) == 1
        assert manager.context.turns[0].role == "user"
        assert manager.context.turns[0].parts == parts

    def test_trim_history(self, manager):
        # Add more than max_turns
        for i in range(105):
            manager.record_turn("user", [Part(text=f"Turn {i}")])
        assert len(manager.context.turns) == 100  # max_turns default

    def test_create_context_snapshot(self, manager):
        manager.record_turn("user", [Part(text="Test")])
        snapshot = manager._create_context_snapshot()
        assert len(snapshot.turns) == 1
        assert snapshot.turns[0].role == "user"
        # Ensure it's a deep copy
        snapshot.turns.clear()
        assert len(manager.context.turns) == 1

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, manager):
        await manager.start_monitoring()
        assert manager._monitor_task is not None
        await manager.stop_monitoring()
        assert manager._monitor_task.done()

    @pytest.mark.asyncio
    async def test_handle_unexpected_disconnect(self, manager):
        mock_facade = AsyncMock()
        mock_facade.state = None
        manager.set_voice_facade(mock_facade)

        with patch.object(manager, "_establish_new_session", new_callable=AsyncMock) as mock_establish:
            mock_establish.return_value = AsyncMock()
            with patch.object(manager, "_atomic_session_swap", new_callable=AsyncMock):
                result = await manager.handle_unexpected_disconnect(Exception("Test"))
                assert result is True
                mock_establish.assert_called_once()
