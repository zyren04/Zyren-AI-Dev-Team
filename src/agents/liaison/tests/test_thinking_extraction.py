"""
Tests for Thinking Process Extraction
"""

import pytest

from src.agents.liaison.reasoning.core import ReasoningCore, ReasoningResult


class TestThinkingExtraction:
    """Test thinking extraction from model responses."""

    @pytest.fixture
    def core(self):
        """Create a ReasoningCore with mocked router."""
        from unittest.mock import AsyncMock, MagicMock
        from src.routing.router import SwitchyardRouter
        from src.governance.guardrails import GuardrailsEngine
        
        mock_router = AsyncMock()
        mock_router.route = AsyncMock()
        mock_router.list_targets.return_value = ["default", "reasoning", "coding", "fast"]
        
        mock_guardrails = MagicMock()
        
        return ReasoningCore(router=mock_router, guardrails=MagicMock())

    @pytest.mark.asyncio
    async def test_extract_thinking_here_is_thinking_process(self, core):
        """Test extraction of 'Here's a thinking process:' pattern."""
        response = """Here's a thinking process: The user is asking about their name. I should check the conversation history.

The user's name is Mohamed."""
        
        thought, clean = core._extract_thinking(response)
        
        assert thought is not None
        assert "The user is asking about their name" in thought
        assert "The user's name is Mohamed" in clean
        assert "Here's a thinking process:" not in clean

    @pytest.mark.asyncio
    async def test_extract_thinking_thinking_process(self, core):
        """Test extraction of 'Thinking Process:' pattern."""
        response = """Thinking Process: The user wants to know if I remember their name. I should check the conversation history.

Yes, I remember your name is Mohamed."""
        
        thought, clean = core._extract_thinking(response)
        
        assert thought is not None
        assert "The user wants to know if I remember their name" in thought
        assert "Yes, I remember your name is Mohamed" in clean
        assert "Thinking Process:" not in clean

    @pytest.mark.asyncio
    async def test_extract_thinking_thought_tags(self, core):
        """Test extraction of <thought>...</thought> tags."""
        response = """<thought>User is asking about their name. I should check conversation history.</thought>

Your name is Mohamed."""
        
        thought, clean = core._extract_thinking(response)
        
        assert thought is not None
        assert "User is asking about their name" in thought
        assert "Your name is Mohamed" in clean
        assert "<thought>" not in clean
        assert "</thought>" not in clean

    @pytest.mark.asyncio
    async def test_extract_thinking_thinking_tags(self, core):
        """Test extraction of <thinking>...</thinking> tags."""
        response = """<thinking>User wants to know if I remember their name. Check history.</thinking>

Yes, your name is Mohamed."""
        
        thought, clean = core._extract_thinking(response)
        
        assert thought is not None
        assert "User wants to know if I remember their name" in thought
        assert "Yes, your name is Mohamed" in clean
        assert "<thinking>" not in clean
        assert "</thinking>" not in clean

    @pytest.mark.asyncio
    async def test_extract_thinking_multiple_blocks(self, core):
        """Test extraction of multiple thinking blocks."""
        response = """<thought>First thought: check history.</thought>

Response 1.

<thought>Second thought: confirm name.</thought>

Response 2."""
        
        thought, clean = core._extract_thinking(response)
        
        assert thought is not None
        assert "First thought: check history" in thought
        assert "Second thought: confirm name" in thought
        assert "Response 1" in clean
        assert "Response 2" in clean

    @pytest.mark.asyncio
    async def test_no_thinking_blocks(self, core):
        """Test response with no thinking blocks returns None thought."""
        response = "This is a normal response without any thinking blocks."
        
        thought, clean = core._extract_thinking(response)
        
        assert thought is None
        assert clean == response

    @pytest.mark.asyncio
    async def test_reason_returns_reasoning_result(self, core):
        """Test that reason() returns ReasoningResult with thought and final_response."""
        from unittest.mock import AsyncMock
        
        core.router.route = AsyncMock(return_value="<thought>Thinking here</thought>\n\nFinal answer.")
        
        result = await core.reason("Test prompt", task_type="reasoning")
        
        assert isinstance(result, ReasoningResult)
        assert result.thought == "Thinking here"
        assert result.final_response == "Final answer."

    @pytest.mark.asyncio
    async def test_delegate_task_returns_reasoning_result(self, core):
        """Test that delegate_task returns ReasoningResult."""
        from unittest.mock import AsyncMock
        
        core.router.route = AsyncMock(return_value="<thought>Delegated thinking</thought>\n\nDelegated response.")
        
        result = await core.delegate_task(
            task_type="code_analysis",
            prompt="Review this code",
            preferred_model="coding",
        )
        
        assert isinstance(result, ReasoningResult)
        assert result.thought == "Delegated thinking"
        assert result.final_response == "Delegated response."
