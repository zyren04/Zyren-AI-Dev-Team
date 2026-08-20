"""
Tests for Reasoning Core Router Integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.liaison.reasoning.core import ReasoningCore
from src.routing.router import SwitchyardRouter
from src.governance.guardrails import GuardrailsEngine


class TestReasoningCoreRouterIntegration:
    @pytest.fixture
    def mock_router(self):
        router = AsyncMock(spec=SwitchyardRouter)
        router.route = AsyncMock(return_value="Reasoned response")
        router.list_targets.return_value = ["default", "reasoning", "coding", "fast"]
        return router

    @pytest.fixture
    def mock_guardrails(self):
        return MagicMock(spec=GuardrailsEngine)

    @pytest.mark.asyncio
    async def test_reason_uses_router(self, mock_router, mock_guardrails):
        core = ReasoningCore(router=mock_router, guardrails=mock_guardrails)
        result = await core.reason("Analyze this architecture", task_type="reasoning")

        mock_router.route.assert_called_once()
        call_args = mock_router.route.call_args
        assert call_args.kwargs["task_type"] == "reasoning"
        assert "Analyze this architecture" in call_args.kwargs["prompt"]
        assert result == "Reasoned response"

    @pytest.mark.asyncio
    async def test_reason_auto_routes_coding(self, mock_router, mock_guardrails):
        core = ReasoningCore(router=mock_router, guardrails=mock_guardrails)
        await core.reason("Implement a binary search function", task_type="auto")

        call_args = mock_router.route.call_args
        assert call_args.kwargs["task_type"] == "auto"

    @pytest.mark.asyncio
    async def test_reason_with_tools_not_passed_to_router(self, mock_router, mock_guardrails):
        """Tools are not passed to router (NIM client doesn't support FunctionDeclaration format).
        Tools are handled by Voice Facade via ToolHandler instead."""
        core = ReasoningCore(router=mock_router, guardrails=mock_guardrails)
        tools = [{"name": "test_tool", "description": "Test"}]

        await core.reason_with_tools("Test", task_type="reasoning", tools=tools)

        call_args = mock_router.route.call_args
        # Tools should NOT be passed to router
        assert "tools" not in call_args.kwargs

    @pytest.mark.asyncio
    async def test_delegate_task_maps_to_router(self, mock_router, mock_guardrails):
        core = ReasoningCore(router=mock_router, guardrails=mock_guardrails)
        result = await core.delegate_task(
            task_type="code_analysis",
            prompt="Review this code",
            preferred_model="coding",
        )

        mock_router.route.assert_called_once()
        call_args = mock_router.route.call_args
        assert call_args.kwargs["task_type"] == "coding"
        assert "Review this code" in call_args.kwargs["prompt"]
        assert result == "Reasoned response"

    @pytest.mark.asyncio
    async def test_delegate_task_defaults_to_reasoning(self, mock_router, mock_guardrails):
        core = ReasoningCore(router=mock_router, guardrails=mock_guardrails)
        await core.delegate_task(task_type="unknown_type", prompt="Test")

        call_args = mock_router.route.call_args
        assert call_args.kwargs["task_type"] == "reasoning"
