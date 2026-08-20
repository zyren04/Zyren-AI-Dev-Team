"""
Liaison Agent Reasoning Core
Deep reasoning via Nemotron-3-Ultra through SwitchyardRouter.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from google.genai import types

from src.routing.router import get_router, SwitchyardRouter
from src.governance.guardrails import get_guardrails_engine, GuardrailsEngine
from src.vectorstore.faiss_store import FAISSVectorStore
from src.state.event_store import get_event_store, EventStore
from ..config import ReasoningCoreConfig
from ..tools.definitions import REASONING_CORE_TOOLS
from ..tools.handlers import ToolHandler
from ..dispatch_gate import PLANNER_DISPATCH_TOOL

logger = logging.getLogger(__name__)


class ReasoningCore:
    """Deep reasoning engine using SwitchyardRouter for model selection."""

    def __init__(
        self,
        router: Optional[SwitchyardRouter] = None,
        guardrails: Optional[GuardrailsEngine] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        event_store: Optional[EventStore] = None,
        config: Optional[ReasoningCoreConfig] = None,
    ):
        self.router = router or get_router()
        self.guardrails = guardrails
        self.vector_store = vector_store
        self.event_store = event_store
        self.config = config or ReasoningCoreConfig()
        self.tool_handler = ToolHandler()

        # System prompt for reasoning tasks
        self.system_prompt = self.config.system_prompt

    async def reason(
        self,
        prompt: str,
        task_type: str = "reasoning",
        system_prompt: Optional[str] = None,
        tools: Optional[List[types.FunctionDeclaration]] = None,
        **kwargs,
    ) -> str:
        """Execute a reasoning task via SwitchyardRouter."""
        combined_prompt = (system_prompt or self.system_prompt) + "\n\n" + prompt

        # Note: Tools are not passed to the router because the NIM client
        # (langchain-nvidia-ai-endpoints) doesn't support google-genai's
        # FunctionDeclaration format. Tools are handled by the Voice Facade
        # via the ToolHandler instead.
        response = await self.router.route(
            prompt=combined_prompt,
            task_type=task_type,
            system_prompt=None,
            **kwargs,
        )

        # Validate response through guardrails if needed
        # (Implementation depends on guardrails integration pattern)

        return response

    async def reason_with_tools(
        self,
        prompt: str,
        task_type: str = "reasoning",
        tools: Optional[List[types.FunctionDeclaration]] = None,
        **kwargs,
    ) -> str:
        """Execute reasoning with function calling support."""
        if tools is None:
            tools = REASONING_CORE_TOOLS
        return await self.reason(prompt=prompt, task_type=task_type, tools=tools, **kwargs)

    def get_available_tools(self) -> List[types.FunctionDeclaration]:
        return REASONING_CORE_TOOLS

    async def delegate_task(
        self,
        task_type: str,
        prompt: str,
        preferred_model: str = "auto",
        require_verification: bool = True,
        max_tokens: int = 8192,
    ) -> str:
        """Handle delegation from Voice Facade."""
        # Map task_type to router task_type
        router_task_type = preferred_model if preferred_model != "auto" else task_type
        if router_task_type not in ("reasoning", "coding", "fast", "default", "auto"):
            router_task_type = "reasoning"

        return await self.reason(
            prompt=prompt,
            task_type=router_task_type,
            max_tokens=max_tokens,
        )
