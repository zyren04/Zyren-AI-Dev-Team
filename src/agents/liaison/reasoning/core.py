"""
Liaison Agent Reasoning Core
Deep reasoning via Nemotron-3-Ultra through SwitchyardRouter.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

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


@dataclass
class ReasoningResult:
    """Result of a reasoning task with separated thinking and final response."""
    thought: Optional[str]
    final_response: str


class ReasoningCore:
    """Deep reasoning engine using SwitchyardRouter for model selection."""

    # Regex patterns for thinking extraction
    THINKING_PATTERNS = [
        # Pattern: <thought>...</thought> (most specific, check first)
        re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE),
        # Pattern: <thinking>...</thinking>
        re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE),
        # Pattern: <|thinking|>...<|end_thinking|>
        re.compile(r"<\|thinking\|>(.*?)<\|end_thinking\|>", re.DOTALL | re.IGNORECASE),
        # Pattern: 【Thinking】...【/Thinking】 or similar
        re.compile(r"【Thinking】(.*?)【/Thinking】", re.DOTALL | re.IGNORECASE),
        # Pattern: Here's a thinking process: ... (ends at double newline or end)
        # Must match at START of string, not inside other phrases
        re.compile(r"^Here's a thinking process:\s*(.*?)(?=\n\n|\Z)", re.DOTALL | re.IGNORECASE),
        # Pattern: Thinking Process: ... (ends at double newline or end)
        # Must match at START of string
        re.compile(r"^Thinking Process:\s*(.*?)(?=\n\n|\Z)", re.DOTALL | re.IGNORECASE),
        # Pattern: Here's my thinking: ... (ends at double newline or end)
        re.compile(r"^Here's my thinking:\s*(.*?)(?=\n\n|\Z)", re.DOTALL | re.IGNORECASE),
        # Pattern: 【Thinking】...【/Thinking】 or similar
        re.compile(r"【Thinking】(.*?)【/Thinking】", re.DOTALL | re.IGNORECASE),
    ]

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

    def _extract_thinking(self, response: str) -> Tuple[Optional[str], str]:
        """
        Extract thinking process from model response.

        Returns:
            Tuple of (thought, clean_response)
            - thought: Extracted thinking trace or None if not found
            - clean_response: Response with thinking blocks removed
        """
        # Collect all matches from all patterns first
        all_matches = []
        for pattern in self.THINKING_PATTERNS:
            for match in pattern.finditer(response):
                thought_content = match.group(1).strip()
                if thought_content:
                    all_matches.append((match.start(), match.end(), thought_content))

        # Sort matches by start position (reverse order for safe removal)
        all_matches.sort(key=lambda x: x[0], reverse=True)

        # Build clean response by removing thinking blocks from end to start
        clean_response = response
        thought_parts = []

        for start, end, thought_content in all_matches:
            if thought_content:
                thought_parts.append(thought_content)
            # Remove the thinking block from clean response
            clean_response = clean_response[:start] + clean_response[end:]

        # Clean up the response - remove extra whitespace
        clean_response = re.sub(r'\n{3,}', '\n\n', clean_response).strip()

        # Reverse thought_parts to maintain original order
        thought_parts.reverse()
        thought = "\n\n".join(thought_parts) if thought_parts else None

        return thought, clean_response

    async def reason(
        self,
        prompt: str,
        task_type: str = "reasoning",
        system_prompt: Optional[str] = None,
        tools: Optional[List[types.FunctionDeclaration]] = None,
        **kwargs,
    ) -> ReasoningResult:
        """Execute a reasoning task via SwitchyardRouter with thinking extraction."""
        # Use provided system_prompt or fall back to config default
        effective_system_prompt = system_prompt or self.system_prompt

        # Note: Tools are not passed to the router because the NIM client
        # (langchain-nvidia-ai-endpoints) doesn't support google-genai's
        # FunctionDeclaration format. Tools are handled by the Voice Facade
        # via the ToolHandler instead.
        response = await self.router.route(
            prompt=prompt,
            task_type=task_type,
            system_prompt=effective_system_prompt,
            **kwargs,
        )

        # Extract thinking from response
        thought, clean_response = self._extract_thinking(response)

        # Validate response through guardrails if needed
        # (Implementation depends on guardrails integration pattern)

        return ReasoningResult(thought=thought, final_response=clean_response)

    async def reason_with_tools(
        self,
        prompt: str,
        task_type: str = "reasoning",
        tools: Optional[List[types.FunctionDeclaration]] = None,
        **kwargs,
    ) -> ReasoningResult:
        """Execute reasoning with function calling support."""
        if tools is None:
            tools = REASONING_CORE_TOOLS
        # For now, delegate to reason() since tools aren't passed to router
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
    ) -> ReasoningResult:
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
