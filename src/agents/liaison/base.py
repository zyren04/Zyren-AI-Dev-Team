"""
Liaison Agent Base Classes and Protocols
Defines interfaces for Voice Facade, Reasoning Core, and Tool Handler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable
from google.genai.types import FunctionDeclaration, FunctionCall, FunctionResponse


@runtime_checkable
class VoiceFacadeProtocol(Protocol):
    """Protocol for the Voice Facade (Gemini Live API wrapper)."""

    @property
    def state(self) -> "VoiceState":
        """Current voice session state."""
        ...

    @property
    def is_active(self) -> bool:
        """Whether voice session is actively streaming."""
        ...

    async def start_session(self) -> "VoiceSessionResult":
        """Start a new voice session (WebSocket connection)."""
        ...

    async def stop_session(self) -> "VoiceSessionResult":
        """Stop the current voice session gracefully."""
        ...

    async def mute_microphone(self) -> "VoiceSessionResult":
        """Mute microphone input."""
        ...

    async def unmute_microphone(self) -> "VoiceSessionResult":
        """Unmute microphone input."""
        ...

    async def toggle_camera(self, enabled: bool) -> "VoiceSessionResult":
        """Enable or disable camera feed."""
        ...

    async def toggle_screen_share(self, enabled: bool) -> "VoiceSessionResult":
        """Enable or disable screen sharing."""
        ...

    async def send_text(self, text: str) -> None:
        """Send text input to the Live API session."""
        ...

    def register_tool_handler(self, handler: "ToolHandlerProtocol") -> None:
        """Register tool handler for function calling."""
        ...

    async def handle_tool_call(self, call: FunctionCall) -> FunctionResponse:
        """Handle a tool call from the Live API."""
        ...


@runtime_checkable
class ReasoningCoreProtocol(Protocol):
    """Protocol for the Reasoning Core (Nemotron via SwitchyardRouter)."""

    async def reason(
        self,
        prompt: str,
        task_type: str = "reasoning",
        system_prompt: Optional[str] = None,
        tools: Optional[list[FunctionDeclaration]] = None,
        **kwargs,
    ) -> str:
        """Execute a reasoning task."""
        ...

    async def reason_with_tools(
        self,
        prompt: str,
        task_type: str = "reasoning",
        tools: Optional[list[FunctionDeclaration]] = None,
        **kwargs,
    ) -> str:
        """Execute a reasoning task with function calling."""
        ...

    def get_available_tools(self) -> list[FunctionDeclaration]:
        """Get list of available tool declarations."""
        ...


@runtime_checkable
class ToolHandlerProtocol(Protocol):
    """Protocol for tool execution handler."""

    async def handle_tool_call(self, call: FunctionCall) -> FunctionResponse:
        """Execute a tool call and return response."""
        ...

    def get_tool_declarations(self) -> list[FunctionDeclaration]:
        """Get all registered tool declarations."""
        ...


# Forward references for type hints
class VoiceState:
    pass


class VoiceSessionResult:
    pass