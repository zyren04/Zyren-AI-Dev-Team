"""
Liaison Agent Custom Exceptions
Hierarchy of exceptions for precise error handling.
"""

from __future__ import annotations


class LiaisonError(Exception):
    """Base exception for all Liaison Agent errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class VoiceSessionError(LiaisonError):
    """Errors related to voice session lifecycle and WebSocket management."""

    pass


class SessionResumptionError(LiaisonError):
    """Errors during automatic session resumption (15-min timeout handling)."""

    pass


class DispatchGateError(LiaisonError):
    """Errors related to Planner dispatch gating logic."""

    pass


class ToolError(LiaisonError):
    """Errors during tool execution (web_search, read_files, execute_commands)."""

    pass


class ConfigurationError(LiaisonError):
    """Errors related to invalid or missing configuration."""

    pass


class AudioPipelineError(LiaisonError):
    """Errors in audio capture, playback, or processing."""

    pass


class VideoPipelineError(LiaisonError):
    """Errors in video capture, encoding, or streaming."""

    pass


class TranscriptionError(LiaisonError):
    """Errors in live transcription handling."""

    pass


class ReasoningCoreError(LiaisonError):
    """Errors in the Reasoning Core (Nemotron via SwitchyardRouter)."""

    pass


class GuardrailsValidationError(LiaisonError):
    """Errors when guardrails validation fails."""

    pass