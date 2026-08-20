"""
Liaison Agent Configuration
Pydantic v2 models for all configuration options.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Literal


class VoiceSessionConfig(BaseModel):
    """Configuration for the Voice Facade (Gemini Live API session)."""

    model: str = "gemini-3.1-flash-live-preview"
    audio_input_sample_rate: int = 16000
    audio_output_sample_rate: int = 24000
    video_fps: int = 1
    video_mime_type: Literal["image/jpeg"] = "image/jpeg"
    enable_input_transcription: bool = True
    enable_output_transcription: bool = True
    session_timeout_seconds: int = 900
    resumption_trigger_seconds: int = 840
    prepare_advance_seconds: int = 30
    voice_name: str = "Zephyr"
    speaking_rate: float = 1.0
    pitch: float = 0.0
    volume_gain_db: float = 0.0


class ReasoningCoreConfig(BaseModel):
    """Configuration for the Reasoning Core (Nemotron via SwitchyardRouter)."""

    router_config_path: str = "config/models.yaml"
    default_task_type: Literal["reasoning", "coding", "auto", "fast"] = "reasoning"
    guardrails_config: Optional[str] = None
    vector_store_path: str = "data/vectorstore"
    system_prompt: str = """You are the Reasoning Core of the Liaison Agent.
You receive delegated tasks from the Voice Facade for deep reasoning.
You have access to research tools (web_search, read_files, execute_commands).
You can request Planner dispatch ONLY via the request_planner_dispatch tool
when the DispatchGate approves. You CANNOT write files or modify code."""
    temperature: float = 0.3
    max_tokens: int = 8192
    top_p: float = 0.95


class SessionPersistenceConfig(BaseModel):
    """Configuration for cross-session conversation persistence."""

    enabled: bool = True
    session_file: str = ".liaison_session"
    max_history_turns: int = 100
    auto_save: bool = True


class LiaisonConfig(BaseModel):
    """Top-level Liaison Agent configuration."""

    voice: VoiceSessionConfig = Field(default_factory=VoiceSessionConfig)
    reasoning: ReasoningCoreConfig = Field(default_factory=ReasoningCoreConfig)
    session: SessionPersistenceConfig = Field(default_factory=SessionPersistenceConfig)
    event_store_path: str = "data/events.db"
    log_level: str = "INFO"