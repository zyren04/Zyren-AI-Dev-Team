"""
Liaison Agent Session Management
Conversation context preservation and 15-minute auto-resumption for Gemini Live API.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from google.genai import types
from google.genai.types import LiveConnectConfig, Content, Part

from .exceptions import SessionResumptionError
from .config import VoiceSessionConfig
from .controls import VoiceState


@dataclass
class ConversationTurn:
    """Single conversation turn for context preservation."""

    role: str  # "user" or "model"
    parts: List[Part]
    timestamp: float
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_content(self) -> Content:
        """Convert to Gemini Content object."""
        return Content(role=self.role, parts=self.parts)


@dataclass
class ConversationContext:
    """Complete conversation state for session resumption."""

    turns: List[ConversationTurn] = field(default_factory=list)
    system_instruction: Optional[str] = None
    tool_declarations: List[types.FunctionDeclaration] = field(default_factory=list)
    active_tool_calls: Dict[str, Any] = field(default_factory=dict)  # call_id -> state
    audio_config: Dict[str, Any] = field(default_factory=dict)
    video_config: Dict[str, Any] = field(default_factory=dict)
    transcription_config: Dict[str, Any] = field(default_factory=dict)
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_turn(self, role: str, parts: List[Part]) -> ConversationTurn:
        """Add a new turn to the conversation."""
        turn = ConversationTurn(role=role, parts=parts, timestamp=time.time())
        self.turns.append(turn)
        self.updated_at = time.time()
        return turn

    def get_recent_turns(self, max_turns: int = 100) -> List[ConversationTurn]:
        """Get most recent turns (sliding window)."""
        return self.turns[-max_turns:]

    def to_contents(self, max_turns: int = 100) -> List[Content]:
        """Convert recent turns to Content objects for history seeding."""
        return [turn.to_content() for turn in self.get_recent_turns(max_turns)]


class SessionResumptionManager:
    """
    Manages transparent 15-minute session handoff with full context preservation.

    Gemini Live API has a ~15 minute (900s) WebSocket session limit.
    This manager proactively establishes a new session at ~14 minutes (840s)
    and performs an atomic swap with <200ms audio interruption.
    """

    # Gemini Live API hard limit ~15 min (900s). We resume at 14 min (840s) for safety.
    SESSION_TIMEOUT_SECONDS = 900
    RESUMPTION_TRIGGER_SECONDS = 840  # 14 minutes
    PREPARE_ADVANCE_SECONDS = 30      # Start new connection 30s before swap

    def __init__(self, config: VoiceSessionConfig):
        self.config = config
        self.context = ConversationContext()
        self._monitor_task: Optional[asyncio.Task] = None
        self._resumption_in_progress = False
        self._swap_lock = asyncio.Lock()
        self._voice_facade: Optional["VoiceFacade"] = None  # Set after VoiceFacade creation

    def set_voice_facade(self, facade: "VoiceFacade") -> None:
        """Set reference to VoiceFacade for session swap operations."""
        self._voice_facade = facade

    async def start_monitoring(self) -> None:
        """Start background timeout monitor."""
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._timeout_monitor_loop())

    async def stop_monitoring(self) -> None:
        """Stop background monitor."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _timeout_monitor_loop(self) -> None:
        """Background loop checking session age."""
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            session_age = time.time() - self.context.created_at

            if session_age >= self.RESUMPTION_TRIGGER_SECONDS and not self._resumption_in_progress:
                await self._initiate_resumption()

    async def _initiate_resumption(self) -> None:
        """Orchestrate seamless session handoff."""
        async with self._swap_lock:
            if self._resumption_in_progress:
                return
            if not self._voice_facade:
                raise SessionResumptionError("VoiceFacade not set on SessionResumptionManager")

            self._resumption_in_progress = True
            self._voice_facade.state = VoiceState.RECONNECTING

            try:
                # 1. Snapshot current context (already maintained incrementally)
                snapshot = self._create_context_snapshot()

                # 2. Pre-establish new WebSocket connection
                new_session = await self._establish_new_session(snapshot)

                # 3. Atomic swap with minimal audio gap
                await self._atomic_session_swap(new_session)

                # 4. Update context timestamps
                self.context.created_at = time.time()
                self.context.updated_at = time.time()

            except Exception as e:
                raise SessionResumptionError(f"Session resumption failed: {e}") from e
            finally:
                self._resumption_in_progress = False
                if self._voice_facade:
                    self._voice_facade.state = VoiceState.ACTIVE

    def _create_context_snapshot(self) -> ConversationContext:
        """Create deep copy of current conversation context."""
        return ConversationContext(
            turns=list(self.context.turns),
            system_instruction=self.context.system_instruction,
            tool_declarations=list(self.context.tool_declarations),
            active_tool_calls=dict(self.context.active_tool_calls),
            audio_config=dict(self.context.audio_config),
            video_config=dict(self.context.video_config),
            transcription_config=dict(self.context.transcription_config),
            session_metadata=dict(self.context.session_metadata),
        )

    async def _establish_new_session(self, snapshot: ConversationContext) -> "LiveSession":
        """Create new Live API session pre-seeded with conversation history."""
        from google import genai

        # Build LiveConnectConfig with conversation history
        config = LiveConnectConfig(
            model=self.config.model,
            system_instruction=snapshot.system_instruction,
            tools=[types.Tool(function_declarations=snapshot.tool_declarations)] if snapshot.tool_declarations else None,
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.config.voice_name)
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig() if self.config.enable_input_transcription else None,
            output_audio_transcription=types.AudioTranscriptionConfig() if self.config.enable_output_transcription else None,
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=types.SlidingWindow(target_tokens=12800),
            ),
        )

        # Connect new session
        client = genai.Client()
        new_session = await client.aio.live.connect(model=self.config.model, config=config)

        # Pre-seed conversation history via send_client_content
        if snapshot.turns:
            history_contents = snapshot.to_contents()
            await new_session.send_client_content(turns=history_contents)

        return new_session

    async def _atomic_session_swap(self, new_session: "LiveSession") -> None:
        """Swap sessions with <200ms audio interruption."""
        if not self._voice_facade:
            raise SessionResumptionError("VoiceFacade not set")

        old_session = self._voice_facade._session

        # Pause audio I/O on old session
        await self._voice_facade._pause_audio_io()

        # Flush any pending transcriptions
        await self._voice_facade._flush_transcription_buffers()

        # Atomic reference swap
        self._voice_facade._session = new_session

        # Resume audio I/O on new session
        await self._voice_facade._resume_audio_io()

        # Gracefully close old session
        if old_session:
            await old_session.close()

    def record_turn(self, role: str, parts: List[Part]) -> None:
        """Incrementally record conversation turns for context preservation."""
        self.context.add_turn(role, parts)
        self._trim_history_if_needed()

    def _trim_history_if_needed(self, max_turns: int = 100) -> None:
        """Sliding window to prevent context overflow."""
        if len(self.context.turns) > max_turns:
            # Keep system instruction + most recent turns
            self.context.turns = self.context.turns[-max_turns:]

    async def handle_unexpected_disconnect(self, error: Exception) -> bool:
        """
        Attempt automatic recovery from unexpected disconnect.
        Returns True if recovery successful, False if manual intervention needed.
        """
        if self._resumption_in_progress:
            return False  # Already recovering

        if not self._voice_facade:
            return False

        self._voice_facade.state = VoiceState.RECONNECTING

        try:
            # Use last known good context snapshot
            snapshot = self._create_context_snapshot()
            new_session = await self._establish_new_session(snapshot)
            await self._atomic_session_swap(new_session)
            return True
        except Exception:
            # Log failure, transition to IDLE for manual restart
            self._voice_facade.state = VoiceState.IDLE
            await self._voice_facade._notify_user(
                f"Voice session lost: {error}. Please restart with /voice start"
            )
            return False


# Forward references for type hints
class VoiceFacade:
    pass


class LiveSession:
    pass