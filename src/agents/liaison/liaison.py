"""
Liaison Agent - Main Orchestrator
Coordinates Voice Facade and Reasoning Core.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.genai import types

from .config import LiaisonConfig
from .voice.facade import VoiceFacade
from .reasoning.core import ReasoningCore
from .session import SessionResumptionManager
from .controls import VoiceLifecycleController, VoiceState, VoiceSessionResult
from .dispatch_gate import PlannerDispatchGate, DispatchDecision
from .tools.definitions import VOICE_FACADE_TOOLS, REASONING_CORE_TOOLS
from .tools.handlers import ToolHandler

logger = logging.getLogger(__name__)


class LiaisonAgent:
    """Main Liaison Agent orchestrating Voice Facade and Reasoning Core."""

    def __init__(
        self,
        config: Optional[LiaisonConfig] = None,
        router=None,
        guardrails=None,
        vector_store=None,
        event_store=None,
        session_id: Optional[str] = None,
    ):
        self.config = config or LiaisonConfig()
        self.session_id = session_id
        self._session_file = Path(self.config.session.session_file)
        
        # Initialize components
        self.tool_handler = ToolHandler()
        self.dispatch_gate = PlannerDispatchGate()
        
        # Voice Facade
        self.voice = VoiceFacade(
            config=self.config.voice,
            tool_handler=self.tool_handler,
            dispatch_gate=self.dispatch_gate,
            on_transcription=self._on_transcription,
            on_audio_output=self._on_audio_output,
            on_notification=self._on_notification,
        )
        
        # Reasoning Core
        self.reasoning = ReasoningCore(
            router=router,
            guardrails=guardrails,
            vector_store=vector_store,
            event_store=event_store,
            config=self.config.reasoning,
        )
        
        # State
        self._running = False
        self._conversation_history: List[Dict] = []
        self._pending_delegations: Dict[str, asyncio.Future] = {}
        self._event_store = event_store

    async def initialize(self):
        """Initialize all components and load conversation history."""
        logger.info("Liaison Agent initialized")
        
        # Load or create session ID
        if not self.session_id:
            self.session_id = self._load_or_create_session_id()
        
        # Load conversation history from EventStore
        if self.config.session.enabled and self._event_store:
            await self._load_conversation_history()

    def _load_or_create_session_id(self) -> str:
        """Load session ID from file or create new one."""
        if self._session_file.exists():
            try:
                session_id = self._session_file.read_text().strip()
                if session_id:
                    logger.info(f"Loaded existing session: {session_id}")
                    return session_id
            except Exception as e:
                logger.warning(f"Failed to load session file: {e}")
        
        # Create new session ID
        session_id = str(uuid.uuid4())
        try:
            self._session_file.write_text(session_id)
            logger.info(f"Created new session: {session_id}")
        except Exception as e:
            logger.warning(f"Failed to save session file: {e}")
        return session_id

    async def _load_conversation_history(self):
        """Load conversation history from EventStore."""
        try:
            events = await self._event_store.get_events(
                execution_id=self.session_id,
                event_type="liaison_conversation_turn"
            )
            
            # Sort by timestamp
            events.sort(key=lambda e: e.timestamp)
            
            # Reconstruct conversation history
            self._conversation_history = []
            for event in events:
                payload = event.payload
                if "user_input" in payload:
                    self._conversation_history.append({
                        "role": "user",
                        "content": payload["user_input"],
                        "timestamp": event.timestamp.isoformat()
                    })
                if "agent_response" in payload:
                    self._conversation_history.append({
                        "role": "assistant",
                        "content": payload["agent_response"],
                        "timestamp": event.timestamp.isoformat()
                    })
            
            # Trim to max history
            max_turns = self.config.session.max_history_turns
            if len(self._conversation_history) > max_turns:
                self._conversation_history = self._conversation_history[-max_turns:]
            
            logger.info(f"Loaded {len(self._conversation_history)} conversation turns from session {self.session_id}")
        except Exception as e:
            logger.warning(f"Failed to load conversation history: {e}")

    async def _save_conversation_turn(self, user_input: str, agent_response: str):
        """Save a conversation turn to EventStore."""
        if not self.config.session.enabled or not self._event_store or not self.config.session.auto_save:
            return
        
        try:
            await self._event_store.record_event(
                execution_id=self.session_id,
                event_type="liaison_conversation_turn",
                payload={
                    "user_input": user_input,
                    "agent_response": agent_response,
                    "timestamp": asyncio.get_event_loop().time()
                },
                node_name="liaison_agent",
                iteration=len(self._conversation_history)
            )
        except Exception as e:
            logger.warning(f"Failed to save conversation turn: {e}")

    async def start_voice(self) -> VoiceSessionResult:
        """Start voice session."""
        return await self.voice.start_session()

    async def stop_voice(self) -> VoiceSessionResult:
        """Stop voice session."""
        return await self.voice.stop_session()

    async def mute_microphone(self) -> VoiceSessionResult:
        return await self.voice.mute_microphone()

    async def unmute_microphone(self) -> VoiceSessionResult:
        return await self.voice.unmute_microphone()

    async def toggle_camera(self, enabled: bool) -> VoiceSessionResult:
        return await self.voice.toggle_camera(enabled)

    async def toggle_screen_share(self, enabled: bool) -> VoiceSessionResult:
        return await self.voice.toggle_screen_share(enabled)

    async def send_text(self, text: str):
        """Send text input (for non-voice interaction)."""
        await self.voice.send_text(text)
        self._conversation_history.append({"role": "user", "content": text})

    def _build_conversation_context(self) -> str:
        """Build conversation context string from history."""
        if not self._conversation_history:
            return ""
        
        context_lines = ["Previous conversation:"]
        for turn in self._conversation_history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            if role == "user":
                context_lines.append(f"User: {content}")
            elif role == "assistant":
                context_lines.append(f"Assistant: {content}")
        context_lines.append("---")
        return "\n".join(context_lines)

    async def process_text(self, text: str) -> str:
        """Process text input through Reasoning Core directly with conversation context."""
        # Build context from conversation history
        context = self._build_conversation_context()
        
        # Prepare prompt with conversation context
        if context:
            prompt_with_context = f"{context}\n\nCurrent user message: {text}"
        else:
            prompt_with_context = text
        
        self._conversation_history.append({"role": "user", "content": text})
        response = await self.reasoning.reason(prompt_with_context, task_type="auto")
        self._conversation_history.append({"role": "assistant", "content": response})
        
        # Save to EventStore
        await self._save_conversation_turn(text, response)
        
        return response

    async def delegate_to_reasoning(
        self,
        task_type: str,
        prompt: str,
        preferred_model: str = "auto",
        require_verification: bool = True,
        max_tokens: int = 8192,
    ) -> str:
        """Delegate task to Reasoning Core (called by Voice Facade via tool)."""
        return await self.reasoning.delegate_task(
            task_type=task_type,
            prompt=prompt,
            preferred_model=preferred_model,
            require_verification=require_verification,
            max_tokens=max_tokens,
        )

    def evaluate_dispatch(self, user_input: str, is_voice_mode: bool) -> DispatchDecision:
        """Evaluate whether to dispatch to Planner."""
        return self.dispatch_gate.evaluate(user_input, is_voice_mode, self._conversation_history)

    def confirm_dispatch(self, confirmed: bool) -> DispatchDecision:
        """Confirm or deny Planner dispatch."""
        return self.dispatch_gate.confirm_dispatch(confirmed)

    def _on_transcription(self, text: str, is_final: bool, speaker: str):
        """Handle transcription callback."""
        logger.debug(f"Transcription [{speaker}]: {text} (final={is_final})")
        # Could emit to EventStore here

    def _on_audio_output(self, data: bytes):
        """Handle audio output callback."""
        # Audio playback handled by VoiceFacade
        pass

    def _on_notification(self, message: str):
        """Handle notification callback."""
        logger.info(f"Notification: {message}")

    def get_conversation_history(self) -> List[Dict]:
        return self._conversation_history.copy()

    def get_voice_state(self) -> VoiceState:
        return self.voice.state

    async def shutdown(self):
        """Graceful shutdown."""
        if self.voice.is_active:
            await self.voice.stop_session()
        self._running = False
        logger.info("Liaison Agent shutdown complete")
