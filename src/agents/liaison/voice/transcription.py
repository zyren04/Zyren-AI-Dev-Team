"""
Liaison Agent Transcription Handler
Live bidirectional transcription for STT/TTS display.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionEntry:
    """Single transcription entry."""
    text: str
    is_final: bool
    speaker: str  # "user" or "assistant"
    timestamp: float


class TranscriptionHandler:
    """Handles live transcription from Gemini Live API."""

    def __init__(
        self,
        on_transcription: Optional[Callable[[str, bool, str], None]] = None,
        max_history: int = 1000,
    ):
        self.on_transcription = on_transcription
        self.max_history = max_history
        self._history: List[TranscriptionEntry] = []
        self._partial_user = ""
        self._partial_assistant = ""

    def handle_input_transcription(self, text: str, is_final: bool):
        """Handle user speech transcription (STT)."""
        if is_final:
            self._partial_user = ""
            entry = TranscriptionEntry(text=text, is_final=True, speaker="user", timestamp=asyncio.get_event_loop().time())
            self._history.append(entry)
        else:
            self._partial_user = text
            entry = TranscriptionEntry(text=text, is_final=False, speaker="user", timestamp=asyncio.get_event_loop().time())

        if self.on_transcription:
            self.on_transcription(text, is_final, "user")
        self._trim_history()

    def handle_output_transcription(self, text: str, is_final: bool):
        """Handle assistant speech transcription (TTS)."""
        if is_final:
            self._partial_assistant = ""
            entry = TranscriptionEntry(text=text, is_final=True, speaker="assistant", timestamp=asyncio.get_event_loop().time())
            self._history.append(entry)
        else:
            self._partial_assistant = text
            entry = TranscriptionEntry(text=text, is_final=False, speaker="assistant", timestamp=asyncio.get_event_loop().time())

        if self.on_transcription:
            self.on_transcription(text, is_final, "assistant")
        self._trim_history()

    def get_partial_user(self) -> str:
        return self._partial_user

    def get_partial_assistant(self) -> str:
        return self._partial_assistant

    def get_history(self, limit: Optional[int] = None) -> List[TranscriptionEntry]:
        if limit:
            return self._history[-limit:]
        return self._history.copy()

    def _trim_history(self):
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

    def clear(self):
        self._history.clear()
        self._partial_user = ""
        self._partial_assistant = ""


class LiveTranscriptDisplay:
    """Rich-based live transcript display for TUI."""

    def __init__(self, handler: TranscriptionHandler):
        self.handler = handler
        self._running = False
        self._task = None

    async def start(self):
        """Start display loop."""
        self._running = True
        self._task = asyncio.create_task(self._display_loop())

    async def stop(self):
        """Stop display loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _display_loop(self):
        """Display loop - renders transcript to console."""
        while self._running:
            # This would use Rich for pretty display
            # For now, simple console output
            await asyncio.sleep(0.5)
