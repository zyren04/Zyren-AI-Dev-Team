"""
Liaison Agent Voice Lifecycle Controls
Explicit on-demand voice session management with state machine.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .exceptions import VoiceSessionError
from .config import VoiceSessionConfig


class VoiceState(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    ACTIVE = "active"
    MUTED = "muted"
    CAMERA_OFF = "camera_off"
    SCREEN_OFF = "screen_off"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"


@dataclass
class VoiceSessionResult:
    success: bool
    message: str = ""
    session_id: Optional[str] = None
    previous_state: Optional[VoiceState] = None
    new_state: Optional[VoiceState] = None


class VoiceLifecycleController:
    def __init__(self, config: VoiceSessionConfig):
        self.config = config
        self.state = VoiceState.IDLE
        self._session = None
        self._session_id = None
        self._audio_task = None
        self._video_task = None
        self._transcription_task = None

    @property
    def session_id(self):
        return self._session_id

    @property
    def is_active(self):
        return self.state in (VoiceState.ACTIVE, VoiceState.MUTED, VoiceState.CAMERA_OFF, VoiceState.SCREEN_OFF)

    async def start_voice_session(self):
        if self.state != VoiceState.IDLE:
            raise VoiceSessionError(f"Cannot start from state {self.state.value}")

        previous_state = self.state
        self.state = VoiceState.CONNECTING

        try:
            from google import genai
            from google.genai import types

            config = types.LiveConnectConfig(
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

            client = genai.Client()
            self._session = await client.aio.live.connect(model=self.config.model, config=config)
            self._session_id = f"session_{int(asyncio.get_event_loop().time() * 1000)}"
            self.state = VoiceState.ACTIVE

            return VoiceSessionResult(
                success=True, message="Voice session started",
                session_id=self._session_id,
                previous_state=previous_state, new_state=self.state
            )
        except Exception as e:
            self.state = VoiceState.IDLE
            self._session = None
            self._session_id = None
            raise VoiceSessionError(f"Failed to start voice session: {e}") from e

    async def stop_voice_session(self):
        if self.state == VoiceState.IDLE:
            return VoiceSessionResult(success=True, message="Already stopped",
                previous_state=self.state, new_state=self.state)

        previous_state = self.state
        self.state = VoiceState.CLOSING

        try:
            for task in [self._audio_task, self._video_task, self._transcription_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            if self._session:
                await self._session.close()
                self._session = None
            self._session_id = None
            self.state = VoiceState.IDLE
            return VoiceSessionResult(success=True, message="Voice session stopped",
                previous_state=previous_state, new_state=self.state)
        except Exception as e:
            self.state = VoiceState.IDLE
            raise VoiceSessionError(f"Failed to stop voice session: {e}") from e

    async def mute_microphone(self):
        if self.state != VoiceState.ACTIVE:
            raise VoiceSessionError(f"Can only mute in ACTIVE state, current: {self.state.value}")
        previous_state = self.state
        if self._session:
            await self._session.send_realtime_input(audio=None)
        self.state = VoiceState.MUTED
        return VoiceSessionResult(success=True, message="Microphone muted",
            previous_state=previous_state, new_state=self.state)

    async def unmute_microphone(self):
        if self.state != VoiceState.MUTED:
            raise VoiceSessionError(f"Can only unmute from MUTED state, current: {self.state.value}")
        previous_state = self.state
        self.state = VoiceState.ACTIVE
        return VoiceSessionResult(success=True, message="Microphone unmuted",
            previous_state=previous_state, new_state=self.state)

    async def toggle_camera(self, enabled: bool):
        if not self.is_active:
            raise VoiceSessionError(f"Camera toggle only available in active states, current: {self.state.value}")
        previous_state = self.state
        if enabled:
            self.state = VoiceState.ACTIVE
            message = "Camera enabled"
        else:
            self.state = VoiceState.CAMERA_OFF
            message = "Camera disabled"
        return VoiceSessionResult(success=True, message=message,
            previous_state=previous_state, new_state=self.state)

    async def toggle_screen_share(self, enabled: bool):
        if not self.is_active:
            raise VoiceSessionError(f"Screen share toggle only available in active states, current: {self.state.value}")
        previous_state = self.state
        if enabled:
            self.state = VoiceState.ACTIVE
            message = "Screen sharing started"
        else:
            self.state = VoiceState.SCREEN_OFF
            message = "Screen sharing stopped"
        return VoiceSessionResult(success=True, message=message,
            previous_state=previous_state, new_state=self.state)


class LiveSession:
    pass
