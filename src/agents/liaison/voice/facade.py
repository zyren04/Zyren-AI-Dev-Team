"""
Liaison Agent Voice Facade
Gemini Live API WebSocket wrapper for real-time multimodal interaction.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from google import genai
from google.genai import types

from ..config import VoiceSessionConfig
from ..session import SessionResumptionManager
from ..controls import VoiceLifecycleController, VoiceState, VoiceSessionResult
from ..dispatch_gate import PlannerDispatchGate
from ..tools.definitions import VOICE_FACADE_TOOLS
from ..tools.handlers import ToolHandler

logger = logging.getLogger(__name__)


class VoiceFacade:
    def __init__(
        self,
        config: VoiceSessionConfig,
        tool_handler: ToolHandler,
        dispatch_gate: PlannerDispatchGate,
        on_transcription: Optional[Callable[[str, bool, str], None]] = None,
        on_audio_output: Optional[Callable[[bytes], None]] = None,
        on_notification: Optional[Callable[[str], None]] = None,
    ):
        self.config = config
        self.tool_handler = tool_handler
        self.dispatch_gate = dispatch_gate
        self.on_transcription = on_transcription
        self.on_audio_output = on_audio_output
        self.on_notification = on_notification

        self.state = VoiceState.IDLE
        self._session = None
        self._lifecycle = VoiceLifecycleController(config)
        self._resumption_manager = SessionResumptionManager(config)
        self._resumption_manager.set_voice_facade(self)

        self._audio_in_queue = asyncio.Queue()
        self._audio_out_queue = asyncio.Queue(maxsize=5)
        self._video_out_queue = asyncio.Queue(maxsize=5)
        self._text_out_queue = asyncio.Queue()

        self._tasks = []
        self._conversation_history = []

    @property
    def is_active(self):
        return self.state in (VoiceState.ACTIVE, VoiceState.MUTED, VoiceState.CAMERA_OFF, VoiceState.SCREEN_OFF)

    async def start_session(self):
        result = await self._lifecycle.start_voice_session()
        if result.success:
            self._session = self._lifecycle._session
            self.state = self._lifecycle.state
            await self._resumption_manager.start_monitoring()
            await self._start_pipeline_tasks()
        return result

    async def stop_session(self):
        await self._resumption_manager.stop_monitoring()
        await self._stop_pipeline_tasks()
        result = await self._lifecycle.stop_voice_session()
        self.state = self._lifecycle.state
        self._session = None
        return result

    async def mute_microphone(self):
        result = await self._lifecycle.mute_microphone()
        self.state = self._lifecycle.state
        return result

    async def unmute_microphone(self):
        result = await self._lifecycle.unmute_microphone()
        self.state = self._lifecycle.state
        return result

    async def toggle_camera(self, enabled: bool):
        result = await self._lifecycle.toggle_camera(enabled)
        self.state = self._lifecycle.state
        return result

    async def toggle_screen_share(self, enabled: bool):
        result = await self._lifecycle.toggle_screen_share(enabled)
        self.state = self._lifecycle.state
        return result

    async def send_text(self, text: str):
        if self._session and self.is_active:
            await self._session.send_realtime_input(text=text)
            self._conversation_history.append({"role": "user", "content": text, "type": "text"})

    async def _start_pipeline_tasks(self):
        self._tasks = [
            asyncio.create_task(self._receive_loop()),
            asyncio.create_task(self._send_audio_loop()),
            asyncio.create_task(self._send_video_loop()),
            asyncio.create_task(self._send_text_loop()),
        ]

    async def _stop_pipeline_tasks(self):
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._tasks.clear()

    async def _receive_loop(self):
        if not self._session:
            return
        try:
            async for response in self._session.receive():
                await self._handle_response(response)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive loop error: {e}")
            await self._notify_user(f"Connection error: {e}")

    async def _handle_response(self, response):
        if hasattr(response, "data") and response.data:
            if self.on_audio_output:
                self.on_audio_output(response.data)

        if hasattr(response, "text") and response.text:
            self._conversation_history.append({"role": "model", "content": response.text, "type": "text"})
            if self.on_transcription:
                self.on_transcription(response.text, True, "assistant")

        if hasattr(response, "server_content"):
            server_content = response.server_content
            if hasattr(server_content, "input_transcription") and server_content.input_transcription:
                text = server_content.input_transcription.text
                is_final = getattr(server_content.input_transcription, "done", False)
                if self.on_transcription:
                    self.on_transcription(text, is_final, "user")
                self._conversation_history.append({"role": "user", "content": text, "type": "transcription"})

            if hasattr(server_content, "output_transcription") and server_content.output_transcription:
                text = server_content.output_transcription.text
                is_final = getattr(server_content.output_transcription, "done", False)
                if self.on_transcription:
                    self.on_transcription(text, is_final, "assistant")

        if hasattr(response, "tool_call") and response.tool_call:
            for tool_call in response.tool_call.function_calls:
                await self._handle_tool_call(tool_call)

        if hasattr(response, "server_content") and getattr(response.server_content, "turn_complete", False):
            await self._handle_turn_complete()

    async def _handle_tool_call(self, tool_call):
        try:
            from google.genai.types import FunctionCall
            call = FunctionCall(
                name=tool_call.name,
                args=dict(tool_call.args) if tool_call.args else {},
                id=tool_call.id
            )
            result = await self.tool_handler.handle_tool_call(call)
            if self._session:
                await self._session.send_tool_response(function_responses=[result])
        except Exception as e:
            logger.error(f"Tool call error: {e}")
            from google.genai.types import FunctionResponse
            error_response = FunctionResponse(
                name=tool_call.name,
                response={"error": str(e)},
                id=tool_call.id
            )
            if self._session:
                await self._session.send_tool_response(function_responses=[error_response])

    async def _handle_turn_complete(self):
        while not self._audio_in_queue.empty():
            try:
                self._audio_in_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _send_audio_loop(self):
        while True:
            try:
                audio_data = await self._audio_out_queue.get()
                if self._session and self.is_active:
                    blob = types.Blob(data=audio_data, mime_type="audio/pcm")
                    await self._session.send_realtime_input(audio=blob)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio send error: {e}")

    async def _send_video_loop(self):
        while True:
            try:
                frame = await self._video_out_queue.get()
                if self._session and self.is_active:
                    blob = types.Blob(data=frame["data"], mime_type=frame["mime_type"])
                    await self._session.send_realtime_input(video=blob)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Video send error: {e}")

    async def _send_text_loop(self):
        while True:
            try:
                text = await self._text_out_queue.get()
                if self._session and self.is_active:
                    await self._session.send_realtime_input(text=text)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Text send error: {e}")

    def queue_audio(self, data: bytes):
        try:
            self._audio_out_queue.put_nowait(data)
        except asyncio.QueueFull:
            pass

    def queue_video(self, data: bytes, mime_type: str = "image/jpeg"):
        try:
            self._video_out_queue.put_nowait({"data": data, "mime_type": mime_type})
        except asyncio.QueueFull:
            pass

    def queue_text(self, text: str):
        self._text_out_queue.put_nowait(text)

    async def _pause_audio_io(self):
        while not self._audio_in_queue.empty():
            try:
                self._audio_in_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self._audio_out_queue.empty():
            try:
                self._audio_out_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _resume_audio_io(self):
        pass

    async def _flush_transcription_buffers(self):
        pass

    async def _notify_user(self, message: str):
        if self.on_notification:
            self.on_notification(message)

    def record_turn(self, role: str, parts: List):
        self._resumption_manager.record_turn(role, parts)

    def get_conversation_history(self):
        return self._conversation_history.copy()
