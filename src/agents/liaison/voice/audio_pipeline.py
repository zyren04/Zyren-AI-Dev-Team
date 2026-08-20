"""
Liaison Agent Audio Pipeline
Real-time audio capture and playback for Gemini Live API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AudioPipeline:
    """Manages audio capture (16kHz) and playback (24kHz)."""

    def __init__(
        self,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        channels: int = 1,
        chunk_size: int = 1024,
        format_type: int = 16,  # paInt16
    ):
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format_type = format_type
        self._audio = None
        self._input_stream = None
        self._output_stream = None
        self._capture_task = None
        self._playback_task = None
        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()

    async def start(self):
        """Initialize audio streams."""
        try:
            import pyaudio
            self._audio = pyaudio.PyAudio()

            # Input stream (microphone)
            self._input_stream = await asyncio.to_thread(
                self._audio.open,
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.input_sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )

            # Output stream (speaker)
            self._output_stream = await asyncio.to_thread(
                self._audio.open,
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.output_sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
            )

            self._capture_task = asyncio.create_task(self._capture_loop())
            self._playback_task = asyncio.create_task(self._playback_loop())

        except ImportError:
            logger.warning("PyAudio not available, audio disabled")
        except Exception as e:
            logger.error(f"Failed to start audio pipeline: {e}")
            raise

    async def stop(self):
        """Stop audio streams."""
        for task in [self._capture_task, self._playback_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._input_stream:
            self._input_stream.stop_stream()
            self._input_stream.close()
        if self._output_stream:
            self._output_stream.stop_stream()
            self._output_stream.close()
        if self._audio:
            self._audio.terminate()

    async def _capture_loop(self):
        """Capture audio from microphone."""
        while True:
            try:
                data = await asyncio.to_thread(
                    self._input_stream.read,
                    self.chunk_size,
                    exception_on_overflow=False
                )
                await self._input_queue.put(data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio capture error: {e}")
                await asyncio.sleep(0.1)

    async def _playback_loop(self):
        """Play audio to speaker."""
        while True:
            try:
                data = await self._output_queue.get()
                await asyncio.to_thread(self._output_stream.write, data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio playback error: {e}")

    def get_input_queue(self):
        return self._input_queue

    def get_output_queue(self):
        return self._output_queue

    def queue_playback(self, data: bytes):
        """Queue audio for playback."""
        try:
            self._output_queue.put_nowait(data)
        except asyncio.QueueFull:
            pass
