"""
Liaison Agent Video Pipeline
Real-time camera and screen capture for Gemini Live API (1 FPS JPEG).
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VideoPipeline:
    """Manages camera and screen capture at 1 FPS JPEG."""

    def __init__(
        self,
        fps: int = 1,
        max_dimension: int = 1024,
        jpeg_quality: int = 80,
    ):
        self.fps = fps
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality
        self._camera_task = None
        self._screen_task = None
        self._camera_queue = asyncio.Queue()
        self._screen_queue = asyncio.Queue()
        self._camera_active = False
        self._screen_active = False

    async def start_camera(self):
        """Start camera capture."""
        if self._camera_active:
            return
        try:
            import cv2
            self._camera_active = True
            self._camera_task = asyncio.create_task(self._camera_loop())
        except ImportError:
            logger.warning("OpenCV not available, camera disabled")
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            raise

    async def stop_camera(self):
        """Stop camera capture."""
        self._camera_active = False
        if self._camera_task and not self._camera_task.done():
            self._camera_task.cancel()
            try:
                await self._camera_task
            except asyncio.CancelledError:
                pass

    async def start_screen(self, monitor_index: int = 0):
        """Start screen capture."""
        if self._screen_active:
            return
        try:
            import mss
            self._screen_active = True
            self._monitor_index = monitor_index
            self._screen_task = asyncio.create_task(self._screen_loop())
        except ImportError:
            logger.warning("MSS not available, screen capture disabled")
        except Exception as e:
            logger.error(f"Failed to start screen capture: {e}")
            raise

    async def stop_screen(self):
        """Stop screen capture."""
        self._screen_active = False
        if self._screen_task and not self._screen_task.done():
            self._screen_task.cancel()
            try:
                await self._screen_task
            except asyncio.CancelledError:
                pass

    async def _camera_loop(self):
        """Capture frames from camera."""
        import cv2
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        try:
            while self._camera_active:
                ret, frame = await asyncio.to_thread(cap.read)
                if not ret:
                    await asyncio.sleep(1.0 / self.fps)
                    continue

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize if needed
                h, w = frame_rgb.shape[:2]
                if max(h, w) > self.max_dimension:
                    scale = self.max_dimension / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

                # Encode to JPEG
                import PIL.Image
                img = PIL.Image.fromarray(frame_rgb)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=self.jpeg_quality)
                buf.seek(0)
                jpeg_data = buf.read()

                await self._camera_queue.put({"data": jpeg_data, "mime_type": "image/jpeg"})
                await asyncio.sleep(1.0 / self.fps)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Camera capture error: {e}")
        finally:
            cap.release()

    async def _screen_loop(self):
        """Capture screen frames."""
        import mss
        import PIL.Image
        sct = mss.mss()
        try:
            while self._screen_active:
                monitor = sct.monitors[self._monitor_index]
                screenshot = await asyncio.to_thread(sct.grab, monitor)
                img = PIL.Image.frombytes("RGB", screenshot.size, screenshot.rgb)

                # Resize if needed
                w, h = img.size
                if max(w, h) > self.max_dimension:
                    scale = self.max_dimension / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h))

                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=self.jpeg_quality)
                buf.seek(0)
                jpeg_data = buf.read()

                await self._screen_queue.put({"data": jpeg_data, "mime_type": "image/jpeg"})
                await asyncio.sleep(1.0 / self.fps)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Screen capture error: {e}")

    def get_camera_queue(self):
        return self._camera_queue

    def get_screen_queue(self):
        return self._screen_queue
