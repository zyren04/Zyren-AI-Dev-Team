"""
Tests for Voice Lifecycle Controller
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.liaison.controls import VoiceLifecycleController, VoiceState, VoiceSessionResult
from src.agents.liaison.config import VoiceSessionConfig
from src.agents.liaison.exceptions import VoiceSessionError


class TestVoiceLifecycleController:
    @pytest.fixture
    def config(self):
        return VoiceSessionConfig()

    @pytest.fixture
    def controller(self, config):
        return VoiceLifecycleController(config)

    def test_initial_state(self, controller):
        assert controller.state == VoiceState.IDLE
        assert not controller.is_active

    @pytest.mark.asyncio
    async def test_start_voice_session_success(self, controller):
        with patch("google.genai.Client") as mock_client_class:
            mock_client = AsyncMock()
            mock_session = AsyncMock()
            mock_client.aio.live.connect = AsyncMock(return_value=mock_session)
            mock_client_class.return_value = mock_client

            result = await controller.start_voice_session()

            assert result.success
            assert result.session_id is not None
            assert controller.state == VoiceState.ACTIVE
            assert controller.is_active

    @pytest.mark.asyncio
    async def test_start_voice_session_fails_if_not_idle(self, controller):
        controller.state = VoiceState.ACTIVE
        with pytest.raises(VoiceSessionError, match="Cannot start from state"):
            await controller.start_voice_session()

    @pytest.mark.asyncio
    async def test_stop_voice_session(self, controller):
        controller.state = VoiceState.ACTIVE
        mock_session = AsyncMock()
        controller._session = mock_session
        controller._session_id = "test_session"

        result = await controller.stop_voice_session()

        assert result.success
        assert controller.state == VoiceState.IDLE
        assert not controller.is_active
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_mute_unmute_microphone(self, controller):
        controller.state = VoiceState.ACTIVE
        controller._session = AsyncMock()

        # Mute
        result = await controller.mute_microphone()
        assert result.success
        assert controller.state == VoiceState.MUTED
        controller._session.send_realtime_input.assert_called_with(audio=None)

        # Unmute
        result = await controller.unmute_microphone()
        assert result.success
        assert controller.state == VoiceState.ACTIVE

    @pytest.mark.asyncio
    async def test_mute_fails_if_not_active(self, controller):
        controller.state = VoiceState.IDLE
        with pytest.raises(VoiceSessionError, match="Can only mute in ACTIVE state"):
            await controller.mute_microphone()

    @pytest.mark.asyncio
    async def test_toggle_camera(self, controller):
        controller.state = VoiceState.ACTIVE

        result = await controller.toggle_camera(False)
        assert result.success
        assert controller.state == VoiceState.CAMERA_OFF

        result = await controller.toggle_camera(True)
        assert result.success
        assert controller.state == VoiceState.ACTIVE

    @pytest.mark.asyncio
    async def test_toggle_screen_share(self, controller):
        controller.state = VoiceState.ACTIVE

        result = await controller.toggle_screen_share(False)
        assert result.success
        assert controller.state == VoiceState.SCREEN_OFF

        result = await controller.toggle_screen_share(True)
        assert result.success
        assert controller.state == VoiceState.ACTIVE
