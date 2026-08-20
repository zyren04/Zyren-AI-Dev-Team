"""
Tests for CLI Thinking Render
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.liaison.cli import LiaisonCLI
from src.agents.liaison.config import LiaisonConfig
from src.agents.liaison.reasoning.core import ReasoningResult


class TestCLIThinkingRender:
    """Test CLI thinking panel rendering."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with thinking support."""
        agent = MagicMock()
        agent.get_last_thought = MagicMock(return_value=None)
        agent.process_text = AsyncMock(return_value="Hello! How can I help you?")
        agent.shutdown = AsyncMock()
        return agent

    @pytest.fixture
    def cli(self, mock_agent):
        """Create CLI with mocked agent."""
        config = LiaisonConfig()
        cli = LiaisonCLI(config)
        cli.agent = mock_agent
        return cli

    @pytest.mark.asyncio
    async def test_render_thinking_panel_when_thought_exists(self, cli, mock_agent):
        """Test that thinking panel is rendered when thought exists."""
        mock_agent.get_last_thought.return_value = "User is asking about their name. Check conversation history."
        mock_agent.process_text = AsyncMock(return_value="Your name is Mohamed.")
        
        with patch("src.agents.liaison.cli.console") as mock_console:
            with patch("src.agents.liaison.cli.Prompt.ask", side_effect=["Hello", "quit"]):
                await cli.run_interactive()
            
            panel_calls = [call for call in mock_console.print.call_args_list if "Panel" in str(call)]
            assert len(panel_calls) > 0, "Thinking panel should be rendered"
            
            panel_call = panel_calls[0]
            panel = panel_call[0][0]
            assert panel.title == "🧠 Thinking Process"
            assert panel.border_style == "dim white"
            assert panel.style == "dim"
            assert "User is asking about their name" in str(panel.renderable)

    @pytest.mark.asyncio
    async def test_no_thinking_panel_when_no_thought(self, cli, mock_agent):
        mock_agent.get_last_thought.return_value = None
        mock_agent.process_text = AsyncMock(return_value="Hello! How can I help?")
        
        with patch("src.agents.liaison.cli.console") as mock_console:
            with patch("src.agents.liaison.cli.Prompt.ask", side_effect=["Hello", "quit"]):
                await cli.run_interactive()
            
            panel_calls = [call for call in mock_console.print.call_args_list if "Panel" in str(call)]
            assert len(panel_calls) == 0, "No thinking panel should be rendered when no thought"

    @pytest.mark.asyncio
    async def test_final_response_always_displayed(self, cli, mock_agent):
        mock_agent.get_last_thought.return_value = "Some thinking..."
        mock_agent.process_text = AsyncMock(return_value="Final response here.")
        
        with patch("src.agents.liaison.cli.console") as mock_console:
            with patch("src.agents.liaison.cli.Prompt.ask", side_effect=["Hello", "quit"]):
                await cli.run_interactive()
            
            printed_text = " ".join(str(call[0][0]) for call in mock_console.print.call_args_list)
            assert "Final response here" in printed_text

    @pytest.mark.asyncio
    async def test_thought_cleared_after_display(self, cli, mock_agent):
        mock_agent.get_last_thought.side_effect = ["First thought", None]
        mock_agent.process_text = AsyncMock(side_effect=["Response 1", "Response 2"])
        
        with patch("src.agents.liaison.cli.console") as mock_console:
            with patch("src.agents.liaison.cli.Prompt.ask", side_effect=["First", "Second", "quit"]):
                await cli.run_interactive()
        
        panel_calls = [call for call in mock_console.print.call_args_list if "Panel" in str(call)]
        assert len(panel_calls) == 1, "Thinking panel should only be shown once"
