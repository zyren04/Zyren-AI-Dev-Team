"""
Tests for Tool Handlers
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.liaison.tools.handlers import ToolHandler
from google.genai.types import FunctionCall, FunctionResponse


class TestToolHandler:
    @pytest.fixture
    def handler(self):
        return ToolHandler()

    @pytest.mark.asyncio
    async def test_unknown_tool(self, handler):
        call = FunctionCall(name="unknown_tool", args={}, id="test_1")
        result = await handler.handle_tool_call(call)
        assert isinstance(result, FunctionResponse)
        assert "error" in result.response
        assert "Unknown tool" in result.response["error"]

    @pytest.mark.asyncio
    async def test_read_files_file(self, handler, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3")

        call = FunctionCall(name="read_files", args={"path": str(test_file)}, id="test_1")
        result = await handler.handle_tool_call(call)

        assert result.response["path"] == str(test_file)
        assert result.response["lines"] == ["line1", "line2", "line3"]
        assert result.response["total_lines"] == 3

    @pytest.mark.asyncio
    async def test_read_files_with_line_range(self, handler, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\nline4\nline5")

        call = FunctionCall(name="read_files", args={"path": str(test_file), "start_line": 2, "end_line": 4}, id="test_1")
        result = await handler.handle_tool_call(call)

        assert result.response["lines"] == ["line2", "line3", "line4"]

    @pytest.mark.asyncio
    async def test_read_files_directory(self, handler, tmp_path):
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.txt").write_text("content3")

        call = FunctionCall(name="read_files", args={"path": str(tmp_path)}, id="test_1")
        result = await handler.handle_tool_call(call)

        assert "files" in result.response
        assert len(result.response["files"]) >= 3

    @pytest.mark.asyncio
    async def test_execute_commands(self, handler):
        call = FunctionCall(name="execute_commands", args={"command": "echo hello"}, id="test_1")
        result = await handler.handle_tool_call(call)

        assert result.response["success"]
        assert "hello" in result.response["stdout"]
        assert result.response["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_execute_commands_failure(self, handler):
        call = FunctionCall(name="execute_commands", args={"command": "exit 1"}, id="test_1")
        result = await handler.handle_tool_call(call)

        assert not result.response["success"]
        assert result.response["exit_code"] == 1

    @pytest.mark.asyncio
    async def test_web_search_placeholder(self, handler):
        call = FunctionCall(name="web_search", args={"query": "test query"}, id="test_1")
        result = await handler.handle_tool_call(call)

        assert "results" in result.response
        assert result.response["query"] == "test query"
        assert "not yet implemented" in result.response["note"]
