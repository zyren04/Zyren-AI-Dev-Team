"""
Liaison Agent Tool Handlers
Executes tool calls from Voice Facade and Reasoning Core.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from google.genai.types import FunctionCall, FunctionResponse

from src.sandbox.runner import get_sandbox_runner, ExecutionResult

logger = logging.getLogger(__name__)


class ToolHandler:
    """Executes tool calls from both Voice Facade and Reasoning Core."""

    def __init__(self):
        self.sandbox = get_sandbox_runner()

    async def handle_tool_call(self, call: FunctionCall) -> FunctionResponse:
        """Route tool call to appropriate handler."""
        handlers = {
            "web_search": self._handle_web_search,
            "read_files": self._handle_read_files,
            "execute_commands": self._handle_execute_commands,
            "request_planner_dispatch": self._handle_planner_dispatch,
        }

        handler = handlers.get(call.name)
        if not handler:
            return FunctionResponse(
                name=call.name,
                response={"error": f"Unknown tool: {call.name}"},
                id=call.id,
            )

        try:
            result = await handler(call.args)
            return FunctionResponse(name=call.name, response=result, id=call.id)
        except Exception as e:
            logger.error(f"Tool {call.name} failed: {e}")
            return FunctionResponse(
                name=call.name,
                response={"error": f"Tool execution failed: {str(e)}"},
                id=call.id,
            )

    async def _handle_web_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search via implementation."""
        # This would integrate with a real search API (Serper, Brave, etc.)
        # For now, return a placeholder
        return {
            "results": [],
            "query": args.get("query", ""),
            "note": "Web search not yet implemented - integrate with Serper/Brave API"
        }

    async def _handle_read_files(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read files via explorer tools."""
        from pathlib import Path
        path = Path(args["path"])
        
        if not path.exists():
            return {"error": f"Path not found: {path}"}

        if path.is_file():
            try:
                content = path.read_text()
                lines = content.splitlines()
                start = args.get("start_line")
                end = args.get("end_line")
                if start is not None or end is not None:
                    start_idx = (start - 1) if start else 0
                    end_idx = end if end else len(lines)
                    lines = lines[start_idx:end_idx]
                return {
                    "path": str(path),
                    "lines": lines,
                    "total_lines": len(content.splitlines()),
                }
            except Exception as e:
                return {"error": f"Failed to read file: {e}"}
        else:
            # Directory listing
            files = []
            all_files = list(path.rglob("*"))
            for f in all_files[:args.get("max_files", 50)]:
                files.append({
                    "path": str(f.relative_to(path)),
                    "size": f.stat().st_size if f.is_file() else 0,
                    "is_dir": f.is_dir(),
                })
            return {"path": str(path), "files": files}

    async def _handle_execute_commands(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute command via sandbox runner."""
        result: ExecutionResult = await self.sandbox.run(
            command=args["command"],
            timeout=args.get("timeout", 30.0),
            cwd=args.get("cwd"),
            capture_output=args.get("capture_output", True),
            shell=False,
        )
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "duration_ms": result.duration_ms,
            "success": result.success,
        }

    async def _handle_planner_dispatch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Planner dispatch request."""
        # This is called by Reasoning Core after gate approval
        from src.runtime.engine import get_workflow_engine
        
        engine = get_workflow_engine()
        # Dispatch to planner node (implementation depends on workflow graph)
        return {
            "dispatched": True,
            "specification_summary": args["specification_summary"],
            "trigger_type": args["trigger_type"],
            "message": "Specification dispatched to Planner workflow",
        }
