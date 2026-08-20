"""
Liaison Agent Tool Definitions
Allowed tools: web_search, read_files, execute_commands (read-only research tools)
Forbidden: Any file writing/editing tools
"""

from __future__ import annotations

from google.genai import types


WEB_SEARCH_TOOL = types.FunctionDeclaration(
    name="web_search",
    description=(
        "Perform multi-hop live web search for technical facts, documentation, "
        "API references, and current information. Returns structured results with "
        "titles, URLs, snippets, and relevance scores. Use for research, fact-checking, "
        "and discovering up-to-date technical details."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING,
                description="Search query. Be specific and technical for best results."
            ),
            "max_results": types.Schema(
                type=types.Type.INTEGER,
                description="Maximum number of results to return (default: 5, max: 20)",
                default=5,
                minimum=1,
                maximum=20,
            ),
            "recency_days": types.Schema(
                type=types.Type.INTEGER,
                description="Limit results to last N days (default: 30, 0 = no limit)",
                default=30,
                minimum=0,
                maximum=365,
            ),
            "site_filter": types.Schema(
                type=types.Type.STRING,
                description="Optional site restriction (e.g., 'github.com', 'docs.python.org')",
            ),
        },
        required=["query"],
    ),
)

READ_FILES_TOOL = types.FunctionDeclaration(
    name="read_files",
    description=(
        "Read and inspect project files, directory trees, and codebase content. "
        "Supports reading specific line ranges for large files. Use for code review, "
        "understanding architecture, finding implementations, and analyzing patterns. "
        "Returns file content with line numbers."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "path": types.Schema(
                type=types.Type.STRING,
                description="Absolute or relative path to file or directory. "
                            "Directories return tree listing with file sizes."
            ),
            "start_line": types.Schema(
                type=types.Type.INTEGER,
                description="Optional 1-based starting line (inclusive). Null = start of file.",
            ),
            "end_line": types.Schema(
                type=types.Type.INTEGER,
                description="Optional 1-based ending line (inclusive). Null = end of file.",
            ),
            "max_files": types.Schema(
                type=types.Type.INTEGER,
                description="Max files to return when reading directory (default: 50)",
                default=50,
            ),
        },
        required=["path"],
    ),
)

EXECUTE_COMMANDS_TOOL = types.FunctionDeclaration(
    name="execute_commands",
    description=(
        "Execute safe, read-only inspection terminal commands via the deterministic "
        "sandbox runner. Use for: git status, grep/ripgrep searches, file listings, "
        "process inspection, environment checks, and other non-mutating operations. "
        "Commands run with 30s timeout, 10MB output limit, in project working directory. "
        "FORBIDDEN: write operations, network calls, privilege escalation, package installs."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "command": types.Schema(
                type=types.Type.STRING,
                description="Command to execute (string) or list of args. "
                            "Examples: 'git status', 'rg -n pattern', 'ls -la', 'python -c import sys; print(sys.version)'"
            ),
            "timeout": types.Schema(
                type=types.Type.NUMBER,
                description="Timeout in seconds (default: 30, max: 300)",
                default=30,
                minimum=1,
                maximum=300,
            ),
            "cwd": types.Schema(
                type=types.Type.STRING,
                description="Working directory (default: project root)",
            ),
            "capture_output": types.Schema(
                type=types.Type.BOOLEAN,
                description="Capture stdout/stderr (default: true)",
                default=True,
            ),
        },
        required=["command"],
    ),
)

# Tool registries
VOICE_FACADE_TOOLS = [
    WEB_SEARCH_TOOL,
    READ_FILES_TOOL,
    EXECUTE_COMMANDS_TOOL,
]

REASONING_CORE_TOOLS = [
    WEB_SEARCH_TOOL,
    READ_FILES_TOOL,
    EXECUTE_COMMANDS_TOOL,
]
