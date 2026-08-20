"""
Liaison Agent CLI
Command-line interface for the Liaison Agent.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .config import LiaisonConfig
from .factory import create_liaison_agent
from .controls import VoiceState

console = Console()
app = typer.Typer(name="liaison", help="Liaison Agent - Hybrid Dual-Brain Conversational Agent")


class LiaisonCLI:
    def __init__(self, config: Optional[LiaisonConfig] = None):
        self.config = config or LiaisonConfig()
        self.agent = None
        self._running = False

    async def initialize(self):
        self.agent = await create_liaison_agent(self.config)
        console.print(Panel.fit("[bold green]Liaison Agent initialized[/bold green]"))

    async def run_interactive(self):
        """Run interactive text mode."""
        self._running = True
        console.print("[bold]Liaison Agent Interactive Mode[/bold]")
        console.print("Type 'help' for commands, 'quit' to exit\n")

        while self._running:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
                if not user_input.strip():
                    continue

                if user_input.lower() in ("quit", "exit", "q"):
                    break
                elif user_input.lower() == "help":
                    self._show_help()
                elif user_input.startswith("/voice "):
                    await self._handle_voice_command(user_input[7:])
                else:
                    response = await self.agent.process_text(user_input)
                    console.print(f"[bold green]Liaison[/bold green]: {response}")

            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")

        await self.shutdown()

    async def _handle_voice_command(self, cmd: str):
        parts = cmd.split()
        if not parts:
            console.print("[yellow]Usage: /voice <start|stop|mute|unmute|camera|screen>[/yellow]")
            return

        subcmd = parts[0].lower()
        try:
            if subcmd == "start":
                result = await self.agent.start_voice()
                console.print(f"[green]{result.message}[/green]")
            elif subcmd == "stop":
                result = await self.agent.stop_voice()
                console.print(f"[green]{result.message}[/green]")
            elif subcmd == "mute":
                result = await self.agent.mute_microphone()
                console.print(f"[green]{result.message}[/green]")
            elif subcmd == "unmute":
                result = await self.agent.unmute_microphone()
                console.print(f"[green]{result.message}[/green]")
            elif subcmd == "camera":
                enabled = parts[1].lower() != "off" if len(parts) > 1 else True
                result = await self.agent.toggle_camera(enabled)
                console.print(f"[green]{result.message}[/green]")
            elif subcmd == "screen":
                enabled = parts[1].lower() != "off" if len(parts) > 1 else True
                result = await self.agent.toggle_screen_share(enabled)
                console.print(f"[green]{result.message}[/green]")
            else:
                console.print(f"[yellow]Unknown voice command: {subcmd}[/yellow]")
        except Exception as e:
            console.print(f"[red]Voice command failed: {e}[/red]")

    def _show_help(self):
        console.print("""
[bold]Available Commands:[/bold]
  /voice start          - Start voice session
  /voice stop           - Stop voice session
  /voice mute           - Mute microphone
  /voice unmute         - Unmute microphone
  /voice camera [on|off] - Toggle camera
  /voice screen [on|off] - Toggle screen share
  help                  - Show this help
  quit                  - Exit
""")

    async def shutdown(self):
        if self.agent:
            await self.agent.shutdown()
        console.print("[bold yellow]Goodbye![/bold yellow]")


@app.command()
def main(
    mode: str = typer.Option("text", "--mode", "-m", help="Mode: text, voice, camera, screen"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Run Liaison Agent CLI."""
    logging.basicConfig(level=logging.INFO)

    config = LiaisonConfig()
    if config_path:
        # Load from file if provided
        pass

    cli = LiaisonCLI(config)
    asyncio.run(cli.initialize())
    asyncio.run(cli.run_interactive())


if __name__ == "__main__":
    app()
