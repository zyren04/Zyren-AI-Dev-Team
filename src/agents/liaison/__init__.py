"""
Liaison Agent Package - Hybrid Dual-Brain Conversational Agent
Voice Facade (gemini-3.1-flash-live-preview) + Reasoning Core (Nemotron-3-Ultra via SwitchyardRouter)
"""

from .config import LiaisonConfig, VoiceSessionConfig, ReasoningCoreConfig
from .exceptions import (
    LiaisonError,
    VoiceSessionError,
    DispatchGateError,
    ToolError,
    SessionResumptionError,
)
from .base import VoiceFacadeProtocol, ReasoningCoreProtocol, ToolHandlerProtocol
from .session import ConversationTurn, ConversationContext, SessionResumptionManager
from .controls import VoiceState, VoiceSessionConfig as ControlsVoiceSessionConfig, VoiceLifecycleController
from .dispatch_gate import PlannerDispatchGate, DispatchDecision, DispatchTrigger, ConversationStateTracker
from .liaison import LiaisonAgent
from .factory import create_liaison_agent

__all__ = [
    "LiaisonConfig",
    "VoiceSessionConfig",
    "ReasoningCoreConfig",
    "LiaisonError",
    "VoiceSessionError",
    "DispatchGateError",
    "ToolError",
    "SessionResumptionError",
    "VoiceFacadeProtocol",
    "ReasoningCoreProtocol",
    "ToolHandlerProtocol",
    "ConversationTurn",
    "ConversationContext",
    "SessionResumptionManager",
    "VoiceState",
    "VoiceLifecycleController",
    "PlannerDispatchGate",
    "DispatchDecision",
    "DispatchTrigger",
    "ConversationStateTracker",
    "LiaisonAgent",
    "create_liaison_agent",
]