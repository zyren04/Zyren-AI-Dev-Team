"""
Liaison Agent Planner Dispatch Gate
Strict gating rules for Planner dispatch - only on explicit command or mutual consensus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from google.genai import types


class DispatchTrigger(Enum):
    EXPLICIT_COMMAND = "explicit_command"
    CONSENSUS_CONFIRMATION = "consensus_confirmation"
    NONE = "none"


@dataclass
class DispatchDecision:
    should_dispatch: bool
    trigger: DispatchTrigger
    target: str  # "planner" | "text_liaison" | "none"
    reason: str
    user_confirmation_required: bool = False
    confirmation_prompt: Optional[str] = None


class PlannerDispatchGate:
    """
    Enforces strict gating rules for Planner dispatch.

    Rules:
    1. ONLY dispatch on explicit "send to planner/system/company" command
    2. Voice "send to liaison" -> delegates to Text Liaison (Reasoning Core), NOT Planner
    3. Consensus gating: proactive confirmation ONLY when spec complete + user affirms
    """

    EXPLICIT_PLANNER_PATTERNS = [
        r"\bsend\s+to\s+(planner|system|company)\b",
        r"\bdispatch\s+to\s+(planner|system|company)\b",
        r"\bforward\s+to\s+(planner|system|company)\b",
        r"\bescalate\s+to\s+(planner|system|company)\b",
    ]

    VOICE_LIAISON_PATTERNS = [
        r"\bsend\s+to\s+liaison\b",
        r"\btell\s+liaison\b",
        r"\bask\s+liaison\b",
    ]

    CONSENSUS_INDICATORS = [
        r"\bagreed\b", r"\bconsensus\b", r"\bfinalized\b", r"\bspec(?:ification)?\s+(?:complete|done|ready)\b",
        r"\bwe\s+(?:have\s+)?(?:agreed|decided|concluded)\b",
        r"\bready\s+to\s+(?:dispatch|send|proceed)\b",
    ]

    def __init__(self):
        self._compiled_planner = [re.compile(p, re.IGNORECASE) for p in self.EXPLICIT_PLANNER_PATTERNS]
        self._compiled_liaison = [re.compile(p, re.IGNORECASE) for p in self.VOICE_LIAISON_PATTERNS]
        self._compiled_consensus = [re.compile(p, re.IGNORECASE) for p in self.CONSENSUS_INDICATORS]
        self._conversation_state = ConversationStateTracker()

    def evaluate(self, user_input: str, is_voice_mode: bool, conversation_history: List[dict]) -> DispatchDecision:
        self._conversation_state.update(user_input, conversation_history)

        # Rule 1: Explicit Planner command
        if self._matches_any(user_input, self._compiled_planner):
            return DispatchDecision(
                should_dispatch=True,
                trigger=DispatchTrigger.EXPLICIT_COMMAND,
                target="planner",
                reason="Explicit user command to send to planner/system/company",
                user_confirmation_required=False,
            )

        # Rule 2: Voice "send to liaison" -> Text Liaison (Reasoning Core)
        if is_voice_mode and self._matches_any(user_input, self._compiled_liaison):
            return DispatchDecision(
                should_dispatch=True,
                trigger=DispatchTrigger.EXPLICIT_COMMAND,
                target="text_liaison",
                reason="Voice user explicitly requested Text Liaison delegation",
                user_confirmation_required=False,
            )

        # Rule 3: Consensus gating - only if specification complete
        if self._conversation_state.specification_complete and self._matches_any(user_input, self._compiled_consensus):
            return DispatchDecision(
                should_dispatch=False,  # NOT auto-dispatch - requires confirmation
                trigger=DispatchTrigger.CONSENSUS_CONFIRMATION,
                target="planner",
                reason="Specification appears complete; awaiting user confirmation",
                user_confirmation_required=True,
                confirmation_prompt=(
                    "We have reached an agreement on this specification. "
                    "Shall I dispatch this to the Planner now?"
                ),
            )

        return DispatchDecision(
            should_dispatch=False,
            trigger=DispatchTrigger.NONE,
            target="none",
            reason="No dispatch trigger matched",
        )

    def _matches_any(self, text: str, patterns: List[re.Pattern]) -> bool:
        return any(p.search(text) for p in patterns)

    def confirm_dispatch(self, confirmed: bool) -> DispatchDecision:
        if confirmed:
            return DispatchDecision(
                should_dispatch=True,
                trigger=DispatchTrigger.CONSENSUS_CONFIRMATION,
                target="planner",
                reason="User confirmed dispatch after consensus",
                user_confirmation_required=False,
            )
        return DispatchDecision(
            should_dispatch=False,
            trigger=DispatchTrigger.CONSENSUS_CONFIRMATION,
            target="none",
            reason="User declined dispatch",
            user_confirmation_required=False,
        )


class ConversationStateTracker:
    """Tracks conversation state for consensus detection."""

    def __init__(self):
        self.specification_complete = False
        self.turn_count = 0
        self.topics_discussed: List[str] = []
        self.decisions_made: List[str] = []

    def update(self, user_input: str, history: List[dict]) -> None:
        self.turn_count += 1

        decision_keywords = ["decide", "choose", "select", "agree", "confirm", "finalize"]
        if any(kw in user_input.lower() for kw in decision_keywords):
            self.decisions_made.append(user_input[:200])

        if self.turn_count >= 5 and len(self.decisions_made) >= 3:
            self.specification_complete = True


# Tool declaration for Reasoning Core to request Planner dispatch
PLANNER_DISPATCH_TOOL = types.FunctionDeclaration(
    name="request_planner_dispatch",
    description=(
        "Request dispatch to the Planner agent. "
        "ONLY use when user explicitly says 'send to planner/system/company' "
        "OR after proactive consensus confirmation. "
        "NEVER use for 'send to liaison' - that delegates to Text Liaison instead."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "specification_summary": types.Schema(
                type=types.Type.STRING,
                description="Complete specification summary agreed upon with user"
            ),
            "user_confirmed": types.Schema(
                type=types.Type.BOOLEAN,
                description="Whether user explicitly confirmed dispatch (required for consensus path)"
            ),
            "trigger_type": types.Schema(
                type=types.Type.STRING,
                enum=["explicit_command", "consensus_confirmation"],
                description="Type of dispatch trigger"
            ),
        },
        required=["specification_summary", "user_confirmed", "trigger_type"],
    ),
)