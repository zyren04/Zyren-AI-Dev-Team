"""
Tests for Planner Dispatch Gate
"""

import pytest

from src.agents.liaison.dispatch_gate import PlannerDispatchGate, DispatchDecision, DispatchTrigger


class TestPlannerDispatchGate:
    @pytest.fixture
    def gate(self):
        return PlannerDispatchGate()

    def test_explicit_planner_command(self, gate):
        decision = gate.evaluate("send to planner", is_voice_mode=False, conversation_history=[])
        assert decision.should_dispatch
        assert decision.trigger == DispatchTrigger.EXPLICIT_COMMAND
        assert decision.target == "planner"
        assert not decision.user_confirmation_required

    def test_explicit_planner_command_variations(self, gate):
        for cmd in ["dispatch to planner", "forward to system", "escalate to company"]:
            decision = gate.evaluate(cmd, is_voice_mode=False, conversation_history=[])
            assert decision.should_dispatch
            assert decision.target == "planner"

    def test_voice_send_to_liaison_delegates_to_text_liaison(self, gate):
        decision = gate.evaluate("send to liaison", is_voice_mode=True, conversation_history=[])
        assert decision.should_dispatch
        assert decision.target == "text_liaison"
        assert not decision.user_confirmation_required

    def test_voice_send_to_liaison_not_in_text_mode(self, gate):
        decision = gate.evaluate("send to liaison", is_voice_mode=False, conversation_history=[])
        assert not decision.should_dispatch
        assert decision.target == "none"

    def test_consensus_gating_requires_confirmation(self, gate):
        # Build up conversation state to trigger consensus
        history = []
        for i in range(6):
            gate._conversation_state.update(f"We decide to use option {i}", history)
        
        decision = gate.evaluate("we have agreed on the specification", is_voice_mode=False, conversation_history=history)
        assert not decision.should_dispatch  # Requires confirmation
        assert decision.trigger == DispatchTrigger.CONSENSUS_CONFIRMATION
        assert decision.user_confirmation_required
        assert decision.confirmation_prompt is not None

    def test_confirm_dispatch(self, gate):
        decision = gate.confirm_dispatch(True)
        assert decision.should_dispatch
        assert decision.target == "planner"

        decision = gate.confirm_dispatch(False)
        assert not decision.should_dispatch
        assert decision.target == "none"

    def test_no_trigger_for_casual_conversation(self, gate):
        decision = gate.evaluate("hello how are you", is_voice_mode=False, conversation_history=[])
        assert not decision.should_dispatch
        assert decision.trigger == DispatchTrigger.NONE
