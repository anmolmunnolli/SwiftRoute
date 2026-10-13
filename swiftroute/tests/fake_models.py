"""Fake phone-agent models for the learning tests (no LLM server needed)."""
from evals.rule_agent import RuleAgentLLM
from swiftroute.llm import LLMReply


class LessonSensitiveLLM(RuleAgentLLM):
    """Behaves like the reference agent, except it never calls hold_at_depot on its own:
    the kind of gap a small model has. A lesson telling it to fixes that."""
    model = "fake-lesson-sensitive"

    def chat(self, messages, tools=None):
        reply = super().chat(messages, tools)
        knows = "call hold_at_depot right after" in messages[0]["content"]
        if not knows and any(c["name"] == "hold_at_depot" for c in reply.tool_calls):
            return LLMReply("Okay. Is there anything else I can help with?")
        return reply
