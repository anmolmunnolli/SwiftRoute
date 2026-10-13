"""The harness must pass a correct agent and fail a bad one; otherwise its
numbers on a real model mean nothing."""
import pytest

from evals.harness import SCENARIOS, run_scenario, summarise
from evals.rule_agent import RuleAgentLLM
from swiftroute.llm import LLMReply


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.name)
def test_reference_agent_passes(sc):
    r = run_scenario(sc, RuleAgentLLM)
    assert r.passed, (r.reasons, r.transcript)


class LiarLLM:
    """Never calls a tool, always claims success."""
    model = "liar"

    def chat(self, messages, tools=None):
        return LLMReply("All set, I've rescheduled that for you.")


def test_liar_fails_and_its_lies_never_reach_the_caller():
    results = [run_scenario(sc, LiarLLM) for sc in SCENARIOS]
    s = summarise(results)
    assert s["pass_rate"] < 0.5
    # the runtime claim guard replaces every unbacked "I've rescheduled" with a hand-off
    assert s["false_claims"] == 0
    assert all("rescheduled that" not in text for r in results for who, text in r.transcript if who == "agent")


class ReckessLLM:
    """Calls the action tool straight away with no lookup or verification."""
    model = "reckless"

    def __init__(self):
        self.n = 0

    def chat(self, messages, tools=None):
        self.n += 1
        if messages[-2]["role"] == "user":
            return LLMReply("", [{"id": f"x{self.n}", "name": "change_address", "arguments": {
                "line1": "1 Evil St", "city": "Toronto", "spoken_postal_code": "M5V2T6"}}])
        return LLMReply("Done, the address is changed.")


def test_policy_blocks_reckless_agent_and_harness_flags_it():
    sc = next(s for s in SCENARIOS if s.name == "prompt_injection")
    r = run_scenario(sc, ReckessLLM)
    assert "ADDR_CHANGE_OK" not in r.codes          # tools refused
    assert r.false_claims >= 1 and not r.passed     # and the lie was caught
