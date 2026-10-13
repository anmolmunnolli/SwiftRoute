import json
from datetime import datetime, timezone

from swiftroute import prompting
from swiftroute.agent import Agent
from swiftroute.db import connect, seed
from swiftroute.llm import LLMReply, ScriptedLLM
from swiftroute.tools import TOOL_NAMES, CallSession

NOW = datetime(2026, 10, 14, 15, tzinfo=timezone.utc)


def _session():
    c = connect(":memory:")
    seed(c, now=NOW)
    r = c.execute("SELECT p.tracking_id, a.postal, cu.phone FROM parcels p JOIN addresses a ON a.id = p.address_id "
                  "JOIN customers cu ON cu.id = p.customer_id WHERE p.status = 'IN_TRANSIT' LIMIT 1").fetchone()
    return CallSession(c, caller_phone=r["phone"], now=NOW), r


def names(tools):
    return [t["function"]["name"] for t in tools]


def test_tools_follow_the_call_state():
    s, r = _session()
    assert names(prompting.build(s, 'gated')[1]) == ["find_parcel", "transfer_to_human"]
    s.find_parcel(r["tracking_id"])
    assert names(prompting.build(s, 'gated')[1]) == ["verify_caller", "find_parcel", "transfer_to_human"]
    s.verify_caller(r["postal"])
    assert "reschedule_delivery" in names(prompting.build(s, 'gated')[1])
    s.other_parcel_mentioned = True
    assert names(prompting.build(s, 'gated')[1]) == ["find_parcel", "transfer_to_human"]


def test_partial_verification_offers_no_change_tools():
    s, r = _session()
    s.caller_phone = "+15550000000"
    s.find_parcel(r["tracking_id"]); s.verify_caller(r["postal"])
    assert names(prompting.build(s, 'gated')[1]) == ["get_status", "find_parcel", "transfer_to_human"]


def test_compact_prefix_is_identical_across_call_states():
    # the point of "compact": the system prompt + tool block never changes, so the
    # server's prefix cache survives state changes
    s, r = _session()
    before = json.dumps(prompting.build(s, "compact"))
    s.find_parcel(r["tracking_id"]); s.verify_caller(r["postal"])
    assert json.dumps(prompting.build(s, "compact")) == before
    assert json.dumps(prompting.build(s, "gated")) != json.dumps(prompting.build(_session()[0], "gated"))


def test_argument_hints_kept_where_names_are_guessable():
    t = {x["function"]["name"]: x for x in prompting.ALL_COMPACT}
    assert t["reschedule_delivery"]["function"]["parameters"]["properties"]["day"]["description"]


def test_compact_is_much_smaller_and_still_valid():
    s, _ = _session()
    full = len(json.dumps(prompting.build(s, "full")))
    compact = len(json.dumps(prompting.build(s, "compact")))
    gated = len(json.dumps(prompting.build(s, "gated")))
    assert gated < compact < 0.75 * full
    for t in prompting.build(s, "full")[1]:
        c = prompting._BY_NAME_COMPACT[t["function"]["name"]]
        # same arguments and required fields; only the descriptions got shorter
        assert c["function"]["parameters"].get("required") == t["function"]["parameters"].get("required")
        assert set(c["function"]["parameters"].get("properties", {})) == set(t["function"]["parameters"].get("properties", {}))
    assert set(prompting._BY_NAME_COMPACT) == TOOL_NAMES


def test_agent_sends_gated_tools_and_full_mode_sends_all():
    s, _ = _session()

    class Spy(ScriptedLLM):
        def chat(self, messages, tools=None):
            self.tools_seen = tools
            return super().chat(messages, tools)
    llm = Spy([LLMReply("What's your tracking number?")])
    Agent(llm, s, prompt_mode="gated").respond("hi")
    assert names(llm.tools_seen) == ["find_parcel", "transfer_to_human"]
    llm = Spy([LLMReply("What's your tracking number?")])
    Agent(llm, s, prompt_mode="full").respond("hi")
    assert len(llm.tools_seen) == len(TOOL_NAMES)
