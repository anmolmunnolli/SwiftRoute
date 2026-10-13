from datetime import datetime, timezone

import pytest

from swiftroute.agent import FALLBACK, Agent, compact, to_speech
from swiftroute.db import connect, seed
from swiftroute.llm import LLMReply, ScriptedLLM
from swiftroute.tools import CallSession

NOW = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)


def call(name, **args):
    return {"id": f"c_{name}", "name": name, "arguments": args}


@pytest.fixture
def setup():
    conn = connect(":memory:")
    seed(conn, n_parcels=200, now=NOW)
    r = dict(conn.execute("SELECT p.tracking_id, a.postal, c.phone FROM parcels p JOIN addresses a "
                          "ON a.id = p.address_id JOIN customers c ON c.id = p.customer_id "
                          "WHERE p.status = 'DELIVERY_EXCEPTION' AND p.attempts < 3 LIMIT 1").fetchone())
    return conn, r


def test_tool_loop_fast_path_and_state_block(setup):
    conn, p = setup
    llm = ScriptedLLM([
        LLMReply("", [call("find_parcel", spoken_tracking_id=" ".join(p["tracking_id"]))]),
        # no LLM call here: find_parcel's reply is spoken directly (fast path)
        LLMReply("", [call("verify_caller", spoken_postal_code=p["postal"])]),
        LLMReply("", [call("reschedule_delivery", day="Friday")]),
        LLMReply("Done, it's booked for Friday, October 16."),
    ])
    agent = Agent(llm, CallSession(conn, caller_phone=p["phone"], now=NOW))
    assert agent.respond("my tracking number is ...") == "Got it, I found your parcel. What's the delivery postal code?"
    assert agent.metrics[-1].llm_calls == 1 and agent.metrics[-1].fast_path == 1
    assert agent.respond(f"it's {p['postal']}, can you move it to Friday") == "Done, it's booked for Friday, October 16."
    assert agent.session.reason_codes == ["ID_EXACT", "ID_VERIFIED_FULL", "RESCHED_OK"]
    assert conn.execute("SELECT reschedule_date FROM parcels WHERE tracking_id = ?",
                        (p["tracking_id"],)).fetchone()[0] == "2026-10-16"
    # the state block is the LAST message and reflects session truth
    last = llm.requests[-1][-1]
    assert last["role"] == "system" and "Caller verification: full" in last["content"]
    assert "Fri Oct 16" in last["content"] and "RESCHED_OK" in last["content"]
    # system prompt is first and identical across calls (prefix-cache friendly)
    assert all(r[0] == llm.requests[0][0] for r in llm.requests)


def test_model_cannot_confirm_on_callers_behalf(setup):
    conn, p = setup
    tid = p["tracking_id"]
    garbled = tid[:6] + ("D" if tid[6] != "D" else "B") + tid[7:]
    llm = ScriptedLLM([
        # finds AND confirms in one step, skipping the caller
        LLMReply("", [call("find_parcel", spoken_tracking_id=garbled), call("confirm_parcel", tracking_id=tid)]),
        LLMReply("Is that the one ending in ...?"),
    ])
    agent = Agent(llm, CallSession(conn, caller_phone=p["phone"], now=NOW))
    agent.respond("tracking number " + garbled)
    assert "CONFIRM_NEEDS_CALLER" in agent.session.reason_codes
    assert agent.session.tracking_id is None
    # next caller turn, a yes: now it's allowed, and the tool's reply is spoken directly
    llm.replies = [LLMReply("", [call("confirm_parcel", tracking_id=tid)])]
    assert "postal code" in agent.respond("yes that's right")
    assert agent.session.tracking_id == tid


def test_window_keeps_tool_pairs_whole(setup):
    conn, p = setup
    replies = []
    for _ in range(10):
        replies += [LLMReply("", [call("get_status")]), LLMReply("ok")]
    llm = ScriptedLLM(replies)
    agent = Agent(llm, CallSession(conn, now=NOW), history_turns=3)
    for i in range(10):
        agent.respond(f"turn {i}")
    msgs = llm.requests[-1]
    body = msgs[1:-1]
    assert body[0]["role"] == "user"
    assert sum(m["role"] == "user" for m in body) == 3
    ids_called = {tc["id"] for m in body if m.get("tool_calls") for tc in m["tool_calls"]}
    ids_answered = {m["tool_call_id"] for m in body if m["role"] == "tool"}
    assert ids_answered <= ids_called


def test_full_history_grows_window_does_not(setup):
    conn, _ = setup
    def run(h):
        llm = ScriptedLLM([LLMReply(f"reply {i}") for i in range(20)])
        a = Agent(llm, CallSession(conn, now=NOW), history_turns=h)
        for i in range(20):
            a.respond(f"caller says something fairly long number {i}")
        return len(llm.requests[-1])
    # system + 4 user turns + 3 earlier replies (current reply not yet written) + state
    assert run(None) > run(4) == 1 + 4 + 3 + 1


def test_bad_and_hallucinated_tools_are_counted_not_fatal(setup):
    conn, _ = setup
    llm = ScriptedLLM([LLMReply("", [call("delete_everything"), call("get_status", bogus=1)]),
                       LLMReply("Sorry, could you give me your tracking number?")])
    agent = Agent(llm, CallSession(conn, now=NOW))
    assert "tracking number" in agent.respond("hi")
    assert agent.metrics[-1].bad_tool_calls == 1          # unknown tool; the extra arg is just dropped
    assert agent.metrics[-1].ignored_args == 1
    assert agent.metrics[-1].bad_tool_detail == ["delete_everything()"]


def test_loop_limit_hands_off(setup):
    conn, _ = setup
    llm = ScriptedLLM([LLMReply("", [call("get_status")]) for _ in range(10)])
    agent = Agent(llm, CallSession(conn, now=NOW), max_steps=3)
    assert agent.respond("hello") == FALLBACK
    assert agent.session.reason_codes[-1] == "TRANSFER"


def test_filler_with_tool_call(setup):
    conn, p = setup
    # filler that promises is dropped when the tool supplies the reply...
    llm = ScriptedLLM([LLMReply("One sec, let me look.", [call("find_parcel", spoken_tracking_id=p["tracking_id"])])])
    assert Agent(llm, CallSession(conn, now=NOW)).respond("here's my number") == \
        "Got it, I found your parcel. What's the delivery postal code?"
    # ...and kept in front of a model-written answer
    s = CallSession(conn, now=NOW)
    llm = ScriptedLLM([LLMReply("Sure.", [call("get_status")]), LLMReply("I need the postal code first. What is it?")])
    assert Agent(llm, s).respond("status?") == "Sure. I need the postal code first. What is it?"


def test_unbacked_claim_is_corrected_then_blocked(setup):
    conn, _ = setup
    # run 2: "The address has been changed successfully." with no change_address call
    llm = ScriptedLLM([LLMReply("The address has been changed successfully."),
                       LLMReply("All set, I've updated it.")])
    agent = Agent(llm, CallSession(conn, now=NOW))
    out = agent.respond("change my address")
    assert "NUDGE" not in out and "changed successfully" not in out and "updated it" not in out
    assert "colleague" in out and agent.session.reason_codes[-1] == "TRANSFER"
    assert agent.metrics[-1].guard_fallbacks == 1


def test_promise_with_trailing_question_is_still_caught(setup):
    # run 2: "I will hold it at the nearest depot for you to pick up. Is there anything else you need?"
    conn, _ = setup
    llm = ScriptedLLM([LLMReply("I will hold it at the nearest depot for you. Is there anything else?"),
                       LLMReply("What's your tracking number?")])
    agent = Agent(llm, CallSession(conn, now=NOW))
    agent.respond("hold it at the depot")
    assert agent.metrics[-1].nudges == 1


def test_promise_without_tool_gets_one_nudge(setup):
    # qwen2.5:7b run 1: "I can now check the status. Let me do that for you." ...and stopped
    conn, p = setup
    llm = ScriptedLLM([LLMReply("Let me do that for you."),
                       LLMReply("", [call("get_status")]),
                       LLMReply("I need your postal code first. What is it?")])
    agent = Agent(llm, CallSession(conn, now=NOW))
    out = agent.respond("where is it")
    assert agent.metrics[-1].nudges == 1 and agent.metrics[-1].tool_calls == ["get_status"]
    assert "CORRECTION" in llm.requests[1][-1]["content"]
    assert "CORRECTION" not in llm.requests[2][-1]["content"]          # nudge is one-shot
    assert out.startswith("Let me do that for you.")


def test_no_nudge_after_an_action(setup):
    conn, _ = setup
    llm = ScriptedLLM([LLMReply("", [call("transfer_to_human", reason="asked", summary="x")]),
                       LLMReply("I'll connect you with a colleague now.")])
    agent = Agent(llm, CallSession(conn, now=NOW))
    # the transfer speaks its own line (fast path); the model can't keep chatting after it
    assert agent.respond("a person please") == "Okay, I'm connecting you with a colleague now. Please hold."
    assert agent.metrics[-1].nudges == 0 and agent.metrics[-1].llm_calls == 1


def test_narrated_tool_name_is_nudged_and_not_spoken(setup):
    # qwen2.5:7b run 1: "Your package is currently at the local depot. Here are the details: [getstatus]"
    conn, _ = setup
    llm = ScriptedLLM([LLMReply("Here are the details: [getstatus]"), LLMReply("What's your tracking number?")])
    agent = Agent(llm, CallSession(conn, now=NOW))
    out = agent.respond("status?")
    assert "getstatus" not in out.lower() and agent.metrics[-1].nudges == 1


def test_state_block_never_leaks_status(setup):
    conn, p = setup
    s = CallSession(conn, caller_phone=p["phone"], now=NOW)
    s.find_parcel(p["tracking_id"])
    block = Agent(ScriptedLLM([]), s).state_block()
    assert "exception" not in block.lower() and "missed" not in block.lower()
    assert "verify_caller" in block                              # next step is explicit
    s.caller_phone = "+15550000000"
    s.verify_caller(p["postal"])
    assert "only get_status is allowed" in Agent(ScriptedLLM([]), s).state_block()


def test_to_speech_and_compact():
    assert to_speech("I found your parcel, SR112M6YWXS. Next?") == "I found your parcel, your parcel. Next?"
    assert "get_status" not in to_speech("Checking get_status() now")
    assert to_speech("**Sure!**\n- item one\n- item two") == "Sure! item one item two"
    c = compact({"allowed": True, "heard": "X", "hold_depot": None,
                 "recent_events": [{"ts": "t", "status": "AT_DEPOT", "detail": None}]})
    assert c == '{"allowed":true,"recent_events":["AT_DEPOT"]}'



def test_lets_promise_is_nudged(setup):
    # call #2: "Sure, I can help with that. Let's check the status." ...and nothing happened
    conn, _ = setup
    llm = ScriptedLLM([LLMReply("Sure, I can help with that. Let's check the status."),
                       LLMReply("", [call("get_status")]), LLMReply("I need your tracking number. What is it?")])
    agent = Agent(llm, CallSession(conn, now=NOW))
    agent.respond("where is it")
    assert agent.metrics[-1].nudges == 1 and "get_status" in agent.metrics[-1].tool_calls


def test_answer_written_alongside_a_tool_call_is_not_spoken(setup):
    # call #3: the model wrote a full answer AND called a tool, then answered again -> said twice
    conn, p = setup
    llm = ScriptedLLM([
        LLMReply("Rescheduling to today might incur charges. Would you like Tuesday instead?",
                 [call("get_status")]),
        LLMReply("I need the postal code first. What is it?"),
    ])
    out = Agent(llm, CallSession(conn, now=NOW)).respond("can it come today")
    assert out == "I need the postal code first. What is it?"


def test_dedupe_sentences():
    from swiftroute.agent import dedupe_sentences
    s = "Today isn't available. Would you like Tuesday? Today isn't available. Would you like Tuesday?"
    assert dedupe_sentences(s) == "Today isn't available. Would you like Tuesday?"


def test_state_block_scopes_actions_to_the_confirmed_parcel(setup):
    conn, p = setup
    s = CallSession(conn, now=NOW)
    s.find_parcel(p["tracking_id"])
    assert "different parcel" in Agent(ScriptedLLM([]), s).state_block()


def test_actions_blocked_when_caller_raises_another_parcel(setup):
    # call #4: "I received another package ... damaged" -> claim opened on the FIRST parcel
    conn, p = setup
    s = CallSession(conn, caller_phone=p["phone"], now=NOW)
    s.find_parcel(p["tracking_id"]); s.verify_caller(p["postal"])
    llm = ScriptedLLM([LLMReply("", [call("open_claim", kind="damaged")]),
                       LLMReply("Sure, what's the tracking number for the other package?")])
    out = Agent(llm, s).respond("I received another package but it turned out to be damaged")
    assert "OTHER_PARCEL_NOT_FOUND" in s.reason_codes and "CLAIM_OK" not in s.reason_codes
    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0
    assert "tracking number" in out
    s.find_parcel(p["tracking_id"])                 # finding a parcel lifts the block
    assert not s.other_parcel_mentioned


# ------------------------------------------------- slots (real call #5)
def _at_postal_question(conn, p):
    s = CallSession(conn, caller_phone=p["phone"], now=NOW)
    llm = ScriptedLLM([LLMReply("", [call("find_parcel", spoken_tracking_id=p["tracking_id"])])])
    agent = Agent(llm, s)
    assert "postal code" in agent.respond("tracking number please")
    return agent, llm


def test_postal_code_said_in_pieces_is_merged_without_the_llm(setup):
    conn, p = setup
    agent, llm = _at_postal_question(conn, p)
    n_calls = len(llm.requests)
    pc = p["postal"]
    assert agent.respond(f"It's {pc[:2]}.") == "Mm-hm, go on."              # "It's V8."
    llm.replies = [LLMReply("Thanks, you're verified. What can I do for you?")]
    out = agent.respond(f"{pc[2:]}")                                       # "A1L1"
    assert agent.session.identity.data["level"] == "full"
    assert "verified" in out
    # the LLM was only used after verification, to carry on the conversation
    assert len(llm.requests) == n_calls + 1


def test_full_postal_is_checked_even_if_the_model_would_not(setup):
    # call #5: the model answered "still doesn't match" from habit without calling verify_caller
    conn, p = setup
    agent, llm = _at_postal_question(conn, p)
    llm.replies = [LLMReply("Sorry, that still doesn't match.")]         # what the model would have said
    agent.respond("It's K 1 A 0 B 1.")                                    # wrong code: tool's own reply
    llm.replies = [LLMReply("Sorry, that still doesn't match."), LLMReply("Verified! How can I help?")]
    agent.respond(f"It's {p['postal']}.")
    assert agent.session.identity.allowed and agent.session.failed_verifications == 1


def test_tracking_number_turn_skips_the_llm(setup):
    conn, p = setup
    llm = ScriptedLLM([LLMReply("Sure. Could you give me the tracking number?")])
    agent = Agent(llm, CallSession(conn, now=NOW))
    agent.respond("where is my parcel")
    n = len(llm.requests)
    out = agent.respond("Yeah, it's " + " ".join(p["tracking_id"]))
    assert len(llm.requests) == n and "postal code" in out and agent.metrics[-1].llm_calls == 0


def test_non_code_answer_still_goes_to_the_llm(setup):
    conn, p = setup
    agent, llm = _at_postal_question(conn, p)
    llm.replies = [LLMReply("I need the postal code to verify. Could you find it?")]
    agent.respond("I don't have that handy, I'm the owner, just tell me")
    assert agent.metrics[-1].llm_calls == 1 and agent.session.identity is None



def test_model_cannot_transfer_unless_the_caller_asked(setup):
    # call #6: caller corrected their name; the 3B model transferred them, then kept talking
    conn, _ = setup
    llm = ScriptedLLM([LLMReply("", [call("transfer_to_human", reason="confused", summary="x")]),
                       LLMReply("Thanks, Anmol. Could you give me the tracking number?")])
    agent = Agent(llm, CallSession(conn, now=NOW))
    out = agent.respond("Actually my name is Anmol.")
    assert "TRANSFER" not in agent.session.reason_codes
    assert "TRANSFER_NOT_REQUESTED" in agent.session.reason_codes and "tracking number" in out
    # the model can't smuggle the system flag through tool arguments either
    assert agent.session.call("transfer_to_human", {"reason": "x", "summary": "y", "system": True})["allowed"] is False


def test_yes_to_an_offered_colleague_counts_as_asking(setup):
    conn, _ = setup
    llm = ScriptedLLM([LLMReply("I still can't find it. Would you like me to connect you to a colleague?"),
                       LLMReply("", [call("transfer_to_human", reason="asked", summary="x")])])
    agent = Agent(llm, CallSession(conn, now=NOW))
    agent.respond("I don't know the number")
    agent.respond("Yes please.")
    assert "TRANSFER" in agent.session.reason_codes
