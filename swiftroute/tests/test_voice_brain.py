"""Frame-level tests for the voice brain, using Pipecat's own test runner.
No audio models needed: STT output is simulated with TranscriptionFrames."""
import time
from datetime import datetime, timezone

import pytest

pipecat = pytest.importorskip("pipecat")

from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test  # noqa: E402

from swiftroute.agent import Agent  # noqa: E402
from swiftroute.db import connect, seed  # noqa: E402
from swiftroute.llm import LLMReply, ScriptedLLM  # noqa: E402
from swiftroute.tools import CallSession  # noqa: E402
from swiftroute.voice.brain import SwiftRouteBrain, sentences  # noqa: E402

NOW = datetime(2026, 10, 14, 15, tzinfo=timezone.utc)


class SlowLLM(ScriptedLLM):
    def __init__(self, replies, delay):
        super().__init__(replies)
        self.delay = delay

    def chat(self, messages, tools=None):
        time.sleep(self.delay)
        return super().chat(messages, tools)


def make_agent(llm):
    conn = connect(":memory:")
    seed(conn, n_parcels=50, now=NOW)
    return Agent(llm, CallSession(conn, now=NOW))


def tr(text):
    return TranscriptionFrame(text=text, user_id="caller", timestamp="")


def spoken(frames):
    return [f.text for f in frames if isinstance(f, TTSSpeakFrame)]


@pytest.mark.asyncio
async def test_turn_produces_sentence_by_sentence_speech(tmp_path):
    agent = make_agent(ScriptedLLM([LLMReply("Sure. What's your tracking number?")]))
    brain = SwiftRouteBrain(agent, latency_log=tmp_path / "lat.jsonl")
    down, _ = await run_test(brain, frames_to_send=[
        UserStartedSpeakingFrame(), tr("where's my parcel"), VADUserStoppedSpeakingFrame(),
        UserStoppedSpeakingFrame(), SleepFrame(sleep=0.5)])
    assert spoken(down) == ["Sure.", "What's your tracking number?"]
    assert not any(isinstance(f, TranscriptionFrame) for f in down)   # never read the caller back


@pytest.mark.asyncio
async def test_slow_turn_gets_a_filler(tmp_path):
    agent = make_agent(SlowLLM([LLMReply("What's your tracking number?")], delay=0.6))
    brain = SwiftRouteBrain(agent, filler_after_s=0.2)
    down, _ = await run_test(brain, frames_to_send=[
        UserStartedSpeakingFrame(), tr("hi"), UserStoppedSpeakingFrame(), SleepFrame(sleep=1.0)])
    said = spoken(down)
    assert said[0] in ("One moment.", "Just a sec.", "Let me look.") and said[-1] == "What's your tracking number?"


@pytest.mark.asyncio
async def test_barge_in_drops_stale_reply_and_tells_the_agent(tmp_path):
    agent = make_agent(SlowLLM([LLMReply("This answer is stale."), LLMReply("Fresh answer.")], delay=0.4))
    brain = SwiftRouteBrain(agent, filler_after_s=5)
    down, _ = await run_test(brain, frames_to_send=[
        UserStartedSpeakingFrame(), tr("where is it"), UserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.1),
        UserStartedSpeakingFrame(),            # caller talks over the thinking agent
        tr("actually can I change the address"), UserStoppedSpeakingFrame(),
        SleepFrame(sleep=1.2)])
    assert spoken(down) == ["Fresh answer."]
    # the unheard reply is gone from the model's history, not annotated (call #2: 3B copied annotations)
    said = [m["content"] for m in agent.history if m["role"] == "assistant"]
    assert said == ["Fresh answer."]


@pytest.mark.asyncio
async def test_latency_record_written(tmp_path):
    log = tmp_path / "lat.jsonl"
    agent = make_agent(ScriptedLLM([LLMReply("Hello there.")]))
    brain = SwiftRouteBrain(agent, latency_log=log)
    await run_test(brain, frames_to_send=[
        UserStartedSpeakingFrame(), tr("hi"), VADUserStoppedSpeakingFrame(), UserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.3), BotStartedSpeakingFrame(), SleepFrame(sleep=0.1)])
    import json
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["caller"] == "hi" and rec["voice_to_voice_ms"] is not None and rec["agent_ms"] is not None


def test_sentences():
    assert sentences("Done. Friday, October 16. Anything else?") == ["Done.", "Friday, October 16.", "Anything else?"]
    rb = ("Just to check, is that the parcel ending in V as in Victor, X as in X-ray, "
          "J as in Juliet, V as in Victor, P as in Papa?")
    chunks = sentences(rb)
    assert chunks[0] == "Just to check, is that the parcel ending in V as in Victor,"
    assert " ".join(chunks) == rb and all(len(c) >= 15 for c in chunks)


# ------------------------------------------------------------- endpointing
from swiftroute.voice import endpointing  # noqa: E402


@pytest.mark.parametrize("reply,expect", [
    ("Could you please provide the tracking number?", "id"),
    ("I couldn't find that number. Could you read it again, one character at a time?", "id"),
    ("Got it, I found your parcel. What's the delivery postal code?", "postal"),
    ("Done, delivery is rescheduled to Friday, October 16. Anything else?", "open"),
    # a statement mentioning the postal code isn't a question about it
    ("Your postal code matches. What would you like to do?", "open"),
])
def test_expectation(reply, expect):
    assert endpointing.expectation(reply) == expect


def test_answer_is_complete_uses_the_checksum():
    assert endpointing.answer_is_complete("id", "S for Snake. R for rat. V for 1. C for car M4MAT C for car. 7. "
                                                "R 4 RAT A4 and... J for Joker 1.")
    assert not endpointing.answer_is_complete("id", "S R V C M C 7")            # half-read: keep listening
    assert endpointing.answer_is_complete("postal", "V 8 A 1 L 1")
    assert not endpointing.answer_is_complete("postal", "V 8 A")
    assert not endpointing.answer_is_complete("open", "yes")


class FakeAnalyzer:
    class P:
        stop_secs = 3.0
    def __init__(self):
        self._params = self.P()
        self._stop_ms = 3000


@pytest.mark.asyncio
async def test_fallback_follows_the_question_and_complete_id_ends_turn_early():
    an = FakeAnalyzer()
    agent = make_agent(ScriptedLLM([
        LLMReply("Sure, what's your tracking number?"),
        LLMReply("", [{"id": "c1", "name": "find_parcel", "arguments": {"spoken_tracking_id": "x"}}]),
    ]))
    tid = agent.session.conn.execute("SELECT tracking_id FROM parcels LIMIT 1").fetchone()[0]
    agent.llm.replies[1].tool_calls[0]["arguments"]["spoken_tracking_id"] = " ".join(tid)
    brain = SwiftRouteBrain(agent, turn_analyzer=an, filler_after_s=5)
    down, _ = await run_test(brain, frames_to_send=[
        UserStartedSpeakingFrame(), tr("where's my parcel"), UserStoppedSpeakingFrame(), SleepFrame(sleep=0.3),
        # now expecting an ID: fallback lengthened
        UserStartedSpeakingFrame(), tr(" ".join(tid)),
        SleepFrame(sleep=0.3),          # no UserStoppedSpeakingFrame yet: the checksum ended the turn
    ])
    said = spoken(down)
    assert said[0].startswith("Sure") and "postal code" in said[-1]
    assert an._stop_ms == endpointing.FALLBACK_S["postal"] * 1000   # retuned again for the next question



def test_partial_interruption_is_flagged_in_state_not_in_history():
    agent = make_agent(ScriptedLLM([LLMReply("A long answer about your parcel.")]))
    agent.respond("where is it")
    for _ in range(3):                              # barge-in reported several times: still one effect
        agent.mark_last_reply_unheard(partial=True)
    last = [m for m in agent.history if m["role"] == "assistant"][-1]
    assert last["content"] == "A long answer about your parcel."
    assert "cut off your previous reply" in agent.state_block()
    msgs = agent.build_messages()
    assert all(not k.startswith("_") for m in msgs for k in m)    # private flags never sent to the model


def test_bracket_notes_are_never_spoken():
    from swiftroute.agent import to_speech
    out = to_speech("Let's open a claim. [caller interrupted; this reply was NOT heard] [caller interrupted]")
    assert out == "Let's open a claim."


@pytest.mark.asyncio
async def test_after_transfer_bot_only_holds_the_line():
    agent = make_agent(ScriptedLLM([LLMReply("", [{"id": "t", "name": "transfer_to_human",
                                                    "arguments": {"reason": "asked", "summary": "x"}}]),
                                    LLMReply("I'll connect you now."), LLMReply("SHOULD NOT BE SAID")]))
    brain = SwiftRouteBrain(agent, filler_after_s=5)
    down, _ = await run_test(brain, frames_to_send=[
        UserStartedSpeakingFrame(), tr("agent please"), UserStoppedSpeakingFrame(), SleepFrame(sleep=0.3),
        UserStartedSpeakingFrame(), tr("hey are you there"), UserStoppedSpeakingFrame(), SleepFrame(sleep=0.3)])
    said = spoken(down)
    assert said[-1].startswith("You're in the queue") and "SHOULD NOT BE SAID" not in said
