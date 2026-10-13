"""SwiftRouteBrain: our Agent as a Pipecat frame processor.

Pipecat's built-in LLM services expect Pipecat to own the context and the tool
loop. We don't want that: our Agent has its own tool loop, runtime guards,
state block and fast path, all covered by the eval harness. So this processor
sits where an LLM service would and adapts frames to Agent.respond().

    ... -> STT -> UserTurnProcessor -> [SwiftRouteBrain] -> TTS -> output

Responsibilities:
  * collect the caller's final transcript for the turn; run the agent off the
    event loop (it blocks on HTTP to the model server)
  * barge-in: when the caller starts talking, stop caring about the reply in
    flight and tell the agent what the caller did or didn't hear
  * filler: if a turn is slow (tool calls, a long LLM generation), say
    "One moment." so the line is never silent for long
  * sentence-by-sentence TTS so the first words play as soon as possible
  * a per-turn latency record (the numbers that matter on a phone line)
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    MetricsFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .. import trace
from ..agent import Agent
from . import endpointing

GREETING = "Thanks for calling SwiftRoute, this is Sam. How can I help?"
FILLERS = ["One moment.", "Just a sec.", "Bear with me."]
HOLD_MESSAGE = "You're in the queue for a colleague. They'll be with you shortly."


def sentences(text: str, max_chars: int = 70, min_chunk: int = 25) -> list[str]:
    """Split a reply into TTS chunks. Long sentences are split again at commas:
    measured with Kokoro on CPU, a 120-char read-back took 3.9 s to first audio
    as one sentence and 1.7 s when its first clause went out on its own."""
    out: list[str] = []
    for s in (x for x in re.split(r"(?<=[.?!])\s+", text.strip()) if x):
        if len(s) <= max_chars:
            out.append(s)
            continue
        chunk = ""
        for part in re.split(r"(?<=,)\s+", s):
            chunk = f"{chunk} {part}".strip()
            if len(chunk) >= min_chunk:
                out.append(chunk)
                chunk = ""
        if chunk:
            out[-1] = f"{out[-1]} {chunk}" if out else chunk
    return out


class SwiftRouteBrain(FrameProcessor):
    def __init__(self, agent: Agent, *, filler_after_s: float = 2.5, transcript_wait_s: float = 1.5,
                 latency_log: str | Path | None = None, turn_analyzer=None, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.filler_after_s = filler_after_s      # measured from when the CALLER went quiet
        self.turn_analyzer = turn_analyzer        # smart-turn instance, retuned per question
        self._expect = "open"
        self._early = False                        # turn already started from a complete ID/postal
        self._turn_open = False                    # caller is mid-turn (between start and stop)
        self.transcript_wait_s = transcript_wait_s
        self.latency_log = Path(latency_log) if latency_log else None
        self._buf: list[str] = []
        self._seq = 0                      # bumped on barge-in: stale replies are dropped
        self._lock = asyncio.Lock()        # one agent call at a time (it owns the history)
        self._bot_speaking = False
        self._fillers_used = 0
        self._marks: dict = {}             # timestamps for the turn being measured
        self._pending_record: dict | None = None

    async def greet(self, text: str = GREETING) -> None:
        trace.emit("agent.say", text=text, source="greeting", chunks=sentences(text))
        self._set_expectation(text)
        self.agent.history.append({"role": "assistant", "content": text})   # so slot handling sees the question
        for chunk in sentences(text):
            await self.push_frame(TTSSpeakFrame(chunk))

    async def warm(self) -> None:
        """Pay cold-start costs while the greeting plays, not on the caller's first
        question: the model's first load into memory, and the fuzzy-match library
        import (~0.5 s measured)."""
        def _warm():
            from .. import ids
            ids._shortlist([[("SR0", 0.0)]], ["SR0"] * 600)
            try:
                self.agent.llm.chat([{"role": "user", "content": "Reply with: ok"}])
            except Exception as e:
                logger.warning(f"LLM warm-up failed: {e}")
        await asyncio.to_thread(_warm)

    def _set_expectation(self, reply: str) -> None:
        # after a backchannel ("Mm-hm, go on.") the caller is still mid-code
        if getattr(self.agent, "_slot_buf", ""):
            self._expect = self.agent._slot_expect
        else:
            self._expect = endpointing.expectation(reply)
        endpointing.set_silence_fallback(self.turn_analyzer, endpointing.FALLBACK_S[self._expect])
        trace.emit("endpoint.config", expect=self._expect, silence_fallback_s=endpointing.FALLBACK_S[self._expect],
                   after=reply, why={"id": "callers pause between characters of a tracking number",
                                     "postal": "postal codes come in two halves",
                                     "open": "normal conversation: answer fast"}[self._expect])

    # ------------------------------------------------------------------ frames
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        now = time.perf_counter()

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            # These are TextFrames: if forwarded, the TTS would read the caller's words back.
            if isinstance(frame, InterimTranscriptionFrame) and frame.text.strip():
                trace.emit("stt.interim", text=frame.text.strip())
            if isinstance(frame, TranscriptionFrame) and frame.text.strip():
                self._buf.append(frame.text.strip())
                self._marks["transcript"] = now
                vs = self._marks.get("vad_stop")
                trace.emit("stt.final", text=frame.text.strip(), buffered=" ".join(self._buf),
                           after_vad_stop_ms=round((now - vs) * 1000) if vs else None)
                # A checksum-valid ID (or full postal code) means the caller is done,
                # whatever the silence timer thinks. Start now.
                if (self._turn_open and not self._early and self._expect in ("id", "postal")
                        and endpointing.answer_is_complete(self._expect, " ".join(self._buf))):
                    self._early = True
                    self._marks["turn_end"] = now
                    self._marks["early_endpoint"] = True
                    trace.emit("endpoint.early", expect=self._expect, text=" ".join(self._buf),
                               why="checksum-valid ID / complete postal code: don't wait for silence")
                    self.create_task(self._run_turn(self._seq))
            return

        if isinstance(frame, MetricsFrame):
            for d in frame.data:
                if isinstance(d, TurnMetricsData):
                    self._marks["smart_turn_complete"] = d.is_complete
                    self._marks["smart_turn_prob"] = round(d.probability, 3)
                    trace.emit("smart_turn", complete=d.is_complete, probability=round(d.probability, 3),
                               inference_ms=round(getattr(d, "e2e_processing_time_ms", 0) or 0, 1),
                               why="audio model: has the caller finished their thought, or just paused?")

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._marks["vad_stop"] = now

        elif isinstance(frame, UserStartedSpeakingFrame):
            # Pipecat sends an InterruptionFrame alongside this; handling both counted
            # one barge-in two or three times in call #2. React to the turn start only.
            self._turn_open = True
            trace.emit("turn.user_start")
            if self._bot_speaking or self._lock.locked():
                self._seq += 1             # anything still in flight is now stale
                self._marks["barge_in"] = True
                if self._bot_speaking:
                    self.agent.mark_last_reply_unheard(partial=True)
                trace.emit("barge_in", bot_speaking=self._bot_speaking, agent_thinking=self._lock.locked(),
                           action=("reply cut off: flagged in CALL STATE, not in the text the model sees"
                                   if self._bot_speaking else "reply in flight is now stale and will be dropped"))

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._turn_open = False
            vs = self._marks.get("vad_stop")
            trace.emit("turn.user_end", early=self._early,
                       endpoint_ms=round((now - vs) * 1000) if vs else None)
            if self._early:
                self._early = False        # already handled; anything said since is a new turn
                if self._buf:
                    self._marks["turn_end"] = now
                    self.create_task(self._run_turn(self._seq))
            else:
                self._marks["turn_end"] = now
                self.create_task(self._run_turn(self._seq))

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            trace.emit("bot.speaking", on=True)
            if self._pending_record is not None and "first_audio" not in self._pending_record:
                self._pending_record["first_audio"] = now
                self._write_record(self._pending_record)
                self._pending_record = None

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            trace.emit("bot.speaking", on=False)

        await self.push_frame(frame, direction)

    # -------------------------------------------------------------------- turn
    async def _run_turn(self, seq: int) -> None:
        deadline = time.perf_counter() + self.transcript_wait_s
        while not self._buf and time.perf_counter() < deadline:
            await asyncio.sleep(0.03)      # segmented STT may finish just after the turn ends
        if not self._buf:
            return
        text = " ".join(self._buf)
        self._buf.clear()
        marks = dict(self._marks)
        self._marks = {}

        async with self._lock:
            if seq != self._seq:
                return
            if "TRANSFER" in self.agent.session.reason_codes:
                # Handed off. In call #2 the bot said "transferring you" and then kept
                # chatting. Once handed off, it only holds the line (phase 4 dials a real number).
                trace.emit("agent.say", text=HOLD_MESSAGE, source="hold: already handed off, no LLM")
                await self.push_frame(TTSSpeakFrame(HOLD_MESSAGE))
                return
            filler = self.create_task(self._filler(seq, marks.get("vad_stop")))
            t0 = time.perf_counter()
            try:
                reply = await asyncio.to_thread(self.agent.respond, text)
            except Exception as e:                       # model server down, etc.
                logger.exception(f"agent failed: {e}")
                reply = "Sorry, I'm having a technical problem. Let me get a colleague to help you."
            t1 = time.perf_counter()
            await self.cancel_task(filler)
            m = self.agent.metrics[-1] if self.agent.metrics else None
            record = {
                "turn": m.turn if m else None, "caller": text, "agent": reply,
                "vad_stop": marks.get("vad_stop"), "turn_end": marks.get("turn_end"),
                "transcript": marks.get("transcript"), "agent_start": t0, "agent_done": t1,
                "llm_calls": m.llm_calls if m else 0, "fast_path": m.fast_path if m else 0,
                "llm_call_ms": [round(x) for x in m.llm_ms] if m else [],
                "prompt_tokens": m.prompt_tokens if m else [], "nudges": m.nudges if m else 0,
                "tools": m.tool_calls if m else [], "filler": self._fillers_used > 0,
                "barge_in_before": bool(marks.get("barge_in")),
                "expect": self._expect, "early_endpoint": bool(marks.get("early_endpoint")),
                "fallback_s": endpointing.FALLBACK_S[self._expect],
                "smart_turn_complete": marks.get("smart_turn_complete"),
                "smart_turn_prob": marks.get("smart_turn_prob"),
            }
            self._fillers_used = 0
            if seq != self._seq:
                # caller started talking while we were thinking: never say a stale answer
                self.agent.mark_last_reply_unheard()
                record["dropped_stale"] = True
                trace.emit("guard", kind="stale_reply_dropped", text=reply,
                           why="caller spoke while the agent was thinking; the answer is to an old question")
                self._write_record(record)
                return
            self._pending_record = record
            trace.emit("agent.say", text=reply, source="agent", chunks=sentences(reply),
                       why_chunks="clause-level chunks: first audio starts after the first clause")
            self._set_expectation(reply)
            for s in sentences(reply):
                await self.push_frame(TTSSpeakFrame(s))

    async def _filler(self, seq: int, caller_quiet_at: float | None) -> None:
        waited = time.perf_counter() - caller_quiet_at if caller_quiet_at else 0.0
        await asyncio.sleep(max(0.1, self.filler_after_s - waited))
        if seq == self._seq:
            self._fillers_used += 1
            trace.emit("filler", text=FILLERS[(self.agent.session.turn or 0) % len(FILLERS)],
                       after_s=self.filler_after_s, why="the line has been silent too long while the agent works")
            await self.push_frame(TTSSpeakFrame(FILLERS[(self.agent.session.turn or 0) % len(FILLERS)]))

    # ----------------------------------------------------------------- logging
    def _write_record(self, r: dict) -> None:
        def ms(a, b):
            return round((r[b] - r[a]) * 1000) if r.get(a) and r.get(b) else None
        out = {k: v for k, v in r.items() if not isinstance(v, float)}
        out.update({
            # when the caller stopped making sound -> smart-turn decided they were done
            "endpoint_ms": ms("vad_stop", "turn_end"),
            "stt_ms": ms("vad_stop", "transcript"),
            "agent_ms": ms("agent_start", "agent_done"),
            "tts_first_audio_ms": ms("agent_done", "first_audio"),
            # the number a caller feels: silence after they stop talking
            "voice_to_voice_ms": ms("vad_stop", "first_audio"),
        })
        trace.emit("latency", **{k: out.get(k) for k in (
            "turn", "endpoint_ms", "stt_ms", "agent_ms", "tts_first_audio_ms", "voice_to_voice_ms", "llm_calls",
            "llm_call_ms", "fast_path", "early_endpoint", "expect", "smart_turn_prob", "dropped_stale")})
        logger.info(f"latency turn={out.get('turn')} v2v={out['voice_to_voice_ms']}ms "
                    f"endpoint={out['endpoint_ms']} stt={out['stt_ms']} agent={out['agent_ms']} "
                    f"tts={out['tts_first_audio_ms']} llm_calls={out['llm_calls']} fast={out['fast_path']}")
        if self.latency_log:
            self.latency_log.parent.mkdir(parents=True, exist_ok=True)
            with self.latency_log.open("a") as f:
                f.write(json.dumps(out) + "\n")
