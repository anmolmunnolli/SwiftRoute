"""A synthetic phone call through the REAL voice pipeline, recorded for the inspector.

    python -m swiftroute.inspector.simulate                      # best available stack
    python -m swiftroute.inspector.simulate --stt whisper --llm ollama --tts kokoro
    python -m swiftroute.inspector.simulate --scenario misheard_id

Why: judges (and CI) can watch a full call without a microphone. A rule-based
caller (evals/harness.py RuleCaller, knows the ground truth) answers whatever the
agent asks; its words are synthesised to audio and streamed into the pipeline in
real time, 20 ms at a time, with silence in between, exactly like a mic would:

    caller audio -> Silero VAD -> [spectrogram tap] -> STT -> smart-turn v3 -> SwiftRouteBrain
                 -> Agent (LLM + tools + policy + guards) -> TTS -> [spectrogram tap] -> speaker

What is real and what is a stand-in is written into the trace (call.start "stack"),
and the inspector shows it, so nobody mistakes a stand-in for the real thing:
  * VAD, smart-turn, the brain, slots, ID decoding, tools, policy, guards: always real
  * STT:  faster-whisper if its model is available (--stt whisper), else "scripted":
          the caller's exact words, released when VAD reports silence
  * LLM:  Ollama/any OpenAI-compatible server (--llm ollama), else the rule-based
          reference agent from the evals (--llm rule)
  * TTS:  Kokoro (--tts kokoro), else eSpeak NG (bundled with pip install espeakng-loader)
  * caller voice: eSpeak NG, always (a second synthetic voice)
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evals.harness import NOW, SCENARIOS, RuleCaller, Scenario  # noqa: E402
from swiftroute import trace  # noqa: E402
from swiftroute.agent import Agent  # noqa: E402
from swiftroute.db import connect, seed  # noqa: E402
from swiftroute.tools import CallSession  # noqa: E402
from swiftroute.voice.brain import SwiftRouteBrain  # noqa: E402
from swiftroute.voice.taps import AudioTap, CallAudio  # noqa: E402

SR_IN = 16000
CHUNK_MS = 20

DEMO = Scenario(
    "demo_misheard_weekend", "DELIVERY_EXCEPTION",
    "Hi, I missed my delivery yesterday. Can you bring it on Saturday instead?",
    where="p.attempts < 3", garble=True, date_phrase="Saturday", followups=["Okay, Monday then."],
    expect={"ID_CONFIRMED", "RESCHED_OK"})
# Callers often say "yes" before a long read-back finishes. The demo does that once, so the
# barge-in path (cut audio, flag the cut-off in CALL STATE) is visible in the trace.
DEMO_BARGE_IN = {"Just to check": 3.2}


# --------------------------------------------------------------------------- speech
class Espeak:
    """eSpeak NG through ctypes (the library ships inside the espeakng-loader wheel)."""
    _lock = threading.Lock()

    def __init__(self):
        import espeakng_loader as E
        self.lib = ctypes.CDLL(E.get_library_path())
        self.sr = self.lib.espeak_Initialize(2, 0, E.get_data_path().encode(), 0)
        self._buf: list[np.ndarray] = []
        CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)

        def cb(wav, n, _ev):
            if n > 0:
                self._buf.append(np.ctypeslib.as_array(wav, shape=(n,)).copy())
            return 0
        self._cb = CB(cb)
        self.lib.espeak_SetSynthCallback(self._cb)

    def say(self, text: str, voice: str = "en-us", rate: int = 165, pitch: int = 50) -> np.ndarray:
        with self._lock:
            self._buf = []
            self.lib.espeak_SetVoiceByName(voice.encode())
            self.lib.espeak_SetParameter(1, rate, 0)
            self.lib.espeak_SetParameter(3, pitch, 0)
            b = text.encode() + b"\0"
            self.lib.espeak_Synth(b, len(b), 0, 1, 0, 0, None, None)
            self.lib.espeak_Synchronize()
            return np.concatenate(self._buf) if self._buf else np.zeros(0, np.int16)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or not len(x):
        return x
    t = np.arange(int(len(x) * sr_to / sr_from)) * (sr_from / sr_to)
    return np.interp(t, np.arange(len(x)), x.astype(np.float32)).astype(np.int16)


class EspeakTTS(FrameProcessor):
    """Stand-in TTS: TTSSpeakFrame in, TTSAudioRawFrames out (what KokoroTTSService does)."""

    def __init__(self, engine: Espeak, **kw):
        super().__init__(**kw)
        self.engine = engine

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame):
            audio = await asyncio.to_thread(self.engine.say, frame.text, "en-us+f3", 175, 60)
            await self.push_frame(TTSStartedFrame())
            step = int(self.engine.sr * CHUNK_MS / 1000)
            for i in range(0, len(audio), step):
                await self.push_frame(TTSAudioRawFrame(audio[i:i + step].tobytes(), self.engine.sr, 1))
            await self.push_frame(TTSStoppedFrame())
            return
        await self.push_frame(frame, direction)


class TTSGate(FrameProcessor):
    """Counts replies handed to the TTS that haven't finished playing yet, so the
    simulated caller doesn't talk over a reply that's still being synthesised."""

    def __init__(self, speaker: "Speaker", **kw):
        super().__init__(**kw)
        self.speaker = speaker

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame):
            self.speaker.pending += 1
        await self.push_frame(frame, direction)


class Speaker(FrameProcessor):
    """Stand-in output transport: 'plays' agent audio in real time, and tells the rest of
    the pipeline when the bot starts/stops talking (BotStarted/StoppedSpeakingFrame), as the
    real WebRTC output does. A barge-in (InterruptionFrame) discards what's queued."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.queue_s = 0.0
        self.play_until = 0.0
        self.speaking = False
        self.pending = 0                 # replies sent to TTS and not yet finished (see TTSGate)
        self.speaking_since: float | None = None
        self._watch: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            now = time.monotonic()
            self.play_until = max(self.play_until, now) + len(frame.audio) / 2 / frame.sample_rate
            if not self.speaking:
                self.speaking = True
                await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            if self.speaking_since is None:
                self.speaking_since = now
            if self._watch is None or self._watch.done():
                self._watch = self.create_task(self._until_done())
            return
        if isinstance(frame, InterruptionFrame):
            self.play_until = time.monotonic()
            self.pending = 0
        if isinstance(frame, TTSStoppedFrame):
            self.pending = max(0, self.pending - 1)
            return
        if isinstance(frame, TTSStartedFrame):
            return
        await self.push_frame(frame, direction)

    async def _until_done(self):
        while time.monotonic() < self.play_until:
            await asyncio.sleep(0.02)
        self.speaking = False
        self.speaking_since = None
        await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)


class ScriptedSTT(FrameProcessor):
    """Stand-in STT: when VAD reports silence after the caller finished an utterance, emit
    that utterance as the transcript, after a delay similar to Whisper small on a laptop."""

    def __init__(self, shared: SimpleNamespace, delay_s: float = 0.35, **kw):
        super().__init__(**kw)
        self.caller, self.delay_s = shared, delay_s

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame) and self.caller.unsent_transcript:
            text, self.caller.unsent_transcript = self.caller.unsent_transcript, None
            self.create_task(self._emit(text))

    async def _emit(self, text: str):
        await asyncio.sleep(self.delay_s)
        await self.push_frame(TranscriptionFrame(text, "caller", datetime.now(timezone.utc).isoformat(),
                                                 finalized=True))


# --------------------------------------------------------------------------- caller
class SimCaller:
    """Streams the caller's audio into the pipeline in real time and decides what to say
    next with the eval harness's RuleCaller."""

    def __init__(self, task: PipelineTask, rule: RuleCaller, session: CallSession, agent: Agent,
                 brain: SwiftRouteBrain, speaker: Speaker, voice: Espeak, shared: SimpleNamespace,
                 max_turns: int = 14, barge_in: dict[str, float] | None = None):
        self.task, self.rule, self.session, self.agent = task, rule, session, agent
        self.brain, self.speaker, self.voice = brain, speaker, voice
        self.max_turns = max_turns
        self.shared = shared             # .unsent_transcript: read by ScriptedSTT
        self.barge_in = dict(barge_in or {})   # agent phrase -> interrupt after this many seconds of it
        self.transcript: list[tuple[str, str]] = []

    async def run(self, first_line: str):
        silence = np.zeros(int(SR_IN * CHUNK_MS / 1000), np.int16).tobytes()
        clock = time.monotonic()
        said_to: str | None = None
        line: str | None = None
        turns = 0
        quiet_since = None

        async def push(chunk: bytes):
            nonlocal clock
            clock += CHUNK_MS / 1000
            await self.task.queue_frame(InputAudioRawFrame(chunk, SR_IN, 1))
            await asyncio.sleep(max(0.0, clock - time.monotonic()))

        # wait for the greeting to finish
        while True:
            await push(silence)
            if self._agent_idle():
                quiet_since = quiet_since or time.monotonic()
                if time.monotonic() - quiet_since > 0.6:
                    break
            else:
                quiet_since = None
        line = first_line
        while line and turns < self.max_turns:
            turns += 1
            said_to = self._last_agent()
            trace.emit("sim.caller_says", text=line, why="rule-based caller (evals/harness.py) answering the agent")
            self.transcript.append(("caller", line))
            audio = resample(await asyncio.to_thread(self.voice.say, line, "en-us", 160, 45), self.voice.sr, SR_IN)
            step = len(silence) // 2
            for i in range(0, len(audio), step):
                c = audio[i:i + step]
                await push(np.pad(c, (0, step - len(c))).tobytes())
            self.shared.unsent_transcript = line
            if self.rule.done:
                break
            # silence until the agent has answered and finished speaking
            quiet_since = None
            t_wait = time.monotonic()
            while time.monotonic() - t_wait < 40:
                await push(silence)
                cue = next((k for k in self.barge_in if k in self._last_agent()), None)
                if (cue and self.speaker.speaking and self.speaker.speaking_since
                        and time.monotonic() - self.speaker.speaking_since > self.barge_in[cue]):
                    del self.barge_in[cue]
                    trace.emit("sim.note", text="the caller answers before the read-back finishes")
                    break
                answered = self._last_agent() != said_to and not self.agent._slot_buf or \
                    (self.agent._slot_buf and self._last_agent() != said_to)
                if answered and self._agent_idle():
                    quiet_since = quiet_since or time.monotonic()
                    if time.monotonic() - quiet_since > 0.7:
                        break
                else:
                    quiet_since = None
            reply = self._last_agent()
            self.transcript.append(("agent", reply))
            line = self.rule.reply(reply, self.session)
        for _ in range(int(1500 / CHUNK_MS)):
            await push(silence)
        await self.task.queue_frame(EndFrame())

    def _agent_idle(self) -> bool:
        return (not self.speaker.speaking and not self.speaker.pending and not self.brain._lock.locked()
                and not self.brain._buf)

    def _last_agent(self) -> str:
        for m in reversed(self.agent.history):
            if m["role"] == "assistant" and not m.get("tool_calls") and m.get("content"):
                return m["content"]
        return ""


# --------------------------------------------------------------------------- main
def _ollama_up(url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(url.rstrip("/").removesuffix("/v1") + "/api/tags", timeout=1.5)
        return True
    except Exception:
        return False


async def simulate(scenario: Scenario, stt_mode: str, llm_mode: str, tts_mode: str, out_dir: Path) -> Path:
    from evals.rule_agent import RuleAgentLLM
    from swiftroute.llm import OpenAICompatLLM

    conn = connect(":memory:")
    seed(conn, now=NOW)
    q = ("SELECT p.tracking_id, a.postal, c.phone FROM parcels p JOIN addresses a ON a.id = p.address_id "
         "JOIN customers c ON c.id = p.customer_id WHERE p.status = ?"
         + (f" AND {scenario.where}" if scenario.where else "") + " ORDER BY p.tracking_id LIMIT 1")
    row = conn.execute(q, (scenario.status,)).fetchone()
    tid, postal, phone = row["tracking_id"], row["postal"].replace(" ", ""), row["phone"]
    session = CallSession(conn, caller_phone=phone if scenario.phone == "owner" else
                          (None if scenario.phone == "withheld" else "+15559990000"), now=NOW)

    url = os.environ.get("SWIFTROUTE_LLM_URL", "http://localhost:11434/v1")
    if llm_mode == "auto":
        llm_mode = "ollama" if _ollama_up(url) else "rule"
    llm = OpenAICompatLLM(max_tokens=120) if llm_mode == "ollama" else RuleAgentLLM()
    from swiftroute.learning import confusions
    from swiftroute.learning import store as lesson_store
    learned_pairs = confusions.apply_promoted()
    agent = Agent(llm, session, lessons=lesson_store.promoted())
    voice = Espeak()

    smart_turn = LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=0.9))
    brain = SwiftRouteBrain(agent, filler_after_s=1.8, turn_analyzer=smart_turn,
                            latency_log=ROOT / "data" / "latency_sim.jsonl")
    speaker = Speaker()

    if tts_mode == "auto":
        try:
            import kokoro_onnx  # noqa: F401
            tts_mode = "kokoro"
        except ImportError:
            tts_mode = "espeak"
    if tts_mode == "kokoro":
        from pipecat.services.kokoro.tts import KokoroTTSService
        tts = KokoroTTSService(settings=KokoroTTSService.Settings(voice=os.environ.get("SWIFTROUTE_VOICE", "af_heart")))
    else:
        tts = EspeakTTS(voice)

    if stt_mode == "auto":
        stt_mode = "scripted"
    shared = SimpleNamespace(unsent_transcript=None)
    if stt_mode == "whisper":
        from swiftroute.voice.bot import make_stt
        stt = make_stt()
    else:
        stt = ScriptedSTT(shared)

    stack = {
        "vad": "silero (real)", "turn": "smart-turn v3 (real)", "brain/agent/tools/policy/guards": "real",
        "stt": "faster-whisper (real)" if stt_mode == "whisper" else
               "scripted: the caller's exact words, released on VAD silence (no Whisper model in this environment)",
        "llm": f"{llm.model} (real)" if llm_mode == "ollama" else
               "rule-based reference agent from evals/ (no LLM server in this environment)",
        "tts": "kokoro (real)" if tts_mode == "kokoro" else "eSpeak NG (stand-in for Kokoro)",
        "caller": "synthetic: rule-based caller + eSpeak NG voice",
        "id_noise": "first reading of the tracking number has one sound-alike error" if scenario.garble else "none",
    }
    path = trace.start_call(session.call_id, {
        "source": "simulation", "scenario": scenario.name, "stack": stack, "llm": llm.model,
        "prompt_mode": agent.prompt_mode, "demo_parcel": tid, "demo_postal": postal,
        "demo_status": scenario.status, "caller_id": session.caller_phone or "withheld",
        "today": NOW.strftime("%A %Y-%m-%d"), "lessons": agent.lessons,
        "learned_id_confusions": learned_pairs}, path=out_dir)

    rule = RuleCaller(scenario, tid, postal)
    audio = CallAudio()
    pipeline = Pipeline([
        VADProcessor(vad_analyzer=SileroVADAnalyzer()), AudioTap("caller", recorder=audio), stt,
        UserTurnProcessor(user_turn_strategies=UserTurnStrategies(
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=smart_turn)])),
        brain, TTSGate(speaker), tts, AudioTap("agent", recorder=audio), speaker])
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, audio_in_sample_rate=SR_IN))
    caller = SimCaller(task, rule, session, agent, brain, speaker, voice, shared,
                       barge_in=DEMO_BARGE_IN if scenario is DEMO else None)

    async def drive():
        await asyncio.sleep(1.0)                 # pipeline start-up
        await brain.greet()
        await caller.run(scenario.opening)

    runner = PipelineRunner(handle_sigint=False)
    await asyncio.gather(runner.run(task), drive())
    trace.end_call(reason_codes=session.reason_codes,
                   outcome={"expected": sorted(scenario.expect),
                            "reached": sorted(scenario.expect & set(session.reason_codes))})
    if path:
        audio.save(path.with_suffix(".wav"))
    logger.info(f"reason codes: {session.reason_codes}")
    logger.info(f"trace written to {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="demo", help="demo, or any evals/harness.py scenario name")
    ap.add_argument("--stt", default="auto", choices=["auto", "scripted", "whisper"])
    ap.add_argument("--llm", default="auto", choices=["auto", "rule", "ollama"])
    ap.add_argument("--tts", default="auto", choices=["auto", "espeak", "kokoro"])
    ap.add_argument("--out", default=str(trace.TRACE_DIR))
    a = ap.parse_args()
    sc = DEMO if a.scenario == "demo" else next(s for s in SCENARIOS if s.name == a.scenario)
    asyncio.run(simulate(sc, a.stt, a.llm, a.tts, Path(a.out)))


if __name__ == "__main__":
    main()
