"""SwiftRoute voice bot (Pipecat). Same code serves the browser and, in phase 4, a phone line.

    python -m swiftroute.voice.bot                 # open http://localhost:7860 and click Connect
                                                   # and http://localhost:7860/inspector to watch every step
    python -m swiftroute.voice.bot -t twilio -x <your-ngrok-host>     # Twilio: webhook -> https://<host>/twiml
    python -m swiftroute.voice.bot -t telnyx -x <your-ngrok-host>     # Telnyx: TeXML app URL -> https://<host>/texml

Fully local and open source:
    VAD          Silero
    turn end     smart-turn v3 (bundled ONNX): decides when the caller is DONE, not just silent
    STT          faster-whisper, biased toward spelled-out IDs with hotwords + a prompt
    LLM          our Agent (tool loop + guards) over Ollama / any OpenAI-compatible server
    TTS          Kokoro (82M params, runs on CPU)

Environment knobs (all optional):
    SWIFTROUTE_MODEL        qwen2.5:7b      LLM served by Ollama
    SWIFTROUTE_STT          faster-whisper  or "mlx" (Apple GPU, faster, but no hotwords)
    SWIFTROUTE_STT_MODEL    small.en        faster-whisper: tiny.en / base.en / small.en ...;
                                            mlx default: mlx-community/whisper-large-v3-turbo-q4
    SWIFTROUTE_VOICE        af_heart        any Kokoro v1.0 voice
    SWIFTROUTE_CALLER       owner           browser calls have no caller ID; pretend to call from
                                            the demo parcel's phone (owner) or a stranger's (unknown)
    SWIFTROUTE_OWNER_PHONE  (unset)         your mobile, e.g. +18125551234: calls from it pass caller-ID verification
    SWIFTROUTE_FILLER_S     1.8             say "one moment" if the caller has heard nothing for this long
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import Request  # module level: FastAPI resolves string annotations from module globals
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from .. import trace
from ..agent import Agent
from ..learning import confusions
from ..learning import store as lesson_store
from ..db import DEFAULT_DB, connect, seed
from ..llm import OpenAICompatLLM
from ..tools import CallSession
from .brain import SwiftRouteBrain
from .taps import AudioTap, CallAudio

ROOT = Path(__file__).resolve().parents[2]
LATENCY_LOG = ROOT / "data" / "latency.jsonl"

# Bias Whisper toward how callers read IDs. hotwords nudge token probabilities;
# the prompt sets the expected *style* (spaced capitals and digits, not words).
STT_HOTWORDS = ("SwiftRoute tracking postal code Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel Juliet "
                "Kilo Lima Mike November Papa Quebec Romeo Sierra Tango Uniform Victor Whiskey X-ray Yankee Zulu")
STT_PROMPT = "My tracking number is S R 4 K 7 B X 2 M 9 Q. The postal code is M5V 2T6."


def demo_parcel(conn):
    if conn.execute("SELECT COUNT(*) FROM parcels").fetchone()[0] == 0:
        seed(conn)
    row = conn.execute(
        "SELECT p.tracking_id, p.status, a.postal, c.phone FROM parcels p JOIN addresses a ON a.id = p.address_id "
        "JOIN customers c ON c.id = p.customer_id WHERE p.status = 'DELIVERY_EXCEPTION' AND p.attempts < 3 "
        "ORDER BY p.tracking_id LIMIT 1").fetchone()
    link_owner_phone(conn, row)
    return conn.execute("SELECT p.tracking_id, p.status, a.postal, c.phone FROM parcels p "
                        "JOIN addresses a ON a.id = p.address_id JOIN customers c ON c.id = p.customer_id "
                        "WHERE p.tracking_id = ?", (row["tracking_id"],)).fetchone()


def link_owner_phone(conn, demo) -> None:
    """Make your real mobile the demo customer's number on file, so a phone call from
    it counts as the first verification factor (caller ID)."""
    owner = os.environ.get("SWIFTROUTE_OWNER_PHONE")
    if not owner:
        return
    digits = "+" + "".join(c for c in owner if c.isdigit())
    with conn:
        conn.execute("UPDATE customers SET phone = ? WHERE phone = ? AND id != "
                     "(SELECT customer_id FROM parcels WHERE tracking_id = ?)",
                     (f"{digits}-old", digits, demo["tracking_id"]))
        conn.execute("UPDATE customers SET phone = ? WHERE id = "
                     "(SELECT customer_id FROM parcels WHERE tracking_id = ?)", (digits, demo["tracking_id"]))


PENDING = ROOT / "data" / "pending_outbound.json"


def own_numbers() -> set[str]:
    return {_norm(os.environ.get(k, "")) for k in ("TWILIO_FROM", "TELNYX_FROM")} - {""}


def _norm(n: str | None) -> str:
    d = "".join(c for c in (n or "") if c.isdigit())
    return f"+{d}" if d else ""


def register_outbound(to: str, tracking_id: str, reason: str) -> None:
    """call_me runs in another process and some providers (Telnyx) can't carry custom
    stream parameters, so the parcel for an outbound call is handed over via a small
    file keyed by the number being called."""
    import json
    import time
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(PENDING.read_text())
    except (OSError, ValueError):
        data = {}
    data[_norm(to)] = {"tracking_id": tracking_id, "reason": reason, "ts": time.time()}
    PENDING.write_text(json.dumps(data))


def _take_pending(number: str, max_age_s: float = 180) -> dict | None:
    import json
    import time
    try:
        data = json.loads(PENDING.read_text())
    except (OSError, ValueError):
        return None
    item = data.pop(_norm(number), None)
    PENDING.write_text(json.dumps(data))
    if item and time.time() - item["ts"] <= max_age_s:
        return item
    return None


def outbound_greeting(conn, session: CallSession, runner_args) -> str | None:
    """Proactive call about a parcel (see swiftroute/voice/call_me.py). We already know
    which parcel, so it's pre-selected; whoever picks up still has to give the postal code
    before hearing anything about it. A phone can be answered by anyone."""
    body = getattr(getattr(runner_args, "call_data", None), "body", None) or {}
    if not body.get("tracking_id") and session.caller_phone:
        body = _take_pending(session.caller_phone) or {}
    tid = body.get("tracking_id")
    if not tid or not conn.execute("SELECT 1 FROM parcels WHERE tracking_id = ?", (tid,)).fetchone():
        return None
    session.tracking_id = tid
    reason = body.get("reason", "update")
    lead = ("we tried to deliver your parcel today but couldn't" if reason == "missed"
            else "we have an update about your parcel")
    from ..ids import read_back
    return (f"Hi, this is Sam from SwiftRoute. I'm calling because {lead}, "
            f"the one {read_back(tid)}. Before I go on, could you confirm the delivery postal code?")


def caller_phone_for(runner_args: RunnerArguments, demo) -> str | None:
    call = getattr(runner_args, "call_data", None)
    if call is not None and getattr(call, "from_number", None):
        frm, to = _norm(call.from_number), _norm(getattr(call, "to_number", None))
        if frm in own_numbers() and to:
            return to                                        # outbound: the customer is who we called
        return frm                                           # inbound: real caller ID
    mode = os.environ.get("SWIFTROUTE_CALLER", "owner")
    return {"owner": demo["phone"], "unknown": "+15559990000", "withheld": None}.get(mode, demo["phone"])


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, {
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        "twilio": lambda: _ws_params(),
        "telnyx": lambda: _ws_params(),
    })

    conn = connect(DEFAULT_DB)
    demo = demo_parcel(conn)
    session = CallSession(conn, caller_phone=caller_phone_for(runner_args, demo))
    greeting = outbound_greeting(conn, session, runner_args)
    llm = OpenAICompatLLM(max_tokens=120)
    # what the system has learned between calls (both gated by measurement; see swiftroute/learning)
    learned_pairs = confusions.apply_promoted()
    agent = Agent(llm, session, lessons=lesson_store.promoted())
    if os.environ.get("SWIFTROUTE_TEAM") == "1":    # sub-agents coach the front desk (swiftroute/agents/team.py)
        from ..agents.team import CallTeam
        agent = CallTeam(agent)
    logger.info(f"Demo parcel {demo['tracking_id']} ({demo['status']}), postal {demo['postal']}, "
                f"calling as {session.caller_phone or 'withheld'}")

    stt = make_stt()
    tts = KokoroTTSService(settings=KokoroTTSService.Settings(voice=os.environ.get("SWIFTROUTE_VOICE", "af_heart")))
    # smart-turn's silence fallback is retuned by the brain after every question it asks
    smart_turn = LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=0.9))
    brain = SwiftRouteBrain(agent, filler_after_s=float(os.environ.get("SWIFTROUTE_FILLER_S", "1.8")),
                            latency_log=LATENCY_LOG, turn_analyzer=smart_turn)

    path = trace.start_call(session.call_id, {
        "source": "live", "transport": getattr(runner_args, "transport_type", None) or "webrtc",
        "llm": llm.model, "llm_url": llm.base_url, "stt": os.environ.get("SWIFTROUTE_STT", "faster-whisper"),
        "stt_model": os.environ.get("SWIFTROUTE_STT_MODEL", "small.en"),
        "tts": f"kokoro/{os.environ.get('SWIFTROUTE_VOICE', 'af_heart')}", "vad": "silero",
        "turn": "smart-turn v3", "prompt_mode": agent.prompt_mode,
        "demo_parcel": demo["tracking_id"], "demo_postal": demo["postal"], "demo_status": demo["status"],
        "caller_id": session.caller_phone or "withheld",
        "lessons": agent.lessons, "learned_id_confusions": learned_pairs})
    logger.info(f"inspector: http://localhost:7860/inspector  (recording {path})")
    # both sides of the call, saved next to the trace for replay (local only; set
    # SWIFTROUTE_RECORD_AUDIO=0 to keep just the spectrogram)
    audio = CallAudio() if os.environ.get("SWIFTROUTE_RECORD_AUDIO", "1") != "0" else None

    pipeline = Pipeline([
        transport.input(),
        VADProcessor(vad_analyzer=SileroVADAnalyzer()),   # 0.2 s stop: smart-turn makes the real call
        AudioTap("caller", recorder=audio),               # spectrogram + VAD events for the inspector
        stt,
        UserTurnProcessor(user_turn_strategies=UserTurnStrategies(
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=smart_turn)])),
        brain,
        tts,
        AudioTap("agent", recorder=audio),
        transport.output(),
    ])
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    @transport.event_handler("on_client_connected")
    async def on_connected(_transport, _client):
        await brain.greet(greeting) if greeting else await brain.greet()
        await brain.warm()           # model load + library imports happen during the greeting

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_transport, _client):
        logger.info(f"call ended, reason codes: {session.reason_codes}")
        trace.end_call(reason_codes=session.reason_codes)
        if hasattr(agent, "review"):                    # the supervisor's review feeds the improvement crew
            from ..agents.team import save_review
            save_review(agent.review())
        if audio and path:
            audio.save(path.with_suffix(".wav"))
        await task.cancel()

    await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)


def make_stt():
    """faster-whisper (default) takes hotwords + a prompt, which matters for spelled-out IDs.
    MLX Whisper (SWIFTROUTE_STT=mlx) runs on the Apple GPU and is faster, but takes neither.
    Which wins on ID accuracy vs latency is exactly what phase 5 measures."""
    if os.environ.get("SWIFTROUTE_STT", "faster-whisper") == "mlx":
        from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
        model = os.environ.get("SWIFTROUTE_STT_MODEL", MLXModel.LARGE_V3_TURBO_Q4.value)
        logger.info(f"STT: MLX Whisper {model}")
        return WhisperSTTServiceMLX(settings=WhisperSTTServiceMLX.Settings(model=model, no_speech_prob=0.5))
    model = os.environ.get("SWIFTROUTE_STT_MODEL", "small.en")
    logger.info(f"STT: faster-whisper {model} (int8, hotwords on)")
    return WhisperSTTService(
        device="auto", compute_type="int8",
        settings=WhisperSTTService.Settings(model=model, hotwords=STT_HOTWORDS,
                                            initial_prompt=STT_PROMPT, no_speech_prob=0.5))


def _ws_params():
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
    return FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True)


def add_twiml_route(app) -> None:
    """Twilio only sends the caller's number into the media stream if the TwiML asks
    for it, and Pipecat's built-in TwiML (at POST /) doesn't. Point the Twilio number's
    webhook at /twiml instead: it passes From through, so caller ID becomes a real
    verification factor on the phone line."""
    from urllib.parse import parse_qs
    from xml.sax.saxutils import quoteattr

    from fastapi.responses import Response

    @app.api_route("/twiml", methods=["GET", "POST"])
    async def twiml(request: Request):
        # Twilio's webhook POSTs a form; the console's "Try out Voice" may GET with query params
        form = {k: v[0] for k, v in parse_qs((await request.body()).decode()).items()}
        for k, v in request.query_params.items():          # only fill gaps; Twilio's own values win
            if k[:1].isupper():
                form.setdefault(k, v)
        for k in ("From", "To"):                             # "+1..." can arrive as " 1..." after URL-decoding
            if form.get(k) and form[k].strip()[:1].isdigit():
                form[k] = "+" + form[k].strip()
        host = request.headers.get("x-forwarded-host") or request.headers.get("host")
        outbound = form.get("Direction", "").startswith("outbound")
        # the customer is whoever is NOT our Twilio number: From on inbound, To on outbound
        customer = form.get("To", "") if outbound else form.get("From", "")
        logger.info(f"{'outbound call to' if outbound else 'incoming call from'} {customer or 'unknown'}")
        extra = "".join(f'<Parameter name={quoteattr(k)} value={quoteattr(v)}/>'
                        for k, v in request.query_params.items() if k in ("tracking_id", "reason"))
        xml = (f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
               f'<Stream url={quoteattr(f"wss://{host}/ws")}>'
               f'<Parameter name="from_number" value={quoteattr(customer)}/>'
               f'<Parameter name="to_number" value={quoteattr(form.get("To", ""))}/>'
               f'{extra}</Stream></Connect></Response>')
        return Response(content=xml, media_type="application/xml")


def add_texml_route(app) -> None:
    """Telnyx (TeXML). Telnyx sends the caller and called numbers in the stream's start
    message itself, so caller ID needs nothing extra. bidirectionalMode="rtp" lets the
    bot's audio go back down the same stream."""
    from xml.sax.saxutils import quoteattr

    from fastapi.responses import Response

    @app.api_route("/texml", methods=["GET", "POST"])
    async def texml(request: Request):
        host = request.headers.get("x-forwarded-host") or request.headers.get("host")
        logger.info("telnyx call: sending TeXML")
        xml = (f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
               f'<Stream url={quoteattr(f"wss://{host}/ws")} bidirectionalMode="rtp"></Stream>'
               f'</Connect></Response>')
        return Response(content=xml, media_type="application/xml")


if __name__ == "__main__":
    from pipecat.runner.run import app, main
    add_twiml_route(app)
    add_texml_route(app)
    from ..inspector.server import add_inspector_routes
    add_inspector_routes(app)
    main()
