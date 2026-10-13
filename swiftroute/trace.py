"""Glass-box tracing: every decision the system makes, as a stream of events.

    from swiftroute import trace
    trace.emit("tool.result", name="verify_caller", allowed=False, reason_code="ID_NOT_VERIFIED")

Nothing is emitted unless a call is being traced (`trace.start_call`), so the evals and
tests pay nothing. One process-wide HUB fans events out to:
  * the live inspector page (websocket subscribers, see swiftroute/inspector)
  * a JSONL file per call in data/traces/, which the inspector can replay later

Events are plain dicts: {"t": ms since call start, "seq", "call", "kind", ...data}.
Emitting is thread-safe: the agent runs in a worker thread (asyncio.to_thread), the
audio taps run on the event loop.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "data" / "traces"
BACKLOG_MAX = 20000          # events kept in memory so a late-opened inspector sees the whole call


class Hub:
    def __init__(self):
        self._lock = threading.Lock()
        self._subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self.backlog: list[dict] = []
        self.call: str | None = None
        self._t0 = 0.0
        self._seq = 0
        self._file = None

    # ------------------------------------------------------------- calls
    def start_call(self, call_id: str, meta: dict | None = None, path: Path | None = TRACE_DIR) -> Path | None:
        with self._lock:
            if self._file:
                self._file.close()
            self.call, self._t0, self._seq, self.backlog = call_id, time.perf_counter(), 0, []
            out = None
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)
                out = path / f"{time.strftime('%Y%m%d-%H%M%S')}_{call_id}.jsonl"
                self._file = out.open("w")
            else:
                self._file = None
        self.emit("call.start", **(meta or {}))
        return out

    def end_call(self, **data) -> None:
        self.emit("call.end", **data)
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None
            self.call = None

    def now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    @property
    def active(self) -> bool:
        return self.call is not None

    # ------------------------------------------------------------- events
    def emit(self, _event: str, /, **data: Any) -> None:
        if self.call is None:
            return
        kind = _event
        if "kind" in data:                 # e.g. guard kind=... -> keep as "guard"
            data["guard"] = data.pop("kind")
        with self._lock:
            self._seq += 1
            ev = {"t": round((time.perf_counter() - self._t0) * 1000, 1), "seq": self._seq,
                  "call": self.call, "kind": kind, **data}
            if len(self.backlog) < BACKLOG_MAX:
                self.backlog.append(ev)
            if self._file:
                self._file.write(json.dumps(ev, default=str, separators=(",", ":")) + "\n")
                if kind != "audio.mel":
                    self._file.flush()
            subs = list(self._subs)
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(_put, q, ev)
            except RuntimeError:           # subscriber's loop closed
                self.unsubscribe(q)

    def subscribe(self, maxsize: int = 5000) -> tuple[asyncio.Queue, list[dict]]:
        """Returns (queue of new events, snapshot of events so far)."""
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.append((asyncio.get_running_loop(), q))
            return q, list(self.backlog)

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs = [(l, x) for l, x in self._subs if x is not q]


def _put(q: asyncio.Queue, ev: dict) -> None:
    if q.full():                           # slow browser: drop the oldest, never block the call
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(ev)


HUB = Hub()
_enabled: ContextVar[bool] = ContextVar("swiftroute_trace", default=True)


def emit(_event: str, /, **data: Any) -> None:
    if HUB.call is not None and _enabled.get():
        HUB.emit(_event, **data)


def active() -> bool:
    return HUB.call is not None


def start_call(call_id: str, meta: dict | None = None, path: Path | None = TRACE_DIR) -> Path | None:
    return HUB.start_call(call_id, meta, path)


def end_call(**data) -> None:
    HUB.end_call(**data)


def session_snapshot(s) -> dict:
    """What the system knows about the call right now (the authoritative state, held in
    code, not in the model's context). Tracking IDs are shown in full: it's demo data."""
    ident = s.identity.data if s.identity else {}
    return {
        "turn": s.turn,
        "parcel": s.tracking_id,
        "pending_readback": list(s.pending_ids),
        "verification": (ident.get("level") if s.identity and s.identity.allowed
                         else ("failed" if s.identity else "not checked")),
        "caller_id": s.caller_phone or "withheld",
        "failed_verifications": s.failed_verifications,
        "id_readings": list(s.id_readings),
        "other_parcel_mentioned": s.other_parcel_mentioned,
        "caller_wants_human": s.caller_wants_human,
        "reason_codes": list(s.reason_codes),
    }
