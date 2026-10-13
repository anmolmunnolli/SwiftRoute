"""Inspector routes, mounted on the voice bot's own FastAPI app (same port as the
Pipecat playground):

    http://localhost:7860/inspector          the page (live while a call runs)
    ws://localhost:7860/inspector/ws         live event stream (backlog first, then live)
    GET /inspector/traces                    recorded calls (data/traces/*.jsonl)
    GET /inspector/traces/{name}             one recorded call, for replay
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from .. import trace

PAGE = Path(__file__).with_name("index.html")


def standalone(page: str) -> str:
    """index.html is written as a page body (that's what the hosted showcase needs);
    wrap it into a full document for serving or for a self-hosted copy."""
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<style>[hidden]{display:none!important}</style></head><body>' + page + "</body></html>")


def add_inspector_routes(app) -> None:
    @app.get("/inspector")
    async def inspector_page():
        return HTMLResponse(standalone(PAGE.read_text()))

    @app.get("/inspector/learning")
    async def learning_state():
        from ..learning import store
        return JSONResponse(store.snapshot())

    @app.get("/inspector/traces")
    async def list_traces():
        files = sorted(trace.TRACE_DIR.glob("*.jsonl"), reverse=True) if trace.TRACE_DIR.exists() else []
        return JSONResponse([{"name": f.name, "bytes": f.stat().st_size,
                              "wav": f.with_suffix(".wav").name if f.with_suffix(".wav").exists() else None}
                             for f in files[:50]])

    @app.get("/inspector/traces/{name}")
    async def get_trace(name: str):
        f = (trace.TRACE_DIR / name).resolve()
        if f.parent != trace.TRACE_DIR.resolve() or not f.exists() or f.suffix not in (".jsonl", ".wav"):
            return PlainTextResponse("not found", status_code=404)
        return FileResponse(f, media_type="audio/wav" if f.suffix == ".wav" else "application/x-ndjson")

    @app.websocket("/inspector/ws")
    async def live(ws: WebSocket):
        await ws.accept()
        q, backlog = trace.HUB.subscribe()
        try:
            await ws.send_text(json.dumps({"kind": "hello", "live": trace.HUB.active, "call": trace.HUB.call}))
            for i in range(0, len(backlog), 200):
                await ws.send_text(json.dumps(backlog[i:i + 200], default=str))
            while True:
                ev = await q.get()
                batch = [ev]
                while not q.empty() and len(batch) < 200:
                    batch.append(q.get_nowait())
                await ws.send_text(json.dumps(batch, default=str))
        except (WebSocketDisconnect, asyncio.CancelledError, RuntimeError):
            pass
        finally:
            trace.HUB.unsubscribe(q)
