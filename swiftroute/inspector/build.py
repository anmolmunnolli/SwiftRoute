"""Bake a recorded call into the inspector page, so it plays anywhere with no server:
a hackathon submission link, GitHub Pages, or an email attachment.

    python -m swiftroute.inspector.build data/traces/<call>.jsonl            # -> dist/swiftroute-inspector.html
    python -m swiftroute.inspector.build data/traces/<call>.jsonl --fragment # page body only (for hosts that add their own <head>)

If <call>.wav sits next to the trace, the call audio is embedded too (plays in sync).
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from .server import PAGE, standalone

ROOT = Path(__file__).resolve().parents[2]


def bake(trace_path: Path, wav: Path | None = None, fragment: bool = False) -> str:
    page = PAGE.read_text()
    events = trace_path.read_text().strip().replace("</", "<\\/")      # JSON-safe, can't close the script tag
    marker = '<script type="application/x-ndjson" id="embedded-trace"></script>'
    assert marker in page
    block = f'<script type="application/x-ndjson" id="embedded-trace">\n{events}\n</script>'
    wav = wav if wav is not None else trace_path.with_suffix(".wav")
    if wav and wav.exists():
        b64 = base64.b64encode(wav.read_bytes()).decode()
        block += f'\n<script type="text/plain" id="embedded-wav">data:audio/wav;base64,{b64}</script>'
    from ..learning import store
    learning = json.dumps(store.snapshot(), default=str).replace("</", "<\\/")
    block += f'\n<script type="application/json" id="embedded-learning">{learning}</script>'
    page = page.replace(marker, block)
    return page if fragment else standalone(page)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--wav", type=Path, default=None)
    ap.add_argument("--fragment", action="store_true")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "dist" / "swiftroute-inspector.html")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(bake(a.trace, a.wav, a.fragment))
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
