"""Summarise data/latency.jsonl:  python -m swiftroute.voice.latency_report [--last N]"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

LOG = Path(__file__).resolve().parents[2] / "data" / "latency.jsonl"
STAGES = [
    ("endpoint_ms", "turn detection (silence -> 'caller is done')"),
    ("stt_ms", "speech-to-text (silence -> transcript)"),
    ("agent_ms", "agent (LLM calls + tools)"),
    ("tts_first_audio_ms", "TTS to first audio"),
    ("voice_to_voice_ms", "VOICE-TO-VOICE (what the caller feels)"),
]


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=0, help="only the last N turns")
    ap.add_argument("--log", default=str(LOG))
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.log).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if not r.get("dropped_stale")]
    if a.last:
        rows = rows[-a.last:]
    print(f"{len(rows)} turns\n")
    print(f"{'stage':<46}{'p50':>8}{'p95':>8}{'max':>8}")
    for key, label in STAGES:
        xs = [r[key] for r in rows if r.get(key) is not None]
        if xs:
            print(f"{label:<46}{statistics.median(xs):>8.0f}{pct(xs, .95):>8.0f}{max(xs):>8.0f}")

    def p50(xs):
        xs = [x for x in xs if x is not None]
        return f"{statistics.median(xs):.0f}" if xs else "-"

    print("\nturn detection by what the agent had just asked:")
    for exp in ("id", "postal", "open"):
        rs = [r for r in rows if r.get("expect") == exp]
        if rs:
            early = sum(bool(r.get("early_endpoint")) for r in rs)
            print(f"  {exp:<7} {len(rs):>3} turns   endpoint p50 {p50(r.get('endpoint_ms') for r in rs):>5} ms"
                  f"   ended early by checksum: {early}/{len(rs)}")
    st = [r for r in rows if r.get("smart_turn_complete") is not None]
    if st:
        done = sum(bool(r["smart_turn_complete"]) for r in st)
        print(f"smart-turn said 'complete' on {done}/{len(st)} turns "
              f"(the rest waited for the silence fallback)")

    print("\nagent time by LLM calls in the turn:")
    for n in sorted({r.get("llm_calls", 0) for r in rows}):
        rs = [r for r in rows if r.get("llm_calls") == n]
        fp = sum(bool(r.get("fast_path")) for r in rs)
        print(f"  {n} call(s): {len(rs):>3} turns   agent p50 {p50(r.get('agent_ms') for r in rs):>5} ms"
              f"   (tool reply spoken directly, 1 call saved: {fp})")
    calls = [x for r in rows for x in r.get("llm_call_ms", [])]
    toks = [x for r in rows for x in r.get("prompt_tokens", []) if x]
    if calls:
        print(f"\nper LLM call: p50 {statistics.median(calls):.0f} ms, p95 {pct(calls, .95):.0f} ms over {len(calls)} calls"
              + (f"; prompt p50 {statistics.median(toks):.0f} tokens, max {max(toks)}" if toks else ""))
        print(f"LLM calls per turn: mean {len(calls) / len(rows):.1f}; nudges (corrective retries): "
              f"{sum(r.get('nudges', 0) for r in rows)}")
        slow = statistics.median(calls) > 2500
        if slow:
            print("  ! each call is slow, not just the number of calls. On a laptop running Whisper, Kokoro and\n"
                  "    the LLM together, check memory pressure (Activity Monitor > Memory: swap used).")
    print(f"\nturns that needed a filler: {sum(bool(r.get('filler')) for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
