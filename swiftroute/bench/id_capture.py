"""ID-capture benchmark: how often does the agent land on the right parcel?

Simulates callers reading tracking numbers over a phone line:
  1. render the ID as speech in mixed styles (letter names, NATO, "B as in
     bravo", "double five", sometimes skipping the SR prefix)
  2. corrupt it with an ASR error model: sound-alike substitutions, random
     substitutions, dropped and stuttered characters. NATO words are far more
     robust than bare letter names, so they get a fraction of the error rate.
  3. compare a naive agent (parse -> exact DB lookup) with the resolver
     (checksum + confusion-weighted edits + DB existence + read-back).

This is a synthetic stand-in. Phase 5 replaces step 2 with real recordings run
through faster-whisper, and the same scorer is reused.

    python bench/id_capture.py --trials 2000 --parcels 20000
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from swiftroute import ids  # noqa: E402
from swiftroute.db import connect, existing_ids, seed  # noqa: E402

_LETTER_SAY = {v: k for k, v in ids._LETTER_NAMES.items() if len(k) > 1}
_LETTER_SAY.update({"A": "ay", "E": "ee", "H": "aitch", "W": "double you", "X": "ex"})
_NATO_SAY = {v: k for k, v in ids._NATO.items() if k not in ("alfa", "juliett", "whisky", "x-ray")}
_DIGIT_SAY = {v: k for k, v in ids._DIGITS.items() if k in
              ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")}


def corrupt(tid: str, err: float, rng: random.Random) -> tuple[str, list[str]]:
    """Return (what ASR + parser produce, list of error kinds injected)."""
    chars = list(tid if rng.random() > 0.2 else tid[2:])          # 20% skip "SR"
    out, kinds = [], []
    for c in chars:
        style = rng.choices(["name", "nato", "as_in"], [0.6, 0.25, 0.15])[0] if c.isalpha() else "name"
        p = err if style == "name" else err / 5
        r = rng.random()
        if r < p * 0.08:
            kinds.append("drop")
            continue
        if r < p * 0.12:
            out.append(c)
            kinds.append("stutter")
        elif r < p:
            grp = ids.CONFUSION_GROUPS[ids._GROUP[c]] if c in ids._GROUP else ""
            mates = [d for d in grp if d != c and d in ids.ALPHABET]
            if mates and rng.random() < 0.8:
                c = rng.choice(mates); kinds.append("soundalike")
            else:
                c = rng.choice([d for d in ids.ALPHABET if d != c]); kinds.append("random")
        out.append(c)
    return "".join(out), kinds


def render_spoken(s: str, rng: random.Random) -> str:
    words = []
    for c in s:
        if c.isdigit():
            words.append(_DIGIT_SAY[c] if rng.random() < 0.7 else c)
        else:
            style = rng.choices(["name", "nato", "as_in"], [0.6, 0.25, 0.15])[0]
            if style == "nato":
                words.append(_NATO_SAY[c])
            elif style == "as_in":
                words.append(f"{c} as in {_NATO_SAY[c]}")
            else:
                words.append(_LETTER_SAY.get(c, c))
    return " ".join(words)


def run(trials: int, n_parcels: int, levels: list[float], seed_val: int) -> dict:
    conn = connect(":memory:")
    seed(conn, n_customers=max(60, n_parcels // 4), n_parcels=n_parcels, seed_val=seed_val,
         now=datetime(2026, 10, 14, tzinfo=timezone.utc))
    all_ids = [r[0] for r in conn.execute("SELECT tracking_id FROM parcels")]
    exists = lambda b: existing_ids(conn, b)  # noqa: E731
    rng = random.Random(seed_val)
    report = {"parcels": n_parcels, "trials_per_level": trials, "levels": []}
    for err in levels:
        naive_ok = top1 = in_cands = silent_wrong = wrong_top1 = not_found = 0
        lat = []
        for _ in range(trials):
            tid = rng.choice(all_ids)
            heard_chars, _ = corrupt(tid, err, rng)
            spoken = render_spoken(heard_chars, rng)
            parsed = ids.parse_spoken(spoken)
            # naive: exact lookup of the parse (with the same prefix fix, to be fair)
            naive_ok += ids.normalise_heard(parsed) == tid
            t0 = time.perf_counter()
            res = ids.resolve(parsed, exists)
            lat.append((time.perf_counter() - t0) * 1000)
            cands = [c.tracking_id for c in res.candidates]
            top1 += res.best == tid
            in_cands += tid in cands
            not_found += res.status == "not_found"
            if res.best and res.best != tid:
                wrong_top1 += 1
                silent_wrong += res.status == "exact"   # acted on without confirmation
        lat.sort()
        report["levels"].append({
            "per_char_error": err,
            "naive_exact_lookup": naive_ok / trials,
            "resolver_top1": top1 / trials,
            "resolver_in_offered_candidates": in_cands / trials,
            "wrong_top1_caught_by_readback": wrong_top1 / trials,
            "silent_wrong_parcel": silent_wrong / trials,
            "not_found_reask": not_found / trials,
            "resolve_ms_p50": round(statistics.median(lat), 2),
            "resolve_ms_p95": round(lat[int(0.95 * len(lat))], 2),
        })
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--parcels", type=int, default=20000)
    ap.add_argument("--levels", default="0.03,0.08,0.15")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "id_capture.json"))
    a = ap.parse_args()
    rep = run(a.trials, a.parcels, [float(x) for x in a.levels.split(",")], a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(f"{a.parcels} parcels, {a.trials} calls per noise level\n")
    hdr = f"{'char err':>8} | {'naive':>6} | {'top-1':>6} | {'offered':>7} | {'silent wrong':>12} | {'re-ask':>6} | {'p95 ms':>6}"
    print(hdr); print("-" * len(hdr))
    for L in rep["levels"]:
        print(f"{L['per_char_error']:>8.0%} | {L['naive_exact_lookup']:>6.1%} | {L['resolver_top1']:>6.1%} | "
              f"{L['resolver_in_offered_candidates']:>7.1%} | {L['silent_wrong_parcel']:>12.2%} | "
              f"{L['not_found_reask']:>6.1%} | {L['resolve_ms_p95']:>6}")


if __name__ == "__main__":
    main()
