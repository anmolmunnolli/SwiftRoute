"""Learn which characters speech recognition really confuses, from confirmed calls.

Ground truth comes for free on a phone line: when the agent reads back a corrected
tracking number and the caller says yes, we know what was HEARD and what was MEANT.
Every character that differs is a real ASR confusion on this deployment's audio,
accent mix and microphone (real call #1: an accented R came out as "odd"/"hard").

    python -m swiftroute.learning.confusions            # mine, evaluate, promote if it passes
    python -m swiftroute.learning.confusions --dry-run  # mine and evaluate only

Update rule: a confirmed pair (heard a -> true b) that the static table prices as a
far swap gets cheaper with each confirmation: cost = max(min_cost, FAR * decay**count).
It needs `min_count` confirmations before it counts at all, so one odd call can't move it.

Gate (both must hold, or nothing changes):
  * replay: every mined real reading resolves to its confirmed parcel at least as often
    as before (the point of learning)
  * benchmark: the synthetic ID benchmark's top-1 doesn't drop by more than 0.5 points
    and the silent-wrong-parcel rate stays at zero (the regression check)
Exact matches never depend on these costs, so no update can make the agent act on a
parcel without the caller confirming it.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from .. import ids, trace
from ..db import DEFAULT_DB, existing_ids
from . import store

FILE = "confusions.json"


# ------------------------------------------------------------------ mining
def mine_audit(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """(heard, confirmed tracking id) for every call where the parcel was confirmed.
    Includes earlier failed readings in the same call, when they have the same length."""
    rows = conn.execute("SELECT call_id, tool, args, reason_code, result FROM audit ORDER BY id").fetchall()
    by_call: dict[str, list] = {}
    for r in rows:
        by_call.setdefault(r[0], []).append(r)
    out = []
    for call, rs in by_call.items():
        heard_list, final = [], None
        for _, tool, args, code, result in rs:
            res = json.loads(result or "{}")
            if tool == "find_parcel" and res.get("heard"):
                heard_list.append(res["heard"])
                if code == "ID_EXACT":
                    final = res.get("tracking_id")
            if tool == "confirm_parcel" and code == "ID_CONFIRMED":
                final = json.loads(args or "{}").get("tracking_id") or final
        if final:
            out += [(call, h, final) for h in heard_list if h != final]
    return out


def mine_traces(trace_dir: Path | None = None) -> list[tuple[str, str, str]]:
    """Same evidence from recorded calls (data/traces), for calls whose audit rows live in
    another database (e.g. simulated calls run on an in-memory DB)."""
    d = trace_dir or trace.TRACE_DIR
    out = []
    for f in sorted(d.glob("*.jsonl")) if d.exists() else []:
        heard_list, final, call = [], None, f.stem
        for line in f.read_text().splitlines():
            e = json.loads(line)
            call = e.get("call", call)
            if e.get("kind") != "tool.result":
                continue
            r = e.get("result") or {}
            if e.get("tool") == "find_parcel" and r.get("heard"):
                heard_list.append(r["heard"])
                if e.get("reason_code") == "ID_EXACT":
                    final = r.get("tracking_id")
            if e.get("tool") == "confirm_parcel" and e.get("reason_code") == "ID_CONFIRMED":
                final = (e.get("args") or {}).get("tracking_id") or final
        if final:
            out += [(call, h, final) for h in heard_list if h != final]
    return out


def pairs_from(readings: list[tuple[str, str]], max_diffs: int = 3) -> Counter:
    c: Counter = Counter()
    for heard, true in readings:
        if len(heard) != len(true):
            continue                             # indels: not a character confusion
        diffs = [(a, b) for a, b in zip(heard, true) if a != b]
        if 0 < len(diffs) <= max_diffs:
            c.update(diffs)
    return c


def costs_from(counts: Counter, min_count: int = 2, decay: float = 0.8, min_cost: float = 0.3) -> dict:
    out = {}
    for (a, b), n in counts.items():
        if n < min_count or a not in ids.ALPHABET or b not in ids.ALPHABET:
            continue
        base = _static(a, b)
        new = max(min_cost, base * decay ** n)
        if new < base - 1e-9:
            out[(a, b)] = round(new, 3)
    return out


def _static(a: str, b: str) -> float:
    ga, gb = ids._GROUP.get(a), ids._GROUP.get(b)
    return ids.NEAR if ga is not None and ga == gb else ids.FAR


# ------------------------------------------------------------------ gate
def replay(readings: list[tuple[str, str]], conn: sqlite3.Connection) -> dict:
    hit = offered = 0
    for heard, true in readings:
        res = ids.resolve(heard, lambda b: existing_ids(conn, b))
        hit += res.best == true
        offered += true in [c.tracking_id for c in res.candidates]
    n = max(1, len(readings))
    return {"n": len(readings), "top1": round(hit / n, 4), "offered": round(offered / n, 4)}


def bench(trials: int = 300, parcels: int = 5000, level: float = 0.08, seed: int = 11) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from bench.id_capture import run
    L = run(trials, parcels, [level], seed)["levels"][0]
    return {"top1": round(L["resolver_top1"], 4), "silent_wrong": round(L["silent_wrong_parcel"], 4)}


def evaluate(readings, conn, candidate: dict, current: dict, bench_trials: int = 300) -> dict:
    ids.set_learned(current)
    before = {"replay": replay(readings, conn), "bench": bench(bench_trials)}
    ids.set_learned({**current, **candidate})
    after = {"replay": replay(readings, conn), "bench": bench(bench_trials)}
    ids.set_learned(current)
    ok = (after["replay"]["top1"] >= before["replay"]["top1"]
          and after["bench"]["top1"] >= before["bench"]["top1"] - 0.005
          and after["bench"]["silent_wrong"] <= before["bench"]["silent_wrong"])
    why = ("replayed real readings resolve at least as well, benchmark holds" if ok else
           "replay got worse" if after["replay"]["top1"] < before["replay"]["top1"] else
           "benchmark top-1 dropped" if after["bench"]["top1"] < before["bench"]["top1"] - 0.005 else
           "silent wrong-parcel rate rose")
    return {"accept": ok, "why": why, "before": before, "after": after}


# ------------------------------------------------------------------ state
def load(base: Path | None = None) -> dict:
    p = store._path(FILE, base)
    return json.loads(p.read_text()) if p.exists() else {"promoted": {}, "history": []}


def _key(pair) -> str:
    return f"{pair[0]}>{pair[1]}"


def apply_promoted(base: Path | None = None) -> int:
    """Load promoted costs into the ID decoder. Called by the bot, simulation and MCP server."""
    costs = {tuple(k.split(">")): v for k, v in load(base)["promoted"].items()}
    ids.set_learned(costs)
    return len(costs)


def learn(conn: sqlite3.Connection, base: Path | None = None, dry_run: bool = False,
          min_count: int = 2, bench_trials: int = 300, trace_dir: Path | None = None,
          use_traces: bool = True) -> dict:
    state = load(base)
    current = {tuple(k.split(">")): v for k, v in state["promoted"].items()}
    from_audit = mine_audit(conn)
    from_traces = mine_traces(trace_dir) if use_traces else []
    # (call, heard, confirmed): a call recorded in both the audit table and a trace counts once
    readings = [(h, t) for _, h, t in dict.fromkeys(from_audit + from_traces)]
    counts = pairs_from(readings)
    proposed = {k: v for k, v in costs_from(counts, min_count).items() if current.get(k) != v}
    report = {"readings": len(readings), "from_audit": len(from_audit), "from_traces": len(from_traces),
              "pairs_seen": {_key(k): n for k, n in counts.most_common(20)},
              "proposed": {_key(k): v for k, v in proposed.items()}}
    if not proposed:
        report.update(accept=False, why=f"no pair confirmed {min_count}+ times beyond what's already learned")
    else:
        report.update(evaluate(readings, conn, proposed, current, bench_trials))
    entry = {"event": "confusions.update", **report, "dry_run": dry_run}
    store.log(entry, base)
    trace.emit("learning.confusions", **report)
    if report.get("accept") and not dry_run:
        state["promoted"].update({_key(k): v for k, v in proposed.items()})
        state["history"].append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "added": report["proposed"],
                                 "before": report["before"], "after": report["after"]})
        store._path(FILE, base).write_text(json.dumps(state, indent=2))
        apply_promoted(base)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-count", type=int, default=2)
    a = ap.parse_args()
    from ..db import connect, seed
    conn = connect(a.db)
    if conn.execute("SELECT COUNT(*) FROM parcels").fetchone()[0] == 0:
        seed(conn)                     # same deterministic parcels the bot and simulation use
    rep = learn(conn, dry_run=a.dry_run, min_count=a.min_count)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
