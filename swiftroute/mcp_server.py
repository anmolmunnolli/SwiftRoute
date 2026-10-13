"""SwiftRoute as an MCP server.

    python -m swiftroute.mcp_server                       # stdio (Claude Desktop, Cursor, the improvement agent)
    python -m swiftroute.mcp_server --http --port 8765    # streamable HTTP at http://localhost:8765/mcp

Three groups of tools:

  call tools    the same policy-checked tools the phone agent uses (find_parcel,
                verify_caller, reschedule_delivery, ...). Same CallSession, same policy.py
                verdicts, same audit table, same traces. Any MCP client becomes another
                front end with identical rules.
  ops tools     read what happened: recent calls, a call's audit trail, reason-code counts.
  learning      run evals, list failures, propose / evaluate / promote lessons, learn
                ID confusions. Promotion is gated in code (swiftroute/learning/lessons.py).
  agents        run the multi-agent improvement crew, read its memory and the messages its
                agents exchanged, and the call team's supervisor reviews (swiftroute/agents).

Security: over MCP the "caller ID" is whatever the client sends, so it is NOT trusted
by default: every MCP session counts as a withheld number, which gives status-only
access after the postal code (policy.verify_identity). Start the server with
--trust-caller-id only for a front end that really knows the number (a telephony bridge).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swiftroute import trace  # noqa: E402
from swiftroute.db import DEFAULT_DB, connect, seed  # noqa: E402
from swiftroute.learning import confusions, lessons, store  # noqa: E402
from swiftroute.tools import CallSession  # noqa: E402

INSTRUCTIONS = """SwiftRoute parcel-courier backend.
For a customer conversation: start_call, then find_parcel with the tracking number exactly as the
customer said it, confirm_parcel only after the customer says yes to a read-back, verify_caller with
the delivery postal code, then the action tools. Every action returns allowed / reason_code /
explanation from the policy engine: relay the explanation; you cannot override it.
For improving the phone agent: run_baseline, list_failures, propose_lesson, evaluate_lesson.
A lesson is promoted only if evaluate_lesson's gate accepts it."""

mcp = FastMCP("swiftroute", instructions=INSTRUCTIONS, log_level="WARNING")
STATE = {"conn": None, "sessions": {}, "trust_caller_id": False, "model": None, "baseline": None}


def _conn():
    if STATE["conn"] is None:
        STATE["conn"] = connect(os.environ.get("SWIFTROUTE_DB", str(DEFAULT_DB)))
        if STATE["conn"].execute("SELECT COUNT(*) FROM parcels").fetchone()[0] == 0:
            seed(STATE["conn"])
        confusions.apply_promoted()
    return STATE["conn"]


def _session(call_id: str) -> CallSession:
    s = STATE["sessions"].get(call_id)
    if s is None:
        raise ValueError(f"unknown call_id {call_id}; call start_call first")
    return s


def _run(call_id: str, tool: str, args: dict) -> dict:
    s = _session(call_id)
    s.turn += 1                              # each MCP call is a new "turn" for the confirm guard
    return s.call(tool, args)


# ================================================================== call tools
@mcp.tool()
def start_call(caller_phone: str | None = None) -> dict:
    """Start a customer conversation. Returns call_id for the other call tools.
    caller_phone is ignored unless the server trusts caller ID (see server docs)."""
    phone = caller_phone if STATE["trust_caller_id"] else None
    s = CallSession(_conn(), caller_phone=phone)
    STATE["sessions"][s.call_id] = s
    return {"call_id": s.call_id, "caller_id": phone or "withheld (MCP clients can't prove a phone number)",
            "access_after_postal_code": "full" if phone else "status only"}


@mcp.tool()
def find_parcel(call_id: str, spoken_tracking_id: str) -> dict:
    """Look up a tracking number exactly as the customer said it (e.g. 'S R one one two M as in Mike...')."""
    return _run(call_id, "find_parcel", {"spoken_tracking_id": spoken_tracking_id})


@mcp.tool()
def confirm_parcel(call_id: str, tracking_id: str = "") -> dict:
    """Only after the customer said yes to the read-back from find_parcel."""
    return _run(call_id, "confirm_parcel", {"tracking_id": tracking_id})


@mcp.tool()
def verify_caller(call_id: str, spoken_postal_code: str) -> dict:
    """Check the delivery postal code the customer gave."""
    return _run(call_id, "verify_caller", {"spoken_postal_code": spoken_postal_code})


@mcp.tool()
def get_status(call_id: str) -> dict:
    """Status, ETA and next delivery of the confirmed parcel (needs verification)."""
    return _run(call_id, "get_status", {})


@mcp.tool()
def reschedule_delivery(call_id: str, day: str) -> dict:
    """Move delivery to the day the customer said, in their words ('Friday', 'Monday the 19th')."""
    return _run(call_id, "reschedule_delivery", {"day": day})


@mcp.tool()
def change_address(call_id: str, line1: str, city: str, spoken_postal_code: str) -> dict:
    """Redirect the parcel to a new address."""
    return _run(call_id, "change_address", {"line1": line1, "city": city, "spoken_postal_code": spoken_postal_code})


@mcp.tool()
def hold_at_depot(call_id: str, depot_code: str | None = None) -> dict:
    """Hold the parcel at a depot for pickup (nearest if no depot_code)."""
    return _run(call_id, "hold_at_depot", {"depot_code": depot_code} if depot_code else {})


@mcp.tool()
def open_claim(call_id: str, kind: str, description: str = "") -> dict:
    """Open a claim. kind: 'damaged' or 'missing'."""
    return _run(call_id, "open_claim", {"kind": kind, "description": description})


@mcp.tool()
def transfer_to_human(call_id: str, customer_asked_for_person: bool, reason: str, summary: str = "") -> dict:
    """Hand off to a person. Allowed only if the customer asked for one (TRANSFER_NOT_REQUESTED otherwise)."""
    if customer_asked_for_person:
        _session(call_id).caller_wants_human = True   # on the phone this comes from the caller's own words
    return _run(call_id, "transfer_to_human", {"reason": reason, "summary": summary})


@mcp.tool()
def end_call(call_id: str) -> dict:
    """End the conversation; returns every reason code it produced."""
    s = STATE["sessions"].pop(call_id, None)
    return {"call_id": call_id, "reason_codes": s.reason_codes if s else []}


# ================================================================== ops tools
@mcp.tool()
def recent_calls(limit: int = 20) -> list[dict]:
    """Recent calls from the audit table (phone, browser and MCP), newest first."""
    rows = _conn().execute(
        "SELECT call_id, MIN(ts), MAX(ts), GROUP_CONCAT(tool, ' '), GROUP_CONCAT(COALESCE(reason_code,''), ' ') "
        "FROM audit WHERE call_id IS NOT NULL GROUP BY call_id ORDER BY MAX(id) DESC LIMIT ?", (limit,)).fetchall()
    return [{"call_id": r[0], "started": r[1], "ended": r[2], "tools": r[3].split(),
             "reason_codes": [c for c in r[4].split() if c]} for r in rows]


@mcp.tool()
def call_audit(call_id: str) -> list[dict]:
    """Every tool call in one call, with arguments, verdict and reason code."""
    rows = _conn().execute("SELECT ts, tool, args, allowed, reason_code FROM audit WHERE call_id = ? ORDER BY id",
                           (call_id,)).fetchall()
    return [{"ts": r[0], "tool": r[1], "args": json.loads(r[2] or "{}"), "allowed": r[3], "reason_code": r[4]}
            for r in rows]


@mcp.tool()
def reason_code_stats(limit_calls: int = 500) -> dict:
    """How often each policy outcome happened across recent calls."""
    rows = _conn().execute("SELECT reason_code FROM audit WHERE reason_code IS NOT NULL ORDER BY id DESC LIMIT ?",
                           (limit_calls * 8,)).fetchall()
    return dict(Counter(r[0] for r in rows).most_common())


@mcp.tool()
def trace_findings(limit_files: int = 20) -> dict:
    """What the runtime guards caught in recorded calls (data/traces): unbacked claims, promises
    without action, dropped text, bad tool calls, unresolved tracking numbers, slow turns."""
    found: Counter = Counter()
    examples: dict[str, list] = {}
    slow = []
    files = sorted(trace.TRACE_DIR.glob("*.jsonl"), reverse=True)[:limit_files] if trace.TRACE_DIR.exists() else []
    for f in files:
        for line in f.read_text().splitlines():
            e = json.loads(line)
            k = e["kind"]
            key = (f"guard:{e.get('guard')}" if k == "guard" else
                   "tool_rejected" if k == "tool.rejected" else
                   "id_not_found" if k == "tool.result" and e.get("reason_code") == "ID_NOT_FOUND" else None)
            if key:
                found[key] += 1
                examples.setdefault(key, [])
                if len(examples[key]) < 3:
                    examples[key].append({"call": f.name, "text": e.get("text") or e.get("explanation") or e.get("args")})
            if k == "latency" and (e.get("voice_to_voice_ms") or 0) > 4000:
                slow.append({"call": f.name, "turn": e.get("turn"), "v2v_ms": e["voice_to_voice_ms"],
                             "llm_calls": e.get("llm_calls")})
    return {"files": len(files), "counts": dict(found), "examples": examples, "slow_turns": slow[:10]}


# ================================================================== learning
def _make_llm(model: str | None):
    from evals.rule_agent import RuleAgentLLM
    from swiftroute.llm import OpenAICompatLLM
    m = model or STATE["model"] or os.environ.get("SWIFTROUTE_MODEL", "qwen2.5:3b")
    if m == "rule-agent":
        return RuleAgentLLM
    if m.startswith("test:"):                    # tests inject a fake model by import path
        mod, _, attr = m[5:].partition(":")
        return getattr(__import__(mod, fromlist=[attr]), attr)
    return lambda: OpenAICompatLLM(model=m)


@mcp.tool()
def run_baseline(model: str | None = None, repeat: int = 2) -> dict:
    """Score the phone agent as deployed (with its promoted lessons) on all 15 eval scenarios.
    Slow with a real model: minutes."""
    s = lessons.score(_make_llm(model), store.promoted(), repeat)
    STATE["baseline"] = s
    store._path("baseline.json").write_text(json.dumps(s, indent=2))
    store.log({"event": "baseline", "model": model or STATE["model"], "totals": s["totals"]})
    return {"totals": s["totals"], "failing_scenarios": [f["scenario"] for f in lessons.failures(s)]}


@mcp.tool()
def list_failures() -> list[dict]:
    """Failing eval scenarios from the last baseline, with reasons and the end of a failing transcript."""
    base = STATE["baseline"]
    if base is None and store._path("baseline.json").exists():
        base = json.loads(store._path("baseline.json").read_text())
    if base is None:
        return [{"error": "no baseline yet; call run_baseline first"}]
    return lessons.failures(base)


@mcp.tool()
def list_lessons() -> list[dict]:
    """Every lesson ever proposed: text, status (promoted / rejected / candidate), gate results."""
    return [{k: x.get(k) for k in ("id", "text", "status", "reason", "before", "after", "source")}
            for x in store.load()["lessons"]]


@mcp.tool()
def propose_lesson(text: str, evidence: str = "") -> dict:
    """Propose ONE short lesson (one line, <= 200 chars) for the phone agent's prompt, e.g.
    'When the caller asks to pick the parcel up, call hold_at_depot right after verification.'
    Lessons can't loosen rules (checked). Returns the lesson id to pass to evaluate_lesson."""
    item = store.add(text, {"by": "mcp", "evidence": evidence[:500]})
    return {"id": item["id"], "status": item["status"], "reason": item.get("reason")}


@mcp.tool()
def evaluate_lesson(lesson_id: str, model: str | None = None, repeat: int = 2, rounds: int = 2) -> dict:
    """Run the evals with and without the lesson (rounds x repeat x 15 scenarios each) and apply
    the gate: more passing runs, no new safety failures, no new false claims, held-out not worse.
    Promotes the lesson if and only if the gate accepts. Slow: minutes."""
    r = lessons.evaluate_and_maybe_promote(lesson_id, _make_llm(model), repeat, rounds)
    return {k: r.get(k) for k in ("accept", "why", "checks", "passed_before", "passed_after", "runs")} | {
        "status": r["lesson"]["status"]}


@mcp.tool()
def retire_lesson(lesson_id: str, reason: str) -> dict:
    """Remove a promoted lesson from the prompt (kept in the log)."""
    return store.update(lesson_id, status="retired", reason=reason)


@mcp.tool()
def learn_id_confusions(dry_run: bool = False) -> dict:
    """Mine confirmed read-backs for real speech-recognition confusions and update the ID
    decoder if the replay + benchmark gate passes."""
    return confusions.learn(_conn(), dry_run=dry_run)


# ================================================================== agents (swiftroute/agents)
@mcp.tool()
def run_improvement_crew(driver: str = "playbook", cycles: int = 1, max_lessons: int = 2, repeat: int = 2,
                         rounds: int = 2, team: bool = False, curate: bool = False, model: str | None = None,
                         writer_model: str = "qwen2.5:7b") -> dict:
    """Run the improvement crew: analyst picks failing scenarios, writer and critic agree on a lesson,
    the evaluator's gate decides, the curator retires lessons that stopped helping. driver: playbook
    (templates) or llm (writer_model drafts). team=True evaluates the phone agent with its call team.
    Slow with a real model: minutes per lesson."""
    from swiftroute.agents.crew import Crew
    from swiftroute.llm import OpenAICompatLLM
    writer = OpenAICompatLLM(model=writer_model, max_tokens=120) if driver == "llm" else None
    out = Crew(_make_llm(model), writer, repeat, rounds, team, verbose=False).run(cycles, max_lessons, curate)
    return {"accepted": sum(c["accepted"] for c in out["cycles"]), "promoted_now": out["promoted_now"],
            "messages": out["messages"],
            "evaluated": [{k: e.get(k) for k in ("scenario", "text", "accept", "why")}
                          for c in out["cycles"] for e in c["evaluated"]],
            "skipped": out["cycles"][-1]["skipped"] if out["cycles"] else []}


@mcp.tool()
def crew_memory() -> dict:
    """What the improvement crew has tried per scenario, with the gate's verdict, and past cycles."""
    from swiftroute.agents.crew import Memory
    return Memory().data


@mcp.tool()
def agent_messages(limit: int = 50) -> list[dict]:
    """The latest messages the crew's agents sent each other (sender, to, kind, body)."""
    p = store._path("crew_messages.jsonl")
    lines = p.read_text().splitlines()[-limit:] if p.exists() else []
    return [{k: v for k, v in json.loads(x).items() if k in ("sender", "to", "kind", "body", "reply_to")}
            for x in lines]


@mcp.tool()
def call_reviews(limit: int = 20) -> list[dict]:
    """The call-team supervisor's reviews of recent calls: the caller's goal, whether it was handled,
    and what went wrong (forgot_goal, repeated_reply, caller_frustrated, ...)."""
    from swiftroute.agents.team import load_reviews
    return load_reviews(limit=limit)


# ================================================================== resources
@mcp.resource("swiftroute://policy")
def policy_resource() -> str:
    """The policy configuration the tools enforce."""
    from dataclasses import asdict
    from swiftroute.policy import DEFAULT
    return json.dumps(asdict(DEFAULT), default=str, indent=2)


@mcp.resource("swiftroute://lessons")
def lessons_resource() -> str:
    """Promoted lessons, exactly as they're added to the phone agent's prompt."""
    return json.dumps(store.promoted(), indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--trust-caller-id", action="store_true")
    ap.add_argument("--model", default=None, help="model used by the learning tools (default qwen2.5:3b)")
    a = ap.parse_args()
    STATE["trust_caller_id"], STATE["model"] = a.trust_caller_id, a.model
    if a.http:
        mcp.settings.port = a.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
