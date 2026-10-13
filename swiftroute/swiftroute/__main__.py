"""CLI:  python -m swiftroute {seed|sample|simulate|resolve|demo|chat}"""
from __future__ import annotations

import argparse
import json

from . import ids
from .db import DEFAULT_DB, connect, existing_ids, seed
from .simulator import Simulator
from .tools import CallSession


def main() -> None:
    ap = argparse.ArgumentParser(prog="swiftroute")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed"); s.add_argument("--parcels", type=int, default=250)
    s = sub.add_parser("simulate"); s.add_argument("--ticks", type=int, default=5)
    s = sub.add_parser("resolve"); s.add_argument("spoken")
    sub.add_parser("sample", help="print a few parcels to test with")
    sub.add_parser("demo", help="scripted call through the tool layer")
    s = sub.add_parser("chat", help="talk to the agent by text (needs Ollama running)")
    s.add_argument("--model", default=None, help="default qwen2.5:7b or $SWIFTROUTE_MODEL")
    s.add_argument("--url", default=None, help="default http://localhost:11434/v1")
    s.add_argument("--status", default="DELIVERY_EXCEPTION", help="pick a parcel in this status to play with")
    s.add_argument("--debug", action="store_true", help="print tool calls and results")
    s.add_argument("--team", action="store_true", help="sub-agents coach the agent (goal tracker, supervisor, scribe)")
    a = ap.parse_args()
    conn = connect(a.db)

    if a.cmd == "chat":
        return chat(conn, a)
    if a.cmd == "seed":
        seed(conn, n_parcels=a.parcels)
        print(f"seeded {a.parcels} parcels into {a.db}")
    elif a.cmd == "simulate":
        sim = Simulator(conn)
        sim.subscribe(lambda t, st, d: st == "DELIVERY_EXCEPTION" and print(f"  -> outbound-call candidate {t} ({d})"))
        for i in range(a.ticks):
            print(f"tick {i + 1}: {len(sim.tick())} parcels moved")
    elif a.cmd == "resolve":
        raw = ids.parse_spoken(a.spoken)
        r = ids.resolve(raw, lambda b: existing_ids(conn, b))
        print(json.dumps({"parsed": raw, "heard": r.heard, "status": r.status,
                          "candidates": [vars(c) for c in r.candidates],
                          "read_back": ids.read_back(r.best) if r.best else None}, indent=2))
    elif a.cmd == "sample":
        q = ("SELECT p.tracking_id, p.status, a.postal, c.phone FROM parcels p "
             "JOIN addresses a ON a.id = p.address_id JOIN customers c ON c.id = p.customer_id "
             "WHERE p.status = ? LIMIT 1")
        for st in ("IN_TRANSIT", "AT_DEPOT", "OUT_FOR_DELIVERY", "DELIVERY_EXCEPTION", "DELIVERED"):
            r = conn.execute(q, (st,)).fetchone()
            if r:
                print(f"{r['tracking_id']}  {r['status']:<20} postal={r['postal']}  phone={r['phone']}")
    elif a.cmd == "demo":
        r = conn.execute("SELECT p.tracking_id, a.postal, c.phone FROM parcels p JOIN addresses a "
                         "ON a.id = p.address_id JOIN customers c ON c.id = p.customer_id "
                         "WHERE p.status = 'DELIVERY_EXCEPTION' LIMIT 1").fetchone()
        tid = r["tracking_id"]
        # simulate a mishearing: swap one body char for a sound-alike
        garbled = tid[:5] + ("D" if tid[5] != "D" else "B") + tid[6:]
        spoken = " ".join(garbled)
        sess = CallSession(conn, caller_phone=r["phone"])
        steps = [("find_parcel", {"spoken_tracking_id": spoken})]
        for name, args in steps:
            print(f"\n> {name}({args})\n{json.dumps(sess.call(name, args), indent=2, default=str)}")
        sess.turn += 1          # the caller answers "yes" to the read-back
        follow = [("confirm_parcel", {"tracking_id": (sess.pending_ids or [tid])[0]}),
                  ("verify_caller", {"spoken_postal_code": " ".join(r["postal"])}),
                  ("get_status", {}),
                  ("change_address", {"line1": "1 Test St", "city": "Toronto", "spoken_postal_code": "M5V 2T6"}),
                  ("hold_at_depot", {})]
        for name, args in follow:
            print(f"\n> {name}({args})\n{json.dumps(sess.call(name, args), indent=2, default=str)}")
        print(f"\nreason codes: {sess.reason_codes}  resolved={sess.resolved}")


def chat(conn, a) -> None:
    from .agent import Agent
    from .llm import OpenAICompatLLM

    r = conn.execute("SELECT p.tracking_id, p.status, a.postal, c.phone FROM parcels p JOIN addresses a "
                     "ON a.id = p.address_id JOIN customers c ON c.id = p.customer_id WHERE p.status = ? "
                     "LIMIT 1", (a.status,)).fetchone()
    if not r:
        raise SystemExit(f"No parcel in status {a.status}. Run: python -m swiftroute seed")
    llm = OpenAICompatLLM(model=a.model, base_url=a.url)
    sess = CallSession(conn, caller_phone=r["phone"])
    agent = Agent(llm, sess)
    if a.team:
        from .agents.team import CallTeam, save_review
        agent = CallTeam(agent)
    print(f"model={llm.model}  You're calling from the customer's phone.\n"
          f"Your parcel: {r['tracking_id']} ({r['status']}), delivery postal code {r['postal']}.\n"
          f"Try reading the number with a mistake in it. Ctrl-C to hang up.\n")
    print("Sam: Thanks for calling SwiftRoute, this is Sam. How can I help?")
    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in ("quit", "exit", "bye"):
                break
            n_audit = conn.execute("SELECT COUNT(*) FROM audit WHERE call_id = ?", (sess.call_id,)).fetchone()[0]
            try:
                reply = agent.respond(user)
            except RuntimeError as e:
                raise SystemExit(f"\n{e}")
            m = agent.metrics[-1]
            if a.debug:
                for row in conn.execute("SELECT tool, args, result FROM audit WHERE call_id = ? LIMIT -1 OFFSET ?",
                                        (sess.call_id, n_audit)):
                    print(f"   [{row['tool']}] {row['args']} -> {row['result'][:200]}")
            if a.debug and a.team:
                for msg in agent.bus.log[-12:]:
                    if msg.kind in ("goal", "notes") and any(msg.body.values()):
                        print(f"   <{msg.sender} -> {msg.to}> {msg.kind}: {json.dumps(msg.body)[:200]}")
            print(f"Sam: {reply}")
            print(f"   ({m.llm_calls} LLM calls, {sum(m.llm_ms):.0f} ms LLM, "
                  f"{max(m.prompt_tokens or [0])} prompt tok, tools={m.tool_calls})")
    except (KeyboardInterrupt, EOFError):
        pass
    print(f"\ncall ended. reason codes: {sess.reason_codes}")
    if a.team:
        rv = agent.review()
        save_review(rv)
        print(f"supervisor review: goal={rv.goal} handled={rv.goal_handled} issues={[i['kind'] for i in rv.issues]}")


if __name__ == "__main__":
    main()
