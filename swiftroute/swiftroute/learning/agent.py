"""The improvement agent: an MCP client that tries to make the phone agent better.

    python -m swiftroute.learning.agent                              # LLM-driven (qwen2.5:7b reflects, 3B is improved)
    python -m swiftroute.learning.agent --driver playbook            # deterministic, no reflector LLM
    python -m swiftroute.learning.agent --target qwen2.5:3b --reflector qwen2.5:7b --max-lessons 2

It starts the SwiftRoute MCP server (swiftroute/mcp_server.py) and works only through
its tools, the same way any outside agent would:

    run_baseline -> list_failures / trace_findings -> propose_lesson -> evaluate_lesson

The reflector model decides which failure to work on and writes the lesson. It can't
promote anything itself: evaluate_lesson runs the eval suite with and without the lesson
and promotes only if the gate in swiftroute/learning/lessons.py accepts it.

Two drivers:
  llm       the reflector LLM chooses the tool calls (agentic; needs Ollama)
  playbook  a fixed plan with lesson templates per failing scenario (deterministic; for
            CI and for machines without a model server). It still goes through MCP and the gate.

Held-out scenario types (prompt_injection, second_parcel_claim, withheld_caller_id) are
never used as evidence for a lesson, so the gate's held-out check measures transfer,
not memorisation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from swiftroute.learning import store  # noqa: E402

AGENT_TOOLS = ["run_baseline", "list_failures", "trace_findings", "list_lessons", "propose_lesson", "evaluate_lesson"]

SYSTEM = """You improve a phone support agent for a parcel courier by writing short lessons for its prompt.
Work only through the tools. Plan:
1. run_baseline once.
2. list_failures (and trace_findings) to see what goes wrong. Ignore failures marked held_out.
3. Pick the failing scenario with the most failed runs. Read its reasons and transcript tail.
4. Write ONE lesson: one sentence, under 200 characters, telling the agent what to do in that kind
   of situation (which tool to call, when). Never weaken a rule: no skipping verification, no sharing
   personal details, no fees. Call propose_lesson with it and the evidence.
5. Call evaluate_lesson with the returned id. It decides whether the lesson is kept.
6. Repeat 3-5 for a different failure until you have evaluated {max_lessons} lessons, then reply DONE
   with a one-line summary."""

PLAYBOOK = {
    "status_happy": "Once the caller is verified and they asked where the parcel is, call get_status straight away "
                    "and tell them the status and ETA it returns.",
    "hold_at_depot": "If the caller wants to pick the parcel up or have it held, call hold_at_depot right after "
                     "verification instead of asking again.",
    "damaged_claim": "If the caller says the parcel arrived damaged, call open_claim with kind damaged right after "
                     "verification.",
    "missing_too_early": "If the caller reports a parcel missing, call open_claim with kind missing; the tool decides "
                         "if it's too early.",
    "wants_human": "If the caller asks for a person, call transfer_to_human right away; don't ask for details first.",
    "reschedule_friday": "For a new delivery day, call reschedule_delivery with the caller's own words for the day.",
    "reschedule_weekend_then_monday": "If a reschedule is refused, offer the bookable dates and call "
                                      "reschedule_delivery again with the day the caller picks.",
    "misheard_id": "After the caller confirms the read-back, call confirm_parcel before anything else.",
    "address_change_ok": "For an address change, call change_address with the street, city and postal code the "
                         "caller gave.",
    "address_change_locked": "For an address change, call change_address; if it's refused, explain the reason it "
                             "gives and offer hold_at_depot.",
}


def _openai_tools(mcp_tools) -> list[dict]:
    return [{"type": "function", "function": {"name": t.name, "description": (t.description or "")[:600],
                                              "parameters": t.inputSchema}}
            for t in mcp_tools if t.name in AGENT_TOOLS]


class Run:
    def __init__(self, session: ClientSession, target: str, repeat: int, rounds: int, verbose: bool = True):
        self.s, self.target, self.repeat, self.rounds, self.verbose = session, target, repeat, rounds, verbose
        self.evaluated: list[dict] = []

    def say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    async def call(self, name: str, args: dict) -> object:
        if name in ("run_baseline", "evaluate_lesson"):
            args = {"model": self.target, "repeat": self.repeat, **args}
            if name == "evaluate_lesson":
                args["rounds"] = self.rounds
        self.say(f"  -> {name}({json.dumps({k: v for k, v in args.items() if k != 'evidence'})[:160]})")
        res = await self.s.call_tool(name, args)
        text = "".join(getattr(c, "text", "") for c in res.content)
        try:
            out = json.loads(text)
        except ValueError:
            out = text
        if isinstance(res.structuredContent, dict) and "result" in res.structuredContent:
            out = res.structuredContent["result"]
        if name == "evaluate_lesson":
            self.evaluated.append(out)
            self.say(f"     gate: {'ACCEPT' if out.get('accept') else 'REJECT'} ({out.get('passed_before')} -> "
                     f"{out.get('passed_after')} of {out.get('runs')} runs) {out.get('why')}")
        return out


async def drive_playbook(run: Run, max_lessons: int) -> None:
    base = await run.call("run_baseline", {})
    run.say(f"  baseline: {base['totals']['passed']}/{base['totals']['runs']} runs pass")
    failures = await run.call("list_failures", {})
    tried = {x["text"] for x in await run.call("list_lessons", {})}
    for f in failures:
        if len(run.evaluated) >= max_lessons:
            break
        if f.get("held_out") or f["scenario"] not in PLAYBOOK or PLAYBOOK[f["scenario"]] in tried:
            continue
        p = await run.call("propose_lesson", {"text": PLAYBOOK[f["scenario"]],
                                              "evidence": json.dumps({k: f[k] for k in ("scenario", "reasons")})})
        if p.get("status") == "rejected":
            continue
        await run.call("evaluate_lesson", {"lesson_id": p["id"]})


async def drive_llm(run: Run, reflector, max_lessons: int, tools: list[dict], max_steps: int = 24) -> None:
    messages = [{"role": "system", "content": SYSTEM.format(max_lessons=max_lessons)},
                {"role": "user", "content": "Start."}]
    for _ in range(max_steps):
        reply = await asyncio.to_thread(reflector.chat, messages, tools)
        messages.append({"role": "assistant", "content": reply.content or "", "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for c in reply.tool_calls]} if reply.tool_calls else {"role": "assistant", "content": reply.content or ""})
        if not reply.tool_calls:
            run.say(f"  reflector: {reply.content}")
            if len(run.evaluated) >= max_lessons or "DONE" in (reply.content or ""):
                return
            messages.append({"role": "user", "content": "Continue with the plan using the tools."})
            continue
        for c in reply.tool_calls:
            if c["name"] not in AGENT_TOOLS:
                out = {"error": f"{c['name']} isn't available to you"}
            else:
                out = await run.call(c["name"], c["arguments"] or {})
            messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                             "content": json.dumps(out, default=str)[:3500]})
        if len(run.evaluated) >= max_lessons:
            return


async def main_async(a) -> dict:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    params = StdioServerParameters(command=sys.executable, args=["-m", "swiftroute.mcp_server", "--model", a.target],
                                   cwd=str(ROOT), env=env)
    store.log({"event": "agent.start", "driver": a.driver, "target": a.target, "reflector": a.reflector})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            run = Run(s, a.target, a.repeat, a.rounds)
            print(f"improvement agent: driver={a.driver} target={a.target} via MCP ({len(tools)} tools)")
            if a.driver == "playbook":
                await drive_playbook(run, a.max_lessons)
            else:
                from swiftroute.llm import OpenAICompatLLM
                await drive_llm(run, OpenAICompatLLM(model=a.reflector, max_tokens=400), a.max_lessons,
                                _openai_tools(tools))
    summary = {"evaluated": len(run.evaluated), "accepted": sum(bool(x.get("accept")) for x in run.evaluated),
               "promoted_now": store.promoted()}
    store.log({"event": "agent.end", **summary})
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", choices=["llm", "playbook"], default="llm")
    ap.add_argument("--target", default="qwen2.5:3b", help="the phone-agent model being improved")
    ap.add_argument("--reflector", default="qwen2.5:7b", help="the model that writes lessons (llm driver)")
    ap.add_argument("--max-lessons", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=2, help="runs per scenario per evaluation")
    ap.add_argument("--rounds", type=int, default=2, help="independent evaluation rounds per lesson")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
