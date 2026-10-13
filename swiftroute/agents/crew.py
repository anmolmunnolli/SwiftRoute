"""The improvement crew: agents that talk to each other to make the phone agent better between calls.

    python -m swiftroute.agents.crew --driver playbook                    # deterministic, no reflector model
    python -m swiftroute.agents.crew --target qwen2.5:3b --writer qwen2.5:7b --team --cycles 2
    python -m swiftroute.agents.crew --curate                             # also re-test promoted lessons

    coordinator ──► analyst ──► evaluator            "what's failing, and what did we already try?"
         │
         ├──► writer ◄──► critic                     draft, critique, revise (up to MAX_REVISIONS)
         │
         ├──► evaluator                              the code gate (lessons.py): the only one that promotes
         │
         └──► curator ──► evaluator                  ablate promoted lessons; retire ones that stopped helping

Each arrow is a message on the bus (bus.py), allowed by ROUTES and nothing else. The writer
can't reach the evaluator, so no lesson gets tested without passing the critic, and nothing
gets promoted without passing the gate.

How it improves itself, not just the phone agent:
  * Memory (data/learning/crew_memory.json). Every attempt is kept with the gate's verdict.
    The analyst stops spending eval minutes on a scenario after MAX_ATTEMPTS rejected lessons,
    and the writer is shown its own rejected drafts and why, so it doesn't propose them again.
  * The critic's feedback is a conversation, not a filter: a rejected draft goes back to the
    writer with the reason, and the writer revises.
  * The curator removes lessons that no longer earn their place (a later lesson or a model
    change can make one redundant). Fewer lessons = a shorter, more stable prompt.
  * Evidence includes the call team's supervisor reviews (team.py) from real calls, not only
    eval failures.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable

from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from swiftroute.agents.bus import Bus, Message  # noqa: E402
from swiftroute.agents.team import load_reviews  # noqa: E402
from swiftroute.learning import lessons, store  # noqa: E402
from swiftroute.learning.agent import PLAYBOOK  # noqa: E402
from swiftroute.llm import LLM  # noqa: E402

ROUTES = {
    "coordinator": {"analyst", "writer", "evaluator", "curator"},
    "analyst": {"evaluator"},
    "writer": {"critic"},
    "curator": {"evaluator"},
}
MAX_REVISIONS = 3
MAX_ATTEMPTS = 3            # rejected lessons per scenario before the analyst moves on
TOOL_NAMES = ("find_parcel", "confirm_parcel", "verify_caller", "get_status", "reschedule_delivery",
              "change_address", "hold_at_depot", "open_claim", "transfer_to_human")

# Second phrasings, used by the playbook writer after the first was rejected or criticised.
PLAYBOOK_ALT = {
    "status_happy": "When a verified caller asked where their parcel is, call get_status in your very next reply.",
    "hold_at_depot": "When the caller asked to collect the parcel, call hold_at_depot as soon as they're verified.",
    "damaged_claim": "When a verified caller says the parcel was damaged, call open_claim with kind damaged; "
                     "don't ask them to repeat it.",
    "second_parcel_claim": "If a verified caller reports a problem with another parcel, call find_parcel with "
                           "its number before any other tool.",
    "missing_too_early": "When the caller says the parcel never came, call open_claim with kind missing right "
                         "after verification.",
}


# ================================================================= memory
class Memory:
    """What the crew has tried, per scenario, and what the gate said."""

    def __init__(self, base: Path | None = None):
        self.path = store._path("crew_memory.json", base)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"attempts": {}, "cycles": []}

    def attempts(self, scenario: str) -> list[dict]:
        return self.data["attempts"].get(scenario, [])

    def rejected_count(self, scenario: str) -> int:
        return sum(not a["accepted"] for a in self.attempts(scenario))

    def remember(self, scenario: str, text: str, accepted: bool, why: str, gain: int | None) -> None:
        self.data["attempts"].setdefault(scenario, []).append(
            {"text": text, "accepted": accepted, "why": why, "gain": gain, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))


# ================================================================= agents
class Agent:
    name = "agent"
    bus: Bus | None = None

    def ask(self, to: str, kind: str, body: dict | None = None) -> dict:
        return self.bus.request(self.name, to, kind, body)


class Evaluator(Agent):
    """Wraps the code gate. Holds the model factory, so no other agent runs evals."""
    name = "evaluator"

    def __init__(self, make_llm: Callable, repeat: int, rounds: int, team: bool, base: Path | None):
        self.make_llm, self.repeat, self.rounds, self.team, self.base = make_llm, repeat, rounds, team, base
        self.baseline: dict | None = None

    def handle(self, msg: Message, bus: Bus) -> Message | None:
        b = msg.body
        if msg.kind == "baseline.run":
            self.baseline = lessons.score(self.make_llm, store.promoted(self.base), self.repeat, team=self.team)
            store.log({"event": "baseline", "by": "crew", "totals": self.baseline["totals"]}, self.base)
            return msg.reply("baseline", {"totals": self.baseline["totals"],
                                          "failures": lessons.failures(self.baseline)})
        if msg.kind == "lesson.evaluate":
            item = store.add(b["text"], {"by": "crew", "scenario": b["scenario"],
                                         "evidence": json.dumps(b.get("evidence", ""))[:500]}, self.base)
            if item["status"] == "rejected":
                return msg.reply("verdict", {"accept": False, "why": item["reason"], "id": item["id"]})
            r = lessons.evaluate_and_maybe_promote(item["id"], self.make_llm, self.repeat, self.rounds,
                                                   base=self.base, team=self.team)
            return msg.reply("verdict", {k: r.get(k) for k in ("accept", "why", "passed_before", "passed_after",
                                                                "runs")} | {"id": item["id"]})
        if msg.kind == "score":
            s = lessons.score(self.make_llm, b["lessons"], self.repeat, team=self.team)
            return msg.reply("scored", {"totals": s["totals"]})
        return None


class Analyst(Agent):
    """Decides what to work on: failing scenarios, ranked by evidence, minus what memory says is hopeless."""
    name = "analyst"

    def __init__(self, memory: Memory, base: Path | None):
        self.memory, self.base = memory, base

    def handle(self, msg: Message, bus: Bus) -> Message | None:
        if msg.kind != "targets.find":
            return None
        base = self.ask("evaluator", "baseline.run")
        field_issues: dict[str, int] = {}
        for rv in load_reviews(self.base):
            for i in rv.get("issues", []):
                if i["kind"] == "forgot_goal" and i.get("goal"):
                    field_issues[i["goal"]] = field_issues.get(i["goal"], 0) + 1
        targets, skipped = [], []
        for f in base["failures"]:
            if f["held_out"]:
                skipped.append({"scenario": f["scenario"], "why": "held out: used only to measure transfer"})
            elif self.memory.rejected_count(f["scenario"]) >= MAX_ATTEMPTS:
                skipped.append({"scenario": f["scenario"],
                                "why": f"{MAX_ATTEMPTS} lessons already rejected: needs a structural fix, not a lesson"})
            else:
                targets.append({**f, "priority": f["failed_runs"] - self.memory.rejected_count(f["scenario"]),
                                "past_attempts": self.memory.attempts(f["scenario"])})
        targets.sort(key=lambda t: -t["priority"])
        return msg.reply("targets", {"baseline": base["totals"], "targets": targets, "skipped": skipped,
                                     "field_forgot_goal": field_issues})


class Critic(Agent):
    """Reviews a draft before any eval time is spent on it. Feedback goes back to the writer."""
    name = "critic"

    def __init__(self, base: Path | None):
        self.base = base

    def handle(self, msg: Message, bus: Bus) -> Message | None:
        if msg.kind != "lesson.review":
            return None
        text = msg.body["text"]
        problems = []
        err = store.validate_lesson(text)
        if err:
            problems.append(f"invalid: {err}. A lesson says what to do, in one short line, and never loosens a rule")
        if not any(t in text for t in TOOL_NAMES):
            problems.append("too vague: name the tool to call (e.g. hold_at_depot) and when to call it")
        if not re.search(r"\b(when|if|after|once|as soon as|before|for an?)\b", text, re.I):
            problems.append("no trigger: say in which situation it applies (start with When/If/After)")
        promoted = store.promoted(self.base)
        for old in promoted:
            if fuzz.token_set_ratio(text, old) >= 85:
                problems.append(f"duplicate of a promoted lesson: \"{old}\"")
        for a in msg.body.get("rejected", []):
            if fuzz.token_set_ratio(text, a["text"]) >= 90:
                problems.append(f"already tried and rejected ({a['why']}): change the approach, not the wording")
        return msg.reply("critique", {"ok": not problems, "problems": problems})


class Writer(Agent):
    """Drafts a lesson, gets it critiqued, revises. LLM-driven or playbook (templates)."""
    name = "writer"

    PROMPT = """You write ONE lesson for a small phone-support model at a parcel courier.
A lesson is one sentence under 200 characters: in which situation, which tool to call, when.
It must never weaken a rule (verification, privacy, no fees). Reply with the lesson only."""

    def __init__(self, llm: LLM | None = None):
        self.llm = llm

    def handle(self, msg: Message, bus: Bus) -> Message | None:
        if msg.kind != "lesson.write":
            return None
        t = msg.body["target"]
        rejected = [a for a in t.get("past_attempts", []) if not a["accepted"]]
        feedback: list[str] = []
        drafts: list[dict] = []
        for _ in range(MAX_REVISIONS):
            text = self._draft(t, rejected, feedback, [d["text"] for d in drafts])
            if not text:
                break
            crit = self.ask("critic", "lesson.review", {"text": text, "scenario": t["scenario"], "rejected": rejected})
            drafts.append({"text": text, **crit})
            if crit["ok"]:
                return msg.reply("lesson", {"text": text, "drafts": drafts})
            feedback = crit["problems"]
        return msg.reply("lesson", {"text": None, "drafts": drafts})

    def _draft(self, t: dict, rejected: list[dict], feedback: list[str], mine: list[str]) -> str | None:
        if self.llm is None:
            tried = {a["text"] for a in rejected} | set(mine)
            for cand in (PLAYBOOK.get(t["scenario"]), PLAYBOOK_ALT.get(t["scenario"])):
                if cand and cand not in tried:
                    return cand
            return None
        user = [f"Failing scenario: {t['scenario']} ({t['failed_runs']}/{t['runs']} runs failed)",
                "Why it failed: " + "; ".join(t.get("reasons", [])),
                "End of a failing call:\n" + "\n".join(t.get("transcript_tail", [])[-8:])]
        if t.get("supervisor_issues"):
            user.append("The supervisor flagged: " + ", ".join(t["supervisor_issues"]))
        if rejected:
            user.append("Lessons already tried and rejected by the eval gate (don't repeat them):\n"
                        + "\n".join(f"- {a['text']} ({a['why']})" for a in rejected[-4:]))
        if mine:
            user.append(f"Your previous draft: {mine[-1]}\nThe critic said: " + "; ".join(feedback)
                        + "\nRevise it to fix every point.")
        try:
            out = self.llm.chat([{"role": "system", "content": self.PROMPT},
                                 {"role": "user", "content": "\n\n".join(user)}])
        except RuntimeError:
            return None
        lines = (out.content or "").strip().splitlines()
        text = re.sub(r"^\W*(lesson:\s*)?", "", lines[0], flags=re.I).strip().strip('"') if lines else ""
        return text or None


class Curator(Agent):
    """Leave-one-out test of every promoted lesson; retire the ones that no longer help."""
    name = "curator"

    def __init__(self, base: Path | None):
        self.base = base

    def handle(self, msg: Message, bus: Bus) -> Message | None:
        if msg.kind != "lessons.curate":
            return None
        items = [x for x in store.load(self.base)["lessons"] if x["status"] == "promoted"]
        if not items:
            return msg.reply("curated", {"retired": [], "kept": []})
        current = [x["text"] for x in items]
        full = self.ask("evaluator", "score", {"lessons": current})["totals"]
        retired, kept = [], []
        for x in items:
            without = self.ask("evaluator", "score", {"lessons": [t for t in current if t != x["text"]]})["totals"]
            # retire only if dropping it loses nothing: same or more passes, no new safety problems
            if (without["passed"] >= full["passed"] and without["safety_failures"] <= full["safety_failures"]
                    and without["false_claims"] <= full["false_claims"]):
                store.update(x["id"], self.base, status="retired",
                             reason=f"curator: no loss without it ({without['passed']} vs {full['passed']} passes)")
                current.remove(x["text"])
                full = without
                retired.append(x["text"])
            else:
                kept.append(x["text"])
        return msg.reply("curated", {"retired": retired, "kept": kept})


class Coordinator(Agent):
    name = "coordinator"

    def __init__(self, memory: Memory, verbose: bool = True):
        self.memory, self.verbose = memory, verbose

    def say(self, s: str) -> None:
        if self.verbose:
            print(s, flush=True)

    def cycle(self, max_lessons: int, curate: bool) -> dict:
        found = self.ask("analyst", "targets.find")
        t = found["baseline"]
        self.say(f"analyst: baseline {t['passed']}/{t['runs']} runs pass; {len(found['targets'])} targets, "
                 f"{len(found['skipped'])} skipped")
        for s in found["skipped"]:
            self.say(f"  skip {s['scenario']}: {s['why']}")
        evaluated = []
        for target in found["targets"]:
            if len(evaluated) >= max_lessons:
                break
            w = self.ask("writer", "lesson.write", {"target": target})
            for d in w["drafts"]:
                self.say(f"  writer -> critic [{target['scenario']}]: {d['text'][:90]!r} "
                         f"{'OK' if d['ok'] else 'REVISE: ' + '; '.join(d['problems'])}")
            if not w["text"]:
                continue
            v = self.ask("evaluator", "lesson.evaluate", {"text": w["text"], "scenario": target["scenario"],
                                                          "evidence": {"reasons": target["reasons"]}})
            gain = (v["passed_after"] - v["passed_before"]) if v.get("passed_after") is not None else None
            self.memory.remember(target["scenario"], w["text"], bool(v["accept"]), v["why"], gain)
            self.say(f"  evaluator: {'PROMOTED' if v['accept'] else 'REJECTED'} ({v.get('passed_before')} -> "
                     f"{v.get('passed_after')} of {v.get('runs')}) {v['why']}")
            evaluated.append({"scenario": target["scenario"], "text": w["text"], **v})
        curated = self.ask("curator", "lessons.curate") if curate else None
        if curated:
            self.say(f"curator: retired {len(curated['retired'])}, kept {len(curated['kept'])}")
        summary = {"baseline": t, "evaluated": evaluated, "accepted": sum(bool(e["accept"]) for e in evaluated),
                   "skipped": found["skipped"], "curated": curated}
        self.memory.data["cycles"].append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "baseline_passed": t["passed"],
                                           "runs": t["runs"], "accepted": summary["accepted"],
                                           "evaluated": len(evaluated)})
        self.memory.save()
        return summary


class Crew:
    def __init__(self, make_llm: Callable, writer_llm: LLM | None = None, repeat: int = 2, rounds: int = 2,
                 team: bool = False, base: Path | None = None, verbose: bool = True):
        self.memory = Memory(base)
        self.bus = Bus(routes=ROUTES, log_path=store._path("crew_messages.jsonl", base))
        self.coordinator = self.bus.register(Coordinator(self.memory, verbose))
        self.bus.register(Analyst(self.memory, base))
        self.bus.register(Writer(writer_llm))
        self.bus.register(Critic(base))
        self.bus.register(Evaluator(make_llm, repeat, rounds, team, base))
        self.bus.register(Curator(base))
        self.base = base

    def run(self, cycles: int = 1, max_lessons: int = 2, curate: bool = False) -> dict:
        store.log({"event": "crew.start", "cycles": cycles}, self.base)
        out = []
        for i in range(cycles):
            self.coordinator.say(f"== cycle {i + 1}/{cycles}")
            out.append(self.coordinator.cycle(max_lessons, curate and i == cycles - 1))
            if not out[-1]["evaluated"]:
                break                              # nothing left worth trying
        summary = {"cycles": out, "promoted_now": store.promoted(self.base), "messages": len(self.bus.log)}
        store.log({"event": "crew.end", "accepted": sum(c["accepted"] for c in out)}, self.base)
        return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", choices=["llm", "playbook"], default="llm")
    ap.add_argument("--target", default="qwen2.5:3b", help="phone-agent model being improved (or rule-agent)")
    ap.add_argument("--writer", default="qwen2.5:7b", help="model that drafts lessons (llm driver)")
    ap.add_argument("--team", action="store_true", help="evaluate the phone agent with its call team")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--max-lessons", type=int, default=2, help="lessons evaluated per cycle")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--curate", action="store_true", help="leave-one-out test of promoted lessons at the end")
    a = ap.parse_args()
    from swiftroute.mcp_server import _make_llm
    from swiftroute.llm import OpenAICompatLLM
    writer = OpenAICompatLLM(model=a.writer, max_tokens=120) if a.driver == "llm" else None
    crew = Crew(_make_llm(a.target), writer, a.repeat, a.rounds, a.team)
    summary = crew.run(a.cycles, a.max_lessons, a.curate)
    print(json.dumps({k: v for k, v in summary.items() if k != "cycles"} | {
        "accepted": sum(c["accepted"] for c in summary["cycles"])}, indent=2))


if __name__ == "__main__":
    main()
