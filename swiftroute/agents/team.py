"""The call team: the phone agent works with sub-agents during a live call.

    front_desk     the existing Agent (agent.py): the only one that talks to the caller
    goal_tracker   remembers what the caller asked for, in their words, across the call
    supervisor     watches every turn and coaches the front desk through CALL STATE notes
    scribe         writes the hand-off for the human colleague when the call is transferred

Why: the eval log says qwen2.5:3b "forgets the caller's goal after verification" (it asks
"how can I help?" after the caller said "my parcel arrived damaged" four turns earlier),
and real calls showed it repeating itself. Both are things a second pair of eyes catches
cheaply. The goal tracker and supervisor are deterministic by default (no extra LLM call,
no added latency); the goal tracker can use a small LLM for utterances its patterns miss.

What the supervisor does, in escalating order:
  1. notes and the "Next:" line in CALL STATE (the front desk reads them on every LLM call)
  2. reviews each final reply before it's spoken, and sends it back once if it ignores the goal
  3. if the turn still ends asking "how can I help?", the caller hears a yes/no offer instead
     ("you'd like me to hold it at the depot?"); a yes, or the caller asking again in their own
     words, has the front desk run that tool in code, the way slots.py runs ID and
     postal-code turns. The LLM only phrases the result.
What the sub-agents can NOT do: call tools themselves, change the session, or speak to the caller.
Every tool still goes through CallSession and policy.py, so a wrong judgement can at worst
cost a turn or run an action the caller asked for twice; never skip a safety check.

    team = CallTeam(Agent(llm, session))
    reply = team.respond("my parcel arrived damaged")      # same interface as Agent
    team.review()                                         # the supervisor's after-call review
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .. import dates, trace
from ..agent import Agent
from ..llm import LLM
from .bus import Bus, Message

ROUTES = {
    "front_desk": {"goal_tracker", "supervisor"},
    "supervisor": {"goal_tracker", "scribe"},
}

# goal -> (what the caller wants, the tool that handles it, reason-code prefixes that mean it was handled)
GOALS = {
    "human": ("to speak to a person", "transfer_to_human", ("TRANSFER",)),
    "claim_damaged": ("to report the parcel arrived damaged", "open_claim", ("CLAIM_",)),
    "claim_missing": ("to report the parcel missing", "open_claim", ("CLAIM_",)),
    "change_address": ("to send the parcel to a different address", "change_address", ("ADDR_",)),
    "hold_at_depot": ("to have the parcel held at a depot for pickup", "hold_at_depot", ("HOLD_",)),
    "reschedule": ("a different delivery day", "reschedule_delivery", ("RESCHED_",)),
    "status": ("to know where the parcel is / when it arrives", "get_status", ("STATUS_",)),
}
_PATTERNS = [   # order matters: the most specific request wins
    ("human", r"\b(real person|human|representative|operator|speak to (someone|a person)|talk to (someone|a person))\b"),
    ("claim_damaged", r"\b(damaged|crushed|broken|smashed|squashed|ripped|torn|dented|wet)\b"),
    ("claim_missing", r"\b(missing|never (came|arrived|got it)|lost|stolen|didn'?t (get|receive) it)\b"),
    ("change_address", r"\b(address|redirect|send (my |the )?(parcel|package|it) to|deliver it to)\b"),
    ("hold_at_depot", r"\b(depot|pick (it )?up|pickup|collect it|hold it)\b"),
    ("reschedule", r"\b(reschedul\w*|redeliver\w*|missed (my |the )?delivery|another day|instead|"
                   r"(on )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b"),
    ("status", r"\b(where('s| is)|when('s| is| will)|status|arriv\w*|tracking (update|info)|update on|eta)\b"),
]
_COMPILED = [(g, re.compile(p, re.I)) for g, p in _PATTERNS]
_ASKS_WHAT = re.compile(r"\b(how can i help|what can i (do|help)|anything else|what would you like|"
                        r"what do you need|how may i help)\b", re.I)
_YES = re.compile(r"^\W*(yes|yeah|yep|sure|please|ok(ay)?|go ahead|right|correct|that'?s right)\b", re.I)
# What the supervisor offers when the front desk dropped the goal even after a correction.
# Only goals whose tool arguments are certain; the caller's yes is the consent to run it in code.
OFFERS = {
    "status": "Would you like me to check where your parcel is now?",
    "hold_at_depot": "Just to confirm, you'd like me to hold the parcel at the depot for pickup?",
    "claim_damaged": "Just to confirm, you'd like me to open a damage claim for this parcel?",
    "claim_missing": "Just to confirm, you'd like me to report this parcel missing?",
}
_FRUSTRATED = re.compile(r"\b(already (told|said|gave)|i just said|again\?|ridiculous|useless|not listening|"
                         r"for the (second|third) time)\b", re.I)


def classify(text: str) -> str | None:
    for goal, rx in _COMPILED:
        if rx.search(text or ""):
            return goal
    return None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower()).strip()


class GoalTracker:
    """Keeps the caller's current request. A new request replaces the old one once the old
    one has been handled (or immediately, if the caller clearly changed their mind)."""
    name = "goal_tracker"
    LABELS = list(GOALS)

    def __init__(self, llm: LLM | None = None):
        self.llm = llm
        self.goal: str | None = None
        self.said: str = ""
        self.since_codes = 0                    # reason codes before the goal was set
        self.history: list[dict] = []

    def handle(self, msg: Message, bus: Bus) -> Message | None:
        if msg.kind == "caller.said":
            goal = classify(msg.body["text"]) or self._ask_llm(msg.body["text"])
            if goal and goal != self.goal:
                self.goal, self.said = goal, msg.body["text"]
                self.since_codes = msg.body.get("n_codes", 0)
                self.history.append({"turn": msg.body.get("turn"), "goal": goal, "said": self.said})
            return msg.reply("goal", self.state(msg.body.get("codes", [])))
        if msg.kind == "goal.get":
            return msg.reply("goal", self.state(msg.body.get("codes", [])))
        return None

    def state(self, codes: list[str]) -> dict:
        if not self.goal:
            return {"goal": None}
        want, tool, done_prefixes = GOALS[self.goal]
        handled = any(c.startswith(done_prefixes) for c in codes[self.since_codes:])
        return {"goal": self.goal, "want": want, "tool": tool, "said": self.said, "handled": handled}

    def _ask_llm(self, text: str) -> str | None:
        """Only for utterances the patterns miss, and only if a model was given. One word out."""
        if self.llm is None or len(text.split()) < 3:
            return None
        prompt = ("Classify what a parcel-courier caller wants. Reply with exactly one label from: "
                  + ", ".join(self.LABELS) + ", none.")
        try:
            out = self.llm.chat([{"role": "system", "content": prompt}, {"role": "user", "content": text}])
        except RuntimeError:
            return None
        label = _norm(out.content).replace(" ", "_")
        return label if label in GOALS else None


class Supervisor:
    """Watches the front desk turn by turn. Coaches it with short notes and keeps a review."""
    name = "supervisor"

    def __init__(self):
        self.replies: list[str] = []
        self.issues: list[dict] = []
        self.notes_given: list[str] = []
        self.frustration = 0
        self.handoff: dict | None = None
        self._pending: list[str] = []           # one-shot notes for the next turn
        self.dropped_goal: str | None = None    # a whole turn ended with the goal doable but not done
        self.offered_goal: str | None = None    # the supervisor's offer replaced that turn's reply

    def handle(self, msg: Message, bus: Bus) -> Message | None:
        if msg.kind == "notes.request":
            return msg.reply("notes", self._notes(msg.body, bus))
        if msg.kind == "turn.plan":
            return msg.reply("plan", {"plan": self._plan(msg.body, bus)})
        if msg.kind == "reply.check":
            return msg.reply("verdict", {"correction": self._check(msg.body, bus)})
        if msg.kind == "turn.done":
            return msg.reply("turn.review", self._review_turn(msg.body, bus))
        if msg.kind == "caller.said" and _FRUSTRATED.search(msg.body.get("text", "")):
            self.frustration += 1
            self._issue(msg.body.get("turn"), "caller_frustrated", msg.body["text"])
            self._pending.append("The caller sounds frustrated: acknowledge it briefly, don't ask for "
                                 "anything they already gave, and offer a colleague if you can't help.")
        return None

    def _actionable(self, g: dict, level: str) -> bool:
        """The goal is known, not handled yet, and the caller is verified enough for its tool."""
        return bool(g.get("goal")) and not g["handled"] and g["goal"] != "human" and (
            level == "full" or (level == "partial" and g["goal"] == "status"))

    def _notes(self, body: dict, bus: Bus) -> dict:
        """Notes for CALL STATE, and, when the next step is clear, the "Next:" line itself:
        small models follow that line far better than anything else in the prompt."""
        g = bus.request(self.name, "goal_tracker", "goal.get", {"codes": body["codes"]})
        notes, nxt = list(self._pending), None
        level = body["level"]
        if self._actionable(g, level):
            nxt = (f"call {g['tool']} now. The caller already said they want {g['want']} "
                   f"(they said: \"{g['said'][:90]}\"); don't ask what they need.")
        elif g.get("goal") and not g["handled"] and g["goal"] != "human":
            if level == "partial":
                notes.append(f"The caller wants {g['want']}, which needs full verification: explain they "
                             "must call from the phone number on the account. Status is still allowed.")
            elif level == "not checked":
                notes.append(f"The caller wants {g['want']}. Do it as soon as they're verified; don't ask again.")
        return {"notes": notes, "next": nxt}

    def _plan(self, body: dict, bus: Bus) -> list | None:
        """Escalation after coaching failed. The front desk had a full turn with the goal known and the
        caller verified, and didn't act; now the caller asks again in their own words. Run the tool in
        code (policy still decides), as slots.py does for codes. Only goals whose arguments are certain:
        an address change needs details the model has to collect, so it is never planned."""
        g = bus.request(self.name, "goal_tracker", "goal.get", {"codes": body["codes"]})
        accepted_offer = self.offered_goal == g.get("goal") and bool(_YES.match(body["text"]))
        self.offered_goal = None
        if not (self.dropped_goal and self._actionable(g, body["level"]) and g["goal"] == self.dropped_goal
                and (classify(body["text"]) == g["goal"] or accepted_offer)):
            return None
        args = {"status": {}, "hold_at_depot": {},
                "claim_damaged": {"kind": "damaged", "description": g["said"][:120]},
                "claim_missing": {"kind": "missing", "description": g["said"][:120]}}.get(g["goal"])
        if g["goal"] == "reschedule" and dates.parse_day(body["text"], date.fromisoformat(body["today"])):
            args = {"day": body["text"]}
        if args is None:
            return None
        self._issue(body.get("turn"), "goal_run_in_code", body["text"], goal=g["goal"])
        return [g["tool"], args]

    def _check(self, body: dict, bus: Bus) -> str | None:
        """Before a reply is spoken: is the front desk asking what the caller wants when it already knows?"""
        g = bus.request(self.name, "goal_tracker", "goal.get", {"codes": body["codes"]})
        if self._actionable(g, body["level"]) and _ASKS_WHAT.search(body["reply"]):
            self._issue(body.get("turn"), "forgot_goal_caught", body["reply"], goal=g["goal"])
            return (f"The caller already said they want {g['want']}. Don't ask what they need: call {g['tool']} now.")
        return None

    def _review_turn(self, body: dict, bus: Bus) -> dict:
        turn, reply = body["turn"], body["reply"]
        self._pending = []
        if self.replies and _norm(reply) and _norm(reply) == _norm(self.replies[-1]) and not body["fast_path"]:
            self._issue(turn, "repeated_reply", reply)
            self._pending.append(
                "You said exactly the same thing last turn. Don't repeat it word for word: offer to connect them "
                "with a colleague." if body["level"] == "failed" else
                "You said exactly the same thing last turn. Don't repeat it: either call the tool the caller "
                "needs, or ask a different, specific question.")
        g = bus.request(self.name, "goal_tracker", "goal.get", {"codes": body["codes"]})
        actionable = self._actionable(g, body["level"])
        self.dropped_goal = g["goal"] if actionable else None
        out: dict = {}
        if actionable and _ASKS_WHAT.search(reply):
            self._issue(turn, "forgot_goal", reply, goal=g["goal"])
            if g["goal"] in OFFERS:              # coaching failed twice this turn: offer it instead
                self._issue(turn, "offered_goal", OFFERS[g["goal"]], goal=g["goal"])
                self.offered_goal = g["goal"]
                out = {"replace_reply": OFFERS[g["goal"]]}
                reply = out["replace_reply"]
        for kind in ("nudges", "guard_fallbacks", "bad_tool_calls"):
            if body["metrics"].get(kind):
                self._issue(turn, kind, reply, n=body["metrics"][kind])
        self.replies.append(reply)
        if "TRANSFER" in body["new_codes"] and self.handoff is None:
            self.handoff = bus.request(self.name, "scribe", "handoff.write",
                                       {**body, "goal": g, "issues": self.issues})
        return out

    def _issue(self, turn, kind: str, text: str, **extra) -> None:
        self.issues.append({"turn": turn, "kind": kind, "text": (text or "")[:160], **extra})
        trace.emit("supervisor.issue", turn=turn, issue=kind, text=(text or "")[:160])


class Scribe:
    """Writes what the human colleague needs to know, from the session (not the model's memory)."""
    name = "scribe"

    def handle(self, msg: Message, bus: Bus) -> Message | None:
        if msg.kind != "handoff.write":
            return None
        b = msg.body
        g = b.get("goal") or {}
        done = [c for c in b["codes"] if c.endswith("_OK") or c in ("STATUS_GIVEN", "ID_CONFIRMED")]
        lines = [f"Caller wants: {g.get('want', 'not stated')}" + (f' ("{g["said"][:100]}")' if g.get("said") else ""),
                 f"Parcel: {'ending ' + b['parcel'][-4:] if b.get('parcel') else 'not identified'}",
                 f"Verification: {b['level']}",
                 f"Done this call: {', '.join(done) or 'nothing yet'}",
                 f"Why transferred: {b.get('transfer_reason') or 'caller asked'}"]
        if b.get("issues"):
            lines.append("Watch out: " + "; ".join(sorted({i['kind'] for i in b["issues"]})))
        return msg.reply("handoff", {"summary": "\n".join(lines)})


@dataclass
class CallReview:
    call_id: str
    goal: str | None
    goal_handled: bool
    turns: int
    reason_codes: list[str]
    issues: list[dict] = field(default_factory=list)
    notes_given: int = 0
    handoff: str | None = None
    messages: int = 0


class CallTeam:
    """Drop-in for Agent: same respond() / metrics / session, plus the sub-agents."""

    def __init__(self, front_desk: Agent, classifier_llm: LLM | None = None, bus: Bus | None = None):
        self.front_desk = front_desk
        self.bus = bus or Bus(routes=ROUTES)
        self.goal_tracker = self.bus.register(GoalTracker(classifier_llm))
        self.supervisor = self.bus.register(Supervisor())
        self.scribe = self.bus.register(Scribe())
        front_desk.advisor = self._advice
        front_desk.reply_reviewer = self._review_reply
        front_desk.turn_planner = self._plan_turn

    def __getattr__(self, name):                 # history, metrics, session, mark_last_reply_unheard, ...
        return getattr(self.front_desk, name)

    def _level(self) -> str:
        s = self.front_desk.session
        if s.identity and s.identity.allowed:
            return s.identity.data.get("level", "full")
        return "failed" if s.identity else "not checked"

    def _advice(self) -> dict:
        """Called by the front desk every time it builds a prompt, so notes follow state changes
        inside a turn (verification happens mid-turn, then the model decides what to do)."""
        advice = self.bus.request("front_desk", "supervisor", "notes.request",
                                  {"codes": list(self.front_desk.session.reason_codes), "level": self._level()})
        for n in advice.get("notes", []) + ([advice["next"]] if advice.get("next") else []):
            if n not in self.supervisor.notes_given:
                self.supervisor.notes_given.append(n)
        return advice

    def _plan_turn(self, text: str) -> tuple[str, dict] | None:
        s = self.front_desk.session
        plan = self.bus.request("front_desk", "supervisor", "turn.plan",
                                {"text": text, "codes": list(s.reason_codes), "level": self._level(),
                                 "turn": s.turn, "today": s._now().date().isoformat()}).get("plan")
        return tuple(plan) if plan else None

    def _review_reply(self, reply: str) -> str | None:
        """The supervisor sees a final reply before the caller does and can send it back once.
        It uses the front desk's existing one-correction-per-turn budget."""
        s = self.front_desk.session
        return self.bus.request("front_desk", "supervisor", "reply.check",
                                {"reply": reply, "codes": list(s.reason_codes), "level": self._level(),
                                 "turn": s.turn}).get("correction")

    def respond(self, user_text: str) -> str:
        s = self.front_desk.session
        n_codes = len(s.reason_codes)
        said = {"text": user_text, "turn": s.turn + 1, "n_codes": n_codes, "codes": list(s.reason_codes)}
        self.bus.request("front_desk", "goal_tracker", "caller.said", said)
        self.bus.tell("front_desk", "supervisor", "caller.said", said)
        reply = self.front_desk.respond(user_text)
        m = self.front_desk.metrics[-1]
        audit = s.conn.execute("SELECT args FROM audit WHERE call_id = ? AND tool = 'transfer_to_human' "
                               "ORDER BY id DESC LIMIT 1", (s.call_id,)).fetchone()
        review = self.bus.request("front_desk", "supervisor", "turn.done", {
            "turn": m.turn, "reply": reply, "codes": list(s.reason_codes), "new_codes": s.reason_codes[n_codes:],
            "level": self._level(), "parcel": s.tracking_id, "fast_path": m.fast_path and not m.llm_calls,
            "metrics": {"nudges": m.nudges, "guard_fallbacks": m.guard_fallbacks, "bad_tool_calls": m.bad_tool_calls},
            "transfer_reason": json.loads(audit[0]).get("reason") if audit else None})
        if review.get("replace_reply"):
            # the front desk dropped the caller's goal even after a correction: the caller hears the
            # supervisor's yes/no offer instead, and a yes runs the tool in code next turn
            reply = review["replace_reply"]
            for msg in reversed(self.front_desk.history):
                if msg["role"] == "assistant" and not msg.get("tool_calls"):
                    msg["content"] = reply
                    break
        return reply

    def review(self) -> CallReview:
        s = self.front_desk.session
        g = self.goal_tracker.state(s.reason_codes)
        return CallReview(s.call_id, g.get("goal"), bool(g.get("handled")), len(self.front_desk.metrics),
                          list(s.reason_codes), list(self.supervisor.issues), len(self.supervisor.notes_given),
                          (self.supervisor.handoff or {}).get("summary"), len(self.bus.log))


def save_review(review: CallReview, base: Path | None = None) -> None:
    """Reviews feed the improvement crew (crew.py): failures from real calls, not just evals."""
    from ..learning import store
    with store._path("reviews.jsonl", base).open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **review.__dict__}, default=str) + "\n")


def load_reviews(base: Path | None = None, limit: int = 200) -> list[dict]:
    from ..learning import store
    p = store._path("reviews.jsonl", base)
    return [json.loads(x) for x in p.read_text().splitlines()[-limit:]] if p.exists() else []
