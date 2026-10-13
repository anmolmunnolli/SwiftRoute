"""How SwiftRoute's agents talk to each other: typed messages over a small in-process bus.

    bus = Bus(routes={"front_desk": {"goal_tracker", "supervisor"}, ...})
    bus.register(GoalTracker())
    reply = bus.request("front_desk", "goal_tracker", "caller.said", {"text": "..."})

Same principle as the tools: what agents may do to each other is enforced in code,
not in their prompts.
  * Routes are an allow-list. A sub-agent can only message the agents it was wired to,
    so a lesson writer can't talk to the promoter and skip the evaluator.
  * Every message has a hop count. A reply chain that bounces more than MAX_HOPS times
    is dropped (two LLM agents politely revising each other forever is a real failure).
  * Every message is kept in `bus.log` and emitted as a trace event, so the inspector
    and the tests see the whole conversation between agents, not just the outcome.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .. import trace

MAX_HOPS = 12

_ids = itertools.count(1)


@dataclass
class Message:
    sender: str
    to: str
    kind: str                       # e.g. "caller.said", "turn.done", "lesson.draft", "lesson.critique"
    body: dict = field(default_factory=dict)
    reply_to: int | None = None     # id of the message this answers
    hops: int = 0
    id: int = field(default_factory=lambda: next(_ids))
    ts: float = field(default_factory=time.time)

    def reply(self, kind: str, body: dict | None = None) -> "Message":
        return Message(self.to, self.sender, kind, body or {}, reply_to=self.id, hops=self.hops + 1)


class Handler(Protocol):
    name: str

    def handle(self, msg: Message, bus: "Bus") -> Message | None: ...


class RouteError(PermissionError):
    pass


class Bus:
    def __init__(self, routes: dict[str, set[str]] | None = None, log_path: Path | None = None):
        self.routes = routes                    # None = anyone may message anyone (tests, ad-hoc use)
        self.agents: dict[str, Handler] = {}
        self.log: list[Message] = []
        self.dropped: list[tuple[Message, str]] = []
        self.log_path = log_path

    def register(self, agent: Handler) -> Handler:
        self.agents[agent.name] = agent
        if hasattr(agent, "bus"):
            agent.bus = self
        return agent

    def allowed(self, sender: str, to: str) -> bool:
        return self.routes is None or to in self.routes.get(sender, set())

    def send(self, msg: Message) -> Message | None:
        """Deliver one message and return the recipient's reply (if any). Replies travel back
        along the route they came from, so they don't need a route of their own."""
        if msg.reply_to is None and not self.allowed(msg.sender, msg.to):
            self._drop(msg, f"route {msg.sender} -> {msg.to} not allowed")
            raise RouteError(f"{msg.sender} may not message {msg.to}")
        if msg.hops > MAX_HOPS:
            self._drop(msg, f"more than {MAX_HOPS} hops")
            return None
        self._record(msg)
        agent = self.agents.get(msg.to)
        if agent is None:                       # the caller of request() reads the reply itself
            return None
        out = agent.handle(msg, self)
        if out is not None:
            if out.reply_to is None:
                out.reply_to, out.hops = msg.id, msg.hops + 1
            self._record(out)
        return out

    def request(self, sender: str, to: str, kind: str, body: dict | None = None) -> dict:
        out = self.send(Message(sender, to, kind, body or {}))
        return out.body if out else {}

    def tell(self, sender: str, to: str, kind: str, body: dict | None = None) -> None:
        self.send(Message(sender, to, kind, body or {}))

    def conversation(self, a: str, b: str) -> list[Message]:
        return [m for m in self.log if {m.sender, m.to} == {a, b}]

    # ------------------------------------------------------------ records
    def _record(self, msg: Message) -> None:
        self.log.append(msg)
        trace.emit("agent.msg", id=msg.id, sender=msg.sender, to=msg.to, msg_kind=msg.kind,
                   reply_to=msg.reply_to, body=_short(msg.body))
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(asdict(msg), default=str) + "\n")

    def _drop(self, msg: Message, why: str) -> None:
        self.dropped.append((msg, why))
        trace.emit("agent.dropped", sender=msg.sender, to=msg.to, msg_kind=msg.kind, why=why)


def _short(body: dict[str, Any], limit: int = 400) -> dict:
    s = json.dumps(body, default=str)
    return body if len(s) <= limit else {"_truncated": s[:limit]}
