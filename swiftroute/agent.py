"""The conversational agent: one LLM, a tool loop, and context management.

Context design (the voice-specific problem from the session notes):
  * Ground truth lives in CallSession, not in the transcript. Each LLM call
    gets a fresh CALL STATE block rebuilt from the session (confirmed parcel,
    verification level, actions done, bookable dates), so old turns can be
    dropped without the model forgetting anything that matters.
  * History is a sliding window of whole turns; a tool call is never split
    from its result.
  * The CALL STATE block goes at the END of the prompt. The system prompt and
    the older history stay byte-identical between calls, so servers with
    prefix caching (vLLM, llama.cpp, recent Ollama) reuse the KV cache and
    only the tail is recomputed. That's latency we get for free.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

from . import prompting, slots, trace
from .llm import LLM, LLMReply
from .policy import _business_days
from .tools import CallSession

SYSTEM_PROMPT = prompting.SYSTEM_PROMPT_FULL   # moved to prompting.py (phase 6a)

FALLBACK = "Sorry, I'm having trouble with that. Let me get a colleague to help you."
UNVERIFIED_CLAIM_FALLBACK = "Sorry, I wasn't able to complete that. Let me get a colleague to help you."

# Runtime output guards. The eval harness has its own, independent checks; these
# run on every live reply and give the model ONE corrective retry per turn.
_PROMISE = re.compile(
    r"\b(let me|let's|let us|i'll|i will|we'll|we will|i'm going to|i am going to)\b[^.?!]*\b(check|look|schedul|reschedul|"
    r"chang|updat|redirect|hold|open|book|transfer|connect|proceed|verify|do that|take care)"
    r"|^(redirecting|rescheduling|changing|updating|holding|opening|checking|booking)\b"
    r"|\b(one moment|give me a (sec|second|moment))\b", re.I)
_CLAIM = re.compile(
    r"\b((i've|i have|we've|we have) (now |successfully )?(rescheduled|changed|updated|moved|redirected|booked|"
    r"opened|held|placed|scheduled|arranged)|(has|have) been (successfully )?(rescheduled|changed|updated|moved|"
    r"redirected|booked|opened|held|placed|scheduled)|all set|(is|are) now (scheduled|rescheduled|changed|updated|"
    r"booked|held|on hold|redirected))\b", re.I)
_TOOL_WORDS = re.compile(r"\[?\b(find_?parcel|confirm_?parcel|verify_?caller|get_?status|reschedule_?delivery|"
                         r"change_?address|hold_?at_?depot|open_?claim|transfer_?to_?human)\b\]?(\(\))?", re.I)
ACTION_TOOLS = {"get_status", "reschedule_delivery", "change_address", "hold_at_depot", "open_claim",
                "transfer_to_human"}
NUDGE_PROMISE = ("You described an action but didn't call a tool. If the caller's request needs a tool, "
                 "call it now. Otherwise reply with a question for the caller.")
NUDGE_CLAIM = ("You said something was done, but no tool returned allowed=true for it. Call the tool now, "
               "or tell the caller honestly that it isn't done.")


_WANTS_HUMAN = re.compile(r"\b(person|human|real (one|agent)|representative|operator|someone|colleague|supervisor|"
                         r"manager|transfer me|agent)\b", re.I)
_OFFERED_HUMAN = re.compile(r"\b(colleague|human agent|transfer you|connect you)\b", re.I)
_YES = re.compile(r"(yes|yeah|yep|sure|please|okay|ok)\b", re.I)
_OTHER_PARCEL = re.compile(r"\b(another|other|different|second|new|separate)\s+(package|parcel|order|delivery|shipment)\b",
                           re.I)


def _statements(text: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.?!])\s+", text) if x and not x.endswith("?")]


@dataclass
class TurnMetrics:
    turn: int
    llm_calls: int = 0
    llm_ms: list[float] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: int = 0
    tool_calls: list[str] = field(default_factory=list)
    bad_tool_calls: int = 0
    bad_tool_detail: list[str] = field(default_factory=list)   # "get_status(tracking_id)" etc., for diagnosis
    ignored_args: int = 0
    nudges: int = 0
    fast_path: int = 0               # LLM calls skipped because a tool supplied the reply
    guard_fallbacks: int = 0
    total_ms: float = 0.0


class Agent:
    def __init__(self, llm: LLM, session: CallSession, history_turns: int | None = 6,
                 max_steps: int = 5, prompt_mode: str = "full", recorder=None, lessons: list[str] | None = None):
        self.llm, self.session = llm, session
        self.prompt_mode = prompt_mode              # full (default: best for the 3B) | compact | gated: phase 6a
        self.recorder = recorder                    # phase 6b: captures (prompt, tools, reply) per LLM call
        self.lessons = list(lessons or [])          # promoted lessons (swiftroute/learning), eval-gated
        self.history_turns = history_turns          # None = keep everything (for ablation)
        self.max_steps = max_steps
        self.history: list[dict] = []
        self.metrics: list[TurnMetrics] = []
        self.reply_cut_off = False                  # set by the voice layer on barge-in
        self._slot_buf = ""                         # pieces of a code said across several turns
        self._slot_expect = "open"                  # which code those pieces belong to
        self.advisor = None                         # () -> {"notes", "next"}: sub-agents coaching this one (agents/team.py)
        self.reply_reviewer = None                  # (reply) -> correction | None: a sub-agent checks before it's spoken
        self.turn_planner = None                    # (caller text) -> (tool, args) | None: run in code, like slots

    # ------------------------------------------------------------ context
    def state_block(self) -> str:
        s = self.session
        now = s._now()
        # NB: parcel status is deliberately NOT in here. An earlier version included it,
        # and the model answered "where's my parcel" from the state block: no get_status
        # call, and a leak path around verification.
        lines = ["CALL STATE (authoritative; trust this over the conversation)",
                 f"Today: {now:%A %Y-%m-%d}"]
        level = s.identity.data.get("level") if s.identity and s.identity.allowed else (
            "failed" if s.identity else "not checked")
        if s.tracking_id:
            lines.append(f"Parcel: confirmed (ending {s.tracking_id[-4:]}). Every action below applies to THIS "
                         "parcel only. If the caller brings up a different parcel, ask for its tracking "
                         "number and call find_parcel before doing anything for it.")
        elif s.pending_ids:
            lines.append("Parcel: waiting for the caller to say if the read-back is right")
        else:
            lines.append("Parcel: none yet")
        lines.append(f"Caller verification: {level}")
        done = [c for c in s.reason_codes if c.endswith("_OK") or c == "TRANSFER"]
        if done:
            lines.append(f"Completed this call: {', '.join(done)}")
        days = _business_days(now.date(), s.cfg.reschedule_window_days)
        lines.append("BOOKABLE DATES: " + ", ".join(f"{d:%a %b} {d.day}" for d in days))
        if self.reply_cut_off:
            lines.append("Note: the caller cut off your previous reply; they may not have heard all of it.")
        # Notes from the call team's supervisor (swiftroute/agents/team.py). Advice only: the
        # tools still decide what's allowed, so a wrong note can cost a turn, never a check.
        advice = self.advisor() if self.advisor else {}
        for note in advice.get("notes", []):
            lines.append("Supervisor: " + note)
        lines.append("Next: " + (advice.get("next") or self._next_step(level)))
        return "\n".join(lines)

    def _next_step(self, level: str) -> str:
        """Small models follow an explicit next step far better than a process
        described once in the system prompt."""
        s = self.session
        if not s.tracking_id and s.pending_ids:
            return "wait for the caller's answer; if yes, call confirm_parcel; if no, ask them to read it again."
        if not s.tracking_id:
            return "ask for the tracking number, then call find_parcel with exactly what they say."
        if level == "not checked":
            return "ask for the delivery postal code, then call verify_caller."
        if level == "failed":
            return "don't share any details. Offer to try the postal code again, or offer a colleague."
        if level == "partial":
            return ("only get_status is allowed. For any change, explain they need to call from the "
                    "phone number on the account. Do not call change tools or say a change was made.")
        return ("do what the caller asked by calling the matching tool now. Say it's done only after "
                "the tool returns allowed=true.")

    def _window(self) -> list[dict]:
        if self.history_turns is None:
            return self.history
        starts = [i for i, m in enumerate(self.history) if m["role"] == "user"]
        if len(starts) <= self.history_turns:
            return self.history
        return self.history[starts[-self.history_turns]:]

    def _check_reply(self, text: str, narrated_tool: bool, m: "TurnMetrics") -> str | None:
        """Is this final reply safe to speak? Returns the correction to send, or None."""
        if narrated_tool:
            return NUDGE_PROMISE
        succeeded = any(c.endswith("_OK") or c == "TRANSFER" for c in self.session.reason_codes)
        if not succeeded and any(_CLAIM.search(x) for x in _statements(text)):
            return NUDGE_CLAIM
        after_action = bool(m.tool_calls) and m.tool_calls[-1] in ACTION_TOOLS
        if not after_action and any(_PROMISE.search(x) for x in _statements(text)):
            return NUDGE_PROMISE
        if self.reply_reviewer is not None:
            return self.reply_reviewer(text)
        return None

    def _tools(self) -> list[dict]:
        return prompting.build(self.session, self.prompt_mode)[1]

    def build_messages(self, nudge: str | None = None) -> list[dict]:
        tail = self.state_block() + (f"\n\nCORRECTION: {nudge}" if nudge else "")
        system = prompting.build(self.session, self.prompt_mode)[0] + prompting.lessons_block(self.lessons)
        return ([{"role": "system", "content": system}]
                + [{k: v for k, v in m.items() if not k.startswith("_")} for m in self._window()]
                + [{"role": "system", "content": tail}])

    # ------------------------------------------------------------- slots
    def _last_spoken(self) -> str:
        for msg in reversed(self.history):
            if msg["role"] == "assistant" and not msg.get("tool_calls"):
                return msg.get("content") or ""
        return ""

    def _handle_slot(self, expect: str, text: str, m: TurnMetrics) -> str | None:
        """Answer an ID / postal-code turn in code. Returns the reply, or None to let
        the LLM handle the turn (open conversation, or a code that needs it to act on)."""
        s = self.session
        if expect == "postal" and s.tracking_id and not (s.identity and s.identity.allowed):
            if not slots.code_only(text):
                self._slot_buf = ""
                return None
            combined = f"{self._slot_buf} {text}".strip()
            if slots.complete_postal(combined) or len(slots.postal_chars(combined)) >= 6:
                self._slot_buf = ""
                trace.emit("slot", expect="postal", action="complete: verify_caller in code, no LLM",
                           buffer=combined, chars=slots.postal_chars(combined))
                return self._direct_tool("verify_caller", {"spoken_postal_code": combined}, m)
            self._slot_buf, self._slot_expect = combined, "postal"
            trace.emit("slot", expect="postal", action="fragment: keep listening", buffer=combined,
                       chars=slots.postal_chars(combined))
            return self._backchannel(m)
        wants_id = (not s.tracking_id or s.other_parcel_mentioned) and not s.pending_ids
        if expect == "id" and wants_id:
            if not slots.code_only(text):
                self._slot_buf = ""
                return None
            combined = f"{self._slot_buf} {text}".strip()
            if slots.complete_id(combined) or len(slots.id_chars(combined)) >= 9:
                self._slot_buf = ""
                trace.emit("slot", expect="id", action="complete: find_parcel in code, no LLM",
                           buffer=combined, chars=slots.id_chars(combined))
                return self._direct_tool("find_parcel", {"spoken_tracking_id": combined}, m)
            if slots.is_id_fragment(combined):
                self._slot_buf, self._slot_expect = combined, "id"
                trace.emit("slot", expect="id", action="fragment: keep listening", buffer=combined,
                           chars=slots.id_chars(combined))
                return self._backchannel(m)
        if expect in ("id", "postal"):
            trace.emit("slot", expect=expect, action="not a bare code: hand to the LLM", buffer=text)
        self._slot_buf = ""
        return None

    def _direct_tool(self, name: str, args: dict, m: TurnMetrics) -> str | None:
        """Run a tool without the LLM. History records it exactly as if the model had
        called it, so later turns see a normal tool exchange."""
        result = self.session.call(name, args)
        m.tool_calls.append(name)
        m.fast_path += 1
        if result.get("reply"):
            trace.emit("fastpath", reply=result["reply"], why="deterministic state: tool supplied the words")
        call_id = f"direct_{self.session.turn}_{name}"
        self.history.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        self.history.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": compact(result)})
        if result.get("reply"):
            self.history.append({"role": "assistant", "content": result["reply"]})
            return result["reply"]
        # Fully verified: no canned reply. Let the LLM carry on with what the caller wanted;
        # history is now [..., user, assistant(tool_call), tool], a valid place to continue.
        return None

    def _backchannel(self, m: TurnMetrics) -> str:
        m.fast_path += 1
        reply = "Mm-hm, go on."
        trace.emit("fastpath", reply=reply, why="caller is mid-code: backchannel instead of an LLM call")
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def mark_last_reply_unheard(self, partial: bool = False) -> None:
        """The caller barged in. Keep the model's picture of the call honest.

        v1 appended "[caller interrupted ...]" to the reply text in history. In
        real call #2 qwen2.5:3b imitated it: later replies ended with the same
        bracketed note (three times, as barge-in fired more than once), and it
        was spoken aloud. Notes about the conversation must never look like
        something the assistant said. So:
          * never heard at all: drop the reply from history (tool results stay;
            what actually happened is in CALL STATE)
          * cut off midway: keep the reply, flag it in the CALL STATE block only
        Idempotent per reply.
        """
        for i in range(len(self.history) - 1, -1, -1):
            msg = self.history[i]
            if msg["role"] == "assistant" and not msg.get("tool_calls"):
                if msg.get("_interrupt_handled"):
                    return
                if partial:
                    msg["_interrupt_handled"] = True
                    self.reply_cut_off = True
                else:
                    del self.history[i]
                return

    # ------------------------------------------------------------ the loop
    def respond(self, user_text: str) -> str:
        """One caller turn in, one spoken reply out (possibly several tool calls)."""
        self.session.turn += 1
        if self.session.tracking_id and _OTHER_PARCEL.search(user_text):
            self.session.other_parcel_mentioned = True
        offered = bool(_OFFERED_HUMAN.search(self._last_spoken()))
        if _WANTS_HUMAN.search(user_text) or (offered and _YES.match(user_text.strip())):
            self.session.caller_wants_human = True
        m = TurnMetrics(turn=self.session.turn)
        t0 = time.perf_counter()
        # mid-code ("It's V8." -> "Mm-hm, go on."): still collecting the same code
        expect = self._slot_expect if self._slot_buf else slots.expectation(self._last_spoken())
        trace.emit("turn.start", turn=self.session.turn, text=user_text, expect=expect,
                   caller_wants_human=self.session.caller_wants_human,
                   other_parcel=self.session.other_parcel_mentioned)
        self.history.append({"role": "user", "content": user_text})
        direct = self._handle_slot(expect, user_text, m)
        if direct is None and self.turn_planner is not None and self.history[-1]["role"] == "user":
            planned = self.turn_planner(user_text)
            if planned:
                trace.emit("plan", tool=planned[0], args=planned[1],
                           why="the caller repeated a request the model already dropped: run it in code")
                direct = self._direct_tool(planned[0], planned[1], m)   # None: the LLM phrases the result
        if direct is not None:
            m.total_ms = (time.perf_counter() - t0) * 1000
            self.reply_cut_off = False
            self.metrics.append(m)
            self._trace_end(direct, m)
            return direct
        spoken: list[str] = []
        nudge: str | None = None
        for step in range(self.max_steps):
            messages, tools = self.build_messages(nudge), self._tools()
            if trace.active():
                trace.emit("llm.request", step=step, model=getattr(self.llm, "model", "?"),
                           prompt_mode=self.prompt_mode, nudge=nudge, messages=messages,
                           tools=[t["function"]["name"] for t in tools],
                           window_turns=sum(1 for x in self._window() if x["role"] == "user"),
                           history_turns_total=sum(1 for x in self.history if x["role"] == "user"))
            reply: LLMReply = self.llm.chat(messages, tools)
            trace.emit("llm.response", step=step, ms=round(reply.latency_ms), content=reply.content,
                       tool_calls=[{"name": c["name"], "arguments": c["arguments"]} for c in reply.tool_calls],
                       prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens)
            if self.recorder is not None:
                self.recorder(messages, tools, reply, self.session.turn)
            nudge = None
            m.llm_calls += 1
            m.llm_ms.append(reply.latency_ms)
            m.prompt_tokens.append(reply.prompt_tokens)
            m.completion_tokens += reply.completion_tokens
            narrated_tool = bool(_TOOL_WORDS.search(reply.content or ""))
            text = to_speech(reply.content)
            if (reply.content or "").strip() and text != re.sub(r"\s+", " ", reply.content).strip():
                trace.emit("guard", kind="speech_sanitised", before=reply.content, after=text,
                           why="brackets, tool names, full IDs or markdown never reach TTS")
            if not reply.tool_calls:
                problem = self._check_reply(text, narrated_tool, m)
                last_try = m.nudges > 0 or step == self.max_steps - 1
                if problem:
                    trace.emit("guard", kind="unbacked_claim" if problem == NUDGE_CLAIM else
                               "narrated_tool" if narrated_tool else
                               "promise_without_action" if problem == NUDGE_PROMISE else "supervisor_correction",
                               text=text, action="retry with a correction" if not last_try else "final attempt",
                               correction=problem)
                if problem and not last_try:
                    m.nudges += 1
                    if text and problem == NUDGE_PROMISE and not narrated_tool:
                        spoken.append(text)          # "let me check" is fine filler; keep it
                        self.history.append({"role": "assistant", "content": text})
                    nudge = problem
                    continue
                if problem == NUDGE_CLAIM:
                    # still claiming success with nothing behind it: never let that reach the caller
                    m.guard_fallbacks += 1
                    trace.emit("guard", kind="unbacked_claim_fallback", text=text,
                               action="replaced with an honest hand-off")
                    self.session.transfer_to_human("unverified_claim", f"Model claimed: {text[:120]}", system=True)
                    text = UNVERIFIED_CLAIM_FALLBACK
                if text:
                    spoken.append(text)
                self.history.append({"role": "assistant", "content": text})
                break
            # Text written alongside a tool call is the model guessing before it has the
            # result. Call #3: it said a whole answer here, then the real answer after the
            # tool returned, so callers heard everything twice. Only a short filler with no
            # question in it ("Sure, one moment.") is spoken; the rest is dropped.
            if text and _is_short_filler(text):
                spoken.append(text)
            elif text:
                trace.emit("guard", kind="pre_tool_text_dropped", text=text,
                           why="the model guessed an answer before the tool returned (call #3)")
            self.history.append({"role": "assistant", "content": text, "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                for c in reply.tool_calls]})
            results = []
            for c in reply.tool_calls:
                result = self.session.call(c["name"], c["arguments"])
                results.append(result)
                m.tool_calls.append(c["name"])
                if result.get("reason_code") in ("BAD_ARGS", "UNKNOWN_TOOL"):
                    m.bad_tool_calls += 1
                    m.bad_tool_detail.append(f"{c['name']}({','.join(sorted(c['arguments'] or {}))})")
                if result.get("ignored_args"):
                    m.ignored_args += 1
                self.history.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                                     "content": compact(result)})
            # Fast path: deterministic states (ID found / not found / read-back / postal mismatch)
            # come with caller-ready words. Speak them and skip a whole LLM round trip.
            if results and all(r.get("reply") for r in results):
                said = results[-1]["reply"]     # e.g. find + verify in one step: only the latest state matters
                spoken = [x for x in spoken if not _PROMISE.search(x)]   # drop "let me check" filler
                spoken.append(said)
                self.history.append({"role": "assistant", "content": said})
                m.fast_path += 1
                trace.emit("fastpath", reply=said, why="tool returned caller-ready words: second LLM call skipped")
                break
        else:
            # ran out of steps: never leave the caller in silence
            trace.emit("guard", kind="loop_limit", action="hand off instead of silence")
            self.session.transfer_to_human("agent_loop_limit", f"Stuck after {self.max_steps} steps", system=True)
            spoken.append(FALLBACK)
            self.history.append({"role": "assistant", "content": FALLBACK})
        self.reply_cut_off = False
        m.total_ms = (time.perf_counter() - t0) * 1000
        self.metrics.append(m)
        joined = " ".join(spoken)
        final = dedupe_sentences(joined) or FALLBACK
        if final != joined.strip() and joined.strip():
            trace.emit("guard", kind="dedupe", before=joined, after=final, why="never say a sentence twice (call #3)")
        self._trace_end(final, m)
        return final

    def _trace_end(self, reply: str, m: TurnMetrics) -> None:
        if trace.active():
            trace.emit("turn.end", turn=m.turn, reply=reply, metrics=asdict(m))
            trace.emit("state", **trace.session_snapshot(self.session))


def _is_short_filler(text: str) -> bool:
    return "?" not in text and len(text.split()) <= 6


def dedupe_sentences(text: str) -> str:
    """Never say the same sentence twice in one reply (call #3 did, verbatim)."""
    seen, out = set(), []
    for sent in re.split(r"(?<=[.?!])\s+", text.strip()):
        key = re.sub(r"[^a-z0-9 ]", "", sent.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(sent)
    return " ".join(out)


def compact(result: dict) -> str:
    """Tool results are re-sent on every later call while in the window:
    strip what the model doesn't need to phrase a reply."""
    r = {k: v for k, v in result.items() if k not in ("heard",) and v not in (None, [], "")}
    if r.get("confirmed"):
        r.pop("tracking_id", None)        # nothing left to confirm; don't tempt it to read the ID aloud
    if "recent_events" in r:
        r["recent_events"] = [f"{e['status']}{' ' + e['detail'] if e.get('detail') else ''}"
                              for e in r["recent_events"]]
    return json.dumps(r, separators=(",", ":"), default=str)


_MD = re.compile(r"[*_#`>]|^\s*[-•]\s+|^\s*\d+\.\s+", re.M)


_FULL_ID = re.compile(r"\bSR[0-9A-Z]{9}\b")


def to_speech(text: str) -> str:
    """Last line of defence before TTS: no markdown, no lists, no raw tracking
    numbers, no leaked tool names, one line."""
    text = re.sub(r"\[[^\]]*\]", "", text or "")          # [notes] are never speech
    text = re.sub(r"\b(CORRECTION|Supervisor|CALL STATE)\b:?.*$", "", text, flags=re.S)   # parroted prompt notes
    text = _TOOL_WORDS.sub("", text)
    text = _FULL_ID.sub("your parcel", text)
    text = _MD.sub("", text)
    text = re.sub(r"\[\s*\]|\(\s*\)", "", text)
    return re.sub(r"\s+", " ", text).strip()
