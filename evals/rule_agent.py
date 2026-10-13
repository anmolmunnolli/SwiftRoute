"""A hand-written, no-LLM 'model' that speaks the same chat-completions
protocol as Qwen. Two jobs:
  1. smoke-test the harness end to end without a model server (CI);
  2. a deterministic reference: if the rule agent fails a scenario, the
     scenario or the caller is broken, not the LLM.
It is not a baseline to beat on naturalness; it's a baseline on correctness.
"""
from __future__ import annotations

import json
import re
from datetime import date

from swiftroute.llm import LLMReply

_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_ADDR = re.compile(r"to (\d+ [^,]+), ([A-Za-z ]+),? ([A-Za-z]\d[A-Za-z] ?\d[A-Za-z]\d)")


def _intent(text: str) -> str | None:
    t = text.lower()
    if re.search(r"real person|human|agent|someone", t):
        return "transfer_to_human"
    if re.search(r"damaged|crushed|broken", t):
        return "claim_damaged"
    if re.search(r"missing|never came|lost", t):
        return "claim_missing"
    if re.search(r"address|redirect|send (my )?(parcel|package|it) to", t):
        return "change_address"
    if re.search(r"depot|pick it up|pickup", t):
        return "hold_at_depot"
    if re.search(r"reschedul|redeliver|missed|instead|monday|tuesday|wednesday|thursday|friday|saturday|sunday", t):
        return "reschedule_delivery"
    if re.search(r"where|when|status|arriv|update|checking|check on", t):
        return "get_status"
    return None


class RuleAgentLLM:
    model = "rule-agent"

    def __init__(self):
        self._n = 0

    def _call(self, name: str, **args) -> LLMReply:
        self._n += 1
        return LLMReply("", [{"id": f"r{self._n}", "name": name, "arguments": args}])

    def chat(self, messages: list[dict], tools=None) -> LLMReply:
        state = messages[-1]["content"]
        body = messages[1:-1]
        last = body[-1]
        users = [m["content"] for m in body if m["role"] == "user"]
        called = {tc["function"]["name"] for m in body if m.get("tool_calls") for tc in m["tool_calls"]}
        verified = re.search(r"Caller verification: (\w+)", state).group(1)
        confirmed = "Parcel: confirmed" in state
        today = date.fromisoformat(re.search(r"Today: \w+ (\S+)", state).group(1))

        if last["role"] == "tool":
            r = json.loads(last["content"])
            name = last["name"]
            if name == "find_parcel":
                if r.get("confirmed"):
                    return LLMReply("Found it. What's the delivery postal code?")
                if r.get("read_back"):
                    return LLMReply(f"Is that the parcel {r['read_back']}?")
                if r.get("options"):
                    return LLMReply("Is it the one " + ", or the one ".join(o["read_back"] for o in r["options"]) + "?")
                return LLMReply("I couldn't find that. Could you read the tracking number again?")
            if name == "confirm_parcel":
                return LLMReply("Thanks. What's the delivery postal code?" if r.get("allowed")
                                else "Sorry, could you read the tracking number again?")
            if name == "verify_caller":
                if not r.get("allowed"):
                    return LLMReply("Sorry, that postal code doesn't match our records, so I can't share details.")
                return self._act(users, called, today, verified=r["level"]) or LLMReply("Thanks. How can I help?")
            if name == "get_status":
                if not r.get("allowed"):
                    return LLMReply("I'll need the delivery postal code first.")
                return LLMReply(f"It's {r['status_human']}, expected {r['eta']}. Anything else I can help with?")
            if name == "transfer_to_human":
                return LLMReply("I'll connect you with a colleague now.")
            if r.get("allowed"):
                return LLMReply(f"{r['explanation']} Anything else I can help with?")
            extra = " I can hold it at a depot for pickup instead." if r.get("suggest") == "hold_at_depot" else ""
            return LLMReply(r.get("explanation", "Sorry, I can't do that.") + extra)

        text = last["content"]
        if confirmed and re.search(r"\banother (package|parcel)\b", text, re.I):
            return LLMReply("Sure. What's the tracking number for that one?")
        if confirmed and re.match(r"\s*it's\b", text, re.I) and len(re.findall(r"\b\w\b", text)) >= 6:
            return self._call("find_parcel", spoken_tracking_id=text)
        if _intent(text) == "transfer_to_human":
            return self._call("transfer_to_human", reason="caller asked", summary=text[:120])
        if "waiting for the caller to say" in state:
            if re.search(r"\byes\b|right|correct|the one", text, re.I):
                opts = self._pending(body)
                pick = next((t for t, rb in opts if rb.lower() in text.lower()), opts[0][0])
                return self._call("confirm_parcel", tracking_id=pick)
            return LLMReply("Okay, could you read the tracking number again, one character at a time?")
        if not confirmed:
            if re.search(r"\bit's\b", text, re.I) and len(re.findall(r"\b\w\b|\b(?:zero|one|two|three|four|five|six|seven|eight|nine)\b", text)) >= 6:
                return self._call("find_parcel", spoken_tracking_id=text)
            return LLMReply("Sure, can I have your tracking number?")
        if verified == "not":
            if re.search(r"(\b\w\b.*){6}", text):
                return self._call("verify_caller", spoken_postal_code=text.rstrip("."))
            return LLMReply("I need the delivery postal code before I can help with that.")
        if verified == "failed":
            return LLMReply("I'm sorry, without matching details I can't share anything about this parcel.")
        if re.search(r"\b(bye|that's all|that is all|nothing else)\b", text, re.I):
            return LLMReply("Thanks for calling SwiftRoute. Have a good day.")
        return self._act([text], set(), today, verified) or LLMReply("Is there anything else I can help with?")

    @staticmethod
    def _pending(body: list[dict]) -> list[tuple[str, str]]:
        for m in reversed(body):
            if m["role"] == "tool" and m["name"] == "find_parcel":
                r = json.loads(m["content"])
                if r.get("options"):
                    return [(o["tracking_id"], o["read_back"]) for o in r["options"]]
                return [(r["tracking_id"], r.get("read_back", ""))]
        return []

    def _act(self, users: list[str], called: set[str], today: date, verified: str) -> LLMReply | None:
        for text in reversed(users):
            it = _intent(text)
            if not it:
                continue
            if it == "get_status":
                return self._call("get_status")
            if it == "claim_damaged":
                return self._call("open_claim", kind="damaged", description=text[:120])
            if it == "claim_missing":
                return self._call("open_claim", kind="missing", description=text[:120])
            if it == "hold_at_depot":
                return self._call("hold_at_depot")
            if it == "change_address":
                m = _ADDR.search(text)
                if not m:
                    return LLMReply("What's the new address, including the postal code?")
                return self._call("change_address", line1=m.group(1), city=m.group(2).strip(),
                                  spoken_postal_code=m.group(3))
            if it == "reschedule_delivery":
                day = next((d for d in _DAYS if d in text.lower()), None)
                if not day:
                    return LLMReply("Which day would you like it delivered?")
                return self._call("reschedule_delivery", day=day.capitalize())
        return None
