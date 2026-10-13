"""The tool surface the LLM sees.

Design rules:
  * Tools return facts + a policy verdict; they never let the model decide.
  * State that matters for safety (which parcel, how verified) lives in the
    CallSession, not in the model's context, so a hallucinated tracking ID or
    "I already verified them" can't skip a check.
  * Every call is written to the audit table with its reason code; the ops
    dashboard (phase 7) is just a query over that table.
"""
from __future__ import annotations

import inspect

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from . import dates, ids, policy, trace
from .db import DEPOTS, existing_ids, iso, log_audit, now_utc


@dataclass
class CallSession:
    conn: sqlite3.Connection
    caller_phone: str | None = None
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    now: datetime | None = None                   # injectable clock for tests/evals
    cfg: policy.PolicyConfig = policy.DEFAULT
    turn: int = 0                                 # caller turns so far; the agent increments it
    pending_ids: list[str] = field(default_factory=list)   # offered to the caller, awaiting yes/no
    pending_turn: int = -1
    tracking_id: str | None = None                # confirmed parcel for this call
    identity: policy.Decision | None = None
    failed_verifications: int = 0
    id_readings: list[str] = field(default_factory=list)   # every ID reading this call, for the fallback
    other_parcel_mentioned: bool = False   # caller raised a different parcel; actions blocked until it's found
    caller_wants_human: bool = False       # set from the caller's own words; gates model-initiated transfers
    reason_codes: list[str] = field(default_factory=list)
    resolved: bool = False

    # ------------------------------------------------------------------ utils
    def _now(self) -> datetime:
        return self.now or now_utc()

    def _out(self, tool: str, args: dict, result: dict) -> dict:
        log_audit(self.conn, tool, args, result, self.call_id)
        if result.get("reason_code"):
            self.reason_codes.append(result["reason_code"])
        if trace.active():
            trace.emit("tool.result", tool=tool, args=args, allowed=result.get("allowed"),
                       reason_code=result.get("reason_code"), explanation=result.get("explanation"),
                       reply=result.get("reply"), policy=tool in POLICY_GATED,
                       result={k: v for k, v in result.items() if k not in ("reply", "explanation")})
            trace.emit("state", **trace.session_snapshot(self))
        return result

    def _parcel(self) -> dict | None:
        if not self.tracking_id:
            return None
        r = self.conn.execute("SELECT * FROM parcels WHERE tracking_id = ?", (self.tracking_id,)).fetchone()
        return dict(r) if r else None

    def _need_parcel(self) -> dict | None:
        if not self.tracking_id:
            return {"allowed": False, "reason_code": "NO_PARCEL_CONFIRMED",
                    "explanation": "Ask for the tracking number and confirm it first."}
        return None

    # ----------------------------------------------------------------- tools
    def find_parcel(self, spoken_tracking_id: str) -> dict:
        """Resolve a tracking number as heard.

        Exact, checksum-valid matches are confirmed immediately: making the
        caller confirm what they just said correctly costs a turn (~3 s on a
        call) for no safety gain. Anything we corrected must be confirmed.

        Results carry `reply`, caller-ready words. These states are fully
        deterministic, so the agent speaks the reply directly and skips a
        second LLM call (and the model can't garble the read-back).
        """
        raw = ids.parse_spoken(spoken_tracking_id) if " " in spoken_tracking_id.strip() else spoken_tracking_id
        trace.emit("id.parse", spoken=spoken_tracking_id, parsed=raw)
        res = ids.resolve(raw, lambda batch: existing_ids(self.conn, batch))
        out: dict[str, Any] = {"status": res.status, "heard": res.heard}
        self.pending_ids = []
        self.other_parcel_mentioned = False
        if res.status == "not_found":
            self.id_readings.append(raw)
            rec = self._recover_from_all_readings()
            if rec:
                res = rec
                out.update(status="recovered", reason_code="ID_RECOVERED_MULTI")
        if res.status == "not_found":
            n = len(self.id_readings)
            out["reply"] = (
                "I couldn't find that number. Could you read it again, one character at a time?" if n <= 1 else
                "Still not finding it. Let's try using a word for each letter, like R for Romeo or A for apple."
                if n == 2 else
                "I'm sorry, I still can't find it. You can read it once more, or I can connect you to a colleague.")
            out["reason_code"] = "ID_NOT_FOUND"
        elif res.status == "exact":
            self.tracking_id, self.identity = res.best, None
            out.update(tracking_id=res.best, confirmed=True,
                       reply="Got it, I found your parcel. What's the delivery postal code?")
            out["reason_code"] = "ID_EXACT"
        else:
            all_ids = [c.tracking_id for c in res.candidates] if res.status == "ambiguous" else [res.best]
            self.pending_ids, self.pending_turn = all_ids, self.turn
            if res.status == "ambiguous":
                rbs = [ids.smart_read_back(t, others=[o for o in all_ids if o != t]) for t in all_ids]
                out["options"] = [{"tracking_id": t, "read_back": rb} for t, rb in zip(all_ids, rbs)]
                out["reply"] = ("I found a couple of close matches. Is it the one "
                                + ", or the one ".join(rbs) + "?")
                out["reason_code"] = "ID_AMBIGUOUS"
            else:
                out["tracking_id"] = res.best
                out["read_back"] = ids.smart_read_back(res.best, heard=res.heard)
                out["reply"] = f"Just to check, is that the parcel {out['read_back']}?"
                out.setdefault("reason_code", "ID_CORRECTED")
        return self._out("find_parcel", {"spoken": spoken_tracking_id}, out)

    def _recover_from_all_readings(self) -> "ids.Resolution | None":
        """No single reading resolved: score every parcel against all of this
        call's readings together. Only a clear winner is offered, and it still
        has to be confirmed by the caller."""
        all_ids = [r[0] for r in self.conn.execute("SELECT tracking_id FROM parcels")]
        top = ids.scan(self.id_readings, all_ids)
        if not top:
            return None
        best, runner_up = top[0], (top[1].cost if len(top) > 1 else 99.0)
        if best.cost <= 2.6 and runner_up - best.cost >= 1.5:
            return ids.Resolution(ids.normalise_heard(self.id_readings[-1]), "corrected", [best])
        return None

    def confirm_parcel(self, tracking_id: str = "") -> dict:
        """Call only after the caller says yes to the read-back. The ID may be left out when
        exactly one number was read back (small models often send `{}` here)."""
        tid = (tracking_id or "").upper()
        if not tid and len(self.pending_ids) == 1:
            tid = self.pending_ids[0]
        if tid not in self.pending_ids:
            return self._out("confirm_parcel", {"tracking_id": tid},
                             {"allowed": False, "reason_code": "ID_UNKNOWN",
                              "explanation": "That number wasn't offered to the caller; run find_parcel first."})
        if self.turn == self.pending_turn:
            # structural guard: the model can't 'confirm' on the caller's behalf
            return self._out("confirm_parcel", {"tracking_id": tid},
                             {"allowed": False, "reason_code": "CONFIRM_NEEDS_CALLER",
                              "explanation": "Read the read_back to the caller and wait for their answer."})
        self.tracking_id, self.pending_ids, self.identity = tid, [], None
        self.id_readings = []
        return self._out("confirm_parcel", {"tracking_id": tid},
                         {"allowed": True, "reason_code": "ID_CONFIRMED",
                          "reply": "Great, thanks. What's the delivery postal code?"})

    def verify_caller(self, spoken_postal_code: str) -> dict:
        if (err := self._need_parcel()):
            return self._out("verify_caller", {}, err)
        if self.failed_verifications >= self.cfg.max_verify_attempts:
            # eval run 3: the agent let a caller guess postal codes 13 times. That's brute force.
            self.transfer_to_human("verification_locked", "Too many failed postal code attempts", system=True)
            return self._out("verify_caller", {}, {
                "allowed": False, "reason_code": "VERIFY_LOCKED",
                "reply": "For security I can't try that again on this call. "
                         "Let me get a colleague who can help you."})
        p = self._parcel()
        cust = self.conn.execute("SELECT * FROM customers WHERE id = ?", (p["customer_id"],)).fetchone()
        addr = self.conn.execute("SELECT * FROM addresses WHERE id = ?", (p["address_id"],)).fetchone()
        self.identity = policy.verify_identity(p, cust, addr, self.caller_phone, spoken_postal_code)
        out = self.identity.to_dict()
        level = out.get("level")
        if level == "none":
            self.failed_verifications += 1
            out["reply"] = ("Sorry, that postal code doesn't match what we have for this parcel. "
                            "Could you check it and say it again?")
        elif level == "partial":
            out["reply"] = ("Thanks. Since you're not calling from the number on the account, I can give you "
                            "updates but can't make changes. Would you like to know where it is?")
        # full: no canned reply, the model carries on with what the caller asked for
        return self._out("verify_caller", {"postal": spoken_postal_code}, out)

    def get_status(self) -> dict:
        if (err := self._need_parcel()):
            return self._out("get_status", {}, err)
        if self.identity is None or not self.identity.allowed:
            return self._out("get_status", {}, {
                "allowed": False, "reason_code": "NEEDS_VERIFICATION",
                "explanation": "Verify the caller with the delivery postal code first."})
        p = self._parcel()
        evs = self.conn.execute("SELECT ts, status, detail FROM events WHERE tracking_id = ? "
                                "ORDER BY ts DESC LIMIT 3", (self.tracking_id,)).fetchall()
        res = {"allowed": True, "reason_code": "STATUS_GIVEN", "status": p["status"],
               "status_human": policy._human(p["status"]), "eta": dates.say_date(date.fromisoformat(p["eta"])),
               "sender": p["sender"], "attempts": p["attempts"], "exception_reason": p["exception_reason"],
               "hold_depot": p["hold_depot"], "recent_events": [dict(e) for e in evs]}
        # Ground the follow-up questions callers actually ask. In call #2 the model was
        # asked "when will it be re-attempted?" / "what time?" and invented answers.
        nxt = self._next_delivery(p)
        if nxt:
            res["next_delivery"] = nxt
            res["delivery_window"] = f"between {self.cfg.delivery_window}; we can't give an exact time"
        return self._out("get_status", {}, res)

    def _next_delivery(self, p: dict) -> str | None:
        st = p["status"]
        today = self._now().date()
        if st in ("DELIVERED", "RETURNED_TO_SENDER", "HELD_AT_DEPOT"):
            return None
        if p.get("reschedule_date"):
            return dates.say_date(date.fromisoformat(p["reschedule_date"]))
        if st == "OUT_FOR_DELIVERY":
            return f"today, {dates.say_date(today)}"
        if st in ("DELIVERY_EXCEPTION", "AT_DEPOT"):
            return dates.say_date(policy._business_days(today, 1)[0])     # automatic re-attempt
        eta = date.fromisoformat(p["eta"])
        return dates.say_date(max(eta, today))

    def _guarded(self, tool: str, args: dict) -> dict | None:
        if (err := self._need_parcel()):
            return self._out(tool, args, err)
        if self.other_parcel_mentioned:
            # Call #4: "I received another package... damaged" -> the model opened a damage
            # claim on the parcel already confirmed. A prompt rule didn't stop it; this does.
            return self._out(tool, args, {
                "allowed": False, "reason_code": "OTHER_PARCEL_NOT_FOUND",
                "explanation": "The caller is talking about a different parcel. Ask for its tracking number first."})
        if (d := policy.require_full(self.identity)):
            return self._out(tool, args, d.to_dict())
        return None

    def reschedule_delivery(self, day: str) -> dict:
        """`day` is what the caller said ("Friday", "Monday the 19th", "tomorrow"),
        parsed here rather than by the LLM."""
        args = {"day": day}
        if (g := self._guarded("reschedule_delivery", args)):
            return g
        req = dates.parse_day(day, self._now().date())
        if req is None:
            avail = policy._business_days(self._now().date(), self.cfg.reschedule_window_days)
            return self._out("reschedule_delivery", args, {
                "allowed": False, "reason_code": "BAD_DATE",
                "explanation": "I didn't catch which day. We can deliver on "
                               + ", ".join(dates.say_date(d) for d in avail) + "."})
        d = policy.can_reschedule(self._parcel(), req, self._now(), self.cfg)
        d.data["requested"] = dates.say_date(req)
        if not d.allowed and "available_dates" in d.data:
            d.data["available_dates"] = [dates.say_date(date.fromisoformat(x)) for x in d.data["available_dates"]]
        if d.allowed:
            d.explanation = f"Done, delivery is rescheduled to {dates.say_date(req)}."
            with self.conn:
                # a held parcel goes back into the delivery flow; others keep their status
                self.conn.execute(
                    "UPDATE parcels SET reschedule_date = ?, updated_at = ?, "
                    "status = CASE WHEN status = 'HELD_AT_DEPOT' THEN 'AT_DEPOT' ELSE status END, "
                    "hold_depot = CASE WHEN status = 'HELD_AT_DEPOT' THEN NULL ELSE hold_depot END "
                    "WHERE tracking_id = ?", (req.isoformat(), iso(self._now()), self.tracking_id))
                status = self._parcel()["status"]
                self.conn.execute("INSERT INTO events (tracking_id, ts, status, detail) VALUES (?,?,?,?)",
                                  (self.tracking_id, iso(self._now()), status,
                                   f"RESCHEDULED_TO_{req.isoformat()}"))
            self.resolved = True
        return self._out("reschedule_delivery", args, d.to_dict())

    def change_address(self, line1: str, city: str, spoken_postal_code: str) -> dict:
        args = {"line1": line1, "city": city, "postal": spoken_postal_code}
        if (g := self._guarded("change_address", args)):
            return g
        p = self._parcel()
        d = policy.can_change_address(p, self.cfg)
        postal = ids.normalise_postal(spoken_postal_code)
        if d.allowed and not postal:
            d = policy.Decision(False, "ADDR_BAD_POSTAL", "I didn't catch a valid postal code for the new address.")
        if d.allowed:
            with self.conn:
                cur = self.conn.execute(
                    "INSERT INTO addresses (customer_id, line1, city, province, postal) "
                    "SELECT customer_id, ?, ?, province, ? FROM addresses WHERE id = ?",
                    (line1, city, postal, p["address_id"]))
                self.conn.execute("UPDATE parcels SET address_id = ?, address_changes = address_changes + 1, "
                                  "updated_at = ? WHERE tracking_id = ?",
                                  (cur.lastrowid, iso(self._now()), self.tracking_id))
            d.data["new_address"] = f"{line1}, {city} {postal[:3]} {postal[3:]}"
            self.resolved = True
        return self._out("change_address", args, d.to_dict())

    def hold_at_depot(self, depot_code: str | None = None) -> dict:
        args = {"depot_code": depot_code}
        if (g := self._guarded("hold_at_depot", args)):
            return g
        p = self._parcel()
        d = policy.can_hold_at_depot(p, self.cfg)
        if d.allowed:
            if not depot_code:
                city = self.conn.execute("SELECT city FROM addresses WHERE id = ?", (p["address_id"],)).fetchone()[0]
                depot_code = next((c for c, _, dc in DEPOTS if dc == city), DEPOTS[0][0])
            name = next((n for c, n, _ in DEPOTS if c == depot_code), None)
            if not name:
                return self._out("hold_at_depot", args, {"allowed": False, "reason_code": "DEPOT_UNKNOWN",
                                                         "depots": [c for c, *_ in DEPOTS]})
            with self.conn:
                self.conn.execute("UPDATE parcels SET status = 'HELD_AT_DEPOT', hold_depot = ?, updated_at = ? "
                                  "WHERE tracking_id = ?", (depot_code, iso(self._now()), self.tracking_id))
                self.conn.execute("INSERT INTO events (tracking_id, ts, status, detail) VALUES (?,?,?,?)",
                                  (self.tracking_id, iso(self._now()), "HELD_AT_DEPOT", depot_code))
            d.data["depot"] = name
            self.resolved = True
        return self._out("hold_at_depot", args, d.to_dict())

    def open_claim(self, kind: str, description: str = "") -> dict:
        args = {"kind": kind, "description": description}
        if (g := self._guarded("open_claim", args)):
            return g
        has_open = self.conn.execute("SELECT 1 FROM claims WHERE tracking_id = ? AND status = 'OPEN'",
                                     (self.tracking_id,)).fetchone() is not None
        d = policy.can_open_claim(self._parcel(), kind, self._now(), has_open, self.cfg)
        if d.allowed:
            with self.conn:
                cur = self.conn.execute("INSERT INTO claims (tracking_id, kind, description, opened_at, status) "
                                        "VALUES (?,?,?,?, 'OPEN')",
                                        (self.tracking_id, kind, description, iso(self._now())))
            d.data["claim_ref"] = f"CL{cur.lastrowid:06d}"
            self.resolved = True
        return self._out("open_claim", args, d.to_dict())

    def transfer_to_human(self, reason: str = "caller_request", summary: str = "", *, system: bool = False) -> dict:
        """Hand off. `system=True` is for the code's own decisions (verification lockout,
        loop limit, an unbacked claim). A transfer the MODEL asks for is only allowed if
        the caller asked for a person. Call #6: qwen2.5:3b transferred a caller who had
        just corrected their name, then kept talking; the call was stuck in the queue."""
        if not system and not self.caller_wants_human:
            return self._out("transfer_to_human", {"reason": reason}, {
                "allowed": False, "reason_code": "TRANSFER_NOT_REQUESTED",
                "explanation": "Only transfer when the caller asks for a person. Keep helping them."})
        return self._out("transfer_to_human", {"reason": reason},
                         {"allowed": True, "reason_code": "TRANSFER", "summary": summary,
                          "tracking_id": self.tracking_id,
                          "verified": self.identity.data.get("level") if self.identity else "none",
                          "reply": "Okay, I'm connecting you with a colleague now. Please hold."})

    # -------------------------------------------------------------- dispatch
    def call(self, name: str, arguments: dict) -> dict:
        fn = getattr(self, name, None)
        if name.startswith("_") or name not in TOOL_NAMES or fn is None:
            trace.emit("tool.rejected", tool=name, args=arguments, reason_code="UNKNOWN_TOOL")
            return {"allowed": False, "reason_code": "UNKNOWN_TOOL", "explanation": f"No tool {name}."}
        # Phase 6: drop arguments the tool doesn't take instead of failing the whole call.
        # The 3B with short schemas sent e.g. get_status(tracking_id=...): harmless intent,
        # but a TypeError cost a full extra LLM round trip. `system` is always dropped:
        # the model can't claim system rights. Missing REQUIRED args still fail loudly.
        params = inspect.signature(fn).parameters
        extra = sorted(k for k in arguments if k not in params or k == "system")
        arguments = {k: v for k, v in arguments.items() if k not in extra}
        trace.emit("tool.call", tool=name, args=arguments, ignored_args=extra)
        try:
            out = fn(**arguments)
        except TypeError as e:
            trace.emit("tool.rejected", tool=name, args=arguments, reason_code="BAD_ARGS", explanation=str(e))
            return {"allowed": False, "reason_code": "BAD_ARGS", "explanation": str(e)}
        if extra and isinstance(out, dict):
            out = {**out, "ignored_args": extra}
        return out


# OpenAI-style schemas; Ollama, vLLM and llama.cpp all accept this shape.
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "find_parcel",
        "description": "Look up a tracking number exactly as the caller said it.",
        "parameters": {"type": "object", "properties": {
            "spoken_tracking_id": {"type": "string", "description": "Verbatim, e.g. 'S R four K seven B as in bravo...'"}},
            "required": ["spoken_tracking_id"]}}},
    {"type": "function", "function": {
        "name": "confirm_parcel",
        "description": "Call only after the caller says yes to the read_back. If they say no, ask them to "
                       "read the number again and call find_parcel.",
        "parameters": {"type": "object", "properties": {"tracking_id": {"type": "string"}},
                       "required": ["tracking_id"]}}},
    {"type": "function", "function": {
        "name": "verify_caller",
        "description": "Verify the caller with the delivery postal code they say.",
        "parameters": {"type": "object", "properties": {"spoken_postal_code": {"type": "string"}},
                       "required": ["spoken_postal_code"]}}},
    {"type": "function", "function": {
        "name": "get_status", "description": "Current status, ETA and recent scans of the confirmed parcel.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "reschedule_delivery",
        "description": "Reschedule delivery. Pass the day exactly as the caller said it; the tool works out the date.",
        "parameters": {"type": "object", "properties": {"day": {"type": "string", "description": "e.g. 'Friday', 'Monday the 19th', 'tomorrow'"}},
                       "required": ["day"]}}},
    {"type": "function", "function": {
        "name": "change_address", "description": "Redirect the parcel to a new address.",
        "parameters": {"type": "object", "properties": {
            "line1": {"type": "string"}, "city": {"type": "string"}, "spoken_postal_code": {"type": "string"}},
            "required": ["line1", "city", "spoken_postal_code"]}}},
    {"type": "function", "function": {
        "name": "hold_at_depot", "description": "Hold the parcel at a depot for pickup. Omit depot_code for the nearest.",
        "parameters": {"type": "object", "properties": {"depot_code": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "open_claim", "description": "Open a claim for a damaged or missing parcel.",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["damaged", "missing"]}, "description": {"type": "string"}},
            "required": ["kind"]}}},
    {"type": "function", "function": {
        "name": "transfer_to_human", "description": "Hand off to a human agent with a short summary.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"}, "summary": {"type": "string"}}, "required": ["reason"]}}},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}
# tools whose verdict comes from policy.py (allowed / reason_code), not just a lookup
POLICY_GATED = {"verify_caller", "get_status", "reschedule_delivery", "change_address", "hold_at_depot",
                "open_claim", "transfer_to_human", "confirm_parcel"}
