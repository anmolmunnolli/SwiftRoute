"""Deterministic policy engine.

The LLM never decides whether an action is allowed. It calls a tool, the tool
asks this module, and the agent is told the verdict plus a reason code it must
explain to the caller. Every rule is a pure function of (parcel, request, now),
so it is unit-testable and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from . import trace
from .ids import normalise_postal


@dataclass(frozen=True)
class PolicyConfig:
    max_address_changes: int = 1
    reschedule_window_days: int = 5          # business days ahead a caller may pick
    claim_late_days: int = 5                 # parcel this late (vs ETA) counts as missing
    claim_window_days: int = 30              # claims must be filed within this of ETA
    max_delivery_attempts: int = 3
    max_verify_attempts: int = 3            # failed postal codes per call before lockout
    delivery_window: str = "9 am and 6 pm"
    hold_eligible: frozenset = frozenset({"IN_TRANSIT", "AT_DEPOT", "DELIVERY_EXCEPTION"})
    address_change_eligible: frozenset = frozenset(
        {"LABEL_CREATED", "PICKED_UP", "IN_TRANSIT", "AT_DEPOT", "DELIVERY_EXCEPTION"})
    reschedule_eligible: frozenset = frozenset({"DELIVERY_EXCEPTION", "AT_DEPOT", "IN_TRANSIT", "HELD_AT_DEPOT"})


DEFAULT = PolicyConfig()


@dataclass
class Decision:
    allowed: bool
    reason_code: str
    explanation: str                 # plain-language, safe for the agent to paraphrase
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reason_code": self.reason_code,
                "explanation": self.explanation, **self.data}


def _d(row) -> dict:
    return dict(row) if row is not None and not isinstance(row, dict) else (row or {})


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------
def verify_identity(parcel: dict, customer: dict, address: dict,
                    caller_phone: str | None, spoken_postal: str | None) -> Decision:
    """Two factors: the number they're calling from (caller ID) and the postal
    code they say. Full access (changes) needs both. The postal code alone is
    enough for status only. Note: a withheld number is treated like an unknown
    one; giving it full access would let anyone skip a factor by hiding caller ID.
    (Phase 4 adds an SMS one-time code to the number on file as step-up.)"""
    parcel, customer, address = _d(parcel), _d(customer), _d(address)
    phone_ok = bool(caller_phone) and _digits(caller_phone)[-10:] == _digits(customer["phone"])[-10:]
    postal = normalise_postal(spoken_postal) if spoken_postal else None
    postal_ok = postal is not None and postal == address["postal"].replace(" ", "")
    level = "full" if phone_ok and postal_ok else ("partial" if postal_ok else "none")
    trace.emit("policy.verify", caller_id=bool(caller_phone), caller_id_match=phone_ok,
               postal_said=spoken_postal, postal_normalised=postal, postal_match=postal_ok, level=level)
    if level == "none":
        return Decision(False, "ID_NOT_VERIFIED",
                        "The details given don't match this parcel.",
                        {"level": level, "postal_heard": postal})
    return Decision(True, "ID_VERIFIED_" + level.upper(),
                    "Caller verified." if level == "full" else
                    "Caller partly verified: status only, no changes.",
                    {"level": level, "postal_heard": postal})


def _digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


def require_full(identity: Decision | None) -> Decision | None:
    if identity is None or not identity.allowed:
        return Decision(False, "NEEDS_VERIFICATION",
                        "I need the delivery postal code before I can help with that.")
    if identity.data.get("level") != "full":
        return Decision(False, "NEEDS_FULL_VERIFICATION",
                        "For changes I need you to call from the phone number on the account. "
                        "I can still tell you where the parcel is.")
    return None


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------
def can_change_address(parcel: dict, cfg: PolicyConfig = DEFAULT) -> Decision:
    p = _d(parcel)
    if p["status"] not in cfg.address_change_eligible:
        return Decision(False, "ADDR_LOCKED_STATUS",
                        f"The parcel is {_human(p['status'])}, so the address can't be changed now.")
    if p["address_changes"] >= cfg.max_address_changes:
        return Decision(False, "ADDR_CHANGE_LIMIT",
                        "This parcel has already had its address changed once, which is the limit.")
    if p["signature_required"]:
        return Decision(False, "ADDR_SIGNATURE_REQUIRED",
                        "Signature-required parcels can only be redirected by the sender.",
                        {"suggest": "hold_at_depot"})
    return Decision(True, "ADDR_CHANGE_OK", "The address can be changed.")


def can_reschedule(parcel: dict, requested: date, now: datetime,
                   cfg: PolicyConfig = DEFAULT) -> Decision:
    p = _d(parcel)
    if p["status"] not in cfg.reschedule_eligible:
        return Decision(False, "RESCHED_LOCKED_STATUS",
                        f"The parcel is {_human(p['status'])}, so delivery can't be rescheduled.")
    if p["attempts"] >= cfg.max_delivery_attempts:
        return Decision(False, "RESCHED_ATTEMPTS_EXHAUSTED",
                        "We've reached the maximum delivery attempts; it can be held at a depot for pickup.",
                        {"suggest": "hold_at_depot"})
    today = now.date()
    allowed_days = _business_days(today, cfg.reschedule_window_days)
    if requested not in allowed_days:
        return Decision(False, "RESCHED_DATE_INVALID",
                        "That date isn't available. Deliveries run weekdays within the next "
                        f"{cfg.reschedule_window_days} business days.",
                        {"available_dates": [d.isoformat() for d in allowed_days]})
    return Decision(True, "RESCHED_OK", f"Delivery can be rescheduled to {requested:%A %B %d}.")


def can_hold_at_depot(parcel: dict, cfg: PolicyConfig = DEFAULT) -> Decision:
    p = _d(parcel)
    if p["status"] == "HELD_AT_DEPOT":
        return Decision(False, "HOLD_ALREADY", "The parcel is already being held at a depot.",
                        {"depot": p.get("hold_depot")})
    if p["status"] not in cfg.hold_eligible:
        return Decision(False, "HOLD_LOCKED_STATUS",
                        f"The parcel is {_human(p['status'])}, so it can't be redirected to a depot now.")
    return Decision(True, "HOLD_OK", "The parcel can be held at a depot for pickup.")


def can_open_claim(parcel: dict, kind: str, now: datetime, has_open_claim: bool,
                   cfg: PolicyConfig = DEFAULT) -> Decision:
    p = _d(parcel)
    eta = date.fromisoformat(p["eta"])
    days_late = (now.date() - eta).days
    if has_open_claim:
        return Decision(False, "CLAIM_EXISTS", "There's already an open claim for this parcel.")
    if days_late > cfg.claim_window_days:
        return Decision(False, "CLAIM_WINDOW_CLOSED",
                        f"Claims must be filed within {cfg.claim_window_days} days of the expected delivery date.")
    if kind == "damaged":
        if p["status"] != "DELIVERED":
            return Decision(False, "CLAIM_DAMAGED_NOT_DELIVERED",
                            "A damage claim can be opened once the parcel shows as delivered.")
        return Decision(True, "CLAIM_OK", "A damage claim can be opened.")
    if kind == "missing":
        if p["status"] == "DELIVERED":
            return Decision(True, "CLAIM_OK", "A missing-parcel claim can be opened since it shows as delivered.")
        if days_late >= cfg.claim_late_days and p["status"] not in ("RETURNED_TO_SENDER", "HELD_AT_DEPOT"):
            return Decision(True, "CLAIM_OK",
                            f"The parcel is {days_late} days past its expected date, so a missing-parcel claim can be opened.")
        return Decision(False, "CLAIM_TOO_EARLY",
                        f"A missing-parcel claim opens {cfg.claim_late_days} days after the expected date.",
                        {"eligible_from": (eta + timedelta(days=cfg.claim_late_days)).isoformat()})
    return Decision(False, "CLAIM_KIND_UNKNOWN", "Claims can be for damaged or missing parcels.")


# --------------------------------------------------------------------------
def _business_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


_HUMAN = {
    "LABEL_CREATED": "not yet picked up", "PICKED_UP": "picked up", "IN_TRANSIT": "in transit",
    "AT_DEPOT": "at the local depot", "OUT_FOR_DELIVERY": "already out for delivery",
    "DELIVERED": "already delivered", "DELIVERY_EXCEPTION": "pending after a missed delivery",
    "HELD_AT_DEPOT": "being held at a depot", "RETURNED_TO_SENDER": "being returned to the sender",
}


def _human(status: str) -> str:
    return _HUMAN.get(status, status.lower().replace("_", " "))
