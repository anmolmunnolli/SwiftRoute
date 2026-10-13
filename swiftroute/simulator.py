"""Moves parcels through the lifecycle so every call hits live state.

Listeners get (tracking_id, new_status, detail). Phase 7 subscribes the
outbound-call scheduler to DELIVERY_EXCEPTION events.
"""
from __future__ import annotations

import random
import sqlite3
from datetime import datetime
from typing import Callable

from .db import iso, now_utc

Listener = Callable[[str, str, str | None], None]

NEXT = {
    "LABEL_CREATED": [("PICKED_UP", 1.0)],
    "PICKED_UP": [("IN_TRANSIT", 1.0)],
    "IN_TRANSIT": [("AT_DEPOT", 1.0)],
    "AT_DEPOT": [("OUT_FOR_DELIVERY", 1.0)],
    "OUT_FOR_DELIVERY": [("DELIVERED", 0.85), ("DELIVERY_EXCEPTION", 0.15)],
    "DELIVERY_EXCEPTION": [("OUT_FOR_DELIVERY", 1.0)],   # automatic re-attempt
}
EXCEPTION_REASONS = ["NO_ONE_HOME", "ACCESS_ISSUE", "ADDRESS_ISSUE"]


class Simulator:
    def __init__(self, conn: sqlite3.Connection, seed: int | None = None,
                 move_prob: float = 0.3, max_attempts: int = 3):
        self.conn = conn
        self.rng = random.Random(seed)
        self.move_prob = move_prob
        self.max_attempts = max_attempts
        self.listeners: list[Listener] = []

    def subscribe(self, fn: Listener) -> None:
        self.listeners.append(fn)

    def tick(self, now: datetime | None = None) -> list[tuple[str, str, str | None]]:
        now = now or now_utc()
        changes = []
        rows = self.conn.execute(
            "SELECT tracking_id, status, attempts, reschedule_date FROM parcels "
            "WHERE status IN ({})".format(",".join("?" * len(NEXT))), list(NEXT)).fetchall()
        for r in rows:
            if self.rng.random() > self.move_prob:
                continue
            # a caller-chosen reschedule date blocks the next delivery attempt until that day
            if (r["status"] in ("AT_DEPOT", "DELIVERY_EXCEPTION") and r["reschedule_date"]
                    and r["reschedule_date"] > now.date().isoformat()):
                continue
            if r["status"] == "DELIVERY_EXCEPTION" and r["attempts"] >= self.max_attempts:
                new, detail = "RETURNED_TO_SENDER", "MAX_ATTEMPTS"
            else:
                opts = NEXT[r["status"]]
                new = self.rng.choices([o[0] for o in opts], [o[1] for o in opts])[0]
                detail = self.rng.choice(EXCEPTION_REASONS) if new == "DELIVERY_EXCEPTION" else None
            self._apply(r["tracking_id"], r["status"], new, detail, now)
            changes.append((r["tracking_id"], new, detail))
        for c in changes:
            for fn in self.listeners:
                fn(*c)
        return changes

    def _apply(self, tid: str, old: str, new: str, detail: str | None, now: datetime) -> None:
        with self.conn:
            sets = ["status = ?", "updated_at = ?"]
            vals: list = [new, iso(now)]
            if new == "OUT_FOR_DELIVERY":
                sets.append("attempts = attempts + 1")
                sets.append("reschedule_date = NULL")
            if new == "DELIVERY_EXCEPTION":
                sets.append("exception_reason = ?")
                vals.append(detail)
            self.conn.execute(f"UPDATE parcels SET {', '.join(sets)} WHERE tracking_id = ?", vals + [tid])
            self.conn.execute("INSERT INTO events (tracking_id, ts, status, detail) VALUES (?,?,?,?)",
                              (tid, iso(now), new, detail))
