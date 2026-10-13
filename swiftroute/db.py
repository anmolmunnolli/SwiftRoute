"""SQLite store for the mock courier. Zero setup; swap for Postgres later
by changing connect() only (all SQL here is portable)."""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .ids import ALPHABET, make_id

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "swiftroute.db"

STATUSES = [
    "LABEL_CREATED", "PICKED_UP", "IN_TRANSIT", "AT_DEPOT", "OUT_FOR_DELIVERY",
    "DELIVERED", "DELIVERY_EXCEPTION", "HELD_AT_DEPOT", "RETURNED_TO_SENDER",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, phone TEXT NOT NULL UNIQUE, email TEXT
);
CREATE TABLE IF NOT EXISTS addresses (
    id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
    line1 TEXT NOT NULL, city TEXT NOT NULL, province TEXT NOT NULL, postal TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS depots (
    code TEXT PRIMARY KEY, name TEXT NOT NULL, city TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS parcels (
    tracking_id TEXT PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    address_id INTEGER NOT NULL REFERENCES addresses(id),
    sender TEXT NOT NULL,
    status TEXT NOT NULL,
    eta TEXT NOT NULL,
    signature_required INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    address_changes INTEGER NOT NULL DEFAULT 0,
    hold_depot TEXT REFERENCES depots(code),
    reschedule_date TEXT,
    exception_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, tracking_id TEXT NOT NULL REFERENCES parcels(tracking_id),
    ts TEXT NOT NULL, status TEXT NOT NULL, detail TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_tid ON events(tracking_id, ts);
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY, tracking_id TEXT NOT NULL REFERENCES parcels(tracking_id),
    kind TEXT NOT NULL, description TEXT, opened_at TEXT NOT NULL, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, call_id TEXT, tool TEXT NOT NULL,
    args TEXT NOT NULL, allowed INTEGER, reason_code TEXT, result TEXT NOT NULL
);
"""

DEPOTS = [
    ("YYZ1", "Mississauga Airport Rd Depot", "Mississauga"),
    ("YYZ2", "Toronto Downtown Depot", "Toronto"),
    ("YOW1", "Ottawa East Depot", "Ottawa"),
    ("YUL1", "Montreal Dorval Depot", "Montreal"),
    ("YVR1", "Richmond Depot", "Vancouver"),
]
_CITIES = [("Toronto", "ON", "M"), ("Mississauga", "ON", "L"), ("Ottawa", "ON", "K"),
           ("Montreal", "QC", "H"), ("Vancouver", "BC", "V")]
_FIRST = ["Aisha", "Ben", "Chloe", "Dev", "Elena", "Farhan", "Grace", "Hiro", "Isla", "Jamal",
          "Kiran", "Lena", "Marco", "Nadia", "Owen", "Priya", "Quinn", "Ravi", "Sofia", "Theo"]
_LAST = ["Patel", "Nguyen", "Smith", "Tremblay", "Wong", "Garcia", "Khan", "Roy", "Martin", "Singh"]
_STREETS = ["Maple Ave", "King St W", "Queen St E", "Bloor St", "Lakeshore Blvd", "Elm Dr",
            "Rue Sainte-Catherine", "Granville St", "Bank St", "Dundas St"]
_SENDERS = ["Northwind Outfitters", "PixelHub Electronics", "Greenleaf Pharmacy",
            "BookNook", "Urban Threads", "HomeBox Furniture"]


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def connect(path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def _postal(rng: random.Random, first: str) -> str:
    L = "ABCEGHJKLMNPRSTVWXYZ"
    return f"{first}{rng.randint(0, 9)}{rng.choice(L)}{rng.randint(0, 9)}{rng.choice(L)}{rng.randint(0, 9)}"


def seed(conn: sqlite3.Connection, n_customers: int = 60, n_parcels: int = 250,
         seed_val: int = 7, now: datetime | None = None) -> None:
    """Deterministic demo data spread across every lifecycle status."""
    rng = random.Random(seed_val)
    now = now or now_utc()
    with conn:
        for t in ("audit", "claims", "events", "parcels", "addresses", "customers", "depots"):
            conn.execute(f"DELETE FROM {t}")
        conn.executemany("INSERT INTO depots VALUES (?,?,?)", DEPOTS)
        for cid in range(1, n_customers + 1):
            name = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
            phone = f"+1{rng.choice(['416', '647', '905', '613', '514', '604'])}{rng.randint(2000000, 9999999)}"
            conn.execute("INSERT INTO customers VALUES (?,?,?,?)",
                         (cid, name, phone, f"{name.split()[0].lower()}{cid}@example.com"))
            city, prov, letter = rng.choice(_CITIES)
            conn.execute("INSERT INTO addresses VALUES (?,?,?,?,?,?)",
                         (cid, cid, f"{rng.randint(1, 2400)} {rng.choice(_STREETS)}", city, prov,
                          _postal(rng, letter)))
        weights = [5, 6, 20, 12, 15, 25, 10, 3, 2]
        seen: set[str] = set()
        for _ in range(n_parcels):
            while True:
                tid = make_id("".join(rng.choice(ALPHABET) for _ in range(8)))
                if tid not in seen:
                    seen.add(tid)
                    break
            cid = rng.randint(1, n_customers)
            status = rng.choices(STATUSES, weights)[0]
            created = now - timedelta(days=rng.randint(1, 12), hours=rng.randint(0, 23))
            if status == "DELIVERED":
                eta = now - timedelta(days=rng.randint(0, 20))
            elif status in ("LABEL_CREATED", "PICKED_UP", "IN_TRANSIT"):
                eta = now + timedelta(days=rng.randint(1, 5))
            else:
                eta = now + timedelta(days=rng.randint(-8, 1))
            exc = rng.choice(["NO_ONE_HOME", "ACCESS_ISSUE", "ADDRESS_ISSUE"]) if status == "DELIVERY_EXCEPTION" else None
            attempts = rng.randint(1, 2) if status == "DELIVERY_EXCEPTION" else (1 if status == "DELIVERED" else 0)
            hold = rng.choice(DEPOTS)[0] if status == "HELD_AT_DEPOT" else None
            conn.execute(
                """INSERT INTO parcels (tracking_id, customer_id, address_id, sender, status, eta,
                   signature_required, attempts, address_changes, hold_depot, exception_reason,
                   created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (tid, cid, cid, rng.choice(_SENDERS), status, eta.date().isoformat(),
                 int(rng.random() < 0.25), attempts, 0, hold, exc, iso(created), iso(now)))
            # plausible history up to the current status
            path = STATUSES[: STATUSES.index(status) + 1] if STATUSES.index(status) <= 5 else STATUSES[:5] + [status]
            for k, st in enumerate(path):
                ts = created + (now - created) * (k + 1) / (len(path) + 1)
                conn.execute("INSERT INTO events (tracking_id, ts, status, detail) VALUES (?,?,?,?)",
                             (tid, iso(ts), st, exc if st == "DELIVERY_EXCEPTION" else None))


def existing_ids(conn: sqlite3.Connection, ids: list[str]) -> set[str]:
    """Batch existence check used by the ID resolver (indexed PK lookup)."""
    out: set[str] = set()
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        q = f"SELECT tracking_id FROM parcels WHERE tracking_id IN ({','.join('?' * len(chunk))})"
        out.update(r[0] for r in conn.execute(q, chunk))
    return out


def log_audit(conn: sqlite3.Connection, tool: str, args: dict, result: dict,
              call_id: str | None = None) -> None:
    with conn:
        conn.execute(
            "INSERT INTO audit (ts, call_id, tool, args, allowed, reason_code, result) VALUES (?,?,?,?,?,?,?)",
            (iso(now_utc()), call_id, tool, json.dumps(args, default=str),
             None if "allowed" not in result else int(bool(result["allowed"])),
             result.get("reason_code"), json.dumps(result, default=str)))
