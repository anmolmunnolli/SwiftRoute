from datetime import date, datetime, timedelta, timezone

import pytest

from swiftroute import ids, policy
from swiftroute.db import connect, existing_ids, seed
from swiftroute.simulator import Simulator
from swiftroute.tools import CallSession

NOW = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)  # a Wednesday


@pytest.fixture
def conn():
    c = connect(":memory:")
    seed(c, n_parcels=300, now=NOW)
    return c


def pick(conn, status, **extra):
    where = " AND ".join([f"p.{k} = ?" for k in extra])
    q = ("SELECT p.*, a.postal, c.phone FROM parcels p JOIN addresses a ON a.id = p.address_id "
         "JOIN customers c ON c.id = p.customer_id WHERE p.status = ?" + (f" AND {where}" if where else ""))
    r = conn.execute(q + " LIMIT 1", (status, *extra.values())).fetchone()
    assert r, f"no seeded parcel with {status} {extra}"
    return dict(r)


# ---------------------------------------------------------------- checksum
def test_checksum_catches_every_single_substitution():
    tid = ids.make_id("4K7BX2M9")
    assert ids.is_valid(tid)
    for i in range(2, len(tid)):
        for c in ids.ALPHABET:
            if c != tid[i]:
                assert not ids.is_valid(tid[:i] + c + tid[i + 1:])


def test_checksum_catches_adjacent_transpositions():
    tid = ids.make_id("4K7BX2M9")
    for i in range(2, len(tid) - 2):
        if tid[i] != tid[i + 1]:
            assert not ids.is_valid(tid[:i] + tid[i + 1] + tid[i] + tid[i + 2:])


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("spoken,expected", [
    ("S R four K seven B as in bravo X two M nine", "SR4K7BX2M9"),
    ("sierra romeo four kilo seven bravo x-ray two mike nine", "SR4K7BX2M9"),
    ("um it's S R double five three", "SR553"),
    ("double you three oh", "W30"),
    ("S R 4 K 7", "SR4K7"),
    ("sr4k7bx2", "SR4K7BX2"),
    ("it's for kay", "4K"),
])
def test_parse_spoken(spoken, expected):
    assert ids.parse_spoken(spoken) == expected


# ---------------------------------------------------------------- resolving
def test_resolve_exact(conn):
    tid = pick(conn, "IN_TRANSIT")["tracking_id"]
    r = ids.resolve(tid, lambda b: existing_ids(conn, b))
    assert r.status == "exact" and r.best == tid and not r.needs_confirmation


@pytest.mark.parametrize("mutate", [
    lambda t: t[:4] + ("D" if t[4] != "D" else "B") + t[5:],   # sound-alike swap
    lambda t: t[:6] + t[7:],                                    # dropped char
    lambda t: t[:5] + t[5] + t[5:],                             # stutter
    lambda t: t[2:],                                            # prefix omitted
])
def test_resolve_recovers_common_asr_errors(conn, mutate):
    tid = pick(conn, "AT_DEPOT")["tracking_id"]
    r = ids.resolve(mutate(tid), lambda b: existing_ids(conn, b))
    assert tid in [c.tracking_id for c in r.candidates]
    assert r.needs_confirmation or r.best == tid


def test_resolve_not_found_for_garbage(conn):
    assert ids.resolve("SRZZZZZZZZZ", lambda b: existing_ids(conn, b)).status in ("not_found", "corrected", "ambiguous")
    assert ids.resolve("hello", lambda b: existing_ids(conn, b)).status == "not_found"


def test_read_back():
    assert ids.read_back("SR4K7BX2M9Q") == "ending in two, M as in Mike, nine, Q as in Quebec"
    assert ids.read_back("SR4K7", last_n=None).startswith("S as in Sierra, R as in Romeo, four")


@pytest.mark.parametrize("spoken,expected", [
    ("M5V 2T6", "M5V2T6"),
    ("m five v two t six", "M5V2T6"),
    ("M5V ZT6", "M5V2T6"),        # Z heard in a digit slot -> 2
    ("8 5 N 1 A 1", "B5N1A1"),    # 8 heard in a letter slot -> B
])
def test_postal(spoken, expected):
    assert ids.normalise_postal(spoken) == expected


def test_smart_read_back_covers_the_corrected_char():
    chosen = "SRZAQN79FCJ"
    rb = ids.smart_read_back(chosen, heard="SRZAQD79FCJ")
    assert rb.startswith("ending in N as in November")       # the fixed char is read back
    assert ids.smart_read_back(chosen, heard=chosen) == ids.read_back(chosen)
    # ambiguous options are read from where they diverge
    assert "K as in Kilo" in ids.smart_read_back("SR4K7BX2M9Q", others=["SR4A7BX2M9Q"])


def test_postal_rejects_wrong_length():
    assert ids.normalise_postal("M5V 2T") is None


# ------------------------------------------------------------------ policy
def test_address_change_blocked_out_for_delivery():
    p = {"status": "OUT_FOR_DELIVERY", "address_changes": 0, "signature_required": 0}
    d = policy.can_change_address(p)
    assert not d.allowed and d.reason_code == "ADDR_LOCKED_STATUS"


def test_address_change_limit_and_signature():
    base = {"status": "IN_TRANSIT", "address_changes": 0, "signature_required": 0}
    assert policy.can_change_address(base).allowed
    assert policy.can_change_address({**base, "address_changes": 1}).reason_code == "ADDR_CHANGE_LIMIT"
    assert policy.can_change_address({**base, "signature_required": 1}).reason_code == "ADDR_SIGNATURE_REQUIRED"


def test_reschedule_weekdays_only():
    p = {"status": "DELIVERY_EXCEPTION", "attempts": 1}
    sat = date(2026, 10, 17)
    assert policy.can_reschedule(p, sat, NOW).reason_code == "RESCHED_DATE_INVALID"
    assert policy.can_reschedule(p, date(2026, 10, 15), NOW).allowed
    assert policy.can_reschedule(p, NOW.date(), NOW).reason_code == "RESCHED_DATE_INVALID"  # not same day
    assert policy.can_reschedule({**p, "attempts": 3}, date(2026, 10, 15), NOW).reason_code == "RESCHED_ATTEMPTS_EXHAUSTED"


def test_claims():
    eta = (NOW.date() - timedelta(days=2)).isoformat()
    transit = {"status": "IN_TRANSIT", "eta": eta}
    assert policy.can_open_claim(transit, "missing", NOW, False).reason_code == "CLAIM_TOO_EARLY"
    late = {**transit, "eta": (NOW.date() - timedelta(days=6)).isoformat()}
    assert policy.can_open_claim(late, "missing", NOW, False).allowed
    assert policy.can_open_claim(transit, "damaged", NOW, False).reason_code == "CLAIM_DAMAGED_NOT_DELIVERED"
    assert policy.can_open_claim({"status": "DELIVERED", "eta": eta}, "damaged", NOW, True).reason_code == "CLAIM_EXISTS"
    old = {"status": "DELIVERED", "eta": (NOW.date() - timedelta(days=40)).isoformat()}
    assert policy.can_open_claim(old, "damaged", NOW, False).reason_code == "CLAIM_WINDOW_CLOSED"


# ---------------------------------------------------------- tool layer flow
def test_cannot_act_without_verification(conn):
    p = pick(conn, "IN_TRANSIT")
    s = CallSession(conn, caller_phone=p["phone"], now=NOW)
    assert s.get_status()["reason_code"] == "NO_PARCEL_CONFIRMED"
    s.find_parcel(p["tracking_id"])
    s.confirm_parcel(p["tracking_id"])
    assert s.get_status()["reason_code"] == "NEEDS_VERIFICATION"
    assert s.hold_at_depot()["reason_code"] == "NEEDS_VERIFICATION"


def test_withheld_caller_id_is_not_a_bypass(conn):
    p = pick(conn, "IN_TRANSIT")
    s = CallSession(conn, caller_phone=None, now=NOW)
    s.find_parcel(p["tracking_id"])
    assert s.verify_caller(p["postal"])["level"] == "partial"
    assert s.hold_at_depot()["reason_code"] == "NEEDS_FULL_VERIFICATION"


def test_wrong_postal_rejected(conn):
    p = pick(conn, "IN_TRANSIT")
    s = CallSession(conn, caller_phone="+15550000000", now=NOW)
    s.find_parcel(p["tracking_id"]); s.confirm_parcel(p["tracking_id"])
    assert s.verify_caller("A1A 1A1")["reason_code"] == "ID_NOT_VERIFIED"


def test_partial_verification_allows_status_not_changes(conn):
    p = pick(conn, "IN_TRANSIT")
    s = CallSession(conn, caller_phone="+15550000000", now=NOW)   # unknown caller ID
    s.find_parcel(p["tracking_id"]); s.confirm_parcel(p["tracking_id"])
    v = s.verify_caller(p["postal"])
    assert v["level"] == "partial"
    assert s.get_status()["allowed"]
    assert s.hold_at_depot()["reason_code"] == "NEEDS_FULL_VERIFICATION"


def test_full_flow_hold_at_depot(conn):
    p = pick(conn, "DELIVERY_EXCEPTION")
    s = CallSession(conn, caller_phone=p["phone"], now=NOW)
    r = s.find_parcel(" ".join(p["tracking_id"]))
    assert r["tracking_id"] == p["tracking_id"]
    s.confirm_parcel(r["tracking_id"])
    assert s.verify_caller(" ".join(p["postal"]))["level"] == "full"
    out = s.hold_at_depot()
    assert out["allowed"] and s.resolved
    assert conn.execute("SELECT status FROM parcels WHERE tracking_id = ?", (p["tracking_id"],)).fetchone()[0] == "HELD_AT_DEPOT"
    assert conn.execute("SELECT COUNT(*) FROM audit WHERE call_id = ?", (s.call_id,)).fetchone()[0] >= 4


def test_hallucinated_tool_and_args_are_contained(conn):
    s = CallSession(conn, now=NOW)
    assert s.call("drop_tables", {})["reason_code"] == "UNKNOWN_TOOL"
    assert s.call("_parcel", {})["reason_code"] == "UNKNOWN_TOOL"
    r = s.call("get_status", {"nope": 1})          # extra arg dropped, tool still runs its own checks
    assert r["reason_code"] != "BAD_ARGS" and r["ignored_args"] == ["nope"]
    assert s.call("verify_caller", {})["reason_code"] == "BAD_ARGS"   # missing required arg still fails


def test_confirm_without_id_uses_single_pending(conn):
    s = CallSession(conn, now=NOW)
    s.pending_ids, s.pending_turn, s.turn = ["SRAAAAAAAA1"], 0, 1
    assert s.call("confirm_parcel", {})["reason_code"] == "ID_CONFIRMED"
    assert s.tracking_id == "SRAAAAAAAA1"


def test_model_transfer_without_summary_is_not_bad_args(conn):
    s = CallSession(conn, now=NOW)
    s.caller_wants_human = True
    assert s.call("transfer_to_human", {"reason": "asked"})["reason_code"] == "TRANSFER"


def test_confirm_cannot_jump_to_unoffered_id(conn):
    s = CallSession(conn, now=NOW)
    assert s.confirm_parcel("SR00000000X")["reason_code"] == "ID_UNKNOWN"


# --------------------------------------------------------------- simulator
def test_simulator_emits_events_and_respects_reschedule(conn):
    sim = Simulator(conn, seed=1, move_prob=1.0)
    seen = []
    sim.subscribe(lambda t, st, d: seen.append(st))
    p = pick(conn, "DELIVERY_EXCEPTION")
    conn.execute("UPDATE parcels SET reschedule_date = ?, attempts = 1 WHERE tracking_id = ?",
                 ((NOW.date() + timedelta(days=2)).isoformat(), p["tracking_id"]))
    sim.tick(NOW)
    assert seen
    assert conn.execute("SELECT status FROM parcels WHERE tracking_id = ?", (p["tracking_id"],)).fetchone()[0] == "DELIVERY_EXCEPTION"


def test_postal_code_guessing_is_locked_after_three_tries(conn):
    p = pick(conn, "IN_TRANSIT")
    s = CallSession(conn, caller_phone=p["phone"], now=NOW)
    s.find_parcel(p["tracking_id"])
    for guess in ("A1A1A1", "B2B2B2", "C3C3C3"):
        assert s.verify_caller(guess)["reason_code"] == "ID_NOT_VERIFIED"
    out = s.verify_caller(p["postal"])                 # even the right one, now
    assert out["reason_code"] == "VERIFY_LOCKED" and "TRANSFER" in s.reason_codes
    assert s.identity is None or not s.identity.allowed



def test_status_grounds_next_delivery_and_window(conn):
    p = pick(conn, "DELIVERY_EXCEPTION")
    s = CallSession(conn, caller_phone=p["phone"], now=NOW)
    s.find_parcel(p["tracking_id"]); s.verify_caller(p["postal"])
    st = s.get_status()
    assert st["next_delivery"] == "Thursday, October 15"          # next business day after Wed Oct 14
    assert "9 am and 6 pm" in st["delivery_window"] and "exact time" in st["delivery_window"]
