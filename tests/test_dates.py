from datetime import date

import pytest

from swiftroute.dates import parse_day, say_date

WED = date(2026, 10, 14)


@pytest.mark.parametrize("said,expected", [
    ("Friday", date(2026, 10, 16)),
    ("friday, please", date(2026, 10, 16)),
    ("Saturday", date(2026, 10, 17)),            # parsed faithfully; policy rejects the weekend
    ("Monday then", date(2026, 10, 19)),
    ("next Friday", date(2026, 10, 16)),
    ("Friday next week", date(2026, 10, 23)),
    ("Wednesday", date(2026, 10, 21)),           # said on a Wednesday = next week
    ("tomorrow", date(2026, 10, 15)),
    ("Monday the 26th", date(2026, 10, 26)),
    ("the 19th", date(2026, 10, 19)),
    ("October 20th", date(2026, 10, 20)),
    ("20 Oct", date(2026, 10, 20)),
    ("2026-10-21", date(2026, 10, 21)),
    ("whenever", None),
])
def test_parse_day(said, expected):
    assert parse_day(said, WED) == expected


def test_say_date():
    assert say_date(date(2026, 10, 16)) == "Friday, October 16"


def test_weekend_reschedule_is_refused_with_spoken_options():
    from swiftroute.db import connect, seed
    from swiftroute.tools import CallSession
    from datetime import datetime, timezone
    now = datetime(2026, 10, 14, 15, tzinfo=timezone.utc)
    conn = connect(":memory:")
    seed(conn, n_parcels=200, now=now)
    r = conn.execute("SELECT p.tracking_id, a.postal, c.phone FROM parcels p JOIN addresses a ON a.id = p.address_id "
                     "JOIN customers c ON c.id = p.customer_id WHERE p.status = 'DELIVERY_EXCEPTION' "
                     "AND p.attempts < 3 LIMIT 1").fetchone()
    s = CallSession(conn, caller_phone=r["phone"], now=now)
    s.find_parcel(r["tracking_id"]); s.verify_caller(r["postal"])
    out = s.reschedule_delivery("Saturday")
    assert out["reason_code"] == "RESCHED_DATE_INVALID" and out["requested"] == "Saturday, October 17"
    assert "Monday, October 19" in out["available_dates"]
    ok = s.reschedule_delivery("Monday then")
    assert ok["allowed"] and "Monday, October 19" in ok["explanation"]
