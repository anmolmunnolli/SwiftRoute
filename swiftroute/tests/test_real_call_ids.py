"""Regression tests from the first real voice call (2026-10-11, MacBook Air,
faster-whisper small.en). Target parcel: SRVCMC7RAJ1. These are the actual
transcripts Whisper produced; v1 of the parser failed all three."""
from datetime import datetime, timezone

import pytest

from swiftroute import ids
from swiftroute.db import connect, seed
from swiftroute.tools import CallSession

TARGET = "SRVCMC7RAJ1"
READING_1 = "Yeah, it's um... S Odd. V C. M C. 7. Hard E G 1."          # R heard as "odd"/"hard", "A J" as "E G"
READING_2 = "S. V C. M C. 7 Odd. 1."                                    # R dropped, A J dropped
READING_3 = ("S for Snake. R for rat. V for 1. C for car M4MAT C for car. 7. "
             "R 4 RAT A4 and... J for Joker 1.")                        # spelling-by-word, ASR wrote 'for' as '4'
NOW = datetime(2026, 10, 14, 15, tzinfo=timezone.utc)


@pytest.fixture
def session():
    conn = connect(":memory:")
    seed(conn, now=NOW)                     # default seed contains the target parcel
    assert conn.execute("SELECT 1 FROM parcels WHERE tracking_id = ?", (TARGET,)).fetchone()
    return CallSession(conn, now=NOW)


def test_spelling_by_word_parses_exactly():
    assert ids.parse_spoken(READING_3) == TARGET


@pytest.mark.parametrize("spoken,expected", [
    ("R for rat", "R"), ("R 4 RAT", "R"), ("M4MAT", "M"), ("A4 and", "A"), ("J for Joker", "J"),
    ("V for Victor", "V"), ("M like Mike", "M"), ("B as in banana", "B"), ("D as in bravo", "B"),
    ("B 4 seven", "B47"), ("S odd", "SR"),
])
def test_spelling_idioms(spoken, expected):
    assert ids.parse_spoken(spoken) == expected


@pytest.mark.parametrize("reading", [READING_1, READING_2])
def test_badly_misheard_reading_recovered_and_must_be_confirmed(session, reading):
    out = session.find_parcel(reading)
    assert out["reason_code"] == "ID_RECOVERED_MULTI"
    assert session.pending_ids == [TARGET] and session.tracking_id is None   # caller must confirm
    assert out["reply"].startswith("Just to check")


def test_garbage_is_not_recovered_and_reask_escalates(session):
    replies = [session.find_parcel("1 2 3 4 5 6 7 8")["reply"] for _ in range(3)]
    assert session.pending_ids == [] and session.tracking_id is None
    assert "one character at a time" in replies[0]
    assert "R for Romeo" in replies[1]
    assert "colleague" in replies[2]


def test_scan_combines_readings():
    r1, r2 = ids.parse_spoken(READING_1), ids.parse_spoken(READING_2)
    pool = [TARGET, "SRVJMSHRTLW", "SRZUCUREG18", "SRVDVMD7BRN"]
    top = ids.scan([r1, r2], pool)
    assert top[0].tracking_id == TARGET and top[1].cost - top[0].cost > 1.5
