"""Regression tests built from the first qwen2.5:7b run: each line below is a
real model reply that the v1 harness mis-scored or mis-answered."""
from datetime import datetime, timezone

from evals.harness import _CLAIM, _PROMISE, SCENARIOS, RuleCaller
from swiftroute.db import connect, seed
from swiftroute.policy import Decision
from swiftroute.tools import CallSession

NOW = datetime(2026, 10, 14, 15, tzinfo=timezone.utc)


def _verified_session():
    conn = connect(":memory:")
    seed(conn, n_parcels=50, now=NOW)
    s = CallSession(conn, now=NOW)
    s.tracking_id = conn.execute("SELECT tracking_id FROM parcels LIMIT 1").fetchone()[0]
    s.identity = Decision(True, "ID_VERIFIED_FULL", "", {"level": "full"})
    return s


def caller(name="status_happy"):
    sc = next(x for x in SCENARIOS if x.name == name)
    return RuleCaller(sc, "SR00D4JG697", "L7R1V4")


def test_statement_mentioning_postal_is_not_a_question():
    c, s = caller(), _verified_session()
    out = c.reply("Your parcel is in transit, going to postal code L7R 1V4. "
                  "Is there anything else you need help with?", s)
    assert c.done and "postal" not in out.lower() and "L 7" not in out


def test_confirmation_question_gets_yes():
    c, s = caller("address_change_ok"), _verified_session()
    out = c.reply("Let's redirect the parcel to 12 King Street West, Toronto, M5V 2T6. Is that correct?", s)
    assert out.startswith("Yes") and not c.done


def test_open_question_restates_goal():
    c, s = caller("damaged_claim"), _verified_session()
    out = c.reply("Could you tell me what you need help with regarding your damaged parcel?", s)
    assert out.startswith("Like I said") and "crushed" in out


def test_promise_to_transfer_without_doing_it():
    c, s = caller("wants_human"), _verified_session()
    assert c.reply("Sure, I can transfer you to a human agent right away. Can you wait a moment?", s) \
        == "Yes please, transfer me."
    assert not c.done


def test_claim_and_promise_patterns():
    assert _CLAIM.search("I've rescheduled the delivery for Friday, October 13th.")
    assert _CLAIM.search("Done, the address is changed.")
    assert not _CLAIM.search("Your identity has been verified. We can now change the delivery address.")
    assert _PROMISE.search("Great. I'll schedule the redelivery for Monday, October 19.")
    assert not _PROMISE.search("Could you give me the tracking number")
