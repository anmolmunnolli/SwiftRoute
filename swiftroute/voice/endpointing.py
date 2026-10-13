"""Context-aware endpointing: when is the caller done talking?

Generic turn detection has to guess. This agent knows what it just asked for,
and for IDs it can *check* whether the answer is complete.

Real call #1 (MacBook Air): every turn waited exactly 3.0 s after the caller
stopped. That was Pipecat's silence fallback, firing because smart-turn judged
most turns "incomplete". A single fixed number can't be right here:

  * reading an ID, people pause between chunks ("S R V C ... M C 7"), so a
    short fallback cuts them off mid-ID
  * "yes" or "where's my parcel?" is clearly finished, and 3 s of dead air
    there is what makes a bot feel slow

So:
  1. the silence fallback follows the question the agent just asked
     (long while collecting an ID, short for open questions)
  2. if a transcript already parses to a checksum-valid tracking number, or a
     full postal code, the turn is over: start the agent immediately instead
     of waiting for the timer. The checksum makes this safe; a half-read ID
     can't pass it.
"""
from __future__ import annotations


from loguru import logger

from .. import ids
from ..slots import expectation  # noqa: F401  (one definition, shared with the agent)

FALLBACK_S = {"id": 2.2, "postal": 1.6, "open": 0.9}


def answer_is_complete(expect: str, text: str) -> bool:
    if expect == "id":
        parsed = ids.parse_spoken(text)
        return any(ids.is_valid(v) for v, _ in ids.heard_variants(parsed)) if parsed else False
    if expect == "postal":
        return ids.normalise_postal(text) is not None
    return False


def set_silence_fallback(analyzer, secs: float) -> None:
    """Retune smart-turn's silence fallback between turns.

    Pipecat reads it from analyzer params once at init; there's no public
    setter in 1.12, so set both fields it uses. Guarded so a Pipecat upgrade
    degrades to "fixed fallback" rather than crashing a call.
    """
    if analyzer is None:
        return
    try:
        analyzer._params.stop_secs = secs
        analyzer._stop_ms = secs * 1000
    except AttributeError:
        logger.warning("smart-turn internals changed; adaptive fallback disabled")
