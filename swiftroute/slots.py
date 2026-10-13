"""Slots: the answers this agent collects that have an exact, checkable form.

A tracking number and a postal code are not open conversation. When the agent
has just asked for one, the caller's next words are handled in code:

  * fragments are merged across turns ("It's V8." ... "A1L1"), because callers
    pause mid-code and turn detection, however good, will sometimes split them
  * a complete answer goes straight to the tool, with no LLM call

Real call #5: the postal code arrived in pieces, and when it finally came in
full, qwen2.5:3b didn't call verify_caller. It repeated its previous "that
doesn't match" line. Taking the LLM out of this step fixes that, and saves a
round trip (~1.5-4 s on a laptop) on every ID and postal-code turn.
"""
from __future__ import annotations

import re

from . import ids

_ASKS_ID = re.compile(r"tracking number|read it (again|once more)|one character at a time|word for each letter", re.I)
_ASKS_POSTAL = re.compile(r"postal code", re.I)


def expectation(agent_reply: str) -> str:
    """What the caller's next turn is likely to be, from what the agent just asked."""
    question = " ".join(s for s in re.split(r"(?<=[.?!])\s+", agent_reply or "") if s.endswith("?")) or ""
    if _ASKS_ID.search(question):
        return "id"
    if _ASKS_POSTAL.search(question):
        return "postal"
    # "Sorry, that postal code doesn't match. Could you check it and say it again?"
    if _ASKS_POSTAL.search(agent_reply or "") and re.search(r"\b(again|check)\b", question, re.I):
        return "postal"
    return "open"


_CHATTER = set(ids._FILLER) | {"yeah", "yes", "sure", "code", "postal", "it", "that's", "thats", "its", "it's",
                                "as", "in", "for", "like", "sorry", "again", "uh", "hmm"}


def code_only(text: str) -> bool:
    """True if the utterance is just a (piece of a) code plus filler, e.g. "Yeah, it's V8."
    Anything else ("I don't have it, I'm the owner") goes to the LLM as usual."""
    toks = re.findall(r"[a-z0-9'-]+", text.lower())
    if not toks:
        return False
    for i, t in enumerate(toks):
        if t in _CHATTER or ids._tok_to_chars(t) is not None or t in ids._NATO:
            continue
        # the word after "for"/"as in" is a spelling aid ("R for rat")
        if i > 0 and toks[i - 1] in ("for", "in", "like", "4"):
            continue
        if re.fullmatch(r"[a-z]4[a-z]*", t):
            continue
        return False
    return True


def postal_chars(text: str) -> str:
    """Alphanumerics that could belong to a postal code ('It's V8.' -> 'V8')."""
    spoken = ids.parse_spoken(text) if " " in text.strip() else text.upper()
    return re.sub(r"[^A-Z0-9]", "", spoken)


def complete_postal(text: str) -> str | None:
    return ids.normalise_postal(text)


def is_postal_fragment(text: str) -> bool:
    n = len(postal_chars(text))
    return 1 <= n <= 5


def id_chars(text: str) -> str:
    return ids.parse_spoken(text)


def complete_id(text: str) -> bool:
    parsed = id_chars(text)
    return bool(parsed) and any(ids.is_valid(v) for v, _ in ids.heard_variants(parsed))


def is_id_fragment(text: str) -> bool:
    return 1 <= len(id_chars(text)) <= 8
