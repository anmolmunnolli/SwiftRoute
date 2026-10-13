"""Tracking-ID and postal-code capture from noisy speech transcripts.

This is the part most voice-agent demos skip. Callers read alphanumeric IDs
over a phone line, ASR mangles them (B/D/E/P/T/V all sound alike at 8 kHz),
and a naive agent either fails or acts on the wrong parcel.

Pipeline:
    transcript --parse_spoken--> raw chars --resolve--> ranked candidates
                                                  |
                         checksum filter + confusion-weighted edits + DB lookup

Tracking ID format:  SR + 8 body chars + 1 check char   e.g. SR4K7BX2M9Q
  - alphabet excludes I and O (too easily confused with 1 and 0)
  - check char is Luhn mod N over the body, so a single substitution
    or adjacent transposition is always detected.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable

from . import trace

ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I, no O
N = len(ALPHABET)
PREFIX = "SR"
BODY_LEN = 8
ID_LEN = len(PREFIX) + BODY_LEN + 1
_IDX = {c: i for i, c in enumerate(ALPHABET)}


# --------------------------------------------------------------------------
# Checksum (Luhn mod N)
# --------------------------------------------------------------------------
def check_char(body: str) -> str:
    factor, total = 2, 0
    for ch in reversed(body):
        addend = factor * _IDX[ch]
        factor = 1 if factor == 2 else 2
        total += addend // N + addend % N
    return ALPHABET[(N - total % N) % N]


def is_valid(tracking_id: str) -> bool:
    t = tracking_id.upper()
    if len(t) != ID_LEN or not t.startswith(PREFIX):
        return False
    body, chk = t[2:-1], t[-1]
    if any(c not in _IDX for c in body + chk):
        return False
    return check_char(body) == chk


def make_id(body: str) -> str:
    return PREFIX + body + check_char(body)


# --------------------------------------------------------------------------
# Spoken transcript -> characters
# --------------------------------------------------------------------------
_DIGITS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "won": "1", "two": "2", "to": "2",
    "too": "2", "three": "3", "four": "4", "for": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "ate": "8", "nine": "9",
}
_NATO = {
    "alpha": "A", "alfa": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E",
    "foxtrot": "F", "golf": "G", "hotel": "H", "india": "I", "juliet": "J", "juliett": "J",
    "kilo": "K", "lima": "L", "mike": "M", "november": "N", "oscar": "O", "papa": "P",
    "quebec": "Q", "romeo": "R", "sierra": "S", "tango": "T", "uniform": "U",
    "victor": "V", "whiskey": "W", "whisky": "W", "xray": "X", "x-ray": "X",
    "yankee": "Y", "zulu": "Z",
}
_LETTER_NAMES = {
    "ay": "A", "a": "A", "bee": "B", "be": "B", "see": "C", "sea": "C", "cee": "C",
    "dee": "D", "ee": "E", "eff": "F", "ef": "F", "gee": "G", "jee": "G", "aitch": "H",
    "jay": "J", "kay": "K", "el": "L", "elle": "L", "em": "M", "en": "N", "pee": "P",
    "pea": "P", "cue": "Q", "queue": "Q", "are": "R", "ar": "R", "ess": "S", "es": "S",
    "tee": "T", "tea": "T", "you": "U", "vee": "V", "ex": "X", "why": "Y", "zed": "Z",
    "zee": "Z",
    # Accent homophones seen in real calls: an "aar" R came out of Whisper as "odd"/"hard".
    "odd": "R", "hard": "R", "aar": "R", "our": "R",
}
_SPELL_LINKS = {"for", "4", "like"}            # "R for rat", "R 4 RAT", "M like Mike"
_FILLER = {"um", "uh", "er", "erm", "like", "so", "its", "it's", "is", "the", "number",
           "tracking", "my", "and", "then", "dash", "hyphen", "space", "ok", "okay"}
_MULT = {"double": 2, "triple": 3}


def _tok_to_chars(tok: str) -> str | None:
    if tok in _DIGITS:
        return _DIGITS[tok]
    if tok in _NATO:
        return _NATO[tok]
    if tok in _LETTER_NAMES:
        return _LETTER_NAMES[tok]
    if re.fullmatch(r"[a-z0-9]+", tok) and (
        len(tok) == 1
        or any(c.isdigit() for c in tok)
        or (len(tok) <= 4 and not re.search(r"[aeiouy]", tok))
    ):
        # ASR often emits chunks like "4k7", "sr" or "kbx"; spell them out.
        # Vowel-bearing words ("it", "yes") are real words, not ID chunks.
        return tok.upper()
    return None


def _letter(tok: str) -> str | None:
    if re.fullmatch(r"[a-z]", tok):
        return tok.upper()
    return _LETTER_NAMES.get(tok)


def _is_word(tok: str) -> bool:
    """A spelling-aid word like 'rat' or 'Joker', not another ID character."""
    return (tok.isalpha() and len(tok) >= 2 and tok not in _DIGITS and tok not in _LETTER_NAMES
            and tok not in _FILLER)


def parse_spoken(text: str) -> str:
    """Turn 'S R four seven B as in bravo double five...' into 'SR47B55...'.

    Also handles how people actually spell over the phone, taken from real
    test calls: 'R for rat', 'J for Joker', and Whisper's renderings of it,
    'R 4 RAT', 'A4 and', 'M4MAT' (ASR writes the word 'for' as the digit 4).
    """
    toks = re.findall(r"[a-z0-9'-]+", text.lower().replace("x ray", "xray"))
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        nxt2 = toks[i + 2] if i + 2 < len(toks) else None
        # "M4MAT" = "M for mat" fused into one token
        m = re.fullmatch(r"([a-z])4([a-z]{2,})", t)
        if m:
            out.append(m[1].upper())
            i += 1
            continue
        # "A4 and ..." = "A for <word>"
        m = re.fullmatch(r"([a-z])4", t)
        if m and nxt and nxt.isalpha() and len(nxt) >= 2 and nxt not in _DIGITS and nxt not in _LETTER_NAMES:
            out.append(m[1].upper())
            i += 2
            continue
        L = _letter(t)
        # "B as in bravo" / "B as in banana": trust the NATO word if it is one, else the letter
        if L and nxt == "as" and nxt2 in ("in", "and") and i + 3 < len(toks):
            out.append(_NATO.get(toks[i + 3], L))
            i += 4
            continue
        # "R for rat" / "V for Victor" / "M like Mike". After a letter, the word "for" is
        # always this idiom; a bare "4" only when a real word follows ("B 4 seven" is B,4,7).
        if L and nxt in _SPELL_LINKS and nxt2 and (nxt in ("for", "like") or _is_word(nxt2)):
            out.append(_NATO.get(nxt2, L))
            i += 3
            continue
        # "double you" is W, "double five" is 55
        if t == "double" and i + 1 < len(toks) and toks[i + 1] == "you":
            out.append("W")
            i += 2
            continue
        if t in _MULT and i + 1 < len(toks):
            ch = _tok_to_chars(toks[i + 1])
            if ch and len(ch) == 1:
                out.append(ch * _MULT[t])
                i += 2
                continue
        if t in _FILLER:
            i += 1
            continue
        ch = _tok_to_chars(t)
        if ch:
            out.append(ch)
        i += 1
    s = "".join(out)
    # In this alphabet O and I can't occur; a heard O/I is almost always 0/1
    return s.replace("O", "0").replace("I", "1")


# --------------------------------------------------------------------------
# Confusion-aware candidate generation
# --------------------------------------------------------------------------
# Characters that sound alike on a narrowband phone line. Same-group swaps
# are cheap (0.4), any other swap costs 1.0.
CONFUSION_GROUPS = [
    "BDEGPTVZ3C",  # the 'ee' set
    "MN",
    "FSX5",
    "AJK8H",
    "QU",
    "1",
    "9Y",
    "0",
    "R",
    "L",
    "W2",
    "46",
    "7",
]
_GROUP = {c: gi for gi, g in enumerate(CONFUSION_GROUPS) for c in g}
NEAR, FAR, INDEL = 0.4, 1.0, 0.8


# Learned from real calls (swiftroute/learning/confusions.py): heard -> true character
# pairs that callers confirmed after a read-back, with their reduced cost. Empty unless
# a benchmark-gated update has been promoted and applied. Exact matches never depend on
# these costs, so learning can change which candidate is offered, but never make the
# agent act on a parcel without the caller confirming it.
LEARNED: dict[tuple[str, str], float] = {}
_LEARNED_NEAR: dict[str, set[str]] = {}


def set_learned(costs: dict[tuple[str, str], float]) -> None:
    LEARNED.clear()
    LEARNED.update(costs)
    _LEARNED_NEAR.clear()
    for (a, b), c in costs.items():
        if c <= NEAR + 1e-9:
            _LEARNED_NEAR.setdefault(a, set()).add(b)


def sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    learned = LEARNED.get((a, b))
    ga, gb = _GROUP.get(a), _GROUP.get(b)
    base = NEAR if ga is not None and ga == gb else FAR
    return min(base, learned) if learned is not None else base


@dataclass
class Candidate:
    tracking_id: str
    cost: float
    edits: int


def _neighbours(s: str) -> Iterable[tuple[str, float, int]]:
    """Yield (string, cost, n_edits) within ~2 plausible ASR errors of s."""
    yield s, 0.0, 0
    L = len(s)
    near = {c: [d for d in ALPHABET if d != c and (_GROUP.get(d) == _GROUP.get(c) or d in _LEARNED_NEAR.get(c, ()))]
            for c in set(s)}
    # one substitution, any char
    for i in range(L):
        for d in ALPHABET:
            if d != s[i]:
                yield s[:i] + d + s[i + 1:], sub_cost(s[i], d), 1
    # two near substitutions
    for i, j in combinations(range(L), 2):
        for di in near.get(s[i], []):
            for dj in near.get(s[j], []):
                t = list(s)
                t[i], t[j] = di, dj
                yield "".join(t), sub_cost(s[i], di) + sub_cost(s[j], dj), 2
    # one dropped char (ASR missed it) -> insert
    for i in range(L + 1):
        for d in ALPHABET:
            yield s[:i] + d + s[i:], INDEL, 1
    # one extra char (ASR stuttered) -> delete
    for i in range(L):
        yield s[:i] + s[i + 1:], INDEL, 1
    # adjacent transposition
    for i in range(L - 1):
        if s[i] != s[i + 1]:
            yield s[:i] + s[i + 1] + s[i] + s[i + 2:], FAR, 1


def normalise_heard(raw: str) -> str:
    return heard_variants(raw)[0][0]


def heard_variants(raw: str) -> list[tuple[str, float]]:
    """Plausible readings of the start of the ID, each with a small penalty.

    Callers drop or mangle the "SR" prefix ("S odd V C ..." lost the R entirely
    in a real call), so instead of guessing one fix, try the few that make sense
    and let the checksum + DB decide.
    """
    s = re.sub(r"[^A-Z0-9]", "", raw.upper()).replace("O", "0").replace("I", "1")
    if s.startswith(PREFIX):
        return [(s, 0.0)]
    out: list[tuple[str, float]] = []
    if len(s) >= ID_LEN - 3:
        out.append((PREFIX + s, 0.2))                  # prefix skipped entirely
    if s[:1] == "S":
        out.append((PREFIX + s[1:], 0.2))              # "S <R missing> ..."
    if len(s) >= 2:
        out.append((PREFIX + s[2:], 0.3))              # prefix garbled ("ES", "5R", "FR")
    out.append((s, 0.0))
    seen, uniq = set(), []
    for v, pen in out:
        if v not in seen:
            seen.add(v)
            uniq.append((v, pen))
    return uniq


def weighted_distance(a: str, b: str) -> float:
    """Edit distance where sound-alike swaps are cheap: the scoring used when
    candidate generation from a single reading isn't enough."""
    prev = [j * INDEL for j in range(len(b) + 1)]
    for i in range(1, len(a) + 1):
        cur = [i * INDEL] + [0.0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + INDEL, cur[j - 1] + INDEL, prev[j - 1] + sub_cost(a[i - 1], b[j - 1]))
        prev = cur
    return prev[-1]


def _shortlist(variants: list[list[tuple[str, float]]], pool: list[str], per_variant: int = 150) -> list[str]:
    """Plain Levenshtein in C (rapidfuzz) to pick ~hundreds of plausible IDs,
    then the slower confusion-weighted score only on those."""
    try:
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein
    except ImportError:                    # still correct without it, just slower
        return pool
    keep: set[str] = set()
    for vs in variants:
        for v, _ in vs:
            keep.update(m[0] for m in process.extract(v, pool, scorer=Levenshtein.distance, limit=per_variant))
    return list(keep)


def scan(readings: list[str], all_ids: Iterable[str], top_k: int = 3) -> list[Candidate]:
    """Fallback when no single reading resolves: score every known ID against
    ALL readings the caller gave this call. Independent mishearings rarely hit
    the same character, so evidence adds up (in a real call, readings 1 and 2
    each had two errors, in different places).

    O(#ids) per call; fine to tens of thousands of active parcels. At carrier
    scale this would sit behind a phonetic n-gram index instead of a full scan.
    """
    variants = [heard_variants(r) for r in readings if r]
    if not variants:
        return []
    pool = list(all_ids)
    if len(pool) > 500:
        pool = _shortlist(variants, pool)
    scored = []
    for tid in pool:
        total = sum(min(weighted_distance(v, tid) + pen for v, pen in vs) for vs in variants)
        scored.append(Candidate(tid, total / len(variants), len(variants)))
    scored.sort(key=lambda c: (c.cost, c.tracking_id))
    return scored[:top_k]


@dataclass
class Resolution:
    heard: str
    status: str                 # "exact" | "corrected" | "ambiguous" | "not_found"
    candidates: list[Candidate]

    @property
    def best(self) -> str | None:
        return self.candidates[0].tracking_id if self.candidates else None

    @property
    def needs_confirmation(self) -> bool:
        return self.status != "exact"


def resolve(raw: str, exists: Callable[[list[str]], set[str]], top_k: int = 3) -> Resolution:
    """Map a heard ID to real tracking IDs.

    `exists` takes a batch of syntactically valid IDs and returns the subset
    that exist in the system, so this scales with an indexed DB lookup
    rather than a scan over every parcel.
    """
    t0 = time.perf_counter()
    variants = heard_variants(raw)
    heard = variants[0][0]
    best: dict[str, Candidate] = {}
    generated = 0
    for variant, penalty in variants:
        for cand, cost, edits in _neighbours(variant):
            generated += 1
            if not is_valid(cand):
                continue                   # checksum prunes ~97% of noise for free
            cost += penalty
            prev = best.get(cand)
            if prev is None or cost < prev.cost:
                best[cand] = Candidate(cand, cost, edits)
    real = exists(list(best)) if best else set()
    ranked = sorted((c for c in best.values() if c.tracking_id in real), key=lambda c: (c.cost, c.tracking_id))
    if not ranked:
        res = Resolution(heard, "not_found", [])
    elif ranked[0].cost == 0:
        res = Resolution(heard, "exact", ranked[:1])
    elif len(ranked) > 1 and ranked[1].cost - ranked[0].cost < 0.3:
        res = Resolution(heard, "ambiguous", ranked[:top_k])
    else:
        res = Resolution(heard, "corrected", ranked[:top_k])
    if trace.active():
        trace.emit("id.decode", raw=raw, heard=heard, heard_checksum_ok=is_valid(heard),
                   variants=[{"text": v, "penalty": round(p, 2)} for v, p in variants[:6]],
                   n_variants=len(variants), generated=generated, checksum_valid=len(best),
                   exist_in_db=len(real), status=res.status,
                   ranked=[{"id": c.tracking_id, "cost": round(c.cost, 2), "edits": c.edits,
                            "diff": [i for i, (a, b) in enumerate(zip(heard, c.tracking_id)) if a != b]}
                           for c in ranked[:5]],
                   ms=round((time.perf_counter() - t0) * 1000, 2))
    return res


# --------------------------------------------------------------------------
# Read-back
# --------------------------------------------------------------------------
_NATO_OUT = {v: k.capitalize() for k, v in _NATO.items() if k not in ("alfa", "juliett", "whisky", "x-ray")}
_NATO_OUT["X"] = "X-ray"
_DIGIT_OUT = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
              "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def read_back(tracking_id: str, last_n: int | None = 4) -> str:
    """Speakable confirmation. By default only reads the last 4 chars:
    reading all 11 aloud wastes ~4 s of call time for little extra safety."""
    t = tracking_id.upper()
    part = t[-last_n:] if last_n else t
    words = [_DIGIT_OUT[c] if c.isdigit() else f"{c} as in {_NATO_OUT[c]}" for c in part]
    lead = "ending in " if last_n and last_n < len(t) else ""
    return lead + ", ".join(words)


def first_diff(a: str, b: str) -> int:
    """Index in `b` of the first position where it departs from `a`
    (alignment-aware, so a dropped char doesn't shift everything)."""
    import difflib
    for tag, _, _, j1, _ in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal":
            return j1
    return len(b)


def smart_read_back(chosen: str, heard: str | None = None, others: list[str] | None = None) -> str:
    """Read back enough to actually confirm the uncertain part.

    A plain 'ending in ...' is useless when the correction happened earlier in
    the ID: the caller confirms the part they already said right. So start the
    read-back at the earliest corrected position (or, for ambiguous matches,
    where the candidates diverge), and never later than the last 4 chars.
    """
    start = len(chosen) - 4
    if heard:
        start = min(start, first_diff(heard, chosen))
    for o in others or []:
        start = min(start, next((i for i, (x, y) in enumerate(zip(chosen, o)) if x != y), start))
    start = max(start, len(PREFIX))
    return read_back(chosen, last_n=len(chosen) - start)


# --------------------------------------------------------------------------
# Postal codes (Canadian A1A 1A1)
# --------------------------------------------------------------------------
_POSTAL_RE = re.compile(r"^[A-Z]\d[A-Z]\d[A-Z]\d$")
_LETTER_FOR_DIGIT = {"0": "O", "1": "L", "5": "S", "8": "B", "2": "Z", "6": "G", "4": "A"}
_DIGIT_FOR_LETTER = {"O": "0", "L": "1", "I": "1", "S": "5", "B": "8", "Z": "2", "G": "6", "A": "4"}


def normalise_postal(raw: str) -> str | None:
    """Position-aware repair: postal codes alternate letter/digit, so a heard
    '0' in a letter slot is almost certainly 'O'."""
    s = re.sub(r"[^A-Z0-9]", "", parse_spoken(raw) if " " in raw.strip() else raw.upper())
    if len(s) != 6:
        return None
    fixed = []
    for i, c in enumerate(s):
        if i % 2 == 0 and c.isdigit():
            c = _LETTER_FOR_DIGIT.get(c, c)
        elif i % 2 == 1 and c.isalpha():
            c = _DIGIT_FOR_LETTER.get(c, c)
        fixed.append(c)
    out = "".join(fixed)
    return out if _POSTAL_RE.match(out) else None
