"""Turn what the caller said ("Friday", "Monday the 19th", "tomorrow") into a date.

Same principle as tracking IDs: don't let the LLM do deterministic work.
In eval run 2, qwen2.5:7b heard "Saturday", booked Wednesday the 21st and told
the caller "rescheduled for Saturday, October 21". The model now passes the
caller's words through, this parses them, and the policy engine judges the result.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]


def parse_day(text: str, today: date) -> date | None:
    t = text.lower().strip()
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
    if "day after tomorrow" in t:
        return today + timedelta(days=2)
    if "tomorrow" in t:
        return today + timedelta(days=1)
    if re.search(r"\btoday\b", t):
        return today
    # "October 19", "Oct 19th", "19 October"
    for i, name in enumerate(_MONTHS):
        pat = rf"\b{name[:3]}[a-z]*\.?\s+(\d{{1,2}})(st|nd|rd|th)?\b|\b(\d{{1,2}})(st|nd|rd|th)?\s+(of\s+)?{name[:3]}[a-z]*\b"
        m = re.search(pat, t)
        if m:
            d = int(m[1] or m[3])
            try:
                cand = date(today.year, i + 1, d)
            except ValueError:
                return None
            return cand if cand >= today else date(today.year + 1, i + 1, d)
    day_name = next((d for d in _DAYS if re.search(rf"\b{d}\b", t)), None)
    m = re.search(r"\bthe (\d{1,2})(st|nd|rd|th)\b", t)
    if day_name:
        delta = (_DAYS.index(day_name) - today.weekday()) % 7
        delta = delta or 7                             # "Wednesday" said on a Wednesday = next week
        if "next week" in t or "week after" in t:      # plain "next Friday" = the coming one
            delta += 7
        cand = today + timedelta(days=delta)
        if m and int(m[1]) != cand.day:                # "Monday the 26th": trust the number
            for k in range(1, 5):
                alt = cand + timedelta(days=7 * k)
                if alt.day == int(m[1]):
                    return alt
        return cand
    if m:                                              # "the 19th"
        d = int(m[1])
        for k in range(0, 62):
            cand = today + timedelta(days=k)
            if cand.day == d:
                return cand
    return None


def say_date(d: date) -> str:
    return f"{d:%A}, {d:%B} {d.day}"
