"""Lessons and the learning log, as small JSON files under data/learning/."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
def default_dir() -> Path:
    return Path(os.environ.get("SWIFTROUTE_LEARNING_DIR", ROOT / "data" / "learning"))
MAX_PROMOTED = 8
MAX_CHARS = 200

# A lesson is advice on HOW to handle a kind of call. It can never loosen a rule.
_FORBIDDEN = re.compile(
    r"\b(ignore|override|bypass|skip|disable|forget)\b.*\b(rule|polic|verif|check|instruction|guard)"
    r"|\b(without|no need (for|to))\b.*\b(verif|postal|confirm)"
    r"|\badmin\b|\bsystem prompt\b|\bjailbreak\b|\bshare (the )?(address|phone)\b|\bfee|\bcharge", re.I)


def _path(name: str, base: Path | None = None) -> Path:
    d = base or default_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def validate_lesson(text: str) -> str | None:
    """None if acceptable, else why not."""
    t = (text or "").strip()
    if not t:
        return "empty"
    if "\n" in t:
        return "must be one line"
    if len(t) > MAX_CHARS:
        return f"longer than {MAX_CHARS} characters"
    if _FORBIDDEN.search(t):
        return "tries to loosen a rule or touches a forbidden topic"
    return None


def load(base: Path | None = None) -> dict:
    p = _path("lessons.json", base)
    if not p.exists():
        return {"lessons": []}
    return json.loads(p.read_text())


def save(data: dict, base: Path | None = None) -> None:
    _path("lessons.json", base).write_text(json.dumps(data, indent=2))


def promoted(base: Path | None = None) -> list[str]:
    return [x["text"] for x in load(base)["lessons"] if x["status"] == "promoted"][:MAX_PROMOTED]


def add(text: str, source: dict, base: Path | None = None) -> dict:
    data = load(base)
    item = {"id": uuid.uuid4().hex[:8], "text": text.strip(), "status": "candidate", "source": source,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    err = validate_lesson(text)
    if err:
        item.update(status="rejected", reason=f"validation: {err}")
    data["lessons"].append(item)
    save(data, base)
    log({"event": "lesson.proposed", **item}, base)
    return item


def update(lesson_id: str, base: Path | None = None, **fields) -> dict:
    data = load(base)
    for x in data["lessons"]:
        if x["id"] == lesson_id:
            x.update(fields)
            save(data, base)
            log({"event": f"lesson.{fields.get('status', 'updated')}", **x}, base)
            return x
    raise KeyError(lesson_id)


def get(lesson_id: str, base: Path | None = None) -> dict:
    for x in load(base)["lessons"]:
        if x["id"] == lesson_id:
            return x
    raise KeyError(lesson_id)


def log(entry: dict, base: Path | None = None) -> None:
    with _path("log.jsonl", base).open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}, default=str) + "\n")


def read_log(base: Path | None = None, limit: int = 200) -> list[dict]:
    p = _path("log.jsonl", base)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines()[-limit:]]


def snapshot(base: Path | None = None) -> dict:
    """Everything the inspector shows about learning."""
    conf = _path("confusions.json", base)
    return {"lessons": load(base)["lessons"],
            "confusions": json.loads(conf.read_text()) if conf.exists() else {"promoted": {}, "history": []},
            "log": [e for e in read_log(base, 400) if e.get("event") in
                    ("agent.start", "agent.end", "baseline", "confusions.update")][-40:]}
