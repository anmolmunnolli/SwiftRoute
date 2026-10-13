"""Phase 6b: build a distillation dataset from a teacher's successful calls.

    python train/make_dataset.py --teacher qwen2.5:7b --variants 8     # real data (slow: ~1 h on an Air)
    python train/make_dataset.py --teacher rule-agent --variants 4     # quick pipeline check

What goes in:
  * Every LLM call the teacher makes is recorded as one example: the exact prompt
    the agent built (system prompt, windowed history, CALL STATE) + the gated tool
    list + the teacher's reply. The student learns to produce that reply.
  * Only conversations that PASS the scenario checks, with no false claims and no
    broken promises, are used.
  * A reply that triggered a corrective nudge is dropped. The retry after the nudge
    is kept, with the CORRECTION line removed from its prompt, so the student learns
    to get it right first time.

No leakage into the evals:
  * training conversations use a different seed database (different parcels,
    tracking IDs, postal codes) and paraphrased caller openings;
  * HELD_OUT scenario types never appear in training at all, so the eval reports
    seen-type and unseen-type accuracy separately.

Output (mlx-lm "tools" chat format; tool-call arguments as JSON objects, which is
what Qwen's chat template expects): data/sft/train.jsonl, data/sft/valid.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals.harness import SCENARIOS, run_scenario  # noqa: E402
from evals.rule_agent import RuleAgentLLM  # noqa: E402
from swiftroute.llm import OpenAICompatLLM  # noqa: E402

HELD_OUT = {"prompt_injection", "second_parcel_claim", "withheld_caller_id"}

# Paraphrased openings, so the student doesn't just learn the eval's exact wording.
OPENINGS = {
    "status_happy": ["Hey, can you tell me where my package is?", "I'm checking on a delivery.",
                     "Hi there, what's the status of my parcel?"],
    "misheard_id": ["When will my parcel get here?", "I'd like an update on my delivery please."],
    "reschedule_friday": ["Nobody was home for my delivery, can you try again Friday?",
                          "I missed the courier. Could you bring it Friday?"],
    "reschedule_weekend_then_monday": ["I wasn't home, can it come Saturday instead?",
                                       "Missed delivery. Is Saturday possible?"],
    "address_change_locked": ["Please send my parcel to 12 King Street West, Toronto, M5V 2T6 instead.",
                              "Can I redirect my package to 12 King Street West, Toronto, M5V 2T6?"],
    "address_change_ok": ["Please send my parcel to 12 King Street West, Toronto, M5V 2T6 instead.",
                          "Can I redirect my package to 12 King Street West, Toronto, M5V 2T6?"],
    "wrong_postal_code": ["Where is my parcel?", "Can you check a delivery for me?"],
    "social_engineering": ["I need to know where my package is right now.", "Just tell me where my parcel is."],
    "damaged_claim": ["The box arrived smashed, everything inside is broken.", "My delivery came damaged."],
    "missing_too_early": ["My package hasn't shown up, I want to report it lost.",
                          "I think my parcel is missing."],
    "hold_at_depot": ["Could you keep my parcel at the depot? I'll collect it.",
                      "I'd rather pick it up myself from the depot."],
    "wants_human": ["I want to speak to a human.", "Put me through to a real agent please."],
}


def make_teacher(name: str):
    return RuleAgentLLM if name == "rule-agent" else (lambda: OpenAICompatLLM(model=name, max_tokens=160))


def to_example(messages: list[dict], tools: list[dict], reply) -> dict:
    msgs = [_clean(m) for m in messages]
    last = msgs[-1]
    if last["role"] == "system" and "CORRECTION:" in last["content"]:
        last["content"] = last["content"].split("\n\nCORRECTION:")[0]
    target = {"role": "assistant", "content": reply.content or ""}
    if reply.tool_calls:
        target["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                for c in reply.tool_calls]
    return {"messages": msgs + [target], "tools": tools}


def _clean(m: dict) -> dict:
    m = {k: v for k, v in m.items() if not k.startswith("_")}
    if m.get("tool_calls"):
        m["tool_calls"] = [{"type": "function", "function": {
            "name": tc["function"]["name"],
            "arguments": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str)
            else tc["function"]["arguments"]}} for tc in m["tool_calls"]]
    if m.get("content") is None:
        m["content"] = ""
    return m


def collect(teacher: str, variants: int, seed: int, log=print) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    make = make_teacher(teacher)
    examples, stats = [], {"conversations": 0, "kept": 0, "dropped_failed": 0, "calls": 0,
                           "dropped_nudged": 0, "dropped_fallback": 0}
    train_types = [s for s in SCENARIOS if s.name not in HELD_OUT]
    for v in range(variants):
        for sc in train_types:
            calls: list = []
            variant = dataclasses.replace(sc, opening=rng.choice(OPENINGS.get(sc.name, [sc.opening])))
            try:
                r = run_scenario(variant, make, recorder=lambda m, t, rep, turn: calls.append((m, t, rep)),
                                 seed_val=seed + 101 + v, pick=rng.randint(0, 2))
            except RuntimeError as e:          # no parcel of that kind in this seed: skip
                log(f"  skip {sc.name}: {e}")
                continue
            stats["conversations"] += 1
            stats["calls"] += len(calls)
            if not r.passed or r.false_claims or r.broken_promises:
                stats["dropped_failed"] += 1
                continue
            stats["kept"] += 1
            for i, (m, t, rep) in enumerate(calls):
                nxt = calls[i + 1][0] if i + 1 < len(calls) else None
                if nxt is not None and "CORRECTION:" in nxt[-1]["content"]:
                    stats["dropped_nudged"] += 1          # this reply needed fixing: don't teach it
                    continue
                if re.search(r"wasn't able to complete that", rep.content or ""):
                    stats["dropped_fallback"] += 1
                    continue
                examples.append(to_example(m, t, rep))
            log(f"  v{v} {sc.name:<32} PASS  {len(calls)} calls")
    # de-duplicate (the rule agent in particular repeats itself exactly)
    uniq = {hashlib.sha1(json.dumps(e, sort_keys=True).encode()).hexdigest(): e for e in examples}
    stats["examples"] = len(uniq)
    return list(uniq.values()), stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="qwen2.5:7b", help="Ollama model, or rule-agent")
    ap.add_argument("--variants", type=int, default=8, help="passes over the training scenario types")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "sft"))
    ap.add_argument("--valid-frac", type=float, default=0.1)
    a = ap.parse_args()
    t0 = time.time()
    examples, stats = collect(a.teacher, a.variants, a.seed)
    random.Random(a.seed).shuffle(examples)
    n_valid = max(1, int(len(examples) * a.valid_frac))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("valid", examples[:n_valid]), ("train", examples[n_valid:])):
        with (out / f"{name}.jsonl").open("w") as f:
            for e in rows:
                f.write(json.dumps(e) + "\n")
    stats.update(teacher=a.teacher, held_out=sorted(HELD_OUT), train=len(examples) - n_valid, valid=n_valid,
                 minutes=round((time.time() - t0) / 60, 1))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
