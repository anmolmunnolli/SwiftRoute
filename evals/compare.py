"""Compare models on the same evals: the phase-6 results table.

    python evals/compare.py \\
        --model "7B (Ollama)=qwen2.5:7b" \\
        --model "3B (Ollama)=qwen2.5:3b" \\
        --model "1.5B base (MLX)=default_model@http://localhost:8080/v1" \\
        --repeat 2

Spec: "label=model" (Ollama) or "label=model@base_url". Start each MLX model with
train/serve.sh before its row runs (one at a time: they share port 8080 unless you
set PORT). Rows are appended to evals/results/compare.md, so you can run one model
per invocation.

Columns separate the scenario types the student trained on from the HELD_OUT types
it never saw (see train/make_dataset.py), so "it memorised the evals" shows up
as a gap between the two.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals.harness import SCENARIOS, run_scenario  # noqa: E402
from evals.rule_agent import RuleAgentLLM  # noqa: E402
from swiftroute.llm import OpenAICompatLLM  # noqa: E402
from train.make_dataset import HELD_OUT  # noqa: E402

OUT = ROOT / "evals" / "results"


def parse_spec(spec: str):
    label, _, rest = spec.partition("=")
    model, _, url = rest.partition("@")
    if model == "rule-agent":
        return label, RuleAgentLLM
    return label, (lambda: OpenAICompatLLM(model=model, base_url=url or None))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def run_model(label: str, make, repeat: int, prompt: str) -> dict:
    if not isinstance(make, type):
        make().chat([{"role": "user", "content": "Say ok."}])          # warm-up / fail fast
    seen, unseen, ms, toks, claims, bad = [], [], [], [], 0, 0
    bad_kinds, ignored, fails = Counter(), 0, Counter()
    t0 = time.time()
    for sc in SCENARIOS:
        for _ in range(repeat):
            r = run_scenario(sc, make, prompt_mode=prompt)
            (unseen if sc.name in HELD_OUT else seen).append(r.passed)
            ms += r.llm_ms
            toks += [t for t in r.prompt_tokens if t]
            claims += r.false_claims + r.broken_promises
            bad += r.bad_tool_calls
            bad_kinds.update(r.bad_detail)
            ignored += r.ignored_args
            if not r.passed:
                fails.update(x.split(":")[0] for x in r.reasons)
            print(f"  {label:<22} {'PASS' if r.passed else 'FAIL'}  {sc.name}")
    return {
        "label": label, "prompt": prompt, "repeat": repeat,
        "pass_seen": round(sum(seen) / len(seen), 3), "pass_held_out": round(sum(unseen) / len(unseen), 3),
        "pass_all": round(sum(seen + unseen) / len(seen + unseen), 3),
        "llm_ms_p50": round(statistics.median(ms)) if ms else None, "llm_ms_p95": round(pct(ms, .95)) if ms else None,
        "prompt_tokens_mean": round(statistics.mean(toks)) if toks else None,
        "claims_or_broken_promises": claims, "bad_tool_calls": bad, "ignored_args": ignored,
        "top_bad_calls": bad_kinds.most_common(5), "top_fail_reasons": fails.most_common(5), "minutes": round((time.time() - t0) / 60, 1),
    }


def to_markdown(rows: list[dict]) -> str:
    head = ("| model | prompt | pass (all) | pass (trained types) | pass (held-out types) | "
            "LLM p50 ms | LLM p95 ms | prompt tok | false claims | bad tool calls |\n|" + "---|" * 10)
    lines = [f"| {r['label']} | {r['prompt']} | {r['pass_all']:.0%} | {r['pass_seen']:.0%} | {r['pass_held_out']:.0%} | "
             f"{r['llm_ms_p50']} | {r['llm_ms_p95']} | {r['prompt_tokens_mean']} | "
             f"{r['claims_or_broken_promises']} | {r['bad_tool_calls']} |" for r in rows]
    return head + "\n" + "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help='"label=model[@base_url]"')
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--prompt", default="full", choices=["compact", "full", "gated"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    store = OUT / "compare.json"
    rows = json.loads(store.read_text()) if store.exists() else []
    for spec in a.model:
        label, make = parse_spec(spec)
        try:
            row = run_model(label, make, a.repeat, a.prompt)
        except RuntimeError as e:
            print(f"skipping {label}: {e}")
            continue
        rows = [r for r in rows if not (r["label"] == label and r["prompt"] == a.prompt)] + [row]
        store.write_text(json.dumps(rows, indent=2))
    md = to_markdown(rows)
    for r in rows:   # diagnosis: what the bad calls and failures actually were
        if r.get("top_bad_calls") or r.get("top_fail_reasons"):
            md += (f"\n- **{r['label']} / {r['prompt']}**: bad calls {r.get('top_bad_calls')}; "
                   f"ignored extra args {r.get('ignored_args', 0)}; fail reasons {r.get('top_fail_reasons')}")
    md += "\n"
    (OUT / "compare.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
