"""Run the scripted-call evals.

    python evals/run_text.py                          # qwen2.5:7b on local Ollama
    python evals/run_text.py --model qwen2.5:3b       # compare a smaller model
    python evals/run_text.py --model rule-agent       # no server needed (harness smoke test)
    python evals/run_text.py --history all            # context ablation: keep every turn
    python evals/run_text.py --repeat 3 --show-failures
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evals.harness import SCENARIOS, run_scenario, summarise  # noqa: E402
from evals.rule_agent import RuleAgentLLM  # noqa: E402
from swiftroute.llm import OpenAICompatLLM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--url", default=None)
    ap.add_argument("--history", default="6", help="turns of history to keep, or 'all'")
    ap.add_argument("--repeat", type=int, default=1, help="runs per scenario (temperature 0.2 varies)")
    ap.add_argument("--only", default=None, help="comma-separated scenario names")
    ap.add_argument("--show-failures", action="store_true")
    ap.add_argument("--prompt", default="full", choices=["compact", "full", "gated"],
                    help="compact = short prompt, all tools (stable prefix); gated = only state-valid tools; full = original")
    ap.add_argument("--team", action="store_true",
                    help="run the call team: front desk + goal tracker, supervisor, scribe (swiftroute/agents)")
    a = ap.parse_args()

    hist = None if a.history == "all" else int(a.history)
    make = (RuleAgentLLM if a.model == "rule-agent"
            else lambda: OpenAICompatLLM(model=a.model, base_url=a.url))
    scs = [s for s in SCENARIOS if not a.only or s.name in a.only.split(",")]
    if a.model != "rule-agent":
        try:   # fail fast, and warm the model so the first scenario's latency isn't a cold load
            make().chat([{"role": "user", "content": "Say ok."}])
        except RuntimeError as e:
            raise SystemExit(str(e))

    results = []
    t0 = time.time()
    for sc in scs:
        for _ in range(a.repeat):
            r = run_scenario(sc, make, history_turns=hist, prompt_mode=a.prompt, team=a.team)
            results.append(r)
            mark = "PASS" if r.passed else "FAIL"
            p50 = sorted(r.llm_ms)[len(r.llm_ms) // 2] if r.llm_ms else 0
            print(f"{mark}  {sc.name:<32} turns={r.turns:<2} tools={r.tool_calls:<2} "
                  f"llm_p50={p50:>6.0f}ms  {'; '.join(r.reasons)}")
            if a.show_failures and not r.passed:
                for who, text in r.transcript:
                    print(f"        {who:>6}: {text}")
    s = summarise(results)
    s.update(team=a.team, model=a.model, history=a.history, prompt=a.prompt, repeat=a.repeat, wall_s=round(time.time() - t0, 1))
    print("\n" + json.dumps(s, indent=2))

    out = Path(__file__).parent / "results" / f"text_{a.model.replace(':', '-').replace('/', '_')}_h{a.history}_{a.prompt}{'_team' if a.team else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": s, "runs": [
        {k: v for k, v in asdict(r).items() if k != "llm_ms"} for r in results]}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
