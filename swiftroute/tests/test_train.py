import json

from train.make_dataset import HELD_OUT, collect


def test_dataset_is_clean_and_leaves_held_out_types_unseen():
    examples, stats = collect("rule-agent", variants=1, seed=0, log=lambda *_: None)
    assert stats["kept"] == stats["conversations"] > 0 and examples
    blob = json.dumps(examples)
    assert "CORRECTION:" not in blob
    for e in examples:
        assert e["messages"][-1]["role"] == "assistant"
        for m in e["messages"]:
            for tc in m.get("tool_calls") or []:
                assert isinstance(tc["function"]["arguments"], dict)       # Qwen template wants objects
    # held-out scenario openings never appear in training
    from evals.harness import SCENARIOS
    held = [s.opening for s in SCENARIOS if s.name in HELD_OUT]
    assert not any(h in blob for h in held)


def test_training_parcels_differ_from_eval_parcels():
    from evals.harness import NOW
    from swiftroute.db import connect, seed
    a, b = connect(":memory:"), connect(":memory:")
    seed(a, now=NOW)                    # eval DB (seed 7)
    seed(b, now=NOW, seed_val=101)      # first training DB
    ids_a = {r[0] for r in a.execute("SELECT tracking_id FROM parcels")}
    ids_b = {r[0] for r in b.execute("SELECT tracking_id FROM parcels")}
    assert not ids_a & ids_b
