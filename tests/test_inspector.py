"""Tracing, the spectrogram tap maths, and the baked inspector page."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.harness import SCENARIOS, run_scenario  # noqa: E402
from evals.rule_agent import RuleAgentLLM  # noqa: E402
from swiftroute import trace  # noqa: E402


@pytest.fixture
def traced(tmp_path):
    path = trace.start_call("test", {"source": "test"}, path=tmp_path)
    yield path
    if trace.active():
        trace.end_call()


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_nothing_is_emitted_without_a_call(tmp_path):
    assert not trace.active()
    trace.emit("guard", kind="x")          # must be a silent no-op
    assert trace.HUB.backlog == [] or trace.HUB.call is None


def test_a_text_call_traces_every_decision(traced):
    sc = next(s for s in SCENARIOS if s.name == "misheard_id")
    r = run_scenario(sc, RuleAgentLLM)
    trace.end_call()
    assert r.passed
    kinds = {e["kind"] for e in _events(traced)}
    for k in ("call.start", "turn.start", "slot", "id.parse", "id.decode", "tool.call", "tool.result",
              "policy.verify", "llm.request", "llm.response", "fastpath", "turn.end", "state", "call.end"):
        assert k in kinds, k


def test_id_decode_event_shows_the_checksum_funnel(traced):
    sc = next(s for s in SCENARIOS if s.name == "misheard_id")
    run_scenario(sc, RuleAgentLLM)
    trace.end_call()
    dec = next(e for e in _events(traced) if e["kind"] == "id.decode")
    assert dec["generated"] > dec["checksum_valid"] >= dec["exist_in_db"] >= 1
    assert dec["status"] in ("corrected", "exact", "ambiguous")
    assert dec["ranked"] and all("diff" in r for r in dec["ranked"])


def test_guard_kind_does_not_clash_with_event_kind(traced):
    trace.emit("guard", kind="unbacked_claim", text="All set!")
    trace.end_call()
    g = next(e for e in _events(traced) if e["kind"] == "guard")
    assert g["guard"] == "unbacked_claim"


def test_prompt_is_captured_with_state_block_last(traced):
    sc = next(s for s in SCENARIOS if s.name == "reschedule_friday")
    run_scenario(sc, RuleAgentLLM)
    trace.end_call()
    req = next(e for e in _events(traced) if e["kind"] == "llm.request")
    assert req["messages"][0]["role"] == "system"
    assert req["messages"][-1]["content"].startswith("CALL STATE")


def test_log_mel_puts_a_tone_in_the_right_band():
    from swiftroute.voice.taps import LogMel
    m = LogMel(16000)
    t = np.arange(16000) / 16000
    cols, _ = m.push((np.sin(2 * np.pi * 1000 * t) * 8000).astype(np.int16).tobytes())
    assert cols.shape[1] == 80 and 95 <= len(cols) <= 100      # 10 ms hop
    peak_bin = int(cols.mean(0).argmax())
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    assert abs(peak_bin - mel(1000) / mel(8000) * 80) <= 2


def test_call_audio_cut_only_removes_that_lane(tmp_path):
    from swiftroute.voice.taps import CallAudio
    a = CallAudio()
    one_s = (np.ones(16000) * 1000).astype(np.int16).tobytes()
    a.add("caller", 0, one_s, 16000)
    a.add("agent", 500, one_s, 16000)
    a.cut("agent", 800)
    lanes = {ln: len(x) for ln, _, x in a.parts}
    assert lanes["caller"] == 16000 and lanes["agent"] == int(0.3 * 16000)
    a.save(tmp_path / "x.wav")
    assert (tmp_path / "x.wav").stat().st_size > 30000


def test_bake_embeds_trace_safely(tmp_path):
    from swiftroute.inspector.build import bake
    tr = tmp_path / "c.jsonl"
    tr.write_text(json.dumps({"t": 0, "seq": 1, "kind": "stt.final", "text": "</script><b>x"}) + "\n")
    page = bake(tr, fragment=True)
    assert "</script><b>" not in page                   # can't break out of the data block
    assert page.count('id="embedded-trace"') == 1
    full = bake(tr)
    assert full.startswith("<!doctype html>")
