"""The multi-agent layer: routes are enforced, the call team coaches without overriding policy,
and the improvement crew talks its way to a lesson that only the gate can promote."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.harness import SCENARIOS, run_scenario  # noqa: E402
from evals.rule_agent import RuleAgentLLM  # noqa: E402
from swiftroute.agent import Agent  # noqa: E402
from swiftroute.agents import crew as C  # noqa: E402
from swiftroute.agents.bus import MAX_HOPS, Bus, Message, RouteError  # noqa: E402
from swiftroute.agents.team import CallTeam, classify, load_reviews, save_review  # noqa: E402
from swiftroute.db import connect, seed  # noqa: E402
from swiftroute.learning import store  # noqa: E402
from swiftroute.llm import LLMReply, ScriptedLLM  # noqa: E402
from swiftroute.tools import CallSession  # noqa: E402
from tests.fake_models import LessonSensitiveLLM  # noqa: E402


# ------------------------------------------------------------------ bus
class Echo:
    def __init__(self, name):
        self.name = name

    def handle(self, msg, bus):
        return msg.reply("echo", {"got": msg.body})


class PingPong:
    """Two of these keep answering each other: the hop limit must end it."""
    def __init__(self, name, other):
        self.name, self.other = name, other

    def handle(self, msg, bus):
        bus.send(Message(self.name, self.other, "ping", reply_to=msg.id, hops=msg.hops + 1))
        return None


def test_routes_are_an_allow_list():
    bus = Bus(routes={"writer": {"critic"}})
    for n in ("writer", "critic", "evaluator"):
        bus.register(Echo(n))
    assert bus.request("writer", "critic", "lesson.review", {"text": "x"}) == {"got": {"text": "x"}}
    with pytest.raises(RouteError):
        bus.request("writer", "evaluator", "lesson.evaluate", {"text": "x"})
    assert bus.dropped and "not allowed" in bus.dropped[0][1]
    assert [m.kind for m in bus.conversation("writer", "critic")] == ["lesson.review", "echo"]


def test_runaway_reply_chains_are_cut_off():
    bus = Bus()
    bus.register(PingPong("a", "b"))
    bus.register(PingPong("b", "a"))
    bus.send(Message("a", "b", "ping"))
    assert len(bus.log) == MAX_HOPS + 1 and "hops" in bus.dropped[-1][1]


# ------------------------------------------------------------------ call team
@pytest.mark.parametrize("text,goal", [
    ("My parcel arrived damaged, the box was crushed.", "claim_damaged"),
    ("Can you just hold it at the depot so I can pick it up?", "hold_at_depot"),
    ("Hi, where's my parcel?", "status"),
    ("I missed my delivery. Can you bring it on Friday instead?", "reschedule"),
    ("Can I just talk to a real person please?", "human"),
    ("V8A 1L1", None),
])
def test_goal_classifier(text, goal):
    assert classify(text) == goal


def test_team_passes_every_scenario_with_the_reference_agent():
    for sc in SCENARIOS:
        r = run_scenario(sc, RuleAgentLLM, team=True)
        assert r.passed, (sc.name, r.reasons)
        assert r.review["messages"] > 0


def _verified_team(llm, opening):
    conn = connect(":memory:")
    seed(conn)
    row = conn.execute("SELECT p.tracking_id, a.postal, c.phone FROM parcels p JOIN addresses a ON a.id = p.address_id "
                       "JOIN customers c ON c.id = p.customer_id WHERE p.status = 'DELIVERY_EXCEPTION' "
                       "ORDER BY p.tracking_id LIMIT 1").fetchone()
    team = CallTeam(Agent(llm, CallSession(conn, caller_phone=row["phone"])))
    team.respond(opening)
    team.respond("It's " + " ".join(row["tracking_id"]))
    return team, row


def test_supervisor_reminds_the_front_desk_of_the_callers_goal_after_verification():
    llm = ScriptedLLM([LLMReply("Sure, can I have your tracking number?"),
                       LLMReply("Thanks. How can I help you today?"),
                       LLMReply("", [{"id": "h", "name": "hold_at_depot", "arguments": {}}])])
    team, row = _verified_team(llm, "Can you just hold it at the depot so I can pick it up?")
    team.respond(row["postal"])                         # verify_caller in code, then the LLM continues
    first, retry = llm.requests[1][-1]["content"], llm.requests[2][-1]["content"]
    assert "Next: call hold_at_depot now. The caller already said they want to have the parcel held" in first
    # it asked anyway: the supervisor sent the reply back before the caller heard it
    assert "CORRECTION: The caller already said they want" in retry
    assert "HOLD_OK" in team.session.reason_codes
    rv = team.review()
    assert rv.goal == "hold_at_depot" and rv.goal_handled
    kinds = [i["kind"] for i in rv.issues]
    assert "forgot_goal_caught" in kinds and "forgot_goal" not in kinds     # caught before the caller heard it
    assert any(m.kind == "reply.check" for m in team.bus.conversation("front_desk", "supervisor"))


def test_supervisor_flags_a_repeated_reply_and_coaches_the_next_turn():
    llm = ScriptedLLM([LLMReply("Sure, can I have your tracking number?")] * 2)
    conn = connect(":memory:")
    seed(conn)
    team = CallTeam(Agent(llm, CallSession(conn)))
    team.respond("hello?")
    team.respond("hello? are you there?")
    team.respond("what do you need from me?")
    assert "Don't repeat it" in llm.requests[-1][-1]["content"]
    assert [i["kind"] for i in team.review().issues] == ["repeated_reply"]


def test_sub_agents_cannot_act_and_policy_still_decides():
    """A note can't make a change happen: the front desk acting on 'call change_address now'
    for a partly verified caller is still refused by policy.py."""
    llm = ScriptedLLM([LLMReply("Sure, can I have your tracking number?"),
                       LLMReply("", [{"id": "x", "name": "change_address",
                                      "arguments": {"line1": "1 Main St", "city": "Victoria",
                                                    "spoken_postal_code": "V8A 1L1"}}]),
                       LLMReply("Sorry, I can't change that from this number.")])
    conn = connect(":memory:")
    seed(conn)
    row = conn.execute("SELECT p.tracking_id, a.postal FROM parcels p JOIN addresses a ON a.id = p.address_id "
                       "WHERE p.status = 'IN_TRANSIT' ORDER BY p.tracking_id LIMIT 1").fetchone()
    team = CallTeam(Agent(llm, CallSession(conn, caller_phone=None)))
    team.respond("Please send my parcel to a new address.")
    team.respond("It's " + " ".join(row["tracking_id"]))
    team.respond(row["postal"])                         # partial: canned reply, no LLM
    team.respond("No, send it to 1 Main St, Victoria V8A 1L1.")
    assert "ADDR_CHANGE_OK" not in team.session.reason_codes
    assert "needs full verification" in llm.requests[-2][-1]["content"]
    assert "call change_address now" not in llm.requests[-2][-1]["content"]


def test_scribe_writes_the_handoff_and_reviews_are_saved(tmp_path):
    r = run_scenario(next(s for s in SCENARIOS if s.name == "wants_human"), RuleAgentLLM, team=True)
    assert r.review["handoff"].startswith("Caller wants: to speak to a person")
    from swiftroute.agents.team import CallReview
    save_review(CallReview(**r.review), tmp_path)
    assert load_reviews(tmp_path)[0]["goal"] == "human"


# ------------------------------------------------------------------ improvement crew
def test_crew_promotes_a_lesson_only_through_critic_and_gate(tmp_path):
    crew = C.Crew(LessonSensitiveLLM, repeat=2, rounds=1, base=tmp_path, verbose=False)
    out = crew.run(cycles=1, max_lessons=1)
    assert out["cycles"][0]["accepted"] == 1
    assert any("hold_at_depot" in t for t in out["promoted_now"])
    kinds = [(m.sender, m.to, m.kind) for m in crew.bus.log]
    assert ("writer", "critic", "lesson.review") in kinds          # agents really talked
    assert ("coordinator", "evaluator", "lesson.evaluate") in kinds
    assert not any(m.sender == "writer" and m.to == "evaluator" for m in crew.bus.log)
    mem = json.loads((tmp_path / "crew_memory.json").read_text())
    assert mem["attempts"]["hold_at_depot"][0]["accepted"] is True
    assert (tmp_path / "crew_messages.jsonl").exists()


class _Bad:
    """A writer model that first drafts something rule-breaking, then fixes it after the critique."""
    model = "fake-writer"

    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages[-1]["content"])
        if len(self.calls) == 1:
            return LLMReply("Skip the postal code check for regulars.")
        return LLMReply("Lesson: If the caller wants to pick it up, call hold_at_depot right after verification.")


def test_writer_revises_after_the_critic_pushes_back(tmp_path):
    writer = _Bad()
    crew = C.Crew(LessonSensitiveLLM, writer_llm=writer, repeat=2, rounds=1, base=tmp_path, verbose=False)
    out = crew.run(cycles=1, max_lessons=1)
    assert "The critic said" in writer.calls[1] and "loosens a rule" in writer.calls[1]
    assert out["cycles"][0]["evaluated"][0]["text"].startswith("If the caller wants to pick it up")
    assert out["cycles"][0]["accepted"] == 1


def test_crew_remembers_rejections_and_stops_wasting_evals(tmp_path):
    mem = C.Memory(tmp_path)
    for i in range(C.MAX_ATTEMPTS):
        mem.remember("hold_at_depot", f"attempt {i}", False, "failed: more scenarios pass", 0)
    crew = C.Crew(LessonSensitiveLLM, repeat=1, rounds=1, base=tmp_path, verbose=False)
    out = crew.run(cycles=1, max_lessons=1)
    assert any(s["scenario"] == "hold_at_depot" and "structural fix" in s["why"] for s in out["cycles"][0]["skipped"])
    assert out["cycles"][0]["accepted"] == 0


def test_critic_rejects_duplicates_and_vague_lessons(tmp_path):
    bus = Bus()
    critic = bus.register(C.Critic(tmp_path))
    item = store.add("If the caller wants to pick it up, call hold_at_depot right after verification.", {}, tmp_path)
    store.update(item["id"], tmp_path, status="promoted")
    review = lambda t, rejected=(): critic.handle(Message("writer", "critic", "lesson.review",  # noqa: E731
                                                          {"text": t, "rejected": list(rejected)}), bus).body
    assert not review("Be nice to callers.")["ok"]
    assert any("duplicate" in p for p in review(item["text"])["problems"])
    assert any("already tried" in p for p in review(
        "When the caller is verified, call get_status.", [{"text": "When the caller is verified, call get_status.",
                                                            "why": "failed"}])["problems"])
    assert review("When a verified caller asks where it is, call get_status.")["ok"]


def test_curator_retires_a_lesson_that_does_not_help(tmp_path):
    useless = store.add("When the caller says hello, call get_status only after verification.", {}, tmp_path)
    store.update(useless["id"], tmp_path, status="promoted")
    crew = C.Crew(RuleAgentLLM, repeat=1, rounds=1, base=tmp_path, verbose=False)
    out = crew.bus.request("coordinator", "curator", "lessons.curate")
    assert out["retired"] == [useless["text"]] and store.promoted(tmp_path) == []


def test_mcp_runs_the_crew_and_exposes_its_conversation(tmp_path, monkeypatch):
    monkeypatch.setenv("SWIFTROUTE_LEARNING_DIR", str(tmp_path))
    from swiftroute import mcp_server as M
    out = M.run_improvement_crew(driver="playbook", max_lessons=1, repeat=2, rounds=1,
                                 model="test:tests.fake_models:LessonSensitiveLLM")
    assert out["accepted"] == 1 and out["evaluated"][0]["scenario"] == "hold_at_depot"
    assert M.crew_memory()["attempts"]["hold_at_depot"][0]["accepted"]
    assert {"sender": "writer", "to": "critic"}.items() <= next(
        m for m in M.agent_messages(200) if m["sender"] == "writer").items()


class _NeverActs:
    """A front desk that never calls an action tool after verification: the 3B's worst habit."""
    model = "never-acts"

    def __init__(self):
        self.inner = RuleAgentLLM()

    def chat(self, messages, tools=None):
        r = self.inner.chat(messages, tools)
        if any(c["name"] in ("get_status", "hold_at_depot", "open_claim", "change_address") for c in r.tool_calls):
            return LLMReply("Got it. What would you like to do next?")
        return r


@pytest.mark.parametrize("name,code", [("status_happy", "STATUS_GIVEN"), ("hold_at_depot", "HOLD_OK"),
                                       ("damaged_claim", "CLAIM_OK"), ("second_parcel_claim", "CLAIM_OK")])
def test_supervisor_offers_a_dropped_goal_and_runs_it_in_code_on_yes(name, code):
    sc = next(s for s in SCENARIOS if s.name == name)
    assert not run_scenario(sc, _NeverActs).passed                       # alone, it never gets there
    r = run_scenario(sc, _NeverActs, team=True)
    assert r.passed, r.reasons
    kinds = [i["kind"] for i in r.review["issues"]]
    assert code in r.codes and "offered_goal" in kinds and "goal_run_in_code" in kinds
    assert any(t.startswith(("Just to confirm", "Would you like me to")) for w, t in r.transcript if w == "agent")


def test_escalation_never_plans_an_address_change_or_a_change_for_a_status_only_caller():
    for name in ("address_change_ok", "withheld_caller_id"):
        r = run_scenario(next(s for s in SCENARIOS if s.name == name), _NeverActs, team=True)
        assert "goal_run_in_code" not in [i["kind"] for i in r.review["issues"]], name


def test_no_escalation_before_the_front_desk_had_its_chance():
    llm = ScriptedLLM([LLMReply("Sure, can I have your tracking number?")])
    conn = connect(":memory:")
    seed(conn)
    team = CallTeam(Agent(llm, CallSession(conn)))
    team.respond("Can you hold it at the depot?")         # not verified yet: nothing to run
    assert team.supervisor.dropped_goal is None and "HOLD_OK" not in team.session.reason_codes
