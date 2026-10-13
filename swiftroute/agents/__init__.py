"""SwiftRoute's multi-agent layer.

    bus.py    typed messages between agents, with an allow-list of who may talk to whom
    team.py   during a call: the phone agent (front desk) + goal tracker, supervisor, scribe
    crew.py   between calls: coordinator, analyst, writer <-> critic, evaluator, curator,
              improving the phone agent and remembering what worked

Safety is unchanged: only tools change anything, and policy.py decides every tool call.
Agents advise each other; the eval gate (learning/lessons.py) decides what's kept.
"""
