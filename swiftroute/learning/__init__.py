"""How SwiftRoute improves between calls. Two loops, both gated by measurement:

  confusions.py  learns which characters speech recognition really confuses, from
                 read-backs that callers confirmed. Kept only if the ID benchmark
                 doesn't get worse and replayed real readings resolve as well or better.
  agent.py       an improvement agent (over MCP) that reads failures, drafts a short
                 "lesson" for the phone agent's prompt, and keeps it only if the eval
                 suite scores higher with no new safety failures.

Neither touches policy.py, the tools, or model weights. Distillation (train/) is the
third, slower loop: passing conversations become training data.
"""
