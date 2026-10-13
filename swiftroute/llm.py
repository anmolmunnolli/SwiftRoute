"""Minimal chat-completions client. No SDK, no extra dependencies.

Talks to any OpenAI-compatible endpoint, so the same code drives:
  * Ollama      http://localhost:11434/v1   (phase 2, laptop)
  * vLLM        http://localhost:8000/v1    (phase 6, GPU benchmarks)
  * llama.cpp   http://localhost:8080/v1
Swapping models is a flag, not a code change: that's the point of the cascade.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMReply:
    content: str
    tool_calls: list[dict] = field(default_factory=list)   # [{"id", "name", "arguments": dict}]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class LLM(Protocol):
    model: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply: ...


class OpenAICompatLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 temperature: float = 0.2, max_tokens: int = 200, timeout: float = 120):
        self.model = model or os.environ.get("SWIFTROUTE_MODEL", "qwen2.5:7b")
        self.base_url = (base_url or os.environ.get("SWIFTROUTE_LLM_URL", "http://localhost:11434/v1")).rstrip("/")
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        body: dict[str, Any] = {"model": self.model, "messages": messages,
                                "temperature": self.temperature, "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer local"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise RuntimeError(f"Model '{self.model}' isn't available at {self.base_url}. "
                                   f"Pull it first: ollama pull {self.model}") from e
            raise RuntimeError(f"Model server error {e.code}: {e.read()[:300]!r}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Can't reach the model at {self.base_url} ({e}). Is Ollama running? "
                f"Try: ollama serve  and  ollama pull {self.model}") from e
        latency = (time.perf_counter() - t0) * 1000
        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                args = {"__unparseable__": raw}
            calls.append({"id": tc.get("id") or f"call_{len(calls)}", "name": tc["function"]["name"],
                          "arguments": args if isinstance(args, dict) else {}})
        usage = data.get("usage") or {}
        return LLMReply(content=msg.get("content") or "", tool_calls=calls,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0), latency_ms=latency)


class ScriptedLLM:
    """Deterministic stand-in for tests: returns queued replies in order and
    records every request so tests can inspect the exact prompt sent."""
    model = "scripted"

    def __init__(self, replies: list[LLMReply]):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        self.requests.append(json.loads(json.dumps(messages)))
        if not self.replies:
            return LLMReply("Is there anything else I can help with?")
        return self.replies.pop(0)
