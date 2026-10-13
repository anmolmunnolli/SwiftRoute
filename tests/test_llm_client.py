"""Exercises the HTTP client against a fake OpenAI-compatible server shaped
like Ollama's /v1/chat/completions response (arguments as a JSON string)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from swiftroute.llm import OpenAICompatLLM

SEEN = {}


class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN["body"] = body
        msg = {"role": "assistant", "content": ""}
        if body.get("tools"):
            msg["tool_calls"] = [{"id": "call_1", "type": "function", "function": {
                "name": "find_parcel", "arguments": json.dumps({"spoken_tracking_id": "S R 4"})}},
                {"id": "call_2", "type": "function", "function": {"name": "get_status", "arguments": "{bad json"}}]
        else:
            msg["content"] = "Hello there."
        out = json.dumps({"choices": [{"message": msg}],
                          "usage": {"prompt_tokens": 321, "completion_tokens": 12}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_text_reply(server):
    r = OpenAICompatLLM(model="m", base_url=server).chat([{"role": "user", "content": "hi"}])
    assert r.content == "Hello there." and r.prompt_tokens == 321 and r.latency_ms > 0
    assert SEEN["body"]["model"] == "m" and "tools" not in SEEN["body"]


def test_tool_calls_parsed_and_bad_json_contained(server):
    r = OpenAICompatLLM(model="m", base_url=server).chat([{"role": "user", "content": "x"}], tools=[{"t": 1}])
    assert r.tool_calls[0] == {"id": "call_1", "name": "find_parcel", "arguments": {"spoken_tracking_id": "S R 4"}}
    assert r.tool_calls[1]["arguments"] == {"__unparseable__": "{bad json"}


def test_unreachable_server_gives_actionable_error():
    with pytest.raises(RuntimeError, match="Is Ollama running"):
        OpenAICompatLLM(base_url="http://127.0.0.1:9/v1", timeout=2).chat([{"role": "user", "content": "x"}])
