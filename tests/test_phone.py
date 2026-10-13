"""Phase 4: phone-line plumbing."""
import pytest

pytest.importorskip("pipecat")
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from swiftroute.db import connect, seed  # noqa: E402
from swiftroute.voice.bot import add_twiml_route, caller_phone_for, demo_parcel  # noqa: E402


def test_twiml_passes_caller_id_into_the_stream():
    app = fastapi.FastAPI()
    add_twiml_route(app)
    r = TestClient(app).post("/twiml", content="From=%2B18125551234&To=%2B13125550000",
                             headers={"host": "example-tunnel.ngrok-free.dev",
                                      "content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 200 and "xml" in r.headers["content-type"]
    assert 'url="wss://example-tunnel.ngrok-free.dev/ws"' in r.text
    assert '<Parameter name="from_number" value="+18125551234"/>' in r.text


def test_owner_phone_becomes_the_demo_customers_number(monkeypatch):
    monkeypatch.setenv("SWIFTROUTE_OWNER_PHONE", "+1 (812) 555-1234")
    conn = connect(":memory:")
    seed(conn)
    demo = demo_parcel(conn)
    assert demo["phone"] == "+18125551234"
    # every parcel of that customer now matches the caller's real number
    n = conn.execute("SELECT COUNT(*) FROM parcels p JOIN customers c ON c.id = p.customer_id "
                     "WHERE c.phone = '+18125551234'").fetchone()[0]
    assert n >= 3


def test_real_caller_id_wins_over_the_browser_default():
    class Args:
        class call_data:
            from_number = "+18125551234"
    assert caller_phone_for(Args(), {"phone": "+16135550000"}) == "+18125551234"


def test_outbound_twiml_identifies_the_customer_and_carries_the_parcel():
    app = fastapi.FastAPI()
    add_twiml_route(app)
    r = TestClient(app).post("/twiml?tracking_id=SR112M6YWXS&reason=missed",
                             content="From=%2B12025550188&To=%2B12025550143&Direction=outbound-api",
                             headers={"host": "h.ngrok-free.dev",
                                      "content-type": "application/x-www-form-urlencoded"})
    assert '<Parameter name="from_number" value="+12025550143"/>' in r.text      # the person we called
    assert '<Parameter name="tracking_id" value="SR112M6YWXS"/>' in r.text


def test_outbound_greeting_preselects_parcel_but_still_needs_postal_code():
    from swiftroute.agent import Agent
    from swiftroute.llm import ScriptedLLM
    from swiftroute.tools import CallSession
    from swiftroute.voice.bot import outbound_greeting
    conn = connect(":memory:")
    seed(conn)
    demo = demo_parcel(conn)

    class Args:
        class call_data:
            body = {"tracking_id": demo["tracking_id"], "reason": "missed"}
    s = CallSession(conn, caller_phone=demo["phone"])
    g = outbound_greeting(conn, s, Args())
    assert "couldn't" in g and g.endswith("postal code?") and s.tracking_id == demo["tracking_id"]
    assert s.get_status()["reason_code"] == "NEEDS_VERIFICATION"          # nothing shared yet
    agent = Agent(ScriptedLLM([]), s)
    agent.history.append({"role": "assistant", "content": g})
    agent.respond(" ".join(demo["postal"]))                                # handled as a slot, no LLM
    assert s.identity.data["level"] == "full"


def test_twiml_also_answers_get_from_the_console():
    app = fastapi.FastAPI()
    add_twiml_route(app)
    r = TestClient(app).get("/twiml?tracking_id=SR112M6YWXS&reason=missed&To=%2B12025550143&Direction=outbound-api",
                            headers={"host": "h.ngrok-free.dev"})
    assert r.status_code == 200 and '<Parameter name="from_number" value="+12025550143"/>' in r.text


def test_query_params_never_override_twilios_values_and_plus_is_restored():
    app = fastapi.FastAPI()
    add_twiml_route(app)
    # the console re-encoded "+" as a space in the query string; Twilio's form value must win
    r = TestClient(app).post("/twiml?To=%2012025550143&Direction=outbound-api",
                             content="From=%2B12025550188&To=%2B12025550143&Direction=outbound-api",
                             headers={"host": "h", "content-type": "application/x-www-form-urlencoded"})
    assert '<Parameter name="from_number" value="+12025550143"/>' in r.text
    r = TestClient(app).get("/twiml?To=%2012025550143&Direction=outbound-api", headers={"host": "h"})
    assert '<Parameter name="from_number" value="+12025550143"/>' in r.text


# ----------------------------------------------------------------- Telnyx
def test_texml_route_streams_bidirectional_rtp():
    from swiftroute.voice.bot import add_texml_route
    app = fastapi.FastAPI()
    add_texml_route(app)
    r = TestClient(app).post("/texml", headers={"host": "example-tunnel.ngrok-free.dev"})
    assert r.status_code == 200
    assert '<Stream url="wss://example-tunnel.ngrok-free.dev/ws" bidirectionalMode="rtp">' in r.text


def test_outbound_caller_id_is_the_number_we_called(monkeypatch):
    monkeypatch.setenv("TELNYX_FROM", "+1 202 555 0188")

    class Args:
        class call_data:
            from_number, to_number = "+12025550188", "+12025550143"     # Telnyx outbound: from = us
    assert caller_phone_for(Args(), {"phone": "x"}) == "+12025550143"

    class Inbound:
        class call_data:
            from_number, to_number = "+12025550143", "+12025550188"
    assert caller_phone_for(Inbound(), {"phone": "x"}) == "+12025550143"


def test_pending_outbound_hands_the_parcel_to_the_bot(monkeypatch, tmp_path):
    import swiftroute.voice.bot as bot
    from swiftroute.tools import CallSession
    monkeypatch.setattr(bot, "PENDING", tmp_path / "pending.json")
    conn = connect(":memory:")
    seed(conn)
    demo = demo_parcel(conn)
    bot.register_outbound("+1 (202) 555-0143", demo["tracking_id"], "missed")

    class Args:
        call_data = None
    s = CallSession(conn, caller_phone="+12025550143")
    g = bot.outbound_greeting(conn, s, Args())
    assert g and "couldn't" in g and s.tracking_id == demo["tracking_id"]
    assert bot.outbound_greeting(conn, CallSession(conn, caller_phone="+12025550143"), Args()) is None  # used once


def test_call_me_telnyx_request_shape(monkeypatch, tmp_path):
    import json
    import swiftroute.voice.bot as bot
    import swiftroute.voice.call_me as cm
    monkeypatch.setattr(bot, "PENDING", tmp_path / "pending.json")
    monkeypatch.setenv("TELNYX_API_KEY", "KEY123")
    monkeypatch.setenv("TELNYX_APP_ID", "999")
    monkeypatch.setenv("TELNYX_FROM", "+17375550000")
    seen = {}

    def fake(url, body, headers):
        seen.update(url=url, body=json.loads(body), headers=headers)
        return {"call_sid": "v3:abc", "status": "queued"}
    monkeypatch.setattr(cm, "_request", fake)
    out = cm.place_call("+12025550143", "h.ngrok-free.dev", "SR112M6YWXS")
    assert seen["url"] == "https://api.telnyx.com/v2/texml/calls/999"
    assert seen["body"] == {"From": "+17375550000", "To": "+12025550143",
                            "Url": "https://h.ngrok-free.dev/texml", "Method": "POST"}
    assert seen["headers"]["Authorization"] == "Bearer KEY123"
    assert out["_via"] == "telnyx" and out["status"] == "queued"
    assert json.loads((tmp_path / "pending.json").read_text())["+12025550143"]["tracking_id"] == "SR112M6YWXS"
