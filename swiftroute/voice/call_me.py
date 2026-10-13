"""Have the bot call a phone: the proactive "we missed you" call.

Telnyx (.env: TELNYX_API_KEY, TELNYX_APP_ID, TELNYX_FROM):
    python -m swiftroute.voice.call_me --to +12025550143 --host <ngrok-host>
Twilio  (.env: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM):
    python -m swiftroute.voice.call_me --provider twilio --to ... --host ...

The bot server must already be running with the matching -t telnyx / -t twilio.
No SDKs: one REST call each. Phase 7 triggers this from the simulator's
DELIVERY_EXCEPTION events instead of by hand.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from xml.sax.saxutils import quoteattr

from ..db import DEFAULT_DB, connect
from .bot import demo_parcel, register_outbound


class CallRejected(SystemExit):
    pass


def _request(url: str, body: bytes, headers: dict) -> dict:
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            err = json.loads(raw)
        except ValueError:
            err = {"message": raw[:300].decode(errors="replace")}
        raise CallRejected(f"HTTP {e.code}: {json.dumps(err)[:600]}") from None


# ----------------------------------------------------------------------- Telnyx
def call_telnyx(to: str, host: str) -> dict:
    key, app_id, from_ = os.environ["TELNYX_API_KEY"], os.environ["TELNYX_APP_ID"], os.environ["TELNYX_FROM"]
    body = json.dumps({"From": from_, "To": to, "Url": f"https://{host}/texml", "Method": "POST"}).encode()
    out = _request(f"https://api.telnyx.com/v2/texml/calls/{app_id}", body,
                   {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    return {"status": out.get("status"), "sid": out.get("call_sid"), "_via": "telnyx"}


# ----------------------------------------------------------------------- Twilio
def _twiml(host: str, customer: str, tracking_id: str | None, reason: str) -> str:
    params = {"from_number": customer, "tracking_id": tracking_id or "", "reason": reason}
    p = "".join(f'<Parameter name={quoteattr(k)} value={quoteattr(v)}/>' for k, v in params.items() if v)
    return f'<Response><Connect><Stream url={quoteattr(f"wss://{host}/ws")}>{p}</Stream></Connect></Response>'


def call_twilio(to: str, host: str, tracking_id: str | None, reason: str) -> dict:
    """Trial accounts reject some parameters; send only To/From/Url, then inline TwiML."""
    sid, token, from_ = os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"], os.environ["TWILIO_FROM"]
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json"
    auth = {"Authorization": "Basic " + base64.b64encode(f"{sid}:{token}".encode()).decode()}
    q = urllib.parse.urlencode({k: v for k, v in {"tracking_id": tracking_id, "reason": reason}.items() if v})
    last = None
    for label, params in [("webhook", {"To": to, "From": from_, "Url": f"https://{host}/twiml?{q}"}),
                          ("inline TwiML", {"To": to, "From": from_, "Twiml": _twiml(host, to, tracking_id, reason)})]:
        try:
            out = _request(url, urllib.parse.urlencode(params).encode(), auth)
            return {"status": out.get("status"), "sid": out.get("sid"), "_via": f"twilio {label}"}
        except CallRejected as e:
            print(f"  {label}: refused, {e}")
            last = e
    raise last


# ------------------------------------------------------------------------- main
def place_call(to: str, host: str, tracking_id: str | None, reason: str = "missed",
               provider: str | None = None) -> dict:
    provider = provider or ("telnyx" if os.environ.get("TELNYX_API_KEY") else "twilio")
    if tracking_id:
        register_outbound(to, tracking_id, reason)        # the bot picks this up when the call connects
    if provider == "telnyx":
        return call_telnyx(to, host)
    return call_twilio(to, host, tracking_id, reason)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True, help="phone to call, e.g. +12025550143")
    ap.add_argument("--host", required=True, help="ngrok host, e.g. example-tunnel.ngrok-free.dev")
    ap.add_argument("--provider", choices=["telnyx", "twilio"], default=None,
                    help="default: telnyx if TELNYX_API_KEY is set, else twilio")
    ap.add_argument("--tracking", default=None, help="parcel to call about (default: the demo missed delivery)")
    ap.add_argument("--reason", default="missed", choices=["missed", "update"])
    a = ap.parse_args()
    tid = a.tracking or demo_parcel(connect(DEFAULT_DB))["tracking_id"]
    try:
        call = place_call(a.to, a.host, tid, a.reason, a.provider)
    except KeyError as e:
        raise SystemExit(f"Missing {e.args[0]} in your environment (.env loaded? set -a; source .env; set +a)")
    except CallRejected as e:
        raise SystemExit(f"The provider rejected the call: {e}")
    print(f"calling {a.to} about {tid} via {call['_via']}: status={call.get('status')} id={call.get('sid')}")


if __name__ == "__main__":
    main()
