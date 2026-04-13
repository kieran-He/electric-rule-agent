import hashlib
import hmac
import time

from app.security import EventDeduplicator, verify_signature, verify_token


def test_verify_token():
    payload = {"header": {"token": "abc"}}
    assert verify_token(payload, "abc")
    assert not verify_token(payload, "wrong")


def test_verify_signature():
    ts = str(int(time.time()))
    body = b'{"hello":"world"}'
    secret = "mysecret"
    sign = hmac.new(secret.encode("utf-8"), f"{ts}:{body.decode()}".encode("utf-8"), hashlib.sha256).hexdigest()
    assert verify_signature(ts, body, sign, secret)


def test_event_deduplicator():
    dedup = EventDeduplicator(ttl_seconds=60)
    assert dedup.seen("evt-1") is False
    assert dedup.seen("evt-1") is True

