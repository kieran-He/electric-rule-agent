import hashlib
import hmac
import time
from threading import Lock
from typing import Dict


def verify_token(payload: dict, expected_token: str) -> bool:
    if not expected_token:
        return True
    token = payload.get("token") or payload.get("header", {}).get("token")
    return token == expected_token


def verify_signature(timestamp: str, body: bytes, provided_signature: str, secret: str) -> bool:
    if not secret:
        return True
    if not timestamp or not provided_signature:
        return False
    try:
        now = int(time.time())
        ts = int(timestamp)
    except ValueError:
        return False
    if abs(now - ts) > 300:
        return False
    base = f"{timestamp}:{body.decode('utf-8')}".encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, provided_signature)


class EventDeduplicator:
    def __init__(self, ttl_seconds: int = 600) -> None:
        self.ttl_seconds = ttl_seconds
        self._seen: Dict[str, float] = {}
        self._lock = Lock()

    def seen(self, event_id: str) -> bool:
        now = time.time()
        with self._lock:
            expired = [k for k, ts in self._seen.items() if now - ts > self.ttl_seconds]
            for key in expired:
                self._seen.pop(key, None)
            if event_id in self._seen:
                return True
            self._seen[event_id] = now
        return False