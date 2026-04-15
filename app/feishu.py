import time
import json
from typing import Optional

import requests


class FeishuClient:
    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self._token = None
        self._token_expire_at = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.app_id and self.app_secret)

    def _tenant_access_token(self) -> Optional[str]:
        if not self.enabled:
            return None
        now = time.time()
        if self._token and now < self._token_expire_at - 60:
            return self._token
        resp = requests.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("code") != 0:
            return None
        self._token = payload.get("tenant_access_token")
        self._token_expire_at = now + int(payload.get("expire", 7200))
        return self._token

    def send_text(self, chat_id: str, text: str) -> bool:
        token = self._tenant_access_token()
        if not token:
            return False
        resp = requests.post(
            "https://open.feishu.cn/open-apis/im/v1/messages",
            params={"receive_id_type": "chat_id"},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
            },
            timeout=15,
        )
        if resp.status_code >= 400:
            return False
        payload = resp.json()
        return payload.get("code") == 0
