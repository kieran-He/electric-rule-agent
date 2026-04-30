from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from dataprocess.config import DocProcSettings


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout_sec: int


def build_llm_config(settings: DocProcSettings) -> LLMConfig:
    return LLMConfig(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout_sec=settings.llm_timeout_sec,
    )


def call_llm_json(*, cfg: LLMConfig, system_prompt: str, user_prompt: str, max_retries: int = 2) -> dict[str, Any]:
    normalized_base = cfg.base_url.rstrip("/")
    if normalized_base.endswith("/v1"):
        url = normalized_base + "/chat/completions"
    else:
        url = normalized_base + "/v1/chat/completions"

    payload = {
        "model": cfg.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=cfg.timeout_sec) as response:
                raw = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            if attempt < max_retries:
                print(f"[LLM] HTTPError {exc.code}, retrying... ({attempt + 1}/{max_retries})")
                continue
            raise RuntimeError(f"LLM HTTPError {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            if attempt < max_retries:
                print(f"[LLM] URLError {exc}, retrying... ({attempt + 1}/{max_retries})")
                continue
            raise RuntimeError(f"LLM URLError: {exc}") from exc
        except Exception as exc:
            if attempt < max_retries:
                print(f"[LLM] {type(exc).__name__}: {exc}, retrying... ({attempt + 1}/{max_retries})")
                continue
            raise RuntimeError(f"LLM error: {exc}") from exc

    parsed = json.loads(raw)
    if "choices" not in parsed:
        raise RuntimeError(f"LLM response missing choices: {str(parsed)[:500]}")

    message = parsed["choices"][0].get("message", {})
    content_obj = message.get("content")

    if isinstance(content_obj, str):
        content = content_obj
    elif isinstance(content_obj, list):
        text_parts: list[str] = []
        for item in content_obj:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
        content = "\n".join(part for part in text_parts if part)
    elif isinstance(parsed["choices"][0].get("text"), str):
        content = parsed["choices"][0]["text"]
    else:
        raise RuntimeError(f"LLM message content missing/unsupported: {str(parsed)[:500]}")

    return _extract_json_object(content, cfg)


def _extract_json_object(text: str, cfg: LLMConfig | None = None) -> dict[str, Any]:
    stripped = text.strip()
    
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if lines[0].startswith("```json"):
            lines = lines[1:]
        elif lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    
    def try_parse_json(s: str) -> dict[str, Any] | None:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None
    
    result = try_parse_json(stripped)
    if result is not None:
        return result
    
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        json_str = stripped[start : end + 1]
        result = try_parse_json(json_str)
        if result is not None:
            return result
        
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r'[\x00-\x1f]', ' ', json_str)
        
        result = try_parse_json(json_str)
        if result is not None:
            return result
    
    if cfg:
        print(f"[LLM Debug] Model: {cfg.model}, URL: {cfg.base_url}")
        print(f"[LLM Debug] Response length: {len(stripped)}")
        print(f"[LLM Debug] Response preview: {stripped[:300]}...")
    
    raise json.JSONDecodeError(
        f"Failed to parse JSON from response (len={len(stripped)}): {stripped[:200] if stripped else '<empty>'}",
        stripped,
        0
    )