"""Thin wrapper around the SGLang / OpenAI-compatible local API."""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from openai import OpenAI


def _make_client(base_url: str, api_key: str = "EMPTY") -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


_clients: dict[str, OpenAI] = {}


def _is_external(base_url: str) -> bool:
    return not base_url.startswith("http://localhost") and not base_url.startswith("http://127.")


def get_client(base_url: str = "http://localhost:8000/v1") -> OpenAI:
    if base_url not in _clients:
        if "openrouter.ai" in base_url:
            api_key = os.environ.get("OPENROUTER_API_KEY", "EMPTY")
        else:
            api_key = "EMPTY"
        _clients[base_url] = _make_client(base_url, api_key)
    return _clients[base_url]


def call_llm(
    system: str,
    user: str,
    model: str = "default",
    base_url: str = "http://localhost:8000/v1",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    response_format: dict | None = None,
    service_tier: str | None = None,
) -> str:
    client = get_client(base_url)
    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not _is_external(base_url):
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    elif service_tier is not None:
        kwargs["extra_body"] = {"service_tier": service_tier}
    if response_format is not None:
        kwargs["response_format"] = response_format
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    return text


def _sanitize_control(s: str) -> str:
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', s)


def parse_json_response(raw: str) -> dict:
    import ast
    cleaned = _strip_think(raw)

    def _try_one(s: str) -> dict | None:
        # 1. standard json
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # 2. sanitize control chars, then json
        s2 = _sanitize_control(s)
        try:
            return json.loads(s2)
        except json.JSONDecodeError:
            pass
        # 3. ast.literal_eval for single-quoted JSON
        try:
            result = ast.literal_eval(s)
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        return None

    # Try from every '{' position — handles gemma4's leading stray '{' or fences
    for m in re.finditer(r'\{', cleaned):
        candidate = cleaned[m.start():]
        result = _try_one(candidate)
        if result is not None:
            return result

    raise ValueError(f"No JSON found in response:\n{raw[:300]}")
