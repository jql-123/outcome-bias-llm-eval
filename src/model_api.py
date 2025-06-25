# ==================== NEW IMPLEMENTATION FOR LIGHTWEIGHT MODELS ====================
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import dotenv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {}


def _load_config():
    import yaml

    global CONFIG
    if CONFIG:
        return CONFIG
    with open(ROOT / "config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
    return CONFIG


dotenv.load_dotenv()

# -----------------------------------------------------------------------------
# Helper functions for each provider
# -----------------------------------------------------------------------------


def _openai_call(messages: List[dict], n: int, cfg) -> List[str]:
    import openai
    # Convert messages to OpenAI ChatCompletionMessageParam format
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            chat_messages.append({"role": "system", "content": m["content"]})
        elif m["role"] == "user":
            chat_messages.append({"role": "user", "content": m["content"]})
        else:
            raise ValueError(f"Unknown message role: {m['role']}")
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=cfg["id"],
        messages=chat_messages,
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 0.95),
        max_tokens=cfg.get("max_tokens", 600),
        n=n,
    )
    return [choice.message.content.strip() if choice.message and choice.message.content else "" for choice in response.choices]


def _anthropic_call(messages: List[dict], n: int, cfg) -> List[str]:
    """Call Anthropic Claude-3-Haiku using new Messages API."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Separate system prompt from the message list
    system_prompt = cfg.get("system_prompt")
    user_messages: List[dict] = []  # type: ignore[var-annotated]
    for m in messages:
        role = m["role"]
        if role == "system":
            system_prompt = m["content"]
        else:
            # Anthropic expects only 'user' and 'assistant' roles inside list
            user_messages.append({"role": role, "content": m["content"]})

    outs: List[str] = []
    for _ in range(n):
        resp = client.messages.create(
            model=cfg["id"],
            system=system_prompt,
            messages=user_messages,  # type: ignore[arg-type]
            max_tokens=_load_config().get("max_tokens", 600),
            temperature=cfg.get("temperature", 0.9),
            top_p=_load_config().get("top_p", 0.95),
        )  # type: ignore[arg-type]
        # First content block should be text
        outs.append(resp.content[0].text.strip())  # type: ignore[attr-defined]
    return outs


def _deepseek_call(messages: List[dict], n: int, cfg) -> List[str]:
    """Call DeepSeek REST API (https://api.deepseek.com/chat/completions)."""
    import requests, json

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("HF_API_TOKEN")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable not set.")

    url = "https://api.deepseek.com/chat/completions"

    # DeepSeek expects OpenAI-style messages
    outs: List[str] = []
    payload_base = {
        "model": cfg.get("id", "deepseek-chat"),
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # DeepSeek doesn't support multiple completions in a single request; loop.
    for _ in range(n):
        payload = {**payload_base, "messages": messages}
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
        resp.raise_for_status()
        data = resp.json()
        outs.append(data["choices"][0]["message"]["content"].strip())
    return outs


CALL_MAP = {
    "nano": _openai_call,
    "haiku": _anthropic_call,
    "deepseek": _deepseek_call,
}


def generate(model_key: str, messages: List[dict], n: int) -> List[str]:
    cfg = _load_config()["models"][model_key]
    if model_key not in CALL_MAP:
        raise ValueError(f"Unknown model key '{model_key}'")
    return CALL_MAP[model_key](messages, n, cfg) 