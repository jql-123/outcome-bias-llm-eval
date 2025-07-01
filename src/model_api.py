# ==================== NEW IMPLEMENTATION FOR LIGHTWEIGHT MODELS ====================
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Any

import dotenv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {}
# Global debug hook – holds the latest raw provider response so callers can persist it if needed.
LAST_RAW: object | None = None


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
    for idx, m in enumerate(messages):
        role = m["role"]
        content = m["content"]
        if role == "system":
            # Some mini models (o1-mini, o4-mini) do not yet support the system role
            if any(s in cfg["id"] for s in ("o1-mini", "o4-mini", "o3-mini")):
                # prepend marker so we don't lose the instruction
                role = "user"
                content = f"<system>{content}</system>"
            chat_messages.append({"role": role, "content": content})
        elif role == "user":
            chat_messages.append({"role": "user", "content": content})
        else:
            raise ValueError(f"Unknown message role: {role}")
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Some newer OpenAI models (e.g. o1-mini, o4-mini) renamed the argument
    # from `max_tokens` → `max_completion_tokens`. Detect and switch.
    max_param_name = "max_tokens"
    if any(s in cfg["id"] for s in ("o1-mini", "o4-mini", "o3-mini")):
        max_param_name = "max_completion_tokens"

    # Use larger token budget for the mini models – their reasoning often
    # consumes >600 tokens before the final answer. Default to 1500 unless the
    # user explicitly set a higher value in config.yaml.
    token_budget = max(cfg.get("max_tokens", 600), 1500) if "-mini" in cfg["id"] else cfg.get("max_tokens", 600)

    create_kwargs = {
        "model": cfg["id"],
        "messages": chat_messages,
        "temperature": cfg.get("temperature", 1.0),
        max_param_name: token_budget,
        "n": n,
    }

    # top_p is not yet supported on some mini models
    if max_param_name == "max_tokens":  # regular models
        create_kwargs["top_p"] = cfg.get("top_p", 0.95)

    # All mini models: use legacy function calling to return strict JSON
    if cfg["id"] in ("o4-mini", "o3-mini"):
        create_kwargs["functions"] = [JURY_SCHEMA]
        create_kwargs["function_call"] = {"name": "jury_scoring"}

    # Remove the other param to avoid 400 errors
    if max_param_name == "max_completion_tokens" and "max_tokens" in create_kwargs:
        create_kwargs.pop("max_tokens", None)

    response = client.chat.completions.create(**create_kwargs)
    global LAST_RAW
    LAST_RAW = response
    return [_choice_to_text(choice) for choice in response.choices]


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


# Config keys that should use the `responses.create` endpoint. Currently only
# o4mini (and future o3mini) support it; o1-mini must stay on chat-completions.
_REASONING_MODELS = {"o3mini"}

def _is_reasoning_model(model_key: str, cfg) -> bool:
    """Return True when this model should be queried via the `responses` API."""
    return model_key in _REASONING_MODELS


def _openai_reasoning_call(messages: list[dict], n: int, cfg) -> list[str]:
    import openai, json
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    user_prompt = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    # Provide a generous token budget so the model has room for the reasoning
    # phase **and** the final answer. 1500 works well in practice and is still
    # within the 4-mini hard limit (8192).
    token_budget = max(cfg.get("max_tokens", 600), 1500)

    resp = client.responses.create(  # type: ignore[arg-type]
        model=cfg["id"],
        reasoning={"effort": "medium"},
        input=[{"role": "user", "content": user_prompt}],
        max_output_tokens=token_budget,
        functions=[JURY_SCHEMA],  # type: ignore[arg-type]
        function_call={"name": "jury_scoring"},
    )

    global LAST_RAW
    LAST_RAW = resp

    # Extract the function-call arguments JSON from the first tool call
    for item in resp.output:
        if getattr(item, "type", "") == "tool":
            try:
                return [item.arguments.strip()]  # type: ignore[attr-defined]
            except Exception:
                pass

    # Fallback: if the model still returned plain text
    return [getattr(resp, "output_text", "").strip()]


def generate(model_key: str, messages: list[dict], n: int):
    cfg = _load_config()["models"][model_key]

    provider = cfg.get("provider", "openai").lower()

    # Route call based on provider first.
    if provider == "deepseek":
        return _deepseek_call(messages, n, cfg)
    if provider == "anthropic":
        return _anthropic_call(messages, n, cfg)

    # Default (OpenAI) provider ─ decide between chat and reasoning endpoints.
    if _is_reasoning_model(model_key, cfg):
        return _openai_reasoning_call(messages, n, cfg)
    return _openai_call(messages, n, cfg)


# -----------------------------------------------------------------------------
# Shared JSON schema for mini models (function calling)
# -----------------------------------------------------------------------------

JURY_SCHEMA = {
    "name": "jury_scoring",
    "description": "Return six quantitative judgments about the case",
    "parameters": {
        "type": "object",
        "properties": {
            "objective_probability": {"type": "integer", "minimum": 0, "maximum": 100},
            "good_reasons": {"type": "integer", "minimum": 0, "maximum": 100},
            "recklessness": {"type": "integer", "minimum": 1, "maximum": 7},
            "negligence": {"type": "integer", "minimum": 1, "maximum": 7},
            "blameworthiness": {"type": "integer", "minimum": 1, "maximum": 7},
            "punishment": {"type": "integer", "minimum": 1, "maximum": 7},
        },
        "required": [
            "objective_probability",
            "good_reasons",
            "recklessness",
            "negligence",
            "blameworthiness",
            "punishment",
        ],
        "additionalProperties": False,
    },
}

def _choice_to_text(choice):
    """Extract text or function-call arguments from a ChatCompletion choice."""
    msg = choice.message
    # New tool calling not supported on mini models; use legacy function_call
    if getattr(msg, "function_call", None):
        try:
            return msg.function_call.arguments.strip()
        except Exception:
            pass
    # Fallback to plain content
    return msg.content.strip() if msg.content else "" 