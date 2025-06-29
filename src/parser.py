import re
from typing import Tuple
import json as _json

_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")

RANGES = [
    (0, 100),  # objective_probability
    (0, 100),  # good_reasons
    (1, 7),    # risk_awareness
    (1, 7),    # blameworthiness
    (1, 7),    # punishment
]

def parse_numbers(text: str) -> Tuple[float, float, float, float, float]:
    """Extract **exactly five** numeric values from a string.

    If more than five numbers are found we take the first five. Values are clipped
    to the expected range.
    """
    matches = [float(x) for x in _NUMBER_RE.findall(text)]
    if len(matches) < 5:
        raise ValueError("Fewer than five numbers found in completion")

    values = matches[:5]
    clipped = [float(max(rng[0], min(v, rng[1]))) for v, rng in zip(values, RANGES)]
    return tuple(clipped)  # type: ignore 

def parse_two(text: str):
    text = text.strip()
    # Try JSON first
    if text.startswith("{"):
        try:
            data = _json.loads(text)
            return (
                float(data["objective_probability"]),
                float(data["good_reasons"]),
            )
        except Exception:
            pass
    # fallback comma parsing
    matches = _NUMBER_RE.findall(text)
    if len(matches) < 2:
        raise ValueError("Need two numbers")
    vals = [float(matches[i]) for i in range(2)]
    vals = [max(0, min(v, 100)) for v in vals]  # both 0-100
    return tuple(vals)

def _remove_fences(text: str) -> str:
    """Remove leading/trailing ``` or ```json fences produced by some models."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
        # fallback: drop first fence
        return "```".join(parts[1:]).strip()
    return text

def _strip_leading_numbers(text: str) -> str:
    """Remove leading enumeration numbers like '1. ' at line starts."""
    return re.sub(r"(?m)^\s*\d+\.?\s*", "", text)

def parse_six(text: str):
    text = _strip_leading_numbers(_remove_fences(text))
    if text.startswith("{"):
        try:
            data = _json.loads(text)
            lower = {k.lower(): v for k, v in data.items()}
            return (
                float(lower["objective_probability"]),
                float(lower["good_reasons"]),
                float(lower["recklessness"]),
                float(lower["negligence"]),
                float(lower["blameworthiness"]),
                float(lower["punishment"]),
            )
        except Exception:
            pass
    # fallback numeric list
    matches = _NUMBER_RE.findall(text)
    if len(matches) < 6:
        raise ValueError("Need six numbers")
    vals = [float(matches[i]) for i in range(6)]
    ranges = [
        (0, 100),
        (0, 100),
        (1, 7),
        (1, 7),
        (1, 7),
        (1, 7),
    ]
    clipped = [max(r[0], min(v, r[1])) for v, r in zip(vals, ranges)]
    return tuple(clipped) 