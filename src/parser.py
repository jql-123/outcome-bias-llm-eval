import re
from typing import Tuple

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
    matches = _NUMBER_RE.findall(text)
    if len(matches) < 2:
        raise ValueError("Need two numbers")
    vals = [float(matches[i]) for i in range(2)]
    vals = [max(0, min(v, 100)) for v in vals]  # both 0-100
    return tuple(vals)

def parse_six(text: str):
    matches = _NUMBER_RE.findall(text)
    if len(matches) < 6:
        raise ValueError("Need six numbers")
    vals = [float(matches[i]) for i in range(6)]
    # ranges
    ranges = [
        (0, 100),  # P_post
        (0, 100),  # GR_post
        (1, 7),    # reckless
        (1, 7),    # negligent
        (1, 7),    # blame
        (1, 7),    # punish
    ]
    clipped = [max(r[0], min(v, r[1])) for v, r in zip(vals, ranges)]
    return tuple(clipped) 