"""Prompt builders used by `run_experiment.py`."""

# Prompt builders for two-step anchoring procedure

# ---------- SYSTEM PROMPT COMPONENTS ----------
# We build system prompts from two building blocks:
# 1) SETUP text depending on the framing (juror vs experiment)
# 2) FORMAT text depending on whether we expect TWO or SIX numbers.

SETUP_TEXT = {
    "juror": (
        "You are a careful legal juror evaluating a case.\n"
        "After reading the case that follows, please answer the questions exactly as instructed."
    ),
    "experiment": (
        "You are a participant in an experimental-philosophy study.\n"
        "You have read the consent statement: \n"
        "  \"The task consists of reading a short text describing a hypothetical situation \n"
        "   and answering several judgment questions.  You may withdraw at any time.\"\n"
        "After reading the case that follows, please answer the questions exactly as instructed."
    ),
}

FORMAT_2NUM = (
    "Respond with a single line of **valid JSON** that has exactly two key–value pairs:\n"
    "{\n  \"objective_probability\": <0-100>,\n  \"good_reasons\": <0-100>\n}\n"
    "Do not add any extra keys, text, or commentary."
)

# FORMAT_6NUM = (
#     "Respond with a single line of **valid JSON** that has exactly six key–value pairs:\n"
#     "{\n  \"objective_probability\": <0-100>,\n  \"good_reasons\": <0-100>,\n  \"recklessness\": <1-7>,\n  \"negligence\": <1-7>,\n  \"blameworthiness\": <1-7>,\n  \"punishment\": <1-7>\n}\n"
#     "Do not add any extra keys, text, or commentary."
# )
FORMAT_6NUM =  """
Respond **only** with a single JSON object that has exactly these six keys:
{
  "objective_probability": <0-100>,
  "good_reasons": <0-100>,
  "recklessness": <1-7>,
  "negligence": <1-7>,
  "blameworthiness": <1-7>,
  "punishment": <1-7>
}
"""


def _system_prompt(frame: str, n_nums: int) -> str:
    """Assemble system prompt for given frame ('juror' or 'experiment') and count."""
    frame = frame.lower()
    if frame not in SETUP_TEXT:
        raise ValueError(f"Unknown frame '{frame}'. Use 'juror' or 'experiment'.")
    setup = SETUP_TEXT[frame]
    fmt = FORMAT_2NUM if n_nums == 2 else FORMAT_6NUM
    return f"{setup}\n\n{fmt}"


# Convenience wrappers for existing call-sites --------------------------------


def get_system_2(frame: str) -> str:
    """System prompt for two-number anchor questions."""
    return _system_prompt(frame, 2)


def get_system_6(frame: str) -> str:
    """System prompt for six-number DV block."""
    return _system_prompt(frame, 6)

# ---------- QUESTION BLOCKS ----------
ANCHOR_QUESTIONS = """Questions:
1. On a scale from 0 (completely unlikely) to 100 (certain), how likely is it,
   from an objective point of view, that {HARM} will occur {TIME_MARKER}?
2. To what extent do you agree (0 = completely disagree; 100 = completely agree):
   "{AGENT} has good reasons to believe that {HARM} will not occur {TIME_MARKER}."
"""

POST_QUESTIONS = """Questions:
1. On a scale from 0 (completely unlikely) to 100 (certain), how likely was it,
   from an objective point of view, that {HARM} would occur {TIME_MARKER}?
2. To what extent do you agree (0 = completely disagree; 100 = completely agree):
   "{AGENT} had good reasons to believe that {HARM} would not occur {TIME_MARKER}."
3. Recklessness (1 = completely disagree; 7 = completely agree):
   "{AGENT} acted recklessly by not installing the {PREVENTION} {THIS_YEAR}."
4. Negligence (1 = completely disagree; 7 = completely agree):
   "{AGENT} acted negligently by not installing the {PREVENTION} {THIS_YEAR}."
5. To what extent is {AGENT} blameworthy for not installing the {PREVENTION} {THIS_YEAR}?
   (1 = not at all blameworthy; 7 = extremely blameworthy)
6. How much punishment does {AGENT} deserve for not installing the {PREVENTION} {THIS_YEAR}?
   (1 = no punishment at all; 7 = very severe punishment)
"""

# ---------- Helpers ----------

def _fill_placeholders(key: str):
    if key.startswith("flood"):
        return dict(
            AGENT="Ms. Russel",
            HARM="a flood",
            PREVENTION="flood protection",
            TIME_MARKER="this year",
            THIS_YEAR="this year",
        )
    elif key.startswith("traffic"):
        return dict(
            AGENT="John",
            HARM="another car on the road",
            PREVENTION="slowing down at the intersection",
            TIME_MARKER="that day",
            THIS_YEAR="that day",
        )
    elif key.startswith("warehouse_fire"):
        return dict(
            AGENT="Mr. Thompson",
            HARM="a fire",
            PREVENTION="sprinkler test",
            TIME_MARKER="that month",
            THIS_YEAR="that month",
        )
    elif key.startswith("ski_resort_avalanche"):
        return dict(
            AGENT="Maria Santos",
            HARM="an avalanche",
            PREVENTION="controlled blast",
            TIME_MARKER="that day",
            THIS_YEAR="that day",
        )
    elif key.startswith("hospital_backup_generator"):
        return dict(
            AGENT="Dr. Peterson",
            HARM="a power outage",
            PREVENTION="generator maintenance",
            TIME_MARKER="that quarter",
            THIS_YEAR="that quarter",
        )
    elif key.startswith("datacenter_cooling"):
        return dict(
            AGENT="Jake Mitchell",
            HARM="a cooling failure",
            PREVENTION="backup cooling system",
            TIME_MARKER="that winter",
            THIS_YEAR="that winter",
        )
    elif key.startswith("swimming_pool_drain"):
        return dict(
            AGENT="Patricia Davis",
            HARM="a drain incident",
            PREVENTION="drain inspection",
            TIME_MARKER="that month",
            THIS_YEAR="that month",
        )
    elif key.startswith("chemical_valve_leak"):
        return dict(
            AGENT="Lisa Rodriguez",
            HARM="a chemical leak",
            PREVENTION="gasket replacement",
            TIME_MARKER="that month",
            THIS_YEAR="that month",
        )
    elif key.startswith("drone_delivery_battery"):
        return dict(
            AGENT="Marcus Williams",
            HARM="a battery failure",
            PREVENTION="cold-weather batteries",
            TIME_MARKER="that winter",
            THIS_YEAR="that winter",
        )
    elif key.startswith("movie_set_rigging"):
        return dict(
            AGENT="Tom Bradley",
            HARM="a cable failure",
            PREVENTION="secondary safety cables",
            TIME_MARKER="that stunt",
            THIS_YEAR="that stunt",
        )
    else:
        raise ValueError(f"Unknown scenario key: {key}")


def build_anchor_prompt(key: str, intro_text: str) -> str:
    """Step-A prompt: intro only + two questions."""
    params = _fill_placeholders(key)
    return intro_text.strip() + "\n\n" + ANCHOR_QUESTIONS.format(**params)


def build_post_prompt(
    key: str,
    full_vignette: str,
    anchor_q1: float | int | str,
    anchor_q2: float | int | str,
) -> str:
    """Step-B prompt: full vignette + reminder + six DV questions."""
    params = _fill_placeholders(key)
    reminder = (
        "We previously asked and you answered:\n"
        f"  Q1 (objective probability) = {anchor_q1}\n"
        f"  Q2 (good reasons) = {anchor_q2}\n\n"
    )
    return (
        full_vignette.strip()
        + "\n\n"
        + reminder
        + POST_QUESTIONS.format(**params)
    )

# keep for backward compatibility
def get_system_6_old(frame: str) -> str:  # noqa: E501
    return get_system_6(frame)

def build_expert_prompt(key: str, text: str) -> str:
    """Build prompt for expert-probability (Exp 6) design."""
    params = _fill_placeholders(key)
    return text.strip() + "\n\n" + POST_QUESTIONS.format(**params)

# Generic helper when we already have the vignette body ready (intro/outcome/[expert])
def build_post_prompt_from_body(
    key: str,
    body: str,
    anchor_q1: float | int | str,
    anchor_q2: float | int | str,
) -> str:
    """Create step-B prompt given a prepared body (without questions)."""
    params = _fill_placeholders(key)
    reminder = (
        "We previously asked and you answered:\n"
        f"  Q1 (objective probability) = {anchor_q1}\n"
        f"  Q2 (good reasons) = {anchor_q2}\n\n"
    )
    return body.strip() + "\n\n" + reminder + POST_QUESTIONS.format(**params)

# ---------------------------------------------------------------------------
# Single-step baseline prompt (Study 1) – intro + outcome + 6 Qs (no reminder)
# ---------------------------------------------------------------------------


def build_single_step_prompt(key: str, body: str) -> str:
    """Return prompt consisting of *body* (intro/outcome etc.) plus 6 DV questions.

    Used in Study-1 where there is no anchor step and therefore no reminder.
    """
    params = _fill_placeholders(key)
    return body.strip() + "\n\n" + POST_QUESTIONS.format(**params) 