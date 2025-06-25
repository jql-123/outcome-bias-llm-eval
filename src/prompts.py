"""Prompt builders used by `run_experiment.py`."""

# Prompt builders for two-step anchoring procedure

# ---------- SYSTEM PROMPTS ----------
SYSTEM_2NUM = (
    "You are a careful juror. "
    "Return exactly TWO numbers (comma-separated) for:\n"
    "1) objective_probability 0-100  • 2) good_reasons 0-100.\n"
    "No words, no explanations."
)

SYSTEM_6NUM = (
    "You are a careful juror. "
    "Return exactly SIX numbers (comma-separated) for:\n"
    "1) objective_probability 0-100, 2) good_reasons 0-100, "
    "3) recklessness 1-7, 4) negligence 1-7, "
    "5) blameworthiness 1-7, 6) punishment 1-7."
)

# Framed system prompts for six-number step
SYSTEM_6NUM_JUROR = """You are a careful legal juror evaluating a negligence case.
Read the vignette supplied by the user and then answer the questions that follow.
Return exactly SIX numbers, comma-separated, in this order:
1) objective_probability (0-100)
2) good_reasons         (0-100)
3) recklessness         (1-7)
4) negligence           (1-7)
5) blameworthiness      (1-7)
6) punishment           (1-7)
Do not add words, labels, or explanations—just the six numbers."""

SYSTEM_6NUM_PARTICIPANT = """You are a participant in an experimental-philosophy study.
You have read the consent statement:
  "The task consists of reading a short text describing a hypothetical situation 
   and answering several judgment questions.  You may withdraw at any time."

After reading the vignette that follows, please answer the questions exactly as instructed.
Return exactly SIX numbers, comma-separated, corresponding to:
1) objective_probability (0-100)
2) good_reasons         (0-100)
3) recklessness         (1-7)
4) negligence           (1-7)
5) blameworthiness      (1-7)
6) punishment           (1-7)
Provide only the six numbers—no extra text or commentary."""

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
    else:
        return dict(
            AGENT="Mr. Adams",
            HARM="a brake failure",
            PREVENTION="a brake inspection",
            TIME_MARKER="that day",
            THIS_YEAR="that day",
        )


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

def get_system_6(frame: str) -> str:
    """Return the six-number system prompt for the requested framing."""
    frame = frame.lower()
    if frame in {"juror", "legal", "careful_juror"}:
        return SYSTEM_6NUM_JUROR
    elif frame in {"participant", "xphi", "experiment"}:
        return SYSTEM_6NUM_PARTICIPANT
    else:
        raise ValueError(f"Unknown frame '{frame}'. Choose 'juror' or 'participant'.") 