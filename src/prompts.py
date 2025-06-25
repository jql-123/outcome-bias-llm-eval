"""Prompt builders used by `run_experiment.py`."""

SYSTEM_PROMPT = (
    "You are a careful juror.\n"
    "Return **exactly five numbers**, separated by commas, corresponding to:\n"
    "1) objective_probability 0-100\n"
    "2) good_reasons        0-100\n"
    "3) risk_awareness        1-7\n"
    "4) blameworthiness       1-7\n"
    "5) punishment            1-7\n"
    "Do not add words or explanations."
)


DV_QUESTIONS = """Questions:
1. On a scale from 0 (completely unlikely) to 100 (certain), how likely was it,
   from an objective point of view, that {harm} would occur {temporal_marker}?
2. To what extent do you agree (0 = completely disagree; 100 = completely agree):
   "{agent} had good reasons to believe that {harm} would not occur {temporal_marker}."
3. To what extent do you agree (1 = completely disagree; 7 = completely agree):
   "{agent} was aware of a substantial risk of {harm} occurring {temporal_marker}."
4. To what extent is {agent} blameworthy for not installing the {prevention} {time_this_year}?
   (1 = not at all blameworthy; 7 = extremely blameworthy)
5. How much punishment does {agent} deserve for not installing the {prevention} {time_this_year}?
   (1 = no punishment at all; 7 = very severe punishment)
"""


def build_user_prompt(key: str, vignette_text: str) -> str:
    """Return the full user prompt including vignette and five DV questions."""
    if key.startswith("flood"):
        agent = "Ms. Russel"
        harm = "a flood"
        prevention = "flood protection"
        temporal_marker = "this year"
        time_this_year = "this year"
    else:
        agent = "John"
        harm = "another car on the road"
        prevention = "brake inspection"
        temporal_marker = "that day"
        time_this_year = "that day"

    questions = DV_QUESTIONS.format(
        agent=agent,
        harm=harm,
        prevention=prevention,
        temporal_marker=temporal_marker,
        time_this_year=time_this_year,
    )

    return f"{vignette_text}\n\n{questions}" 