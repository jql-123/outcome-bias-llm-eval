SYSTEM = (
    "You are a careful juror. Return **exactly five numbers** separated by commas "
    "(objective_probability 0-100, good_reasons 0-100, risk_awareness 1-7, "
    "blameworthiness 1-7, punishment 1-7)."
)


def user_prompt(vignette: str) -> str:
    """Build the standard user prompt given a single-ending vignette."""
    return f"""{vignette}

Questions:
1. Objective probability harm occurs (0-100):
2. Were there good reasons? (0-100):
3. Risk awareness (1-7):
4. Blameworthiness (1-7):
5. Deserved punishment (1-7):
"""


def within_subject_prompt(vignette_within: str) -> str:
    """Study-2 prompt – the within-subject DOCX already contains both endings."""
    # We just append the same questionnaire.
    return user_prompt(vignette_within) 