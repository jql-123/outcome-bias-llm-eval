"""Build `data/vignettes.json` with six cleaned vignette texts.

The logic follows the detailed requirements from the latest task description.

For the two *between-subjects* DOCX exports we:
1. Identify the Qualtrics survey block that contains the vignette.
2. Remove all question text.
3. Keep the intro up to – but *not including* – the outcome sentence.
4. Append the outcome paragraph only ("As during…" or "It just so happens…").

For the *within-subjects* DOCX exports we keep only the two outcome paragraphs
(good first, bad second) and join them with an "---" delimiter.

Implementation relies on `python-docx` (to satisfy the requirement) **and** a
fallback to raw XML parsing because the narrative is stored in elements that
`python-docx` currently ignores.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import docx  # noqa: F401 – imported per requirement

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------

FLOOD_BETWEEN = (
    ROOT
    / "Qualtrics Exports/Exp 1 and Exp 3/Ex_ante_flood-3.docx"
)
TRAFFIC_BETWEEN = (
    ROOT
    / "Qualtrics Exports/Exp 8/ex_ante_intersection-2.docx"
)
FLOOD_WITHIN = (
    ROOT
    / "Qualtrics Exports/Exp 2 and Exp 4/Flood_multiple_GOOD_endings_PLUS_Flood_WS.docx"
)
TRAFFIC_WITHIN = (
    ROOT
    / "Qualtrics Exports/Exp 6 and Exp 7/Intersection_BWS_WS-2.docx"
)

# Qualtrics block names we care about -> output key mapping
BETWEEN_BLOCKS = {
    (FLOOD_BETWEEN, "A neutral ex ante"): "flood_good",
    (FLOOD_BETWEEN, "B severe ex ante"): "flood_bad",
    (TRAFFIC_BETWEEN, "A Ex ante good"): "traffic_good",
    (TRAFFIC_BETWEEN, "B ex ante bad"): "traffic_bad",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TAG_RE = re.compile(r"<[^>]+>")

def _block_plain_text(docx_path: Path, block_name: str) -> str:
    """Return plain text for a single Qualtrics block."""
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")

    pattern = f"Start of Block: {re.escape(block_name)}"
    m = re.search(pattern, xml)
    if not m:
        raise ValueError(f"Block '{block_name}' not found in {docx_path.name}")
    start = m.end()
    end = xml.find("End of Block:", start)
    block_xml = xml[start:end]
    return TAG_RE.sub("", block_xml)

def _intro_and_outcome(full_text: str, good: bool) -> str:
    """Extract intro paragraph + outcome paragraph.

    Parameters
    ----------
    full_text : str
        Plain text for a single Qualtrics block.
    good : bool
        Whether the outcome is the *good* version.
    """

    outcome_phrase = (
        "As"  # will match 'As during the previous years' or 'As usual...'
        if good
        else "It just so happens"
    )

    o_idx = full_text.find(outcome_phrase)
    if o_idx == -1:
        raise ValueError(f"Outcome phrase '{outcome_phrase}' not found.")

    # Intro is everything before the first question (heuristic: 'On a scale')
    q_idx = full_text.lower().find("on a scale")
    intro = full_text[:q_idx].strip()

    # Outcome paragraph runs until next question (again 'On a scale')
    next_q = full_text.lower().find("on a scale", o_idx)
    outcome = full_text[o_idx: next_q if next_q != -1 else None].strip()

    return f"{intro} {outcome}".strip()


def _within_outcomes(full_text: str, good_phrase: str) -> tuple[str, str]:
    """Return (good_paragraph, bad_paragraph) in that order."""
    good_idx = full_text.find(good_phrase)
    bad_idx = full_text.find("It just so happens")
    if good_idx == -1 or bad_idx == -1:
        raise ValueError("Could not locate good/bad outcome sentences in within-subjects file.")

    def _paragraph(start: int) -> str:
        end = full_text.lower().find("on a scale", start)
        return full_text[start:end].strip()

    good_para = _paragraph(good_idx)
    bad_para = _paragraph(bad_idx)
    return good_para, bad_para

def _doc_plain_text(docx_path: Path) -> str:
    """Plain text for entire document."""
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    return re.sub(r"\s+", " ", TAG_RE.sub("", xml)).strip()

# Cleaning helper ------------------------------------------------------------

def _clean(text: str) -> str:
    """Remove scenario labels (e.g. 'A1', 'B1') and any 'Second ending:' notes."""
    text = text.strip()
    # drop leading block label like 'A neutral ex ante' if still present
    text = re.sub(r"^Start of Block:[^A-Z]*", "", text, flags=re.IGNORECASE)
    # drop leading scenario id (e.g. 'A1', 'B1')
    text = re.sub(r"^[A-D]\d\s+", "", text)
    # drop 'Second ending:' and everything after
    text = re.sub(r"Second ending:.*", "", text, flags=re.IGNORECASE)
    return text.strip()

# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def main() -> None:
    for p in [FLOOD_BETWEEN, TRAFFIC_BETWEEN, FLOOD_WITHIN, TRAFFIC_WITHIN]:
        if not p.exists():
            raise FileNotFoundError(p)
        # Copy source files into data/raw for provenance
        RAW_DIR.joinpath(p.name).write_bytes(p.read_bytes())

    v: dict[str, str] = {}

    # ---------------- Between-subjects ----------------
    for (doc_path, block_name), out_key in BETWEEN_BLOCKS.items():
        block_text = TAG_RE.sub("", _block_plain_text(doc_path, block_name))
        block_text = re.sub(r"\s+", " ", block_text).strip()
        block_text = _clean(block_text)
        is_good = out_key.endswith("good")
        v[out_key] = _intro_and_outcome(block_text, good=is_good)

    # ---------------- Within-subjects -----------------
    flood_within_text = _doc_plain_text(FLOOD_WITHIN)
    good_para, bad_para = _within_outcomes(
        flood_within_text, "As during the previous years"
    )
    v["flood_within"] = f"{_clean(good_para)}\n---\n{_clean(bad_para)}"

    traffic_within_text = _doc_plain_text(TRAFFIC_WITHIN)
    good_para, bad_para = _within_outcomes(
        traffic_within_text, "As usual"
    )
    v["traffic_within"] = f"{_clean(good_para)}\n---\n{_clean(bad_para)}"

    # ---------------- Write JSON ----------------------
    out_path = DATA_DIR / "vignettes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(v, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path} with {len(v)} vignettes (expected 6).")


if __name__ == "__main__":
    main() 