"""
Loughran-McDonald Lexicon Scorer
==================================
Downloads the Loughran-McDonald Master Dictionary and scores financial text
by counting matches against sentiment and tone categories.
"""

import logging
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import pandas as pd  # noqa: E402
import requests  # noqa: E402

logger = logging.getLogger(__name__)

# Category columns of interest in the CSV
_CATEGORIES = ["Negative", "Positive", "Uncertainty", "Litigious", "Constraining"]


# ---------------------------------------------------------------------------
# Dictionary download
# ---------------------------------------------------------------------------
def download_lm_dictionary() -> None:
    """Download the Loughran-McDonald Master Dictionary CSV if it doesn't exist.

    The file is saved to ``config.LM_DICTIONARY_PATH``.  If the file is
    already present the download is skipped.
    """
    dest = Path(config.LM_DICTIONARY_PATH)
    if dest.exists():
        logger.debug("LM dictionary already present at %s — skipping download.", dest)
        return

    logger.info("Downloading Loughran-McDonald dictionary from %s …", config.LM_DICTIONARY_URL)
    try:
        resp = requests.get(config.LM_DICTIONARY_URL, timeout=120)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        logger.info("Saved LM dictionary to %s (%d bytes).", dest, len(resp.content))
    except requests.RequestException as exc:
        logger.error("Failed to download LM dictionary: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Dictionary loading
# ---------------------------------------------------------------------------
def load_lm_dictionary() -> dict[str, set[str]]:
    """Load the Loughran-McDonald dictionary and build per-category word sets.

    A word belongs to a category if the corresponding column value is > 0
    (the value represents a year such as 2009, not a simple boolean).

    Returns
    -------
    dict[str, set[str]]
        Mapping from lowercased category name to a set of uppercased words.
        Keys: ``'positive'``, ``'negative'``, ``'uncertainty'``,
        ``'litigious'``, ``'constraining'``.
    """
    download_lm_dictionary()

    logger.info("Loading LM dictionary from %s …", config.LM_DICTIONARY_PATH)
    df = pd.read_csv(config.LM_DICTIONARY_PATH)

    word_lists: dict[str, set[str]] = {}
    for cat in _CATEGORIES:
        if cat not in df.columns:
            logger.warning("Column '%s' not found in dictionary CSV — skipping.", cat)
            continue
        # Values > 0 indicate the word belongs to that category
        words = df.loc[df[cat] > 0, "Word"].str.upper().tolist()
        word_lists[cat.lower()] = set(words)
        logger.info("  %s: %d words", cat.lower(), len(word_lists[cat.lower()]))

    return word_lists


# ---------------------------------------------------------------------------
# Text scoring
# ---------------------------------------------------------------------------
def score_text(text: str, word_lists: dict[str, set[str]] | None = None) -> dict:
    """Score *text* against the Loughran-McDonald lexicon.

    Parameters
    ----------
    text : str
        Raw text to score.
    word_lists : dict[str, set[str]], optional
        Pre-loaded dictionary (from :func:`load_lm_dictionary`).  If *None*
        the dictionary is loaded on-the-fly.

    Returns
    -------
    dict
        Metrics including ``lm_net_score``, per-category percentages
        (``positive_pct``, ``negative_pct``, ``uncertainty_pct``,
        ``litigious_pct``, ``constraining_pct``), raw counts, and
        ``total_words``.
    """
    if word_lists is None:
        word_lists = load_lm_dictionary()

    # Tokenize: extract alphabetic tokens, uppercase for matching
    tokens = [t.upper() for t in re.findall(r"[a-zA-Z]+", text)]
    total = len(tokens)

    if total == 0:
        logger.warning("No alphabetic tokens found in text.")
        return {
            "lm_net_score": 0.0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "uncertainty_pct": 0.0,
            "litigious_pct": 0.0,
            "constraining_pct": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "uncertainty_count": 0,
            "litigious_count": 0,
            "constraining_count": 0,
            "total_words": 0,
        }

    counts: dict[str, int] = {}
    for cat, words in word_lists.items():
        counts[cat] = sum(1 for t in tokens if t in words)

    pos = counts.get("positive", 0)
    neg = counts.get("negative", 0)

    lm_net_score = (pos - neg) / total

    result = {
        "lm_net_score": round(lm_net_score, 6),
        "positive_pct": round(pos / total * 100, 4),
        "negative_pct": round(neg / total * 100, 4),
        "uncertainty_pct": round(counts.get("uncertainty", 0) / total * 100, 4),
        "litigious_pct": round(counts.get("litigious", 0) / total * 100, 4),
        "constraining_pct": round(counts.get("constraining", 0) / total * 100, 4),
        "positive_count": pos,
        "negative_count": neg,
        "uncertainty_count": counts.get("uncertainty", 0),
        "litigious_count": counts.get("litigious", 0),
        "constraining_count": counts.get("constraining", 0),
        "total_words": total,
    }

    logger.info(
        "LM scoring complete — net_score=%.4f, +%d/-%d out of %d words.",
        lm_net_score,
        pos,
        neg,
        total,
    )
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    sample = (
        "Revenue growth was strong and management expressed optimism about future "
        "earnings. However, uncertainty around regulatory changes and potential "
        "litigation risk remain significant concerns. The company acknowledged "
        "constraining factors in the current economic environment, including "
        "inflationary pressures and supply chain disruptions."
    )

    logger.info("=== Loughran-McDonald Scorer Demo ===")
    scores = score_text(sample)
    print("\n--- LM Scores ---")
    for key, val in scores.items():
        print(f"  {key}: {val}")
