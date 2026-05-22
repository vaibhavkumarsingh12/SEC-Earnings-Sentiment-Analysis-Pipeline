"""
Text Preprocessing & Chunking for SEC Filings
================================================
Splits long MD&A text into FinBERT-compatible chunks using a
sliding-window approach with sentence-level overlap.
"""

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import nltk  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NLTK bootstrap
# ---------------------------------------------------------------------------
def ensure_nltk_data() -> None:
    """Download the 'punkt_tab' tokenizer data if it is not already present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
        logger.debug("NLTK 'punkt_tab' data already available.")
    except LookupError:
        logger.info("Downloading NLTK 'punkt_tab' tokenizer data …")
        nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Sentence tokenization
# ---------------------------------------------------------------------------
def sentence_tokenize(text: str) -> list[str]:
    """Split *text* into a list of sentences using NLTK's Punkt tokenizer.

    Parameters
    ----------
    text : str
        Raw text (e.g. an MD&A section).

    Returns
    -------
    list[str]
        Individual sentences.
    """
    ensure_nltk_data()
    sentences = nltk.sent_tokenize(text)
    logger.debug("Tokenized text into %d sentences.", len(sentences))
    return sentences


# ---------------------------------------------------------------------------
# Sliding-window chunking
# ---------------------------------------------------------------------------
def sliding_window_chunks(
    sentences: list[str],
    max_tokens: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Combine sentences into chunks that respect a token-count ceiling.

    Parameters
    ----------
    sentences : list[str]
        Ordered list of sentences (output of :func:`sentence_tokenize`).
    max_tokens : int, optional
        Maximum estimated token count per chunk. Defaults to
        ``config.FINBERT_MAX_TOKENS``.
    overlap : int, optional
        Number of trailing sentences from the previous chunk to prepend to the
        next chunk for context continuity.  Defaults to
        ``config.FINBERT_OVERLAP_SENTENCES``.

    Returns
    -------
    list[str]
        Text chunks ready for model inference.
    """
    if max_tokens is None:
        max_tokens = config.FINBERT_MAX_TOKENS
    if overlap is None:
        overlap = config.FINBERT_OVERLAP_SENTENCES

    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_token_count: int = 0

    for sentence in sentences:
        sentence_tokens = len(sentence.split())

        # Edge case: a single sentence already exceeds max_tokens
        if sentence_tokens >= max_tokens and not current_sentences:
            logger.warning(
                "Single sentence exceeds max_tokens (%d >= %d); "
                "including it as its own chunk.",
                sentence_tokens,
                max_tokens,
            )
            chunks.append(sentence)
            continue

        # Would adding this sentence overflow the chunk?
        if current_token_count + sentence_tokens > max_tokens:
            # Finalise the current chunk
            chunks.append(" ".join(current_sentences))
            logger.debug(
                "Chunk %d finalised with ~%d tokens (%d sentences).",
                len(chunks),
                current_token_count,
                len(current_sentences),
            )

            # Start a new chunk with the last `overlap` sentences for context
            overlap_sentences = current_sentences[-overlap:] if overlap > 0 else []
            current_sentences = list(overlap_sentences)
            current_token_count = sum(len(s.split()) for s in current_sentences)

        current_sentences.append(sentence)
        current_token_count += sentence_tokens

    # Flush the remaining sentences
    if current_sentences:
        chunks.append(" ".join(current_sentences))
        logger.debug(
            "Final chunk %d finalised with ~%d tokens (%d sentences).",
            len(chunks),
            current_token_count,
            len(current_sentences),
        )

    logger.info("Created %d chunk(s) from %d sentences.", len(chunks), len(sentences))
    return chunks


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def prepare_chunks(text: str) -> list[str]:
    """End-to-end convenience: sentence-tokenize *text*, then chunk it.

    Parameters
    ----------
    text : str
        Raw document text.

    Returns
    -------
    list[str]
        Chunks suitable for FinBERT inference.
    """
    sentences = sentence_tokenize(text)
    return sliding_window_chunks(sentences)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    sample_text = (
        "The company reported strong revenue growth of 12 percent year over year. "
        "Net income increased to $4.5 billion, exceeding analyst expectations. "
        "Management highlighted continued investment in digital transformation. "
        "Credit quality remained robust across all major portfolios. "
        "Provisions for credit losses decreased by 15 percent compared to the prior year. "
        "The board of directors declared a quarterly dividend of $1.00 per share. "
        "Looking ahead, management expressed cautious optimism about the macroeconomic environment. "
        "Regulatory capital ratios remained well above minimum requirements. "
        "The firm completed the acquisition of a mid-sized fintech platform during the quarter. "
        "Overall, the results reflect disciplined execution of the strategic plan."
    )

    logger.info("=== Text Preprocessor Demo ===")
    chunks = prepare_chunks(sample_text)
    for i, chunk in enumerate(chunks, 1):
        word_count = len(chunk.split())
        print(f"\n--- Chunk {i} (~{word_count} tokens) ---")
        print(chunk)
