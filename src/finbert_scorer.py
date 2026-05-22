"""
FinBERT Sentiment Scorer
=========================
Loads ProsusAI/finbert (or a compatible checkpoint) and scores text chunks,
returning per-chunk probabilities and document-level aggregate metrics.
"""

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

logger = logging.getLogger(__name__)


class FinBERTScorer:
    """Wrapper around a HuggingFace FinBERT model for financial sentiment scoring.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier.  Defaults to ``config.FINBERT_MODEL_NAME``.
    batch_size : int, optional
        Number of chunks processed per forward pass.
        Defaults to ``config.FINBERT_BATCH_SIZE``.
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name or config.FINBERT_MODEL_NAME
        self.batch_size = batch_size or config.FINBERT_BATCH_SIZE

        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        # Load tokenizer & model
        logger.info("Loading tokenizer and model from '%s' …", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Read label ordering from model config (e.g. {0: 'positive', 1: 'negative', 2: 'neutral'})
        self.id2label: dict[int, str] = self.model.config.id2label
        self.label_names: list[str] = [
            self.id2label[i] for i in range(len(self.id2label))
        ]
        logger.info("Label mapping: %s", self.id2label)

    # ------------------------------------------------------------------
    # Per-chunk scoring
    # ------------------------------------------------------------------
    def score_chunks(self, chunks: list[str]) -> list[dict]:
        """Score a list of text chunks and return per-chunk sentiment probabilities.

        Parameters
        ----------
        chunks : list[str]
            Text chunks produced by the text-preprocessor.

        Returns
        -------
        list[dict]
            Each dict contains a probability for every label **plus** a
            ``'label'`` key with the dominant sentiment string.
        """
        results: list[dict] = []

        for start in tqdm(
            range(0, len(chunks), self.batch_size),
            desc="FinBERT scoring",
            unit="batch",
        ):
            batch = chunks[start : start + self.batch_size]
            encodings = self.tokenizer(
                batch,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            # Move every tensor to the target device
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = self.model(**encodings)

            probs = F.softmax(outputs.logits, dim=-1).cpu().tolist()

            for prob_row in probs:
                score_dict = {
                    label: round(prob, 6)
                    for label, prob in zip(self.label_names, prob_row)
                }
                score_dict["label"] = max(score_dict, key=score_dict.get)  # type: ignore[arg-type]
                results.append(score_dict)

        logger.info("Scored %d chunk(s).", len(results))
        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def aggregate_scores(chunk_scores: list[dict]) -> dict:
        """Aggregate per-chunk scores into document-level summary metrics.

        Parameters
        ----------
        chunk_scores : list[dict]
            Output of :meth:`score_chunks`.

        Returns
        -------
        dict
            Keys: ``net_polarity``, ``positive_ratio``, ``negative_ratio``,
            ``avg_positive``, ``avg_negative``, ``avg_neutral``,
            ``dominant_sentiment``, ``num_chunks``.
        """
        n = len(chunk_scores)
        if n == 0:
            return {
                "net_polarity": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "avg_positive": 0.0,
                "avg_negative": 0.0,
                "avg_neutral": 0.0,
                "dominant_sentiment": "neutral",
                "num_chunks": 0,
            }

        positives = [s["positive"] for s in chunk_scores]
        negatives = [s["negative"] for s in chunk_scores]
        neutrals = [s["neutral"] for s in chunk_scores]

        avg_positive = sum(positives) / n
        avg_negative = sum(negatives) / n
        avg_neutral = sum(neutrals) / n

        net_polarity = sum(p - neg for p, neg in zip(positives, negatives)) / n
        positive_ratio = sum(1 for p, neg in zip(positives, negatives) if p > neg) / n
        negative_ratio = sum(1 for p, neg in zip(positives, negatives) if neg > p) / n

        # Dominant sentiment = label with highest average
        avg_map = {
            "positive": avg_positive,
            "negative": avg_negative,
            "neutral": avg_neutral,
        }
        dominant_sentiment = max(avg_map, key=avg_map.get)  # type: ignore[arg-type]

        return {
            "net_polarity": round(net_polarity, 6),
            "positive_ratio": round(positive_ratio, 6),
            "negative_ratio": round(negative_ratio, 6),
            "avg_positive": round(avg_positive, 6),
            "avg_negative": round(avg_negative, 6),
            "avg_neutral": round(avg_neutral, 6),
            "dominant_sentiment": dominant_sentiment,
            "num_chunks": n,
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def score_document(
    text: str,
    scorer: FinBERTScorer | None = None,
) -> dict:
    """End-to-end document scoring: chunk → score → aggregate.

    Parameters
    ----------
    text : str
        Raw document text (e.g. an MD&A section).
    scorer : FinBERTScorer, optional
        Re-use an existing scorer instance to avoid reloading the model.

    Returns
    -------
    dict
        ``{'aggregate': <aggregated metrics>, 'chunk_scores': [<per-chunk dicts>]}``.
    """
    from src.text_preprocessor import prepare_chunks  # noqa: E402  (lazy to avoid circular)

    if scorer is None:
        scorer = FinBERTScorer()

    chunks = prepare_chunks(text)
    chunk_scores = scorer.score_chunks(chunks)
    aggregate = scorer.aggregate_scores(chunk_scores)

    return {
        "aggregate": aggregate,
        "chunk_scores": chunk_scores,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    test_text = (
        "Revenue increased significantly, driven by strong client activity. "
        "However, provisions for credit losses rose due to a weakening economic outlook. "
        "Overall, management remains cautiously optimistic about future performance."
    )

    logger.info("=== FinBERT Scorer Demo ===")
    result = score_document(test_text)

    print("\n--- Per-Chunk Scores ---")
    for i, cs in enumerate(result["chunk_scores"], 1):
        print(f"  Chunk {i}: {cs}")

    print("\n--- Aggregate ---")
    for key, val in result["aggregate"].items():
        print(f"  {key}: {val}")
