"""Financial-text sentiment scoring.

A ``SentimentScorer`` maps a list of strings -> list of polarity scores in
[-1, 1] (bearish→bullish). Backends, selected by ``backend=``:

* ``finbert`` — ProsusAI/finbert via 🤗 transformers (recommended; needs torch).
* ``vader``   — NLTK VADER (``vaderSentiment`` package): a lexicon+rules model
  that natively handles negation, intensifiers and punctuation. Pure-Python,
  no torch — the automatic fallback when transformers isn't available.

(A generative FinGPT LoRA backend was intentionally left out: it needs a gated
13B base model + ``peft`` + a GPU, which is impractical here and gives no gain
over FinBERT for short-headline sentiment.)

No sentiment lexicon is hand-maintained here; the light path defers entirely to
VADER. The model is loaded lazily on first ``score()`` and cached.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

log = logging.getLogger(__name__)


class SentimentScorer:
    def __init__(self, backend: str = "finbert", model_name: str | None = None):
        self.backend = backend
        self.model_name = model_name or _default_model(backend)
        self._pipe = None     # lazy transformers pipeline
        self._vader = None    # lazy VADER analyzer

    def score(self, texts: Sequence[str]) -> List[float]:
        texts = [t or "" for t in texts]
        if not texts:
            return []
        if self.backend == "finbert":
            scores = self._score_transformers(texts)
            if scores is not None:
                return scores
            log.warning("transformers backend unavailable; falling back to VADER")
        return [self._score_vader(t) for t in texts]

    # --- transformers (FinBERT / FinGPT) -------------------------------------
    def _score_transformers(self, texts: Sequence[str]) -> List[float] | None:
        if self._pipe is None:
            try:
                from transformers import pipeline
            except ImportError:
                return None
            try:
                self._pipe = pipeline(
                    "text-classification", model=self.model_name, truncation=True, top_k=None
                )
            except Exception as e:
                log.warning("failed to load %s: %s", self.model_name, e)
                self._pipe = None
                return None
        try:
            raw = self._pipe(list(texts))
        except Exception as e:
            log.warning("sentiment inference failed: %s", e)
            return None

        out = []
        for row in raw:
            by_label = {d["label"].lower(): d["score"] for d in row}
            out.append(float(by_label.get("positive", 0.0) - by_label.get("negative", 0.0)))
        return out

    # --- VADER fallback ------------------------------------------------------
    def _score_vader(self, text: str) -> float:
        analyzer = self._get_vader()
        if analyzer is None:
            return 0.0  # no model available -> neutral, never crash
        return float(analyzer.polarity_scores(text)["compound"])  # already in [-1, 1]

    def _get_vader(self):
        if self._vader is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

                self._vader = SentimentIntensityAnalyzer()
            except ImportError:
                log.warning("vaderSentiment not installed; sentiment falls back to neutral")
                self._vader = False
        return self._vader or None


def _default_model(backend: str) -> str:
    return {
        "finbert": "ProsusAI/finbert",
        "vader": "vader",
    }.get(backend, "ProsusAI/finbert")
