"""Financial-text sentiment scoring.

A ``SentimentScorer`` maps a list of strings -> list of polarity scores in
[-1, 1] (bearish→bullish). Three backends, selected by ``backend=``:

* ``finbert`` — ProsusAI/finbert via 🤗 transformers (recommended; needs torch).
* ``fingpt``  — FinGPT sentiment LoRA; same transformers pipeline contract.
* ``lexicon`` — dependency-free Loughran-McDonald-style word list. Always works,
  used as the automatic fallback when transformers/torch aren't available.

The model is loaded lazily on first ``score()`` and cached on the instance.
"""
from __future__ import annotations

import logging
import re
from typing import List, Sequence

log = logging.getLogger(__name__)

# Compact finance lexicon — not a substitute for FinBERT, but a sane fallback.
_POSITIVE = {
    "beat", "beats", "surge", "surged", "soar", "soared", "rally", "gain", "gains",
    "growth", "record", "upgrade", "upgraded", "outperform", "bullish", "profit",
    "strong", "exceeds", "raises", "raised", "approval", "approved", "wins", "win",
}
_NEGATIVE = {
    "miss", "misses", "missed", "plunge", "plunged", "drop", "fall", "fell", "slump",
    "downgrade", "downgraded", "underperform", "bearish", "loss", "losses", "weak",
    "cut", "cuts", "lawsuit", "probe", "investigation", "recall", "warning", "bankruptcy",
    "default", "fraud", "decline", "declines", "halt", "halted",
}
_NEGATORS = {"not", "no", "never", "without", "fails", "fail", "failed"}
_WORD_RE = re.compile(r"[a-z']+")


class SentimentScorer:
    def __init__(self, backend: str = "finbert", model_name: str | None = None):
        self.backend = backend
        self.model_name = model_name or _default_model(backend)
        self._pipe = None  # lazy transformers pipeline

    def score(self, texts: Sequence[str]) -> List[float]:
        texts = [t or "" for t in texts]
        if not texts:
            return []
        if self.backend in ("finbert", "fingpt"):
            scores = self._score_transformers(texts)
            if scores is not None:
                return scores
            log.warning("transformers backend unavailable; using lexicon fallback")
        return [self._score_lexicon(t) for t in texts]

    # --- transformers (FinBERT / FinGPT) -------------------------------------
    def _score_transformers(self, texts: Sequence[str]) -> List[float] | None:
        if self._pipe is None:
            try:
                from transformers import pipeline
            except ImportError:
                return None
            try:
                self._pipe = pipeline(
                    "text-classification",
                    model=self.model_name,
                    truncation=True,
                    top_k=None,
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
            # row is a list of {label, score} dicts (top_k=None)
            by_label = {d["label"].lower(): d["score"] for d in row}
            pos = by_label.get("positive", 0.0)
            neg = by_label.get("negative", 0.0)
            out.append(float(pos - neg))  # in [-1, 1]
        return out

    # --- lexicon fallback ----------------------------------------------------
    @staticmethod
    def _score_lexicon(text: str) -> float:
        tokens = _WORD_RE.findall(text.lower())
        score = 0
        for i, tok in enumerate(tokens):
            polarity = 1 if tok in _POSITIVE else (-1 if tok in _NEGATIVE else 0)
            if polarity and i > 0 and tokens[i - 1] in _NEGATORS:
                polarity = -polarity
            score += polarity
        hits = sum(1 for t in tokens if t in _POSITIVE or t in _NEGATIVE)
        return score / hits if hits else 0.0


def _default_model(backend: str) -> str:
    return {
        "finbert": "ProsusAI/finbert",
        "fingpt": "FinGPT/fingpt-sentiment_llama2-13b_lora",
        "lexicon": "lexicon",
    }.get(backend, "ProsusAI/finbert")
