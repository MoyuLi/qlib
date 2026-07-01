"""Aggregate AI outputs into a cross-sectional, per-symbol alpha signal.

The strategy layer wants one number per symbol on a given day. This module
collapses many per-article sentiment/event scores into that, with two pieces of
quant hygiene baked in:

* **Recency decay** — older articles count less (exponential half-life).
* **Confidence weighting** — low-confidence LLM extractions count less.

Returns a ``pd.Series`` indexed by symbol (the raw aggregate). Cross-sectional
standardization is left to ``strategy.combine.zscore`` so all signals are scaled
the same way before blending.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import List, Optional, Sequence

import pandas as pd

from trading.ai.events import EventExtractor
from trading.ai.sentiment import SentimentScorer
from trading.data.sources.base import ExtractedEvent, NewsItem

log = logging.getLogger(__name__)


def _decay_weight(when: datetime, asof: datetime, half_life_days: float) -> float:
    if when is None:
        return 1.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (asof - when).total_seconds() / 86400.0)
    return 0.5 ** (age_days / half_life_days) if half_life_days > 0 else 1.0


def news_sentiment_signal(
    news: Sequence[NewsItem],
    scorer: Optional[SentimentScorer] = None,
    half_life_days: float = 2.0,
    asof: Optional[datetime] = None,
) -> pd.Series:
    """Recency-weighted average headline sentiment per symbol."""
    if not news:
        return pd.Series(dtype="float64", name="news_sentiment")
    scorer = scorer or SentimentScorer()
    asof = asof or datetime.now(timezone.utc)

    texts = [f"{n.headline}. {n.summary}".strip() for n in news]
    scores = scorer.score(texts)

    num: dict = {}
    den: dict = {}
    for n, s in zip(news, scores):
        w = _decay_weight(n.datetime, asof, half_life_days)
        num[n.symbol] = num.get(n.symbol, 0.0) + w * s
        den[n.symbol] = den.get(n.symbol, 0.0) + w
    agg = {sym: num[sym] / den[sym] for sym in num if den[sym] > 0}
    return pd.Series(agg, name="news_sentiment").sort_index()


def event_signal(
    news: Sequence[NewsItem],
    extractor: Optional[EventExtractor] = None,
    half_life_days: float = 5.0,
    asof: Optional[datetime] = None,
) -> pd.Series:
    """Confidence- and recency-weighted event polarity per symbol."""
    if not news:
        return pd.Series(dtype="float64", name="events")
    extractor = extractor or EventExtractor()
    events = extractor.extract(news)
    return events_to_signal(events, half_life_days=half_life_days, asof=asof)


def events_to_signal(
    events: Sequence[ExtractedEvent],
    half_life_days: float = 5.0,
    asof: Optional[datetime] = None,
) -> pd.Series:
    if not events:
        return pd.Series(dtype="float64", name="events")
    asof = asof or datetime.now(timezone.utc)
    num: dict = {}
    den: dict = {}
    for e in events:
        w = _decay_weight(e.datetime, asof, half_life_days) * max(e.confidence, 0.0)
        if w <= 0:
            continue
        num[e.symbol] = num.get(e.symbol, 0.0) + w * e.polarity
        den[e.symbol] = den.get(e.symbol, 0.0) + w
    agg = {sym: num[sym] / den[sym] for sym in num if den[sym] > 0}
    return pd.Series(agg, name="events").sort_index()
