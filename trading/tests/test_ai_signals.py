from datetime import datetime, timedelta, timezone

import pandas as pd

from trading.ai.sentiment import SentimentScorer
from trading.ai.signals import events_to_signal, news_sentiment_signal
from trading.data.sources.base import ExtractedEvent, NewsItem


def test_lexicon_scorer_directionality():
    sc = SentimentScorer(backend="lexicon")
    pos, neg = sc.score(["Company beats earnings, shares surge", "Company misses, stock plunges"])
    assert pos > 0 > neg


def test_lexicon_handles_negation():
    sc = SentimentScorer(backend="lexicon")
    (s,) = sc.score(["the company did not beat expectations"])
    assert s <= 0


def test_news_sentiment_signal_recency_weighting():
    now = datetime.now(timezone.utc)
    news = [
        # fresh bad news should dominate stale good news for AAPL
        NewsItem("AAPL", now, "AAPL plunges on weak guidance, downgrade"),
        NewsItem("AAPL", now - timedelta(days=10), "AAPL beats, record profit, surge"),
        NewsItem("MSFT", now, "MSFT wins major contract, shares gain"),
    ]
    sig = news_sentiment_signal(news, SentimentScorer(backend="lexicon"), half_life_days=2.0)
    assert sig["AAPL"] < 0
    assert sig["MSFT"] > 0


def test_events_to_signal_confidence_weighting():
    now = datetime.now(timezone.utc)
    events = [
        ExtractedEvent("AAPL", now, "legal", polarity=-1.0, confidence=0.9),
        ExtractedEvent("AAPL", now, "product", polarity=1.0, confidence=0.1),
    ]
    sig = events_to_signal(events)
    assert sig["AAPL"] < 0  # high-confidence negative outweighs low-confidence positive


def test_empty_inputs_return_empty_series():
    assert news_sentiment_signal([]).empty
    assert events_to_signal([]).empty
