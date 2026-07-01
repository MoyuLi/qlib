"""News ingestion.

Returns ``List[NewsItem]`` (see ``base.NewsItem``) for downstream sentiment /
event extraction. Backends, in rough order of data quality:

* ``finnhub``  — company news endpoint, needs ``FINNHUB_API_KEY``.
* ``yfinance`` — free, no key, but sparse and recent-only.

Both are lazily imported and degrade to an empty list rather than raising, so a
missing key / offline run never breaks the pipeline.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

from trading.data.sources.base import NewsItem

log = logging.getLogger(__name__)


def get_news(
    symbols: Iterable[str],
    lookback_days: int = 3,
    source: str = "finnhub",
    limit_per_symbol: int = 50,
) -> List[NewsItem]:
    symbols = [str(s).upper() for s in symbols]
    if source == "finnhub":
        return _get_news_finnhub(symbols, lookback_days, limit_per_symbol)
    if source == "yfinance":
        return _get_news_yfinance(symbols, limit_per_symbol)
    raise ValueError(f"unknown news source: {source!r}")


def _get_news_finnhub(symbols: List[str], lookback_days: int, limit: int) -> List[NewsItem]:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        log.warning("FINNHUB_API_KEY not set; returning no news")
        return []
    try:
        import finnhub
    except ImportError:
        log.warning("finnhub-python not installed (`pip install finnhub-python`)")
        return []

    client = finnhub.Client(api_key=key)
    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=lookback_days)).isoformat()
    end = today.isoformat()

    items: List[NewsItem] = []
    for sym in symbols:
        try:
            raw = client.company_news(sym, _from=start, to=end) or []
        except Exception as e:  # rate limits / bad ticker shouldn't sink the batch
            log.warning("finnhub news failed for %s: %s", sym, e)
            continue
        for art in raw[:limit]:
            ts = art.get("datetime")
            when = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)
            items.append(
                NewsItem(
                    symbol=sym,
                    datetime=when,
                    headline=art.get("headline", "") or "",
                    summary=art.get("summary", "") or "",
                    url=art.get("url", "") or "",
                    source=art.get("source", "finnhub") or "finnhub",
                )
            )
    return items


def _get_news_yfinance(symbols: List[str], limit: int) -> List[NewsItem]:
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed; returning no news")
        return []

    items: List[NewsItem] = []
    for sym in symbols:
        try:
            raw = yf.Ticker(sym).news or []
        except Exception as e:
            log.warning("yfinance news failed for %s: %s", sym, e)
            continue
        for art in raw[:limit]:
            # yfinance nests the payload under "content" in newer versions
            content = art.get("content", art)
            ts = content.get("pubDate") or art.get("providerPublishTime")
            if isinstance(ts, (int, float)):
                when = datetime.fromtimestamp(ts, tz=timezone.utc)
            elif isinstance(ts, str):
                try:
                    when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    when = datetime.now(timezone.utc)
            else:
                when = datetime.now(timezone.utc)
            items.append(
                NewsItem(
                    symbol=sym,
                    datetime=when,
                    headline=content.get("title", "") or art.get("title", "") or "",
                    summary=content.get("summary", "") or "",
                    url=(content.get("canonicalUrl", {}) or {}).get("url", "") or art.get("link", ""),
                    source=(content.get("provider", {}) or {}).get("displayName", "yahoo"),
                )
            )
    return items
