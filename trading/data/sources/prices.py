"""Price-history ingestion.

Three interchangeable backends, all returning the canonical long price frame
(see ``base.normalize_price_frame``):

* ``yfinance``      — free daily OHLCV, good default for research/backfill.
* ``alpaca``        — uses the same creds as the live bridge; intraday-capable.
* ``qlib``          — pull straight from an initialized qlib bundle.

Pick with ``get_prices(..., source="yfinance")`` or call the backend directly.
All heavy imports are lazy so this module imports with zero extra deps.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional

import pandas as pd

from trading.data.sources.base import empty_price_frame, normalize_price_frame

log = logging.getLogger(__name__)


def get_prices(
    symbols: Iterable[str],
    start: str,
    end: Optional[str] = None,
    source: str = "yfinance",
) -> pd.DataFrame:
    """Dispatch to a price backend and return a canonical price frame."""
    symbols = [str(s).upper() for s in symbols]
    if source == "yfinance":
        return get_prices_yfinance(symbols, start, end)
    if source == "alpaca":
        return get_prices_alpaca(symbols, start, end)
    if source == "qlib":
        return get_prices_qlib(symbols, start, end)
    raise ValueError(f"unknown price source: {source!r}")


def get_prices_yfinance(symbols: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed (`pip install yfinance`); returning empty frame")
        return empty_price_frame()

    raw = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw is None or raw.empty:
        return empty_price_frame()

    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        for sym in symbols:
            if sym not in raw.columns.get_level_values(0):
                continue
            sub = raw[sym].reset_index().assign(symbol=sym)
            frames.append(sub)
    else:  # single symbol -> flat columns
        frames.append(raw.reset_index().assign(symbol=symbols[0]))

    if not frames:
        return empty_price_frame()
    return normalize_price_frame(pd.concat(frames, ignore_index=True))


def get_prices_alpaca(symbols: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        log.warning("alpaca-py not installed; returning empty frame")
        return empty_price_frame()

    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        log.warning("ALPACA_API_KEY / ALPACA_SECRET_KEY not set; returning empty frame")
        return empty_price_frame()

    client = StockHistoricalDataClient(key, secret)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end) if end else None,
    )
    bars = client.get_stock_bars(req)
    df = bars.df  # MultiIndex (symbol, timestamp)
    if df is None or df.empty:
        return empty_price_frame()
    df = df.reset_index().rename(columns={"timestamp": "datetime"})
    return normalize_price_frame(df)


def get_prices_qlib(symbols: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Pull OHLCV from an *already initialized* qlib bundle (caller runs
    ``qlib.init(...)`` first)."""
    try:
        from qlib.data import D
    except ImportError:
        log.warning("qlib not importable; returning empty frame")
        return empty_price_frame()

    fields = ["$open", "$high", "$low", "$close", "$volume"]
    df = D.features(symbols, fields, start_time=start, end_time=end, freq="day")
    if df is None or df.empty:
        return empty_price_frame()
    df = df.rename(columns={f: f.lstrip("$") for f in fields})
    df = df.reset_index().rename(columns={"instrument": "symbol"})
    return normalize_price_frame(df)
