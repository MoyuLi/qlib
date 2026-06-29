"""Classical price/volume factors computed from a canonical OHLCV frame.

These are deliberately simple, well-known cross-sectional factors so the blend
has something orthogonal to the ML score and the AI sentiment. Each returns a
``pd.Series`` indexed by symbol, evaluated as-of the last available date.

Input is the long price frame from ``data.sources.base`` (columns
``[datetime, symbol, open, high, low, close, volume]``).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _wide_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Pivot to a (datetime x symbol) close-price matrix."""
    if prices is None or prices.empty:
        return pd.DataFrame()
    return prices.pivot_table(index="datetime", columns="symbol", values="close").sort_index()


def momentum(prices: pd.DataFrame, lookback: int = 126, skip: int = 21) -> pd.Series:
    """12-1 style momentum: return over ``lookback`` days, skipping the most
    recent ``skip`` days to avoid short-term reversal."""
    close = _wide_close(prices)
    if len(close) <= lookback + 1:
        return pd.Series(dtype="float64", name="momentum")
    past = close.shift(skip)
    ret = past.iloc[-1] / past.iloc[-1 - (lookback - skip)] - 1.0
    return ret.dropna().rename("momentum")


def short_reversal(prices: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Negative of recent short-horizon return (mean-reversion factor)."""
    close = _wide_close(prices)
    if len(close) <= lookback:
        return pd.Series(dtype="float64", name="reversal")
    ret = close.iloc[-1] / close.iloc[-1 - lookback] - 1.0
    return (-ret).dropna().rename("reversal")


def low_volatility(prices: pd.DataFrame, lookback: int = 63) -> pd.Series:
    """Negative of trailing daily-return volatility (low-vol anomaly)."""
    close = _wide_close(prices)
    if len(close) <= lookback + 1:
        return pd.Series(dtype="float64", name="low_vol")
    rets = close.pct_change().iloc[-lookback:]
    vol = rets.std(ddof=0)
    return (-vol).dropna().rename("low_vol")


def compute_factors(prices: pd.DataFrame) -> Dict[str, pd.Series]:
    """Convenience: all factors as a {name: Series} dict ready for ``combine``."""
    return {
        "momentum": momentum(prices),
        "reversal": short_reversal(prices),
        "low_vol": low_volatility(prices),
    }
