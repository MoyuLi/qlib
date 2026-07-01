"""Classical price/volume factors, computed with the ``ta`` library.

Rather than hand-rolling indicator math, each factor wraps a well-known
indicator from the open-source ``ta`` package (Bukosabino) and reduces it to a
single cross-sectional value per symbol (the latest observation). Output is a
``pd.Series`` indexed by symbol, ready for ``strategy.combine``.

Input is the long OHLCV frame from ``data.sources.base``. If ``ta`` isn't
installed the factors return empty (the blend simply runs without them) — we do
not silently re-implement the indicators.

Note: qlib's Alpha158 handler (used by the ML leg) already engineers a large
momentum/volatility factor set. These ``ta`` factors are here for an explicit,
interpretable blend leg and for non-qlib (e.g. vectorbt) workflows.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict

import pandas as pd

log = logging.getLogger(__name__)


def _per_symbol_last(prices: pd.DataFrame, fn: Callable[[pd.DataFrame], pd.Series], name: str) -> pd.Series:
    """Apply ``fn`` (a ``ta`` indicator over one symbol's OHLCV) to every symbol
    and collect the latest value into a cross-sectional Series."""
    if prices is None or prices.empty:
        return pd.Series(dtype="float64", name=name)
    out: Dict[str, float] = {}
    for sym, g in prices.sort_values("datetime").groupby("symbol"):
        try:
            series = fn(g)
            val = series.dropna().iloc[-1] if series is not None and not series.dropna().empty else None
        except Exception as e:  # short history / degenerate symbol -> skip
            log.debug("factor %s failed for %s: %s", name, sym, e)
            val = None
        if val is not None and pd.notna(val):
            out[str(sym)] = float(val)
    return pd.Series(out, name=name).sort_index()


def momentum(prices: pd.DataFrame, window: int = 126) -> pd.Series:
    """Rate of change over ~6 months — ``ta.momentum.ROCIndicator``."""
    from ta.momentum import ROCIndicator

    return _per_symbol_last(
        prices, lambda g: ROCIndicator(close=g["close"], window=window).roc(), "momentum"
    )


def short_reversal(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    """Mean-reversion: ``50 - RSI`` (oversold names score high) — ``ta.momentum.RSIIndicator``."""
    from ta.momentum import RSIIndicator

    return _per_symbol_last(
        prices, lambda g: 50.0 - RSIIndicator(close=g["close"], window=window).rsi(), "reversal"
    )


def low_volatility(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """Low-vol anomaly: negative Bollinger band width — ``ta.volatility.BollingerBands``."""
    from ta.volatility import BollingerBands

    return _per_symbol_last(
        prices,
        lambda g: -BollingerBands(close=g["close"], window=window).bollinger_wband(),
        "low_vol",
    )


def compute_factors(prices: pd.DataFrame) -> Dict[str, pd.Series]:
    """All factors as a {name: Series} dict ready for ``combine``."""
    try:
        import ta  # noqa: F401
    except ImportError:
        log.warning("`ta` not installed (`pip install ta`); skipping factor legs")
        return {}
    return {
        "momentum": momentum(prices),
        "reversal": short_reversal(prices),
        "low_vol": low_volatility(prices),
    }
