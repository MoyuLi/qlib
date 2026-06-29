"""Macroeconomic data ingestion (FRED).

Pulls macro series from the St. Louis Fed (FRED) via ``fredapi`` (needs
``FRED_API_KEY``) or, as a keyless fallback, ``pandas_datareader``.

Returns a wide frame indexed by date, one column per series id. A small,
opinionated default basket (rates, curve slope, credit spread, inflation
expectations, USD, VIX) is provided as ``DEFAULT_SERIES`` — these are regime
features the strategy layer can use to scale gross exposure or gate signals.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, Optional

import pandas as pd

log = logging.getLogger(__name__)

# id -> human label. Sane macro-regime starter basket.
DEFAULT_SERIES: Dict[str, str] = {
    "DGS3MO": "ust_3m",
    "DGS2": "ust_2y",
    "DGS10": "ust_10y",
    "T10Y2Y": "curve_10y2y",
    "BAMLH0A0HYM2": "hy_oas",          # high-yield credit spread
    "T10YIE": "breakeven_10y",         # 10y inflation expectations
    "DTWEXBGS": "usd_broad",           # broad trade-weighted USD
    "VIXCLS": "vix",
}


def get_macro(
    series: Optional[Iterable[str]] = None,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    source: str = "fredapi",
) -> pd.DataFrame:
    """Wide macro frame (date index, one column per series, label-renamed)."""
    ids = list(series) if series is not None else list(DEFAULT_SERIES)
    if source == "fredapi":
        df = _get_fredapi(ids, start, end)
    elif source == "pandas_datareader":
        df = _get_pdr(ids, start, end)
    else:
        raise ValueError(f"unknown macro source: {source!r}")

    if df.empty:
        return df
    return df.rename(columns=DEFAULT_SERIES).sort_index()


def _get_fredapi(ids, start, end) -> pd.DataFrame:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        log.warning("FRED_API_KEY not set; falling back to pandas_datareader")
        return _get_pdr(ids, start, end)
    try:
        from fredapi import Fred
    except ImportError:
        log.warning("fredapi not installed; falling back to pandas_datareader")
        return _get_pdr(ids, start, end)

    fred = Fred(api_key=key)
    cols = {}
    for sid in ids:
        try:
            cols[sid] = fred.get_series(sid, observation_start=start, observation_end=end)
        except Exception as e:
            log.warning("FRED series %s failed: %s", sid, e)
    return pd.DataFrame(cols)


def _get_pdr(ids, start, end) -> pd.DataFrame:
    try:
        import pandas_datareader.data as web
    except ImportError:
        log.warning("pandas_datareader not installed; returning empty macro frame")
        return pd.DataFrame()
    try:
        return web.DataReader(list(ids), "fred", start, end)
    except Exception as e:
        log.warning("pandas_datareader FRED pull failed: %s", e)
        return pd.DataFrame()
