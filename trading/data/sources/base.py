"""Normalized data contracts shared by every market-data source.

Conventions (kept deliberately close to qlib so frames drop straight into a
handler or a vectorbt backtest):

* **Price frames** are tidy/long: columns ``[datetime, symbol, open, high,
  low, close, volume]``, one row per (symbol, day). ``datetime`` is a
  tz-naive ``pd.Timestamp`` normalized to midnight (daily bars).
* **News / filings / events** are lists of frozen dataclasses, each carrying
  the ``symbol`` it pertains to and a ``datetime`` so the AI layer can build
  point-in-time signals without look-ahead.

Nothing here imports a vendor SDK; this module is pure and cheap to import.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime as _dt
from typing import List, Optional

import pandas as pd

PRICE_COLUMNS = ["datetime", "symbol", "open", "high", "low", "close", "volume"]


def empty_price_frame() -> pd.DataFrame:
    """An empty frame with the canonical price schema and dtypes."""
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in PRICE_COLUMNS}).astype(
        {"datetime": "datetime64[ns]", "symbol": "object"}
    )


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame to the canonical schema.

    Drops unknown columns, lower-cases/renames common variants, enforces dtypes,
    sorts, and de-duplicates on (symbol, datetime) keeping the last observation.
    Missing optional columns are tolerated and filled with NaN.
    """
    if df is None or df.empty:
        return empty_price_frame()

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {
        "date": "datetime",
        "timestamp": "datetime",
        "time": "datetime",
        "ticker": "symbol",
        "instrument": "symbol",
        "adj close": "close",
        "adj_close": "close",
        "adjclose": "close",
        "vol": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for col in PRICE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[PRICE_COLUMNS]
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.normalize()
    df["symbol"] = df["symbol"].astype("string").str.upper()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "symbol"])
        .sort_values(["symbol", "datetime"])
        .drop_duplicates(["symbol", "datetime"], keep="last")
        .reset_index(drop=True)
    )
    return df


def to_qlib_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex a canonical price frame to qlib's ``(instrument, datetime)``
    MultiIndex with ``$``-prefixed feature columns, so it can back a custom
    ``DataHandler`` / ``StaticDataLoader`` without further munging."""
    df = normalize_price_frame(df)
    out = df.rename(
        columns={
            "symbol": "instrument",
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
        }
    )
    return out.set_index(["instrument", "datetime"]).sort_index()


@dataclass(frozen=True)
class NewsItem:
    """One headline/article tied to a symbol at a point in time."""

    symbol: str
    datetime: _dt
    headline: str
    summary: str = ""
    url: str = ""
    source: str = ""


@dataclass(frozen=True)
class Filing:
    """An SEC filing reference (8-K, 10-K, 10-Q, …)."""

    symbol: str
    datetime: _dt
    form_type: str
    url: str = ""
    accession_no: str = ""


@dataclass(frozen=True)
class ExtractedEvent:
    """A structured event distilled from text by the AI layer.

    ``polarity`` is in [-1, 1] (bearish→bullish) and ``confidence`` in [0, 1].
    """

    symbol: str
    datetime: _dt
    event_type: str
    polarity: float = 0.0
    confidence: float = 0.0
    evidence: str = ""


def read_universe(path) -> List[str]:
    """Read ``trading/data/universe.txt``-style files: one symbol per line,
    ``#`` comments and blanks ignored."""
    from pathlib import Path

    symbols: List[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip().upper()
        if line:
            symbols.append(line)
    # stable de-dup
    seen = set()
    return [s for s in symbols if not (s in seen or seen.add(s))]
