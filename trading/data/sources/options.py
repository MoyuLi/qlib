"""Options-chain ingestion.

Pulls the live option chain per symbol via ``yfinance`` and derives a couple of
cheap, widely-used sentiment features the strategy layer can consume:

* ``put_call_oi_ratio``  — put open-interest / call open-interest (>1 = bearish).
* ``put_call_vol_ratio`` — put volume / call volume (intraday flow).
* ``iv_atm``             — rough at-the-money implied vol (mean of nearest-strike IVs).

Returns a tidy frame indexed by symbol. Greeks/full surfaces are intentionally
out of scope here — add a dedicated vendor (ORATS, Polygon) when needed.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import pandas as pd

log = logging.getLogger(__name__)

_FEATURE_COLUMNS = ["symbol", "expiry", "put_call_oi_ratio", "put_call_vol_ratio", "iv_atm"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_FEATURE_COLUMNS)


def get_option_features(symbols: Iterable[str], max_expiries: int = 1) -> pd.DataFrame:
    """Per-symbol option-sentiment features for the nearest ``max_expiries``."""
    symbols = [str(s).upper() for s in symbols]
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed; returning empty option features")
        return _empty()

    rows = []
    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            expiries = list(tk.options)[:max_expiries]
        except Exception as e:
            log.warning("option expiries failed for %s: %s", sym, e)
            continue

        spot = _spot(tk)
        for expiry in expiries:
            try:
                chain = tk.option_chain(expiry)
            except Exception as e:
                log.warning("option chain failed for %s %s: %s", sym, expiry, e)
                continue
            calls, puts = chain.calls, chain.puts
            rows.append(
                {
                    "symbol": sym,
                    "expiry": expiry,
                    "put_call_oi_ratio": _ratio(puts, calls, "openInterest"),
                    "put_call_vol_ratio": _ratio(puts, calls, "volume"),
                    "iv_atm": _atm_iv(calls, puts, spot),
                }
            )

    return pd.DataFrame(rows, columns=_FEATURE_COLUMNS) if rows else _empty()


def _spot(tk) -> Optional[float]:
    try:
        fast = getattr(tk, "fast_info", None)
        if fast and fast.get("lastPrice"):
            return float(fast["lastPrice"])
    except Exception:
        pass
    return None


def _ratio(puts: pd.DataFrame, calls: pd.DataFrame, col: str) -> float:
    p = float(puts[col].fillna(0).sum()) if col in puts else 0.0
    c = float(calls[col].fillna(0).sum()) if col in calls else 0.0
    return p / c if c > 0 else float("nan")


def _atm_iv(calls: pd.DataFrame, puts: pd.DataFrame, spot: Optional[float]) -> float:
    if spot is None:
        return float("nan")
    ivs = []
    for df in (calls, puts):
        if "strike" not in df or "impliedVolatility" not in df or df.empty:
            continue
        idx = (df["strike"] - spot).abs().idxmin()
        iv = df.loc[idx, "impliedVolatility"]
        if pd.notna(iv):
            ivs.append(float(iv))
    return sum(ivs) / len(ivs) if ivs else float("nan")
