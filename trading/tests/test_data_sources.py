import numpy as np
import pandas as pd

from trading.data.sources.base import (
    PRICE_COLUMNS,
    normalize_price_frame,
    read_universe,
    to_qlib_frame,
)
from trading.strategy import factors as factor_lib


def test_normalize_renames_and_dedups():
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "Ticker": ["aapl", "aapl", "aapl"],
            "Adj Close": [10.0, 11.0, 12.0],  # dup date -> keep last (11.0)
            "Volume": [100, 200, 300],
        }
    )
    out = normalize_price_frame(raw)
    assert list(out.columns) == PRICE_COLUMNS
    assert len(out) == 2
    first = out[out["datetime"] == pd.Timestamp("2024-01-01")].iloc[0]
    assert first["close"] == 11.0
    assert first["symbol"] == "AAPL"


def test_normalize_empty_returns_schema():
    out = normalize_price_frame(pd.DataFrame())
    assert list(out.columns) == PRICE_COLUMNS
    assert out.empty


def test_to_qlib_frame_multiindex_and_dollar_cols():
    raw = pd.DataFrame(
        {"date": ["2024-01-01"], "symbol": ["AAPL"], "close": [10.0], "volume": [100]}
    )
    q = to_qlib_frame(raw)
    assert q.index.names == ["instrument", "datetime"]
    assert "$close" in q.columns


def test_read_universe(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("# header\nAAPL\nmsft  # inline\n\nAAPL\n")
    assert read_universe(p) == ["AAPL", "MSFT"]


def _synthetic_prices():
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    rows = []
    for sym, drift in [("UP", 0.002), ("DOWN", -0.002)]:
        price = 100.0
        for d in dates:
            price *= 1 + drift
            rows.append({"datetime": d, "symbol": sym, "open": price, "high": price,
                         "low": price, "close": price, "volume": 1_000})
    return pd.DataFrame(rows)


def test_momentum_factor_ranks_uptrend_above_downtrend():
    prices = _synthetic_prices()
    mom = factor_lib.momentum(prices)
    assert mom["UP"] > mom["DOWN"]


def test_compute_factors_returns_named_series():
    facs = factor_lib.compute_factors(_synthetic_prices())
    assert set(facs) == {"momentum", "reversal", "low_vol"}
    for name, s in facs.items():
        assert isinstance(s, pd.Series)
