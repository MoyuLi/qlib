"""Pull daily OHLCV from Alpaca's market-data API, write per-symbol CSVs,
then hand off to qlib's scripts/dump_bin.py to build the .bin store.

Using Alpaca for both training data and live execution avoids train/serve
skew from mixing vendors with different adjustment conventions.

    python trading/data/build_us_bin.py --start 2016-01-01 --out raw_us_csv

Then convert with qlib's own tool:

    python scripts/dump_bin.py dump_all \
        --csv_path  ./raw_us_csv \
        --qlib_dir  ~/.qlib/qlib_data/us_live \
        --include_fields open,high,low,close,volume,factor \
        --date_field_name date --symbol_field_name symbol
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

DEFAULT_UNIVERSE_FILE = Path(__file__).parent / "universe.txt"


def load_universe() -> list:
    if DEFAULT_UNIVERSE_FILE.exists():
        return [s.strip() for s in DEFAULT_UNIVERSE_FILE.read_text().splitlines() if s.strip()]
    raise FileNotFoundError(
        f"No universe file at {DEFAULT_UNIVERSE_FILE}. Create it with one symbol per line "
        "(e.g. a liquid S&P 500 / ETF basket) before running this script."
    )


def fetch_symbol(client: StockHistoricalDataClient, symbol: str, start: str, end: str) -> pd.DataFrame:
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="all",  # splits + dividends folded into a factor-adjusted series
    )
    bars = client.get_stock_bars(request).df
    bars = bars.reset_index()
    bars["symbol"] = symbol
    bars = bars.rename(columns={"timestamp": "date"})
    bars["date"] = pd.to_datetime(bars["date"]).dt.tz_localize(None).dt.date
    # qlib expects a `factor` column; with adjustment="all" the OHLCV is
    # already adjusted, so factor=1.0 is consistent here.
    bars["factor"] = 1.0
    return bars[["date", "symbol", "open", "high", "low", "close", "volume", "factor"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default="raw_us_csv")
    args = parser.parse_args()

    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    client = StockHistoricalDataClient(api_key, secret_key)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for symbol in load_universe():
        df = fetch_symbol(client, symbol, args.start, args.end)
        df.to_csv(out_dir / f"{symbol}.csv", index=False)
        print(f"Wrote {len(df)} rows for {symbol}")


if __name__ == "__main__":
    main()
