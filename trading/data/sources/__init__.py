"""Market-data ingestion layer.

Each module here is a thin adapter around an open-source data source. They all
normalize into the small set of dataclasses / frame conventions defined in
``base.py`` so the AI and strategy layers never depend on a specific vendor.

Heavy / optional dependencies (yfinance, ccxt, sec-edgar-downloader, fredapi,
alpaca) are imported lazily *inside* the function that needs them, so importing
this package never forces them to be installed.
"""
