"""SEC filings ingestion.

Lists recent filings (8-K, 10-K, 10-Q, …) per symbol straight from the SEC
EDGAR full-text/submissions API — no key required, just a descriptive
``User-Agent`` per SEC fair-access policy (set ``SEC_USER_AGENT``).

Returns ``List[Filing]``. The AI layer can then fetch + summarize the document
behind ``Filing.url`` for event extraction. We deliberately only return
*references* here to keep ingestion cheap and avoid pulling multi-MB documents
the strategy may never look at.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Iterable, List, Optional

from trading.data.sources.base import Filing

log = logging.getLogger(__name__)

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


def _user_agent() -> str:
    # SEC requires a contactable UA; fall back to a clearly-labeled default.
    return os.environ.get("SEC_USER_AGENT", "qlib-trading research contact@example.com")


def _ticker_to_cik(session, symbols: List[str]) -> dict:
    resp = session.get(_TICKER_MAP_URL, timeout=20)
    resp.raise_for_status()
    table = resp.json()
    by_ticker = {row["ticker"].upper(): int(row["cik_str"]) for row in table.values()}
    return {s: by_ticker[s] for s in symbols if s in by_ticker}


def get_filings(
    symbols: Iterable[str],
    forms: Optional[Iterable[str]] = None,
    limit_per_symbol: int = 10,
) -> List[Filing]:
    """Recent EDGAR filings per symbol, newest first.

    ``forms`` filters by form type (default: 8-K / 10-K / 10-Q). Symbols that
    don't map to a CIK are skipped with a warning.
    """
    symbols = [str(s).upper() for s in symbols]
    forms = {f.upper() for f in (forms or ["8-K", "10-K", "10-Q"])}

    try:
        import requests
    except ImportError:
        log.warning("requests not installed; cannot reach EDGAR")
        return []

    session = requests.Session()
    session.headers.update({"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"})

    try:
        cik_map = _ticker_to_cik(session, symbols)
    except Exception as e:
        log.warning("EDGAR ticker->CIK lookup failed: %s", e)
        return []

    out: List[Filing] = []
    for sym, cik in cik_map.items():
        try:
            resp = session.get(_SUBMISSIONS_URL.format(cik=cik), timeout=20)
            resp.raise_for_status()
            recent = resp.json().get("filings", {}).get("recent", {})
        except Exception as e:
            log.warning("EDGAR submissions failed for %s: %s", sym, e)
            continue

        types = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])

        count = 0
        for i, form in enumerate(types):
            if count >= limit_per_symbol:
                break
            if form.upper() not in forms:
                continue
            acc = accessions[i] if i < len(accessions) else ""
            acc_nodash = acc.replace("-", "")
            doc = docs[i] if i < len(docs) else ""
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
                if acc_nodash and doc
                else ""
            )
            when = (
                datetime.fromisoformat(dates[i]) if i < len(dates) and dates[i] else datetime.utcnow()
            )
            out.append(
                Filing(symbol=sym, datetime=when, form_type=form.upper(), url=url, accession_no=acc)
            )
            count += 1

    out.sort(key=lambda f: f.datetime, reverse=True)
    return out
