# `trading/` — research → AI → strategy → execution bridge

A comprehensive, layered quant workflow built on top of qlib. Each layer is a
set of thin, **independently testable** adapters; heavy/optional libraries
(transformers, FinRL, yfinance, …) are imported lazily so the whole thing runs
end-to-end on the bundled US data with **no API keys**, and gets richer as you
add them.

```
 Market Data            AI Layer              Strategy Layer        Backtest / Execution
 ───────────            ────────              ──────────────        ────────────────────
 prices    ┐            sentiment ┐           factors  ┐            qlib backtest
 news      │  ──────►   events    │  ──────►  ml_models│  ──────►   ───────────────
 filings   │            signals   ┘           rules    │            Alpaca (paper/live)
 options   │           (FinBERT/FinGPT,       rl(FinRL)│            risk gate + reconcile
 macro     ┘            LLM extraction)       combine  ┘
```

## Layers

### Market Data — `trading/data/sources/`
Normalized adapters; all price backends emit the same long OHLCV frame (see
`base.py`), all text sources emit `NewsItem` / `Filing` dataclasses.

| Module | Source(s) | Needs |
| --- | --- | --- |
| `prices.py` | yfinance · Alpaca · qlib bundle | — / Alpaca keys |
| `news.py` | Finnhub · yfinance | `FINNHUB_API_KEY` (yfinance keyless) |
| `fundamentals.py` | SEC EDGAR filings | `SEC_USER_AGENT` |
| `options.py` | yfinance chains → put/call & ATM-IV features | — |
| `macro.py` | FRED (rates, curve, credit, USD, VIX) | `FRED_API_KEY` (pdr fallback) |

### AI Layer — `trading/ai/`
- `sentiment.py` — `SentimentScorer`: FinBERT via transformers (primary),
  **VADER** (`vaderSentiment`) as the no-torch fallback → polarity in [-1, 1].
  No hand-maintained word lists.
- `events.py` — `EventExtractor` (`auto`: Claude if `ANTHROPIC_API_KEY`, else
  GPT if `OPENAI_API_KEY`, else keyword rules) → structured `ExtractedEvent`s.
- `signals.py` — collapse per-article scores into a per-symbol cross-section
  with **recency decay** and **confidence weighting**.

### Strategy Layer — `trading/strategy/`
- `factors.py` — momentum (ROC), reversal (RSI), low-vol (Bollinger width),
  computed with the `ta` library (not hand-rolled).
- `ml_models.py` — wrap a trained qlib model's latest cross-section as a signal.
- `combine.py` — z-score + winsorize + weighted blend → one ranked score.
- `rules.py` — hard overlays: block/allow lists, liquidity floor, halts,
  long-only, top-N, and a VIX-driven risk-regime scalar.
- `rl.py` — optional FinRL portfolio agent (lazy; disabled if FinRL absent).

### Execution — `trading/live/` (pre-existing)
`run_live.py` reconciles against the broker, sizes target weights, diffs to
delta orders, and submits each through the pure-logic `risk.py` gate. Defaults
to **Alpaca paper**; the non-paper guard must be flipped deliberately.

## Run it

```bash
# 1. Build the blended signal (ML + AI + factors → scores_latest.parquet)
python trading/serving/build_signal.py trading/configs/alpha158_lgb_bundled.yaml

# 2. Backtest the ML leg (IC / portfolio analysis)
python trading/research/run_backtest.py trading/configs/alpha158_lgb_bundled.yaml

# 3. Live bridge (paper) — consumes scores_latest.parquet unchanged
python trading/live/run_live.py
```

`build_signal.py` is the comprehensive replacement for the ML-only
`generate_scores.py`; it writes the same `scores_latest.parquet` contract, so
the live bridge needs no changes.

## Dependencies
Core (qlib, pandas, alpaca-py) only. Optional legs:
`pip install -r trading/requirements-extras.txt`. Missing lib or key → that leg
is simply absent from the blend.

## Tests
```bash
python -m pytest trading/tests/ -q
```
Pure-logic layers (combine, rules, signal aggregation, sentiment lexicon, data
normalization, factors) are fully covered without network or model downloads.
```
