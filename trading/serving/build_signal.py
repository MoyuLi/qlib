"""End-to-end signal builder: Market Data -> AI -> Strategy -> scores_latest.

This is the comprehensive replacement for the ML-only ``generate_scores.py``.
It blends three orthogonal legs into one ranked score and writes the same
``serving/scores_latest.parquet`` contract the live bridge already consumes, so
``run_live.py`` needs no changes:

    ML model score   (qlib)            ─┐
    News sentiment   (FinBERT/FinGPT)  ─┼─► z-score + weighted blend ─► rules ─► scores
    Price factors    (mom/rev/low-vol) ─┘        (strategy.combine)   (overlay)

Every external leg degrades gracefully (missing key/lib/data -> that leg is
simply absent from the blend), so this runs end-to-end on the bundled US data
with no API keys, and gets richer as you add them.

    python trading/serving/build_signal.py trading/configs/alpha158_lgb_bundled.yaml
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from trading.ai.signals import event_signal, news_sentiment_signal
from trading.data.sources.base import read_universe
from trading.data.sources.macro import get_macro
from trading.data.sources.news import get_news
from trading.data.sources.prices import get_prices
from trading.strategy import factors as factor_lib
from trading.strategy.combine import BlendConfig, combine_signals
from trading.strategy.rules import RuleConfig, apply_rules, regime_scalar_from_macro

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("build_signal")

OUT_PATH = Path(__file__).parent / "scores_latest.parquet"
UNIVERSE_PATH = Path(__file__).parent.parent / "data" / "universe.txt"

# How much each leg counts in the blend. Tune via backtest; ML leads by default.
DEFAULT_WEIGHTS = {
    "ml": 1.0,
    "news_sentiment": 0.4,
    "events": 0.3,
    "momentum": 0.3,
    "reversal": 0.1,
    "low_vol": 0.1,
}


def _ml_scores(config: dict) -> pd.Series:
    """Train+predict the qlib ML leg, returning the latest cross-section."""
    import qlib
    from qlib.utils import init_instance_by_config

    qlib.init(**config["qlib_init"])
    model = init_instance_by_config(config["task"]["model"])
    dataset = init_instance_by_config(config["task"]["dataset"])
    model.fit(dataset)

    from trading.strategy.ml_models import latest_ml_scores

    return latest_ml_scores(model, dataset)


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- ML leg (qlib) -------------------------------------------------------
    log.info("Computing ML leg from %s ...", config_path)
    ml = _ml_scores(config)
    universe = list(ml.index) if not ml.empty else read_universe(UNIVERSE_PATH)
    log.info("ML leg: %d symbols", len(ml))

    signals = {"ml": ml}

    # --- AI legs (news sentiment + LLM events) -------------------------------
    try:
        news = get_news(universe, lookback_days=3, source="finnhub")
        if not news:  # keyless fallback
            news = get_news(universe, source="yfinance")
        log.info("Fetched %d news items", len(news))
        if news:
            signals["news_sentiment"] = news_sentiment_signal(news)
            signals["events"] = event_signal(news)
    except Exception as e:
        log.warning("AI legs skipped: %s", e)

    # --- Factor legs (price/volume) ------------------------------------------
    try:
        prices = get_prices(universe, start="2023-01-01", source="yfinance")
        if not prices.empty:
            signals.update(factor_lib.compute_factors(prices))
            log.info("Factor legs computed over %d price rows", len(prices))
    except Exception as e:
        log.warning("Factor legs skipped: %s", e)

    # --- Blend ---------------------------------------------------------------
    weights = {k: w for k, w in DEFAULT_WEIGHTS.items() if k in signals}
    blended = combine_signals(signals, BlendConfig(weights=weights))
    log.info("Blended %d legs -> %d scored symbols", len(signals), len(blended))

    # --- Rule overlay (macro regime throttle) --------------------------------
    regime = 1.0
    try:
        macro = get_macro(start="2024-01-01")
        regime = regime_scalar_from_macro(macro)
        if regime < 1.0:
            log.info("Risk-off regime: scaling scores by %.2f", regime)
    except Exception as e:
        log.warning("Macro regime check skipped: %s", e)

    final = apply_rules(blended, RuleConfig(long_only=True, regime_scalar=regime))

    if final.empty:
        log.warning("No symbols survived the pipeline; writing ML leg unblended as fallback")
        final = ml.sort_values(ascending=False)

    final.to_frame("score").to_parquet(OUT_PATH)
    log.info("Wrote %d blended scores -> %s", len(final), OUT_PATH)
    print(final.head(20).to_string())


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "trading/configs/alpha158_lgb_bundled.yaml"
    main(cfg)
