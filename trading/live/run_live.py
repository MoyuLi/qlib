"""Daily orchestrator. Run once per trading day, after generate_scores.py
has produced trading/serving/scores_latest.parquet.

    python trading/live/run_live.py

Reads ALPACA_API_KEY / ALPACA_SECRET_KEY from the environment. Defaults to
paper trading; do not flip to live until Phase E and only deliberately.
"""
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from trading.live.broker_alpaca import AlpacaBroker
from trading.live.reconcile import reconcile
from trading.live.risk import AccountState, Order, RiskLimits, check_order
from trading.live.target_to_orders import (
    TargetWeightConfig,
    scores_to_target_weights,
    target_weights_to_orders,
)

SCORES_PATH = Path(__file__).parent.parent / "serving" / "scores_latest.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_live")


def get_last_prices(broker: AlpacaBroker, symbols) -> dict:
    """Fetch last trade price per symbol from Alpaca market data.

    Uses the same Alpaca account credentials. Symbols that have no quote
    (delisted, non-US, bad ticker) are simply omitted; target_to_orders
    skips any symbol without a price, so this fails safe.
    """
    symbols = [s for s in {str(s).upper() for s in symbols} if s]
    if not symbols:
        return {}

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest

    data = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    )
    prices: dict = {}
    # batch to keep request URLs sane
    for i in range(0, len(symbols), 100):
        chunk = symbols[i : i + 100]
        try:
            trades = data.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=chunk)
            )
            for sym, trade in trades.items():
                if trade and trade.price:
                    prices[sym] = float(trade.price)
        except Exception as e:  # one bad chunk shouldn't sink the run
            log.warning("price fetch failed for chunk %s..: %s", chunk[:3], e)
    return prices


def alert(message: str):
    """Placeholder: send to email/Slack/Telegram. Silence is the dangerous
    failure mode here, so make sure something pages you if this never fires."""
    log.warning("ALERT: %s", message)


def main():
    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"

    broker = AlpacaBroker(api_key, secret_key, paper=paper)

    # 1. reconcile against the broker — it is the source of truth
    state = reconcile(broker)
    log.info("Reconciled: equity=%.2f cash=%.2f day_pnl_pct=%.2f%%", state.equity, state.cash, state.day_pnl_pct * 100)

    limits = RiskLimits()
    if state.day_pnl_pct <= -abs(limits.daily_loss_halt_pct):
        broker.cancel_all_open_orders()
        alert(f"Daily loss halt triggered ({state.day_pnl_pct:.2%}). Cancelled open orders, exiting.")
        return

    # 2. load today's scores
    if not SCORES_PATH.exists():
        alert(f"No scores file at {SCORES_PATH}; skipping this run.")
        return
    scores = pd.read_parquet(SCORES_PATH)["score"]

    # 3. compute target weights -> delta orders
    # Size against a deployable sleeve (Phase E: "fund a few hundred dollars"),
    # not the whole paper account. Risk math below still uses true equity.
    deploy_capital = float(os.environ.get("TRADING_CAPITAL", state.equity))
    cfg = TargetWeightConfig(equity=deploy_capital)
    log.info("Sizing target weights against deploy_capital=%.2f (equity=%.2f)", deploy_capital, state.equity)
    target_weights = scores_to_target_weights(scores, cfg)
    last_prices = get_last_prices(broker, set(target_weights) | set(state.positions))
    orders = target_weights_to_orders(target_weights, state.positions, last_prices, cfg)

    # 4. risk-check then submit each order; no order may bypass check_order
    account_state = AccountState(
        equity=state.equity,
        cash=state.cash,
        gross_exposure_pct=sum(state.positions.values()) / state.equity if state.equity else 0,
        day_pnl_pct=state.day_pnl_pct,
    )

    submitted, rejected = 0, 0
    for order in orders:
        position_value_after = state.positions.get(order.symbol, 0.0) + (
            order.qty * order.limit_price * (1 if order.side == "buy" else -1)
        )
        risk_order = Order(
            symbol=order.symbol,
            side=order.side,
            notional=order.qty * order.limit_price,
            resulting_position_pct=abs(position_value_after) / state.equity if state.equity else 0,
        )
        allowed, reason = check_order(risk_order, account_state, limits)
        if not allowed:
            log.info("REJECTED %s %s: %s", order.side, order.symbol, reason)
            rejected += 1
            continue

        try:
            broker.submit_limit_order(order)
            log.info("SUBMITTED %s %s qty=%s limit=%s", order.side, order.symbol, order.qty, order.limit_price)
            submitted += 1
            account_state.orders_submitted_this_run += 1
        except Exception as e:
            log.exception("Order submission failed for %s", order.symbol)
            alert(f"Order submission failed for {order.symbol}: {e}")

    log.info("Run complete: submitted=%d rejected=%d", submitted, rejected)


if __name__ == "__main__":
    main()
