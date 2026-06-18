"""Reconcile local state against the broker on every startup.

The broker is always the source of truth. Never assume locally-saved
state survived a crash or restart intact.
"""
from dataclasses import dataclass
from typing import Dict

from trading.live.broker_alpaca import AlpacaBroker


@dataclass
class ReconciledState:
    equity: float
    cash: float
    day_pnl_pct: float
    positions: Dict[str, float]   # symbol -> market value ($)
    last_equity: float            # equity at previous close, for day_pnl_pct sanity


def reconcile(broker: AlpacaBroker) -> ReconciledState:
    account = broker.get_account()
    positions = broker.get_positions()

    equity = float(account.equity)
    cash = float(account.cash)
    last_equity = float(account.last_equity) if account.last_equity else equity

    day_pnl_pct = (equity - last_equity) / last_equity if last_equity else 0.0

    position_values = {p.symbol: float(p.market_value) for p in positions}

    return ReconciledState(
        equity=equity,
        cash=cash,
        day_pnl_pct=day_pnl_pct,
        positions=position_values,
        last_equity=last_equity,
    )
