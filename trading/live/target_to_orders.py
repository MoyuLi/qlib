"""Turn today's scores into target weights, diff against current positions,
and emit only the delta orders needed to move from current -> target.
"""
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from trading.live.broker_alpaca import BrokerOrder


@dataclass
class TargetWeightConfig:
    topk: int = 20
    equity: float = 20_000.0
    min_order_notional: float = 50.0  # skip dust-sized rebalancing orders


def scores_to_target_weights(scores: pd.Series, cfg: TargetWeightConfig) -> Dict[str, float]:
    """Equal-weight the top-k scored symbols. Returns {symbol: weight}, weights sum to <=1."""
    top = scores.sort_values(ascending=False).head(cfg.topk)
    if top.empty:
        return {}
    weight = 1.0 / len(top)
    return {symbol: weight for symbol in top.index}


def target_weights_to_orders(
    target_weights: Dict[str, float],
    current_positions: Dict[str, float],  # symbol -> current market value ($)
    last_prices: Dict[str, float],        # symbol -> last known price, for qty/limit calc
    cfg: TargetWeightConfig,
    as_of=None,
) -> List[BrokerOrder]:
    """Diff target dollar exposure against current and emit delta limit orders.

    Only symbols in the union of target+current are considered. Orders below
    min_order_notional are skipped to avoid churning on rounding noise.
    """
    orders: List[BrokerOrder] = []
    symbols = set(target_weights) | set(current_positions)

    for symbol in sorted(symbols):
        target_value = target_weights.get(symbol, 0.0) * cfg.equity
        current_value = current_positions.get(symbol, 0.0)
        delta = target_value - current_value

        if abs(delta) < cfg.min_order_notional:
            continue

        price = last_prices.get(symbol)
        if not price or price <= 0:
            continue  # can't size an order without a price; skip and log upstream

        side = "buy" if delta > 0 else "sell"
        qty = round(abs(delta) / price, 4)  # fractional shares; adjust if not supported
        limit_price = round(price * (1.005 if side == "buy" else 0.995), 2)  # marketable limit

        orders.append(
            BrokerOrder(
                symbol=symbol,
                side=side,
                qty=qty,
                limit_price=limit_price,
                client_order_id=f"{(as_of or pd.Timestamp.now().date()).isoformat()}-{symbol}-{side}",
            )
        )

    return orders
