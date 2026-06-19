"""Risk layer — every order must pass check_order() before submission.

This module owns no broker/network calls. It is pure logic so it can be
unit tested exhaustively without touching Alpaca.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RiskLimits:
    max_position_pct: float = 0.10       # max fraction of equity in one symbol
    max_gross_exposure: float = 1.0      # no leverage in v1
    max_order_notional: float = 1000.0   # absolute $ cap per single order
    max_orders_per_run: int = 50         # catch runaway loops
    daily_loss_halt_pct: float = 0.03    # halt all trading if down this much today


@dataclass
class Order:
    symbol: str
    side: str          # "buy" or "sell"
    notional: float     # absolute dollar size of this order
    resulting_position_pct: float  # resulting |position| / equity if this order fills


@dataclass
class AccountState:
    equity: float
    cash: float
    gross_exposure_pct: float   # current gross exposure / equity, before this order
    day_pnl_pct: float          # today's P&L as a fraction of starting-day equity
    orders_submitted_this_run: int = 0


def check_order(order: Order, account: AccountState, limits: RiskLimits = RiskLimits()) -> Tuple[bool, str]:
    """Return (allowed, reason). A False result MUST be treated as final —
    nothing downstream may override or retry around a reject."""

    if account.day_pnl_pct <= -abs(limits.daily_loss_halt_pct):
        return False, f"daily loss halt active ({account.day_pnl_pct:.2%} <= -{limits.daily_loss_halt_pct:.2%})"

    if account.orders_submitted_this_run >= limits.max_orders_per_run:
        return False, f"max_orders_per_run reached ({limits.max_orders_per_run})"

    if order.notional <= 0:
        return False, "non-positive order notional"

    if order.notional > limits.max_order_notional:
        return False, f"order notional {order.notional:.2f} exceeds max_order_notional {limits.max_order_notional:.2f}"

    if order.resulting_position_pct > limits.max_position_pct:
        return False, (
            f"resulting position {order.resulting_position_pct:.2%} exceeds "
            f"max_position_pct {limits.max_position_pct:.2%}"
        )

    projected_gross = account.gross_exposure_pct + (order.notional / account.equity if account.equity else 0)
    if projected_gross > limits.max_gross_exposure:
        return False, (
            f"projected gross exposure {projected_gross:.2%} exceeds "
            f"max_gross_exposure {limits.max_gross_exposure:.2%}"
        )

    return True, "ok"
