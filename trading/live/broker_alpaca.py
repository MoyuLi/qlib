"""Thin, idempotent wrapper around alpaca-py.

Defaults to the paper endpoint. Every order gets a deterministic
client_order_id of the form "{date}-{symbol}-{side}" so a retry or
reconnect can never double-submit the same intended order.
"""
from dataclasses import dataclass
from datetime import date
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus


@dataclass
class BrokerOrder:
    symbol: str
    side: str          # "buy" or "sell"
    qty: float
    limit_price: float
    client_order_id: str


class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.paper = paper
        self.client = TradingClient(api_key, secret_key, paper=paper)
        if not paper:
            raise RuntimeError(
                "Refusing to construct a live (non-paper) AlpacaBroker implicitly. "
                "Pass paper=True until Phase E, and flip this guard deliberately."
            )

    @staticmethod
    def make_client_order_id(symbol: str, side: str, as_of: Optional[date] = None) -> str:
        as_of = as_of or date.today()
        return f"{as_of.isoformat()}-{symbol}-{side}"

    def get_account(self):
        return self.client.get_account()

    def get_positions(self):
        return self.client.get_all_positions()

    def get_open_orders(self):
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        return self.client.get_orders(filter=req)

    def cancel_all_open_orders(self):
        return self.client.cancel_orders()

    def submit_limit_order(self, order: BrokerOrder):
        """Submit a marketable limit order. Idempotent via client_order_id —
        if Alpaca already has an order with this id, this call is a no-op
        from the trading-intent point of view (the API will reject the dup)."""
        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        request = LimitOrderRequest(
            symbol=order.symbol,
            qty=order.qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=order.limit_price,
            client_order_id=order.client_order_id,
        )
        return self.client.submit_order(request)
