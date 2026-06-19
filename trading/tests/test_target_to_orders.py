import pandas as pd

from trading.live.target_to_orders import (
    TargetWeightConfig,
    scores_to_target_weights,
    target_weights_to_orders,
)


def test_scores_to_target_weights_equal_weights_topk():
    scores = pd.Series({"AAPL": 0.9, "MSFT": 0.8, "GOOG": 0.1})
    weights = scores_to_target_weights(scores, TargetWeightConfig(topk=2))
    assert set(weights) == {"AAPL", "MSFT"}
    assert weights["AAPL"] == 0.5
    assert weights["MSFT"] == 0.5


def test_scores_to_target_weights_empty():
    assert scores_to_target_weights(pd.Series(dtype=float), TargetWeightConfig()) == {}


def test_target_weights_to_orders_buys_new_position():
    cfg = TargetWeightConfig(equity=10_000.0, min_order_notional=10.0)
    orders = target_weights_to_orders(
        target_weights={"AAPL": 0.5},
        current_positions={},
        last_prices={"AAPL": 100.0},
        cfg=cfg,
    )
    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"
    assert orders[0].side == "buy"


def test_target_weights_to_orders_sells_dropped_position():
    cfg = TargetWeightConfig(equity=10_000.0, min_order_notional=10.0)
    orders = target_weights_to_orders(
        target_weights={},
        current_positions={"AAPL": 500.0},
        last_prices={"AAPL": 100.0},
        cfg=cfg,
    )
    assert len(orders) == 1
    assert orders[0].side == "sell"


def test_target_weights_to_orders_skips_dust():
    cfg = TargetWeightConfig(equity=10_000.0, min_order_notional=100.0)
    orders = target_weights_to_orders(
        target_weights={"AAPL": 0.05},  # target 500, current 495 -> delta 5 < min
        current_positions={"AAPL": 495.0},
        last_prices={"AAPL": 100.0},
        cfg=cfg,
    )
    assert orders == []


def test_target_weights_to_orders_skips_missing_price():
    cfg = TargetWeightConfig(equity=10_000.0, min_order_notional=10.0)
    orders = target_weights_to_orders(
        target_weights={"AAPL": 0.5},
        current_positions={},
        last_prices={},
        cfg=cfg,
    )
    assert orders == []
