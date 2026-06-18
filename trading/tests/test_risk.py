from trading.live.risk import AccountState, Order, RiskLimits, check_order

LIMITS = RiskLimits(
    max_position_pct=0.10,
    max_gross_exposure=1.0,
    max_order_notional=1000.0,
    max_orders_per_run=2,
    daily_loss_halt_pct=0.03,
)


def account(**overrides) -> AccountState:
    base = dict(equity=20_000.0, cash=10_000.0, gross_exposure_pct=0.5, day_pnl_pct=0.0)
    base.update(overrides)
    return AccountState(**base)


def order(**overrides) -> Order:
    base = dict(symbol="AAPL", side="buy", notional=500.0, resulting_position_pct=0.05)
    base.update(overrides)
    return Order(**base)


def test_allows_ordinary_order():
    allowed, _ = check_order(order(), account(), LIMITS)
    assert allowed


def test_blocks_when_daily_loss_halt_active():
    allowed, reason = check_order(order(), account(day_pnl_pct=-0.05), LIMITS)
    assert not allowed
    assert "halt" in reason


def test_blocks_order_exceeding_notional_cap():
    allowed, reason = check_order(order(notional=5000.0), account(), LIMITS)
    assert not allowed
    assert "max_order_notional" in reason


def test_blocks_order_exceeding_position_pct():
    allowed, reason = check_order(order(resulting_position_pct=0.2), account(), LIMITS)
    assert not allowed
    assert "max_position_pct" in reason


def test_blocks_order_exceeding_gross_exposure():
    allowed, reason = check_order(order(notional=900.0), account(gross_exposure_pct=0.98), LIMITS)
    assert not allowed
    assert "max_gross_exposure" in reason


def test_blocks_when_max_orders_per_run_reached():
    allowed, reason = check_order(order(), account(), RiskLimits(max_orders_per_run=0))
    assert not allowed
    assert "max_orders_per_run" in reason


def test_blocks_non_positive_notional():
    allowed, reason = check_order(order(notional=0), account(), LIMITS)
    assert not allowed
    assert "notional" in reason
