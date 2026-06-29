import pandas as pd

from trading.strategy.rules import RuleConfig, apply_rules, regime_scalar_from_macro


def scores():
    return pd.Series({"AAPL": 2.0, "MSFT": 1.0, "TSLA": -0.5, "PENNY": 0.2})


def test_long_only_drops_negative_scores():
    out = apply_rules(scores(), RuleConfig(long_only=True))
    assert "TSLA" not in out.index
    assert (out > 0).all()


def test_blocklist_excludes_symbol():
    out = apply_rules(scores(), RuleConfig(blocklist={"AAPL"}, long_only=False))
    assert "AAPL" not in out.index


def test_allowlist_restricts_universe():
    out = apply_rules(scores(), RuleConfig(allowlist={"AAPL", "MSFT"}, long_only=False))
    assert set(out.index) == {"AAPL", "MSFT"}


def test_halted_symbols_removed():
    out = apply_rules(scores(), RuleConfig(long_only=False), halted=["MSFT"])
    assert "MSFT" not in out.index


def test_liquidity_filter_drops_illiquid():
    dv = pd.Series({"AAPL": 1e9, "MSFT": 1e9, "TSLA": 1e9, "PENNY": 1.0})
    out = apply_rules(scores(), RuleConfig(min_dollar_volume=1e6, long_only=False), dollar_volume=dv)
    assert "PENNY" not in out.index


def test_max_names_keeps_top_n():
    out = apply_rules(scores(), RuleConfig(long_only=True, max_names=1))
    assert list(out.index) == ["AAPL"]


def test_regime_scalar_calm_is_one():
    macro = pd.DataFrame({"vix": [12.0, 15.0]})
    assert regime_scalar_from_macro(macro) == 1.0


def test_regime_scalar_tapers_when_stressed():
    macro = pd.DataFrame({"vix": [40.0]})
    s = regime_scalar_from_macro(macro, vix_risk_off=25.0)
    assert 0.0 <= s < 1.0


def test_regime_scalar_missing_vix_defaults_to_one():
    assert regime_scalar_from_macro(pd.DataFrame()) == 1.0
