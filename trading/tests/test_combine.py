import numpy as np
import pandas as pd

from trading.strategy.combine import BlendConfig, combine_signals, winsorize, zscore


def test_zscore_standardizes_to_unit_variance():
    s = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    z = zscore(s, winsor=0.0)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1.0) < 1e-9


def test_zscore_constant_signal_returns_zeros_not_nan():
    s = pd.Series({"A": 5.0, "B": 5.0, "C": 5.0})
    z = zscore(s)
    assert (z == 0).all()
    assert not z.isna().any()


def test_winsorize_clips_outlier():
    s = pd.Series(list(range(100)) + [10_000])
    w = winsorize(s, 0.05)
    assert w.max() < 10_000


def test_combine_weights_flip_direction_with_negative_weight():
    sig = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    pos = combine_signals({"x": sig}, BlendConfig(weights={"x": 1.0}))
    neg = combine_signals({"x": sig}, BlendConfig(weights={"x": -1.0}))
    # best under +weight should be worst under -weight
    assert pos.index[0] == neg.index[-1]


def test_combine_missing_symbol_treated_as_neutral():
    a = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    b = pd.Series({"A": 3.0, "B": 2.0})  # C absent
    out = combine_signals({"a": a, "b": b}, BlendConfig(weights={"a": 1.0, "b": 1.0}, min_signals=1))
    assert set(out.index) == {"A", "B", "C"}


def test_combine_min_signals_drops_thinly_covered_names():
    a = pd.Series({"A": 1.0, "B": 2.0})
    b = pd.Series({"A": 1.0})  # only A covered twice
    out = combine_signals({"a": a, "b": b}, BlendConfig(weights={"a": 1.0, "b": 1.0}, min_signals=2))
    assert list(out.index) == ["A"]


def test_combine_empty_returns_empty_score_series():
    out = combine_signals({}, BlendConfig())
    assert out.empty
    assert out.name == "score"
