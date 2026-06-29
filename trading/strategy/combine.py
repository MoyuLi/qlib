"""Blend multiple per-symbol signals into one final score.

The quant-standard recipe: cross-sectionally standardize each signal (so a
sentiment score in [-1,1] and a momentum factor in raw return units are
comparable), optionally winsorize tails, then take a weighted sum.

Everything here is pure pandas — no IO, no model loading — so it unit-tests
fast and deterministically. The output ``pd.Series`` of scores is exactly what
``trading.live.target_to_orders.scores_to_target_weights`` expects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, limits: float = 0.05) -> pd.Series:
    """Clip each tail at the given quantile to tame outliers before scaling."""
    if s.empty or limits <= 0:
        return s
    lo, hi = s.quantile(limits), s.quantile(1 - limits)
    return s.clip(lower=lo, upper=hi)


def zscore(s: pd.Series, winsor: float = 0.05) -> pd.Series:
    """Cross-sectional z-score (mean 0, std 1), winsorized first.

    A degenerate (zero-variance / single-name) signal returns all-zeros rather
    than NaN/inf, so it contributes nothing instead of poisoning the blend.
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return s
    s = winsorize(s, winsor)
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=s.index, name=s.name)
    return ((s - s.mean()) / std).rename(s.name)


@dataclass
class BlendConfig:
    # signal name -> weight. Weights need not sum to 1; the result is a score,
    # not a weight vector. Negative weights flip a signal's direction.
    weights: Dict[str, float] = field(default_factory=lambda: {"ml": 1.0})
    winsor: float = 0.05
    # require a symbol to appear in at least this many signals to be scored
    min_signals: int = 1


def combine_signals(
    signals: Mapping[str, pd.Series],
    cfg: Optional[BlendConfig] = None,
) -> pd.Series:
    """Z-score each named signal, align on the symbol union, weighted-sum.

    Missing values (a symbol absent from one signal) are treated as 0 *after*
    standardization — i.e. "no information", not an extreme. Symbols seen in
    fewer than ``cfg.min_signals`` inputs are dropped.
    """
    cfg = cfg or BlendConfig()
    if not signals:
        return pd.Series(dtype="float64", name="score")

    standardized = {name: zscore(s, cfg.winsor) for name, s in signals.items() if s is not None}
    standardized = {name: s for name, s in standardized.items() if not s.empty}
    if not standardized:
        return pd.Series(dtype="float64", name="score")

    frame = pd.DataFrame(standardized)
    coverage = frame.notna().sum(axis=1)

    weighted = pd.Series(0.0, index=frame.index)
    for name, col in frame.items():
        w = cfg.weights.get(name, 0.0)
        if w:
            weighted = weighted.add(col.fillna(0.0) * w, fill_value=0.0)

    weighted = weighted[coverage >= cfg.min_signals]
    return weighted.sort_values(ascending=False).rename("score")
