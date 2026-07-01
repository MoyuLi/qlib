"""Rule overlays and hard filters applied to a final score series.

Signals say *what to prefer*; rules enforce *what we refuse to hold*. These run
after ``combine_signals`` and before target-weight construction. They are
intentionally blunt and explainable — the place to encode tradeability and
risk-regime constraints that shouldn't be averaged away by a blend.

All pure functions over a score ``pd.Series``; compose them in ``apply_rules``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd


@dataclass
class RuleConfig:
    blocklist: Set[str] = field(default_factory=set)   # symbols never to hold
    allowlist: Optional[Set[str]] = None               # if set, restrict to these
    min_dollar_volume: float = 0.0                     # liquidity floor ($/day)
    long_only: bool = True                             # drop negative scores
    max_names: Optional[int] = None                    # keep top-N by score
    # scale all scores by a macro regime factor in [0,1] (1 = risk-on)
    regime_scalar: float = 1.0


def filter_by_liquidity(
    scores: pd.Series, dollar_volume: pd.Series, min_dollar_volume: float
) -> pd.Series:
    if min_dollar_volume <= 0 or dollar_volume is None or dollar_volume.empty:
        return scores
    liquid = dollar_volume[dollar_volume >= min_dollar_volume].index
    return scores[scores.index.isin(liquid)]


def apply_rules(
    scores: pd.Series,
    cfg: RuleConfig,
    dollar_volume: Optional[pd.Series] = None,
    halted: Optional[Iterable[str]] = None,
) -> pd.Series:
    """Apply the full overlay stack and return the surviving, ranked scores."""
    s = scores.copy()

    if cfg.allowlist is not None:
        s = s[s.index.isin({x.upper() for x in cfg.allowlist})]
    if cfg.blocklist:
        s = s[~s.index.isin({x.upper() for x in cfg.blocklist})]
    if halted:
        s = s[~s.index.isin({x.upper() for x in halted})]

    if dollar_volume is not None:
        s = filter_by_liquidity(s, dollar_volume, cfg.min_dollar_volume)

    if cfg.long_only:
        s = s[s > 0]

    if cfg.regime_scalar != 1.0:
        s = s * float(np.clip(cfg.regime_scalar, 0.0, 1.0))

    s = s.sort_values(ascending=False)
    if cfg.max_names:
        s = s.head(cfg.max_names)
    return s.rename("score")


def regime_scalar_from_macro(macro: pd.DataFrame, vix_risk_off: float = 25.0) -> float:
    """Derive a simple risk-on/off scalar from the latest VIX level.

    Returns 1.0 when calm, tapering toward 0 as VIX rises above the threshold.
    Used to throttle gross exposure in stressed regimes (see ``RuleConfig``).
    """
    if macro is None or macro.empty or "vix" not in macro.columns:
        return 1.0
    vix = macro["vix"].dropna()
    if vix.empty:
        return 1.0
    latest = float(vix.iloc[-1])
    if latest <= vix_risk_off:
        return 1.0
    # linear taper: at 2x threshold, scalar ~0
    return float(max(0.0, 1.0 - (latest - vix_risk_off) / vix_risk_off))
