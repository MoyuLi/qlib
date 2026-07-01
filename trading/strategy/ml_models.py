"""Wrap a trained qlib model's prediction as a strategy signal.

This is the bridge between qlib's research stack (Alpha158 + LGBModel, etc.) and
the blend. It returns the latest cross-section of model scores as a plain
``pd.Series`` indexed by instrument, identical in shape to every other signal.

Kept separate from ``serving/generate_scores.py`` so the blend can recompute /
reuse the ML leg without re-implementing it.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


def latest_ml_scores(model, dataset, segment: str = "test") -> pd.Series:
    """Predict with a fitted qlib ``model`` over ``dataset`` and return the most
    recent date's cross-section as a symbol-indexed Series named ``ml``."""
    pred = model.predict(dataset, segment=segment) if _accepts_segment(model) else model.predict(dataset)
    return _latest_cross_section(pred)


def _latest_cross_section(pred) -> pd.Series:
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    if not isinstance(pred, pd.Series):
        pred = pd.Series(pred)

    if isinstance(pred.index, pd.MultiIndex):
        # qlib predictions are MultiIndex (datetime, instrument)
        last_date = pred.index.get_level_values(0).max()
        cross = pred.xs(last_date, level=0)
    else:
        cross = pred
    return cross.astype("float64").rename("ml").sort_values(ascending=False)


def _accepts_segment(model) -> bool:
    try:
        import inspect

        return "segment" in inspect.signature(model.predict).parameters
    except (TypeError, ValueError):
        return False


def load_scores_parquet(path) -> pd.Series:
    """Load a previously-written ``scores_latest.parquet`` as a Series named
    ``ml`` — useful when the ML leg was produced by ``generate_scores.py``."""
    df = pd.read_parquet(path)
    col = "score" if "score" in df.columns else df.columns[0]
    return df[col].astype("float64").rename("ml")
