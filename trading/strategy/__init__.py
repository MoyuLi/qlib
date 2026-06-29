"""Strategy layer.

Replaces the live bridge's hardcoded "equal-weight the top-k ML scores" with a
pluggable signal pipeline:

* ``factors``    — classical price/volume factors computed from an OHLCV frame.
* ``ml_models``  — wrap a trained qlib model's prediction as a signal.
* ``rules``      — hard filters / overlays (liquidity, halts, macro regime).
* ``rl``         — optional FinRL agent producing portfolio weights.
* ``combine``    — z-score and blend many signals into one final score, the
                   contract the live ``target_to_orders`` already consumes.

The blended output is still a ``pd.Series`` of per-symbol scores, so nothing
downstream of ``serving/scores_latest.parquet`` has to change.
"""
