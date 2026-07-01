"""Optional reinforcement-learning portfolio agent (FinRL).

FinRL + stable-baselines3 + a deep-learning backend are heavy and rarely needed
for the daily cross-sectional bridge, so this is fully lazy and isolated: if the
libraries aren't installed, ``available()`` is False and the rest of the system
runs on the factor/ML/AI blend exactly as before.

The contract mirrors the other signals — ``predict_weights`` returns a
symbol-indexed ``pd.Series`` of target weights (summing to ~1) — so an RL leg
can either *be* the strategy or be blended in as one more signal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


def available() -> bool:
    """True iff FinRL (and an RL backend) can be imported."""
    try:
        import finrl  # noqa: F401
        import stable_baselines3  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class RLConfig:
    algo: str = "ppo"            # ppo | a2c | ddpg | sac | td3
    lookback: int = 252
    timesteps: int = 50_000


class FinRLAgent:
    """Thin wrapper to train/load a FinRL portfolio-allocation agent.

    Implementations of ``train``/``predict_weights`` are stubbed against the
    FinRL API surface; they raise a clear error if the libs are missing so the
    caller can fall back to the classical blend.
    """

    def __init__(self, cfg: Optional[RLConfig] = None):
        self.cfg = cfg or RLConfig()
        self.model = None
        if not available():
            log.warning(
                "FinRL/stable-baselines3 not installed; RL leg disabled. "
                "`pip install finrl stable-baselines3` to enable."
            )

    def train(self, prices: pd.DataFrame):
        """Train on a canonical long OHLCV frame. Returns self."""
        if not available():
            raise RuntimeError("FinRL not installed; cannot train RL agent")
        # FinRL expects a long frame with a 'date'/'tic'/'close' schema and a
        # set of technical indicators. We adapt the canonical frame here.
        from finrl.agents.stablebaselines3.models import DRLAgent
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

        df = prices.rename(columns={"datetime": "date", "symbol": "tic"})
        env = self._build_env(StockTradingEnv, df)
        agent = DRLAgent(env=env)
        model = agent.get_model(self.cfg.algo)
        self.model = agent.train_model(
            model=model, tb_log_name=self.cfg.algo, total_timesteps=self.cfg.timesteps
        )
        return self

    def predict_weights(self, prices: pd.DataFrame) -> pd.Series:
        """Return target portfolio weights per symbol (sums to ~1)."""
        if self.model is None:
            raise RuntimeError("RL agent not trained/loaded")
        raise NotImplementedError(
            "Wire FinRL's DRLAgent.DRL_prediction over your eval env here and map "
            "the action vector to {symbol: weight}. Left explicit so the weight "
            "mapping is auditable rather than silently guessed."
        )

    @staticmethod
    def _build_env(env_cls, df):
        raise NotImplementedError(
            "Construct the FinRL StockTradingEnv with your indicator set / cost "
            "model here. Kept explicit to avoid hidden, untested defaults."
        )
