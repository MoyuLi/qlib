"""Produce today's (symbol, score) ranking from a trained model/dataset.

Output contract consumed by the live bridge: trading/serving/scores_latest.parquet
with columns [score], indexed by instrument, for the single latest available date.

Run with: python trading/serving/generate_scores.py trading/configs/alpha158_lgb.yaml
"""
import sys
from pathlib import Path

import qlib
import yaml
from qlib.utils import init_instance_by_config

OUT_PATH = Path(__file__).parent / "scores_latest.parquet"


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    qlib.init(**config["qlib_init"])

    model = init_instance_by_config(config["task"]["model"])
    dataset = init_instance_by_config(config["task"]["dataset"])

    # NOTE: in production this model comes from the rolling-retrain job
    # (see examples/online_srv/), not a freshly re-fit one each run.
    model.fit(dataset)

    pred = model.predict(dataset)
    if hasattr(pred, "to_frame"):
        pred = pred.to_frame("score") if pred.name != "score" else pred.to_frame()

    today = pred.index.get_level_values(0).max()
    today_scores = pred.loc[today].sort_values(by=pred.columns[0] if hasattr(pred, "columns") else None,
                                                ascending=False)

    today_scores.columns = ["score"]
    today_scores.to_parquet(OUT_PATH)
    print(f"Wrote {len(today_scores)} scores for {today} -> {OUT_PATH}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "trading/configs/alpha158_lgb.yaml"
    main(cfg)
