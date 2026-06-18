"""Backtest driver — adapted from examples/workflow_by_code.py.

Run with: python trading/research/run_backtest.py trading/configs/alpha158_lgb.yaml
"""
import sys
from pathlib import Path

import qlib
import yaml
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.contrib.report import analysis_position
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    qlib.init(**config["qlib_init"])

    model = init_instance_by_config(config["task"]["model"])
    dataset = init_instance_by_config(config["task"]["dataset"])

    with R.start(experiment_name="trading_research"):
        model.fit(dataset)
        R.save_objects(trained_model=model)

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        sar = SigAnaRecord(recorder)
        sar.generate()

        par = PortAnaRecord(recorder, config["port_analysis_config"], "day")
        par.generate()

        print(f"Recorder id: {recorder.id}")
        print("IC / Rank-IC and portfolio analysis written to the experiment store.")
        print("Inspect with `qlib.workflow.R` or `mlflow ui` against the experiment tracking dir.")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "trading/configs/alpha158_lgb.yaml"
    main(cfg)
