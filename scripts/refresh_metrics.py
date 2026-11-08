"""
从已有的 mlflow 实验中重新提取指标（不重跑训练）。

用途：调整指标提取逻辑后，无需花几十分钟重跑训练，直接从 recorder 重建。

用法
----
    python scripts/refresh_metrics.py                  # 全部实验
    python scripts/refresh_metrics.py lightgbm_alpha158
"""
from __future__ import annotations

import json
import os
import sys

import mlflow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

os.chdir(ROOT)                      # mlflow 的 ./mlruns 在项目目录下
mlflow.set_tracking_uri("file:./mlruns")

from run_experiment import collect_metrics, print_metrics, save_metrics  # noqa: E402

import yaml  # noqa: E402


def main():
    names = sys.argv[1:]
    client = mlflow.tracking.MlflowClient()
    exps = client.search_experiments()
    if not names:
        names = [e.name for e in exps if e.name and e.name != "Default"]

    for name in names:
        exp = client.get_experiment_by_name(name)
        if exp is None:
            print("[跳过] 没有实验 %s" % name)
            continue
        runs = client.search_runs([exp.experiment_id],
                                   order_by=["attributes.start_time desc"])
        if not runs:
            print("[跳过] %s 没有 run" % name)
            continue
        run = runs[0]

        # 找到对应的 YAML 配置
        cfg_path = os.path.join(ROOT, "configs", name + ".yaml")
        if not os.path.exists(cfg_path):
            print("[跳过] %s 找不到配置文件" % name)
            continue
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # 用一个轻量包装，让 collect_metrics 能拿到 run_id
        class _Rec:
            id = run.info.run_id
        # mlflow 的 end_time/start_time 是毫秒
        elapsed = ((run.info.end_time - run.info.start_time) / 1000.0
                   if run.info.end_time else 0)
        m = collect_metrics(_Rec(), name, elapsed, cfg)

        out_dir = os.path.join(ROOT, "results", name)
        os.makedirs(out_dir, exist_ok=True)
        save_metrics(m, name)
        print()
        print_metrics(m)


if __name__ == "__main__":
    main()
