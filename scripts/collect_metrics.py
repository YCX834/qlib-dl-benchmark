"""
汇总所有实验的指标 → results/metrics.csv

用法
----
    python scripts/collect_metrics.py
    python scripts/collect_metrics.py --markdown      # 输出 Markdown 表格（贴 README 用）
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# 展示顺序（与 README 的叙事顺序一致）
ORDER = ["lightgbm_alpha158", "lstm_alpha360", "transformer_alpha360"]

COLUMNS = [
    ("experiment", "实验"),
    ("model_class", "模型"),
    ("handler", "因子"),
    ("rankic_mean", "RankIC"),
    ("rankic_ir", "RankIC IR"),
    ("ic_mean", "IC"),
    ("annual_return", "年化收益%"),
    ("max_drawdown", "最大回撤%"),
    ("sharpe", "夏普"),
    ("benchmark_return", "基准收益%"),
    ("excess_return", "超额收益%"),
    ("daily_turnover", "日均换手"),
    ("total_cost", "总成本"),
    ("high_vol_annual_return", "高波动期年化%"),
    ("low_vol_annual_return", "低波动期年化%"),
    ("elapsed_sec", "运行秒"),
]


def load_all() -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(RESULTS, "*", "metrics.json")):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        # 去掉内部曲线数据
        d.pop("_nav", None)
        d.pop("_drawdown", None)
        d.pop("_bench_nav", None)
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "experiment" in df:
        df["_rank"] = df["experiment"].apply(
            lambda x: ORDER.index(x) if x in ORDER else 999)
        df = df.sort_values("_rank").drop(columns="_rank")
    return df


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_暂无实验结果_"
    cols = [(k, disp) for k, disp in COLUMNS if k in df.columns]
    lines = ["| " + " | ".join(d for _, d in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for k, _ in cols:
            v = r.get(k)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("—")
            elif isinstance(v, float):
                cells.append("%.4f" % v if abs(v) < 1 else "%.2f" % v)
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    df = load_all()
    if df.empty:
        print("还没有实验结果，先跑 run_experiment.py")
        return

    out_csv = os.path.join(RESULTS, "metrics.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("已写出 %s (%d 行)" % (out_csv, len(df)))

    if args.markdown:
        print()
        print(to_markdown(df))
    else:
        with pd.option_context("display.max_columns", None,
                               "display.width", 200):
            print()
            print(df[[c for c, _ in COLUMNS if c in df.columns]].to_string(
                index=False))


if __name__ == "__main__":
    main()
