"""
出图：累计收益曲线、回撤曲线、分市场环境对比 → results/figures/

用法
----
    python scripts/plot_report.py

生成的图（README 直接引用）：
    fig1_cumulative_return.png   各模型累计净值对比 + 基准
    fig2_drawdown.png            各模型回撤曲线
    fig3_regime.png              高/低波动期表现对比
    fig4_ic.png                  IC / RankIC 时序对比
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")

# 中文字体（Windows）
for cand in ["Microsoft YaHei", "SimHei", "DengXian"]:
    try:
        matplotlib.font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except (ValueError, RuntimeError):
        continue
plt.rcParams["axes.unicode_minus"] = False

LABELS = {
    "lightgbm_alpha158": "LightGBM (Alpha158)",
    "lstm_alpha360": "LSTM (Alpha360)",
    "transformer_alpha360": "Transformer (Alpha360)",
    "lstm_alpha360_small": "LSTM 小容量 (消融)",
    "transformer_alpha360_tuned": "Transformer 调优版 (消融)",
}
COLORS = {
    "lightgbm_alpha158": "#2E5EAA",
    "lstm_alpha360": "#D9822B",
    "transformer_alpha360": "#3A9E6E",
    "lstm_alpha360_small": "#8FB8DE",
    "transformer_alpha360_tuned": "#A8D5BA",
}
# 消融实验用虚线，主线用实线
ABLATION = {"lstm_alpha360_small", "transformer_alpha360_tuned"}


def _style(e):
    """按实验类型返回 (标签, 颜色, 线型, 线宽)。"""
    name = e["experiment"]
    ls = "--" if name in ABLATION else "-"
    lw = 1.3 if name in ABLATION else 1.7
    return LABELS.get(name, name), COLORS.get(name), ls, lw


def load_experiments():
    """读所有实验的指标 + 曲线。

    曲线数据（净值/回撤/IC）存在 results/<name>/curves.pkl，
    标量指标在 results/<name>/metrics.json。
    """
    import pickle
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "*", "metrics.json"))):
        name = os.path.basename(os.path.dirname(p))
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        cp = os.path.join(RESULTS, name, "curves.pkl")
        if os.path.exists(cp):
            with open(cp, "rb") as f:
                curves = pickle.load(f)
            for k, v in curves.items():
                d[k.lstrip("_")] = v
        out.append(d)
    return out


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ✓ %s" % os.path.relpath(path, ROOT))


def fig_cumulative(exps):
    fig, ax = plt.subplots(figsize=(12, 6))
    for e in exps:
        nav = e.get("nav")
        if nav is None or len(nav) == 0:
            continue
        lab, col, ls, lw = _style(e)
        ax.plot(nav.index, nav.values, label=lab, color=col,
                linewidth=lw, linestyle=ls, alpha=0.9 if ls == "-" else 0.75)
    # 基准
    for e in exps:
        b = e.get("bench_nav")
        if b is not None and len(b):
            ax.plot(b.index, b.values, label="沪深300 基准",
                    color="#888888", linewidth=1.4, linestyle=":")
            break
    ax.set_title("累计净值对比（测试期 2017-01 ~ 2020-08，含交易成本）\n"
                 "虚线 = 消融实验", fontsize=13, pad=12)
    ax.set_ylabel("净值（初始 = 1.0）")
    ax.set_xlabel("")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(alpha=0.25, linestyle=":")
    ax.axhline(1.0, color="#ccc", linewidth=0.8)
    save(fig, "fig1_cumulative_return.png")


def fig_drawdown(exps):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for e in exps:
        dd = e.get("drawdown")
        if dd is None or len(dd) == 0:
            continue
        lab, col, ls, lw = _style(e)
        ax.plot(dd.index, dd.values * 100, label=lab, color=col,
                linewidth=lw, linestyle=ls, alpha=0.9 if ls == "-" else 0.75)
    ax.set_title("回撤曲线（测试期）", fontsize=13, pad=12)
    ax.set_ylabel("回撤 (%)")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.grid(alpha=0.25, linestyle=":")
    save(fig, "fig2_drawdown.png")


def fig_regime(exps):
    """高低波动期年化收益对比。"""
    rows = []
    for e in exps:
        if "high_vol_annual_return" not in e:
            continue
        rows.append({
            "model": LABELS.get(e["experiment"], e["experiment"]),
            "高波动期": e.get("high_vol_annual_return"),
            "低波动期": e.get("low_vol_annual_return"),
            "_ablation": e["experiment"] in ABLATION,
        })
    if not rows:
        return
    df = pd.DataFrame(rows).set_index("model")
    abl = df["_ablation"]
    df = df.drop(columns=["_ablation"])

    fig, ax = plt.subplots(figsize=(11, 5.4))
    df.plot(kind="bar", ax=ax, color=["#D9822B", "#4A90D9"], width=0.62)
    # 消融实验加斜线纹理区分。
    # 注意：pandas 的 bar patch 顺序是「按列分组」（第 0 列所有 bar，再第 1 列所有 bar），
    # 所以 patch 下标是 j * n_rows + i，不是 i * n_cols + j。
    n = len(df)
    for i in range(len(df)):
        if abl.iloc[i]:
            for j in range(len(df.columns)):
                ax.patches[j * n + i].set_hatch("//")
                ax.patches[j * n + i].set_alpha(0.55)
    ax.set_title("不同市场环境下的年化收益（按基准60日波动率三分位切分）\n"
                 "斜纹 = 消融实验", fontsize=13, pad=12)
    ax.set_ylabel("年化收益 (%)")
    ax.set_xlabel("模型")
    ax.axhline(0, color="#888", linewidth=0.9)
    ax.legend(frameon=False, title="市场环境")
    ax.grid(alpha=0.25, axis="y", linestyle=":")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=9)
    save(fig, "fig3_regime.png")


def fig_ic(exps):
    """IC / RankIC 时序对比 + 滚动均值。"""
    rows = [e for e in exps if "rankic_series" in e]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for e in rows:
        name = e["experiment"]
        lab = LABELS.get(name, name)
        col = COLORS.get(name)
        ric = e.get("rankic_series")
        ic = e.get("ic_series")
        if ric is not None and len(ric):
            axes[0].plot(ric.index, ric.values, alpha=0.28, color=col, linewidth=0.8)
            axes[1].plot(ric.index, ric.rolling(60).mean().values,
                          label=lab, color=col, linewidth=1.6)
        if ic is not None and len(ic):
            axes[0].plot(ic.index, ic.values, alpha=0.28, color=col,
                         linewidth=0.8, linestyle="--")

    axes[0].axhline(0, color="#888", linewidth=0.8)
    axes[0].set_title("逐日 IC / RankIC（测试期）", fontsize=12, pad=10)
    axes[0].set_ylabel("IC")
    axes[0].grid(alpha=0.25, linestyle=":")

    axes[1].axhline(0, color="#888", linewidth=0.8)
    axes[1].set_title("RankIC 60 日滚动均值", fontsize=12, pad=10)
    axes[1].set_ylabel("滚动 RankIC")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(alpha=0.25, linestyle=":")

    fig.tight_layout()
    save(fig, "fig5_ic_timeseries.png")


def fig_summary_bar(exps):
    """关键指标横向对比条形图。"""
    metrics = [("annual_return", "年化收益 %", "#2E5EAA"),
               ("max_drawdown", "最大回撤 %", "#C0504D"),
               ("sharpe", "夏普", "#3A9E6E"),
               ("rankic_mean", "RankIC", "#D9822B")]
    avail = [(k, t, c) for k, t, c in metrics
             if any(e.get(k) is not None for e in exps)]
    if not avail:
        return
    names = [LABELS.get(e["experiment"], e["experiment"]) for e in exps]
    fig, axes = plt.subplots(1, len(avail), figsize=(4 * len(avail), 4.2))
    if len(avail) == 1:
        axes = [axes]
    for ax, (k, title, color) in zip(axes, avail):
        vals = [e.get(k) if e.get(k) is not None else np.nan for e in exps]
        colors = [COLORS.get(e["experiment"], color) for e in exps]
        bars = ax.bar(range(len(vals)), vals, color=colors, width=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, axis="y", linestyle=":")
        for b, v in zip(bars, vals):
            if not (isinstance(v, float) and np.isnan(v)):
                ax.text(b.get_x() + b.get_width() / 2, v,
                        "%.2f" % v, ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8)
    fig.suptitle("各模型关键指标对比", fontsize=13, y=1.02)
    save(fig, "fig4_metrics_bar.png")


def main():
    exps = load_experiments()
    if not exps:
        print("没有可用的实验结果，先跑 run_experiment.py")
        return
    print("找到 %d 个实验结果，开始出图" % len(exps))
    fig_cumulative(exps)
    fig_drawdown(exps)
    fig_regime(exps)
    fig_ic(exps)
    fig_summary_bar(exps)


if __name__ == "__main__":
    main()
