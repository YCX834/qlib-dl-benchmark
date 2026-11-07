"""
统一实验运行器：读 YAML 配置 → 跑 Qlib workflow → 落盘结果。

用法
----
    python scripts/run_experiment.py --config configs/lightgbm_alpha158.yaml
    python scripts/run_experiment.py --config configs/lstm_alpha360.yaml --tag my_run

设计要点
--------
1. **完全复刻 qlib 的 qrun**：模型 → SignalRecord → SigAnaRecord → PortAnaRecord
   （参考 qlib 官方 examples/workflow_by_code.py）

2. **实验设置与代码分离**：所有超参在 YAML 里，脚本不含任何模型专属逻辑。
   这样 LightGBM / LSTM / Transformer 走的是**同一条代码路径**，
   保证对比的公平性 —— 唯一差异就是 YAML 里写的东西。

3. **结果落盘**：每个实验在 results/ 下生成一个目录，含
   - metrics.json    汇总指标（IC/RankIC/收益/回撤/夏普）
   - predictions.pkl 测试期逐日预测
   - report.pkl      完整回测报告
   便于后续统一出图和横向对比。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")


# --------------------------------------------------------------------------
# 配置处理
# --------------------------------------------------------------------------


def _coerce_numbers(obj):
    """把 YAML 里被解析成字符串的科学计数法转回 float。

    YAML 1.1 的规范里 `1e-3` **不是**合法的浮点数（必须写成 `1.0e-3`），
    所以 PyYAML 会把它当字符串。这会让 torch.optim.Adam 报
    "'<=' not supported between instances of 'float' and 'str'"。

    这里做一次兜底转换，让配置写法更宽松。
    """
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        # 形如 1e-3 / 1E-3 / -2.5e+4
        try:
            if ("e" in s.lower()) and any(c.isdigit() for c in s):
                return float(s)
        except ValueError:
            pass
    return obj


def load_config(path: str) -> dict:
    """读 YAML，处理锚点/别名与科学计数法。"""
    with open(path, "r", encoding="utf-8") as f:
        return _coerce_numbers(yaml.safe_load(f))


def deep_update(base: dict, other: dict) -> dict:
    """递归合并字典（other 覆盖 base）。"""
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


# --------------------------------------------------------------------------
# 实验执行
# --------------------------------------------------------------------------


def run_experiment(cfg: dict, tag: str = "", force: bool = False) -> dict:
    import qlib
    from qlib.constant import REG_CN
    from qlib.utils import init_instance_by_config, flatten_dict
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

    name = tag or os.path.splitext(os.path.basename(cfg["_config_path"]))[0]
    out_dir = os.path.join(RESULTS_DIR, name)
    metrics_path = os.path.join(out_dir, "metrics.json")

    if os.path.exists(metrics_path) and not force:
        with open(metrics_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print("[跳过] %s 已有结果（加 --force 重跑）" % name)
        return cached

    os.makedirs(out_dir, exist_ok=True)

    # ---- 初始化 qlib ----
    init_cfg = cfg.get("qlib_init", {})
    provider_uri = os.path.expanduser(
        init_cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data"))
    region = REG_CN if init_cfg.get("region", "cn") == "cn" else REG_US
    qlib.init(provider_uri=provider_uri, region=region)
    print("[init] provider_uri=%s" % provider_uri)

    task = cfg["task"]

    # ---- 构造模型与数据集 ----
    print("[build] 构造模型与数据集 ...")
    t0 = time.time()
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    print("[build] 用时 %.1fs" % (time.time() - t0))

    # 数据集概览（确认特征维度与样本量）
    try:
        df_train = dataset.prepare("train")
        print("[data] 训练集 %d 行 × %d 列" % df_train.shape)
        print("[data] 列名: %s" % list(df_train.columns)[:8])
    except Exception as e:
        print("[data] 训练集预览失败: %s" % str(e)[:80])

    # ---- 跑 workflow ----
    print("[run] 开始训练与回测 ...")
    t0 = time.time()
    with R.start(experiment_name=name):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        print("[run] 信号生成完成")

        sar = SigAnaRecord(recorder)
        sar.generate()
        print("[run] 信号分析完成")

        par = PortAnaRecord(recorder, cfg["port_analysis_config"], "day")
        par.generate()
        print("[run] 回测完成")

    elapsed = time.time() - t0
    print("[run] 总用时 %.1fs" % elapsed)

    # ---- 汇总指标 ----
    metrics = collect_metrics(recorder, name, elapsed, cfg)
    save_metrics(metrics, name)

    # ---- 保存预测与报告 ----
    try:
        root = _artifact_root(recorder)
        pred = _load_artifact(root, "pred.pkl")
        with open(os.path.join(out_dir, "predictions.pkl"), "wb") as f:
            pickle.dump(pred, f)
    except Exception as e:
        print("[warn] 保存预测失败: %s" % str(e)[:80])
    try:
        root = _artifact_root(recorder)
        pa = _load_artifact(root, "report_normal")
        with open(os.path.join(out_dir, "report.pkl"), "wb") as f:
            pickle.dump(pa, f)
    except Exception as e:
        print("[warn] 保存报告失败: %s" % str(e)[:80])

    print_metrics(metrics)
    return metrics


# --------------------------------------------------------------------------
# 指标提取
# --------------------------------------------------------------------------


def _artifact_root(recorder) -> str | None:
    """定位本次 run 的 artifact 目录。

    mlflow 的 file store 布局是：
        mlruns/<experiment_id>/<run_id>/artifacts/
    注意 run_id 是**第二层**目录名（experiment_id 是第一层），
    两种布局都要试。
    """
    import glob

    run_id = getattr(recorder, "id", None)
    if run_id is None:
        run_id = getattr(getattr(recorder, "info", None), "run_id", None)
    if run_id is None:
        return None

    bases = [os.path.join(os.getcwd(), "mlruns"),
             os.path.expanduser("~/.qlib/mlruns"),
             os.path.join(os.path.expanduser("~"), ".mlflow", "mlruns")]
    for base in bases:
        patterns = [
            os.path.join(base, "*", str(run_id), "artifacts"),
            os.path.join(base, str(run_id), "*", "artifacts"),
            os.path.join(base, str(run_id), "artifacts"),
        ]
        for pat in patterns:
            hits = sorted(glob.glob(pat))
            if hits:
                return hits[0]
    return None


def _load_artifact(root: str, *keywords):
    """按关键词在 artifact 目录里找并加载 pickle。

    qlib 0.9.7 的 artifact 布局（实测）：
        sig_analysis/ic.pkl
        sig_analysis/ric.pkl
        portfolio_analysis/port_analysis_1day.pkl
        portfolio_analysis/report_normal_1day.pkl
        pred.pkl
    名字会随版本变，所以不写死，改为遍历匹配。
    """
    if not root or not os.path.isdir(root):
        raise FileNotFoundError("artifact 目录不存在: %s" % root)

    all_files = []
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            if f.endswith(".pkl"):
                all_files.append(os.path.join(dp, f))

    for kw in keywords:
        kwl = kw.lower()
        for path in all_files:
            rel = os.path.relpath(path, root).replace("\\", "/")
            if kwl in rel.lower():
                with open(path, "rb") as fh:
                    return pickle.load(fh)
    raise FileNotFoundError("未找到 artifact: %s" % (keywords,))


def save_metrics(m: dict, name: str) -> dict:
    """写盘：JSON 存标量指标，曲线数据单独存 curves.pkl。"""
    curves = {k: m.pop(k) for k in list(m)
              if k.startswith("_") and isinstance(
                  m[k], (pd.Series, pd.DataFrame))}
    out_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2, default=str)
    if curves:
        with open(os.path.join(out_dir, "curves.pkl"), "wb") as f:
            pickle.dump(curves, f)
    return m


def collect_metrics(recorder, name, elapsed, cfg) -> dict:
    """从 recorder 提取关键指标。"""
    m = {
        "experiment": name,
        "model_class": cfg["task"]["model"]["class"],
        "handler": cfg["task"]["dataset"]["kwargs"]["handler"]["class"],
        "elapsed_sec": round(elapsed, 1),
    }

    root = _artifact_root(recorder)

    # ---- mlflow 自己记录的指标（最权威）----
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(recorder.id)
        for k, v in run.data.metrics.items():
            if "excess_return_with_cost" in k:
                m["qlib_excess_annual_return"] = round(v, 6)
            elif "excess_return_without_cost" in k:
                m["qlib_excess_no_cost_annual_return"] = round(v, 6)
        for k, v in run.data.metrics.items():
            kk = k.strip()
            if kk == "IC":
                m["mlflow_ic"] = round(v, 6)
            elif kk == "ICIR":
                m["mlflow_icir"] = round(v, 6)
            elif kk == "Rank IC":
                m["mlflow_rank_ic"] = round(v, 6)
            elif kk == "Rank ICIR":
                m["mlflow_rank_icir"] = round(v, 6)
    except Exception as e:
        m["mlflow_metric_error"] = str(e)[:120]

    # ---- 信号分析指标（IC / RankIC 序列）----
    try:
        ic = _load_artifact(root, "sig_analysis/ic", "ic.pkl")
        ric = _load_artifact(root, "sig_analysis/ric", "ric.pkl")
        if hasattr(ic, "columns") and len(ic.columns) > 1:
            ic = ic.iloc[:, 0]
        if hasattr(ric, "columns") and len(ric.columns) > 1:
            ric = ric.iloc[:, 0]
        ic = pd.Series(ic).dropna()
        ric = pd.Series(ric).dropna()
        m["ic_mean"] = round(float(ic.mean()), 5)
        m["ic_std"] = round(float(ic.std()), 5)
        m["icir"] = round(float(ic.mean() / ic.std()), 4) if len(ic) > 1 and ic.std() else None
        m["rankic_mean"] = round(float(ric.mean()), 5)
        m["rankic_std"] = round(float(ric.std()), 5)
        m["rankic_ir"] = round(float(ric.mean() / ric.std()), 4) if len(ric) > 1 and ric.std() else None
        m["ic_positive_ratio"] = round(float((ic > 0).mean()), 4)
        m["rankic_positive_ratio"] = round(float((ric > 0).mean()), 4)
        m["ic_n"] = int(len(ic))
        m["_ic_series"] = ic
        m["_rankic_series"] = ric
    except Exception as e:
        m["signal_analysis_error"] = str(e)[:150]

    # ---- 组合绩效 ----
    try:
        # report_normal_<freq>.pkl 直接就是组合指标表
        # （列：account/return/total_turnover/turnover/total_cost/cost/value/cash/bench）
        pm = _load_artifact(root, "report_normal")
        if isinstance(pm, tuple):
            pm = pm[0]
        if isinstance(pm, dict):
            # 老结构：{列名: Series}
            pm = pd.DataFrame(pm)

        ret_daily = pm["return"]
        nav = (1.0 + ret_daily).cumprod()
        initial = pm["account"].iloc[0]
        final = pm["account"].iloc[-1]

        m["initial_capital"] = float(initial)
        m["final_capital"] = float(final)
        # 累计收益（与 account 变化一致）
        m["total_return"] = round(float(final / initial - 1) * 100, 4)
        # 年化收益（按 252 交易日折算）
        m["annual_return"] = round(
            float((final / initial) ** (252 / len(nav)) - 1) * 100, 4)
        m["max_drawdown"] = round(float((nav / nav.cummax() - 1).min()) * 100, 4)
        # 日波动率（注意 return 列是**日**收益率，不是年化）
        m["daily_vol"] = round(float(ret_daily.std()) * 100, 4)
        m["annual_vol"] = round(float(ret_daily.std() * np.sqrt(244)) * 100, 4)
        m["sharpe"] = round(
            float(ret_daily.mean() / ret_daily.std() * np.sqrt(244)), 4
        ) if ret_daily.std() else None
        m["total_cost"] = round(float(pm["total_cost"].iloc[-1]), 2)
        m["daily_turnover"] = round(float(pm["turnover"].mean()), 5)
        m["n_days"] = int(len(nav))

        # 基准 —— 必须与策略同口径（累计 & 年化各一份）
        if "bench" in pm.columns:
            bnav = (1.0 + pm["bench"]).cumprod()
            m["benchmark_return"] = round(float(bnav.iloc[-1] - 1) * 100, 4)
            m["benchmark_annual_return"] = round(
                float(bnav.iloc[-1] ** (252 / len(bnav)) - 1) * 100, 4)
            m["benchmark_max_drawdown"] = round(
                float((bnav / bnav.cummax() - 1).min()) * 100, 4)
            # 超额：累计口径（与 total_return 同口径）
            m["excess_return"] = round(m["total_return"] - m["benchmark_return"], 4)
            m["excess_annual_return"] = round(
                m["annual_return"] - m["benchmark_annual_return"], 4)

        # 分市场环境（用基准的滚动波动率分位切分）
        try:
            bench_vol = pm["bench"].rolling(60).std()
            hi_mask = bench_vol >= bench_vol.quantile(0.67)
            lo_mask = bench_vol <= bench_vol.quantile(0.33)
            for tag, mask in (("high_vol", hi_mask), ("low_vol", lo_mask)):
                seg = ret_daily[mask].dropna()
                if len(seg) > 20:
                    m[f"{tag}_days"] = int(len(seg))
                    m[f"{tag}_annual_return"] = round(
                        float((1 + seg).prod() ** (244 / len(seg)) - 1) * 100, 4)
                    m[f"{tag}_sharpe"] = round(
                        float(seg.mean() / seg.std() * np.sqrt(244)), 4
                    ) if seg.std() else None
        except Exception:
            pass

        # 净值与回撤曲线（供画图）
        m["_nav"] = nav
        m["_drawdown"] = nav / nav.cummax() - 1
        m["_bench_nav"] = (1.0 + pm["bench"]).cumprod() if "bench" in pm.columns else None
    except Exception as e:
        m["portfolio_error"] = str(e)[:150]

    return m


def print_metrics(m: dict) -> None:
    print()
    print("=" * 62)
    print("  实验结果: %s" % m.get("experiment"))
    print("=" * 62)
    print("模型: %s | 因子: %s | 用时 %.0fs" % (
        m.get("model_class"), m.get("handler"), m.get("elapsed_sec", 0)))
    print()
    print("── 信号分析 ───────────────────────────")
    print("  IC 均值:        %+.5f   (std %.5f, IR %s)" % (
        m.get("ic_mean", float("nan")), m.get("ic_std", float("nan")),
        m.get("icir")))
    print("  RankIC 均值:    %+.5f   (std %.5f, IR %s)" % (
        m.get("rankic_mean", float("nan")), m.get("rankic_std", float("nan")),
        m.get("rankic_ir")))
    print("  IC 正收益率:    %s" % m.get("ic_positive_ratio"))
    print()
    print("── 组合绩效 ───────────────────────────")
    print("  总收益:         %+.2f%%" % m.get("total_return", float("nan")))
    print("  年化收益:       %+.2f%%" % m.get("annual_return", float("nan")))
    print("  最大回撤:       %.2f%%" % m.get("max_drawdown", float("nan")))
    print("  日波动:         %.2f%%  (年化 %.2f%%)" % (
        m.get("daily_vol", float("nan")), m.get("annual_vol", float("nan"))))
    print("  夏普:           %s" % m.get("sharpe"))
    print("  基准累计收益:   %s%%  (年化 %s%%)" % (
        m.get("benchmark_return"), m.get("benchmark_annual_return")))
    print("  超额收益:       %s%%  (年化差 %s%%)" % (
        m.get("excess_return"), m.get("excess_annual_return")))
    print("  日均换手率:     %s" % m.get("daily_turnover"))
    if "high_vol_annual_return" in m:
        print()
        print("── 分市场环境（按基准60日波动率分位）───────────")
        print("  高波动期年化:   %+.2f%% (夏普 %s, %d 天)" % (
            m.get("high_vol_annual_return"), m.get("high_vol_sharpe"),
            m.get("high_vol_days", 0)))
        print("  低波动期年化:   %+.2f%% (夏普 %s, %d 天)" % (
            m.get("low_vol_annual_return"), m.get("low_vol_sharpe"),
            m.get("low_vol_days", 0)))
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML 配置路径")
    ap.add_argument("--tag", default="", help="实验名（默认取配置文件名）")
    ap.add_argument("--force", action="store_true", help="已有结果也重跑")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["_config_path"] = args.config
    run_experiment(cfg, tag=args.tag, force=args.force)


if __name__ == "__main__":
    main()
