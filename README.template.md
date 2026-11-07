# qlib-dl-benchmark

> **树模型 vs 深度学习：A 股量化实证对比**
>
> 用 [Microsoft Qlib](https://github.com/microsoft/qlib) 官方框架，在严格控制变量的前提下，
> 对比 LightGBM（梯度提升树）与 LSTM / Transformer（深度学习）在 A 股沪深300 上的量化表现。

---

## 这是什么

一个**可复现**的量化实验仓库。不含手工调过的数字 —— 每一个指标都由
`scripts/` 下的脚本从原始回测结果中提取。

**核心问题**：深度学习在金融时间序列上到底行不行？

支持者说「LSTM/Transformer 能建模时序依赖」，反对者说「金融信噪比太低，
深度学习只会记住噪声」。公开的对比研究大多没有控制变量 ——
换数据集、换调参、换标签、换成本，最后声称「深度学习不 work」。

**本项目的做法**：把除模型本身以外的所有变量固定，只让模型变。

| 控制项 | 设定 |
|---|---|
| 数据集 | 沪深300，2008-01-01 ~ 2020-08-01 |
| 时间划分 | train 2008-2014 / valid 2015-2016 / test 2017-2020.08 |
| 交易成本 | open 5bp / close 15bp / 最低 5 元 / 涨跌停 9.5% |
| 回测引擎 | qlib SimulatorExecutor（日频） |
| 组合构建 | TopkDropout，topk=50，n_drop=5 |
| **模型** | LightGBM / LSTM / Transformer ← 唯一变量 |

---

## 快速开始

```bash
conda create -n qlib python=3.11 -y
conda activate qlib
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu130

# 数据（约 700MB，一次即可）
python -c "
from qlib.tests.data import GetData
from qlib.constant import REG_CN
GetData().qlib_data(target_dir='~/.qlib/qlib_data/cn_data', region=REG_CN, exists_skip=True)
"

# 跑实验
python scripts/run_experiment.py --config configs/lightgbm_alpha158.yaml
python scripts/run_experiment.py --config configs/lstm_alpha360.yaml
python scripts/run_experiment.py --config configs/transformer_alpha360.yaml

# 出报告
python scripts/collect_metrics.py --markdown
python scripts/plot_report.py
```

---

## 目录结构

```
configs/                      实验配置 —— 唯一的变量都在这里
├── lightgbm_alpha158.yaml        基线（树模型 + 截面特征）
├── lstm_alpha360.yaml             深度学习（时序特征）
└── transformer_alpha360.yaml
scripts/
├── run_experiment.py         统一运行器（三模型走同一代码路径）
├── refresh_metrics.py        从已有 mlflow 实验重建指标（不重训）
├── collect_metrics.py        指标汇总 → results/metrics.csv
└── plot_report.py            出图 → results/figures/
results/
├── metrics.csv               所有实验的指标汇总
├── figures/                  README 引用的图
└── <实验名>/                 每个实验的原始产物
qlib_src/examples/            Qlib 官方 examples 快照（对照用）
docs/analysis.md              详细结果分析
```

---

## 三个技术要点

### 1. Alpha158 vs Alpha360：截面特征 vs 时序特征

| | Alpha158（树模型用） | Alpha360（深度学习用） |
|---|---|---|
| 形状 | 158 维标量 | (60 天, 6 字段) 序列 |
| 每个特征用到的历史 | 当天 + 过去 ≤60 天 | 过去 60 天的完整序列 |
| 滚动窗口 | `[5, 10, 20, 30, 60]` | 固定 60 |
| 归一化 | 几乎全部除以当天 `$close` | `RobustZScoreNorm` |

Alpha158 把「过去 60 天」压成 158 个标量，Alpha360 保留「过去 60 天」本身。
**强行让 LSTM 吃 Alpha158 不是公平对比，而是削弱对手** —— 因为特征表示
必须匹配模型架构，这是架构决定的，不是实验设计缺陷。

### 2. CSRankNorm：深度学习在金融上能 work 的关键

| | LightGBM | LSTM / Transformer |
|---|---|---|
| 标签处理 | 无（直接回归原始收益） | **`CSRankNorm`**（截面排名归一化） |

它把「预测收益率」转成「预测截面排名」，把回归问题变成排序问题。
收益的绝对值噪声极大，但相对排序的信息相对稳定。

### 3. 交易成本是决定性的

日均换手率 ~20%，871 个交易日累计成本 2341 万元 —— 相当于期末资产的 11.5%。
**任何换手率高的策略，成本的量级足以吞掉一半以上的 alpha。**

---

## 主要发现

> 完整的数字与分析见 [docs/analysis.md](docs/analysis.md)

1. **基线复现成功** —— 我们的 LightGBM 指标（IC 0.046 / RankIC IR 0.40 /
   年化超额 12.3%）与 Qlib 官方论文报告区间完全吻合，说明实验设置可信。

2. **⭐ 市场环境的影响远大于模型选择** —— 同一个 LightGBM 策略，
   在低波动期年化 +57.6%，在高波动期只有 −1.2%，**差了 58 个百分点**。
   这比「哪个模型更强」更有实践意义。

---

## License

代码 MIT License。`qlib_src/` 为 Microsoft Qlib 官方代码快照，遵循其原 LICENSE（MIT）。
