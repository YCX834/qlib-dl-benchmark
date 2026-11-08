# PyCharm 使用说明

本项目已预置好 PyCharm 配置，克隆后可直接运行。

---

## 一、环境前提

| 项目 | 值 |
|---|---|
| Python 解释器 | qlib 环境（Python 3.11.18） |
| 解释器路径 | `%USERPROFILE%\.conda\envs\qlib\python.exe` |
| 必需包 | `qlib 0.9.7` / `torch 2.14.1+cu130` / `lightgbm 4.7` / `pandas` / `matplotlib` / `pyyaml` / `mlflow` |
| 数据目录 | `%USERPROFILE%\.qlib\qlib_data\cn_data` |

验证环境是否就绪：

```cmd
conda activate qlib
python -c "import qlib, torch, lightgbm; print(qlib.__version__, torch.__version__)"
```

---

## 二、打开项目

**方式 A：命令行**

```cmd
conda activate qlib
cd /d %USERPROFILE%\Desktop\qlib-dl-benchmark
pycharm64 .
```

**方式 B：桌面启动器**

双击 `打开 Anaconda 终端 (Qlib).bat`，然后在打开的终端里敲 `pycharm64 .`

**方式 C：直接双击项目目录**

在文件资源管理器里进入 `qlib-dl-benchmark` 文件夹 → 右键 → `打开方式` → PyCharm。

---

## 三、预置的运行配置

打开项目后，**右上角运行配置下拉框**里已有 8 个配置，点击 ▶ 即可运行：

| 运行配置 | 作用 | 耗时 |
|---|---|---|
| **LightGBM (Alpha158)** | 树模型基线 | ~5 min |
| **LSTM (Alpha360)** | LSTM 主实验 | ~5 min |
| **Transformer (Alpha360)** | Transformer 主实验 | ~15 min |
| LSTM small (ablation) | LSTM 降容量消融 | ~40 min |
| Transformer tuned (ablation) | Transformer 调优消融 | ~30 min |
| Refresh metrics (no retrain) | 从已有实验重建指标（**不重训，秒级**） | < 1 min |
| Plot report (figures) | 出图 → `results/figures/` | < 10 s |
| Collect metrics (CSV) | 汇总 → `results/metrics.csv` | < 10 s |

> 💡 **想快速看结果**：先跑 `Refresh metrics (no retrain)`，再跑 `Plot report (figures)`。
> 不用重训就能拿到全部指标和 5 张图。

---

## 四、命令行等价命令

所有运行配置都对应一条命令（在项目根目录、qlib 环境下执行）：

```cmd
python scripts\run_experiment.py --config configs\lightgbm_alpha158.yaml
python scripts\run_experiment.py --config configs\lstm_alpha360.yaml
python scripts\run_experiment.py --config configs\lstm_alpha360_small.yaml
python scripts\run_experiment.py --config configs\transformer_alpha360.yaml
python scripts\run_experiment.py --config configs\transformer_alpha360_tuned.yaml

python scripts\refresh_metrics.py lightgbm_alpha158 lstm_alpha360 transformer_alpha360
python scripts\plot_report.py
python scripts\collect_metrics.py --markdown
```

---

## 五、结果去哪看

```
results/
├── metrics.csv                        # 5 个模型的横向对比表
├── figures/
│   ├── fig1_cumulative_return.png     # 累计净值对比
│   ├── fig2_drawdown.png              # 回撤曲线
│   ├── fig3_regime.png                # ⭐ 高低波动期表现对比
│   ├── fig4_metrics_bar.png           # 指标柱状图
│   └── fig5_ic_timeseries.png         # 滚动 RankIC 时序
├── lightgbm_alpha158/metrics.json     # 单实验详细指标
├── lstm_alpha360/metrics.json
├── lstm_alpha360_small/metrics.json
├── transformer_alpha360/metrics.json
└── transformer_alpha360_tuned/metrics.json
```

---

## 六、常见问题

### Q1：PyCharm 里 `import qlib` 标红

解释器没选对。右下角 → `Add New Interpreter` → `Existing environment`
→ 选 `%USERPROFILE%\.conda\envs\qlib\python.exe`。

项目里的 `.idea/qlib.xml` 已经声明了名为 `qlib` 的 SDK，如果下拉里能选到就直接用。

### Q2：运行报 `ModuleNotFoundError: No module named 'qlib'`

同上，解释器问题。也可以在运行配置里改：
`Run` → `Edit Configurations` → `Python interpreter` → 选 `qlib`。

### Q3：运行报编码错误（UnicodeEncodeError）

在运行配置的 `Environment variables` 里加一条：

```
PYTHONIOENCODING=utf-8
```

（预置的 8 个配置已经加好了。）

### Q4：中文路径导致 PyCharm 启动异常

JVM 的 `sun.jnu.encoding = GBK`，中文用户名路径会解码成乱码。
解决：把 IDE 目录移到纯 ASCII 路径。

```cmd
set PYCHARM_PROPERTIES=C:\pycharm_data\idea.properties
pycharm64 .
```

`C:\pycharm_data\idea.properties` 内容：

```properties
idea.config.path=C:/pycharm_data/config
idea.system.path=C:/pycharm_data/system2
idea.log.path=C:/pycharm_data/log
idea.plugins.path=C:/pycharm_data/plugins
idea.paths.selector=PyCharmCE2025.2
```

> ⚠️ **必须用正斜杠**。Java 的 `.properties` 文件里 `\` 是转义符，
> `C:\pycharm_data\config` 会被解析成 `C:pycharm_dataconfig`（相对路径），
> 导致 IDE 在项目目录下误建目录、拿不到锁而启动失败。

### Q5：`.port` 文件删不掉 / 报 AccessDeniedException

`C:\pycharm_data\system\.port` 可能是失效的 reparse point（符号链接）。
换一个新的 system 目录名即可：

```properties
idea.system.path=C:/pycharm_data/system3
```

### Q6：训练很慢 / 想先看结果

Alpha158 首次加载要 5-8 分钟（158 个表达式 × 300 只 × 12 年），
之后命中磁盘缓存会快很多。DL 模型训练在 GPU 上，Transformer 约 15 分钟。

**最快的验证路径**：`Refresh metrics` → `Plot report`，不重训直接出结果。

---

## 七、目录结构

```
qlib-dl-benchmark/
├── configs/                    # 实验配置（唯一的变量都在这里）
│   ├── lightgbm_alpha158.yaml
│   ├── lstm_alpha360.yaml
│   ├── lstm_alpha360_small.yaml
│   ├── transformer_alpha360.yaml
│   └── transformer_alpha360_tuned.yaml
├── scripts/
│   ├── run_experiment.py       # 统一运行器（三模型走同一代码路径）
│   ├── refresh_metrics.py      # 从已有实验重建指标（不重训）
│   ├── collect_metrics.py      # 指标汇总 → results/metrics.csv
│   ├── plot_report.py          # 出图 → results/figures/
│   └── push_to_github.py       # 一键推 GitHub
├── docs/                       # 分析文档
├── notebooks/                  # Jupyter notebook
├── results/                    # 实验结果（指标 + 图表）
├── qlib_src/examples/          # qlib 官方 examples 快照（参考用）
└── .idea/
    ├── qlib.xml                # 解释器定义（qlib）
    └── runConfigurations/       # 8 个预置运行配置
```
