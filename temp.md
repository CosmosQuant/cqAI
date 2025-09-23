总体架构

```markdown
CQML/
  data/                     # 原始/中间/结果数据（按子目录划分）
  results/
  configs/
    exp_tsla_15m.yaml       # 实验配置（数据路径、CV、模型、交易参数）
  src/
    io.py                   # 仅做数据I/O + 基本校验（函数式）
    features.py             # 特征工程（函数式，可按block注册）
    labels.py               # 标签生成（函数式）
    cv.py                   # 各类时间切分（类/策略模式）
    models/
      base.py               # ModelWrapper 抽象类（OOP）
      sklearn_wrap.py       # 具体模型封装（Ridge/LGB/XGB/MLP）
    strategy.py             # 由 yhat → signal → position（函数式+小型类）
    backtest.py             # 撮合/成本/PNL（函数式）
    metrics.py              # 评估与汇总（函数式）
    experiment.py           # ExperimentRunner（OOP，组织全流程）
    utils.py                # 日志、seed、定数、计时
  scripts/
    main.py
  run_experiment.py         # CLI 入口：python run_experiment.py --config configs/exp_tsla_15m.yaml
  README.md

```

# 总体 DoD（Definition of Done）

- 配置化运行：`python run_experiment.py --config configs/exp_tsla_15m.yaml` 一键复现。
- 产物可追溯：`results/<exp_id>/` 下包含 `metrics.csv`、`trades.parquet`、`holdout_metrics.json`。
- 严格无泄漏：fit/标准化/特征选择仅在训练窗内完成；验证/holdout 仅 transform/predict。
- 报告：1 页摘要图（累计收益、滚动 IC/Sharpe、时段分组收益表）。

---

# To-Do 清单（按模块）

## 0) 基础环境与工具

- [ ]  建仓库与目录树；初始化 `pyproject.toml`/`requirements.txt`（pandas, numpy, scikit-learn, lightgbm/xgboost 可二选一, pyyaml, matplotlib, numba 可选）
- [ ]  `utils.py`：日志（简单 logger）、随机种子、时间计时器
- [ ]  `README.md`：安装、数据放置、如何运行

---

## 1) 数据 I/O 与清洗（函数式）

**文件：`src/io.py`**
- [ ] `load_ohlcv(path) -> pd.DataFrame`

- 列标准化：`timestamp, open, high, low, close, volume`
- 时间索引、缺失/重复处理、时区对齐（纽约时区）
- [ ] `align_and_clean(df: pd.DataFrame) -> pd.DataFrame`

- 生成基础 `ret_1m = log(close/close.shift(1))`
- 异常值/零成交过滤策略（可留 TODO 注释）

**DoD：** 读取 TSLA 1m CSV，输出干净 DataFrame（无重复索引，基本列齐全）。

---

## 2) 特征工程（函数式 + 注册表）

**文件：`src/features.py`**
- [ ] 特征注册器
`python   FEATURE_REGISTRY = {}   def register_feature(fn): FEATURE_REGISTRY[fn.__name__] = fn; return fn`
- [ ] 基础特征函数（右对齐，严格滚动）

- [ ] `feat_ret_lag(df, lags=[1,5,15])`
- [ ] `feat_ma(df, windows=[3,10,30])`; `feat_rsi_14(df)`；`feat_atr_14(df)`
- [ ] `feat_range_rank(df, window=20)`（压缩度）
- [ ] `feat_vol_z(df, window=60)`（量能异常）
- [ ] `feat_kbar_parts(df)`（实体/上下影线占比）
- [ ] `feat_time_of_day(df)`（minute_of_day 周期编码；开盘/收盘布尔）
- [ ] `make_features(df, selected: list[str]|None=None) -> pd.DataFrame`

- 仅计算 `selected` 列表指定的特征；合并输出 `X_df`

**DoD：** 给定原始 df，`make_features` 输出与索引对齐的 `X_df`，无未来漏用。

---

## 3) 标签构建（函数式）

**文件：`src/labels.py`**
- [ ] `make_forward_return(df, horizon_min: int) -> pd.Series`

- `y_t = log(C_{t+H}/C_t)`，使用 `.shift(-H)`
- 对齐索引、尾部生成 NA
- [ ]（可选）`make_multi_horizon_labels(df, horizons=[5,15,30]) -> dict[int, pd.Series]`

**DoD：** 给定 horizon=15min 能生成 `y15` 序列，与 `X_df` 可 `dropna` 对齐训练。

---

## 4) 时间切分 / 交叉验证（OOP 策略）

**文件：`src/cv.py`**
- [ ] 抽象协议/基类
```python
from typing import Iterable, Tuple
import numpy as np, pandas as pd

class Splitter:
def split(self, df: pd.DataFrame) -> Iterable[Tuple[np.ndarray, np.ndarray]]: …
```- [ ]`WalkForwardSplitter(train_days, val_days, step_days, embargo_min=0)`- [ ] 工厂方法`build_splitter(cv_cfg: dict) -> Splitter`

**DoD：** `split()` 能在样例数据上产出数对 `(train_idx, val_idx)`，且无重叠泄漏。

---

## 5) 模型封装（OOP，多态）

**文件：`src/models/base.py`**
- [ ] `ModelWrapper` 抽象类
`python   class ModelWrapper:       name: str = "base"       def fit(self, X, y): ...       def predict(self, X): ...`

**文件：`src/models/sklearn_wrap.py`**
- [ ] `RidgeModel(params: dict)`（带 `StandardScaler` 管道）
- [ ] `ElasticNetModel(params: dict)`（带 `StandardScaler`）
- [ ] `LGBModel(params: dict)` 或 `XGBModel(params: dict)`（二选一先实现）
- [ ] `MLPModel(params: dict)`（两层小网络 + 早停，可后补）
- [ ] `build_model(model_cfg: dict) -> ModelWrapper`（工厂）

**DoD：** 至少两个可用模型（Ridge + LGB/XGB），接口一致，fit/predict 正常。

---

## 6) 由 Yhat 生成信号（函数式，首版简单）

**文件：`src/strategy.py`**
- [ ] `yhat_to_signal(yhat: pd.Series, vol: pd.Series, thresh=0.0, k=0.5, pmax=1.0) -> pd.Series`

**DoD：** 输入 yhat/vol 能稳定给出 pos（[-pmax, pmax]），无向前看。

---

## 7) 回测与成本（函数式）

**文件：`src/backtest.py`**
- [ ] `simulate(df, pos, cost_bp=1.0, slip_bp=2.0, hold_min=15) -> dict`

**DoD：** 能产出逐分钟 PnL 与成交明细；成本可配置；边界时段不出错。

---

## 8) 指标评估（函数式）

**文件：`src/metrics.py`**
- [ ] 基础指标（Sharpe、MDD、胜率等）
- [ ] `summarize(res_dict) -> dict`

---

## 9) 实验编排（OOP）

**文件：`src/experiment.py`**
- [ ] `ExperimentRunner(cfg)`

---

## 10) CLI 入口与配置

**文件：`run_experiment.py`**
- [ ] 解析配置，运行 Runner

**文件：`configs/exp_tsla_15m.yaml`（样例）**

---

## 11) 可视化与报告

- [ ]  报告三张图：累计收益、滚动 IC/Sharpe、时段收益

---

## 12) 质量与健壮性

- [ ]  单元测试，关键逻辑检验
- [ ]  TODO 性能优化（numba）

---

# 代码接口样例

（略，已在上面详细写出 cv.py / models/base.py / sklearn_wrap.py / experiment.py 的代码骨架）

---

# 最小可交付路线

**P0（跑通）**

- [ ] IO + 3-5 特征 + 15m 标签 + WalkForwardSplitter

- [ ] Ridge + LGB 两模型

- [ ] 简单 strategy + backtest

- [ ] Runner 跑一组配置，落地 metrics 与图

**P1（增强稳健）**

- [ ] 多 horizon（5/15/30）

- [ ] 更多特征与成本敏感性

- [ ] Holdout 严格评估

**P2（工程化与展示）**

- [ ] 轻量测试、日志

- [ ] 摘要图/报告 + README