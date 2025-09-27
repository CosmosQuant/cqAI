
### 📁 Structure and architecture for bigger quant_equity AI project

quant_equity/
├─ core/
│  ├─ config.py            # 全局配置 dataclass / YAML 解析
│  ├─ types.py             # 别名类型、常量（e.g., BarDF, XSecDF）
│  ├─ utils.py             # 通用工具（time, logging, PIT校验）
│  └─ schedule.py          # Rebalance / Delay 节拍器
├─ data/
│  ├─ schemas.py           # 数据列规范：PIT字段 asof_date, effective_date, sid
│  ├─ loader.py            # 数据加载（行情/基本面/行业/风险暴露）
│  ├─ feature_store.py     # FeatureStore（版本化、延迟、缓存、落盘）
│  └─ universe.py          # Universe选择与过滤（流动性、行业、国家）
├─ transforms/
│  ├─ ops.py               # neutralize, scale, winsorize, decay, zscore
│  └─ risk_filters.py      # beta/industry/sector 中性化、capacity/cap 限制
├─ alpha/
│  ├─ base.py              # AlphaModel 抽象类（predict 返回 ŷ & conf）
│  ├─ alphas/
│  │  └─ channel_break.py  # 例：CB 策略的横截面打分
│  └─ blender.py           # 多 alpha 融合（加权/stacking/贝叶斯）
├─ portfolio/
│  ├─ risk_model.py        # 协方差/因子暴露接口（Barra-like 占位）
│  ├─ optimizer.py         # 优化器（目标：E[ret]-Cost，含约束）
│  ├─ cost_model.py        # 交易成本/冲击（IS/TCA 反馈）
│  └─ position_rules.py    # 仓位规则（上下限、turnover 控制）
├─ exec/
│  ├─ broker_stub.py       # OMS/执行桩（for backtest/paper）
│  ├─ routers.py           # VWAP/TWAP/POV/liq-seeking 桩
│  └─ slippage.py          # 滑点模型（回测用）
├─ backtest/
│  ├─ engine.py            # Delay-1 X-section 回测引擎（EOD）
│  ├─ walkforward.py       # Anchored Walk-Forward / Purged KFold
│  └─ metrics.py           # IR/Sharpe/turnover/TE/归因
├─ ops/
│  ├─ registry.py          # Alpha Registry（元数据、版本、开关）
│  ├─ monitor.py           # 线上监控（覆盖率、延迟、成本偏差）
│  └─ lineage.py           # 审计/可追溯（数据→特征→模型→持仓）
├─ cli.py                  # 命令入口：research/backtest/paper/prod
└─ settings.yaml           # 全局 YAML 配置（路径、超参、约束）


--- config
```python
# core/config.py
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(frozen=True)
class DataConfig:
    root: str
    calendar: str = "XNYS"
    use_adjusted: bool = True

@dataclass(frozen=True)
class RiskConfig:
    neutralize: str = "industry+market"   # none|market|industry|both
    target_vol: float = 0.05
    max_gross: float = 1.0
    max_single_name: float = 0.01

@dataclass(frozen=True)
class CostConfig:
    fee_bps: float = 0.5
    impact_k: float = 0.1

@dataclass(frozen=True)
class ExecConfig:
    mode: str = "close"    # close|next_open
    lot: float = 1.0

@dataclass(frozen=True)
class BacktestConfig:
    start: str
    end: str
    delay: int = 1         # delay-1 为日频金线
    rebalance: str = "1D"  # 每日重平

@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    risk: RiskConfig
    cost: CostConfig
    exec: ExecConfig
    backtest: BacktestConfig
    universe: List[str] = field(default_factory=lambda: ["US_TOP1500"])

```