### 📁 核心文件结构
```
cqAI/
├── data/                    # 数据 (保留现有btc_csv)
├── results/                 # 实验结果 (按日期组织)
├── configs/                 # YAML配置文件
│   └── base.yaml           # 基础实验配置
├── src/
│   ├── data.py             # 数据加载 (函数式，基于现有DataHandler)
│   ├── features.py         # 特征计算 (函数式 + 注册表扩展点)
│   ├── cv.py               # 时间切分策略 (OOP扩展点，策略模式)
│   ├── models.py           # 模型封装 (OOP扩展点，基于现有架构)
│   ├── signals.py          # yhat→signals (函数式，基于现有SignalGenerator)
│   ├── backtest.py         # signals→pnl (函数式，基于现有Simulator)
│   ├── analysis.py         # 结果分析 (基于现有DailyResultAnalyzer)
│   └── experiment.py       # 实验编排 (OOP扩展点)
├── run.py                  # 单一入口: python run.py --config configs/base.yaml
└── requirements.txt        # 最小依赖
```
