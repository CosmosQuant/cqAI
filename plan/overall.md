# 🚀 cqAI - 极简量化ML系统

## 📋 项目综述

**cqAI v3.0** - 极简设计，专注核心ML流程，保留关键扩展点

**核心流程**: Data → Features → Labels → **[Models×CV Loop]** → Results → Analysis

---

### Principles:
- Follows KISS principle: Keep it simple, stupid
- YAGNI 原则：You Aren't Gonna Need It 
- Every time, after the initial try, always follow by a step with eliminating unnecessary complexity


---

### ✅ **极简MVP原则**
- **单一入口**: `python cqAI_main.py` (配置内嵌)
- **函数式主体**: 数据处理、特征、信号、回测都是纯函数，数据流清晰
- **单一职责**: 每个文件只做一件事，职责明确
- **配置驱动**: 所有参数外置到YAML，便于实验管理
- **最小依赖**: pandas, numpy, scikit-learn, lightgbm, pyyaml

### 🔧 **关键OOP扩展点** (未来发展必需)
- **models.py**: 统一模型接口，支持任意算法扩展 (sklearn/LGB/XGB/神经网络)
- **cv.py**: 时间切分策略，支持多种CV方法 (WalkForward/TimeSeriesCV/PurgedCV)
- **experiment.py**: 实验编排OOP，支持复杂实验设计 (参数搜索、并行)
- **features.py**: 注册表模式，支持动态特征添加和组合

### ⚙️ **严格无泄漏保证**
- **时间序列切分**: train/test严格按时间顺序，防止未来信息泄漏
- **fit/transform分离**: 标准化、特征选择只在训练集fit，测试集仅transform
- **前瞻标签**: 使用`.shift(-H)`生成标签，严格无未来信息

### 🚀 **保留现有价值**
- **技术指标**: 现有的`fast_SMA`, `fast_std`, `RSI`等高效实现
- **并行处理**: `multiprocessing`框架，支持大规模参数搜索
- **回测逻辑**: `Simulator`的成本和滑点计算，经过验证的交易逻辑
- **分析工具**: `DailyResultAnalyzer`的可视化和报告功能

---

## 🎯 **设计核心理念**

### 📈 **数据流驱动**
整个系统围绕清晰的数据流设计：`raw data → features → labels → model → yhat → signals → pnl → analysis`。每一步都是纯函数转换，便于测试和调试。

### 🔧 **先简后扩**
- **第一阶段**: 极简MVP，快速跑通完整流程
- **第二阶段**: 在关键扩展点添加OOP设计，支持复杂需求
- **核心思想**: 先能跑，再优雅

### 🎛️ **配置化实验**
所有实验参数都外置到YAML配置文件，支持：
- 一键复现实验结果
- 批量参数搜索
- 实验版本管理
- 结果可追溯

### 🔒 **严格ML规范**
严格遵循机器学习最佳实践，防止数据泄漏，确保模型评估的可靠性。

---

## 📋 **立即开始**

1. **查看结构**: 参考 [structure.md](structure.md) 了解项目文件组织
2. **实施计划**: 参考 [todo.md](todo.md) 开始2周开发计划
3. **核心目标**: 先跑通完整的 `train→test→yhat→signals→pnl` 流程
4. **扩展准备**: 在关键点预留OOP扩展接口，保证未来可扩展

**核心理念**: 极简MVP + 关键扩展点 = 快速迭代 + 长期可维护

---

**项目名称**: cqAI v3.0 - 极简量化ML系统  
**设计原则**: 极简MVP + 关键扩展点  
**实施目标**: 2周内跑通完整流程并具备扩展性  
**最后更新**: 2025年1月


1. src/data.py - 数据加载模块
load_ohlcv() - 从CSV文件加载OHLCV数据
validate_ohlcv() - 数据质量验证
支持多种CSV格式，自动标准化列名

2. src/features.py - 特征工程模块
make_features() - 根据特征列表创建技术指标
make_labels() - 创建前瞻收益标签
create_technical_features() - 创建全套技术指标
支持SMA、EMA、RSI、布林带、MACD、ATR等指标

3. src/models.py - 机器学习模型模块
create_model() - 根据配置创建模型
LightGBMWrapper - LightGBM包装器
evaluate_predictions() - 模型评估
EnsembleModel - 集成模型
支持Ridge、线性回归、随机森林、LightGBM

4. src/cv.py - 交叉验证模块
build_splitter() - 构建交叉验证分割器
WalkForwardCV - 滚动前向交叉验证
PurgedKFoldCV - 净化K折交叉验证
SeasonalCV - 季节性交叉验证
专为时间序列设计，防止数据泄露

5. src/signals.py - 信号生成模块
yhat_to_signals() - 预测转信号
predictions_to_positions() - 预测转仓位
apply_signal_filters() - 信号过滤
optimize_signal_threshold() - 阈值优化
create_ensemble_signals() - 集成信号

6. src/backtest.py - 回测模块
simulate() - 交易模拟，考虑交易成本和滑点
calculate_performance_metrics() - 性能指标计算
run_backtest() - 完整回测流程
portfolio_simulation() - 投资组合模拟
walk_forward_analysis() - 滚动前向分析

7. src/analysis.py - 结果分析模块
compute_cv_metrics() - 交叉验证指标计算
analyze_all_results() - 全面结果分析
save_experiment_results() - 结果保存
create_performance_plots() - 性能图表
generate_report() - 生成分析报告
