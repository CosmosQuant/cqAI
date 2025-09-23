# 📋 cqAI 实施计划

## ✅ 极简实施计划

### 🚀 **第一周 - 核心MVP**
**目标**: 跑通完整流程 `train → test → yhat → signals → pnl`

#### Day 1-2: 基础框架
- [ ] **单一入口**: `run.py` + YAML配置解析
- [ ] **data.py**: 基于现有DataHandler的简化版本
- [ ] **requirements.txt**: 最小依赖包

#### Day 3-4: 核心流程
- [ ] **features.py**: 
  - 保留现有`fast_SMA`, `fast_std`, `RSI`
  - 简单注册表: `FEATURES = {'sma': fast_SMA, 'rsi': RSI}`
  - `make_features(df, feature_list)` 函数
- [ ] **cv.py**: 
  - 抽象基类 `Splitter(split)`
  - `WalkForwardSplitter` 基础实现 (OOP扩展点)
  - `build_splitter(cv_cfg)` 工厂方法
- [ ] **models.py**: 
  - 抽象基类 `BaseModel(fit, predict)`
  - Ridge和LGB封装 (OOP扩展点)

#### Day 5-7: 完整链路
- [ ] **signals.py**: `yhat_to_signals()` 基于现有SignalGenerator
- [ ] **backtest.py**: `simulate()` 基于现有Simulator
- [ ] **analysis.py**: 基础指标计算和保存

**Week 1 DoD**: `python run.py --config configs/base.yaml` 一键运行完整流程

---

### 🔧 **第二周 - 关键扩展**
**目标**: 添加关键的OOP扩展点，保证未来发展

#### 扩展点1: 模型系统
- [ ] **models.py** 完善工厂模式
- [ ] 添加XGB, 神经网络支持
- [ ] 统一的超参数接口

#### 扩展点2: CV策略系统
- [ ] **cv.py** 完善策略模式
- [ ] 添加 `TimeSeriesCV`, `PurgedCV` 支持
- [ ] 支持多种切分策略组合

#### 扩展点3: 实验编排  
- [ ] **experiment.py** OOP设计
- [ ] 支持参数网格搜索
- [ ] 集成现有并行处理能力

#### 扩展点4: 特征注册表
- [ ] **features.py** 注册装饰器
- [ ] 支持动态添加新特征
- [ ] 特征重要性分析

**Week 2 DoD**: 具备良好扩展性，支持快速添加新模型和特征

---

## 📋 **立即开始**

1. **创建极简结构**: 按照structure.md的文件结构组织代码
2. **第一周目标**: 跑通完整的 train→test→yhat→signals→pnl 流程  
3. **第二周目标**: 添加关键OOP扩展点，保证未来可扩展

**核心理念**: 先极简能跑，再优雅扩展
