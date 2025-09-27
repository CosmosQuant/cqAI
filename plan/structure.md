# Features.py 架构总览

## 核心计算函数 (Core Functions)
```python
# 高性能时间序列计算 (Numba加速)
def run_SMA(series, window)           # 简单移动平均
def run_std(series, window)           # 滚动标准差  
def run_zscore(series, window, mu)    # 滚动Z分数
def run_SR(series, smooth_w, sr_w)    # 夏普比率
def fast_rank(series, window)         # 滚动排名

# 数据处理工具
def winsorize(series, winsor_pct)     # 异常值裁剪
def discretize(series, method, ...)   # 连续值离散化
```

## Feature类 - 主要特征计算架构
```python
class Feature:
    def __init__(df, feature_type, **kwargs)    # 初始化: df + 参数
    def calculate()                             # 主流程: 计算 → winsorize → normalize  
    def winsorize(winsorize_config)             # 异常值处理
    def normalize(normalize_config)             # 标准化处理 (rank/zscore)
    def get_feature()                           # 获取结果
    def get_name()                              # 自动生成名称
```

### 参数格式 (统一字典结构):
```python
# 异常值处理
winsorize={'pct': 0.01}

# 标准化处理 (必须显式指定method和window)
normalize={'method': 'rank', 'window': 1000}
normalize={'method': 'zscore', 'window': 500}
```

### 注册特征函数:
```python
@Feature.register("maratio")      # MA比率: MA_short/MA_long - 1
@Feature.register("sr")           # 夏普比率
@Feature.register("rsi")          # RSI指标 (-1到+1)  
@Feature.register("roc")          # 变化率
```

## FeatureOperator类 - 特征后处理
```python
class FeatureOperator:
    def apply(series) -> series       # 应用变换
    def get_name() -> str            # 生成操作名称
```

### 可用操作:
```python
ReLU(threshold, direction)        # ReLU激活
Clip(low, high)                   # 值域裁剪
Scale(mul, add)                   # 线性变换
Sign()                            # 符号提取
Abs()                             # 绝对值
Lag(k)                            # 时间滞后
Smooth(window, method)            # 平滑处理
Crossover(threshold, direction)   # 交叉信号
```

### 批量处理:
```python
operators = [ReLU(threshold=0.0), Clip(low=-1, high=1)]
result = apply_operators(series, operators)
```

## 工厂函数
```python
def generate_feature_objects(df, feature_definitions)
# 根据配置批量生成Feature对象
```

## 使用示例
```python
# 1. 基础特征计算
feature = Feature(df, 'maratio', short=3, long=100,  winsorize={'pct': 0.01}, normalize={'method': 'rank', 'window': 1000})
feature.calculate()
result = feature.get_feature()

# 2. 特征后处理  
processed = ReLU(threshold=0.0).apply(result)
final = Clip(low=-1, high=1).apply(processed)

# 3. 批量生成
features = generate_feature_objects(df, FEATURE_DEFINITIONS)
```

## 设计原则
- **统一接口**: 所有特征遵循相同的计算→处理→命名流程
- **字典参数**: 多参数方法使用字典格式，保持扩展性
- **严格验证**: 注册函数必须设置self.feature，否则报错
- **自动命名**: 根据参数自动生成feature名称 (如: maratio_3_100_w1_rank1000)
- **高性能**: 核心函数使用Numba JIT加速
