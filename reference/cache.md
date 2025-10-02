你提的顾虑是对的：如果以后要 sr(open)，甚至对“临时序列（adhoc transform）”做特征，仅按 close 缓存就会失效或重复计算。

下面给一套极简但通用的缓存设计，既能避免重复计算，又能覆盖 open/close/自定义序列 等场景；不引入复杂框架。

目标

同一 基础序列（如 close、open、或你注册的临时序列）+ 相同窗口/参数 的 SMA/STD/… 只算一次。

支持 任意列名 或 临时 Series（adhoc），后者可以“命名/绑定”后参与缓存。

仅在单次计算会话（per-call）有效，避免脏缓存。

设计要点（简单稳妥）
1) 会话级缓存（FeatureSession/Cache）

缓存在一次特征计算会话内生效，结束即丢弃。

提供统一入口：sma(selector, window, ...)、std(selector, window, ...)，其中 selector 支持：

字符串列名："close", "open" …

已“绑定”的临时序列名："my_price"

直接给 pd.Series（匿名）；匿名只能同对象复用，建议尽量绑定命名。

2) 统一的 缓存键（核心）

key = (op, series_id, window, other_params)

series_id 规则：

若是字符串（列名或绑定名）：('col', name)

若是匿名 pd.Series：('ser', id(series.values), len(series))（同一对象可复用；跨对象不保证，足够 MVP）

最稳的是命名/绑定，匿名仅作兜底。

3) 绑定临时序列（解决 adhoc）

提供 bind(name, series)：把任意临时序列注册到会话，之后用 name 引用即可参与缓存。

例如：sess.bind("mid_price", (df["bid"]+df["ask"])/2)

极简实现示例（可直接粘贴用）
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Union

Selector = Union[str, pd.Series]

class FeatureCache:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cache: Dict[Tuple, np.ndarray] = {}
        self.bound: Dict[str, pd.Series] = {}  # name -> Series

    # --- public API ---
    def bind(self, name: str, series: pd.Series) -> None:
        """Register an ad-hoc series to take part in caching."""
        self.bound[name] = series

    def sma(self, sel: Selector, w: int, exact: bool = True) -> np.ndarray:
        s, sid = self._resolve(sel)
        key = ("sma", sid, w, exact)
        if key not in self.cache:
            arr = self._fast_sma(s, w, exact)
            self.cache[key] = arr
        return self.cache[key]

    def std(self, sel: Selector, w: int, exact: bool = True, sample: bool = True) -> np.ndarray:
        s, sid = self._resolve(sel)
        key = ("std", sid, w, exact, sample)
        if key not in self.cache:
            arr = self._fast_std(s, w, exact, sample)
            self.cache[key] = arr
        return self.cache[key]

    # --- helpers ---
    def _resolve(self, sel: Selector) -> Tuple[pd.Series, Tuple[Any, ...]]:
        if isinstance(sel, str):
            s = self.bound.get(sel, self.df[sel])  # prefer bound, else df column
            sid = ("col", sel)
        elif isinstance(sel, pd.Series):
            s = sel
            sid = ("ser", id(s.values), s.size)    # good enough for session reuse
        else:
            raise TypeError("selector must be str or pd.Series")
        return s, sid

    # --- kernels (replace with your fast_SMA/fast_std) ---
    def _fast_sma(self, s: pd.Series, w: int, exact: bool) -> np.ndarray:
        r = s.rolling(w, min_periods=w if exact else 1).mean()
        return r.to_numpy(dtype="float64", copy=False)

    def _fast_std(self, s: pd.Series, w: int, exact: bool, sample: bool) -> np.ndarray:
        ddof = 1 if sample else 0
        r = s.rolling(w, min_periods=w if exact else 1).std(ddof=ddof)
        return r.to_numpy(dtype="float64", copy=False)

用法示例
sess = FeatureCache(df)

# 1) 默认 close：
ma3  = sess.sma("close", 3)
ma40 = sess.sma("close", 40)
ma100= sess.sma("close", 100)

# 2) maratio (close)
maratio_3_100 = (ma3 / np.where(ma100!=0, ma100, np.nan)) - 1.0

# 3) rank of maratio over prev 1000 days（略）

# 4) 换成 open：零改造
ma3_open  = sess.sma("open", 3)
ma100_open= sess.sma("open", 100)
maratio_open = (ma3_open / np.where(ma100_open!=0, ma100_open, np.nan)) - 1.0

# 5) adhoc：绑定临时序列后即可缓存
mid = (df["bid"] + df["ask"]) / 2
sess.bind("mid", mid)
ma_mid_5  = sess.sma("mid", 5)

为什么它能解决你的担心

sr(open) 轻松支持：把 selector 从 "close" 改成 "open" 即可；缓存键随 series_id 改变，互不污染。

adhoc 也能缓存：对任意临时序列 bind("name", series)，后续用 "name" 当 selector，自动享受缓存。

不强制复杂图谱：没有引入 DAG/框架；只是一个会话字典，极简但覆盖 90% 需求。

不会过度缓存：缓存生命周期 = 会话；不会在下一次计算里“带旧账”。