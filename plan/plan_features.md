
## 🎯 Features Engineering Plan
### **Phase 1: Core Functions**
### **Phase 2: Create Feature Definition Architecture**
### **Phase 3: Auto-Generate Feature Registry**

## 🎯 Thoughts
- 哪类用来定义state，哪类用来预测fwdret,分类清楚应该会更有效
- 制作一本cookbook - 可以做reference，也可以找工作的时候给别人看 （包括具体的单feature的衡量，包括feature combination的demo）
- 动手能力非常重要 - 我是最早一批开始研究使用xgboost的，但是却迟迟没有做到可实盘的产品
- *不急着上很多categories - 更重要的是每个类别（e.g. direction, vol, VB, extreme）把研究的流程给build up 好*
- *感觉可以专注Breakout一类，可能会延伸出来10-20个features来做第一波研究 - rule-based to AI*
  - 先做好one pager evaluation
  - 可以尝试xgboost先
  - 然后尝试 state (VB的情况) + MOM/MR

## 🎯 Thoughts - rule-based -> modern features
[整个流程可以尽量的automatic] [more_details_on_notion]
- **Feature with Combined Components** Use State × Value with soft, dynamic gates and quality scalers
    - *要仔细想想到底有没有predictbility, predictability来源于State还是Value*
        - In rule-based PPT signal （连暴跌4天或以上，抄底1天: setup:=连续暴跌4天，这个本身是State么，还是Value,我们可以这样思考：        
            - like State: Setup=1 then other features (e.g. ret_22 or sr_5) 跟 fwdret1 highly correlated;
            - like Value: avg fwdret1 在Setup=1 vs 0的时候显著不一样，而不需要其它的Value来determine fwdret1了
            - Hybrid: setup=1的情况下，ret_4以及到底连续暴跌了几天都跟fwdret1很相关（比如ret_4越negative, fwdret1越大）
                - 这个例子下，不需要引进更多的feature，变成modern的方式应该都会增加performance

    - *x to f(x) - 更细化的区间划分* 传统的momentum {I(above) - I(below)} to {s_far(s) *f(x) + s_near(s) *f(x) + s_extreme(s) *f(x)}
        - 用神经网络加一层自然是最理想的
        - 但是，pre-define 几个不同的thresholds (比如extreme_2, extreme_2.5) 倒是兼顾了dynamic (可以很快的capture到最近的动态)和避免过拟合（loop 各种thresholds） 
        - 跟选择一个thresholds相比，在threshold附近定义一个sigmoid function或者类似的用连续代替binary的做法会不会更好？

- [ ] start to write codes for AI reformation of used strategies before continue the AI pipeline (this is more longer term prj) - CB, PPT, Pelican, MOM, and EOD intraday patterns (C/VWAP)
        - adding features based on above mentioned needs
        - adding essential quant functions (sr, roc, skew, rank, max, min) earlier

### More to Think:
- **Evaluation**
    - robustness with small changes in parameters
### Decisions to Make
- [] timestamp - 要不要日内和ID的保持统一，都保留真实的时间(e.g. daily time still as 20250926 16pm) 

### 📝 TODO List
- [] **Add caching to avoid duplicating calculation**
- [] **Add more features** - see plans in notion - 每一个都可以很延伸和细化（e.g. VB）
     - *Direction (ROC/SR/MA/RSI)* - strength, cleaness, volume confirmation - 或许一些变化会让简单的因子更有效，比如ROC5变成0附近平滑，两头也平滑，中间更陡
     - *Combined Direction* - Consensus trend (diff freq or mkts), Pullback-in-trend ()
     - *Regime (ATR/SD/Range/Kurtosis + realized-vol estimators)* - vol, vol of vol, NR7 (extend: %time of extreme)
     - *Convexity* - continuously up or dn, or stay above for n days
     - *Market Sentiment from OHLCV* - *Shock or Extreme*, *Drawdown Geometry - Drawdown depth/speed*, *Crowdness*
     - *Structure (wick/body/CLV/position-in-range)*
     - *Participation (volume/dollar vol/rel vol)*
     - *Pressure from Inventory or Position/Valuation*
     - *Calendar (weekday/EOP/EOM + simple dummies)*
     - *Flow (from calendar or other resources)*
     - *Context (breadth/dispersion; stock–sector residual)*
     - *Overall Risk On Off - measured by common macro factors/markets* - rates, dollar, breadth, macro
- [] **Likely redundancy**-multiple closely related SD/range ratios at many (xx, nn) pairs. Consider paring via correlation screening or learning a small set of orthogonal composites
- [] **Monotone transforms for interpretability & stability** Apply signed ranks, winsorized z-scores, or logistic transforms to heavy-tailed pieces like kurtosis or ratio features -  *loop and determine the best needs?*
- [] **feature verification** - compare R ouput vs python output one by one
- [] **one pager evaluation of single feature** - [continous_vs_categorical] [regime_daynight_vol_] [singleMarket_vs_multiMarkets]
- [O] Feature Engineering (optional - careful evaluation is much more important than creating more garbage)
- [] **Add more features** - see plans in notion
    - *return comparison at diff clocks* - Overnight vs intraday returns & vol (close→open, open→close); gap size and gap fill ratio
    - *RV vs IV*
    - *Vol-of-vol & compression diagnostics*
    - *Breakout anatomy* - Days since HH/LL, % above/below recent HH/LL, false-break score (breaks HH intraday, closes back in range)
    - *cross-sectional or vs sector/universe* or vv paired markets
    - *Simple risk-appetite proxies (index-level)* - Without buying options data: realized vol term-structure slope (short vs long window), drawdown depth & speed. These are decent stand-ins for “risk-on/off” conditioning.

