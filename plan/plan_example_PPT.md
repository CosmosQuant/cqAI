# PPT (Parametric Prediction Trading) Implementation Plan

## Overview
Convert classical rule-based trading signal to modern AI approach using Gate × Driver × Quality architecture with JAX/PyTorch backend.

## Phase 1: Core Infrastructure (MVP) ✅ COMPLETED
### 1.1 Data & Feature Engineering
- [x] Basic OHLC data loader (synthetic data generation)
- [x] Core features: returns, ATR, volatility, run-length, compression, momentum
- [x] Feature normalization (z-score with winsorization)
- [x] Forward return labels for horizons H ∈ {1,2,3}

### 1.2 Model Components
- [x] Parameter mapping helpers (map_box, map_symmetric)
- [x] Gate component (run-length + drawdown gates)
- [x] Driver component (drawdown magnitude)
- [x] Quality component (compression + trend)
- [x] Head components (mu_hat, p_hat predictions)
- [x] PPTModel (Gate × Driver × Quality → Heads)

### 1.3 Training Infrastructure
- [x] Loss functions (Huber + BCE)
- [x] Basic optimizer setup (L-BFGS-B via SciPy)
- [x] Single window training function
- [x] Model parameter management (27 parameters total)

## Phase 2: Backtesting & Evaluation
### 2.1 Rolling Training
- [ ] Date grid generation (monthly retraining)
- [ ] Rolling window training loop
- [ ] Model checkpointing

### 2.2 Decision & Backtesting
- [ ] Decision logic (score > 0 & phat > threshold)
- [ ] Simple backtest (binary decisions)
- [ ] Performance metrics (IC, Hit rate, PnL, Turnover)

## Phase 3: Advanced Features (Future)
- [ ] Monotonic splines (JAX implementation)
- [ ] Probability calibration (isotonic regression)
- [ ] Group lasso regularization
- [ ] Advanced backtesting with costs

## Implementation Strategy
1. **Start Simple**: Use linear transformations instead of splines
2. **PyTorch First**: Implement with PyTorch, migrate to JAX later
3. **Test Early**: Synthetic data testing before real data
4. **Modular Design**: Each component independently testable

## File Structure
```
run_PPT.py          # Main implementation
src/ppt_model.py    # Model components (Gate/Driver/Quality)
src/ppt_data.py     # Data loading and feature engineering
src/ppt_train.py    # Training and evaluation utilities
```

## Success Criteria
- [x] Model trains without errors on synthetic data ✅
- [x] Produces reasonable predictions (mu_hat: 0-0.04, p_hat: 0.48-0.91) ✅
- [ ] Rolling backtest completes successfully
- [ ] Performance metrics calculated correctly

## Current Status
**Phase 1 MVP Successfully Completed! 🎉**

**Key Achievements:**
- **27-parameter model** with Gate×Driver×Quality architecture
- **Converged training** from loss 1.064 → 1.033 in 149 iterations
- **Multi-horizon predictions** for 1, 2, 3-day forward returns
- **Reasonable output ranges**: mu_hat ∈ [0, 0.04], p_hat ∈ [0.48, 0.91]
- **Pure NumPy/SciPy implementation** - no external ML dependencies

**Next Priority:** Implement Phase 2 (Rolling Training & Backtesting)

- [ ] encoder-decoder
- [ ] do we need to addCV + HyperPara part?