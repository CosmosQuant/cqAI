# ML Model Implementation Plan

## Overview
Implement XGBoost model for PPT trading strategy, comparing against the neural network approach.

## Architecture

```
run_ML.py
├── Data Loading & Preparation
├── Feature Engineering  
├── Train/Test/Validation Split
├── Model Training (XGBoost)
├── Model Analysis
└── Trading Analysis
```

## Step-by-Step Plan

### Step 0: Data Loading
**Goal**: Load and prepare SPX data

**Implementation**:
```python
def load_data(file_path: str = 'spx.csv') -> pd.DataFrame:
    """Load SPX data with datetime index"""
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df
```

**Output**: DataFrame with OHLCV columns, datetime index

---

### Step 1: Feature & Target Generation
**Goal**: Create features (runlen, dd_abs) and target variable Y

**Implementation**:
```python
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features matching run_PPT.py"""
    # Reuse compute_run_length from run_PPT.py
    # Features: runlen, dd_abs (same as PPT)
    # Target: ret_h (h-day forward return, default h=1)
    return df_with_features

def create_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Generate target variable Y = ret_h > 0"""
    # Binary classification: 1 if future return > 0, else 0
    return y
```

**Features**:
- `runlen`: Run length of consecutive down days
- `dd_abs`: Absolute drawdown (max(0, -ret4))

**Target**:
- `y`: Binary (1 if ret_h > 0, else 0) for classification
- `ret_h`: Continuous return for regression (optional)

---

### Step 2: Model Configuration
**Goal**: Define XGBoost with hyperparameters

**Implementation**:
```python
from dataclasses import dataclass
import xgboost as xgb

@dataclass
class MLConfig:
    # Model hyperparameters
    max_depth: int = 3
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    objective: str = 'binary:logistic'  # or 'reg:squarederror'
    eval_metric: str = 'auc'  # or 'rmse'
    
    # Prediction horizon
    horizon: int = 1
    
    # Trading threshold
    pred_threshold: float = 0.5  # for classification
    
def create_model(config: MLConfig) -> xgb.XGBClassifier:
    """Initialize XGBoost model"""
    return xgb.XGBClassifier(
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        objective=config.objective,
        eval_metric=config.eval_metric,
        random_state=42
    )
```

**Hyperparameters**:
- Conservative defaults for initial MVP
- Easy to extend for hyperparameter tuning later

---

### Step 3: Data Split Strategy
**Goal**: Create holdout set and flexible train/test split supporting future walk-forward

**Implementation**:
```python
@dataclass
class SplitConfig:
    """Flexible split configuration"""
    holdout_year: int = 2020  # Final holdout (never touch)
    method: str = 'fixed'  # 'fixed', 'expanding', 'rolling'
    
    # For 'fixed' method
    train_end: str = '2015'
    
    # For 'expanding'/'rolling' method (walk-forward)
    train_window: int = 252 * 5  # 5 years in days
    test_window: int = 252  # 1 year in days
    step_size: int = 63  # Quarter (for walk-forward)

def create_holdout_split(df: pd.DataFrame, holdout_year: int = 2020) -> tuple:
    """
    First split: separate final holdout from full modeling data
    
    Returns:
        df_full: All data < holdout_year (for train/test)
        df_holdout: Data >= holdout_year (final validation, never touch)
    """
    df_holdout = df.loc[df.index.year >= holdout_year].copy()
    df_full = df.loc[df.index.year < holdout_year].copy()
    
    print(f"Full data: {df_full.index[0]} to {df_full.index[-1]} ({len(df_full)} days)")
    print(f"Holdout:   {df_holdout.index[0]} to {df_holdout.index[-1]} ({len(df_holdout)} days)")
    
    return df_full, df_holdout

def split_train_test(df_full: pd.DataFrame, config: SplitConfig) -> tuple:
    """
    Generic train/test split supporting multiple methods
    
    Methods:
        'fixed': Single train/test split at fixed date
        'expanding': Walk-forward with expanding training window
        'rolling': Walk-forward with rolling (fixed-size) training window
    
    Returns:
        For 'fixed': (df_train, df_test)
        For 'expanding'/'rolling': list of (df_train, df_test) tuples
    """
    if config.method == 'fixed':
        # Simple fixed split
        df_train = df_full.loc[:config.train_end]
        df_test = df_full.loc[config.train_end:]
        
        print(f"\nFixed split:")
        print(f"  Train: {df_train.index[0]} to {df_train.index[-1]} ({len(df_train)} days)")
        print(f"  Test:  {df_test.index[0]} to {df_test.index[-1]} ({len(df_test)} days)")
        
        return df_train, df_test
    
    elif config.method in ['expanding', 'rolling']:
        # Walk-forward splits
        splits = []
        n = len(df_full)
        
        # Initial training period
        train_start_idx = 0
        train_end_idx = config.train_window
        
        while train_end_idx + config.test_window <= n:
            # Define test period
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + config.test_window
            
            # Get data slices
            df_train = df_full.iloc[train_start_idx:train_end_idx]
            df_test = df_full.iloc[test_start_idx:test_end_idx]
            
            splits.append((df_train, df_test))
            
            # Move window
            if config.method == 'expanding':
                # Keep train_start, extend train_end
                train_end_idx = test_end_idx
            else:  # 'rolling'
                # Shift both train_start and train_end
                train_start_idx += config.step_size
                train_end_idx = train_start_idx + config.train_window
        
        print(f"\n{config.method.capitalize()} walk-forward: {len(splits)} splits")
        print(f"  First train: {splits[0][0].index[0]} to {splits[0][0].index[-1]}")
        print(f"  First test:  {splits[0][1].index[0]} to {splits[0][1].index[-1]}")
        print(f"  Last train:  {splits[-1][0].index[0]} to {splits[-1][0].index[-1]}")
        print(f"  Last test:   {splits[-1][1].index[0]} to {splits[-1][1].index[-1]}")
        
        return splits
    
    else:
        raise ValueError(f"Unknown split method: {config.method}")
```

**Split Strategy**:
1. **First split**: `df_full` (< 2020) and `df_holdout` (>= 2020)
2. **Second split** (on `df_full`):
   - **Fixed**: Single train/test (default: train=<2015, test=2015-2019)
   - **Expanding**: Walk-forward with growing training window
   - **Rolling**: Walk-forward with fixed-size sliding window

**Extensible Design**:
- MVP uses 'fixed' method
- Easy to switch to walk-forward by changing `config.method`
- All walk-forward parameters configurable (window sizes, step size)

---

### Step 4: Model Training
**Goal**: Fit XGBoost model with early stopping

**Implementation**:
```python
def train_model(model: xgb.XGBClassifier, X_train, y_train, X_test, y_test):
    """Train XGBoost with early stopping"""
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, 'train'), (X_test, 'test')],
        early_stopping_rounds=10,
        verbose=False
    )
    return model

def predict(model: xgb.XGBClassifier, X) -> tuple:
    """Generate predictions"""
    y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of class 1
    y_pred = model.predict(X)  # Binary prediction
    return y_pred, y_pred_proba
```

**Training Process**:
- Early stopping on test set
- Track training history for analysis

---

### Step 5: Model Analysis
**Goal**: Visualize training progress and evaluate performance

**Implementation**:
```python
def plot_training_curve(model: xgb.XGBClassifier):
    """Plot training vs test loss over iterations"""
    results = model.evals_result()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results['train']['auc'], label='Train AUC')
    ax.plot(results['test']['auc'], label='Test AUC')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('AUC')
    ax.set_title('Model Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('ml_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(y_true, y_pred, y_pred_proba):
    """Calculate classification metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba)
    }
    
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    
    return metrics

def plot_feature_importance(model: xgb.XGBClassifier, feature_names):
    """Plot feature importance"""
    importance = model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feature_names, importance)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ml_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
```

**Analysis Outputs**:
1. Training curve (loss over iterations)
2. Classification metrics (accuracy, precision, recall, F1, AUC)
3. Feature importance
4. Confusion matrix

---

### Step 6: Trading Analysis
**Goal**: Backtest trading strategy based on model predictions

**Implementation**:
```python
def simulate_trading(df: pd.DataFrame, predictions: np.ndarray, threshold: float = 0.5):
    """
    Simulate trading strategy:
    - Buy $1 of SPX if prediction > threshold
    - Hold for h days
    """
    signals = (predictions > threshold).astype(int)
    ret1 = df['close'].pct_change().shift(-1)  # Next day return
    
    # Daily PnL
    pnl = signals * ret1
    pnl = pnl.fillna(0)
    
    # Cumulative NAV
    nav = (1 + pnl).cumprod()
    
    # Trading stats
    n_trades = signals.sum()
    pct_trading_days = (signals.sum() / len(signals)) * 100
    
    return {
        'signals': signals,
        'pnl': pnl,
        'nav': nav,
        'n_trades': n_trades,
        'pct_trading_days': pct_trading_days
    }

def calculate_trading_metrics(result: dict, df: pd.DataFrame):
    """Calculate trading performance metrics"""
    pnl = result['pnl']
    nav = result['nav']
    
    # Sharpe ratio (annualized)
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0
    
    # Max drawdown
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (pnl[pnl > 0].count() / pnl[pnl != 0].count()) if (pnl != 0).any() else 0
    
    # Total return
    total_return = nav.iloc[-1] - 1
    
    metrics = {
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_return': total_return,
        'final_nav': nav.iloc[-1],
        'n_trades': result['n_trades'],
        'pct_trading_days': result['pct_trading_days']
    }
    
    return metrics

def plot_trading_results(result: dict, metrics: dict):
    """Plot NAV curve and trading statistics"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # NAV curve
    axes[0].plot(result['nav'].index, result['nav'].values, label='Strategy NAV', linewidth=2)
    axes[0].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Baseline')
    axes[0].set_title('Trading Strategy NAV Curve', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('NAV')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Trading signals
    signals_plot = result['signals'].copy()
    signals_plot[signals_plot == 0] = np.nan
    axes[1].scatter(signals_plot.index, signals_plot.values, c='green', s=10, alpha=0.5, label='Long Signal')
    axes[1].set_title('Trading Signals', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Signal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add metrics text
    metrics_text = f"Sharpe: {metrics['sharpe']:.2f} | Max DD: {metrics['max_drawdown']:.2%} | Win Rate: {metrics['win_rate']:.2%}\n"
    metrics_text += f"Total Return: {metrics['total_return']:.2%} | Trades: {metrics['n_trades']} ({metrics['pct_trading_days']:.1f}% days)"
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('ml_trading_results.png', dpi=300, bbox_inches='tight')
    plt.show()
```

**Trading Analysis Outputs**:
1. NAV curve over time
2. Trading signals visualization
3. Performance metrics:
   - Sharpe ratio (annualized)
   - Max drawdown
   - Win rate
   - Total return
   - Number of trades
   - Percentage of trading days

---

## File Structure

```python
# run_ML.py structure

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from dataclasses import dataclass
from run_PPT import compute_run_length  # Reuse existing functions

# Step 0: Data Loading
def load_data() -> pd.DataFrame: ...

# Step 1: Feature & Target
def create_features(df) -> pd.DataFrame: ...
def create_target(df, horizon) -> pd.Series: ...

# Step 2: Model Configuration
@dataclass
class MLConfig: ...
def create_model(config) -> xgb.XGBClassifier: ...

# Step 3: Data Split
def split_data(df, val_year) -> tuple: ...

# Step 4: Training
def train_model(model, X_train, y_train, X_test, y_test): ...
def predict(model, X) -> tuple: ...

# Step 5: Model Analysis
def plot_training_curve(model): ...
def evaluate_model(y_true, y_pred, y_pred_proba): ...
def plot_feature_importance(model, feature_names): ...

# Step 6: Trading Analysis
def simulate_trading(df, predictions, threshold) -> dict: ...
def calculate_trading_metrics(result, df) -> dict: ...
def plot_trading_results(result, metrics): ...

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("ML MODEL - XGBoost for PPT Strategy")
    print("="*80)
    
    # Step 0: Load data
    print("\n[Step 0] Loading data...")
    df = load_data('spx.csv')
    
    # Step 1: Create features & target
    print("\n[Step 1] Creating features & target...")
    df = create_features(df)
    y = create_target(df, horizon=1)
    
    # Step 3: Split data (Two-stage split)
    print("\n[Step 3] Splitting data...")
    
    # Stage 1: Create holdout
    df_full, df_holdout = create_holdout_split(df, holdout_year=2020)
    
    # Stage 2: Split df_full into train/test
    split_config = SplitConfig(
        holdout_year=2020,
        method='fixed',  # 'fixed', 'expanding', or 'rolling'
        train_end='2015'
    )
    
    split_result = split_train_test(df_full, split_config)
    
    # Handle different split methods
    if split_config.method == 'fixed':
        df_train, df_test = split_result
        
        # Prepare X, y
        feature_cols = ['runlen', 'dd_abs']
        X_train, y_train = df_train[feature_cols], y.loc[df_train.index]
        X_test, y_test = df_test[feature_cols], y.loc[df_test.index]
        
        # Step 4: Train model
        print("\n[Step 4] Training model...")
        ml_config = MLConfig()
        model = create_model(ml_config)
        model = train_model(model, X_train, y_train, X_test, y_test)
        
        # Step 5: Model analysis
        print("\n[Step 5] Model analysis...")
        plot_training_curve(model)
        y_pred, y_pred_proba = predict(model, X_test)
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        plot_feature_importance(model, feature_cols)
        
        # Step 6: Trading analysis
        print("\n[Step 6] Trading analysis...")
        result = simulate_trading(df_test, y_pred_proba, threshold=ml_config.pred_threshold)
        trading_metrics = calculate_trading_metrics(result, df_test)
        plot_trading_results(result, trading_metrics)
        
        print("\n" + "="*80)
        print("ML MODEL COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nHoldout data available: {len(df_holdout)} days (never touched)")
    
    else:  # walk-forward
        splits = split_result
        
        # For walk-forward, train on each split and aggregate results
        print(f"\n[Step 4-6] Processing {len(splits)} walk-forward splits...")
        
        all_predictions = []
        all_actuals = []
        
        for i, (df_train, df_test) in enumerate(splits):
            print(f"\n  Split {i+1}/{len(splits)}: Train={len(df_train)}, Test={len(df_test)}")
            
            feature_cols = ['runlen', 'dd_abs']
            X_train, y_train = df_train[feature_cols], y.loc[df_train.index]
            X_test, y_test = df_test[feature_cols], y.loc[df_test.index]
            
            ml_config = MLConfig()
            model = create_model(ml_config)
            model = train_model(model, X_train, y_train, X_test, y_test)
            
            y_pred, y_pred_proba = predict(model, X_test)
            all_predictions.append(pd.Series(y_pred_proba, index=df_test.index))
            all_actuals.append(y_test)
        
        # Aggregate results
        y_pred_all = pd.concat(all_predictions)
        y_test_all = pd.concat(all_actuals)
        
        print("\n[Step 5] Overall metrics (walk-forward)...")
        metrics = evaluate_model(y_test_all, (y_pred_all > 0.5).astype(int), y_pred_all)
        
        print("\n[Step 6] Overall trading (walk-forward)...")
        df_test_all = df_full.loc[y_pred_all.index]
        result = simulate_trading(df_test_all, y_pred_all.values, threshold=0.5)
        trading_metrics = calculate_trading_metrics(result, df_test_all)
        plot_trading_results(result, trading_metrics)
        
        print("\n" + "="*80)
        print("ML MODEL (WALK-FORWARD) COMPLETED")
        print("="*80)
        print(f"Splits: {len(splits)} | Holdout: {len(df_holdout)} days")
```

---

## Dependencies

Add to `requirements.txt`:
```
xgboost>=1.7.0
scikit-learn>=1.2.0
```

---

## Future Extensions (TODO)

- [ ] Add more features (technical indicators, volatility measures)
- [ ] Hyperparameter tuning (GridSearch, Bayesian optimization)
- [ ] Walk-forward validation
- [ ] Multiple horizons (H=1, 2, 5)
- [ ] Ensemble models (Random Forest, LightGBM)
- [ ] Feature selection / engineering
- [ ] Transaction costs
- [ ] Position sizing

---

## Key Design Principles (following rule.md)

1. **MVP**: Start simple with 2 features, binary classification
2. **Modularity**: Each step is a separate function, easy to test/modify
3. **Reusability**: Leverage existing functions from run_PPT.py
4. **Flexibility**: SplitConfig for easy experimentation
5. **Visualization**: Clear plots for model and trading analysis
6. **Standalone**: run_ML.py is independent, can be run directly
7. **Concise**: Compact code following user preferences (fewer lines)



# if split_kwargs["method"] in {"expanding", "rolling"}:
#     splits = split_train_test(df_model, **split_kwargs)
#     run_walk_forward_flow(splits, df_holdout, model_params=model_params, pred_threshold=pred_threshold)
# else:
#     df_train, df_test = split_train_test(df_model, **split_kwargs)
#     run_fixed_flow(df_train, df_test, df_holdout, model_params=model_params, pred_threshold=pred_threshold)




def predict(model: xgb.XGBClassifier, X: pd.DataFrame, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Predict using the model"""
    proba = model.predict_proba(X)[:, 1]
    labels = (proba >= threshold).astype(int)
    return labels, proba

    model_params = {
        "max_depth": 3,
        "learning_rate": 0.01,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
        
        "min_child_weight": 50,
        "tree_method": "hist",
    }
    split_kwargs = {
        "method": "fixed",
        "train_end": "2015-12-31",
        "train_window": 252 * 5,
        "test_window": 252,
        "step_size": 63,
    }



def run_fixed_flow(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_holdout: pd.DataFrame,
    *,
    model_params: Dict[str, Union[int, float, str]],
    pred_threshold: float,
) -> None:
    feature_cols = ["runlen", "dd_abs"]
    
    # Convert to numpy arrays
    y_train = df_train["y_cls"].values.astype(int)
    y_test = df_test["y_cls"].values.astype(int)
    X_train = df_train[feature_cols].values
    X_test = df_test[feature_cols].values
    
    model = create_model(**model_params)
    model = train_model(model, X_train, y_train, X_test, y_test)
    y_pred, y_proba = predict(model, df_test[feature_cols], pred_threshold)
    metrics = evaluate_model(df_test["y_cls"], y_pred, y_proba)
    result = simulate_trading(df_test, y_proba, pred_threshold)
    trade_metrics = calculate_trading_metrics(result)
    print("Model metrics:")
    print(json.dumps(metrics, indent=2))
    print("Trading metrics:")
    print(json.dumps(trade_metrics, indent=2))
    plot_training_curve(model, "classification")
    plot_feature_importance(model, feature_cols, "classification")
    plot_trading_results(df_test, result, trade_metrics)
    if df_holdout is not None and not df_holdout.empty:
        holdout_proba = model.predict_proba(df_holdout[feature_cols])[:, 1]
        holdout_result = simulate_trading(df_holdout, holdout_proba, pred_threshold, holding_days=holding_days)
        holdout_metrics = calculate_trading_metrics(holdout_result)
        print("Holdout trading metrics:")
        print(json.dumps(holdout_metrics, indent=2))


def run_walk_forward_flow(
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
    df_holdout: pd.DataFrame,
    *,
    model_params: Dict[str, Union[int, float, str]],
    pred_threshold: float,
) -> None:
    feature_cols = ["runlen", "dd_abs"]
    predictions: List[pd.DataFrame] = []
    for idx, (df_train, df_test) in enumerate(splits, start=1):
        model = create_model(**model_params)
        model = train_model(model, df_train[feature_cols], df_train["y_cls"], df_test[feature_cols], df_test["y_cls"])
        y_pred, y_proba = predict(model, df_test[feature_cols], pred_threshold)
        block = df_test.copy()
        block["proba"] = y_proba
        block["pred"] = y_pred
        block["split"] = idx
        predictions.append(block)
    agg = pd.concat(predictions).sort_index()
    metrics = evaluate_model(agg["y_cls"], agg["pred"], agg["proba"])
    result = simulate_trading(agg, agg["proba"].to_numpy(), pred_threshold)
    trade_metrics = calculate_trading_metrics(result)
    print("Aggregated model metrics:")
    print(json.dumps(metrics, indent=2))
    print("Aggregated trading metrics:")
    print(json.dumps(trade_metrics, indent=2))
    plot_trading_results(agg, result, trade_metrics)
    if not df_holdout.empty:
        model = create_model(**model_params)
        full_train = pd.concat([split[0] for split in splits])
        model = train_model(model, full_train[feature_cols], full_train["y_cls"], df_holdout[feature_cols], df_holdout["y_cls"])
        holdout_proba = model.predict_proba(df_holdout[feature_cols])[:, 1]
        holdout_result = simulate_trading(df_holdout, holdout_proba, pred_threshold, holding_days=holding_days)
        holdout_metrics = calculate_trading_metrics(holdout_result)
        print("Holdout trading metrics:")
        print(json.dumps(holdout_metrics, indent=2))


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        "confusion_tp": int(tp),
        "confusion_fp": int(fp),
        "confusion_fn": int(fn),
        "confusion_tn": int(tn),
    })
    return metrics


def plot_training_curve(model: xgb.XGBClassifier, title: str) -> None:
    if plt is None:
        return
    try:
        history = model.evals_result()
        if not history:
            print("No training history available (eval_set not used)")
            return
        metric = model.get_params().get("eval_metric", "metric")
        train_curve = history.get("train", {}).get(metric)
        valid_curve = history.get("valid", {}).get(metric)
        if train_curve is None or valid_curve is None:
            print("No valid training curves found")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(train_curve, label="train")
        plt.plot(valid_curve, label="valid")
        plt.title(f"Training history - {title}")
        plt.xlabel("Iteration")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not plot training curve: {e}")


def calculate_trading_metrics(result: pd.DataFrame) -> Dict[str, float]:
    ret = pd.Series(result["pnl"], index=result.index, name="strategy_ret")
    nav = pd.Series(result["nav"], index=result.index, name="nav")
    metrics = {
        "total_return": float(nav.iloc[-1] - 1.0) if not nav.empty else 0.0,
        "annualized_sharpe": annualized_sharpe(ret),
        "retmd_ratio": retmd_ratio(ret),
        "max_drawdown": max_drawdown(nav),
        "win_rate": float((ret > 0).mean()) if not ret.empty else 0.0,
        "n_trades": int(result["signal"].sum()),
        "pct_trading_days": float(result["signal"].mean()) if not result.empty else 0.0,
    }
    return metrics


def plot_feature_importance(model: xgb.XGBClassifier, feature_names: List[str], title: str, save_path: str = "ml_analysis.png") -> None:
    if plt is None:
        return
    importance = model.get_booster().get_score(importance_type="gain")
    if not importance:
        return
    scores = pd.Series({feature_names[int(k[1:])]: v for k, v in importance.items()})
    scores.sort_values(ascending=False, inplace=True)
    plt.figure(figsize=(6, 4))
    scores.plot(kind="bar")
    plt.title(f"Feature importance - {title}")
    plt.ylabel("gain")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Feature importance plot saved to {save_path}")
    
    # Automatically open the image file
    if os.path.exists(save_path):
        if os.name == 'nt':  # Windows
            os.startfile(save_path)
        elif os.name == 'posix':  # macOS and Linux
            subprocess.run(['open' if os.uname().sysname == 'Darwin' else 'xdg-open', save_path])
        print(f"Opening {save_path}...")

# Plot feature importance
# plot_feature_importance(model, feature_cols, "XGBoost Classification")


split_kwargs = {
    "method": "fixed",
    "train_end": "2015-12-31",
    "train_window": 252 * 5,
    "test_window": 252,
    "step_size": 63,
}