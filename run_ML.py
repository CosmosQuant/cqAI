import json, os, subprocess
import numpy as np, pandas as pd, xgboost as xgb, matplotlib.pyplot as plt, seaborn as sns
from typing import Dict, List, Tuple, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from run_PPT import compute_run_length
from src.utils import annualized_sharpe, retmd_ratio, max_drawdown, standard_pnl_analysis, create_holdout_split, tsdata_split
from src.data import read_xts
sns.set_style("whitegrid"); sns.set_palette("husl")

# TODO - ensemble models - regression, classification, and regression + classification

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=["date"])
    df = df.rename(columns=str.lower).set_index("date").sort_index()
    for col in ("close", "high", "low"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")
    return df

def _compute_runlen(close: pd.Series) -> pd.Series:
    ret1 = close.pct_change()
    down_mask = ret1.lt(0).to_numpy(dtype=bool)
    run = compute_run_length(down_mask)
    return pd.Series(run, index=close.index, name="runlen")

def create_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    res = pd.DataFrame(index=df.index)
    res["y_reg"] = df["close"].shift(-horizon) / df["close"] - 1.0
    # res["y_reg"] = res["y_reg"] / res["y_reg"].std()
    res["y_reg"] = res["y_reg"] / res["y_reg"].rolling(252*3).std().shift(1).fillna(1)
    res["y_cls"] = (res["y_reg"] > 0).astype(bool)
    return res  

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame(index=df.index)
    res["runlen"] = _compute_runlen(df["close"])
    res["runlen"] = res["runlen"].clip(lower=2, upper=5)
    
    res["dd_abs4"] = np.maximum(0.0, -df["close"].pct_change(4))
    
    # res["mom"] = (df["close"].rolling(3).mean() > df["close"].rolling(250).mean()).astype(int)
    # get mmoving average of mom
    
    # res["volatility"] = df["close"].pct_change(4).rolling(252).std()
    # res["ret4"] = df["close"].pct_change(4)
    return res

def create_model(is_reg: bool = False, **params) -> Union[xgb.XGBClassifier, xgb.XGBRegressor]:
    defaults = {
        "max_depth": 3, "learning_rate": 0.05, "n_estimators": 100,
        "subsample": 1, "colsample_bytree": 1, "tree_method": "hist",
        "enable_categorical": False, "random_state": 42, "verbosity": 0,        
        "objective": "reg:squarederror" if is_reg else "binary:logistic",
        "eval_metric": "rmse" if is_reg else "auc",
    }
    defaults.update(params)
    return xgb.XGBRegressor(**defaults) if is_reg else xgb.XGBClassifier(**defaults)

def train_model(
    model: Union[xgb.XGBClassifier, xgb.XGBRegressor],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Union[xgb.XGBClassifier, xgb.XGBRegressor]:
    """Train XGBoost model"""
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


##################################################################################
# MAIN

horizon, holding_days = 3, 3

model_params = {
    "max_depth": 3,
    "learning_rate": 0.01, "n_estimators": 500,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "early_stopping_rounds": 50, "min_child_weight": 10, #"tree_method": "hist",
    "reg_lambda": 0.0, "reg_alpha": 0.5,
}

# Load and prepare data
df = load_data("spx.csv")
df = df[df.index >= '1940-01-01']
feats, target = create_features(df), create_target(df, horizon)
dataset, feature_cols = pd.concat([feats, target, df["close"]], axis=1), feats.columns.tolist()
df_model, df_holdout = create_holdout_split(dataset, holdout_year=2020)
df_train, df_test = tsdata_split(df_model, train_end="2015-12-31")

# model_params.update({"monotone_constraints": {col: 1 for col in feature_cols if col == "runlen"}})

# Train model
is_reg = 1
col_y, thresholds = "y_reg", [0, 0.02, 0.03, 0.05]
if not is_reg:
    col_y, thresholds = "y_cls", [0.5, 0.54, 0.56, 0.58]
    
X_train, y_train = df_train[feature_cols], df_train[col_y].astype(float if is_reg else bool)
X_test, y_test = df_test[feature_cols], df_test[col_y].astype(float if is_reg else bool)

model = create_model(is_reg=is_reg, **model_params)
model = train_model(model, X_train, y_train)
yhat = pd.Series(model.predict(X_test) if is_reg else model.predict_proba(X_test)[:, 1], index=df_test.index)
print(f"5 Quantiles of yhat: \n{yhat.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).round(2)}")

# Create PnL DataFrame for all strategies
print(f"Train: {df_train.index[0]} to {df_train.index[-1]} ({len(df_train)} days)")
print(f"Test: {df_test.index[0]} to {df_test.index[-1]} ({len(df_test)} days)")      
print(f"Model result - selected iterations: {model.best_iteration}")
print(f"\nGenerating PnL for thresholds: {thresholds} || Holding period: {holding_days} day(s)")

pnl_df = pd.DataFrame({'BnH': df_test["close"].pct_change().fillna(0)}, index=df_test.index)
# pnl_df = pd.DataFrame({'BnH': df_test["close"].diff().fillna(0)}, index=df_test.index)/1000
for th in thresholds:    
    position = yhat.shift(1) > th
    if sum(position)<50: continue
    position = position.astype(float).rolling(holding_days, 1).sum().fillna(0)    
    # position = position.clip(lower=0, upper=2)
    pnl_df[f"th {th}"] = pnl_df['BnH'] * position / holding_days
            
# Run standard analysis
standard_pnl_analysis(pnl_df, save_path='ml_result.png')

# plot histogram of yhat and save to png
plt.figure(figsize=(10, 5))
plt.hist(yhat, bins=100)
plt.savefig("hist.png", dpi=300, bbox_inches="tight")

