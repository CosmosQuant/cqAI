# cqAI_main.py - MVP ML workflow
import datetime
import pandas as pd
from itertools import product
from src import data, features, models, cv, signals, backtest, analysis

# ===== configs =====
# data path
DATA_PATH = 'data/btc_csv'
# features
FEATURES = ['sma_10', 'sma_20', 'rsi_14']
# horizon
HORIZON = 15  # 15 minutes ahead return

# models (algorithm + params)
MODELS = [
    {'id': 'ridge_default', 'algorithm': 'ridge', 'params': {'alpha': 1.0}},
    {'id': 'ridge_strong', 'algorithm': 'ridge', 'params': {'alpha': 10.0}},
    {'id': 'lgb_default', 'algorithm': 'lightgbm', 'params': {'n_estimators': 100, 'learning_rate': 0.1}},
    {'id': 'lgb_deep', 'algorithm': 'lightgbm', 'params': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 8}}
]

# cv strategies
CV_STRATEGIES = [
    {'id': 'walkforward', 'name': 'WalkForward', 'type': 'walk_forward', 'train_days': 252, 'test_days': 63},
    {'id': 'purged', 'name': 'Purged', 'type': 'purged_cv', 'n_splits': 5, 'embargo_days': 5},
    {'id': 'seasonal', 'name': 'Seasonal', 'type': 'seasonal_cv', 'n_splits': 4, 'season_length': 90}
]

# simulation config
SIM_CONFIG = {'signal_threshold': 0.0, 'cost_bp': 1.0, 'slip_bp': 2.0}

# # ===== execution =====
# if __name__ == "__main__":
# data preparation
df = data.load_ohlcv(DATA_PATH)

# load data from crypto_df.csv.gz
df = pd.read_csv('data/crypto_df.csv.gz', compression='gzip', index_col=0, parse_dates=True)
df["Open Time"] = pd.to_datetime(df["Open Time"],  unit="ms", utc=True)
df = df.set_index('Open Time')
df = df[(df.index >= '2024-01-01') & (df.index <= '2025-06-30')]
# save df into a pickle file
df.to_pickle('data/crypto_df_small.pkl')

# X = features.make_features(df, FEATURES)
# y = features.make_labels(df, HORIZON)

# all_results = []
# output_dir = f'results/exp_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
# model_cv_combinations = list(product(MODELS, CV_STRATEGIES))

# print(f"Starting {len(MODELS)} models × {len(CV_STRATEGIES)} CV strategies = {len(model_cv_combinations)} combinations")

# # loop through all model×cv combinations
# for combination_idx, (model_config, cv_config) in enumerate(model_cv_combinations):
#     combination_id = f"{model_config['id']}_{cv_config['id']}"
#     print(f"[{combination_idx+1}/{len(model_cv_combinations)}] {combination_id}")
    
#     model = models.create_model(model_config)
#     splitter = cv.build_splitter(cv_config)
    
#     for cv_idx, (train_idx, test_idx) in enumerate(splitter.split(df)):
#         # train and predict
#         model.fit(X[train_idx], y[train_idx])
#         yhat = model.predict(X[test_idx])
        
#         # generate signals and backtest
#         sigs = signals.yhat_to_signals(yhat, SIM_CONFIG['signal_threshold'])
#         pnl = backtest.simulate(df[test_idx], sigs, SIM_CONFIG['cost_bp'], SIM_CONFIG['slip_bp'])
        
#         # save cv result
#         cv_result = {
#             'combination_id': combination_id, 'model_id': model_config['id'], 'algorithm': model_config['algorithm'],
#             'model_params': model_config['params'], 'cv_id': cv_config['id'], 'cv_strategy': cv_config['name'],
#             'cv_idx': cv_idx, 'train_period': (df.index[train_idx[0]], df.index[train_idx[-1]]),
#             'test_period': (df.index[test_idx[0]], df.index[test_idx[-1]]), 'yhat': yhat,
#             'signals': sigs, 'pnl': pnl, 'metrics': analysis.compute_cv_metrics(pnl, sigs, yhat)
#         }
#         all_results.append(cv_result)

# # analyze and save results
# final_analysis = analysis.analyze_all_results(all_results)
# analysis.save_experiment_results(final_analysis, output_dir)

# print(f"Completed! {len(all_results)} CV experiments saved to: {output_dir}")
