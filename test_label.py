import time, pandas as pd, numpy as np
from src.data import generate_test_data
from src.features import run_SMA, run_std, run_SR, fast_rank, create_features, discretize
from src.labels import Label

# Set pandas display options to show more columns
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 1000); pd.set_option('display.max_colwidth', 1000)

df = generate_test_data(100000, 42)

print("\nTesting Label class...")

y = Label(df, label_type='return', LB=5)
y.calculate()
print(f"Label name: {y.get_name()} \n label: \n {y.get_label().dropna().tail()}")


y = Label(df, label_type='return', LB=5, rescale={'method': 'volatility', 'nscale': 10})
y.calculate()
print(f"Label name: {y.get_name()} \n label: \n {y.get_label().dropna().tail()}")


y = Label(df, label_type='return', LB=5, discretize={'method': 'binary', 'threshold': 0.0})
y.calculate()
print(f"Label name: {y.get_name()} \n label: \n {y.get_label().dropna().tail()}")


y = Label(df, label_type='return', LB=5, discretize={'method': 'quantile', 'bin': 5})
y.calculate()
print(f"Label name: {y.get_name()} \n label: \n {y.get_label().dropna().tail()}")
print(f"Count summary of discrete label: \n {y.get_label().value_counts().sort_index()}")



y = Label(df, label_type='return', LB=5,
          rescale={'method': 'volatility', 'nscale': 10}, discretize={'method': 'quantile', 'bin': 5})
y.calculate()
print(f"Label name: {y.get_name()} \n label: \n {y.get_label().dropna().tail()}")
print(f"Count summary of discrete label: \n {y.get_label().value_counts().sort_index()}")
