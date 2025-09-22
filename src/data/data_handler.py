# src/data/data_handler.py
import os
import glob
import re
import pandas as pd
from src.indicators.technical import Signal

## Class for loading and processing market data from CSV files, with a saving_space option to retain only necessary columns.

class DataHandler:
    def __init__(self, config):
        self.config = config
        self.data = None

    def load_data(self, saving_space=True):
            folder_path = self.config["data_folder"]
            csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
            days_df = []
            for file in csv_files:
                date_str = re.sub(r'\D', '', file)
                try:
                    date_val = pd.to_datetime(date_str, format="%Y%m%d")
                except ValueError:
                    date_val = pd.to_datetime(date_str, errors='coerce')
                df = pd.read_csv(file)
                df['Date'] = pd.to_datetime(date_val)
                if 'Datetime' in df.columns:
                    df = df.sort_values("Datetime").reset_index(drop=True)
                else:
                    df = df.reset_index(drop=True)
                days_df.append(df)
            self.data = pd.concat(days_df, ignore_index=True)
            if saving_space:
                necessary_cols = ["Date", "Close"]
                self.data = self.data[necessary_cols]
            return self.data

    def process_data(self, saving_space=True):
        if self.data is None:
            self.load_data()
        self.data = self.data.sort_values(["Date"]).reset_index(drop=True)
        signal_type = self.config.get("signal_type")
        if signal_type not in Signal.registry:
            raise ValueError("Unknown signal type: " + signal_type)
        for window in self.config["windows"]:
            indicator_vec = Signal.registry[signal_type].compute_indicator(self.data, window)
            col_name = "{}{}".format(signal_type.upper(), window)
            self.data[col_name] = indicator_vec
        if saving_space:
            necessary_cols = ["Date", "Close"] + ["{}{}".format(signal_type.upper(), w) for w in self.config["windows"]]
            self.data = self.data[necessary_cols]
        return self.data
