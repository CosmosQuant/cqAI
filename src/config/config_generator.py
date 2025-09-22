# src/config/config_generator.py
import os
import json
import itertools

## Class for generating simulation configurations from a JSON file, driving different simulation scenarios.

class ConfigGenerator:
    def __init__(self, base_config, config_file="config_list.json"):
        self.base_config = base_config
        self.config_file = config_file
        self.config_list = []
        if config_file and os.path.exists(config_file):
            self.load_configs()
        else:
            self.generate_configs()
            if config_file:
                self.save_configs()

    def generate_configs(self):
        windows = self.base_config["windows"]
        hold_periods = self.base_config["hold_periods"]
        thresholds = self.base_config["thresholds"]
        for w, h, t in itertools.product(windows, hold_periods, thresholds):
            config = self.base_config.copy()
            config["window"] = w         
            config["hold_period"] = h    
            config["threshold"] = t     
            if "signal_type" not in config:
                config["signal_type"] = "RSI"
            self.config_list.append(config)

    def save_configs(self):
        with open(self.config_file, "w") as f:
            json.dump(self.config_list, f, indent=4)

    def load_configs(self):
        with open(self.config_file, "r") as f:
            self.config_list = json.load(f)

    def get_configs(self):
        return self.config_list
