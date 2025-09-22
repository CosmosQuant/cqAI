# parallel main.py
import os
import datetime
import itertools
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing

from src.config.settings import BASE_CONFIG
from src.config.config_generator import ConfigGenerator
from src.data.data_handler import DataHandler
from src.indicators.technical import Signal 
from src.simulation.simulator import Simulator
from src.strategies.simple_strategy import SimpleStrategy
from src.utils.daily_results import DailyResults
from src.utils.performance import compute_monthly_rolling_sharpe 
from src.analysis.daily import DailyResultAnalyzer

class SimulationProject:
    def __init__(self, base_config):
        self.base_config = base_config
        data_handler = DataHandler(base_config)
        self.full_df = data_handler.load_data(saving_space=True)
        config_gen = ConfigGenerator(base_config, config_file="config_list.json")
        self.config_list = config_gen.get_configs()
        self.results_folder = "results_parallel"
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        self.plot_enabled = base_config.get("plot", False)
        self.pdf = None
        if self.plot_enabled:
            self.pdf = PdfPages("%s_my_%s_%s.pdf" % (base_config['ticker'],
                                                     base_config['longshort'],
                                                     datetime.datetime.now().strftime("%m%d")))
        self.simulator = Simulator(self.full_df)
        self.daily_analyzer = DailyResultAnalyzer(results_folder=self.results_folder,
                                                  plot_enabled=self.plot_enabled,
                                                  pdf=self.pdf)

    @staticmethod
    def init_globals(full_df, simulator, daily_analyzer):
        global FULL_DF, SIMULATOR, DAILY_ANALYZER
        FULL_DF = full_df
        SIMULATOR = simulator
        DAILY_ANALYZER = daily_analyzer

    @staticmethod
    def task(cfg):
        signal_config = {
            "signal_type": cfg.get("signal_type", "RSI"),
            "window": cfg["window"],
            "threshold": cfg["threshold"],
            "longshort": cfg["longshort"]
        }
        signal_obj = Signal.create(FULL_DF, signal_config)
        signals = signal_obj.get_signal()
        pnl_gross, cost, slippage, pos = SIMULATOR.simulate(
            signals,
            hold_period=cfg["hold_period"],
            num_of_share=cfg["num_of_share"],
            cost_ratio=cfg["cost_ratio"],
            slippage_ratio=cfg["slippage_ratio"]
        )
        daily_results = DailyResults.get_daily_results_df(FULL_DF, pnl_gross, cost, slippage, pos)
        config_key = "Hold%s_%s_Thr%s" % (cfg["hold_period"], cfg["window"], cfg["threshold"])
        DAILY_ANALYZER.process_and_save(FULL_DF, daily_results, config_key)
        return config_key

    def run(self):
        num_processes = min(4, len(self.config_list))
        with multiprocessing.Pool(processes=num_processes,
                                  initializer=SimulationProject.init_globals,
                                  initargs=(self.full_df, self.simulator, self.daily_analyzer)) as pool:
            results = pool.map(SimulationProject.task, self.config_list)
        if self.plot_enabled:
            self.daily_analyzer.finalize_plots()
        print("Processed configurations:", results)

if __name__ == "__main__":
    project = SimulationProject(BASE_CONFIG)
    project.run()
