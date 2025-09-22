# src/analysis/daily.py
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from src.utils.daily_results import DailyResults
from src.utils.performance import compute_monthly_rolling_sharpe


class DailyResultAnalyzer:
    def __init__(self, results_folder="results", plot_enabled=False, pdf=None):
        self.results_folder = results_folder
        self.plot_enabled = plot_enabled
        self.pdf = pdf

    def process_and_save(self, full_df, daily_results, config_key):
        monthly_sharpe = compute_monthly_rolling_sharpe(daily_results["GrossPnL"])
        daily_results["MonthlySharpe"] = monthly_sharpe

        csv_filename = os.path.join(self.results_folder, config_key + ".csv")
        daily_results.to_csv(csv_filename)
        if self.plot_enabled and self.pdf is not None:
            fig = DailyResults.plot_daily_results(daily_results, "Configuration: " + config_key)
            self.pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        print("Processed configuration {} (saved to {})".format(config_key, csv_filename))
        return csv_filename

    def finalize_plots(self):
        if self.plot_enabled and self.pdf is not None:
            self.pdf.close()
            print("All plots saved.")
