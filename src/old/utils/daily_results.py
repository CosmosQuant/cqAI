# src/utils/daily_results.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class DailyResults:
    @staticmethod
    def convert_to_daily(minutely_series, time_index):
        # Convert the provided time index to datetime
        time_index = pd.to_datetime(time_index)
        # Create a pandas Series using the given minutely series and the time index
        s = pd.Series(minutely_series.values, index=time_index)
        # Group the series by date and take the last value for each day
        daily_series = s.groupby(s.index.date).last()
        # Convert the grouped index back to datetime
        daily_series.index = pd.to_datetime(daily_series.index)
        return daily_series

    @staticmethod
    def get_daily_results_df(df, pnl_gross, cost, slippage, pos):
        # Convert minutely cumulative series to daily series by taking the last value of each day
        daily_gross = DailyResults.convert_to_daily(pnl_gross.cumsum(), df["Date"])
        daily_cost = DailyResults.convert_to_daily(cost.cumsum(), df["Date"])
        daily_slippage = DailyResults.convert_to_daily(slippage.cumsum(), df["Date"])
        daily_position = DailyResults.convert_to_daily(pos, df["Date"])
        total_daily = daily_gross + daily_cost + daily_slippage
        return pd.DataFrame({
            "GrossPnL": daily_gross,
            "Cost": daily_cost,
            "Slippage": daily_slippage,
            "TotalPnL": total_daily,
            "Position": daily_position
        }, index=daily_gross.index)

    @staticmethod
    def plot_daily_results(daily_df, title):
        # Create a figure and axis for plotting
        fig, ax = plt.subplots(figsize=(10, 4))
        # Plot each component of the daily results
        ax.plot(daily_df.index, daily_df["GrossPnL"], label="GrossPnL", color="blue")
        ax.plot(daily_df.index, daily_df["Cost"], label="Cost", color="red")
        ax.plot(daily_df.index, daily_df["Slippage"], label="Slippage", color="green")
        ax.plot(daily_df.index, daily_df["TotalPnL"], label="TotalPnL", color="magenta")
        # Set up the x-axis to show dates nicely
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Value")
        ax.legend()
        return fig
