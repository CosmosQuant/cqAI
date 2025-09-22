# src/indicators/technical.py
import pandas as pd
import numpy as np

class Signal:
    registry = {}

    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.signals = None

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls.registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, df, config):
        signal_type = config.get("signal_type", "RSI")
        if signal_type not in cls.registry:
            raise ValueError("Unknown signal type: " + signal_type)
        return cls.registry[signal_type](df, config)

    def get_signal(self):
        raise NotImplementedError("Subclasses must implement get_signal method.")

    def get_signal_name(self):
        raise NotImplementedError("Subclasses must implement get_signal_name method.")

    @classmethod
    def compute_indicator(cls, df, window, **kwargs):
        raise NotImplementedError("Subclasses must implement compute_indicator method.")


### get rsi and pass it to simulator directly, dont save in input df/databuffer
@Signal.register("RSI")
class RSI(Signal):
    #data_buffer = pd.DataFrame(timeindex)
    def __init__(self, df, config):
        super().__init__(df, config)
        self.window = config.get("window", 14)
        self.threshold = config.get("threshold", 30)
        self.longshort = config.get("longshort", "long")
        # self.rsi_col = "RSI" + str(self.window)

    @staticmethod
    def calculate_rsi(df, window):
        prices = df["Close"]
        diffs = prices.diff()
        gains = diffs.clip(lower=0)
        losses = -diffs.clip(upper=0)
        avg_gain = gains.rolling(window=window, min_periods=window).mean()
        avg_loss = losses.rolling(window=window, min_periods=window).mean()
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        return rsi


    def get_signal(self):
        if self.signals is not None:
            return self.signals
        
        rsi_vec = RSI.calculate_rsi(self.df, self.window)
        if self.longshort.lower() in ["long", "l"]:
            signals = (rsi_vec < self.threshold).astype(int)
        elif self.longshort.lower() in ["short", "s"]:
            signals = -(rsi_vec > (100 - self.threshold)).astype(int)
        else:
            raise ValueError("Invalid longshort parameter. Choose 'long' or 'short'.")
        self.signals = signals
        return signals

    def get_signal_name(self):
        return "RSI_{}_{}".format(self.window, self.threshold)


###### other signals for future use
@Signal.register("MACD")
class MACD(Signal):
    def __init__(self, df, config):
        super().__init__(df, config)
        self.fast = config.get("fast", 12)
        self.slow = config.get("slow", 26)
        self.signal_period = config.get("signal_period", 9)
        self.longshort = config.get("longshort", "long")
        self.macd_vec = None
        self.signal_line = None

    @staticmethod
    def calculate_macd(df, fast, slow, signal_period):
        price = df["Close"]
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal_line

    @classmethod
    def compute_indicator(cls, df, window, **kwargs):
        fast = kwargs.get("fast", 12)
        slow = kwargs.get("slow", 26)
        signal_period = kwargs.get("signal_period", 9)
        macd, _ = cls.calculate_macd(df, fast, slow, signal_period)
        return macd

    def get_signal(self):
        if self.macd_vec is None or self.signal_line is None:
            self.macd_vec, self.signal_line = MACD.calculate_macd(self.df, self.fast, self.slow, self.signal_period)
        if self.longshort == "long":
            signals = (self.macd_vec > self.signal_line).astype(int)
        elif self.longshort == "short":
            signals = -(self.macd_vec < self.signal_line).astype(int)
        else:
            raise ValueError("Invalid longshort parameter. Choose 'long' or 'short'.")
        return signals

    def get_signal_name(self):
        return "MACD_{}_{}_{}".format(self.fast, self.slow, self.signal_period)

@Signal.register("SMA")
class SMA(Signal):
    def __init__(self, df, config):
        super().__init__(df, config)
        # Expect config to contain: window and optionally longshort.
        self.window = config["window"]
        self.longshort = config.get("longshort", "long")
        self.sma_vec = None

    @staticmethod
    def calculate_sma(df, window):
        return df["Close"].rolling(window=window, min_periods=window).mean()

    @classmethod
    def compute_indicator(cls, df, window, **kwargs):
        return cls.calculate_sma(df, window)

    def get_signal(self):
        if self.sma_vec is None:
            self.sma_vec = SMA.calculate_sma(self.df, self.window)
        # For long signal: if current close is above SMA, signal = 1, otherwise 0.
        if self.longshort == "long":
            signals = (self.df["Close"] > self.sma_vec).astype(int)
        elif self.longshort == "short":
            signals = -(self.df["Close"] < self.sma_vec).astype(int)
        else:
            raise ValueError("Invalid longshort parameter. Choose 'long' or 'short'.")
        return signals

    def get_signal_name(self):
        return "SMA_{}".format(self.window)
