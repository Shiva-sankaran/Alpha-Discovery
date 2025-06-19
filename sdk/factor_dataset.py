import pandas as pd
import os

class FactorDataset:
    def __init__(self, symbol, feather_dir="features"):
        self.symbol = symbol
        self.path = os.path.join(feather_dir, f"{symbol}.feather")
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Missing feather file for {symbol}")
        self.df = pd.read_feather(self.path).set_index("date")
        self.df.index = pd.to_datetime(self.df.index)

    def get_factors(self, columns=None):
        if columns:
            return self.df[columns]
        # Exclude OHLCV if not needed
        return self.df.drop(columns=["open", "high", "low", "volume"], errors="ignore")

    def get_close(self):
        return self.df["close"]

    def get_full(self):
        return self.df.copy()
