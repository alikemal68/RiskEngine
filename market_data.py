import numpy as np, pandas as pd #,matplotlib.pyplot as plt 
import yfinance as yf

class MarketData:
    """Market data for a collection of assets."""

    def __init__(self, prices: pd.DataFrame):
        self.prices = prices

    @property
    def assets(self):
         return self.prices.columns.tolist()
    
    @property
    def start(self):
        return self.prices.index.min()

    @property
    def end(self):
        return self.prices.index.max()

    def log_returns(self):
        """Log returns."""
        return np.log(self.prices).diff().dropna()

    @classmethod
    def from_yfinance(cls, assets, start, end):
        prices = yf.download(
            assets,
            start=start,
            end=end,
            auto_adjust=False,
        )["Adj Close"]

        # Keep DataFrame shape even for one asset
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        return cls(prices)

    





