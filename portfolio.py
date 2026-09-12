import numpy as np
import pandas as pd

import utils


class Portfolio:
    """Simple buy-and-hold stock portfolio."""

    def __init__(self, assets, holdings, start, end):
        """
        Parameters
        ----------
        assets : list[str]
            Asset tickers.
        holdings : array-like
            Number of shares held in each asset.
        start : str
            Start date for price data.
        end : str
            End date for price data.
        """

        if len(assets) != len(holdings):
            raise ValueError("Number of assets and holdings must match.")

        self.assets = assets
        self.holdings = pd.Series(holdings, index=assets, dtype=float)

        self.prices = utils.download_prices(assets, start, end)

    def position_values(self):
        """Market value of each individual position over time."""
        return self.prices * self.holdings

    def value(self):
        """Total portfolio market value over time."""
        return self.position_values().sum(
            axis=1,
            min_count=len(self.holdings)
        )

    def weights(self):
        """Portfolio weights of each asset over time."""
        return self.position_values().div(self.value(), axis=0)

    def risk_factors(self):
        """Log returns of the underlying stock prices."""
        return utils.log_returns(self.prices)

    def loss(self):
        """Calculate historical one-day portfolio losses."""
        simple_returns = np.expm1(self.risk_factors())

        previous_position_values = self.position_values().shift(1)

        losses = -(
            simple_returns * previous_position_values
        ).sum(
            axis=1,
            min_count=len(self.holdings)
        )

        return losses.dropna()


if __name__ == "__main__":
    port = Portfolio(
        ["AAPL", "MSFT"],
        [2, 3],
        "2000-01-01",
        "2009-12-31",
    )

    print(port.loss())