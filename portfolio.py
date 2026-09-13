import numpy as np
import pandas as pd

import market_data


class Portfolio:
    """Simple buy-and-hold stock portfolio."""

    def __init__(self, holdings: dict):
        """
        Parameters
        ----------
        holdings : dict
            Mapping from asset ticker to number of shares held.
            Example: {"AAPL": 2, "MSFT": 3}
        """

        if not isinstance(holdings, dict):
            raise TypeError("holdings must be a dictionary.")

        if len(holdings) == 0:
            raise ValueError("holdings must not be empty.")

        # if (holdings < 0).any():
        #     raise ValueError("Negative holdings are not allowed.")

        self.holdings = pd.Series(holdings, dtype=float)

        self.assets = self.holdings.index.tolist()

        # prices = market_data.download_prices(
        #     self.assets,
        #     start,
        #     end
        # )

    def position_values(self, data):
        """
        data value of each individual position over time.

        Returns
        -------
        pandas.DataFrame
        """
        return data.prices * self.holdings

    def value(self, data):
        """
        Total portfolio value over time.

        Returns
        -------
        pandas.Series
        """
        return self.position_values(data).sum(
            axis=1,
            min_count=len(self.holdings)
        )

    # def variance(self, covariance_matrices):
    #     """
    #     Portfolio variance over time.

    #     Returns
    #     -------
    #     pandas.Series
    #     """
    #     return self.weights(data) @ covariance_matrices @ self.weights(data) 

    def weights(self, data):
        """
        Portfolio weights of each asset over time.

        Returns
        -------
        pandas.DataFrame
        """
        return self.position_values(data).div(
            self.value(data),
            axis=0
        )

    def risk_factors(self, data):
        """
        Log returns of the underlying stock data.prices.

        Returns
        -------
        pandas.DataFrame
        """
        return data.log_returns()

    def loss(self, data):
        """
        Historical one-day portfolio losses.

        Positive values represent losses and negative values gains.

        Returns
        -------
        pandas.Series
        """
        simple_returns = np.expm1(
            self.risk_factors(data)
        )

        previous_position_values = (
            self.position_values(data).shift(1)
        )

        losses = -(
            simple_returns * previous_position_values
        ).sum(
            axis=1,
            min_count=len(self.holdings)
        )

        return losses.dropna()


if __name__ == "__main__":

    port = Portfolio(
        holdings={
            "AAPL": 2,
            "MSFT": 3,
        }
    )

    data = market_data.MarketData.from_yfinance(port.assets, "2000-01-01","2009-12-31")

    print(type(data))
    print(type(data.prices))

    print(port.holdings)
    print(port.value(data))
    print(port.weights(data))
    print(port.loss(data))  