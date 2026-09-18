import numpy as np
import pandas as pd

from co_variance_results import CovarianceResult, VarianceResult

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

        # probably have to differentiate between a multi asset portfolio and a single asset portfolio  
        self.holdings = pd.Series(holdings, dtype=float)


        if (self.holdings < 0).any():
            raise ValueError("Negative holdings are not allowed.")



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
    #     # return self.weights(data) @ covariance_matrices @ self.weights(data) 

    def portfolio_variance(
        self,
        data,
        covariance_result: CovarianceResult,
    ) -> pd.Series:
        
        weights = (
        self.weights(data)
        .shift(1)
        .reindex(
            index=covariance_result.dates,
            columns=covariance_result.assets,
            )
        )

        W = weights.to_numpy()
        Sigma = covariance_result.covariances

        portfolio_variance = np.einsum(
            "ti,tij,tj->t",
            W,
            Sigma,
            W,
        )

        return pd.Series(
            portfolio_variance,
            index=covariance_result.dates,
            name="portfolio_variance",
        )

    
    def variance_forecast(
        self,
        data,
        covariance_forecast: pd.DataFrame,
    ) -> pd.Series:

        assets = covariance_forecast.columns

        weights = (
            self.weights(data)
            .iloc[-1]
            .reindex(assets)
            .to_numpy()
        )

        Sigma = covariance_forecast.to_numpy()

        portfolio_variance = (
            weights
            @ Sigma
            @ weights
        )

        return pd.Series(
            [portfolio_variance],
            index=[data.prices.index[-1]],
            name="portfolio_variance",
        )


#     def variance(
#     self,
#     data,
#     covariance: CovarianceResult | pd.DataFrame,
# ) -> pd.Series:

#         weights = self.weights(data)

#         # --------------------------------------------------------
#         # Historical covariance series
#         # --------------------------------------------------------

#         if isinstance(covariance, CovarianceResult):

#             dates = covariance.dates
#             assets = covariance.assets

#             # For return X_t, use portfolio weights known at t-1
#             historical_weights = (
#                 weights
#                 .shift(1)
#                 .reindex(index=dates, columns=assets)
#             )

#             W = historical_weights.to_numpy()
#             Sigma = covariance.covariances

#             portfolio_variance = np.einsum(
#                 "ti,tij,tj->t",
#                 W,
#                 Sigma,
#                 W,
#             )

#             return pd.Series(
#                 portfolio_variance,
#                 index=dates,
#                 name="portfolio_variance",
#             )

#         # --------------------------------------------------------
#         # Single covariance forecast
#         # --------------------------------------------------------

#         elif isinstance(covariance, pd.DataFrame):

#             assets = covariance.columns

#             # Forecast for t+1 uses weights known at t
#             current_weights = (
#                 weights
#                 .iloc[-1]
#                 .reindex(assets)
#                 .to_numpy()
#             )

#             Sigma_forecast = covariance.to_numpy()

#             portfolio_variance = (
#                 current_weights
#                 @ Sigma_forecast
#                 @ current_weights
#             )

#             return pd.Series(
#                 data=[portfolio_variance],
#                 index=[weights.index[-1]],
#                 name="portfolio_variance",
#             )

#         else:
#             raise TypeError(
#                 "covariance must be a CovarianceResult "
#                 "or a covariance forecast DataFrame."
#             )


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

    import market_data

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