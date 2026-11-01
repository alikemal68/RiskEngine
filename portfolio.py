
"""Portfolio valuation and covariance-based variance calculations.

Covariance matrices are assumed to describe *decimal log returns*. The
weight-based methods return approximate portfolio return variances; the
position-value-based methods return approximate monetary loss variances.
"""

import numpy as np
import pandas as pd

from co_variance_results import CovarianceResult


class Portfolio:
    """A fixed-holdings, long-only equity portfolio."""

    def __init__(self, holdings: dict):
        """Initialize from a mapping of asset ticker to number of shares."""
        if not isinstance(holdings, dict):
            raise TypeError("holdings must be a dictionary.")
        if not holdings:
            raise ValueError("holdings must not be empty.")

        self.holdings = pd.Series(holdings, dtype=float)

        if not np.isfinite(self.holdings.to_numpy()).all():
            raise ValueError("Holdings must be finite numbers.")
        if (self.holdings < 0).any():
            raise ValueError("Negative holdings are not allowed.")
        if (self.holdings == 0).all():
            raise ValueError("At least one holding must be positive.")

        self.assets = self.holdings.index.tolist()

    def position_values(self, data) -> pd.DataFrame:
        """Market value of each position, in the price data's currency."""
        missing = self.holdings.index.difference(data.prices.columns)
        if not missing.empty:
            raise ValueError(
                f"Missing prices for portfolio assets: {missing.tolist()}"
            )

        return data.prices.loc[:, self.assets].mul(
            self.holdings, axis="columns"
        )

    def value(self, data) -> pd.Series:
        """Total market value of the portfolio on each date."""
        return self.position_values(data).sum(
            axis=1, min_count=len(self.holdings)
        ).rename("portfolio_value")

    def weights(self, data) -> pd.DataFrame:
        """Portfolio weights, obtained from position values / total value."""
        return self.position_values(data).div(
            self.value(data), axis=0
        )

    def risk_factors(self, data) -> pd.DataFrame:
        """Decimal log returns for the portfolio's underlying assets."""
        return data.log_returns().loc[:, self.assets]

    def portfolio_weighted_log_return(self, data) -> pd.Series:
        """First-order approximation to portfolio return: w[t-1] @ X[t]."""
        returns = self.risk_factors(data)

        previous_weights = self.weights(data).shift(1).reindex(
            index=returns.index,
            columns=returns.columns,
        )

        weighted_returns = (
            previous_weights * returns
        ).sum(
            axis=1,
            min_count=len(self.holdings),
        )

        return weighted_returns.rename(
            "portfolio_weighted_log_return"
        ).dropna()

    def loss(self, data) -> pd.Series:
        """Realized monetary loss; positive values indicate a loss.

        Uses exact simple price changes for a fixed-holdings portfolio,
        rather than the first-order approximation used by covariance VaR.
        """
        simple_returns = np.expm1(self.risk_factors(data))

        previous_position_values = (
            self.position_values(data)
            .shift(1)
            .reindex(
                index=simple_returns.index,
                columns=simple_returns.columns,
            )
        )

        losses = -(
            simple_returns * previous_position_values
        ).sum(
            axis=1,
            min_count=len(self.holdings),
        )

        return losses.rename("portfolio_loss").dropna()

    def _historical_variance(
        self,
        exposures: pd.DataFrame,
        covariance_result: CovarianceResult,
        name: str,
    ) -> pd.Series:
        """Apply one-step-ahead covariance matrices to t-1 exposures."""
        dates = pd.Index(covariance_result.dates)
        assets = pd.Index(covariance_result.assets)
        sigma = np.asarray(
            covariance_result.covariances, dtype=float
        )

        if not assets.is_unique or set(assets) != set(self.assets):
            raise ValueError(
                "Covariance assets must match portfolio assets."
            )

        if sigma.shape != (
            len(dates), len(assets), len(assets)
        ):
            raise ValueError(
                "Covariance array shape does not match dates/assets."
            )

        # Shift BEFORE reindexing to forecast target dates.
        # For loss on date t, exposures are known at t-1.
        previous_exposures = exposures.shift(1).reindex(
            index=dates,
            columns=assets,
        )

        if previous_exposures.isna().any().any():
            raise ValueError(
                "Missing t-1 exposures for covariance forecast dates."
            )

        x = previous_exposures.to_numpy()

        variance = np.einsum(
            "ti,tij,tj->t",
            x,
            sigma,
            x,
        )

        return pd.Series(
            variance,
            index=dates,
            name=name,
        )

    def variance_from_weights(
        self,
        data,
        covariance_result: CovarianceResult,
    ) -> pd.Series:
        """One-step-ahead approximate portfolio return variance (return^2).

        Covariance at t must be forecast using data through t-1.
        Computes w[t-1].T @ Sigma[t|t-1] @ w[t-1].
        """
        return self._historical_variance(
            self.weights(data),
            covariance_result,
            "variance_from_weights",
        )

    def variance_from_position_values(
        self,
        data,
        covariance_result: CovarianceResult,
    ) -> pd.Series:
        """One-step-ahead approximate monetary loss variance (currency^2).

        Covariance at t must be forecast using data through t-1.
        Computes a[t-1].T @ Sigma[t|t-1] @ a[t-1].
        """
        return self._historical_variance(
            self.position_values(data),
            covariance_result,
            "variance_from_position_values",
        )

    def _forecast_variance(
        self,
        exposures: pd.DataFrame,
        covariance_forecast: pd.DataFrame,
        name: str,
    ) -> pd.Series:
        """Apply a single next-day covariance forecast to current exposures."""
        assets = pd.Index(covariance_forecast.columns)

        if (
            not assets.is_unique
            or not covariance_forecast.index.is_unique
            or set(assets) != set(self.assets)
            or set(covariance_forecast.index) != set(self.assets)
        ):
            raise ValueError(
                "Forecast covariance rows/columns must match assets."
            )

        x = exposures.iloc[-1].reindex(assets)

        if x.isna().any():
            raise ValueError(
                "Missing current exposures for covariance forecast."
            )

        sigma = covariance_forecast.reindex(
            index=assets,
            columns=assets,
        ).to_numpy()

        variance = float(
            x.to_numpy() @ sigma @ x.to_numpy()
        )

        # A future trading date may not yet be known.
        # Index the forecast by its origin date.
        origin_date = exposures.index[-1]

        return pd.Series(
            [variance],
            index=[origin_date],
            name=name,
        )

    def forecast_variance_from_weights(
        self,
        data,
        covariance_forecast: pd.DataFrame,
    ) -> pd.Series:
        """Next-day return variance using weights at the forecast origin."""
        return self._forecast_variance(
            self.weights(data),
            covariance_forecast,
            "variance_from_weights",
        )

    def forecast_variance_from_position_values(
        self,
        data,
        covariance_forecast: pd.DataFrame,
    ) -> pd.Series:
        """Next-day monetary loss variance using current position values."""
        return self._forecast_variance(
            self.position_values(data),
            covariance_forecast,
            "variance_from_position_values",
        )








# import numpy as np
# import pandas as pd

# from co_variance_results import CovarianceResult, VarianceResult

# class Portfolio:
#     """Simple buy-and-hold stock portfolio."""

#     def __init__(self, holdings: dict):
#         """
#         Parameters
#         ----------
#         holdings : dict
#             Mapping from asset ticker to number of shares held.
#             Example: {"AAPL": 2, "MSFT": 3}
#         """

#         if not isinstance(holdings, dict):
#             raise TypeError("holdings must be a dictionary.")

#         if len(holdings) == 0:
#             raise ValueError("holdings must not be empty.")

#         # probably have to differentiate between a multi asset portfolio and a single asset portfolio  
#         self.holdings = pd.Series(holdings, dtype=float)


#         if (self.holdings < 0).any():
#             raise ValueError("Negative holdings are not allowed.")



#         self.assets = self.holdings.index.tolist()

#         # prices = market_data.download_prices(
#         #     self.assets,
#         #     start,
#         #     end
#         # )

#     def position_values(self, data):
#         """
#         data value of each individual position over time.

#         Returns
#         -------
#         pandas.DataFrame
#         """
#         return data.prices * self.holdings

#     def value(self, data):
#         """
#         Total portfolio value over time.

#         Returns
#         -------
#         pandas.Series
#         """
#         return self.position_values(data).sum(
#             axis=1,
#             min_count=len(self.holdings)
#         )

#     # def variance(self, covariance_matrices):
#     #     """
#     #     Portfolio variance over time.

#     #     Returns
#     #     -------
#     #     pandas.Series
#     #     """
#     #     # return self.weights(data) @ covariance_matrices @ self.weights(data) 

#     def portfolio_variance(
#         self,
#         data,
#         covariance_result: CovarianceResult,
#     ) -> pd.Series:
        
#         weights = (
#         self.weights(data)
#         .shift(1)
#         .reindex(
#             index=covariance_result.dates,
#             columns=covariance_result.assets,
#             )
#         )

#         W = weights.to_numpy()
#         Sigma = covariance_result.covariances

#         portfolio_variance = np.einsum(
#             "ti,tij,tj->t",
#             W,
#             Sigma,
#             W,
#         )

#         return pd.Series(
#             portfolio_variance,
#             index=covariance_result.dates,
#             name="portfolio_variance",
#         )

    
#     def variance_forecast(
#         self,
#         data,
#         covariance_forecast: pd.DataFrame,
#     ) -> pd.Series:

#         assets = covariance_forecast.columns

#         weights = (
#             self.weights(data)
#             .iloc[-1]
#             .reindex(assets)
#             .to_numpy()
#         )

#         Sigma = covariance_forecast.to_numpy()

#         portfolio_variance = (
#             weights
#             @ Sigma
#             @ weights
#         )

#         return pd.Series(
#             [portfolio_variance],
#             index=[data.prices.index[-1]],
#             name="portfolio_variance",
#         )


# #     def variance(
# #     self,
# #     data,
# #     covariance: CovarianceResult | pd.DataFrame,
# # ) -> pd.Series:

# #         weights = self.weights(data)

# #         # --------------------------------------------------------
# #         # Historical covariance series
# #         # --------------------------------------------------------

# #         if isinstance(covariance, CovarianceResult):

# #             dates = covariance.dates
# #             assets = covariance.assets

# #             # For return X_t, use portfolio weights known at t-1
# #             historical_weights = (
# #                 weights
# #                 .shift(1)
# #                 .reindex(index=dates, columns=assets)
# #             )

# #             W = historical_weights.to_numpy()
# #             Sigma = covariance.covariances

# #             portfolio_variance = np.einsum(
# #                 "ti,tij,tj->t",
# #                 W,
# #                 Sigma,
# #                 W,
# #             )

# #             return pd.Series(
# #                 portfolio_variance,
# #                 index=dates,
# #                 name="portfolio_variance",
# #             )

# #         # --------------------------------------------------------
# #         # Single covariance forecast
# #         # --------------------------------------------------------

# #         elif isinstance(covariance, pd.DataFrame):

# #             assets = covariance.columns

# #             # Forecast for t+1 uses weights known at t
# #             current_weights = (
# #                 weights
# #                 .iloc[-1]
# #                 .reindex(assets)
# #                 .to_numpy()
# #             )

# #             Sigma_forecast = covariance.to_numpy()

# #             portfolio_variance = (
# #                 current_weights
# #                 @ Sigma_forecast
# #                 @ current_weights
# #             )

# #             return pd.Series(
# #                 data=[portfolio_variance],
# #                 index=[weights.index[-1]],
# #                 name="portfolio_variance",
# #             )

# #         else:
# #             raise TypeError(
# #                 "covariance must be a CovarianceResult "
# #                 "or a covariance forecast DataFrame."
# #             )


#     def weights(self, data):
#         """
#         Portfolio weights of each asset over time.

#         Returns
#         -------
#         pandas.DataFrame
#         """
#         return self.position_values(data).div(
#             self.value(data),
#             axis=0
#         )

#     def risk_factors(self, data):
#         """
#         Log returns of the underlying stock data.prices.

#         Returns
#         -------
#         pandas.DataFrame
#         """
#         return data.log_returns()

#     def portfolio_weighted_log_return(self, data):
#         """
#         Portfolio return based on weighted asset log returns.

#         Uses portfolio weights from the previous day.

#         Returns
#         -------
#         pandas.Series
#         """

#         risk_factors = self.risk_factors(data)

#         previous_weights = (
#             self.weights(data)
#             .shift(1)
#             .reindex(risk_factors.index)
#         )

#         portfolio_returns = (
#             previous_weights * risk_factors
#         ).sum(
#             axis=1,
#             min_count=len(self.holdings),
#         )

#         return portfolio_returns.rename(
#             "portfolio_weighted_log_return"
#         ).dropna()

#     def loss(self, data):
#         """
#         Historical one-day portfolio losses.

#         Positive values represent losses and negative values gains.

#         Returns
#         -------
#         pandas.Series
#         """
#         simple_returns = np.expm1(  
#             self.risk_factors(data)
#         )

#         previous_position_values = (
#             self.position_values(data).shift(1)
#         )

#         losses = -(
#             simple_returns * previous_position_values
#         ).sum(
#             axis=1,
#             min_count=len(self.holdings)
#         )

#         return losses.dropna()


# if __name__ == "__main__":

#     import market_data

#     port = Portfolio(
#         holdings={
#             "AAPL": 2,
#             "MSFT": 3,
#         }
#     )

#     data = market_data.MarketData.from_yfinance(port.assets, "2000-01-01","2009-12-31")

#     print(type(data))
#     print(type(data.prices))

#     print(port.holdings)
#     print(port.value(data))
#     print(port.weights(data))
#     print(port.loss(data))  