"""Rolling one-step-ahead covariance and innovation-distribution forecasts.

Forecasters return ``(OneStepForecast, next_state)``.  The rolling engine
assigns each forecast to its *target* date, using only observations strictly
before that date.  Distribution parameters are saved at those same dates.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import models
from co_variance_results import CovarianceResult


@dataclass
class OneStepForecast:
    """One conditional covariance forecast and its innovation distribution.

    Parameters
    ----------
    covariance : pd.DataFrame
        Labeled (N, N) forecast for the next observation.
    distribution : str | None
        Distribution identifier accepted by ``arch`` (e.g. ``"StudentsT"``
        or ``"skewstudent"``).  None means no distribution was estimated.
    distribution_params : pd.Series | None
        Fitted innovation shape parameters, named and in the order required
        by the distribution object's ``parameter_names()``. None for a
        parameterless distribution (Normal) or when unspecified (EWMA).
    """

    covariance: pd.DataFrame
    distribution: str | None = None
    distribution_params: pd.Series | None = None


def ewma_forecaster(
    window: pd.DataFrame,
    state=None,
    decay: float = 0.96,
    initialization_window: int = 60,
    burn_in_window: int = 30,
) -> tuple[OneStepForecast, None]:
    """Estimate EWMA and forecast the next covariance for any asset count.

    EWMA specifies conditional covariance dynamics, but does not estimate
    an innovation distribution. ``state`` is currently unused.
    """
    result = models.ewma_covariance_estimator(
        window,
        decay=decay,
        initialization_window=initialization_window,
        burn_in_window=burn_in_window,
    )

    return OneStepForecast(
        covariance=models.ewma_forecast(window, result),
    ), None


def garch_forecaster(
    window: pd.DataFrame,
    state=None,
    p: int = 1,
    q: int = 1,
    o: int = 0,
    dist: str = "Normal",
) -> tuple[OneStepForecast, None]:
    """Fit univariate GARCH and retain fitted innovation shape parameters.

    ``arch`` jointly estimates the volatility and distribution parameters.
    An optional ``state`` is passed as optimization starting values; no
    state is passed forward by this adapter yet.
    """
    fit_result = models.fit_garch(
        window,
        p=p,
        q=q,
        o=o,
        dist=dist,
        starting_values=state,
    )

    distribution = fit_result.model.distribution
    param_names = distribution.parameter_names()

    # Use names rather than positions: arch's ppf expects this ordering.
    dist_params = (
        fit_result.params.loc[param_names].copy()
        if param_names else None
    )

    return OneStepForecast(
        covariance=models.garch_forecast(fit_result),
        distribution=dist,
        distribution_params=dist_params,
    ), None


def rolling_forecast(
    risk_factors: pd.DataFrame,
    window_length: int,
    forecaster: Callable[..., tuple[OneStepForecast, Any]],
    **forecaster_kwargs,
) -> CovarianceResult:
    """Compute rolling one-step-ahead conditional covariance forecasts.

    At target row t, fit on rows [t-window_length, ..., t-1].
    Save the forecast and fitted distribution parameters at target date t.
    Return one ``CovarianceResult`` for univariate or multivariate models.
    """
    if not isinstance(risk_factors, pd.DataFrame) or risk_factors.empty:
        raise ValueError("risk_factors must be a nonempty DataFrame.")
    if not risk_factors.index.is_unique or not risk_factors.index.is_monotonic_increasing:
        raise ValueError("Risk-factor dates must be unique and increasing.")
    if not risk_factors.columns.is_unique:
        raise ValueError("Risk-factor asset names must be unique.")
    if not np.isfinite(risk_factors.to_numpy(dtype=float)).all():
        raise ValueError("Risk factors must be finite.")
    if not isinstance(window_length, int) or not 1 <= window_length < len(risk_factors):
        raise ValueError("window_length must be between 1 and len(risk_factors) - 1.")
    if not callable(forecaster):
        raise TypeError("forecaster must be callable.")

    assets = risk_factors.columns
    target_dates = risk_factors.index[window_length:]
    matrices = []
    distributions = []
    params_by_date = []
    state = None

    for t in range(window_length, len(risk_factors)):
        window = risk_factors.iloc[t - window_length:t]
        forecast, state = forecaster(
            window,
            state=state,
            **forecaster_kwargs,
        )

        if not isinstance(forecast, OneStepForecast):
            raise TypeError("The forecaster must return a OneStepForecast.")

        matrix_df = forecast.covariance
        if not isinstance(matrix_df, pd.DataFrame):
            raise TypeError("Forecast covariance must be a DataFrame.")
        if (not matrix_df.index.is_unique
                or not matrix_df.columns.is_unique
                or set(matrix_df.index) != set(assets)
                or set(matrix_df.columns) != set(assets)):
            raise ValueError("Forecast matrix row/column labels must match risk factors.")

        # Align by asset names, never by implicit position.
        matrix = matrix_df.reindex(index=assets, columns=assets).to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("The forecaster returned a nonfinite covariance matrix.")
        matrices.append(matrix)

        distributions.append(forecast.distribution)
        params = forecast.distribution_params
        if params is not None:
            if not isinstance(params, pd.Series) or not params.index.is_unique:
                raise TypeError("Distribution parameters must be a named Series.")
            if not np.isfinite(params.to_numpy(dtype=float)).all():
                raise ValueError("Distribution parameters must be finite.")
            params = params.copy()
        params_by_date.append(params)

    if any(dist != distributions[0] for dist in distributions):
        raise ValueError("The innovation distribution changed between forecast windows.")

    if all(params is None for params in params_by_date):
        distribution_params = None
    elif any(params is None for params in params_by_date):
        raise ValueError("Missing distribution parameters in some forecast windows.")
    else:
        first_names = params_by_date[0].index
        if any(not params.index.equals(first_names) for params in params_by_date):
            raise ValueError("Distribution parameter names/order changed across windows.")

        distribution_params = pd.DataFrame(
            params_by_date,
            index=target_dates,
        )

    return CovarianceResult(
        covariances=np.stack(matrices),
        dates=target_dates,
        assets=assets,
        model=getattr(forecaster, "__name__", None),
        metadata={
            "kind": "out_of_sample_forecast",
            "horizon": 1,
            "window_length": window_length,
            "forecaster_kwargs": dict(forecaster_kwargs),
            "distribution": distributions[0],
            "distribution_params": distribution_params,
        },
    )


# import numpy as np, pandas as pd, matplotlib.pyplot as plt 

# # import portfolio
# # import market_data
# import models
# # import simulation
# # import risk_measure

# from co_variance_results import CovarianceResult, VarianceResult

# # def garch_forecaster(
# #     window: pd.DataFrame,
# #     state=None,
# #     p=1,
# #     q=1,
# #     o=0,
# #     dist="Normal",
# # ):
# #     fit_result = models.fit_garch(
# #         window,
# #         p=p,
# #         q=q,
# #         o=o,
# #         dist=dist,
# #         starting_values=state,
# #     )

# #     forecast = models.garch_forecast(fit_result)

# #     # new_state = fit_result.params.to_numpy()
# #     new_state = fit_result.params.to_numpy(copy=True)
    
#     #return forecast, new_state

# def garch_forecaster(
#     window: pd.DataFrame,
#     state=None,
#     p=1,
#     q=1,
#     o=0,
#     dist="Normal",
# ):
#     fit_result = models.fit_garch(
#         window,
#         p=p,
#         q=q,
#         o=o,
#         dist=dist,
#         starting_values=state,
#     )

#     forecast = models.garch_forecast(fit_result)

#     new_state = fit_result.params.to_numpy(copy=True)

#     return forecast,  None #new_state

# def univariate_ewma_forecaster(
#     window: pd.DataFrame,
#     state=None,
#     decay=0.96,
# ):
#     result = models.univariate_ewma_variance_estimator(
#         window,
#         decay=decay,
#     )

#     return models.univariate_ewma_forecast(
#         window,
#         result,
#     ), None

# def multivariate_ewma_forecaster(
#     window: pd.DataFrame,
#     state=None,
#     decay=0.96,
# ):
#     result = models.multivariate_ewma_covariance_estimator(
#         window,
#         decay=decay,
#     )

#     return models.multivariate_ewma_forecast(
#         window,
#         result,
#     ), None


# def rolling_forecast(
#     risk_factors: pd.DataFrame,
#     window_length: int,
#     forecaster,
#     **forecaster_kwargs,
# ) -> VarianceResult | CovarianceResult:

#     results = []
#     dates = []
# #
#     state = None

#     for t in range(window_length, len(risk_factors)):

#         window = risk_factors.iloc[
#             t - window_length:t
#         ]

#         forecast, state = forecaster(
#             window,
#             state=state,
#             **forecaster_kwargs,
#         )

#         target_date = risk_factors.index[t]

#         results.append(forecast)
#         dates.append(target_date)

#     # --------------------------------------------------------
#     # Univariate
#     # --------------------------------------------------------

#     if risk_factors.shape[1] == 1:

#         asset = risk_factors.columns[0]

#         variances = pd.Series(
#             [forecast.iloc[0] for forecast in results],
#             index=dates,
#             name=asset,
#         )

#         return VarianceResult(
#             variances=variances,
#             asset=asset,
#         )

#     # --------------------------------------------------------
#     # Multivariate
#     # --------------------------------------------------------

#     else:

#         assets = risk_factors.columns

#         covariances = np.stack([
#             forecast.to_numpy()
#             for forecast in results
#         ])

#         return CovarianceResult(
#             covariances=covariances,
#             dates=pd.Index(dates),
#             assets=assets,
#         )






# #     risk_factorsBT = portBT.risk_factors(dataBT)

# # import numpy as np, pandas as pd, matplotlib.pyplot as plt 

# # # import portfolio
# # # import market_data
# # # import models
# # # import simulation
# # # import risk_measure

# # import volatility_forecaster

# # class RollingForecastEngine:

# #     def __init__(
# #         self,
# #         window_length: int,
# #     ):
# #         self.window_length = window_length

# #     def run(
# #         self,
# #         risk_factors: pd.DataFrame,
# #         forecaster: volatility_forecaster.VolatilityForecaster,
# #     ):

# #         results = []

# #         for t in range(
# #             self.window_length,
# #             len(risk_factors),
# #         ):

# #             window = risk_factors.iloc[
# #                 t - self.window_length:t
# #             ]

# #             forecast = forecaster.forecast(
# #                 window
# #             )

# #             target_date = risk_factors.index[t]

# #             results.append(
# #                 (target_date, forecast)
# #             )

# #         return results


# # # def ewma_forecaster(window: pd.DataFrame) -> pd.Series:
# # #      variance_ewma_t = models.univariate_ewma_forecast(
# # #         window,
# # #         ewma_result,
# # #     )



# # # # --------------------------------------------------------
# # # # EWMA variance path
# # # # --------------------------------------------------------

# # # ewma_result = models.univariate_ewma_variance_estimator(
# # #     risk_factorsBT
# # # )

# # # # Normal quantiles do not change through time
# # # q = simulation.parametric_quantiles(
# # #     "Normal",
# # #     [0.01, 0.05],
# # # )


# # # # --------------------------------------------------------
# # # # Backtest
# # # # --------------------------------------------------------


# # # var_ewma_results = []
# # # var_garch11_results = []

# # # estimation_window_length = 500 # probably should be more than init_window from ewma 
# # # testing_window_length = len(risk_factorsBT)


# # # for t in range(estimation_window_length, testing_window_length):

# # #     estimation_first_day = t - estimation_window_length
# # #     estimation_last_day = t

# # #     window = risk_factorsBT.iloc[
# # #         estimation_first_day:estimation_last_day
# # #     ]

# # #     # ----------------------------------
# # #     # EWMA: Use all the history or rolling forecast window????
# # #     # ----------------------------------

# # #     # Forecast sigma^2_(t | t-1)
# # #     variance_ewma_t = models.univariate_ewma_forecast(
# # #         window,
# # #         ewma_result,
# # #     )

# # #     # VaR forecast for t
# # #     var_ewma_t = risk_measure.parametric_value_at_risk(
# # #         variance_ewma_t,
# # #         q,
# # #     )

# # #     # ewma_forecast is indexed by forecast origin t-1,
# # #     # but for backtesting we want the target date t
# # #     target_date = risk_factorsBT.index[t]

# # #     var_ewma_t.index = [target_date]

# # #     var_ewma_results.append(var_ewma_t)


# # #     # ----------------------------------
# # #     # GARCH(1,1) with normal innovations
# # #     # ----------------------------------

# # #     garch11_fit = models.fit_garch(window,1,1,0,"Normal")

# # #     variance_garch11_t = models.garch_forecast(garch11_fit)

# # #     var_garch11_t = risk_measure.parametric_value_at_risk(variance_garch11_t,q)

# # #     var_garch11_t.index = [target_date]
    
# # #     var_garch11_results.append(var_garch11_t)


# # # Var_ewma_data_frame = pd.concat(var_ewma_results)
# # # Var_garch11_data_frame = pd.concat(var_garch11_results)


# # # print(Var_ewma_data_frame)
# # # Var_ewma_data_frame.plot()

# # # print(Var_garch11_data_frame)
# # # Var_garch11_data_frame.plot()





