"""Conditional covariance estimation and one-step-ahead forecasting.

Risk-factor observations are decimal log returns. GARCH is fitted to returns
scaled by 100 for numerical stability, and its variances are converted back.
"""

import numpy as np
import pandas as pd
import arch
from arch.univariate.base import ARCHModelResult

from co_variance_results import CovarianceResult


def ewma_covariance_estimator(
    risk_factors: pd.DataFrame,
    initial_covariance=None,
    decay: float = 0.96,
    initialization_window: int = 60,
    burn_in_window: int = 30,
) -> CovarianceResult:
    """Estimate an EWMA conditional covariance path for one or more assets.

    Sigma[t] is the covariance conditional on observations through t-1.
    If no initial matrix is supplied, it is estimated from the first
    ``initialization_window`` observations; earlier output is NaN.
    ``burn_in_window`` additional matrices are masked, but are still
    calculated internally so the recursion remains continuous.
    """
    if not isinstance(risk_factors, pd.DataFrame) or risk_factors.empty:
        raise ValueError("risk_factors must be a nonempty DataFrame.")
    if not risk_factors.index.is_unique or not risk_factors.index.is_monotonic_increasing:
        raise ValueError("Risk-factor dates must be unique and ordered.")
    if not risk_factors.columns.is_unique:
        raise ValueError("Risk-factor asset labels must be unique.")
    if not np.isfinite(risk_factors.to_numpy(dtype=float)).all():
        raise ValueError("Risk factors must contain only finite values.")
    if not 0 < decay < 1:
        raise ValueError("decay must be strictly between 0 and 1.")
    if not isinstance(burn_in_window, int) or burn_in_window < 0:
        raise ValueError("burn_in_window must be a nonnegative integer.")

    X = risk_factors.to_numpy(dtype=float)
    T, N = X.shape

    if initial_covariance is None:
        if (not isinstance(initialization_window, int)
                or initialization_window < 2
                or initialization_window >= T):
            raise ValueError("initialization_window must be between 2 and T-1.")

        sigma0 = risk_factors.iloc[:initialization_window].cov().to_numpy()
        start = initialization_window
    else:
        if isinstance(initial_covariance, pd.DataFrame):
            assets = risk_factors.columns
            if (not initial_covariance.index.is_unique
                    or not initial_covariance.columns.is_unique
                    or set(initial_covariance.index) != set(assets)
                    or set(initial_covariance.columns) != set(assets)):
                raise ValueError("Initial covariance labels must match risk factors.")
            initial_covariance = initial_covariance.loc[assets, assets]

        sigma0 = np.asarray(initial_covariance, dtype=float)
        if N == 1 and sigma0.ndim == 0:
            sigma0 = sigma0.reshape(1, 1)
        start = 0

    if sigma0.shape != (N, N) or not np.isfinite(sigma0).all():
        raise ValueError(f"initial_covariance must have finite shape ({N}, {N}).")
    if not np.allclose(sigma0, sigma0.T, atol=1e-12, rtol=1e-10):
        raise ValueError("initial_covariance must be symmetric.")
    if start + burn_in_window >= T:
        raise ValueError("Not enough observations left after initialization/burn-in.")

    # Always store (T, N, N), including the univariate case (N = 1).
    covariances = np.full((T, N, N), np.nan)
    covariances[start] = sigma0

    # At date t, use the risk-factor realization from date t-1.
    for t in range(start + 1, T):
        covariances[t] = (
            decay * covariances[t - 1]
            + (1 - decay) * np.outer(X[t - 1], X[t - 1])
        )

    covariances[start:start + burn_in_window] = np.nan

    return CovarianceResult(
        covariances=covariances,
        dates=risk_factors.index,
        assets=risk_factors.columns,
        model="EWMA",
        metadata={
            "decay": decay,
            "initialization_window": initialization_window if start else None,
            "burn_in_window": burn_in_window,
            "kind": "fitted_conditional_covariance",
        },
    )


def ewma_forecast(
    risk_factors: pd.DataFrame,
    covariance_result: CovarianceResult,
) -> pd.DataFrame:
    """Forecast Sigma[t+1|t] from the latest EWMA covariance and return.

    Returns a labeled N-by-N DataFrame even when N = 1. The forecast
    origin is the final date in ``risk_factors``; the target is next period.
    """
    if risk_factors.empty or not risk_factors.columns.equals(covariance_result.assets):
        raise ValueError("Risk-factor columns must match covariance-result assets.")
    if risk_factors.index[-1] != covariance_result.dates[-1]:
        raise ValueError("The final risk-factor and covariance dates must match.")

    decay = covariance_result.metadata["decay"]
    sigma_t = covariance_result.covariances[-1]
    x_t = risk_factors.iloc[-1].to_numpy(dtype=float)

    if not np.isfinite(sigma_t).all() or not np.isfinite(x_t).all():
        raise ValueError("The most recent covariance and returns must be finite.")

    sigma_next = decay * sigma_t + (1 - decay) * np.outer(x_t, x_t)
    return pd.DataFrame(
        sigma_next,
        index=covariance_result.assets,
        columns=covariance_result.assets,
    )


def fit_garch(
    risk_factors: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    o: int = 0,
    dist: str = "Normal",
    starting_values=None,
) -> ARCHModelResult:
    """Fit a univariate zero-mean GARCH model to decimal log returns."""
    if not isinstance(risk_factors, pd.DataFrame) or risk_factors.shape[1] != 1:
        raise ValueError("GARCH requires a DataFrame with exactly one risk factor.")
    if risk_factors.empty or not np.isfinite(risk_factors.to_numpy(dtype=float)).all():
        raise ValueError("GARCH risk factors must be nonempty and finite.")

    y = 100 * risk_factors.iloc[:, 0]
    model = arch.arch_model(
        y, mean="Zero", vol="Garch", p=p, o=o, q=q, dist=dist
    )
    return model.fit(disp="off", starting_values=starting_values)


def garch_conditional_variance(
    fit_result: ARCHModelResult,
) -> CovarianceResult:
    """Return the fitted, in-sample GARCH variance path as (T, 1, 1)."""
    y = fit_result.model.y
    if not isinstance(y, pd.Series):
        raise TypeError("The fitted GARCH model must contain a pandas Series.")

    variance = np.asarray(fit_result.conditional_volatility, dtype=float) ** 2 / 100**2
    return CovarianceResult(
        covariances=variance[:, None, None],
        dates=y.index,
        assets=pd.Index([y.name]),
        model="GARCH",
        metadata={
            "parameters": fit_result.params.to_dict(),
            "distribution": fit_result.model.distribution.name,
            "kind": "fitted_conditional_covariance",
        },
    )


def garch_forecast(
    fit_result: ARCHModelResult,
) -> pd.DataFrame:
    """Return the GARCH one-step-ahead variance as a labeled 1-by-1 matrix."""
    asset = fit_result.model.y.name
    forecast = fit_result.forecast(horizon=1, align="origin")
    variance = float(forecast.variance["h.1"].iloc[-1]) / 100**2
    if not np.isfinite(variance):
        raise ValueError("GARCH returned a nonfinite variance forecast.")

    return pd.DataFrame([[variance]], index=[asset], columns=[asset])























# # from dataclasses import dataclass, field
# # from typing import Any

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from co_variance_results import CovarianceResult, VarianceResult

# import arch 
# from arch.univariate.base import ARCHModelResult


# # ============================================================
# # Result objects
# # ============================================================

# # @dataclass
# # class CovarianceResult:
# #     """Time series of covariance matrices."""

# #     covariances: pd.Series
# #     dates: pd.Index
# #     assets: pd.Index

# #     model: str | None = None
# #     metadata: dict[str, Any] = field(default_factory=dict)


# # @dataclass
# # class VarianceResult:
# #     """Time series of variances."""

# #     variances: pd.Series
# #     asset: str

# #     model: str | None = None
# #     metadata: dict[str, Any] = field(default_factory=dict)


# # ============================================================
# # Shared numerical EWMA core
# # ============================================================

# def _ewma_recursion_core(
#     X: np.ndarray,
#     sigma0: np.ndarray,
#     decay: float,
# ) -> np.ndarray:
#     """
#     EWMA recursion for an arbitrary number of risk factors.

#     Parameters
#     ----------
#     X : np.ndarray
#         Shape (T, N).

#     sigma0 : np.ndarray
#         Initial covariance matrix with shape (N, N).

#     decay : float
#         EWMA decay parameter.

#     Returns
#     -------
#     np.ndarray
#         Covariance matrices with shape (T, N, N).
#     """

#     T, N = X.shape

#     covariances = np.empty((T, N, N))

#     covariances[0] = sigma0

#     for t in range(1, T):

#         x_prev = X[t - 1]

#         covariances[t] = (
#             decay * covariances[t - 1]
#             + (1 - decay) * np.outer(x_prev, x_prev)
#         )

#     return covariances


# # ============================================================
# # Univariate EWMA
# # ============================================================

# def univariate_ewma_variance_estimator(
#     risk_factor: pd.DataFrame,
#     initial_var=None,
#     decay=0.96,
#     initialization_window=60,
#     burn_in_window=30,
# ) -> VarianceResult:
#     """
#     Estimate a univariate EWMA variance process.

#     risk_factor must be a one-column DataFrame.
#     """

#     if risk_factor.shape[1] != 1:
#         raise ValueError(
#             "Univariate EWMA requires exactly one risk factor."
#         )

#     T = len(risk_factor)

#     # Already has shape (T, 1)
#     X = risk_factor.to_numpy()

#     asset = risk_factor.columns[0]

#     # --------------------------------------------------------
#     # Initial variance
#     # --------------------------------------------------------

#     if initial_var is None:

#         variance0 = (
#             risk_factor
#             .iloc[:initialization_window, 0]
#             .var()
#         )

#         start = initialization_window

#     else:

#         variance0 = float(initial_var)

#         start = 0

#     # The shared core expects an N x N covariance matrix.
#     # In the univariate case N = 1.
#     sigma0 = np.array([[variance0]])

#     # --------------------------------------------------------
#     # EWMA recursion
#     # --------------------------------------------------------

#     recursion_result = _ewma_recursion_core(
#         X[start:],
#         sigma0,
#         decay,
#     )

#     # Extract the scalar from each 1 x 1 covariance matrix
#     variances = np.full(T, np.nan)

#     variances[start:] = recursion_result[:, 0, 0]

#     # --------------------------------------------------------
#     # Burn-in
#     # --------------------------------------------------------

#     variances[
#         start:start + burn_in_window
#     ] = np.nan

#     # --------------------------------------------------------
#     # Result
#     # --------------------------------------------------------

#     variance_series = pd.Series(
#         variances,
#         index=risk_factor.index,
#         name=asset,
#     )

#     return VarianceResult(
#         variances=variance_series,
#         asset=asset,
#         model="univariate EWMA",
#         metadata={
#             "decay": decay,
#             "initialization_window": (
#                 initialization_window
#                 if initial_var is None
#                 else None
#             ),
#             "burn_in_window": burn_in_window,
#         },
#     )


# # ============================================================
# # Multivariate EWMA
# # ============================================================

# def multivariate_ewma_covariance_estimator(
#     risk_factors: pd.DataFrame,
#     initial_covar=None,
#     decay=0.96,
#     initialization_window=60,
#     burn_in_window=30,
# ) -> CovarianceResult:
#     """
#     Estimate a multivariate EWMA covariance process.

#     risk_factors must contain at least two columns.
#     """

#     if risk_factors.shape[1] < 2:
#         raise ValueError(
#             "Multivariate EWMA requires at least two risk factors."
#         )

#     T, N = risk_factors.shape

#     X = risk_factors.to_numpy()

#     # --------------------------------------------------------
#     # Initial covariance
#     # --------------------------------------------------------

#     if initial_covar is None:

#         sigma0 = (
#             risk_factors
#             .iloc[:initialization_window]
#             .cov()
#             .to_numpy()
#         )

#         # sigma0 is the forecast for the first observation
#         # after the initialization sample
#         start = initialization_window

#     else:

#         sigma0 = np.asarray(
#             initial_covar,
#             dtype=float,
#         )

#         # Supplied sigma0 is assumed to contain information
#         # available before X[0]
#         start = 0

#     # --------------------------------------------------------
#     # EWMA recursion
#     # --------------------------------------------------------

#     recursion_result = _ewma_recursion_core(
#         X[start:],
#         sigma0,
#         decay,
#     )

#     # Full-size output aligned with original dates
#     covariances = np.full(
#         (T, N, N),
#         np.nan,
#     )

#     covariances[start:] = recursion_result

#     # --------------------------------------------------------
#     # Burn-in
#     # --------------------------------------------------------

#     covariances[
#         start:start + burn_in_window
#     ] = np.nan

#     # --------------------------------------------------------
#     # Result
#     # --------------------------------------------------------

#     return CovarianceResult(
#         covariances=covariances,
#         dates=risk_factors.index,
#         assets=risk_factors.columns,
#         model="multivariate EWMA",
#         metadata={
#             "decay": decay,
#             "initialization_window": (
#                 initialization_window
#                 if initial_covar is None
#                 else None
#             ),
#             "burn_in_window": burn_in_window,
#         },
#     )

# # def ewma_forecast(
# #     risk_factor: pd.DataFrame,
# #     variance_result: VarianceResult | CovarianceResult
# # ) -> pd.Series:

# #     date = risk_factor.index[-1]
# #     asset = risk_factor.columns[0]

# #     # check that risk_factor.index[-1] matches variance_result.variances.index[-1] ???

# #     decay = variance_result.metadata["decay"]

# #     last_cond_var = variance_result.variances.iloc[-1]
# #     last_risk_factor = risk_factor.iloc[-1, 0]

# #     forecast = (
# #         decay * last_cond_var
# #         + (1 - decay) * last_risk_factor**2
# #     )

# #     return pd.Series(
# #         data=[forecast],
# #         index=[date],
# #         name=asset,
# #     )

# def _ewma_forecast_core(
#     last_risk_factors: np.ndarray,
#     last_covar: np.ndarray,
#     decay: float,
# ) -> np.ndarray:
#     # """
#     # One-step-ahead EWMA covariance forecast.
#     # """

#     return (
#         decay * last_covar
#         + (1 - decay)
#         * np.outer(last_risk_factors, last_risk_factors)
#     )

# def univariate_ewma_forecast(
#     risk_factor: pd.DataFrame,
#     variance_result: VarianceResult,
# ) -> pd.Series:

#     if risk_factor.shape[1] != 1:
#         raise ValueError(
#             "Univariate EWMA forecast requires exactly one risk factor."
#         )

#     date = risk_factor.index[-1]
#     asset = risk_factor.columns[0]

#     decay = variance_result.metadata["decay"]

#     last_variance = variance_result.variances.loc[date]

#     last_risk_factor = risk_factor.iloc[-1].to_numpy()

#     forecast = _ewma_forecast_core(
#         last_risk_factors=last_risk_factor,
#         last_covar=np.array([[last_variance]]),
#         decay=decay,
#     )

#     return pd.Series(
#         data=[forecast[0, 0]],
#         index=[date],
#         name=asset,
#     )

# def multivariate_ewma_forecast(
#     risk_factors: pd.DataFrame,
#     covariance_result: CovarianceResult,
# ) -> pd.DataFrame:

#     if risk_factors.shape[1] < 2:
#         raise ValueError(
#             "Multivariate EWMA forecast requires at least two risk factors."
#         )

#     assets = risk_factors.columns

#     decay = covariance_result.metadata["decay"]

#     last_covar = covariance_result.covariances[-1]

#     last_risk_factors = (
#         risk_factors.iloc[-1].to_numpy()
#     )

#     forecast = _ewma_forecast_core(
#         last_risk_factors=last_risk_factors,
#         last_covar=last_covar,
#         decay=decay,
#     )

#     return pd.DataFrame(
#         forecast,
#         index=assets,
#         columns=assets,
#     )


# def fit_garch(
#     risk_factors: pd.DataFrame,
#     p=1,
#     q=1,
#     o=0,
#     dist="Normal",
#     starting_values=None,
# ):

#     y = 100 * risk_factors.iloc[:, 0]

#     model = arch.arch_model(
#         y,
#         mean="Zero",
#         vol="Garch",
#         p=p,
#         o=o,
#         q=q,
#         dist=dist,
#     )

#     return model.fit(
#         disp="off",
#         starting_values=starting_values,
#     )
    
# def garch_conditional_variance(
#     fit_result: ARCHModelResult,
# ) -> VarianceResult:

#     y = fit_result.model.y

#     if not isinstance(y, pd.Series):
#         raise TypeError(
#             "Expected the fitted GARCH model to contain a pandas Series."
#         )

#     asset = y.name

#     variance_series = (fit_result.conditional_volatility**2).rename(asset) / 100**2

#     return VarianceResult(
#         variances=variance_series,
#         asset=asset,
#         model="GARCH",
#         metadata={
#             "parameters": fit_result.params.to_dict(),
#             "distribution": fit_result.model.distribution.name,
#         },
#     )

# def garch_forecast(
#     fit_result: ARCHModelResult,
# ) -> pd.Series:

#     asset = fit_result.model.y.name

#     forecast = fit_result.forecast(
#         horizon=1,
#         align="origin",
#     )

#     cond_var = forecast.variance["h.1"].rename(asset) / 100**2

#     return cond_var



