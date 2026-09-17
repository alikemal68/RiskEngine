from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import arch 
from arch.univariate.base import ARCHModelResult


# ============================================================
# Result objects
# ============================================================

@dataclass
class CovarianceResult:
    """Time series of covariance matrices."""

    covariances: np.ndarray
    dates: pd.Index
    assets: pd.Index

    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VarianceResult:
    """Time series of variances."""

    variances: pd.Series
    asset: str

    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Shared numerical EWMA core
# ============================================================

def ewma_recursion_core(
    X: np.ndarray,
    sigma0: np.ndarray,
    decay: float,
) -> np.ndarray:
    """
    EWMA recursion for an arbitrary number of risk factors.

    Parameters
    ----------
    X : np.ndarray
        Shape (T, N).

    sigma0 : np.ndarray
        Initial covariance matrix with shape (N, N).

    decay : float
        EWMA decay parameter.

    Returns
    -------
    np.ndarray
        Covariance matrices with shape (T, N, N).
    """

    T, N = X.shape

    covariances = np.empty((T, N, N))

    covariances[0] = sigma0

    for t in range(1, T):

        x_prev = X[t - 1]

        covariances[t] = (
            decay * covariances[t - 1]
            + (1 - decay) * np.outer(x_prev, x_prev)
        )

    return covariances


# ============================================================
# Univariate EWMA
# ============================================================

def univariate_ewma_variance_estimator(
    risk_factor: pd.DataFrame,
    initial_var=None,
    decay=0.96,
    initialization_window=60,
    burn_in_window=30,
) -> VarianceResult:
    """
    Estimate a univariate EWMA variance process.

    risk_factor must be a one-column DataFrame.
    """

    if risk_factor.shape[1] != 1:
        raise ValueError(
            "Univariate EWMA requires exactly one risk factor."
        )

    T = len(risk_factor)

    # Already has shape (T, 1)
    X = risk_factor.to_numpy()

    asset = risk_factor.columns[0]

    # --------------------------------------------------------
    # Initial variance
    # --------------------------------------------------------

    if initial_var is None:

        variance0 = (
            risk_factor
            .iloc[:initialization_window, 0]
            .var()
        )

        start = initialization_window

    else:

        variance0 = float(initial_var)

        start = 0

    # The shared core expects an N x N covariance matrix.
    # In the univariate case N = 1.
    sigma0 = np.array([[variance0]])

    # --------------------------------------------------------
    # EWMA recursion
    # --------------------------------------------------------

    recursion_result = ewma_recursion_core(
        X[start:],
        sigma0,
        decay,
    )

    # Extract the scalar from each 1 x 1 covariance matrix
    variances = np.full(T, np.nan)

    variances[start:] = recursion_result[:, 0, 0]

    # --------------------------------------------------------
    # Burn-in
    # --------------------------------------------------------

    variances[
        start:start + burn_in_window
    ] = np.nan

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    variance_series = pd.Series(
        variances,
        index=risk_factor.index,
        name=asset,
    )

    return VarianceResult(
        variances=variance_series,
        asset=asset,
        model="univariate EWMA",
        metadata={
            "decay": decay,
            "initialization_window": (
                initialization_window
                if initial_var is None
                else None
            ),
            "burn_in_window": burn_in_window,
        },
    )


# ============================================================
# Multivariate EWMA
# ============================================================

def multivariate_ewma_covariance_estimator(
    risk_factors: pd.DataFrame,
    initial_covar=None,
    decay=0.96,
    initialization_window=60,
    burn_in_window=30,
) -> CovarianceResult:
    """
    Estimate a multivariate EWMA covariance process.

    risk_factors must contain at least two columns.
    """

    if risk_factors.shape[1] < 2:
        raise ValueError(
            "Multivariate EWMA requires at least two risk factors."
        )

    T, N = risk_factors.shape

    X = risk_factors.to_numpy()

    # --------------------------------------------------------
    # Initial covariance
    # --------------------------------------------------------

    if initial_covar is None:

        sigma0 = (
            risk_factors
            .iloc[:initialization_window]
            .cov()
            .to_numpy()
        )

        # sigma0 is the forecast for the first observation
        # after the initialization sample
        start = initialization_window

    else:

        sigma0 = np.asarray(
            initial_covar,
            dtype=float,
        )

        # Supplied sigma0 is assumed to contain information
        # available before X[0]
        start = 0

    # --------------------------------------------------------
    # EWMA recursion
    # --------------------------------------------------------

    recursion_result = ewma_recursion_core(
        X[start:],
        sigma0,
        decay,
    )

    # Full-size output aligned with original dates
    covariances = np.full(
        (T, N, N),
        np.nan,
    )

    covariances[start:] = recursion_result

    # --------------------------------------------------------
    # Burn-in
    # --------------------------------------------------------

    covariances[
        start:start + burn_in_window
    ] = np.nan

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return CovarianceResult(
        covariances=covariances,
        dates=risk_factors.index,
        assets=risk_factors.columns,
        model="multivariate EWMA",
        metadata={
            "decay": decay,
            "initialization_window": (
                initialization_window
                if initial_covar is None
                else None
            ),
            "burn_in_window": burn_in_window,
        },
    )


def fit_garch(
    risk_factors: pd.DataFrame,
    p=1,
    q=1,
    o=0,
    dist="Normal",
) -> ARCHModelResult:

    if risk_factors.shape[1] != 1:
        raise ValueError("Univariate GARCH requires exactly one risk factor.")
    
    y = risk_factors.iloc[:, 0]

    model = arch.arch_model(
        y,
        mean="Zero",
        vol="Garch",
        p=p,
        o=o,
        q=q,
        dist=dist,
    )

    return model.fit(disp="off")

# def garch_conditional_variance(
#     fit_result: ARCHModelResult,
#     risk_factor: pd.DataFrame, # RETHINK THIS!!!!!
# ) -> VarianceResult:

#     if risk_factor.shape[1] != 1:
#         raise ValueError(
#             "GARCH conditional variance requires exactly one risk factor."
#         )

#     asset = risk_factor.columns[0]

#     variance_series = (fit_result.conditional_volatility**2).rename(asset)

#     return VarianceResult(
#         variances=variance_series,
#         asset=asset,
#     )
    
def garch_conditional_variance(
    fit_result: ARCHModelResult,
) -> VarianceResult:

    y = fit_result.model.y

    if not isinstance(y, pd.Series):
        raise TypeError(
            "Expected the fitted GARCH model to contain a pandas Series."
        )

    asset = y.name

    variance_series = (fit_result.conditional_volatility**2).rename(asset)

    return VarianceResult(
        variances=variance_series,
        asset=asset,
        model="GARCH",
        metadata={
            "parameters": fit_result.params.to_dict(),
            "distribution": fit_result.model.distribution.name,
        },
    )

def garch_forecast(
        fit_result: ARCHModelResult
        ):
    forecasts = fit_result.forecast(horizon=1,  align="origin")

    # cond_mean = forecasts.mean["h.1"]
    cond_var = forecasts.variance#.iloc[-1]

    return cond_var



