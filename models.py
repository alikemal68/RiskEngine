import numpy as np, pandas as pd

import portfolio
import market_data

from dataclasses import dataclass, field
from typing import Any

@dataclass
class CovarianceResult:
    """
    Time series of covariance matrices.
    """

    covariances: np.ndarray
    dates: pd.Index
    assets: pd.Index

    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict) 
    # default_factory=dict makes a fresh dictionary for every object 
    # to prevent that multiple instances could accidentally share the same dictionary.


@dataclass
class VarianceResult:
    """
    Time series of variances.
    """

    variances: float
    dates: pd.Index
    assets: pd.Index

    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def ewma_core(
    X: np.ndarray,
    sigma0: np.ndarray,
    decay: float,
) -> np.ndarray:

    T, N = X.shape
    # Doppelt 

    covariances = np.empty((T, N, N))
    covariances[0] = sigma0

    for t in range(1, T):
        x_prev = X[t - 1]

        covariances[t] = (
            decay * covariances[t - 1]
            + (1 - decay) * np.outer(x_prev, x_prev)
        )

    return covariances


def univariate_ewma_variance_estimator(
        risk_factor: pd.Series,
        initial_var=None,
        decay=0.96,
        initialization_window=60,
        burn_in_window=30
) -> VarianceResult:

    T = len(risk_factor)

    X = risk_factor.to_numpy().reshape(-1, 1)

    # Initial variance
    if initial_var is None:

        variance0 = risk_factor.iloc[:initialization_window].var()

        start = initialization_window

    else:

        variance0 = float(initial_var)

        start = 0

    # Core expects an N x N initial covariance matrix.
    # For N = 1 this is just a 1 x 1 matrix.
    sigma0 = np.array([[variance0]])

    core_result = ewma_core(
        X[start:],
        sigma0,
        decay
    )

    variances = np.full(T, np.nan)

    variances[start:] = core_result[:, 0, 0]

    # Burn-in
    variances[
        start:start + burn_in_window
    ] = np.nan

    return VarianceResult(
        variances=pd.Series(
            variances,
            index=risk_factor.index,
            name=risk_factor.name,
        ),
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

def multivariate_ewma_covariance_estimator(
        risk_factor,
        initial_covar=None,
        decay=0.96,
        initialization_window=60,
        burn_in_window=30):

    T, N = risk_factor.shape
    X = risk_factor.to_numpy()

    # Initial covariance
    if initial_covar is None:

        sigma0 = (
            risk_factor
            .iloc[:initialization_window]
            .cov()
            .to_numpy()
        )

        # sigma0 is the covariance forecast for the first
        # observation AFTER the initialization sample
        start = initialization_window

    else:

        sigma0 = np.asarray(initial_covar, dtype=float)

        # supplied sigma0 is assumed to contain information
        # available before X[0]
        start = 0

    # --------------------------------------------------
    # Run recursion only on observations after sigma0
    # --------------------------------------------------
    core_result = ewma_core(
        X[start:],
        sigma0,
        decay
    )

    # Full-size result aligned with original dates
    covariances = np.full(
        (T, N, N),
        np.nan
    )

    covariances[start:] = core_result

    # Burn-in
    covariances[
        start:start + burn_in_window
    ] = np.nan

    return CovarianceResult(
        covariances=covariances,
        dates=risk_factor.index,
        assets=risk_factor.columns,
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


if __name__ == "__main__":

    port = portfolio.Portfolio(
            holdings={
                "AAPL": 2,
                # "MSFT": 3,
            }
        )
    
    data = market_data.MarketData.from_yfinance(port.assets, "2000-01-01", "2000-06-01") #"2009-12-31")

    # print(type(data))
    # print(type(data.prices))

    # print(port.holdings)
    # print(port.value(data))
    # print(port.weights(data))
    # print(port.loss(data))  

    my_risk_factors = port.risk_factors(data)

    print(EWMA_covariances_estimator(my_risk_factors))


