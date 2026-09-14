from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


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

def _ewma_core(
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

    core_result = _ewma_core(
        X[start:],
        sigma0,
        decay,
    )

    # Extract the scalar from each 1 x 1 covariance matrix
    variances = np.full(T, np.nan)

    variances[start:] = core_result[:, 0, 0]

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

    core_result = _ewma_core(
        X[start:],
        sigma0,
        decay,
    )

    # Full-size output aligned with original dates
    covariances = np.full(
        (T, N, N),
        np.nan,
    )

    covariances[start:] = core_result

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



if __name__ == "__main__":

    import portfolio
    import market_data

    port = portfolio.Portfolio(
        holdings={
            "AAPL": 2,
            "MSFT": 3
        }
    )

    data = market_data.MarketData.from_yfinance(
        port.assets,
        "2000-01-01",
        "2009-12-31",
    )

    risk_factors = port.risk_factors(data)

    result = multivariate_ewma_covariance_estimator(
        risk_factors
    )

    print(result)
    # print(result.variances)


# import numpy as np, pandas as pd

# import portfolio
# import market_data

# from dataclasses import dataclass, field
# from typing import Any

# @dataclass
# class CovarianceResult:
#     """
#     Time series of covariance matrices.
#     """

#     covariances: np.ndarray
#     dates: pd.Index
#     assets: pd.Index

#     model: str | None = None
#     metadata: dict[str, Any] = field(default_factory=dict) 
#     # default_factory=dict makes a fresh dictionary for every object 
#     # to prevent that multiple instances could accidentally share the same dictionary.


# @dataclass
# class VarianceResult:
#     """
#     Time series of variances.
#     """

#     variances: float
#     dates: pd.Index
#     assets: pd.Index

#     model: str | None = None
#     metadata: dict[str, Any] = field(default_factory=dict)


# def ewma_core(
#     X: np.ndarray,
#     sigma0: np.ndarray,
#     decay: float,
# ) -> np.ndarray:

#     T, N = X.shape
#     # Doppelt 

#     covariances = np.empty((T, N, N))
#     covariances[0] = sigma0

#     for t in range(1, T):
#         x_prev = X[t - 1]

#         covariances[t] = (
#             decay * covariances[t - 1]
#             + (1 - decay) * np.outer(x_prev, x_prev)
#         )

#     return covariances


# def univariate_ewma_variance_estimator(
#         risk_factor: pd.Series,
#         initial_var=None,
#         decay=0.96,
#         initialization_window=60,
#         burn_in_window=30
# ) -> VarianceResult:

#     T = len(risk_factor)

#     X = risk_factor.to_numpy().reshape(-1, 1)

#     # Initial variance
#     if initial_var is None:

#         variance0 = risk_factor.iloc[:initialization_window].var()

#         start = initialization_window

#     else:

#         variance0 = float(initial_var)

#         start = 0

#     # Core expects an N x N initial covariance matrix.
#     # For N = 1 this is just a 1 x 1 matrix.
#     sigma0 = np.array([[variance0]])

#     core_result = ewma_core(
#         X[start:],
#         sigma0,
#         decay
#     )

#     variances = np.full(T, np.nan)

#     variances[start:] = core_result[:, 0, 0]

#     # Burn-in
#     variances[
#         start:start + burn_in_window
#     ] = np.nan

#     return VarianceResult(
#         variances=pd.Series(
#             variances,
#             index=risk_factor.index,
#             name=risk_factor.columns,
#         ),
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

# def multivariate_ewma_covariance_estimator(
#         risk_factor,
#         initial_covar=None,
#         decay=0.96,
#         initialization_window=60,
#         burn_in_window=30):

#     T, N = risk_factor.shape
#     X = risk_factor.to_numpy()

#     # Initial covariance
#     if initial_covar is None:

#         sigma0 = (
#             risk_factor
#             .iloc[:initialization_window]
#             .cov()
#             .to_numpy()
#         )

#         # sigma0 is the covariance forecast for the first
#         # observation AFTER the initialization sample
#         start = initialization_window

#     else:

#         sigma0 = np.asarray(initial_covar, dtype=float)

#         # supplied sigma0 is assumed to contain information
#         # available before X[0]
#         start = 0

#     # --------------------------------------------------
#     # Run recursion only on observations after sigma0
#     # --------------------------------------------------
#     core_result = ewma_core(
#         X[start:],
#         sigma0,
#         decay
#     )

#     # Full-size result aligned with original dates
#     covariances = np.full(
#         (T, N, N),
#         np.nan
#     )

#     covariances[start:] = core_result

#     # Burn-in
#     covariances[
#         start:start + burn_in_window
#     ] = np.nan

#     return CovarianceResult(
#         covariances=covariances,
#         dates=risk_factor.index,
#         assets=risk_factor.columns,
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


# if __name__ == "__main__":

#     port = portfolio.Portfolio(
#             holdings={
#                 "AAPL": 2,
#                 # "MSFT": 3,
#             }
#         )
    
#     data = market_data.MarketData.from_yfinance(port.assets, "2000-01-01", "2000-06-01") #"2009-12-31")

#     # print(type(data))
#     # print(type(data.prices))

#     # print(port.holdings)
#     # print(port.value(data))
#     # print(port.weights(data))
#     # print(port.loss(data))  

#     my_risk_factors = port.risk_factors(data)


#     print(univariate_ewma_variance_estimator(my_risk_factors))


