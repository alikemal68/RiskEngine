import numpy as np, pandas as pd

def EWMA_covar_estimator(risk_factor, theta=0.04, burn_in_window=100):

    T,N = risk_factor.shape

    covar = np.full((T, N*(N+1)/2), np.nan)

    # Initial covariance estimate
    covar[burn_in_window] = risk_factor.iloc[:burn_in_window].cov()
#     Sigma_1 is usually set as the
# unconditional volatility of the data and some 30 days of data are used to update the
# volatility forecast before it is used
# !!!!!!!!!!!!!!!!!!!!.

    # EWMA recursion
    for t in range(burn_in_window + 1, T):
        x = risk_factor.iloc[t - 1]

        covar[t] = (
            theta * np.outer(x, x)
            + (1 - theta) * covar[t - 1]
        )

    return covar