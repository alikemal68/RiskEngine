import numpy as np, pandas as pd


def EWMA_covar_estimator(
        risk_factor,
        initial_covar, 
        decay=0.96, 
        initialization_window=60, 
        burn_in_window=30):

    T,N = risk_factor.shape

    covariances = np.full((T, N, N), np.nan)

    if initial_covar is None:
        sigma0 = (
                risk_factor
                .iloc[:initialization_window]
                .cov()
                .to_numpy()
            )

        # Sigma_0 is the forecast for the first observation AFTER the initialization sample.
        start = initialization_window   

    else:
        sigma0 = initial_covar
        start = 0 
     

    # Initial covariancesiance estimate

    covariances[start] = sigma0 #risk_factor.iloc[:start].cov()

    # EWMA recursion
    for t in range(start + 1, T):
        x = risk_factor.iloc[t - 1]

        covariances[t] = (
            (1-decay) * np.outer(x, x) + decay * covariances[t - 1]
        )

    covariances[start:start+burn_in_window] = np.nan

    return covariances