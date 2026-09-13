import numpy as np, pandas as pd

import portfolio
import market_data

from dataclasses import dataclass


@dataclass
class CovarianceResult:
    covariances: np.ndarray
    dates: pd.Index
    assets: pd.Index



def EWMA_covar_estimator(
        risk_factor,
        initial_covar=None, 
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
        sigma0 = np.asarray(initial_covar, dtype=float)
        start = 0 
     

    # Initial covariancesiance estimate

    covariances[start] = sigma0 #risk_factor.iloc[:start].cov()

    # EWMA recursion

    X = risk_factor.to_numpy()

    for t in range(start + 1, T):
        x = X.iloc[t - 1]

        covariances[t] = (
            (1-decay) * np.outer(x, x) + decay * covariances[t - 1]
        )

    covariances[start:start+burn_in_window] = np.nan

    return covariances


if __name__ == "__main__":

    port = portfolio.Portfolio(
            holdings={
                "AAPL": 2,
                "MSFT": 3,
            }
        )
    
    data = market_data.MarketData.from_yfinance(port.assets, "2000-01-01","2009-12-31")

    # print(type(data))
    # print(type(data.prices))

    # print(port.holdings)
    # print(port.value(data))
    # print(port.weights(data))
    # print(port.loss(data))  

    X = port.risk_factors(data)

    print(EWMA_covar_estimator(X))


