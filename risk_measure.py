import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# portfolio_vol = np.sqrt(weights @ sigma_tp1 @ weights)

# VaR_var_covar = stats.norm.ppf(p) * port_value * portfolio_vol

def parametric_value_at_risk(
    # cond_mean: pd.Series | None,
    cond_var: pd.Series,
    quantiles: pd.Series,
) -> pd.DataFrame:

    # if not cond_mean.index.equals(cond_var.index):
    #     raise ValueError(
    #         "Conditional mean and variance must have the same index."
        # )

    var = (
        # -cond_mean.to_numpy()[:, None]
        - np.sqrt(cond_var.to_numpy())[:, None]
        * quantiles.to_numpy()[None, :]
    )

    columns = [
        f"{100 * alpha:g}%"
        for alpha in quantiles.index
    ]

    return pd.DataFrame(
        var,
        index=cond_var.index,
        columns=columns,
    )


def parametric_expected_shortfall(
    # cond_mean: pd.Series | None,
    cond_var: pd.Series,
    quantiles: pd.Series,
) -> pd.DataFrame:

    # if not cond_mean.index.equals(cond_var.index):
    #     raise ValueError(
    #         "Conditional mean and variance must have the same index."
        # )

    es = (
        # - np.sqrt(cond_var.to_numpy())[:, None]
        # * quantiles.to_numpy()[None, :]
    )

    columns = [
        f"{100 * alpha:g}%"
        for alpha in quantiles.index
    ]

    return pd.DataFrame(
        es,
        index=cond_var.index,
        columns=columns,
    )

if __name__ == "__main__":

    import portfolio
    import market_data
    import models
    import simulation

    port = portfolio.Portfolio(
        holdings={
            # "AAPL": 2,
            "MSFT": 3
        }
    )

    data = market_data.MarketData.from_yfinance(
        port.assets,
        "2000-01-01",
        "2020-12-31",
    )

    my_risk_factors = port.risk_factors(data)

    ewma_result = models.univariate_ewma_variance_estimator(
        my_risk_factors
    )
    res=models.fit_garch(my_risk_factors)
    cond_var=models.garch_conditional_variance(res, my_risk_factors)
    print(cond_var.variances)


    # print(models.garch_forecast(res))

    q=simulation.parametric_quantiles(res,[0.01,0.05])
    print(q)

    VaR=parametric_value_at_risk(cond_var.variances,q)
    my_risk_factors.plot()
    VaR.plot()
    plt.show()



