"""Parametric Value at Risk and Expected Shortfall for zero-mean models.

The input conditional variances may describe returns or monetary losses.
The resulting risk measures use the corresponding units. Quantiles and
lower-tail means are for *standardized* model innovations.

Risk levels are numeric tail probabilities (e.g. 0.01, not "1%").
"""

import numpy as np
import pandas as pd


def _parametric_risk_measure(
    cond_var: pd.Series,
    tail_values: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Scale constant or date-dependent standardized lower-tail statistics."""
    if not isinstance(cond_var, pd.Series) or cond_var.empty:
        raise ValueError("cond_var must be a nonempty pandas Series.")
    if not cond_var.index.is_unique:
        raise ValueError("Conditional variance dates must be unique.")

    variances = cond_var.to_numpy(dtype=float)
    if not np.isfinite(variances).all() or (variances < 0).any():
        raise ValueError("Conditional variances must be finite and nonnegative.")

    if isinstance(tail_values, pd.Series):
        # One set of quantiles / tail means shared by every forecast date.
        levels = tail_values.index
        values = tail_values.to_numpy(dtype=float)[None, :]

    elif isinstance(tail_values, pd.DataFrame):
        # Each forecast date has its own quantiles / tail means.
        if not tail_values.index.equals(cond_var.index):
            raise ValueError(
                "Tail-statistic dates must exactly match variance forecast dates."
            )
        levels = tail_values.columns
        values = tail_values.to_numpy(dtype=float)

    else:
        raise TypeError("tail_values must be a pandas Series or DataFrame.")

    if len(levels) == 0 or not levels.is_unique:
        raise ValueError("Tail probabilities must be nonempty and unique.")
    if not pd.api.types.is_numeric_dtype(levels.dtype):
        raise ValueError("Use numeric tail probabilities (0.05), not strings ('5%').")

    alpha = levels.to_numpy(dtype=float)
    if not np.isfinite(alpha).all() or ((alpha <= 0) | (alpha >= 1)).any():
        raise ValueError("Tail probabilities must be strictly between 0 and 1.")
    if not np.isfinite(values).all():
        raise ValueError("Quantiles or tail means must be finite.")

    # Series: (T, 1) * (1, A). DataFrame: (T, 1) * (T, A).
    risk_measure = -np.sqrt(variances)[:, None] * values

    return pd.DataFrame(
        risk_measure,
        index=cond_var.index,
        columns=levels,
    )


def parametric_value_at_risk(
    cond_var: pd.Series,
    quantiles: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Calculate VaR from variance and standardized innovation quantiles.

    ``quantiles`` can be a Series indexed by tail probability (constant
    distribution) or a DataFrame indexed by forecast date with numeric
    probability columns (rolling distribution parameters).

    Assumes zero conditional mean. VaR has the units of sqrt(cond_var).
    """
    return _parametric_risk_measure(cond_var, quantiles)


def expected_shortfall(
    cond_var: pd.Series,
    tail_means: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Calculate ES from variance and standardized lower-tail means.

    ``tail_means`` has the same accepted shapes as ``quantiles`` above.
    Assumes zero conditional mean. ES has the units of sqrt(cond_var).
    """
    return _parametric_risk_measure(cond_var, tail_means)



    # import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt


    # # portfolio_vol = np.sqrt(weights @ sigma_tp1 @ weights)

    # # VaR_var_covar = stats.norm.ppf(p) * port_value * portfolio_vol

    # def parametric_value_at_risk(
    #     # cond_mean: pd.Series | None,
    #     cond_var: pd.Series,
    #     quantiles: pd.Series,
    # ) -> pd.DataFrame:

    #     # if not cond_mean.index.equals(cond_var.index):
    #     #     raise ValueError(
    #     #         "Conditional mean and variance must have the same index."
    #         # )

    #     var = (
    #         # -cond_mean.to_numpy()[:, None]
    #         - np.sqrt(cond_var.to_numpy())[:, None]
    #         * quantiles.to_numpy()[None, :]
    #     )

    #     columns = [
    #         f"{100 * alpha:g}%"
    #         for alpha in quantiles.index
    #     ]

    #     return pd.DataFrame(
    #         var,
    #         index=cond_var.index,
    #         columns=columns,
    #     )


    # def expected_shortfall(
    #     # cond_mean: pd.Series | None,
    #     cond_var: pd.Series,
    #     tail_means: pd.Series,
    # ) -> pd.DataFrame:

    #     # if not cond_mean.index.equals(cond_var.index):
    #     #     raise ValueError(
    #     #         "Conditional mean and variance must have the same index."
    #         # )

    #     es = (
    #         #  -cond_mean.to_numpy()[:, None]
    #         - np.sqrt(cond_var.to_numpy())[:, None]
    #         * tail_means.to_numpy()[None, :]
    #     )

    #     columns = [
    #         f"{100 * alpha:g}%"
    #         for alpha in tail_means.index
    #     ]

    #     return pd.DataFrame(
    #         es,
    #         index=cond_var.index,
    #         columns=columns,
    #     )


    # # add a function that checks maybe if es ia always bigger than var???






    # # if __name__ == "__main__":

    # #     import portfolio
    # #     import market_data
    # #     import models
    # #     import simulation

    # #     port = portfolio.Portfolio(
    # #         holdings={
    # #             # "AAPL": 2,
    # #             "MSFT": 3
    # #         }
    # #     )

    # #     data = market_data.MarketData.from_yfinance(
    # #         port.assets,
    # #         "2000-01-01",
    # #         "2020-12-31",
    # #     )

    # #     my_risk_factors = port.risk_factors(data)

    # #     ewma_result = models.univariate_ewma_variance_estimator(
    # #         my_risk_factors
    # #     )
    # #     res=models.fit_garch(my_risk_factors)
    # #     cond_var=models.garch_conditional_variance(res, my_risk_factors)
    # #     print(cond_var.variances)


    # #     # print(models.garch_forecast(res))

    # #     q=simulation.parametric_quantiles(res,[0.01,0.05])
    # #     print(q)

    # #     VaR=parametric_value_at_risk(cond_var.variances,q)
    # #     my_risk_factors.plot()
    # #     VaR.plot()
    # #     plt.show()



