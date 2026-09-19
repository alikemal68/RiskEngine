
# from dataclasses import dataclass, field
# from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import arch 
from arch.univariate.base import ARCHModelResult
from arch.univariate import Normal, StudentsT, SkewStudent



def _get_distribution_and_params(
    distribution: str,
    dist_params: dict[str, float] | None = None,
):
    if distribution == "Normal":
        dist = Normal()

    elif distribution == "StudentsT":
        dist = StudentsT()

    elif distribution == "SkewStudent":
        dist = SkewStudent()

    else:
        raise ValueError(
            "Unsupported distribution. "
            "Use Normal, StudentsT or SkewStudent."
        )

    param_names = dist.parameter_names()

    if param_names:
        params = np.array([
            dist_params[name]
            for name in param_names
        ])
    else:
        params = None

#     if param_names and dist_params is None:
#         raise ValueError(
#             f"{distribution} requires parameters: {param_names}"
# )


    return dist, params



def parametric_quantiles(
    distribution: str,
    levels: list[float],
    dist_params: dict[str, float] | None = None,
) -> pd.Series:

    dist, params = _get_distribution_and_params(distribution,dist_params)

    q = dist.ppf(levels, params)

    return pd.Series(
        q,
        index=levels,
        name="quantile",
    )


# def filtered_historic_quantiles(
#     fit_result: ARCHModelResult,
#     levels: list[float],
# ) -> pd.Series:

#     std_residuals = fit_result.std_resid.dropna()

#     q = std_residuals.quantile(levels)
#     q.name = "quantile"

#     return q




###################################################################################
# def std_residuals(
#     non_std_residuals: pd.Series,
#     cond_var: pd.Series)-> pd.Series:
  
#     return (non_std_residuals/cond_var).dropna()

# def filtered_historic_quantiles(
    
#     levels: list[float],
# ) -> pd.Series:

    

#     q = std_residuals.quantile(levels)
#     q.name = "quantile"

#     return q
###################################################################################


def parametric_tail_means(
    distribution: str,
    levels: list[float],
    dist_params: dict[str, float] | None = None,
) -> pd.Series:

    dist, params = _get_distribution_and_params(
        distribution,
        dist_params,
    )

    tail_means = []

    for alpha in levels:
        q = dist.ppf(alpha, params)

        partial_moment = dist.partial_moment(   
                1, # Order of partial moment 
                q, # Upper bound for partial moment integral
            params,
        )

        tail_means.append(partial_moment / alpha)

    return pd.Series(
        tail_means,
        index=levels,
        name="tail_mean",
    )
