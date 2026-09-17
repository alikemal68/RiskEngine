
# from dataclasses import dataclass, field
# from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import arch 
from arch.univariate.base import ARCHModelResult



def parametric_quantiles(
    res: ARCHModelResult,
    levels: list[float],
) -> pd.Series:

    dist = res.model.distribution
    param_names = dist.parameter_names()

    if param_names:
        dist_params = res.params[param_names].to_numpy()
    else:
        dist_params = None

    q = dist.ppf(levels, dist_params)

    return pd.Series(
        q,
        index=levels,
        name="quantile",
    )


def filtered_historic_quantiles(
    res: ARCHModelResult,
    levels: list[float],
) -> pd.Series:

    std_residuals = res.std_resid.dropna()

    q = std_residuals.quantile(levels)
    q.name = "quantile"

    return q


def parametric_tail_means(
    res: ARCHModelResult,
    levels: list[float],
) -> pd.Series:

    dist = res.model.distribution
    param_names = dist.parameter_names()

    if param_names:
        dist_params = res.params[param_names].to_numpy()
    else:
        dist_params = None

    tail_means = []

    for alpha in levels:
        q = dist.ppf(alpha, dist_params)

        partial_moment = dist.partial_moment(
            1, # Order of partial moment
            q, # Upper bound for partial moment integral
            dist_params,
        )

        tail_means.append(partial_moment / alpha)

    return pd.Series(
        tail_means,
        index=levels,
        name="tail_mean",
    )
