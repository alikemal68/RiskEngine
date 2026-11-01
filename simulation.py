"""Quantiles and lower-tail means of standardized innovation distributions.

Constant parameters return a Series indexed by tail probability.
Rolling parameters return a DataFrame indexed by forecast target date, with
numeric tail probabilities as columns.
"""

import numpy as np
import pandas as pd
from arch.univariate import Normal, StudentsT, SkewStudent


def _get_distribution(distribution: str):
    """Construct an arch distribution, accepting common capitalization."""
    names = {
        "normal": Normal,
        "studentst": StudentsT,
        "skewstudent": SkewStudent,
    }

    try:
        return names[distribution.lower()]()
    except (KeyError, AttributeError) as exc:
        raise ValueError(
            "distribution must be 'Normal', 'StudentsT' or 'SkewStudent'."
        ) from exc


def _get_distribution_and_params(
    distribution: str,
    dist_params: dict[str, float] | None = None,
):
    """Return an arch distribution and its ordered shape-parameter array."""
    dist = _get_distribution(distribution)
    names = dist.parameter_names()

    if not names:
        return dist, None

    if dist_params is None:
        raise ValueError(f"{distribution} requires parameters: {names}.")

    try:
        params = np.array([dist_params[name] for name in names], dtype=float)
    except KeyError as exc:
        raise ValueError(f"Missing distribution parameter: {exc.args[0]}.") from exc

    if not np.isfinite(params).all():
        raise ValueError("Distribution parameters must be finite.")

    return dist, params


def _check_levels(levels: list[float]) -> np.ndarray:
    """Validate tail probabilities without changing their order."""
    alpha = np.asarray(levels, dtype=float)
    if (
        alpha.ndim != 1
        or alpha.size == 0
        or not np.isfinite(alpha).all()
        or np.any((alpha <= 0) | (alpha >= 1))
        or len(np.unique(alpha)) != len(alpha)
    ):
        raise ValueError("levels must contain distinct probabilities in (0, 1).")
    return alpha


def _rolling_parameters(distribution: str, dist_params: pd.DataFrame):
    """Extract parameter values in the order required by arch."""
    dist = _get_distribution(distribution)
    names = dist.parameter_names()

    if not names:
        raise ValueError(
            "Normal has no fitted shape parameters; use parametric_quantiles "
            "or parametric_tail_means instead."
        )
    if not isinstance(dist_params, pd.DataFrame) or dist_params.empty:
        raise ValueError("dist_params must be a nonempty DataFrame.")
    if not dist_params.index.is_unique or not dist_params.index.is_monotonic_increasing:
        raise ValueError("Distribution-parameter dates must be unique and increasing.")
    if not dist_params.columns.is_unique:
        raise ValueError("Distribution-parameter names must be unique.")

    missing = [name for name in names if name not in dist_params.columns]
    if missing:
        raise ValueError(f"Missing distribution parameters: {missing}.")

    values = dist_params.loc[:, names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Distribution parameters must be finite.")

    return dist, values


def parametric_quantiles(
    distribution: str,
    levels: list[float],
    dist_params: dict[str, float] | None = None,
) -> pd.Series:
    """Quantiles of a standardized innovation distribution (fixed parameters)."""
    alpha = _check_levels(levels)
    dist, params = _get_distribution_and_params(distribution, dist_params)
    q = dist.ppf(alpha, parameters=params)
    return pd.Series(q, index=alpha, name="quantile")


def rolling_parametric_quantiles(
    distribution: str,
    levels: list[float],
    dist_params: pd.DataFrame,
) -> pd.DataFrame:
    """Date-dependent quantiles using fitted parameters from each rolling window.

    Input rows and output rows correspond to the same forecast *target* dates.
    """
    alpha = _check_levels(levels)
    dist, values = _rolling_parameters(distribution, dist_params)

    quantiles = np.array(
        [dist.ppf(alpha, parameters=row) for row in values],
        dtype=float,
    )

    return pd.DataFrame(
        quantiles,
        index=dist_params.index,
        columns=alpha,
    )


def parametric_tail_means(
    distribution: str,
    levels: list[float],
    dist_params: dict[str, float] | None = None,
) -> pd.Series:
    """Lower-tail means E[Z | Z <= q_alpha] (fixed parameters)."""
    alpha = _check_levels(levels)
    dist, params = _get_distribution_and_params(distribution, dist_params)
    q = dist.ppf(alpha, parameters=params)

    tail_means = [
        dist.partial_moment(1, float(z), parameters=params) / a
        for a, z in zip(alpha, q)
    ]

    return pd.Series(tail_means, index=alpha, name="tail_mean")


def rolling_parametric_tail_means(
    distribution: str,
    levels: list[float],
    dist_params: pd.DataFrame,
) -> pd.DataFrame:
    """Date-dependent lower-tail means using rolling fitted parameters."""
    alpha = _check_levels(levels)
    dist, values = _rolling_parameters(distribution, dist_params)

    tail_means = np.array([
        [
            dist.partial_moment(1, float(z), parameters=row) / a
            for a, z in zip(alpha, dist.ppf(alpha, parameters=row))
        ]
        for row in values
    ])

    return pd.DataFrame(
        tail_means,
        index=dist_params.index,
        columns=alpha,
    )



# # from dataclasses import dataclass, field
# # from typing import Any

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # import arch 
# from arch.univariate.base import ARCHModelResult
# from arch.univariate import Normal, StudentsT, SkewStudent



# def _get_distribution_and_params(
#     distribution: str,
#     dist_params: dict[str, float] | None = None,
# ):
#     if distribution == "Normal":
#         dist = Normal()

#     elif distribution == "StudentsT":
#         dist = StudentsT()

#     elif distribution == "SkewStudent":
#         dist = SkewStudent()

#     else:
#         raise ValueError(
#             "Unsupported distribution. "
#             "Use Normal, StudentsT or SkewStudent."
#         )

#     param_names = dist.parameter_names()

#     if param_names:
#         params = np.array([
#             dist_params[name]
#             for name in param_names
#         ])
#     else:
#         params = None

# #     if param_names and dist_params is None:
# #         raise ValueError(
# #             f"{distribution} requires parameters: {param_names}"
# # )


#     return dist, params



# def parametric_quantiles(
#     distribution: str,
#     levels: list[float],
#     dist_params: dict[str, float] | None = None,
# ) -> pd.Series:

#     dist, params = _get_distribution_and_params(distribution,dist_params)

#     q = dist.ppf(levels, params)

#     return pd.Series(
#         q,
#         index=levels,
#         name="quantile",
#     )


# # def filtered_historic_quantiles(
# #     fit_result: ARCHModelResult,
# #     levels: list[float],
# # ) -> pd.Series:

# #     std_residuals = fit_result.std_resid.dropna()

# #     q = std_residuals.quantile(levels)
# #     q.name = "quantile"

# #     return q




# ###################################################################################
# # def std_residuals(
# #     non_std_residuals: pd.Series,
# #     cond_var: pd.Series)-> pd.Series:
  
# #     return (non_std_residuals/cond_var).dropna()

# # def filtered_historic_quantiles(
    
# #     levels: list[float],
# # ) -> pd.Series:

    

# #     q = std_residuals.quantile(levels)
# #     q.name = "quantile"

# #     return q
# ###################################################################################


# def parametric_tail_means(
#     distribution: str,
#     levels: list[float],
#     dist_params: dict[str, float] | None = None,
# ) -> pd.Series:

#     dist, params = _get_distribution_and_params(
#         distribution,
#         dist_params,
#     )

#     tail_means = []

#     for alpha in levels:
#         q = dist.ppf(alpha, params)

#         partial_moment = dist.partial_moment(   
#                 1, # Order of partial moment 
#                 q, # Upper bound for partial moment integral
#             params,
#         )

#         tail_means.append(partial_moment / alpha)

#     return pd.Series(
#         tail_means,
#         index=levels,
#         name="tail_mean",
#     )
