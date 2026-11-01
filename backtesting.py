"""One-day Value-at-Risk (VaR) coverage backtests.

All VaR columns are numeric lower-tail probabilities (for example, 0.05,
0.01). VaR and realized portfolio losses must have the same units and be
indexed by the forecast target dates.

The p-values below use asymptotic chi-squared reference distributions.
For very few VaR violations, these approximations can be unreliable.
"""

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2


def _validate_violations(indicators: pd.DataFrame) -> np.ndarray:
    """Validate a non-missing, chronologically ordered Boolean panel."""
    if not isinstance(indicators, pd.DataFrame) or indicators.empty:
        raise ValueError("violations must be a nonempty DataFrame.")
    if not indicators.index.is_unique or not indicators.index.is_monotonic_increasing:
        raise ValueError("Violation dates must be unique and increasing.")
    if not indicators.columns.is_unique or not pd.api.types.is_numeric_dtype(
        indicators.columns.dtype
    ):
        raise ValueError("Columns must be unique numeric tail probabilities.")
    if indicators.isna().any().any() or not all(
        pd.api.types.is_bool_dtype(dtype) for dtype in indicators.dtypes
    ):
        raise ValueError("Violations must contain only non-missing Booleans.")

    alpha = indicators.columns.to_numpy(dtype=float)
    if not np.isfinite(alpha).all() or ((alpha <= 0) | (alpha >= 1)).any():
        raise ValueError("Tail probabilities must lie strictly between 0 and 1.")
    return alpha


def violations(realized_losses: pd.Series, var: pd.DataFrame) -> pd.DataFrame:
    """Return True when realized monetary loss strictly exceeds monetary VaR.

    Only dates with VaR forecasts are evaluated; missing realized losses
    for any of these dates are an error, not nonviolations.
    """
    if not isinstance(realized_losses, pd.Series):
        raise TypeError("realized_losses must be a pandas Series.")
    if not isinstance(var, pd.DataFrame) or var.empty:
        raise ValueError("var must be a nonempty pandas DataFrame.")
    if not realized_losses.index.is_unique or not var.index.is_unique:
        raise ValueError("Dates must be unique.")
    if not var.index.is_monotonic_increasing:
        raise ValueError("VaR dates must be increasing.")
    if not var.columns.is_unique:
        raise ValueError("VaR levels must be unique.")

    losses = realized_losses.reindex(var.index)
    if (
        not np.isfinite(losses.to_numpy(dtype=float)).all()
        or not np.isfinite(var.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Realized losses and VaR must be finite on forecast dates.")

    result = var.lt(losses, axis="index")
    _validate_violations(result)
    return result


def violation_ratio(indicators: pd.DataFrame) -> pd.Series:
    """Observed violation frequency divided by nominal tail probability."""
    alpha = _validate_violations(indicators)
    return pd.Series(
        indicators.mean(axis=0).to_numpy() / alpha,
        index=indicators.columns,
        name="violation_ratio",
    )


def kupiec_test(indicators: pd.DataFrame) -> pd.DataFrame:
    """Kupiec likelihood-ratio test of unconditional VaR coverage.

    H0: unconditional violation probability equals each column's alpha.
    LR_UC is asymptotically chi-squared with one degree of freedom.
    """
    alpha = _validate_violations(indicators)
    T = len(indicators)
    v1 = indicators.sum(axis=0).to_numpy(dtype=int)
    v0 = T - v1
    p_hat = v1 / T

    ll_null = xlogy(v1, alpha) + xlogy(v0, 1 - alpha)
    ll_alt = xlogy(v1, p_hat) + xlogy(v0, 1 - p_hat)
    lr = np.maximum(2 * (ll_alt - ll_null), 0.0)

    return pd.DataFrame(
        {
            "observations": T,
            "violations": v1,
            "violation_rate": p_hat,
            "LR_UC": lr,
            "p_value": chi2.sf(lr, df=1),
        },
        index=pd.Index(alpha, name="alpha"),
    )


def christoffersen_independence_test(indicators: pd.DataFrame) -> pd.DataFrame:
    """Christoffersen test of first-order independence of VaR violations.

    n_ij counts transitions from indicator i at time t-1 to j at time t.
    H0: P(I_t=1 | I_{t-1}=0) = P(I_t=1 | I_{t-1}=1).

    Returns NaN for LR_IND and p_value when either previous-state group
    never occurs, because one of the transition probabilities is undefined.
    Otherwise p-values use the asymptotic chi-squared(1) approximation.
    """
    alpha = _validate_violations(indicators)
    if len(indicators) < 2:
        raise ValueError("At least two observations are needed for transitions.")

    V = indicators.to_numpy(dtype=bool)
    previous, current = V[:-1], V[1:]

    n00 = np.sum(~previous & ~current, axis=0)
    n01 = np.sum(~previous & current, axis=0)
    n10 = np.sum(previous & ~current, axis=0)
    n11 = np.sum(previous & current, axis=0)

    from_zero = n00 + n01
    from_one = n10 + n11
    n_transitions = len(indicators) - 1

    p01 = np.divide(
        n01, from_zero,
        out=np.full(len(alpha), np.nan), where=from_zero > 0,
    )
    p11 = np.divide(
        n11, from_one,
        out=np.full(len(alpha), np.nan), where=from_one > 0,
    )

    # Under H0, both transition rows share one common probability.
    p = (n01 + n11) / n_transitions
    estimable = (from_zero > 0) & (from_one > 0)
    lr = np.full(len(alpha), np.nan)

    if estimable.any():
        valid = estimable
        ll_null = (
            xlogy(n00[valid] + n10[valid], 1 - p[valid])
            + xlogy(n01[valid] + n11[valid], p[valid])
        )
        ll_alt = (
            xlogy(n00[valid], 1 - p01[valid])
            + xlogy(n01[valid], p01[valid])
            + xlogy(n10[valid], 1 - p11[valid])
            + xlogy(n11[valid], p11[valid])
        )
        lr[valid] = np.maximum(2 * (ll_alt - ll_null), 0.0)

    return pd.DataFrame(
        {
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
            "p01": p01,
            "p11": p11,
            "LR_IND": lr,
            "p_value": chi2.sf(lr, df=1),
        },
        index=pd.Index(alpha, name="alpha"),
    )


def christoffersen_conditional_coverage_test(indicators: pd.DataFrame) -> pd.DataFrame:
    """Joint test of correct unconditional coverage and independence.

    LR_CC = LR_UC + LR_IND; asymptotic chi-squared(2) p-value.
    """
    uc = kupiec_test(indicators)
    ind = christoffersen_independence_test(indicators)
    lr_cc = uc["LR_UC"] + ind["LR_IND"]

    return pd.DataFrame(
        {"LR_CC": lr_cc, "p_value": chi2.sf(lr_cc, df=2)},
        index=uc.index,
    )


def var_backtest(indicators: pd.DataFrame) -> pd.DataFrame:
    """Combine violation counts, ratios, and all three VaR coverage tests.

    Row index: numerical alpha (e.g. 0.05, 0.01).
    p_UC, p_IND, p_CC are asymptotic p-values, NOT confidence levels.
    """
    uc = kupiec_test(indicators)
    ind = christoffersen_independence_test(indicators)

    # Joint test is a sum of the two likelihood-ratio statistics.
    lr_cc = uc["LR_UC"] + ind["LR_IND"]
    alpha = uc.index.to_numpy(dtype=float)

    return pd.DataFrame(
        {
            "observations": uc["observations"],
            "violations": uc["violations"],
            "expected_violations": uc["observations"] * alpha,
            "violation_ratio": uc["violation_rate"] / alpha,
            "LR_UC": uc["LR_UC"],
            "p_UC": uc["p_value"],
            "LR_IND": ind["LR_IND"],
            "p_IND": ind["p_value"],
            "LR_CC": lr_cc,
            "p_CC": chi2.sf(lr_cc, df=2),
        },
        index=uc.index,
    )


# import numpy as np
# import pandas as pd

# from scipy.special import xlogy
# from scipy.stats import chi2


# def kupiec_test(
#     violations: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Kupiec unconditional coverage test for VaR violations.

#     Parameters
#     ----------
#     violations : pd.DataFrame
#         Boolean violation indicators.
#         Columns are numeric tail probabilities (e.g. 0.05, 0.01).

#     Returns
#     -------
#     pd.DataFrame
#         Number of observations, violations, observed violation
#         frequency, likelihood-ratio statistic and p-value
#         for each tail probability.
#     """

#     if not isinstance(violations, pd.DataFrame) or violations.empty:
#         raise ValueError("violations must be a nonempty DataFrame.")

#     if not violations.columns.is_unique:
#         raise ValueError("Tail probabilities must be unique.")

#     if (
#         violations.isna().any().any()
#         or not violations.dtypes.map(pd.api.types.is_bool_dtype).all()
#     ):
#         raise ValueError(
#             "violations must contain only non-missing Boolean values."
#         )

#     alpha = violations.columns.to_numpy(dtype=float)

#     if (
#         not np.isfinite(alpha).all()
#         or ((alpha <= 0) | (alpha >= 1)).any()
#     ):
#         raise ValueError("Tail probabilities must be between 0 and 1.")

#     T = len(violations)

#     v1 = violations.sum(axis=0).to_numpy(dtype=int)
#     v0 = T - v1

#     p_hat = v1 / T

#     # Log-likelihood under H0: p = alpha
#     log_likelihood_null = (
#         xlogy(v1, alpha)
#         + xlogy(v0, 1 - alpha)
#     )

#     # Log-likelihood under H1: p = p_hat
#     log_likelihood_alt = (
#         xlogy(v1, p_hat)
#         + xlogy(v0, 1 - p_hat)
#     )

#     lr_uc = 2 * (
#         log_likelihood_alt - log_likelihood_null
#     )

#     # Protect against tiny negative values from floating-point error
#     lr_uc = np.maximum(lr_uc, 0.0)

#     p_values = chi2.sf(lr_uc, df=1)

#     return pd.DataFrame(
#         {
#             "observations": T,
#             "violations": v1,
#             "violation_rate": p_hat,
#             "LR_UC": lr_uc,
#             "p_value": p_values,
#         },
#         index=pd.Index(alpha, name="alpha"),
#     )



# def violations(
#     realized_losses: pd.Series,
#     var: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Identify VaR violations using realized monetary losses.

#     Both realized losses and VaR must use the same currency.
#     VaR forecasts are indexed by their target dates.

#     A violation occurs when the realized loss exceeds VaR.
#     """

#     # Each forecast date must be uniquely identifiable
#     if not var.index.is_unique or not realized_losses.index.is_unique:
#         raise ValueError("Date indices must be unique.")

#     # Align realized losses to the VaR forecast dates
#     realized_losses = realized_losses.reindex(var.index)

#     # Missing observations must not become nonviolations
#     if (
#         realized_losses.isna().any()
#         or var.isna().any().any()
#     ):
#         raise ValueError("Missing VaR forecasts or realized losses.")

#     violations = var.lt(
#         realized_losses,
#         axis="index",
#     )

#     return violations


# def violation_ratio(
#     violations: pd.DataFrame,
# ) -> pd.Series:
#     """Observed violation frequency / expected violation frequency."""

#     if violations.empty:
#         raise ValueError("The violation DataFrame must not be empty.")

#     alpha = pd.Series(
#         violations.columns.to_numpy(dtype=float),
#         index=violations.columns,
#     )


#     if ((alpha <= 0) | (alpha >= 1)).any():
#         raise ValueError("Tail probabilities must be between 0 and 1.")

#     return (violations.mean() / alpha).rename("violation_ratio")



# # def kupiec_unconditional_coverage_test:(

# # ):
# #     a = p^(sum(v))*(1  p)^(length(v)-sum(v));
# # b=(sum(v)/length(v))^(sum(v))*(1-(sum(v)/length(v)))^
# # (length(v)-sum(v));
# # res = 2*log(a/b);
