
import pandas as pd


def violations(
    portfolio_returns: pd.Series,
    var: pd.Series,
) -> pd.Series:
    """
    Identify VaR violations.

    Parameters
    ----------
    portfolio_returns : pandas.Series
        Realized portfolio returns.

    var : pandas.Series
        Positive VaR loss thresholds.

    Returns
    -------
    pandas.Series
        Boolean Series where True indicates a VaR violation.
    """

    realized_loss = -portfolio_returns

    realized_loss, var = realized_loss.align(
        var,
        join="inner",
    )

    valid = realized_loss.notna() & var.notna()

    violations = (
        realized_loss[valid]
        > var[valid]
    )

    return violations.rename("violation")


def violation_ratio(
    violations: pd.Series,
    alpha: float,
) -> float:
    return violations.mean() / ( alpha )