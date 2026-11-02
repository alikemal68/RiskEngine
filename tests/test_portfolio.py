"""Unit tests for portfolio.py.

Run from the project root:
    python -m pytest -v test_portfolio.py

Only synthetic prices are used: no Yahoo Finance download, GARCH fitting,
or internet access is required.
"""

import numpy as np
import pandas as pd
import pytest

from co_variance_results import CovarianceResult
from portfolio import Portfolio
from risk_measure import parametric_value_at_risk


class SyntheticMarketData:
    """Small stand-in providing the two MarketData attributes Portfolio uses."""

    def __init__(self, prices: pd.DataFrame):
        self.prices = prices

    def log_returns(self) -> pd.DataFrame:
        return np.log(self.prices).diff().dropna()


@pytest.fixture
def example():
    """A fixed-holdings two-stock portfolio with known price changes."""
    dates = pd.bdate_range("2024-01-02", periods=8)
    prices = pd.DataFrame(
        {
            "AAPL": [100, 105, 101, 107, 110, 106, 115, 112],
            "MSFT": [50, 48, 53, 49, 54, 55, 51, 52],
        },
        index=dates,
        dtype=float,
    )
    return Portfolio({"AAPL": 2, "MSFT": 3}), SyntheticMarketData(prices)


@pytest.fixture
def covariance_result(example):
    """Target-dated covariance forecasts with intentionally reversed labels."""
    _, data = example
    dates = data.prices.index[3:]
    assets = pd.Index(["MSFT", "AAPL"])  # Opposite of portfolio order.
    base = np.array([[0.0009, 0.00008], [0.00008, 0.0004]])
    matrices = np.stack([base * (1 + 0.1 * i) for i in range(len(dates))])
    return CovarianceResult(
        covariances=matrices,
        dates=dates,
        assets=assets,
        model="synthetic",
    )


def test_position_values_and_total_value(example):
    portfolio, data = example

    positions = portfolio.position_values(data)
    expected = data.prices.mul(pd.Series({"AAPL": 2, "MSFT": 3}))
    pd.testing.assert_frame_equal(positions, expected)

    total = portfolio.value(data)
    pd.testing.assert_series_equal(
        total,
        expected.sum(axis=1).rename("portfolio_value"),
    )
    assert total.iloc[0] == 350  # 2 * 100 + 3 * 50


def test_weights_sum_to_one_and_match_positions(example):
    portfolio, data = example
    weights = portfolio.weights(data)

    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0)
    np.testing.assert_allclose(
        weights.to_numpy(),
        portfolio.position_values(data).div(portfolio.value(data), axis=0).to_numpy(),
    )
    assert weights.loc[data.prices.index[0], "AAPL"] == pytest.approx(200 / 350)


def test_realized_losses_equal_negative_change_in_value(example):
    """Realized P&L is exact for the fixed-holdings price convention."""
    portfolio, data = example

    actual = portfolio.loss(data)
    expected = -portfolio.value(data).diff().dropna()
    pd.testing.assert_series_equal(
        actual,
        expected.rename("portfolio_loss"),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    # On the first return day, the portfolio gains 2*5 + 3*(-2) = 4.
    assert actual.iloc[0] == pytest.approx(-4.0)
    assert (actual > 0).any()  # The example also contains loss days.


def test_weighted_log_returns_use_previous_day_weights(example):
    portfolio, data = example
    returns = portfolio.risk_factors(data)
    previous_weights = portfolio.weights(data).shift(1).reindex(returns.index)
    expected = (previous_weights * returns).sum(axis=1)

    pd.testing.assert_series_equal(
        portfolio.portfolio_weighted_log_return(data),
        expected.rename("portfolio_weighted_log_return"),
    )


def test_target_date_variances_use_previous_exposures(example, covariance_result):
    """The covariance for target date t must use exposures from t-1."""
    portfolio, data = example
    cov = covariance_result

    return_variance = portfolio.variance_from_weights(data, cov)
    loss_variance = portfolio.variance_from_position_values(data, cov)

    assert return_variance.index.equals(cov.dates)
    assert loss_variance.index.equals(cov.dates)
    assert cov.dates.isin(portfolio.loss(data).index).all()

    for i, target in enumerate(cov.dates):
        prior = data.prices.index[data.prices.index.get_loc(target) - 1]
        weights = portfolio.weights(data).loc[prior, cov.assets].to_numpy()
        positions = portfolio.position_values(data).loc[prior, cov.assets].to_numpy()
        matrix = cov.covariances[i]

        assert return_variance.loc[target] == pytest.approx(weights @ matrix @ weights)
        assert loss_variance.loc[target] == pytest.approx(positions @ matrix @ positions)

    # A check that would fail if the implementation accidentally used t weights.
    first_target = cov.dates[0]
    future_weights = portfolio.weights(data).loc[first_target, cov.assets].to_numpy()
    wrong_variance = future_weights @ cov.covariances[0] @ future_weights
    assert not np.isclose(
        return_variance.loc[first_target], wrong_variance, rtol=1e-7, atol=1e-12
    )


def test_monetary_variance_and_var_scale_with_previous_portfolio_value(
    example, covariance_result
):
    """a' Sigma a == V[t-1]**2 * w' Sigma w, including resulting VaR."""
    portfolio, data = example
    return_variance = portfolio.variance_from_weights(data, covariance_result)
    loss_variance = portfolio.variance_from_position_values(data, covariance_result)
    previous_value = portfolio.value(data).shift(1).reindex(covariance_result.dates)

    np.testing.assert_allclose(
        loss_variance.to_numpy(),
        (previous_value.pow(2) * return_variance).to_numpy(),
        rtol=1e-12,
    )

    quantiles = pd.Series([-1.644853627, -2.326347874], index=[0.05, 0.01])
    return_var = parametric_value_at_risk(return_variance, quantiles)
    monetary_var = parametric_value_at_risk(loss_variance, quantiles)
    pd.testing.assert_index_equal(monetary_var.columns, pd.Index([0.05, 0.01]))
    np.testing.assert_allclose(
        monetary_var.to_numpy(),
        return_var.mul(previous_value, axis=0).to_numpy(),
        rtol=1e-12,
    )


def test_single_forecast_uses_origin_exposures_and_matches_history(example):
    """A standalone next-day forecast uses exposures at origin, not target."""
    portfolio, full_data = example
    target_date = full_data.prices.index[-1]
    origin_data = SyntheticMarketData(full_data.prices.iloc[:-1])
    origin_date = origin_data.prices.index[-1]

    asset_order = pd.Index(["MSFT", "AAPL"])
    matrix = pd.DataFrame(
        [[0.0009, 0.00008], [0.00008, 0.0004]],
        index=asset_order,
        columns=asset_order,
    )
    result = CovarianceResult(
        covariances=matrix.to_numpy()[None, :, :],
        dates=pd.Index([target_date]),
        assets=asset_order,
    )

    for stand_alone, historical in [
        (portfolio.forecast_variance_from_weights, portfolio.variance_from_weights),
        (
            portfolio.forecast_variance_from_position_values,
            portfolio.variance_from_position_values,
        ),
    ]:
        issued = stand_alone(origin_data, matrix)
        collected = historical(full_data, result)

        assert issued.index[0] == origin_date
        assert collected.index[0] == target_date
        assert issued.iloc[0] == pytest.approx(collected.iloc[0])


def test_univariate_covariance_result_works_without_special_cases(example):
    """A one-stock covariance forecast still uses shape (T, 1, 1)."""
    _, data = example
    portfolio = Portfolio({"AAPL": 2})
    dates = data.prices.index[2:]
    result = CovarianceResult(
        covariances=np.full((len(dates), 1, 1), 0.0004),
        dates=dates,
        assets=pd.Index(["AAPL"]),
    )

    result_variance = portfolio.variance_from_position_values(data, result)
    prev_position = portfolio.position_values(data)["AAPL"].shift(1).reindex(dates)
    np.testing.assert_allclose(result_variance.to_numpy(), 0.0004 * prev_position**2)


@pytest.mark.parametrize("holdings", [{}, {"AAPL": -2}, {"AAPL": 0}, {"AAPL": np.nan}])
def test_invalid_holdings_are_rejected(holdings):
    with pytest.raises((ValueError, TypeError)):
        Portfolio(holdings)


def test_missing_portfolio_asset_prices_raise_an_error(example):
    portfolio, data = example
    data_without_msft = SyntheticMarketData(data.prices[["AAPL"]])
    with pytest.raises(ValueError, match="Missing prices"):
        portfolio.position_values(data_without_msft)


def test_covariance_asset_mismatch_is_rejected(example):
    portfolio, data = example
    dates = data.prices.index[2:]
    bad_result = CovarianceResult(
        covariances=np.full((len(dates), 1, 1), 0.0004),
        dates=dates,
        assets=pd.Index(["AAPL"]),
    )
    with pytest.raises(ValueError, match="Covariance assets must match"):
        portfolio.variance_from_position_values(data, bad_result)
