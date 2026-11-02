# Quantitative Market Risk Engine

**Python · Portfolio risk · EWMA / GARCH · Value at Risk & Expected Shortfall · Out-of-sample backtesting**

A modular market-risk project that forecasts **one-day portfolio Value at Risk (VaR)** and **Expected Shortfall (ES)** from conditional volatility models, then evaluates VaR forecasts against realized losses. The implementation supports **multivariate EWMA** and **univariate GARCH-family** models with Normal, Student-t and skewed Student-t innovations.

Built to explore the practical link between time-series volatility estimation, portfolio exposures, tail-risk measurement and statistical VaR backtesting.

## What it does

| Component | Implemented |
|---|---|
| Market data | Yahoo Finance adjusted-price downloads; decimal log returns |
| Portfolio | Fixed share holdings, evolving weights, position values and realized price-based losses |
| Conditional risk models | Multivariate EWMA; univariate GARCH(1,1) and GJR-GARCH (`o=1`) |
| Innovation distributions | Normal, standardized Student-t and skewed Student-t (via `arch`) |
| Rolling forecasts | One-step-ahead out-of-sample covariances; date-indexed GARCH distribution parameters |
| Risk measures | Parametric portfolio VaR and ES, using monetary loss variance |
| VaR backtests | Violation ratio; Kupiec unconditional coverage; Christoffersen independence and conditional coverage |

**Not implemented:** historical simulation, filtered historical simulation, Monte Carlo, options pricing, multivariate GARCH and ES backtesting. These are potential extensions—not features of this release.

## Notebooks

| Notebook | Demonstrates |
|---|---|
| [`01_multivariate_ewma.ipynb`](notebooks/01_multivariate_ewma.ipynb) | MSFT + AAPL portfolio; conditional covariance/correlation; 95%/99% one-day **VaR and ES**; comparison with realized losses |
| [`02_univariate_var_backtesting.ipynb`](notebooks/02_univariate_var_backtesting.ipynb) | S&P 500 index-unit risk proxy; **EWMA vs GARCH** with Normal, Student-t, skewed-t and asymmetric GJR-skewed-t; shared rolling evaluation, violation ratios and three coverage tests |

The backtesting notebook defines the model configurations **once** in `MODEL_SPECS` and evaluates them with a common function. This keeps dates, window lengths and statistical tests consistent across model comparisons.

## Getting started

**Requirements:** Python 3.10+ and internet access to retrieve Yahoo Finance prices.

```bash
# From the project root (after cloning or downloading the files)
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
jupyter lab
```

Open either notebook under `notebooks/` and run cells from top to bottom. The notebooks work when Jupyter's working directory is the project root or `notebooks/`.

**Runtime:** the multivariate EWMA example is relatively lightweight. The univariate backtest **refits GARCH for every forecast day and model**, so its full 2000–2026 historical run can take a substantial amount of time. For a quick exploratory run, set, for example, `START_DATE = "2019-01-01"` and `WINDOW_LENGTH = 500` in the notebook settings cell; this changes the evaluation sample and therefore the statistical results.

For fast, deterministic checks of the portfolio calculations:

```bash
pytest -q test_portfolio.py
```

## Minimal example: multivariate EWMA VaR / ES

```python
from market_data import MarketData
from portfolio import Portfolio
import rolling_forecast
import simulation
import risk_measure

port = Portfolio({"MSFT": 3, "AAPL": 2})
data = MarketData.from_yfinance(port.assets, "2019-01-01", "2026-11-01")
returns = port.risk_factors(data)

forecasts = rolling_forecast.rolling_forecast(
    returns,
    window_length=750,
    forecaster=rolling_forecast.ewma_forecaster,
    decay=0.96,
)

# One-day monetary loss variance, using previous-day position values.
loss_variance = port.variance_from_position_values(data, forecasts)
levels = [0.05, 0.01]

var = risk_measure.parametric_value_at_risk(
    loss_variance,
    simulation.parametric_quantiles("Normal", levels),
)
es = risk_measure.expected_shortfall(
    loss_variance,
    simulation.parametric_tail_means("Normal", levels),
)

print(var.tail())
print(es.tail())
```

The 0.05 and 0.01 columns represent **5% and 1% tail probabilities**, i.e. **95% and 99% confidence VaR/ES**. All time-series forecasts are indexed by the *date of the future observation they predict*.

### Rolling GARCH with time-varying tails

For a **single-asset** portfolio, select `rolling_forecast.garch_forecaster` and pass `dist="StudentsT"` or `dist="skewstudent"`. Its returned `CovarianceResult.metadata["distribution_params"]` contains the shape parameters estimated **separately for every rolling window**. Pass these to `simulation.rolling_parametric_quantiles()` (and `rolling_parametric_tail_means()` for ES), then use the same risk-measure functions above. See Notebook 02 for the complete implementation.

## Design and timing conventions

```text
MarketData (adjusted prices → decimal log returns)
    ↓
Rolling model estimation (EWMA or univariate GARCH / GJR-GARCH)
    ↓
CovarianceResult: covariances[T, N, N] + target dates + asset names
    ↓
Portfolio monetary position values known at t−1
    ↓
Approximate conditional portfolio loss variance at t
    ↓
Innovation quantiles / tail means → monetary VaR / ES
    ↓
Realized monetary losses → violation indicators → VaR coverage backtests
```

- **Covariance units:** models estimate variances/covariances of *decimal log returns*; the portfolio applies **monetary position values** to obtain an approximate *monetary loss variance*.
- **No look-ahead:** the covariance forecast at target date `t` uses returns through `t−1`; the portfolio exposures also come from `t−1`.
- **Realized losses:** `Portfolio.loss()` uses exact *simple price changes* of the adjusted-price portfolio, while VaR/ES are based on a **first-order (delta-normal) loss approximation**.
- **Distribution parameters:** Student-t/skewed-t shape parameters are re-estimated with each rolling GARCH window; `arch` uses standardized, unit-variance innovations.
- **Model assumptions:** zero conditional return mean, one-trading-day horizon, and the chosen conditional innovation distribution. Multivariate EWMA in the example uses **Normal** innovations; no multivariate Student-t dependence model is fitted.

## Backtesting interpretation

Let `I_t = 1{realized loss_t > VaR_t}` and `alpha` be the nominal *tail probability*.

| Diagnostic | Interpretation |
|---|---|
| Violation ratio | `mean(I_t) / alpha`; a value near 1 indicates average exceedance frequency close to nominal |
| Kupiec `p_UC` | Tests correct **unconditional** violation frequency |
| Christoffersen `p_IND` | Tests **first-order independence** of successive violations |
| Christoffersen `p_CC` | Tests **coverage and independence jointly** (`LR_CC = LR_UC + LR_IND`) |

At a **5% test significance level**, a p-value below 0.05 means rejecting that test's null hypothesis. This 5% *test level* is different from the 5% *VaR tail probability*. Non-rejection does not prove that a risk model is correct, and the LR tests use **asymptotic chi-squared approximations**, which can be weak with few violations.

The original exploratory comparison suggested that **skewed Student-t specifications calibrated 1% VaR better than Normal specifications** on that sample. This is an empirical finding to recheck after downloading data and rerunning the notebooks—not a general claim that skewed-t/GJR models are always superior. Violation counts alone also do not fully rank forecast quality.

## Project layout

```text
.
├── README.md
├── requirements.txt
├── market_data.py
├── portfolio.py
├── models.py
├── rolling_forecast.py
├── co_variance_results.py
├── simulation.py
├── risk_measure.py
├── backtesting.py
├── test_portfolio.py
└── notebooks/
    ├── 01_multivariate_ewma.ipynb
    └── 02_univariate_var_backtesting.ipynb
```

## Scope and limitations

This is a **research / educational implementation**, not a production trading or regulatory-capital system. Yahoo Finance adjusted prices and a fixed-holdings index proxy simplify valuation, and the code does not model live transaction costs, cash dividends as separate cash flows, option nonlinearities, mean forecasts, multi-day overlapping horizon tests or production-quality model-convergence monitoring. Yahoo Finance may revise historical adjusted data, so exact notebook numbers can change.

**Possible next steps:** quantile (pinball) scores for direct VaR model comparisons, exact-binomial coverage tests for small numbers of violations, historical/filtered historical simulation, and full-revaluation Monte Carlo for nonlinear products.
