import numpy as np, pandas as pd, matplotlib.pyplot as plt 

# import portfolio
# import market_data
import models
# import simulation
# import risk_measure

from co_variance_results import CovarianceResult, VarianceResult

def garch_forecaster(
    window: pd.DataFrame,
    state=None,
    p=1,
    q=1,
    o=0,
    dist="Normal",
):
    fit_result = models.fit_garch(
        window,
        p=p,
        q=q,
        o=o,
        dist=dist,
        starting_values=state,
    )

    forecast = models.garch_forecast(fit_result)

    # new_state = fit_result.params.to_numpy()
    new_state = fit_result.params.to_numpy(copy=True)
    

    return forecast, new_state

def univariate_ewma_forecaster(
    window: pd.DataFrame,
    state=None,
    decay=0.96,
):
    result = models.univariate_ewma_variance_estimator(
        window,
        decay=decay,
    )

    return models.univariate_ewma_forecast(
        window,
        result,
    ), None

def multivariate_ewma_forecaster(
    window: pd.DataFrame,
    state=None,
    decay=0.96,
):
    result = models.multivariate_ewma_covariance_estimator(
        window,
        decay=decay,
    )

    return models.multivariate_ewma_forecast(
        window,
        result,
    ), None


def rolling_forecast(
    risk_factors: pd.DataFrame,
    window_length: int,
    forecaster,
    **forecaster_kwargs,
) -> VarianceResult | CovarianceResult:

    results = []
    dates = []

    state = None

    for t in range(window_length, len(risk_factors)):

        window = risk_factors.iloc[
            t - window_length:t
        ]

        forecast, state = forecaster(
            window,
            state=state,
            **forecaster_kwargs,
        )

        target_date = risk_factors.index[t]

        results.append(forecast)
        dates.append(target_date)

    # --------------------------------------------------------
    # Univariate
    # --------------------------------------------------------

    if risk_factors.shape[1] == 1:

        asset = risk_factors.columns[0]

        variances = pd.Series(
            [forecast.iloc[0] for forecast in results],
            index=dates,
            name=asset,
        )

        return VarianceResult(
            variances=variances,
            asset=asset,
        )

    # --------------------------------------------------------
    # Multivariate
    # --------------------------------------------------------

    else:

        assets = risk_factors.columns

        covariances = np.stack([
            forecast.to_numpy()
            for forecast in results
        ])

        return CovarianceResult(
            covariances=covariances,
            dates=pd.Index(dates),
            assets=assets,
        )






#     risk_factorsBT = portBT.risk_factors(dataBT)

# import numpy as np, pandas as pd, matplotlib.pyplot as plt 

# # import portfolio
# # import market_data
# # import models
# # import simulation
# # import risk_measure

# import volatility_forecaster

# class RollingForecastEngine:

#     def __init__(
#         self,
#         window_length: int,
#     ):
#         self.window_length = window_length

#     def run(
#         self,
#         risk_factors: pd.DataFrame,
#         forecaster: volatility_forecaster.VolatilityForecaster,
#     ):

#         results = []

#         for t in range(
#             self.window_length,
#             len(risk_factors),
#         ):

#             window = risk_factors.iloc[
#                 t - self.window_length:t
#             ]

#             forecast = forecaster.forecast(
#                 window
#             )

#             target_date = risk_factors.index[t]

#             results.append(
#                 (target_date, forecast)
#             )

#         return results


# # def ewma_forecaster(window: pd.DataFrame) -> pd.Series:
# #      variance_ewma_t = models.univariate_ewma_forecast(
# #         window,
# #         ewma_result,
# #     )



# # # --------------------------------------------------------
# # # EWMA variance path
# # # --------------------------------------------------------

# # ewma_result = models.univariate_ewma_variance_estimator(
# #     risk_factorsBT
# # )

# # # Normal quantiles do not change through time
# # q = simulation.parametric_quantiles(
# #     "Normal",
# #     [0.01, 0.05],
# # )


# # # --------------------------------------------------------
# # # Backtest
# # # --------------------------------------------------------


# # var_ewma_results = []
# # var_garch11_results = []

# # estimation_window_length = 500 # probably should be more than init_window from ewma 
# # testing_window_length = len(risk_factorsBT)


# # for t in range(estimation_window_length, testing_window_length):

# #     estimation_first_day = t - estimation_window_length
# #     estimation_last_day = t

# #     window = risk_factorsBT.iloc[
# #         estimation_first_day:estimation_last_day
# #     ]

# #     # ----------------------------------
# #     # EWMA: Use all the history or rolling forecast window????
# #     # ----------------------------------

# #     # Forecast sigma^2_(t | t-1)
# #     variance_ewma_t = models.univariate_ewma_forecast(
# #         window,
# #         ewma_result,
# #     )

# #     # VaR forecast for t
# #     var_ewma_t = risk_measure.parametric_value_at_risk(
# #         variance_ewma_t,
# #         q,
# #     )

# #     # ewma_forecast is indexed by forecast origin t-1,
# #     # but for backtesting we want the target date t
# #     target_date = risk_factorsBT.index[t]

# #     var_ewma_t.index = [target_date]

# #     var_ewma_results.append(var_ewma_t)


# #     # ----------------------------------
# #     # GARCH(1,1) with normal innovations
# #     # ----------------------------------

# #     garch11_fit = models.fit_garch(window,1,1,0,"Normal")

# #     variance_garch11_t = models.garch_forecast(garch11_fit)

# #     var_garch11_t = risk_measure.parametric_value_at_risk(variance_garch11_t,q)

# #     var_garch11_t.index = [target_date]
    
# #     var_garch11_results.append(var_garch11_t)


# # Var_ewma_data_frame = pd.concat(var_ewma_results)
# # Var_garch11_data_frame = pd.concat(var_garch11_results)


# # print(Var_ewma_data_frame)
# # Var_ewma_data_frame.plot()

# # print(Var_garch11_data_frame)
# # Var_garch11_data_frame.plot()





