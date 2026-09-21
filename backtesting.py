
import pandas as pd


def violation(
        risk_factors: pd.DataFrame,
        data: pd.DataFrame,
        window_length: int,
        forecaster,
        **forecaster_kwargs,
        q,


):
    
    my_forecast = rolling_forecast.rolling_forecast(
        risk_factors=risk_factors,
        window_length=window_length,
        forecaster=forecaster #rolling_forecast.forecaster
        *forecaster_kwargs,
    )

    my_portBT_cond_var = portBT.portfolio_variance(data,my_forecast)

    # q = simulation.parametric_quantiles("Normal",[0.01,0.05])
    my_var = risk_measure.parametric_value_at_risk(my_portBT_cond_var,q) 


    my_weighted_returns = portBT.portfolio_weighted_log_return(data)

    # print(my_weighted_returns[500:])
    # print(my_var)

    list_of_violations=  my_weighted_returns.iloc[500:] > my_var["5%"] 

    return list_of_violations



# if __name__ == "__main__":

#     import matplotlib.pyplot as plt 
#     import portfolio
#     import market_data    
#     import simulation
#     import rolling_forecast
#     import risk_measure


#     portBT = portfolio.Portfolio(
#     holdings={
#         "AAPL": 2,
#         "MSFT": 3
#     }
#     )

#     dataBT = market_data.MarketData.from_yfinance(
#     portBT.assets,
#     "2000-01-01",
#     "2020-12-31",
#     )   

#     risk_factorsBT = portBT.risk_factors(dataBT)

#     my_forecast = rolling_forecast.rolling_forecast(
#         risk_factorsBT,
#         window_length=500,
#         forecaster=rolling_forecast.multivariate_ewma_forecaster,
#         decay=0.96
#         # p=1,
#         # q=1,
#         # o=0,
#         # dist="StudentsT",
#     )

#     my_portBT_cond_var = portBT.portfolio_variance(dataBT,my_forecast)

#     q = simulation.parametric_quantiles("Normal",[0.01,0.05])
#     my_var = risk_measure.parametric_value_at_risk(my_portBT_cond_var,q) 

#     tail_means = simulation.parametric_tail_means("Normal",[0.01,0.05])
#     my_es = risk_measure.expected_shortfall(my_portBT_cond_var, tail_means)

#     my_weighted_returns = portBT.portfolio_weighted_log_return(dataBT)

#     # print(my_var)
#     # my_var.plot()  
#     # # my_es.plot()
#     # my_weighted_returns.plot()
#     # plt.show()  

#     my_weighted_returns = portBT.portfolio_weighted_log_return(dataBT)

#     print(my_weighted_returns[500:])
#     print(my_var)

#     print(my_weighted_returns.iloc[500:] > my_var["5%"])
