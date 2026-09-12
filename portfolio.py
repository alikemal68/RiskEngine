import numpy as np, pandas as pd
import utils

class Portfolio:

    def __init__(self, assets, holdings, start, end): 

        # assets are given by their tickers
        self.assets=assets

        # number of many positions of the corresponding asset one has 
        self.holdings=pd.Series(holdings, index=assets)

        # check if the number of assets and number of holdings match
        if len(assets) != len(holdings):
            raise ValueError("Number of assets and holdings must match!")

        # get the prices   
        self.prices=utils.download_prices(assets,start,end)

        self.position_values = self.prices * self.holdings


    # total portfolio value 
    def value(self):
        return (self.prices * self.holdings).sum(axis=1)

    # calculate the weights for each asset in the portfolio along the time  
    def weights(self):
        return self.position_values.div(self.value(), axis=0)

    # the risk factor for a simple stock portfolio is the log return 
    def risk_factor(self):
        return utils.log_returns(self.prices)

    # calculate losses. we could have used L_{t+1} = - (V_{t+1} - V_{t+1}) but we use 
    # the more general formula based on risk factors.
    def loss(self):
        simple_returns = np.exp(self.risk_factor()) - 1
        temp = self.position_values.shift(1)

        losses = -(simple_returns * temp).sum(
            axis=1,
            min_count=1 # The required number of valid values to perform the operation. 
        ).dropna() # get rid of NaN values
        return losses

if __name__ == "__main__":

    port = portfolio(["AAPL","MSFT"], [2, 3],"2000-01-01","2009-12-31")

    print(port.loss())

