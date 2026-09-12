import numpy as np, pandas as pd #,matplotlib.pyplot as plt 
import yfinance as yf


def download_prices(list_of_tickers, start, end):
    data = yf.download(list_of_tickers, start=start,end=end, auto_adjust=False)
    return data["Adj Close"]


def log_returns(prices):
    return np.log(prices).diff().dropna()