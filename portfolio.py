import numpy as np, pandas as pd, matplotlib.pyplot as plt 


def loss_stock_port(weights, risk_factors, port_value=1):
    return - port_value * ( np.exp(risk_factors) - 1) @ weights