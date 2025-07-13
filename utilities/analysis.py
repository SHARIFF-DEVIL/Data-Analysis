import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def decompose_series(df):
    return sm.tsa.seasonal_decompose(df['Close'], model='additive', period=30)

def check_stationarity(series):
    result = adfuller(series)
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Critical Values": result[4]
    }

def plot_acf_pacf(series, lags=30):
    fig_acf = sm.graphics.tsa.plot_acf(series, lags=lags)
    fig_pacf = sm.graphics.tsa.plot_pacf(series, lags=lags)
    return fig_acf, fig_pacf
