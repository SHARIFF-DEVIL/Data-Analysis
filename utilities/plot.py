import matplotlib.pyplot as plt
import pandas as pd

def plot_time_series(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    if 'Close' not in df.columns:
        raise ValueError("Column 'Close' not found in DataFrame")
    df['Close'].plot(ax=ax, title="Stock Closing Price")
    ax.set_ylabel("Price")
    ax.grid(True)
    return fig

def plot_forecast(df, forecast_series, title="Forecast"):
    fig, ax = plt.subplots(figsize=(12, 5))
    forecast_series.plot(ax=ax, color='orange', label='Forecast')
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Close Price")
    ax.legend()
    ax.grid(True)
    return fig

def plot_decomposition(decomposition):
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    return fig