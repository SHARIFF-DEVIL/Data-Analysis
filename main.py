import matplotlib.pyplot as plt
from utilities.data import fetch_stock_data, load_data_from_csv
from utilities.analysis import decompose_series
from utilities.plot import plot_time_series, plot_decomposition, plot_forecast
from utilities.model import arima_forecast

def main():
    print("📈 Welcome to the Stock Forecast CLI App")

    ticker = input("🔹 Enter stock ticker (default = AAPL): ").upper() or "AAPL"

    try:
        days = int(input("🔹 Enter number of days to forecast (default = 30): ") or 30)
    except ValueError:
        print("⚠️ Invalid input. Using default 30 days.")
        days = 30

    print(f"\n📦 Fetching stock data for {ticker}...")
    fetch_stock_data(ticker)

    print("📊 Loading and processing data...")
    df = load_data_from_csv()

    print("📈 Plotting time series...")
    ts_fig = plot_time_series(df)
    ts_fig.savefig(f"{ticker}_timeseries.png")

    print("🧩 Decomposing time series...")
    decomposition = decompose_series(df)
    decomp_fig = plot_decomposition(decomposition)
    decomp_fig.savefig(f"{ticker}_decomposition.png")

    print(f"🔮 Forecasting next {days} days using ARIMA...")
    forecast_series = arima_forecast(df, steps=days)
    forecast_fig = plot_forecast(df, forecast_series, title=f"{ticker} Forecast (ARIMA)")
    forecast_fig.savefig(f"{ticker}_forecast.png")

    print("\n✅ Done!")
    print(f"📂 Files saved:\n- {ticker}_timeseries.png\n- {ticker}_decomposition.png\n- {ticker}_forecast.png")

if __name__ == "__main__":
    main()