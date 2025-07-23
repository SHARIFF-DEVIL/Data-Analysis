import streamlit as st
from datetime import datetime, date
from utilities.data import fetch_stock_data, load_data_from_csv
from utilities.analysis import decompose_series, check_stationarity, plot_acf_pacf
from utilities.model import arima_forecast, sarima_forecast, prophet_forecast, lstm_forecast
from utilities.plot import plot_time_series, plot_forecast, plot_decomposition
import pandas as pd

st.set_page_config(page_title="📈 Stock Forecasting App", layout="wide")
st.title("📈 Stock Price Forecasting App")

st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Enter stock ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2010, 1, 1).date())
end_date = st.sidebar.date_input("End Date", value=date.today(), max_value=date.today())
days_to_forecast = st.sidebar.slider("Forecast Days", min_value=10, max_value=100, value=30)

model_choice = st.sidebar.radio(
    "Choose Forecasting Model",
    ("ARIMA", "SARIMA", "Prophet", "LSTM")
)

if st.sidebar.button("🔄 Run Forecast"):
    with st.spinner("Fetching data and building forecast..."):
        fetch_stock_data(ticker, start_date=start_date, end_date=end_date, interval='1d')
        df = load_data_from_csv()

        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
        df.index.name = 'Date'

        with st.expander("🗃 Raw Data"):
            st.dataframe(df.tail())
              
        csv_original = df.to_csv(index=True)
        st.download_button(
            label="⬇️ Download Original Dataset",
            data=csv_original,
            file_name=f"{ticker}_historical_data_{date.today()}.csv",
            mime='text/csv',
            help="Download the complete historical data."
        )    

        st.subheader("📊 Historical Time Series")
        st.pyplot(plot_time_series(df))

        st.subheader("🧩 Time Series Decomposition")
        decomposition = decompose_series(df)
        st.pyplot(plot_decomposition(decomposition))

        st.subheader("📉 ADF Test (Stationarity Check)")
        adf_result = check_stationarity(df['Close'])
        st.write(adf_result)

        st.subheader("📐 ACF & PACF Plots")
        acf_fig, pacf_fig = plot_acf_pacf(df['Close'])
        st.pyplot(acf_fig)
        st.pyplot(pacf_fig)

        st.subheader(f"🔮 {model_choice} Forecast - Next {days_to_forecast} Days")
        if model_choice == "ARIMA":
            forecast_series = arima_forecast(df, steps=days_to_forecast)
        elif model_choice == "SARIMA":
            forecast_series = sarima_forecast(df, steps=days_to_forecast)
        elif model_choice == "Prophet":
            forecast_series = prophet_forecast(df, steps=days_to_forecast)
        elif model_choice == "LSTM":
            forecast_series = lstm_forecast(df, steps=days_to_forecast)
        else:
            st.error("Unsupported model selected.")
            st.stop()

        try:
            fig = plot_forecast(df, forecast_series, title=f"{model_choice} Forecast")
            st.pyplot(fig)
        except ValueError as e:
            st.error(f"⚠️ Plotting failed: {e}")

        st.subheader("⬇️ Download Forecast Data")
        
        if isinstance(forecast_series, pd.Series):
            forecast_df = forecast_series.to_frame(name=f"{ticker}_Forecasted_Close")
        else:
            forecast_df = forecast_series
        
        forecast_filename = f"{ticker}_{model_choice}_forecast_{date.today()}.csv"

        csv = forecast_df.to_csv(index=True)
        
        st.download_button(
            label="Download Forecasted Series as CSV",
            data=csv,
            file_name=forecast_filename,
            mime='text/csv',
            help="Download the forecasted stock prices."
        )

        st.success("✅ Forecast complete!")
        
        st.markdown("---")
        st.markdown("""
        ## Developed by Group 26

        ### Team Members
        1. Rohit Kumar
        2. Mohammed Shabar
        3. Anupam Kanoongo
        4. Nupur Agarwal
        5. Nilambar Elangbam
        6. Patlolla Hari Haran Reddy
        """)