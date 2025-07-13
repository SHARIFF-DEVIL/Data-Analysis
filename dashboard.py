import streamlit as st
from datetime import datetime, date
from utilities.data import fetch_stock_data, load_data_from_csv
from utilities.analysis import decompose_series, check_stationarity, plot_acf_pacf
from utilities.model import arima_forecast, random_forest_forecast
from utilities.plot import plot_time_series, plot_forecast, plot_decomposition

st.set_page_config(page_title="📈 Stock Forecasting App", layout="wide")
st.title("📈 Stock Price Forecasting App")

st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Enter stock ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2010, 1, 1).date())
end_date = st.sidebar.date_input("End Date", value=date.today(), max_value=date.today())
days_to_forecast = st.sidebar.slider("Forecast Days", min_value=10, max_value=100, value=30)

model_choice = st.sidebar.radio("Choose Forecasting Model", ("ARIMA", "Random Forest"))

if st.sidebar.button("🔄 Run Forecast"):
    with st.spinner("Fetching data and building forecast..."):
        fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
        df = load_data_from_csv()

        with st.expander("🗃 Raw Data"):
            st.dataframe(df.tail())

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
        else:
            forecast_series = random_forest_forecast(df, future_days=days_to_forecast)

        st.pyplot(plot_forecast(df, forecast_series, title=f"{model_choice} Forecast"))
        st.success("✅ Forecast complete!")