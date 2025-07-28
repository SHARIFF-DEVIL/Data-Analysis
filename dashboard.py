
import streamlit as st
from datetime import datetime, date
from utilities.data import fetch_stock_data, load_data_from_csv
from utilities.analysis import decompose_series, check_stationarity, plot_acf_pacf
from utilities.model import arima_forecast, sarima_forecast, prophet_forecast, lstm_forecast
import pandas as pd
import numpy as np
import plotly.graph_objs as go

def plotly_time_series(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    fig.update_layout(title='Historical Time Series', xaxis_title='Date', yaxis_title='Close Price')
    return fig

def plotly_decomposition(decomposition):
    import plotly.subplots as sp
    fig = sp.make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'])
    fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual'), row=4, col=1)
    fig.update_layout(height=900, showlegend=False, title='Time Series Decomposition')
    return fig

def plotly_acf_pacf(acf_vals, pacf_vals, lags):
    import plotly.subplots as sp
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=['ACF', 'PACF'])
    fig.add_trace(go.Bar(x=list(range(lags)), y=acf_vals, name='ACF'), row=1, col=1)
    fig.add_trace(go.Bar(x=list(range(lags)), y=pacf_vals, name='PACF'), row=1, col=2)
    fig.update_layout(title='ACF & PACF Plots')
    return fig



def plotly_ohlc_lines(forecast_df, title="Forecast (OHLC)"):
    fig = go.Figure()
    if all(col in forecast_df.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Close'],
            mode='lines+markers',
            name='Close',
            customdata=np.stack([forecast_df['Open'], forecast_df['High'], forecast_df['Low'], forecast_df['Close']], axis=-1),
            hovertemplate=
                'Date: %{x}<br>' +
                'Open: %{customdata[0]:.2f}<br>' +
                'High: %{customdata[1]:.2f}<br>' +
                'Low: %{customdata[2]:.2f}<br>' +
                'Close: %{customdata[3]:.2f}<extra></extra>'
        ))
    else:
        close_col = forecast_df.columns[0] if forecast_df.shape[1] == 1 else 'Close'
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df[close_col],
            mode='lines+markers',
            name=close_col,
            hovertemplate='Date: %{x}<br>'+f'{close_col}: '+'%{y:.2f}<extra></extra>'
        ))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price')
    return fig

st.set_page_config(page_title="Stock Forecasting App")
st.title("Stock Price Forecasting App")

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter stock ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2010, 1, 1).date())
end_date = st.sidebar.date_input("End Date", value=date.today(), max_value=date.today())

st.sidebar.markdown("### Forecast Days")
col1, col2 = st.sidebar.columns([2, 1])

with col1:
    days_to_forecast = st.number_input(
        "Enter days",
        min_value=1,
        max_value=100,
        value=30,
        step=1,
        label_visibility="collapsed"
    )


st.sidebar.markdown("---")
model_choice = st.sidebar.radio(
    "Choose Forecasting Model",
    ("ARIMA", "SARIMA", "Prophet", "LSTM")
)

if st.sidebar.button("Run Forecast"):
    with st.spinner("Fetching data and building forecast..."):
        fetch_stock_data(ticker, start_date=start_date, end_date=end_date, interval='1d')
        df = load_data_from_csv()
        
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
        df.index.name = 'Date'

        tab1, tab2, tab3 = st.tabs([
            "Forecast Section",
            "Historical Time Series",
            "Time Series Analysis"
            
        ])

        with tab2:
            st.subheader("Raw Data")
            st.dataframe(df.tail())
              
            csv_original = df.to_csv(index=True)
            st.download_button(
                label="⬇️ Download Original Dataset",
                data=csv_original,
                file_name=f"{ticker}_historical_data_{date.today()}.csv",
                mime='text/csv',
                help="Download the complete historical data."
            )    
            st.subheader("Historical Time Series")
            fig1 = plotly_time_series(df)
            st.plotly_chart(fig1, use_container_width=True)

        with tab3:
            st.subheader("Time Series Decomposition")
            decomposition = decompose_series(df)
            fig2 = plotly_decomposition(decomposition)
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("📉 ADF Test (Stationarity Check)")
            adf_result = check_stationarity(df['Close'])
            
            st.markdown("### Results:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ADF Statistic", f"{adf_result['ADF Statistic']:.4f}")
                st.metric("p-value", f"{adf_result['p-value']:.4f}")
            
            st.markdown("### Critical Values:")
            cols = st.columns(3)
            with cols[0]:
                st.metric("1%", f"{adf_result['Critical Values']['1%']:.4f}")
            with cols[1]:
                st.metric("5%", f"{adf_result['Critical Values']['5%']:.4f}")
            with cols[2]:
                st.metric("10%", f"{adf_result['Critical Values']['10%']:.4f}")

            st.markdown("#### Interpretation:")
            if adf_result['p-value'] > 0.05:
                st.warning("❗ The series is **non-stationary** (p-value > 0.05)")
            else:
                st.success("✅ The series is **stationary** (p-value ≤ 0.05)")

            st.subheader("ACF & PACF Plots")
            from statsmodels.graphics.tsaplots import acf, pacf
            lags = min(40, len(df['Close']) // 2)
            acf_vals = acf(df['Close'], nlags=lags)
            pacf_vals = pacf(df['Close'], nlags=lags)
            acf_pacf_fig = plotly_acf_pacf(acf_vals, pacf_vals, lags+1)
            st.plotly_chart(acf_pacf_fig, use_container_width=True)

        with tab1:
            st.subheader(f"{model_choice} Forecast - Next {days_to_forecast} Days")
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

            if isinstance(forecast_series, pd.Series):
                close_values = forecast_series.values
                volatility = 0.02
                
                forecast_df = pd.DataFrame({
                    'Close': close_values,
                    'Open': close_values * (1 + np.random.normal(0, volatility, len(close_values))),
                    'High': close_values * (1 + abs(np.random.normal(0, volatility, len(close_values)))),
                    'Low': close_values * (1 - abs(np.random.normal(0, volatility, len(close_values))))
                }, index=forecast_series.index)
                
                forecast_df['High'] = forecast_df[['Open', 'Close', 'High']].max(axis=1)
                forecast_df['Low'] = forecast_df[['Open', 'Close', 'Low']].min(axis=1)
            else:
                forecast_df = pd.DataFrame(forecast_series)

            if all(col in forecast_df.columns for col in ['Open', 'High', 'Low', 'Close']):
                forecast_df = forecast_df[['Open', 'High', 'Low', 'Close']]

            try:
                fig3 = plotly_ohlc_lines(forecast_df, title=f"{model_choice} Forecast (OHLC)")
                st.plotly_chart(fig3, use_container_width=True)
            except ValueError as e:
                st.error(f"Plotting failed: {e}")

            st.subheader("Forecasted Data Table")
            st.dataframe(forecast_df)

            st.subheader("Download Forecast Data")

            forecast_filename = f"{ticker}_{model_choice}_forecast_OHLC_{date.today()}.csv"
            csv = forecast_df.to_csv(index=True)
            st.download_button(
                label="Download Forecasted Series as CSV",
                data=csv,
                file_name=forecast_filename,
                mime='text/csv',
                help="Download the forecasted stock prices."
            )

            st.success("Forecast complete!")
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
