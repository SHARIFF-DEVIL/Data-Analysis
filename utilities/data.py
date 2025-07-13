import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date, interval, save_as="Data.csv"):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

    data.reset_index(inplace=True)
    
    data.to_csv(save_as, index=False)


def load_data_from_csv(filepath="Data.csv"):
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[numeric_cols].dropna()
    return df