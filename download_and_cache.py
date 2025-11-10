"""
Download and Cache Stock Data

This utility script downloads historical stock data from Yahoo Finance
and saves it locally for offline use in the Portfolio Tracker application.

You can modify the TICKERS list to add or remove stocks.
The PERIOD can be adjusted to download different time ranges.
"""

import yfinance as yf
import os

TICKERS = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "AVGO", "GOOG", "META", "TSLA", "JPM", "LLY", "WMT", "ORCL", "V", "MA", "XOM", "NFLX", "JNJ", "PLTR", "COST", "BAC", "ABBV", "AMD", "HD"]
PERIOD = "10y"
CACHE_DIR = "data_cache"

def download_and_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    data = yf.download(TICKERS, period=PERIOD, group_by="ticker", progress=False)
    for ticker in TICKERS:
        df = data[ticker].dropna()
        df.to_csv(f"{CACHE_DIR}/{ticker}_{PERIOD}history.csv")
        print(f"Saved {ticker}")

if __name__ == "__main__":
    download_and_cache()