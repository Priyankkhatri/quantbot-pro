import sys
import datetime
import sqlite3
import pandas as pd
import yfinance as yf

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

conn = sqlite3.connect('market_data_cache.db')

etf_tickers = [
    "NIFTYBEES.NS",
    "BANKBEES.NS",
    "GOLDBEES.NS",
    "JUNIORBEES.NS",
    "ITBEES.NS",
    "LIQUIDBEES.NS",
    "SILVERBEES.NS"
]

print("Fetching Split-Adjusted Indian ETF historical daily bars (auto_adjust=True)...")
now_str = datetime.datetime.now().isoformat()

for ticker in etf_tickers:
    try:
        print(f"  Downloading {ticker}...")
        df = yf.download(ticker, period="10y", interval="1d", progress=False, auto_adjust=True)
        if df is None or len(df) == 0:
            print(f"    Warning: No data returned for {ticker}")
            continue
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df = df.dropna()
        count = 0
        for dt, row in df.iterrows():
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            o = float(row['Open'])
            h = float(row['High'])
            l = float(row['Low'])
            c = float(row['Close'])
            v = float(row.get('Volume', 0.0))
            
            conn.execute("""
            INSERT OR REPLACE INTO ohlcv_candles (symbol, timeframe, timestamp, open, high, low, close, volume, source, fetched_at)
            VALUES (?, '1d', ?, ?, ?, ?, ?, ?, 'YFINANCE_ETF_ADJUSTED', ?)
            """, (ticker, ts_str, o, h, l, c, v, now_str))
            count += 1
        conn.commit()
        print(f"    ✓ Stored {count} split-adjusted bars for {ticker} ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
    except Exception as e:
        print(f"    Error downloading {ticker}: {e}")

print("ETF Data Refresh Complete.")
