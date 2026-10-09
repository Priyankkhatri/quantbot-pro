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

print("Fetching Total-Return Clean Adjusted ETF data...")
now_str = datetime.datetime.now().isoformat()

for ticker in etf_tickers:
    try:
        print(f"  Downloading {ticker}...")
        raw_df = yf.download(ticker, period="10y", interval="1d", progress=False, auto_adjust=False)
        if raw_df is None or len(raw_df) == 0:
            print(f"    Warning: No data returned for {ticker}")
            continue
        
        # Flatten MultiIndex columns
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = [col[0] for col in raw_df.columns]

        raw_df = raw_df.dropna()
        
        # Compute adjustment ratio: Adj Close / Close
        adj_close = raw_df['Adj Close']
        raw_close = raw_df['Close']
        adj_ratio = adj_close / raw_close
        
        # Standard total return adjustment
        adj_open = raw_df['Open'] * adj_ratio
        adj_high = raw_df['High'] * adj_ratio
        adj_low = raw_df['Low'] * adj_ratio
        
        count = 0
        for dt in raw_df.index:
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            o = float(adj_open.loc[dt])
            h = float(adj_high.loc[dt])
            l = float(adj_low.loc[dt])
            c = float(adj_close.loc[dt])
            v = float(raw_df.loc[dt, 'Volume'])
            
            conn.execute("""
            INSERT OR REPLACE INTO ohlcv_candles (symbol, timeframe, timestamp, open, high, low, close, volume, source, fetched_at)
            VALUES (?, '1d', ?, ?, ?, ?, ?, ?, 'YFINANCE_ETF_CLEAN_ADJUSTED', ?)
            """, (ticker, ts_str, o, h, l, c, v, now_str))
            count += 1
        conn.commit()
        print(f"    ✓ Stored {count} clean adjusted bars for {ticker} ({raw_df.index[0].strftime('%Y-%m-%d')} to {raw_df.index[-1].strftime('%Y-%m-%d')})")
    except Exception as e:
        print(f"    Error downloading {ticker}: {e}")

print("Clean ETF Data Stored.")
