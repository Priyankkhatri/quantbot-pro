import sqlite3
import pandas as pd
import yfinance as yf

conn = sqlite3.connect('market_data_cache.db')
cur = conn.cursor()
cur.execute("SELECT symbol, COUNT(*), MIN(timestamp), MAX(timestamp) FROM ohlcv_candles WHERE timeframe='1d' GROUP BY symbol ORDER BY symbol")
rows = cur.fetchall()
print(f"Total symbols in cache: {len(rows)}")
for r in rows:
    print(f"  {r[0]}: {r[1]} bars ({r[2]} to {r[3]})")
