import sqlite3
import pandas as pd

conn = sqlite3.connect('market_data_cache.db')
query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv_candles WHERE symbol='^NSEI' AND timeframe='1d' ORDER BY timestamp ASC"
df_nifty = pd.read_sql_query(query, conn)
print(f"Nifty bars: {len(df_nifty)}, {df_nifty['timestamp'].min()} to {df_nifty['timestamp'].max()}")

stocks_query = "SELECT DISTINCT symbol FROM ohlcv_candles WHERE timeframe='1d' AND symbol NOT LIKE '^%'"
stocks = [r[0] for r in conn.execute(stocks_query).fetchall()]
print(f"Total equity stocks available: {len(stocks)}: {stocks[:10]}...")
conn.close()
