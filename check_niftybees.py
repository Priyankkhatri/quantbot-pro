import sqlite3
import pandas as pd
conn = sqlite3.connect('market_data_cache.db')
df = pd.read_sql_query("SELECT timestamp, open, high, low, close, source FROM ohlcv_candles WHERE symbol='NIFTYBEES.NS' AND timestamp LIKE '2019-12-%' ORDER BY timestamp", conn)
print(df)
