import sqlite3
import pandas as pd

conn = sqlite3.connect("market_data_cache.db")
df = pd.read_sql_query("SELECT symbol, min(timestamp) as min_t, max(timestamp) as max_t, count(*) as cnt FROM ohlcv_candles WHERE timeframe='1d' GROUP BY symbol ORDER BY symbol ASC", conn)
print(df.to_string())

# Check 2026-10-08 specifically
df_08 = pd.read_sql_query("SELECT symbol, timestamp, close FROM ohlcv_candles WHERE timeframe='1d' AND timestamp LIKE '2026-10-08%'", conn)
print("\nSymbols with data on 2026-10-08:")
print(df_08.to_string())

# Check 2026-10-09 specifically
df_09 = pd.read_sql_query("SELECT symbol, timestamp, close FROM ohlcv_candles WHERE timeframe='1d' AND timestamp LIKE '2026-10-09%'", conn)
print("\nSymbols with data on 2026-10-09:")
print(df_09.to_string())

# Check 2025-10-24 specifically
df_24 = pd.read_sql_query("SELECT symbol, timestamp, close FROM ohlcv_candles WHERE timeframe='1d' AND timestamp LIKE '2025-10-24%'", conn)
print("\nSymbols with data on 2025-10-24:")
print(f"Total symbols on 2025-10-24: {len(df_24)}")

conn.close()
