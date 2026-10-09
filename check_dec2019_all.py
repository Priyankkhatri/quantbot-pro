import sqlite3
import pandas as pd

conn = sqlite3.connect('market_data_cache.db')
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT symbol FROM ohlcv_candles")
symbols = [r[0] for r in cursor.fetchall()]

for sym in symbols:
    df = pd.read_sql_query(f"SELECT timestamp, open, high, low, close FROM ohlcv_candles WHERE symbol='{sym}' AND timestamp >= '2019-12-15' AND timestamp <= '2019-12-25' ORDER BY timestamp ASC", conn)
    if not df.empty:
        print(f"=== {sym} Dec 2019 ===")
        for _, r in df.iterrows():
            print(f"  {r['timestamp'][:10]}: O={r['open']:.2f}, H={r['high']:.2f}, L={r['low']:.2f}, C={r['close']:.2f}")
