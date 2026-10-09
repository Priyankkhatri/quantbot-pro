import sqlite3
import pandas as pd
conn = sqlite3.connect('market_data_cache.db')
for sym in ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "JUNIORBEES.NS", "LIQUIDBEES.NS"]:
    df = pd.read_sql_query(f"SELECT timestamp, close FROM ohlcv_candles WHERE symbol='{sym}' ORDER BY timestamp", conn)
    print(f"{sym}: min={df['close'].min():.2f}, max={df['close'].max():.2f}, start={df['close'].iloc[0]:.2f}, end={df['close'].iloc[-1]:.2f}")
    df['pct'] = df['close'].pct_change()
    print(f"  Largest jumps: {df['pct'].abs().nlargest(3).values}")
