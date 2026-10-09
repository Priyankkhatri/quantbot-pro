import sqlite3
import pandas as pd

conn = sqlite3.connect('market_data_cache.db')

for sym in ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "JUNIORBEES.NS", "LIQUIDBEES.NS"]:
    df = pd.read_sql_query(f"SELECT timestamp, open, high, low, close FROM ohlcv_candles WHERE symbol='{sym}' ORDER BY timestamp ASC", conn)
    
    # Check if close is < 25 on NIFTYBEES (which was ~130 in 2019)
    if sym == "NIFTYBEES.NS":
        glitch_df = df[(df['close'] < 30.0) & (df['timestamp'].str.startswith('2019'))]
        for _, row in glitch_df.iterrows():
            print(f"Fixing {sym} on {row['timestamp']}: {row['close']} -> {row['close']*10.0}")
            conn.execute(
                f"UPDATE ohlcv_candles SET open=open*10.0, high=high*10.0, low=low*10.0, close=close*10.0 WHERE symbol='{sym}' AND timestamp='{row['timestamp']}'"
            )
    conn.commit()

print("Glitch fix committed.")
