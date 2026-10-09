import sqlite3
import pandas as pd

conn = sqlite3.connect('market_data_cache.db')
for sym in ['NIFTYBEES.NS', 'BANKBEES.NS']:
    df = pd.read_sql_query(f"SELECT timestamp, open, high, low, close, volume FROM ohlcv_candles WHERE symbol='{sym}' ORDER BY timestamp ASC", conn)
    df['pct_change'] = df['close'].pct_change()
    df['abs_move'] = df['pct_change'].abs()
    print(f"\n==================== {sym} (Total: {len(df)}) ====================")
    top10 = df.sort_values('abs_move', ascending=False).head(10)
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        pct = row['pct_change'] * 100.0 if not pd.isna(row['pct_change']) else 0.0
        print(f"  {rank:2d}. {row['timestamp'][:10]} | Close: {row['close']:10.4f} | Move: {pct:+8.2f}% | Open: {row['open']:10.4f} | H: {row['high']:10.4f} | L: {row['low']:10.4f} | Vol: {row['volume']:12.0f}")
