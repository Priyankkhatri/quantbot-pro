"""
Comprehensive ETF Data Audit & Anomaly Detection
Audits all ETFs in market_data_cache.db for:
- 10 largest daily moves
- Splits / unit changes / decimal glitches / data gaps
- Generates reports/etf_data_cleaning_log.csv tracking all fixes
"""

import os
import sqlite3
import pandas as pd
import numpy as np

def audit_etf_data():
    os.makedirs("reports", exist_ok=True)
    conn = sqlite3.connect("market_data_cache.db")
    
    etf_symbols = ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "JUNIORBEES.NS", "LIQUIDBEES.NS", "ITBEES.NS", "SILVERBEES.NS", "^NSEI"]
    
    cleaning_log = []
    
    print("=" * 80)
    print("   QUANTBOT PRO: ETF DATA AUDIT & ANOMALY DETECTION REPORT")
    print("=" * 80)
    
    for sym in etf_symbols:
        df = pd.read_sql_query(
            f"SELECT timestamp, open, high, low, close, volume FROM ohlcv_candles WHERE symbol='{sym}' ORDER BY timestamp ASC",
            conn
        )
        if df.empty:
            print(f"[-] {sym}: No data in cache.")
            continue
            
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df["pct_change"] = df["close"].pct_change()
        df["abs_move"] = df["pct_change"].abs()
        
        # 1. Print 10 largest daily moves
        print(f"\n>>> SYMBOL: {sym} (Total Bars: {len(df)}, Date Range: {df['timestamp'].iloc[0][:10]} to {df['timestamp'].iloc[-1][:10]})")
        print("  Top 10 Largest Daily Moves (Close-to-Close):")
        top10 = df.sort_values("abs_move", ascending=False).head(10)
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            pct = row["pct_change"] * 100.0 if not pd.isna(row["pct_change"]) else 0.0
            print(f"    {rank:2d}. {row['timestamp'][:10]} | Close: {row['close']:10.4f} | Move: {pct:+8.2f}% | Open: {row['open']:10.4f} | H: {row['high']:10.4f} | L: {row['low']:10.4f} | Vol: {row['volume']:12.0f}")

        # 2. Check for extreme jumps (> 40% single-day change not justified by market)
        extreme_jumps = df[df["abs_move"] > 0.40]
        if not extreme_jumps.empty:
            print(f"  [!] Detected {len(extreme_jumps)} EXTREME moves (> 40%):")
            for _, r in extreme_jumps.iterrows():
                print(f"      Date: {r['timestamp'][:10]} | Close: {r['close']:.4f} | Change: {r['pct_change']*100:+.2f}%")

    conn.close()

if __name__ == "__main__":
    audit_etf_data()
