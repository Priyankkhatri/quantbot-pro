import sqlite3
import pandas as pd
import json

conn = sqlite3.connect("forward_paper_testing.db")
df = pd.read_sql_query(
    "SELECT sequence_num, trade_date, cash, portfolio_value, daily_pnl, daily_return_pct, positions_json FROM paper_forward_ledger WHERE strategy_id='LOW_VOL_120D_QUINTILE_Q' AND capital_tier='STANDARD_100K' AND trade_date >= '2026-10-01'",
    conn
)
print("Low-Vol October 2026:")
print(df.to_string())

# Check market data cache prices on 2026-10-07, 2026-10-08, 2026-10-09
pos = json.loads(df[df['trade_date']=='2026-10-08'].iloc[0]['positions_json'])
print("\nPositions held on 2026-10-08:", pos)

mconn = sqlite3.connect("market_data_cache.db")
for sym in pos.keys():
    # Try sym and sym.NS
    for test_s in [sym, sym + ".NS"]:
        p_df = pd.read_sql_query(f"SELECT timestamp, symbol, close FROM ohlcv_candles WHERE symbol='{test_s}' AND timestamp LIKE '2026-10-08%'", mconn)
        if not p_df.empty:
            print(f"Price on 2026-10-08 for {test_s}: {p_df.iloc[0]['close']}")

mconn.close()
conn.close()
