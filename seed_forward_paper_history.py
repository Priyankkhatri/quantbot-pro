"""
Seeds Forward Paper-Testing Ledger with Backtest History
========================================================
Runs strict day-by-day forward stepping over the historical evaluation period (2024 to 2026-10-09)
for all 6 registered strategies across both Standard (₹100k) and Micro (₹1k) tiers.
All rows are explicitly tagged with source='BACKTEST_SEED'.
Builds SHA-256 hash chains sequentially from Genesis.
"""

import os
import sys
import time
import sqlite3
import pandas as pd
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from forward_paper_testing import ForwardPaperEngine, REGISTERED_STRATEGIES, DB_PATH
from strategy_research_fair_benchmark import load_all_data

def seed_history(start_date_str: str = "2024-01-01", db_path: str = DB_PATH):
    # Safety Check: Refuse to run if live forward or catchup tracking rows exist
    if os.path.exists(db_path):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(paper_forward_ledger)")
            cols = [r[1] for r in cursor.fetchall()]
            if "source" in cols:
                fwd_rows = conn.execute("SELECT COUNT(*) FROM paper_forward_ledger WHERE source IN ('FORWARD', 'CATCHUP')").fetchone()[0]
                if fwd_rows > 0:
                    raise RuntimeError(f"ABORTED: {db_path} contains {fwd_rows} live FORWARD/CATCHUP rows! Seeding refused to protect forward track record.")
            conn.execute("DELETE FROM paper_forward_ledger;")
            conn.execute("DELETE FROM paper_forward_pending_orders;")
            conn.execute("DELETE FROM paper_forward_orders_journal;")
            conn.commit()
            print(f"Cleared existing rows in {db_path} for fresh genesis hash chain build.")

    print(f"Loading market data from SQLite cache...")
    nifty_df, stock_dfs = load_all_data()
    all_dfs = dict(stock_dfs)
    all_dfs["^NSEI"] = nifty_df

    all_dates = sorted(list(nifty_df.index))
    forward_dates = [d for d in all_dates if d >= pd.to_datetime(start_date_str)]

    print(f"Seeding paper ledger as BACKTEST_SEED across {len(forward_dates)} trading days ({forward_dates[0].strftime('%Y-%m-%d')} to {forward_dates[-1].strftime('%Y-%m-%d')})...")
    
    engine = ForwardPaperEngine(db_path=db_path, cost_model="ANGEL_ONE")
    start_time = time.time()

    open_prices_by_date = {}
    close_prices_by_date = {}
    for d in forward_dates:
        open_prices_by_date[d] = {sym: float(df.loc[d, "Open"]) for sym, df in all_dfs.items() if d in df.index}
        close_prices_by_date[d] = {sym: float(df.loc[d, "Close"]) for sym, df in all_dfs.items() if d in df.index}

    total_steps = len(forward_dates) * len(REGISTERED_STRATEGIES) * 2
    step_count = 0

    for strat_id, strat_def in REGISTERED_STRATEGIES.items():
        for cap_tier in ["STANDARD_100K", "MICRO_1K"]:
            for d in forward_dates:
                engine.step_forward_one_day(
                    strategy_id=strat_id,
                    version=strat_def.version,
                    capital_tier=cap_tier,
                    today_date=d,
                    all_dates=all_dates,
                    historical_data_dict=all_dfs,
                    today_open_prices=open_prices_by_date[d],
                    today_close_prices=close_prices_by_date[d],
                    source="BACKTEST_SEED"
                )
                step_count += 1

    elapsed = time.time() - start_time
    print(f"✓ Backtest seed history recorded: {step_count} blocks processed in {elapsed:.2f}s.")

    # Run cryptographic hash chain audit
    is_valid, msg, count = engine.store.verify_hash_chain()
    print(f"✓ Cryptographic Hash Chain Audit: {msg} (Verified {count} blocks)")

if __name__ == "__main__":
    seed_history("2024-01-01")
