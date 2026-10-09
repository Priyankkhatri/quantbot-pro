import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def recompute_metrics_from_ledger(db_path="forward_paper_testing.db"):
    conn = sqlite3.connect(db_path)
    print("=" * 110)
    print("   QUANTBOT PRO: INDEPENDENT LEDGER RECOMPUTATION & ANOMALY DIAGNOSTIC")
    print("=" * 110)

    strategies = [
        "NIFTY50_BUY_HOLD",
        "EQUAL_WEIGHT_33_SURVIVORSHIP_BIASED",
        "LOW_VOL_120D_QUINTILE_Q",
        "MOMENTUM_12_1M_QUINTILE_M",
        "ML_HYSTERESIS_RANKER_M",
        "NIFTY_200EMA_TREND_FILTER"
    ]
    tiers = ["STANDARD_100K", "MICRO_1K"]

    # First fetch Nifty benchmark daily returns for excess return calculation
    nifty_df_std = pd.read_sql_query(
        "SELECT trade_date, portfolio_value, daily_return_pct FROM paper_forward_ledger WHERE strategy_id='NIFTY50_BUY_HOLD' AND capital_tier='STANDARD_100K' ORDER BY sequence_num ASC",
        conn
    )
    nifty_rets_std = nifty_df_std["daily_return_pct"].values / 100.0

    print(f"\n{'Strategy ID':<35} | {'Tier':<13} | {'Total Ret (%)':<14} | {'CAGR (%)':<10} | {'Max DD (%)':<11} | {'Excess (%/yr)':<13} | {'Costs (₹)':<10}")
    print("-" * 110)

    for strat in strategies:
        for tier in tiers:
            df = pd.read_sql_query(
                f"SELECT sequence_num, trade_date, portfolio_value, daily_return_pct, cum_costs FROM paper_forward_ledger WHERE strategy_id='{strat}' AND capital_tier='{tier}' ORDER BY sequence_num ASC",
                conn
            )
            if df.empty:
                continue

            vals = df["portfolio_value"].values
            init_v = vals[0]
            final_v = vals[-1]
            tot_ret = ((final_v - init_v) / init_v) * 100.0 if init_v > 0 else 0.0

            n_days = len(vals)
            n_years = max(0.01, n_days / 252.0)
            cagr = (((final_v / init_v) ** (1.0 / n_years)) - 1.0) * 100.0 if init_v > 0 and final_v > 0 else 0.0

            peaks = np.maximum.accumulate(vals)
            dds = (vals - peaks) / np.where(peaks > 0, peaks, 1.0)
            max_dd = float(np.min(dds)) * 100.0

            daily_rets = df["daily_return_pct"].values / 100.0
            excess_daily = daily_rets - nifty_rets_std[:len(daily_rets)]
            excess_ann = float(np.mean(excess_daily)) * 252.0 * 100.0

            total_costs = df.iloc[-1]["cum_costs"]

            print(f"{strat:<35} | {tier:<13} | {tot_ret:>+12.2f}% | {cagr:>+8.2f}% | {max_dd:>+9.2f}% | {excess_ann:>+11.2f}% | ₹{total_costs:>8.2f}")

    # Inspect Trades for EQUAL_WEIGHT_33_SURVIVORSHIP_BIASED
    print("\n" + "=" * 110)
    print("   TRADE JOURNAL INSPECTION FOR EQUAL_WEIGHT_33_SURVIVORSHIP_BIASED (STANDARD_100K)")
    print("=" * 110)
    trades_df = pd.read_sql_query(
        "SELECT fill_date, symbol, action, shares, fill_price, total_cost FROM paper_forward_orders_journal WHERE strategy_id='EQUAL_WEIGHT_33_SURVIVORSHIP_BIASED' AND capital_tier='STANDARD_100K' ORDER BY fill_date ASC",
        conn
    )
    print(f"Total Completed Trades in Journal: {len(trades_df)}")
    if not trades_df.empty:
        print("\nFirst 10 Trades:")
        print(trades_df.head(10).to_string(index=False))
        print("\nLast 10 Trades:")
        print(trades_df.tail(10).to_string(index=False))

    # Inspect LOW_VOL_120D_QUINTILE_Q
    print("\n" + "=" * 110)
    print("   LOW_VOL_120D_QUINTILE_Q REBALANCE & HOLDINGS DIAGNOSTIC")
    print("=" * 110)
    lv_df = pd.read_sql_query(
        "SELECT sequence_num, trade_date, cash, portfolio_value, daily_return_pct, positions_json FROM paper_forward_ledger WHERE strategy_id='LOW_VOL_120D_QUINTILE_Q' AND capital_tier='STANDARD_100K' ORDER BY sequence_num ASC",
        conn
    )
    # Check largest daily drops
    lv_df['daily_drop'] = lv_df['daily_return_pct']
    worst_days = lv_df.sort_values('daily_drop', ascending=True).head(5)
    print("5 Worst Daily Drops for Low-Vol Strategy:")
    for _, r in worst_days.iterrows():
        print(f"  • Date: {r['trade_date']} | Daily Return: {r['daily_return_pct']:+.2f}% | Portfolio Value: ₹{r['portfolio_value']:,.2f} | Positions: {r['positions_json']}")

    conn.close()

if __name__ == "__main__":
    recompute_metrics_from_ledger()
