"""
QuantBot Pro - Independent Daily Ledger Value & Position Verification
======================================================================
Recomputes daily portfolio values strictly from:
Calculated_Value_t = Cash_t + Sum(Shares_i * Close_Price_i_t)
Compares against recorded portfolio_value in paper_forward_ledger.
Reports max_abs_diff_INR for every strategy and capital tier.
"""

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

from strategy_research_fair_benchmark import load_all_data

def verify_ledger_independent(db_path: str = "forward_paper_testing.db"):
    nifty_df, stock_dfs = load_all_data()
    all_dfs = dict(stock_dfs)
    all_dfs["^NSEI"] = nifty_df

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT strategy_id, version, capital_tier FROM paper_forward_ledger ORDER BY strategy_id, capital_tier")
    strat_combos = cursor.fetchall()

    print("=" * 110)
    print("   QUANTBOT PRO: INDEPENDENT LEDGER PORTFOLIO VALUE AUDIT (MAX ABS DIFF INR)")
    print("=" * 110)
    print(f"{'Strategy ID':<38} | {'Tier':<14} | {'Blocks':<7} | {'Max Abs Diff (INR)':<20} | {'Status'}")
    print("-" * 110)

    overall_pass = True
    results_summary = []

    for combo in strat_combos:
        strat_id = combo["strategy_id"]
        version = combo["version"]
        tier = combo["capital_tier"]

        cursor.execute("""
            SELECT sequence_num, trade_date, cash, portfolio_value, positions_json, data_status
            FROM paper_forward_ledger
            WHERE strategy_id = ? AND version = ? AND capital_tier = ?
            ORDER BY sequence_num ASC
        """, (strat_id, version, tier))
        rows = cursor.fetchall()

        max_diff = 0.0
        worst_date = None
        worst_rec_val = 0.0
        worst_calc_val = 0.0

        for r in rows:
            t_date = pd.to_datetime(r["trade_date"])
            cash = float(r["cash"])
            rec_val = float(r["portfolio_value"])
            positions = json.loads(r["positions_json"]) if r["positions_json"] else {}

            holdings_val = 0.0
            for sym, shares in positions.items():
                if shares > 0:
                    df_key = sym if sym in all_dfs else f"{sym}.NS"
                    if df_key in all_dfs:
                        df = all_dfs[df_key]
                        sub = df[df.index <= t_date]
                        if not sub.empty:
                            close_p = float(sub["Close"].iloc[-1])
                            holdings_val += shares * close_p

            calc_val = cash + holdings_val
            diff = abs(rec_val - calc_val)
            if diff > max_diff:
                max_diff = diff
                worst_date = r["trade_date"]
                worst_rec_val = rec_val
                worst_calc_val = calc_val

        status_str = "✓ PASS (0.00)" if max_diff < 0.05 else f"FAIL (Diff: ₹{max_diff:.2f} on {worst_date})"
        if max_diff >= 0.05:
            overall_pass = False

        print(f"{strat_id:<38} | {tier:<14} | {len(rows):<7} | ₹{max_diff:>17.4f} | {status_str}")
        results_summary.append({
            "strategy_id": strat_id,
            "capital_tier": tier,
            "blocks": len(rows),
            "max_abs_diff_INR": round(max_diff, 4),
            "passed": bool(max_diff < 0.05)
        })

    print("=" * 110)
    if overall_pass:
        print("🎉 ALL STRATEGIES & TIERS PASSED INDEPENDENT VALUATION AUDIT (max_abs_diff_INR near zero)!")
    else:
        print("⚠️ AUDIT FAILED: Discrepancies detected between recorded and recomputed portfolio values.")

    return results_summary

if __name__ == "__main__":
    verify_ledger_independent()
