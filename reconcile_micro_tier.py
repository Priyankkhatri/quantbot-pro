"""
Micro Tier (₹1,000) Diagnostics & CAGR Reconciliation
=====================================================
Analyzes EQUAL_WEIGHT_4_ETFS at ₹1,000:
1. Computes average % invested in equity vs average % in cash.
2. Explains the cash yield assumption (0% idle cash in broker ledger vs 5.5% LiquidBeES).
3. Dumps holdings over time.
4. Reconciles CAGR mathematically.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from strategy_research_etf_vol_overlay import load_etf_and_index_data, ResearchCostModel

def analyze_micro_tier():
    data_dict = load_etf_and_index_data()
    cost_model = ResearchCostModel(cost_model="ANGEL_ONE", dp_charge_per_sell=15.93)
    
    etf_symbols = ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "JUNIORBEES.NS"]
    common_dates = data_dict["NIFTYBEES.NS"].index
    for sym in etf_symbols:
        common_dates = common_dates.intersection(data_dict[sym].index)
    common_dates = sorted(list(common_dates))
    
    close_prices_df = pd.DataFrame({sym: data_dict[sym].loc[common_dates, "Close"] for sym in etf_symbols})
    open_prices_df = pd.DataFrame({sym: data_dict[sym].loc[common_dates, "Open"] for sym in etf_symbols})
    n_bars = len(common_dates)

    month_end_mask = np.zeros(n_bars, dtype=bool)
    for i in range(n_bars - 1):
        if common_dates[i].month != common_dates[i+1].month:
            month_end_mask[i] = True
    month_end_mask[-1] = True

    init_capital = 1000.0
    cash = init_capital
    pos = {s: 0 for s in etf_symbols}
    costs = 0.0
    turnover = 0.0
    
    daily_records = []

    for i in range(n_bars):
        # Quarterly rebalance
        if i == 0 or (month_end_mask[i-1] and common_dates[i-1].month in [3, 6, 9, 12]):
            cur_total = cash + sum(pos[s] * close_prices_df[s].iloc[max(0, i-1)] for s in etf_symbols)
            target_alloc = cur_total / len(etf_symbols)  # ₹250 target per ETF

            # Sells first
            for s in etf_symbols:
                p_op = open_prices_df[s].iloc[i]
                tgt_shares = int(np.floor(target_alloc / p_op))
                diff = tgt_shares - pos[s]
                if diff < 0:
                    tr_val = abs(diff) * p_op
                    c_info = cost_model.calculate_trade_cost("SELL", tr_val)
                    cash += (tr_val - c_info["total_cost"])
                    costs += c_info["total_cost"]
                    turnover += tr_val
                    pos[s] = tgt_shares

            # Buys second
            for s in etf_symbols:
                p_op = open_prices_df[s].iloc[i]
                tgt_shares = int(np.floor(target_alloc / p_op))
                diff = tgt_shares - pos[s]
                if diff > 0:
                    tr_val = diff * p_op
                    c_info = cost_model.calculate_trade_cost("BUY", tr_val)
                    if cash >= (tr_val + c_info["total_cost"]):
                        cash -= (tr_val + c_info["total_cost"])
                        costs += c_info["total_cost"]
                        turnover += tr_val
                        pos[s] = tgt_shares

        # Mark to market at close
        holdings_val = sum(pos[s] * close_prices_df[s].iloc[i] for s in etf_symbols)
        total_val = cash + holdings_val
        invested_pct = (holdings_val / total_val) * 100.0 if total_val > 0 else 0.0
        cash_pct = (cash / total_val) * 100.0 if total_val > 0 else 0.0

        daily_records.append({
            "date": common_dates[i],
            "total_val": total_val,
            "cash": cash,
            "holdings_val": holdings_val,
            "invested_pct": invested_pct,
            "cash_pct": cash_pct,
            "shares_nifty": pos["NIFTYBEES.NS"],
            "shares_bank": pos["BANKBEES.NS"],
            "shares_gold": pos["GOLDBEES.NS"],
            "shares_junior": pos["JUNIORBEES.NS"]
        })

    df_res = pd.DataFrame(daily_records)
    
    avg_invested = df_res["invested_pct"].mean()
    avg_cash = df_res["cash_pct"].mean()
    
    n_years = n_bars / 252.0
    final_val = df_res["total_val"].iloc[-1]
    cagr = (((final_val / init_capital) ** (1.0 / n_years)) - 1.0) * 100.0

    print("=" * 85)
    print("   MICRO TIER (₹1,000) EQUAL_WEIGHT_4_ETFS HOLDINGS & CASH ALLOCATION")
    print("=" * 85)
    print(f"Initial Capital          : ₹{init_capital:,.2f}")
    print(f"Final Value              : ₹{final_val:,.2f}")
    print(f"Realized Net CAGR        : {cagr:+.2f}%")
    print(f"Average Invested Equity %: {avg_invested:.2f}%")
    print(f"Average Cash Drag %      : {avg_cash:.2f}%")
    print(f"Cash Yield Assumption    : 0.0% (idle uninvested cash in trading account)")
    print(f"Total Transaction Costs  : ₹{costs:,.2f}")
    
    print("\nSample Holdings Over Time (Every 2 Years):")
    sample_dates = df_res.iloc[::500]
    for _, r in sample_dates.iterrows():
        print(f"  • {r['date'].strftime('%Y-%m-%d')}: Total=₹{r['total_val']:.2f} | Cash=₹{r['cash']:.2f} ({r['cash_pct']:.1f}%) | NIFTY={r['shares_nifty']} sh | BANK={r['shares_bank']} sh | GOLD={r['shares_gold']} sh | JUNIOR={r['shares_junior']} sh")

if __name__ == "__main__":
    analyze_micro_tier()
