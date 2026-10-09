"""
Detailed Diagnostic: TOP_1_MOMENTUM_ROTATION_M across ₹10,000 and ₹100,000 Tiers
================================================================================
Dumps monthly rebalance decisions, momentum rankings, trades, and holdings side by side
to identify exact causes of divergence (tie-breaking, rounding, or cash allocation differences).
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

def diagnose_momentum_tiers():
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

    mom_6m = close_prices_df.shift(21) / close_prices_df.shift(126) - 1.0

    month_end_mask = np.zeros(n_bars, dtype=bool)
    for i in range(n_bars - 1):
        if common_dates[i].month != common_dates[i+1].month:
            month_end_mask[i] = True
    month_end_mask[-1] = True

    # Simulate ₹10k and ₹100k side by side
    tiers = {"₹10k": 10000.0, "₹100k": 100000.0}
    state = {
        "₹10k": {"cash": 10000.0, "pos": {s: 0 for s in etf_symbols}, "costs": 0.0, "turnover": 0.0, "vals": []},
        "₹100k": {"cash": 100000.0, "pos": {s: 0 for s in etf_symbols}, "costs": 0.0, "turnover": 0.0, "vals": []}
    }

    rebal_records = []

    for i in range(n_bars):
        if i >= 126 and month_end_mask[i-1]:
            d_str = common_dates[i-1].strftime("%Y-%m-%d")
            m_scores = mom_6m.iloc[i-1]
            valid_moms = m_scores.dropna()
            
            # Sort with explicit deterministic tie-breaking (symbol name secondary sort)
            sorted_items = sorted(valid_moms.items(), key=lambda x: (x[1], x[0]), reverse=True)
            top_sym = sorted_items[0][0]
            top_score = sorted_items[0][1]

            row_rec = {"date": d_str, "top_sym": top_sym, "score": top_score}

            for t_name, init_cap in tiers.items():
                st = state[t_name]
                cur_total = st["cash"] + sum(st["pos"][s] * close_prices_df[s].iloc[i-1] for s in etf_symbols)
                p_op = open_prices_df[top_sym].iloc[i]
                
                # Check target shares
                # Sells first
                for s in etf_symbols:
                    if s != top_sym and st["pos"][s] > 0:
                        s_p_op = open_prices_df[s].iloc[i]
                        shares_to_sell = st["pos"][s]
                        tr_val = shares_to_sell * s_p_op
                        c_info = cost_model.calculate_trade_cost("SELL", tr_val)
                        st["cash"] += (tr_val - c_info["total_cost"])
                        st["costs"] += c_info["total_cost"]
                        st["turnover"] += tr_val
                        st["pos"][s] = 0

                # Buy top_sym
                # How much cash can we spend?
                # Max whole shares = floor(cash / (p_op * 1.002))
                max_buyable = int(np.floor(max(0.0, st["cash"] - 20.0) / (p_op * 1.002)))
                diff = max_buyable - st["pos"][top_sym]
                
                if diff > 0:
                    tr_val = diff * p_op
                    c_info = cost_model.calculate_trade_cost("BUY", tr_val)
                    if st["cash"] >= (tr_val + c_info["total_cost"]):
                        st["cash"] -= (tr_val + c_info["total_cost"])
                        st["costs"] += c_info["total_cost"]
                        st["turnover"] += tr_val
                        st["pos"][top_sym] += diff

                row_rec[f"{t_name}_holdings"] = f"{top_sym} ({st['pos'][top_sym]} sh)"
                row_rec[f"{t_name}_cash"] = round(st["cash"], 2)
                row_rec[f"{t_name}_total"] = round(cur_total, 2)

            rebal_records.append(row_rec)

        for t_name in tiers.keys():
            st = state[t_name]
            port_val = st["cash"] + sum(st["pos"][s] * close_prices_df[s].iloc[i] for s in etf_symbols)
            st["vals"].append(port_val)

    print("=" * 90)
    print("   SAMPLE MONTHLY REBALANCES & HOLDINGS COMPARISON (₹10k vs ₹100k)")
    print("=" * 90)
    rebal_df = pd.DataFrame(rebal_records)
    print(f"Total Monthly Rebalance Events: {len(rebal_df)}")
    print("\nFirst 10 Rebalances:")
    print(rebal_df[["date", "top_sym", "₹10k_holdings", "₹10k_cash", "₹100k_holdings", "₹100k_cash"]].head(10).to_string(index=False))
    print("\nLast 10 Rebalances:")
    print(rebal_df[["date", "top_sym", "₹10k_holdings", "₹10k_cash", "₹100k_holdings", "₹100k_cash"]].tail(10).to_string(index=False))

    n_years = n_bars / 252.0
    for t_name, init_cap in tiers.items():
        st = state[t_name]
        vals = np.array(st["vals"])
        rets = np.diff(vals) / vals[:-1]
        cagr = (((vals[-1] / init_cap) ** (1.0 / n_years)) - 1.0) * 100.0
        vol = float(np.std(rets) * np.sqrt(252)) * 100.0
        peaks = np.maximum.accumulate(vals)
        dd = float(np.min((vals - peaks) / peaks)) * 100.0
        sharpe = (cagr - 5.5) / vol if vol > 0 else 0.0
        ann_turnover = (st["turnover"] / (init_cap * n_years)) * 100.0

        print(f"\n>>> Final {t_name} Tier Results:")
        print(f"    CAGR: {cagr:+.2f}% | Vol: {vol:.2f}% | Sharpe: {sharpe:.3f} | Max DD: {dd:.2f}% | Turnover: {ann_turnover:.1f}%/yr | Costs: ₹{st['costs']:.2f}")

if __name__ == "__main__":
    diagnose_momentum_tiers()
