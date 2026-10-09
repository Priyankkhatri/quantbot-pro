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

EQUITY_33_SYMBOLS = [
    "ADANIPORTS.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
    "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS", "DRREDDY.NS", "HCLTECH.NS",
    "HDFCBANK.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INFY.NS",
    "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "RELIANCE.NS", "SBIN.NS", "SUNPHARMA.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "WIPRO.NS"
]

def run_survivorship_audit_33():
    conn = sqlite3.connect("market_data_cache.db")
    
    print("=" * 100)
    print("   QUANTBOT PRO: SURVIVORSHIP BIAS & HINDSIGHT SENSITIVITY AUDIT (33 EQUITY STOCKS)")
    print("=" * 100)
    
    stock_dfs = {}
    for sym in EQUITY_33_SYMBOLS:
        df = pd.read_sql_query(f"SELECT timestamp, close FROM ohlcv_candles WHERE symbol='{sym}' ORDER BY timestamp ASC", conn)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            stock_dfs[sym] = df
            
    nifty_df = pd.read_sql_query("SELECT timestamp, close FROM ohlcv_candles WHERE symbol='^NSEI' ORDER BY timestamp ASC", conn)
    nifty_df['timestamp'] = pd.to_datetime(nifty_df['timestamp'])
    nifty_df.set_index('timestamp', inplace=True)
    
    # Synchronize dates across all 33 equity stocks and Nifty
    common_dates = sorted(list(nifty_df.index))
    for sym, df in stock_dfs.items():
        common_dates = sorted(list(set(common_dates).intersection(set(df.index))))
        
    n_stocks = len(stock_dfs)
    n_bars = len(common_dates)
    n_years = n_bars / 252.0
    start_d = common_dates[0].strftime('%Y-%m-%d')
    end_d = common_dates[-1].strftime('%Y-%m-%d')
    
    print(f"Universe Definition : Pure Equity Universe (Excludes all ETFs, Cash & Benchmarks)")
    print(f"Universe Size       : {n_stocks} Equity Stocks")
    print(f"Historical Period   : {start_d} to {end_d} ({n_bars} trading days, {n_years:.2f} years)\n")
    
    close_mat = pd.DataFrame({sym: stock_dfs[sym].loc[common_dates, "close"] for sym in EQUITY_33_SYMBOLS})
    ret_mat = close_mat.pct_change().dropna()
    
    nifty_rets = nifty_df.loc[common_dates, "close"].pct_change().dropna()
    nifty_cum = np.cumprod(1.0 + nifty_rets.values)
    nifty_cagr = (((nifty_cum[-1]) ** (1.0 / n_years)) - 1.0) * 100.0
    
    # 1. Baseline: 33 Equity Equal Weight
    base_daily_rets = ret_mat.mean(axis=1).values
    base_cum = np.cumprod(1.0 + base_daily_rets)
    base_cagr = (((base_cum[-1]) ** (1.0 / n_years)) - 1.0) * 100.0
    base_vol = float(np.std(base_daily_rets) * np.sqrt(252)) * 100.0
    peaks_b = np.maximum.accumulate(base_cum)
    base_max_dd = float(np.min((base_cum - peaks_b) / peaks_b)) * 100.0
    base_sharpe = (base_cagr - 5.5) / base_vol if base_vol > 0 else 0.0

    print("1. BASELINE BUY-AND-HOLD COMPARISON:")
    print(f"   • Nifty 50 Benchmark (^NSEI)       : CAGR = {nifty_cagr:+.2f}%")
    print(f"   • 33-Stock Equal-Weight Portfolio : CAGR = {base_cagr:+.2f}% | Vol = {base_vol:.2f}% | Sharpe = {base_sharpe:.3f} | Max DD = {base_max_dd:.2f}%")

    # 2. Experiment A: 200 Monte Carlo Simulations with 20% Random Drops (7 stocks dropped, 26 kept)
    n_drop = int(np.ceil(0.20 * n_stocks))  # 7 stocks
    n_keep = n_stocks - n_drop              # 26 stocks
    n_sims = 200
    rng = np.random.RandomState(42)

    sim_cagrs = []
    sim_vols = []
    sim_sharpes = []
    sim_dds = []

    for sim in range(n_sims):
        dropped_idx = rng.choice(n_stocks, size=n_drop, replace=False)
        kept_syms = [EQUITY_33_SYMBOLS[i] for i in range(n_stocks) if i not in dropped_idx]
        sub_rets = ret_mat[kept_syms].mean(axis=1).values
        sub_cum = np.cumprod(1.0 + sub_rets)
        cagr_s = (((sub_cum[-1]) ** (1.0 / n_years)) - 1.0) * 100.0
        vol_s = float(np.std(sub_rets) * np.sqrt(252)) * 100.0
        pks_s = np.maximum.accumulate(sub_cum)
        dd_s = float(np.min((sub_cum - pks_s) / pks_s)) * 100.0
        sharpe_s = (cagr_s - 5.5) / vol_s if vol_s > 0 else 0.0

        sim_cagrs.append(cagr_s)
        sim_vols.append(vol_s)
        sim_sharpes.append(sharpe_s)
        sim_dds.append(dd_s)

    sim_cagrs = np.array(sim_cagrs)
    print(f"\n2. EXPERIMENT A: RANDOM 20% DROPS (200 Monte Carlo Simulations, 26 Stocks Each):")
    print(f"   • Mean CAGR            : {np.mean(sim_cagrs):+.2f}% (Std: {np.std(sim_cagrs):.2f}%)")
    print(f"   • 5th Percentile CAGR  : {np.percentile(sim_cagrs, 5):+.2f}%")
    print(f"   • Median CAGR (50th)   : {np.percentile(sim_cagrs, 50):+.2f}%")
    print(f"   • 95th Percentile CAGR : {np.percentile(sim_cagrs, 95):+.2f}%")
    print(f"   • Min / Max CAGR Range : [{np.min(sim_cagrs):+.2f}%, {np.max(sim_cagrs):+.2f}%]")
    print(f"   • Mean Sharpe Ratio    : {np.mean(sim_sharpes):.3f} (Range: [{np.min(sim_sharpes):.3f}, {np.max(sim_sharpes):.3f}])")
    print(f"   • Mean Max Drawdown    : {np.mean(sim_dds):.2f}%")

    # 3. Experiment B: Removal of Top-5 Ex-Post Winners Over Study Period
    individual_cum_returns = {sym: float(close_mat[sym].iloc[-1] / close_mat[sym].iloc[0] - 1.0) for sym in EQUITY_33_SYMBOLS}
    sorted_performers = sorted(individual_cum_returns.items(), key=lambda x: x[1], reverse=True)
    top_5_winners = [sym for sym, _ in sorted_performers[:5]]

    print(f"\n3. EXPERIMENT B: REMOVAL OF TOP-5 EX-POST PERFORMERS (HINDSIGHT DRAG):")
    print("   • Top-5 Ex-Post Equity Winners:")
    for rank, (sym, tot_ret) in enumerate(sorted_performers[:5], 1):
        stock_cagr = (((1.0 + tot_ret) ** (1.0 / n_years)) - 1.0) * 100.0
        print(f"     {rank}. {sym:<15}: Total Return = {tot_ret*100:+7.1f}% | CAGR = {stock_cagr:+5.2f}%")

    syms_no_top5 = [s for s in EQUITY_33_SYMBOLS if s not in top_5_winners]
    no_top5_rets = ret_mat[syms_no_top5].mean(axis=1).values
    no_top5_cum = np.cumprod(1.0 + no_top5_rets)
    no_top5_cagr = (((no_top5_cum[-1]) ** (1.0 / n_years)) - 1.0) * 100.0
    no_top5_vol = float(np.std(no_top5_rets) * np.sqrt(252)) * 100.0
    peaks_no = np.maximum.accumulate(no_top5_cum)
    no_top5_dd = float(np.min((no_top5_cum - peaks_no) / peaks_no)) * 100.0
    no_top5_sharpe = (no_top5_cagr - 5.5) / no_top5_vol if no_top5_vol > 0 else 0.0

    cagr_drag = base_cagr - no_top5_cagr

    print(f"\n   • 28-Stock Portfolio Performance (Top-5 Ex-Post Winners Removed):")
    print(f"     - Baseline 33-Stock CAGR       : {base_cagr:+.2f}%")
    print(f"     - No-Top-5 28-Stock CAGR       : {no_top5_cagr:+.2f}%")
    print(f"     - Top-5 Concentration Drag    : {cagr_drag:.2f}% per year")
    print(f"     - No-Top-5 Sharpe Ratio        : {no_top5_sharpe:.3f} (vs {base_sharpe:.3f} baseline)")
    print(f"     - No-Top-5 Max Drawdown        : {no_top5_dd:.2f}% (vs {base_max_dd:.2f}% baseline)")

    print("\n" + "=" * 100)
    print("   CRITICAL METHODOLOGICAL LIMITATION & SURVIVORSHIP DISCLAIMER:")
    print("   =============================================================")
    print("   IMPORTANT: Removing top performers (Experiment B) and random subsampling (Experiment A)")
    print("   measure portfolio concentration and robustness to winner exclusion, but they DO NOT")
    print("   measure TRUE SURVIVORSHIP BIAS from missing delisted or bankrupt companies.")
    print("   Because the 33-stock universe consists exclusively of firms that survived through 2026,")
    print("   any company that went bankrupt, merged under distress, or was delisted between 2016 and 2026")
    print("   (e.g., DHFL, Yes Bank, Reliance Communications, Sintex) is absent from the dataset.")
    print("   True survivorship-bias correction requires point-in-time constituent reconstruction.")
    print("=" * 100)

    conn.close()

if __name__ == "__main__":
    run_survivorship_audit_33()
