"""
QuantBot Pro: Comprehensive Bootstrap & Sub-Period Audit
=========================================================
1. Paired stationary block-bootstrap (5,000 resamples):
   - Excess Return (%/yr) with 95% CI
   - Sharpe Ratio Difference (ΔSR) with 95% CI
   - Max Drawdown Difference (ΔMaxDD) with 95% CI
2. Strategies evaluated vs Benchmarks:
   - EQUAL_WEIGHT_4_ETFS vs NIFTYBEES
   - VOL_TARGET_10PCT_HAR vs NIFTY50 Buy & Hold
   - VOL_TARGET_12PCT_HAR vs NIFTY50 Buy & Hold
   - VOL_TARGET_15PCT_HAR vs NIFTY50 Buy & Hold
   - TOP_1_MOMENTUM_ROTATION_M vs NIFTYBEES
   - BH_GOLDBEES vs NIFTYBEES
3. Sub-period evaluations:
   - 2016-2019 (Pre-COVID)
   - 2020-2022 (COVID Crash & Post-COVID Recovery)
   - 2023-2026 (Recent Bull Market)
4. Deflated Sharpe Ratio with cumulative K = 174.
"""

import os
import sys
import math
import sqlite3
import pandas as pd
import numpy as np
from scipy import stats

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from strategy_research_etf_vol_overlay import (
    load_etf_and_index_data,
    run_etf_rotation_experiments,
    run_volatility_targeting_experiments,
    ResearchAnalytics
)

def compute_sharpe_and_maxdd(daily_rets: np.ndarray, rf_annual: float = 0.055) -> Tuple[float, float]:
    if len(daily_rets) == 0:
        return 0.0, 0.0
    n_years = len(daily_rets) / 252.0
    cum = np.cumprod(1.0 + daily_rets)
    cagr = (((cum[-1]) ** (1.0 / max(0.01, n_years))) - 1.0) * 100.0 if cum[-1] > 0 else -100.0
    vol = float(np.std(daily_rets) * np.sqrt(252)) * 100.0
    sharpe = (cagr - rf_annual * 100.0) / vol if vol > 0 else 0.0
    peaks = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum - peaks) / peaks)) * 100.0
    return round(sharpe, 3), round(max_dd, 2)

def paired_advanced_block_bootstrap(strat_daily_rets: np.ndarray,
                                    bench_daily_rets: np.ndarray,
                                    block_size: int = 20,
                                    n_boot: int = 5000,
                                    seed: int = 42) -> Dict[str, Any]:
    n = min(len(strat_daily_rets), len(bench_daily_rets))
    s_ret = strat_daily_rets[:n]
    b_ret = bench_daily_rets[:n]

    excess_daily = s_ret - b_ret
    mean_excess_ann = float(np.mean(excess_daily)) * 252.0 * 100.0

    orig_s_sharpe, orig_s_dd = compute_sharpe_and_maxdd(s_ret)
    orig_b_sharpe, orig_b_dd = compute_sharpe_and_maxdd(b_ret)
    orig_delta_sharpe = orig_s_sharpe - orig_b_sharpe
    orig_delta_dd = orig_s_dd - orig_b_dd

    rng = np.random.RandomState(seed)
    num_blocks = int(np.ceil(n / block_size))

    boot_excess = np.zeros(n_boot)
    boot_delta_sharpe = np.zeros(n_boot)
    boot_delta_dd = np.zeros(n_boot)

    for b in range(n_boot):
        start_indices = rng.randint(0, max(1, n - block_size + 1), size=num_blocks)
        resamp_s = []
        resamp_b = []
        for s in start_indices:
            resamp_s.extend(s_ret[s:s + block_size])
            resamp_b.extend(b_ret[s:s + block_size])
        
        sample_s = np.array(resamp_s[:n])
        sample_b = np.array(resamp_b[:n])

        boot_excess[b] = np.mean(sample_s - sample_b) * 252.0 * 100.0
        sh_s, dd_s = compute_sharpe_and_maxdd(sample_s)
        sh_b, dd_b = compute_sharpe_and_maxdd(sample_b)
        boot_delta_sharpe[b] = sh_s - sh_b
        boot_delta_dd[b] = dd_s - dd_b

    p_val_excess = float(np.mean(boot_excess <= 0.0)) if mean_excess_ann > 0 else float(np.mean(boot_excess >= 0.0))
    p_val_excess = min(1.0, p_val_excess * 2.0)

    return {
        "n_days": n,
        "mean_excess_ann_pct": round(mean_excess_ann, 2),
        "excess_ci": (round(float(np.percentile(boot_excess, 2.5)), 2), round(float(np.percentile(boot_excess, 97.5)), 2)),
        "p_value_excess": round(p_val_excess, 4),
        "delta_sharpe": round(orig_delta_sharpe, 3),
        "delta_sharpe_ci": (round(float(np.percentile(boot_delta_sharpe, 2.5)), 3), round(float(np.percentile(boot_delta_sharpe, 97.5)), 3)),
        "delta_max_dd": round(orig_delta_dd, 2),
        "delta_max_dd_ci": (round(float(np.percentile(boot_delta_dd, 2.5)), 2), round(float(np.percentile(boot_delta_dd, 97.5)), 2))
    }

def run_audit():
    print("=" * 100)
    print("   QUANTBOT PRO: ADVANCED BOOTSTRAP (SHARPE & DRAWDOWN CIs) & SUB-PERIOD AUDIT")
    print("=" * 100)

    data_dict = load_etf_and_index_data()
    etf_tier_res, _ = run_etf_rotation_experiments(data_dict)
    vol_eval, vol_strat_res = run_volatility_targeting_experiments(data_dict)

    std_res = etf_tier_res["₹100,000"]
    nifty_bench_daily = std_res["BH_NIFTYBEES"]["daily_returns"]
    nifty_oos_bench = vol_strat_res["NIFTY50_BUY_HOLD_BENCHMARK"]["daily_returns"]

    k_cum = 174

    test_pairs = [
        ("EQUAL_WEIGHT_4_ETFS vs NIFTYBEES", std_res["EQUAL_WEIGHT_4_ETFS"]["daily_returns"], std_res["EQUAL_WEIGHT_4_ETFS"]["sharpe"], nifty_bench_daily),
        ("VOL_TARGET_10PCT_HAR vs NIFTY50", vol_strat_res["VOL_TARGET_10PCT_HAR"]["daily_returns"], vol_strat_res["VOL_TARGET_10PCT_HAR"]["sharpe"], nifty_oos_bench),
        ("VOL_TARGET_12PCT_HAR vs NIFTY50", vol_strat_res["VOL_TARGET_12PCT_HAR"]["daily_returns"], vol_strat_res["VOL_TARGET_12PCT_HAR"]["sharpe"], nifty_oos_bench),
        ("VOL_TARGET_15PCT_HAR vs NIFTY50", vol_strat_res["VOL_TARGET_15PCT_HAR"]["daily_returns"], vol_strat_res["VOL_TARGET_15PCT_HAR"]["sharpe"], nifty_oos_bench),
        ("TOP_1_MOMENTUM_ROTATION_M vs NIFTYBEES", std_res["TOP_1_MOMENTUM_ROTATION_M"]["daily_returns"], std_res["TOP_1_MOMENTUM_ROTATION_M"]["sharpe"], nifty_bench_daily),
        ("BH_GOLDBEES vs NIFTYBEES", std_res["BH_GOLDBEES"]["daily_returns"], std_res["BH_GOLDBEES"]["sharpe"], nifty_bench_daily)
    ]

    all_trial_sharpes = [res["sharpe"] for res in std_res.values()] + [res["sharpe"] for res in vol_strat_res.values()]

    print("\n----------------------------------------------------------------------------------------------------")
    print("TABLE A: PAIRED 20-DAY BLOCK-BOOTSTRAP (5,000 RESAMPLES) WITH SHARPE & DRAWDOWN CIs")
    print(f"Cumulative Multiple-Testing Penalty: K = {k_cum}")
    print("----------------------------------------------------------------------------------------------------")
    print(f"{'Strategy Pair':<36} | {'Excess (%/yr)':<13} | {'Δ Sharpe (95% CI)':<22} | {'Δ Max DD (95% CI)':<20} | {'DSR'}")
    print("----------------------------------------------------------------------------------------------------")

    for label, s_s, s_sh, b_s in test_pairs:
        common = s_s.index.intersection(b_s.index)
        s_arr = s_s.loc[common].values
        b_arr = b_s.loc[common].values

        boot_res = paired_advanced_block_bootstrap(s_arr, b_arr, block_size=20, n_boot=5000)
        dsr_info = ResearchAnalytics.calculate_deflated_sharpe_ratio(
            estimated_sharpe_annual=s_sh,
            all_trial_sharpes=all_trial_sharpes,
            n_days=len(s_arr),
            k_cumulative_trials=k_cum
        )

        exc_str = f"{boot_res['mean_excess_ann_pct']:+.2f}%"
        sh_ci_str = f"{boot_res['delta_sharpe']:+.3f} [{boot_res['delta_sharpe_ci'][0]:+.2f}, {boot_res['delta_sharpe_ci'][1]:+.2f}]"
        dd_ci_str = f"{boot_res['delta_max_dd']:+.2f}% [{boot_res['delta_max_dd_ci'][0]:+.1f}%, {boot_res['delta_max_dd_ci'][1]:+.1f}%]"

        print(f"{label:<36} | {exc_str:<13} | {sh_ci_str:<22} | {dd_ci_str:<20} | {dsr_info['deflated_sharpe_prob']:.2f}")

    print("----------------------------------------------------------------------------------------------------")

    # Sub-Period Splits
    sub_periods = {
        "Sub-Period 1 (2016-2019 Pre-COVID)": ("2016-10-10", "2019-12-31"),
        "Sub-Period 2 (2020-2022 COVID Shock & Bull)": ("2020-01-01", "2022-12-31"),
        "Sub-Period 3 (2023-2026 Recent Rally)": ("2023-01-01", "2026-10-09")
    }

    print("\n----------------------------------------------------------------------------------------------------")
    print("TABLE B: SUB-PERIOD PERFORMANCE & EXCESS RETURN BREAKDOWN")
    print("----------------------------------------------------------------------------------------------------")
    print(f"{'Strategy Name':<28} | {'Sub-Period':<35} | {'CAGR (%)':<9} | {'Sharpe':<7} | {'Max DD (%)':<10} | {'Excess (%/yr)'}")
    print("----------------------------------------------------------------------------------------------------")

    eval_strats = [
        ("NIFTYBEES (Benchmark)", std_res["BH_NIFTYBEES"]["daily_returns"]),
        ("EQUAL_WEIGHT_4_ETFS", std_res["EQUAL_WEIGHT_4_ETFS"]["daily_returns"]),
        ("BH_GOLDBEES", std_res["BH_GOLDBEES"]["daily_returns"]),
        ("TOP_1_MOMENTUM_ROTATION_M", std_res["TOP_1_MOMENTUM_ROTATION_M"]["daily_returns"]),
        ("VOL_TARGET_12PCT_HAR", vol_strat_res["VOL_TARGET_12PCT_HAR"]["daily_returns"])
    ]

    for strat_name, rets_series in eval_strats:
        for sub_name, (start_d, end_d) in sub_periods.items():
            sub_rets = rets_series.loc[(rets_series.index >= start_d) & (rets_series.index <= end_d)]
            sub_bench = nifty_bench_daily.loc[(nifty_bench_daily.index >= start_d) & (nifty_bench_daily.index <= end_d)]
            
            if len(sub_rets) > 20:
                common_sub = sub_rets.index.intersection(sub_bench.index)
                s_arr = sub_rets.loc[common_sub].values
                b_arr = sub_bench.loc[common_sub].values

                sh, dd = compute_sharpe_and_maxdd(s_arr)
                n_y = len(s_arr) / 252.0
                cum_s = np.cumprod(1.0 + s_arr)
                cagr_s = (((cum_s[-1]) ** (1.0 / max(0.01, n_y))) - 1.0) * 100.0 if cum_s[-1] > 0 else -100.0
                
                exc_ann = float(np.mean(s_arr - b_arr)) * 252.0 * 100.0

                print(f"{strat_name:<28} | {sub_name:<35} | {cagr_s:<+9.2f} | {sh:<7.2f} | {dd:<10.2f} | {exc_ann:<+10.2f}%")
        print("." * 100)

    # Geometric compounding validation across sub-periods
    print("\n----------------------------------------------------------------------------------------------------")
    print("SUB-PERIOD GEOMETRIC COMPOUNDING ASSERTION VERIFICATION:")
    print("----------------------------------------------------------------------------------------------------")
    for strat_name, rets_series in eval_strats:
        full_arr = rets_series.values
        full_growth = float(np.prod(1.0 + full_arr))
        
        comp_growth = 1.0
        sub_cagrs = []
        for sub_name, (start_d, end_d) in sub_periods.items():
            sub_s = rets_series.loc[(rets_series.index >= start_d) & (rets_series.index <= end_d)]
            sub_g = float(np.prod(1.0 + sub_s.values))
            comp_growth *= sub_g
            n_y = len(sub_s) / 252.0
            sub_cagr = (((sub_g) ** (1.0 / max(0.01, n_y))) - 1.0) * 100.0 if sub_g > 0 else -100.0
            sub_cagrs.append(sub_cagr)
            
        growth_err_pct = abs(comp_growth - full_growth) / full_growth * 100.0
        assert growth_err_pct < 0.05, f"Discrepancy for {strat_name}: comp={comp_growth:.4f}, full={full_growth:.4f}, err={growth_err_pct:.4f}%"
        print(f"  ✓ [{strat_name:<26}] Full Growth = {full_growth:.4f}x | Compounded Sub-Periods = {comp_growth:.4f}x (Err: {growth_err_pct:.4f}%) -> PASS")
    print("----------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    run_audit()
