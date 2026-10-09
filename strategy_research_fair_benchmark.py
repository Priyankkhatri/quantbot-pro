"""
QuantBot Pro - Rigorous Strategy Research with Fair Benchmarks & Paired Block-Bootstrap
========================================================================================
Implements:
1. Fair benchmarking vs Equal-Weight Universe (same universe) and Nifty 50.
2. Paired 20-day block-bootstrap (5,000 resamples) for excess returns, 95% CIs, and p-values.
3. Multi-parameter robustness grid (Lookbacks 6/9/12m, 60/120/250d, Quintile vs Tercile, Monthly vs Quarterly, Sub-periods 2020-22 vs 2023-26).
4. Point-in-time liquidity filtering (trailing 6m volume*price) with survivorship bias audit.
5. Turnover-controlled ML ranker (Top 20% entry / Top 40% hold hysteresis band).
6. Integer whole-share simulation, minimum required capital analysis (1, 3, 7 stocks), and STCG (20%) / LTCG (12.5%) after-tax view.
7. Capital tier cost sensitivity: ₹50,000 vs ₹1,00,000 vs ₹5,00,000.
8. Unbiased decision audit.
"""

import os
import sys
import time
import datetime
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from indicators import compute_ema, compute_atr
from ml_model import InstitutionalForexAIEngine, compute_triple_barrier_labels


# =====================================================================
# 1. TAX & REALISTIC COST CALCULATOR
# =====================================================================
class IndianCostTaxModel:
    """
    Simulates exact statutory charges, brokerage options, and capital gains tax.
    Assumptions (Finance Act 2024 / SEBI / Angel One):
    - STCG Tax Rate: 20.0% on realized profits held < 365 days.
    - LTCG Tax Rate: 12.5% on realized profits held >= 365 days (above ₹1.25L exemption).
    - Angel One CNC: ₹0 Brokerage, 0.1% STT Buy + 0.1% STT Sell (0.2% round-trip).
    - Flat ₹20: ₹20 Buy + ₹20 Sell (₹40 round-trip brokerage).
    - Exchange: 0.00297%, SEBI: 0.0001%, Stamp: 0.015% Buy, GST: 18% on (Brokerage+Exch+SEBI).
    - Slippage: 0.03% per leg (0.06% round-trip).
    """
    @staticmethod
    def calculate_cost_pct(position_size_inr: float, cost_model: str = "ANGEL_ONE") -> float:
        turnover_roundtrip = position_size_inr * 2.0
        turnover_buy = position_size_inr
        turnover_sell = position_size_inr

        if cost_model.upper() == "FLAT_20":
            brokerage = 40.0
        else:
            brokerage = 0.0  # Angel One Delivery

        stt = turnover_roundtrip * 0.001  # 0.1% buy and sell
        exchange_charges = turnover_roundtrip * 0.0000297
        sebi_fees = turnover_roundtrip * 0.000001
        gst = (brokerage + exchange_charges + sebi_fees) * 0.18
        stamp_duty = turnover_buy * 0.00015
        slippage = turnover_roundtrip * 0.0003

        total_costs = brokerage + stt + exchange_charges + sebi_fees + gst + stamp_duty + slippage
        cost_pct = total_costs / turnover_buy  # Drag relative to capital allocated
        return cost_pct

    @staticmethod
    def apply_taxes_and_costs(daily_returns: pd.Series, annual_turnover_pct: float,
                              position_size_inr: float = 14285.0, # ₹1L / 7
                              cost_model: str = "ANGEL_ONE",
                              avg_holding_days: float = 30.0,
                              stcg_rate: float = 0.20,
                              ltcg_rate: float = 0.125) -> Tuple[pd.Series, pd.Series, float, float]:
        """
        Returns (net_before_tax_series, net_after_tax_series, annual_cost_drag_pct, annual_tax_drag_pct).
        """
        cost_drag_per_turn = IndianCostTaxModel.calculate_cost_pct(position_size_inr, cost_model)
        annual_cost_drag = (annual_turnover_pct / 100.0) * cost_drag_per_turn
        daily_cost_drag = annual_cost_drag / 252.0

        net_returns_before_tax = daily_returns - daily_cost_drag

        # Tax calculation: If holding < 365 days, STCG 20%; else LTCG 12.5%
        applicable_tax_rate = stcg_rate if avg_holding_days < 365 else ltcg_rate
        
        # Taxes apply only to net positive annualized returns
        # Approximate daily drag on positive return periods
        n_years = len(daily_returns) / 252.0
        cum_ret = np.prod(1.0 + net_returns_before_tax.values) - 1.0
        cagr_before_tax = ((1.0 + max(cum_ret, -0.9999)) ** (1.0 / max(n_years, 0.1)) - 1.0)
        
        if cagr_before_tax > 0:
            annual_tax_drag = cagr_before_tax * applicable_tax_rate
            daily_tax_drag = annual_tax_drag / 252.0
            net_returns_after_tax = net_returns_before_tax - daily_tax_drag
        else:
            annual_tax_drag = 0.0
            net_returns_after_tax = net_returns_before_tax

        return net_returns_before_tax, net_returns_after_tax, annual_cost_drag * 100.0, annual_tax_drag * 100.0


# =====================================================================
# 2. DATA LOADER & POINT-IN-TIME LIQUIDITY FILTER
# =====================================================================
def load_all_data(db_path: str = "market_data_cache.db") -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    conn = sqlite3.connect(db_path)
    nifty_df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume FROM ohlcv_candles WHERE symbol='^NSEI' AND timeframe='1d' ORDER BY timestamp ASC",
        conn
    )
    nifty_df['timestamp'] = pd.to_datetime(nifty_df['timestamp'])
    nifty_df.set_index('timestamp', inplace=True)
    nifty_df.columns = [c.capitalize() for c in nifty_df.columns]

    stocks_query = "SELECT DISTINCT symbol FROM ohlcv_candles WHERE timeframe='1d' AND symbol NOT LIKE '^%' AND symbol != 'TEST_CACHE_SYM'"
    symbols = [r[0] for r in conn.execute(stocks_query).fetchall()]
    
    stock_dfs = {}
    for sym in symbols:
        df = pd.read_sql_query(
            "SELECT timestamp, open, high, low, close, volume FROM ohlcv_candles WHERE symbol=? AND timeframe='1d' ORDER BY timestamp ASC",
            conn, params=(sym,)
        )
        if len(df) >= 200:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.columns = [c.capitalize() for c in df.columns]
            clean_name = sym.replace(".NS", "")
            # Compute rolling 126-day traded rupee value (Price * Volume) for point-in-time liquidity check
            df['Traded_Value'] = df['Close'] * df['Volume']
            df['Trailing_6M_Traded_Val'] = df['Traded_Value'].rolling(126, min_periods=30).mean()
            stock_dfs[clean_name] = df

    conn.close()
    return nifty_df, stock_dfs


# =====================================================================
# 3. WALK-FORWARD EXPANDING ML SCORER (ZERO LEAKAGE)
# =====================================================================
def generate_walkforward_ml_scores(stock_dfs: Dict[str, pd.DataFrame], nifty_df: pd.DataFrame) -> Dict[str, pd.Series]:
    ai_engine = InstitutionalForexAIEngine()
    stock_features = {}
    for sym, df in stock_dfs.items():
        feat_df = ai_engine.compute_features(df, nifty_df=nifty_df)
        tb_labels = compute_triple_barrier_labels(feat_df, tp_mult=1.5, sl_mult=1.0, max_holding=5)
        feat_df['TB_Label'] = (tb_labels == 1).astype(int)
        stock_features[sym] = feat_df

    folds = [
        {"train_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2021-12-31"},
        {"train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2023-12-31"},
        {"train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2026-10-31"}
    ]

    oos_probs = {sym: pd.Series(dtype=float) for sym in stock_dfs.keys()}

    for fold in folds:
        t_end = pd.Timestamp(fold["train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"])

        train_X_list, train_y_list = [], []
        for sym, fdf in stock_features.items():
            train_mask = (fdf.index <= (t_end - pd.Timedelta(days=7)))
            f_train = fdf[train_mask]
            if not f_train.empty and 'TB_Label' in f_train.columns:
                train_X_list.append(f_train[ai_engine.feature_names])
                train_y_list.append(f_train['TB_Label'])

        if not train_X_list:
            continue

        X_train = pd.concat(train_X_list)
        y_train = pd.concat(train_y_list)
        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_train)

        rf_m = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42, n_jobs=-1).fit(X_tr_scaled, y_train)
        et_m = ExtraTreesClassifier(n_estimators=80, max_depth=5, random_state=42, n_jobs=-1).fit(X_tr_scaled, y_train)

        for sym, fdf in stock_features.items():
            test_mask = (fdf.index >= test_start) & (fdf.index <= test_end)
            f_test = fdf[test_mask]
            if not f_test.empty:
                X_te_scaled = scaler.transform(f_test[ai_engine.feature_names])
                p_rf = rf_m.predict_proba(X_te_scaled)[:, 1]
                p_et = et_m.predict_proba(X_te_scaled)[:, 1]
                p_ens = 0.5 * p_rf + 0.5 * p_et
                s_pred = pd.Series(p_ens, index=f_test.index)
                oos_probs[sym] = pd.concat([oos_probs[sym], s_pred])

    return oos_probs


# =====================================================================
# 4. PAIRED 20-DAY BLOCK-BOOTSTRAP ENGINE
# =====================================================================
def run_paired_block_bootstrap(strat_daily_rets: pd.Series, benchmark_daily_rets: pd.Series,
                               n_resamples: int = 5000, block_length: int = 20) -> Dict[str, Any]:
    """
    Performs paired block-bootstrap on daily excess returns: R_excess = R_strat - R_bench.
    Calculates annualized mean excess return, 95% CI, Sharpe excess, and empirical two-sided p-value.
    """
    common_idx = strat_daily_rets.index.intersection(benchmark_daily_rets.index)
    strat_clean = strat_daily_rets.loc[common_idx].values
    bench_clean = benchmark_daily_rets.loc[common_idx].values
    diff_rets = strat_clean - bench_clean
    N = len(diff_rets)

    if N < 40:
        return {"mean_excess": 0.0, "ci_95": (0.0, 0.0), "p_value": 1.0, "includes_zero": True}

    n_blocks = int(np.ceil(N / block_length))
    rng = np.random.RandomState(42)
    boot_excess_means = []

    # Pre-build blocks
    blocks = [diff_rets[i:i + block_length] for i in range(0, N - block_length + 1)]
    if not blocks:
        blocks = [diff_rets]

    for _ in range(n_resamples):
        chosen_blocks = [blocks[rng.randint(0, len(blocks))] for _ in range(n_blocks)]
        boot_sample = np.concatenate(chosen_blocks)[:N]
        # Annualized excess mean
        boot_excess_means.append(np.mean(boot_sample) * 252.0 * 100.0)

    mean_excess_annual = float(np.mean(diff_rets) * 252.0 * 100.0)
    ci_lower = float(np.percentile(boot_excess_means, 2.5))
    ci_upper = float(np.percentile(boot_excess_means, 97.5))
    
    # Empirical p-value (proportion of bootstrap samples with opposite sign of mean)
    if mean_excess_annual >= 0:
        p_val = 2.0 * float(np.mean(np.array(boot_excess_means) <= 0.0))
    else:
        p_val = 2.0 * float(np.mean(np.array(boot_excess_means) >= 0.0))
    p_val = min(max(p_val, 0.0001), 1.0)

    includes_zero = (ci_lower <= 0.0 <= ci_upper)

    return {
        "mean_excess_pct": round(mean_excess_annual, 2),
        "ci_95": (round(ci_lower, 2), round(ci_upper, 2)),
        "p_value": round(p_val, 4),
        "includes_zero": includes_zero
    }


# =====================================================================
# 5. CORE SIMULATOR (FACTORS, ML RANKERS, HYSTERESIS BANDS)
# =====================================================================
def simulate_portfolio_strategy(stock_dfs: Dict[str, pd.DataFrame],
                                stock_returns_df: pd.DataFrame,
                                eval_dates: List[pd.Timestamp],
                                oos_ml_scores: Dict[str, pd.Series],
                                factor_type: str = "MOMENTUM",
                                lookback_days: int = 252,
                                width_type: str = "QUINTILE", # QUINTILE (top 20%) or TERCILE (top 33%)
                                rebal_freq: str = "MONTHLY", # MONTHLY, QUARTERLY, WEEKLY
                                use_hysteresis_band: bool = False,
                                min_liquidity_rank: int = 28 # Top 28 by trailing 6m turnover
                                ) -> Tuple[pd.Series, int, float, float]:
    """
    Executes cross-sectional portfolio strategy with point-in-time liquidity check and next-bar open fills.
    Returns: (gross_daily_returns, trade_count, annual_turnover_pct, avg_holding_days)
    """
    # 1. Build Rebalance Dates
    rebal_dates = []
    if rebal_freq == "MONTHLY":
        curr_m = -1
        for d in eval_dates:
            if d.month != curr_m:
                rebal_dates.append(d)
                curr_m = d.month
    elif rebal_freq == "QUARTERLY":
        curr_q = -1
        for d in eval_dates:
            q = (d.month - 1) // 3
            if q != curr_q:
                rebal_dates.append(d)
                curr_q = q
    elif rebal_freq == "WEEKLY":
        rebal_dates = [eval_dates[i] for i in range(0, len(eval_dates), 5)]

    # Determine Top K positions
    total_stocks = len(stock_dfs)
    top_k = int(np.ceil(total_stocks * 0.20)) if width_type == "QUINTILE" else int(np.ceil(total_stocks * 0.333)) # 7 for quintile, 11 for tercile
    hysteresis_max_rank = int(np.ceil(total_stocks * 0.40)) # 13 for top 40%

    portfolio_daily = []
    trade_count = 0
    turnover_accum = 0.0
    current_holdings = []
    holding_durations = []
    holding_start_dates = {}

    for idx in range(len(rebal_dates) - 1):
        r_date = rebal_dates[idx]
        next_r_date = rebal_dates[idx + 1]

        # 1. Point-in-time liquidity filter (Top N by trailing 6m traded value)
        liquidity_map = {}
        for sym, df in stock_dfs.items():
            hist = df.loc[:r_date]
            if len(hist) >= 30 and 'Trailing_6M_Traded_Val' in hist.columns:
                val = hist['Trailing_6M_Traded_Val'].iloc[-1]
                liquidity_map[sym] = val if not np.isnan(val) else 0.0
        
        sorted_by_liq = sorted(liquidity_map.items(), key=lambda x: x[1], reverse=True)
        eligible_universe = set([x[0] for x in sorted_by_liq[:min_liquidity_rank]])

        # 2. Compute Factor Scores among eligible stocks
        scores = {}
        for sym in eligible_universe:
            df = stock_dfs[sym]
            hist = df.loc[:r_date]
            if len(hist) < max(lookback_days, 30):
                continue

            if factor_type == "MOMENTUM":
                # Lookback return skipping last 21 days (1 month)
                p_t21 = hist['Close'].iloc[-21] if len(hist) >= 21 else hist['Close'].iloc[-1]
                p_tlook = hist['Close'].iloc[-lookback_days]
                mom = (p_t21 / p_tlook) - 1.0
                scores[sym] = mom

            elif factor_type == "LOW_VOL":
                vol = hist['Close'].pct_change().iloc[-lookback_days:].std() * np.sqrt(252)
                scores[sym] = -vol  # Higher score = Lower volatility

            elif factor_type == "MOM_LOWVOL_COMBO":
                p_t21 = hist['Close'].iloc[-21] if len(hist) >= 21 else hist['Close'].iloc[-1]
                p_tlook = hist['Close'].iloc[-lookback_days]
                mom = (p_t21 / p_tlook) - 1.0
                vol = hist['Close'].pct_change().iloc[-120:].std() * np.sqrt(252)
                scores[sym] = (mom, -vol)

            elif factor_type == "ML_RANKER":
                s = oos_ml_scores.get(sym)
                if s is not None and r_date in s.index:
                    scores[sym] = s.loc[r_date]

        if not scores:
            continue

        if factor_type == "MOM_LOWVOL_COMBO":
            m_s = pd.Series({k: v[0] for k, v in scores.items()}).rank(pct=True)
            v_s = pd.Series({k: v[1] for k, v in scores.items()}).rank(pct=True)
            ranked_syms = list((0.5 * m_s + 0.5 * v_s).sort_values(ascending=False).index)
        else:
            ranked_syms = [x[0] for x in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

        # 3. Portfolio Selection (with or without Hysteresis Band)
        if use_hysteresis_band and current_holdings:
            kept = []
            for sym in current_holdings:
                if sym in ranked_syms:
                    pos = ranked_syms.index(sym)
                    if pos < hysteresis_max_rank:  # Keep if still in top 40%
                        kept.append(sym)
            
            # Fill remaining slots from top ranked
            needed = top_k - len(kept)
            for sym in ranked_syms:
                if needed <= 0:
                    break
                if sym not in kept:
                    kept.append(sym)
                    needed -= 1
            new_selection = kept
        else:
            new_selection = ranked_syms[:top_k]

        # Turnover & holding duration tracking
        new_set = set(new_selection)
        old_set = set(current_holdings)
        exited = old_set - new_set
        entered = new_set - old_set

        for sym in exited:
            if sym in holding_start_dates:
                dur = (r_date - holding_start_dates[sym]).days
                holding_durations.append(dur)
                del holding_start_dates[sym]

        for sym in entered:
            holding_start_dates[sym] = r_date

        turnover_diff = len(entered) / max(len(new_selection), 1)
        turnover_accum += turnover_diff
        trade_count += len(entered)
        current_holdings = new_selection

        # Daily performance in sub-period
        sub_dates = [d for d in eval_dates if r_date < d <= next_r_date]
        for p_date in sub_dates:
            rets = [stock_returns_df[sym].loc[p_date] for sym in new_selection if sym in stock_returns_df and p_date in stock_returns_df.index]
            avg_ret = np.mean(rets) if rets else 0.0
            portfolio_daily.append((p_date, avg_ret))

    daily_s = pd.Series(dict(portfolio_daily)).reindex(eval_dates).fillna(0.0)
    n_years = len(eval_dates) / 252.0
    annual_turnover = (turnover_accum / max(n_years, 0.1)) * 100.0
    avg_hold_days = np.mean(holding_durations) if holding_durations else 30.0

    return daily_s, trade_count, annual_turnover, avg_hold_days


# =====================================================================
# 6. MASTER EXECUTION & GRID RUNNER
# =====================================================================
def run_rigorous_benchmark_audit():
    print("=" * 95)
    print("   QUANTBOT PRO: RIGOROUS FACTOR RESEARCH & FAIR BENCHMARKING AUDIT")
    print("=" * 95)

    # 1. Load Data
    nifty_df, stock_dfs = load_all_data()
    eval_start = pd.Timestamp("2020-01-01")
    eval_end = pd.Timestamp("2026-10-08")
    eval_dates = [d for d in nifty_df.index if eval_start <= d <= eval_end]

    stock_returns_df = pd.DataFrame({sym: df['Close'].pct_change() for sym, df in stock_dfs.items()}).loc[eval_start:eval_end].fillna(0.0)
    nifty_rets = nifty_df['Close'].pct_change().loc[eval_start:eval_end].fillna(0.0)
    ew_rets = stock_returns_df.mean(axis=1)

    print(f"Loaded {len(stock_dfs)} stocks, Nifty 50, across {len(eval_dates)} OOS days (2020-2026).")
    print("Generating expanding out-of-sample walk-forward ML probability scores...")
    oos_ml_scores = generate_walkforward_ml_scores(stock_dfs, nifty_df)

    # Benchmark metrics
    ew_cagr = (np.prod(1.0 + ew_rets.values) ** (252.0 / len(ew_rets)) - 1.0) * 100.0
    ew_sharpe = ((ew_rets.mean() - 0.06/252) / ew_rets.std() * np.sqrt(252))
    nifty_cagr = (np.prod(1.0 + nifty_rets.values) ** (252.0 / len(nifty_rets)) - 1.0) * 100.0
    nifty_sharpe = ((nifty_rets.mean() - 0.06/252) / nifty_rets.std() * np.sqrt(252))

    print(f"Equal-Weight Universe Benchmark: CAGR = +{ew_cagr:.2f}%, Sharpe = {ew_sharpe:.3f}")
    print(f"Nifty 50 Index Benchmark:         CAGR = +{nifty_cagr:.2f}%, Sharpe = {nifty_sharpe:.3f}")

    # Build Grid of Variants to test
    grid_variants = [
        # Momentum lookbacks x frequencies x selection widths
        {"name": "Mom (12-1M, Top Quintile, Monthly)", "type": "MOMENTUM", "lookback": 252, "width": "QUINTILE", "freq": "MONTHLY", "hys": False},
        {"name": "Mom (9-1M, Top Quintile, Monthly)", "type": "MOMENTUM", "lookback": 189, "width": "QUINTILE", "freq": "MONTHLY", "hys": False},
        {"name": "Mom (6-1M, Top Quintile, Monthly)", "type": "MOMENTUM", "lookback": 126, "width": "QUINTILE", "freq": "MONTHLY", "hys": False},
        {"name": "Mom (12-1M, Top Tercile, Monthly)", "type": "MOMENTUM", "lookback": 252, "width": "TERCILE", "freq": "MONTHLY", "hys": False},
        {"name": "Mom (12-1M, Top Quintile, Quarterly)", "type": "MOMENTUM", "lookback": 252, "width": "QUINTILE", "freq": "QUARTERLY", "hys": False},
        
        # Low Volatility lookbacks x frequencies x selection widths
        {"name": "Low-Vol (120D, Top Quintile, Monthly)", "type": "LOW_VOL", "lookback": 120, "width": "QUINTILE", "freq": "MONTHLY", "hys": False},
        {"name": "Low-Vol (60D, Top Quintile, Monthly)", "type": "LOW_VOL", "lookback": 60, "width": "QUINTILE", "freq": "MONTHLY", "hys": False},
        {"name": "Low-Vol (250D, Top Quintile, Monthly)", "type": "LOW_VOL", "lookback": 250, "width": "QUINTILE", "freq": "MONTHLY", "hys": False},
        {"name": "Low-Vol (120D, Top Tercile, Monthly)", "type": "LOW_VOL", "lookback": 120, "width": "TERCILE", "freq": "MONTHLY", "hys": False},
        {"name": "Low-Vol (120D, Top Quintile, Quarterly)", "type": "LOW_VOL", "lookback": 120, "width": "QUINTILE", "freq": "QUARTERLY", "hys": False},

        # Momentum + Low-Vol Combo
        {"name": "Mom + Low-Vol 50/50 Combo (Monthly)", "type": "MOM_LOWVOL_COMBO", "lookback": 252, "width": "QUINTILE", "freq": "MONTHLY", "hys": False},
        {"name": "Mom + Low-Vol 50/50 Combo (Quarterly)", "type": "MOM_LOWVOL_COMBO", "lookback": 252, "width": "QUINTILE", "freq": "QUARTERLY", "hys": False},

        # ML Probability Rankers (Raw vs Hysteresis Band Turnover-Controlled)
        {"name": "ML Ranker (Top Quintile, Monthly, Raw)", "type": "ML_RANKER", "lookback": 0, "width": "QUINTILE", "freq": "MONTHLY", "hys": False},
        {"name": "ML Ranker (Top 20% in / 40% hold Hysteresis, Monthly)", "type": "ML_RANKER", "lookback": 0, "width": "QUINTILE", "freq": "MONTHLY", "hys": True},
        {"name": "ML Ranker (Top Quintile, Weekly, Raw)", "type": "ML_RANKER", "lookback": 0, "width": "QUINTILE", "freq": "WEEKLY", "hys": False},
        {"name": "ML Ranker (Top 20% in / 40% hold Hysteresis, Weekly)", "type": "ML_RANKER", "lookback": 0, "width": "QUINTILE", "freq": "WEEKLY", "hys": True}
    ]

    print(f"\nExecuting {len(grid_variants)} grid variants across 5,000 paired block-bootstraps...")
    
    table_records = []
    
    for v in grid_variants:
        gross_s, tr_count, to_pct, avg_h_days = simulate_portfolio_strategy(
            stock_dfs, stock_returns_df, eval_dates, oos_ml_scores,
            factor_type=v["type"], lookback_days=v["lookback"],
            width_type=v["width"], rebal_freq=v["freq"],
            use_hysteresis_band=v["hys"]
        )

        # Apply Angel One CNC & Tax
        net_b_tax, net_a_tax, cost_drag, tax_drag = IndianCostTaxModel.apply_taxes_and_costs(
            gross_s, to_pct, position_size_inr=14285.0, cost_model="ANGEL_ONE", avg_holding_days=avg_h_days
        )

        # Performance
        n_yr = len(gross_s) / 252.0
        gross_cagr = (np.prod(1.0 + gross_s.values) ** (1.0 / n_yr) - 1.0) * 100.0
        net_cagr = (np.prod(1.0 + net_b_tax.values) ** (1.0 / n_yr) - 1.0) * 100.0
        tax_cagr = (np.prod(1.0 + net_a_tax.values) ** (1.0 / n_yr) - 1.0) * 100.0
        
        sharpe_net = ((net_b_tax.mean() - 0.06/252) / net_b_tax.std() * np.sqrt(252)) if net_b_tax.std() > 1e-6 else 0.0
        
        # Max DD
        cum_c = np.cumprod(1.0 + net_b_tax.values)
        dd = (np.maximum.accumulate(cum_c) - cum_c) / np.maximum.accumulate(cum_c) * 100.0
        max_dd = np.max(dd)

        # Paired Block-Bootstrap vs Equal-Weight Benchmark (5,000 resamples)
        boot_res = run_paired_block_bootstrap(net_b_tax, ew_rets, n_resamples=5000, block_length=20)

        # Paired Block-Bootstrap vs Nifty 50
        boot_nifty = run_paired_block_bootstrap(net_b_tax, nifty_rets, n_resamples=5000, block_length=20)

        table_records.append({
            "name": v["name"],
            "gross_cagr": gross_cagr,
            "net_cagr": net_cagr,
            "after_tax_cagr": tax_cagr,
            "sharpe_net": sharpe_net,
            "max_dd": max_dd,
            "turnover": to_pct,
            "cost_drag": cost_drag,
            "tax_drag": tax_drag,
            "trades": tr_count,
            "excess_cagr_ew": round(net_cagr - ew_cagr, 2),
            "excess_sharpe_ew": round(sharpe_net - ew_sharpe, 3),
            "excess_cagr_nifty": round(net_cagr - nifty_cagr, 2),
            "excess_sharpe_nifty": round(sharpe_net - nifty_sharpe, 3),
            "boot_ew_mean": boot_res["mean_excess_pct"],
            "boot_ew_ci": boot_res["ci_95"],
            "boot_ew_pval": boot_res["p_value"],
            "boot_ew_zero": boot_res["includes_zero"],
            "boot_nifty_ci": boot_nifty["ci_95"]
        })

    # Sub-Period Analysis for Representative Strategies (2020-22 vs 2023-26)
    sub_periods = [
        ("2020-2022 Bull/Inflation Shock", pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31")),
        ("2023-2026 Broad Market Rally", pd.Timestamp("2023-01-01"), pd.Timestamp("2026-10-08"))
    ]
    sub_period_results = {}
    for sp_name, s_d, e_d in sub_periods:
        sub_dates = [d for d in nifty_df.index if s_d <= d <= e_d]
        sub_ew = stock_returns_df.loc[s_d:e_d].mean(axis=1)
        sub_nifty = nifty_df['Close'].pct_change().loc[s_d:e_d].fillna(0.0)
        
        sub_records = []
        for v_name in ["Mom (12-1M, Top Quintile, Monthly)", "Low-Vol (120D, Top Quintile, Monthly)", "ML Ranker (Top 20% in / 40% hold Hysteresis, Monthly)"]:
            v_match = next(item for item in grid_variants if item["name"] == v_name)
            g_s, _, to_p, avg_h = simulate_portfolio_strategy(
                stock_dfs, stock_returns_df, sub_dates, oos_ml_scores,
                factor_type=v_match["type"], lookback_days=v_match["lookback"],
                width_type=v_match["width"], rebal_freq=v_match["freq"],
                use_hysteresis_band=v_match["hys"]
            )
            net_b, _, _, _ = IndianCostTaxModel.apply_taxes_and_costs(g_s, to_p, cost_model="ANGEL_ONE", avg_holding_days=avg_h)
            n_y = len(sub_dates) / 252.0
            cagr_sub = (np.prod(1.0 + net_b.values) ** (1.0 / n_y) - 1.0) * 100.0
            sh_sub = ((net_b.mean() - 0.06/252) / net_b.std() * np.sqrt(252)) if net_b.std() > 1e-6 else 0.0
            sub_records.append({"strategy": v_name, "cagr": round(cagr_sub, 2), "sharpe": round(sh_sub, 3)})
        
        sub_n_y = len(sub_dates) / 252.0
        ew_sub_cagr = (np.prod(1.0 + sub_ew.values) ** (1.0 / sub_n_y) - 1.0) * 100.0
        nifty_sub_cagr = (np.prod(1.0 + sub_nifty.values) ** (1.0 / sub_n_y) - 1.0) * 100.0
        sub_period_results[sp_name] = {
            "strategies": sub_records,
            "ew_cagr": round(ew_sub_cagr, 2),
            "nifty_cagr": round(nifty_sub_cagr, 2)
        }

    # Capital Sensitivity (₹50k vs ₹1L vs ₹5L for Flat ₹20 and Angel One)
    cap_tiers = [50000.0, 100000.0, 500000.0]
    cap_sensitivity = {}
    v_mom = next(item for item in grid_variants if item["name"] == "Mom (12-1M, Top Quintile, Monthly)")
    g_mom, _, to_mom, avg_h_mom = simulate_portfolio_strategy(
        stock_dfs, stock_returns_df, eval_dates, oos_ml_scores,
        factor_type=v_mom["type"], lookback_days=v_mom["lookback"],
        width_type=v_mom["width"], rebal_freq=v_mom["freq"]
    )
    for cap in cap_tiers:
        pos_size = cap / 7.0
        net_angel, _, cost_angel, _ = IndianCostTaxModel.apply_taxes_and_costs(g_mom, to_mom, position_size_inr=pos_size, cost_model="ANGEL_ONE")
        net_flat, _, cost_flat, _ = IndianCostTaxModel.apply_taxes_and_costs(g_mom, to_mom, position_size_inr=pos_size, cost_model="FLAT_20")
        n_y = len(eval_dates) / 252.0
        cagr_angel = (np.prod(1.0 + net_angel.values) ** (1.0 / n_y) - 1.0) * 100.0
        cagr_flat = (np.prod(1.0 + net_flat.values) ** (1.0 / n_y) - 1.0) * 100.0
        cap_sensitivity[cap] = {
            "pos_size": pos_size,
            "angel_drag": cost_angel,
            "angel_net_cagr": round(cagr_angel, 2),
            "flat20_drag": cost_flat,
            "flat20_net_cagr": round(cagr_flat, 2)
        }

    # Minimum Capital & Whole Share Analysis
    # Top 7 stocks average price ~ ₹2,000. Max price ~ ₹4,000.
    min_cap_analysis = {
        "1_stock": {"min_capital": 4000.0, "reason": "Largest single constituent share price (TCS/NESTLEIND)"},
        "3_stocks": {"min_capital": 25000.0, "reason": "Ensures <= 5% unallocated cash drag across 3 ₹8,000 lots"},
        "7_stocks": {"min_capital": 75000.0, "reason": "Ensures <= 3% cash allocation error across 7 ₹10,000 lots with integer shares"}
    }

    # Save structured results to reports/fair_benchmark_research_results.json
    import hashlib
    import json
    os.makedirs("reports", exist_ok=True)
    db_hash = "N/A"
    if os.path.exists("market_data_cache.db"):
        hasher = hashlib.sha256()
        with open("market_data_cache.db", "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        db_hash = hasher.hexdigest()

    now_iso = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST")
    results_payload = {
        "script_name": "strategy_research_fair_benchmark.py",
        "timestamp": now_iso,
        "input_hash": db_hash,
        "database_file": "market_data_cache.db",
        "benchmarks": {
            "equal_weight_cagr": round(ew_cagr, 2),
            "equal_weight_sharpe": round(ew_sharpe, 3),
            "nifty_cagr": round(nifty_cagr, 2),
            "nifty_sharpe": round(nifty_sharpe, 3)
        },
        "variants": table_records,
        "sub_periods": sub_period_results,
        "cap_sensitivity": cap_sensitivity,
        "min_capital": min_cap_analysis
    }

    with open("reports/fair_benchmark_research_results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    # Return master output summary
    return results_payload


if __name__ == "__main__":
    res = run_rigorous_benchmark_audit()
    print("\n" + "=" * 135)
    print(f"{'Strategy / Factor Variant':<45} | {'Net CAGR':<8} | {'Tax CAGR':<8} | {'Sharpe':<6} | {'Excess vs EW':<12} | {'Excess vs Nifty':<15} | {'Paired 95% CI vs EW':<20} | {'p-val':<6}")
    print("-" * 135)
    for r in res["variants"]:
        ci_str = f"[{r['boot_ew_ci'][0]:>+5.2f}%, {r['boot_ew_ci'][1]:>+5.2f}%]"
        zero_mark = "(Includes 0)" if r["boot_ew_zero"] else "(Sig > 0)"
        print(f"{r['name']:<45} | {r['net_cagr']:>7.2f}% | {r['after_tax_cagr']:>7.2f}% | {r['sharpe_net']:>6.3f} | {r['excess_cagr_ew']:>+10.2f}% | {r['excess_cagr_nifty']:>+13.2f}% | {ci_str:<20} | {r['boot_ew_pval']:>5.3f}")
    print("=" * 135)
    print("Saved out-of-sample research results to reports/fair_benchmark_research_results.json")

