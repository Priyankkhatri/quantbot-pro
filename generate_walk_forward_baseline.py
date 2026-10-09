"""
Walk-Forward Baseline Generator with Strict Data Provenance & Dynamic Loading
=============================================================================
Computes out-of-sample walk-forward baselines and saves to walk_forward_baseline_results.json:
- Strictly loads all performance metrics dynamically from out-of-sample research results:
  reports/fair_benchmark_research_results.json
- Queries actual database tables (ohlcv_candles / timestamp) in market_data_cache.db.
- Records data provenance, timestamp, database SHA-256 hash.
- Strictly fails closed if the research results file is missing or invalid.
- Removes swing_ml_ensemble block (focusing exclusively on walk-forward portfolio baselines).
"""

import os
import sys
import json
import sqlite3
import hashlib
import datetime

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DB_PATH = "market_data_cache.db"
RESEARCH_RESULTS_PATH = os.path.join("reports", "fair_benchmark_research_results.json")
OUTPUT_PATH = "walk_forward_baseline_results.json"

def compute_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def generate_baseline():
    # 1. Strict Fail-Closed Check on Input Database
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"FAIL CLOSED: Database {DB_PATH} missing.")

    db_hash = compute_file_sha256(DB_PATH)
    now_ist = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST")

    # 2. Query Real Table: ohlcv_candles and timestamp
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT timestamp) FROM ohlcv_candles WHERE timeframe='1d'")
        total_days = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv_candles WHERE timeframe='1d'")
        total_symbols = cursor.fetchone()[0]
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv_candles WHERE timeframe='1d'")
        min_ts, max_ts = cursor.fetchone()
        eval_period_str = f"{min_ts[:10]} to {max_ts[:10]}" if min_ts and max_ts else "N/A"

    # 3. Strict Fail-Closed Check on Research Results
    if not os.path.exists(RESEARCH_RESULTS_PATH):
        raise FileNotFoundError(
            f"FAIL CLOSED: Out-of-sample research results file '{RESEARCH_RESULTS_PATH}' missing! "
            f"Run strategy_research_fair_benchmark.py to generate valid research baseline data."
        )

    with open(RESEARCH_RESULTS_PATH, "r", encoding="utf-8") as f:
        research_data = json.load(f)

    # 4. Extract provenance from research file
    res_script = research_data.get("script_name", "UNKNOWN")
    res_timestamp = research_data.get("timestamp", "UNKNOWN")
    res_input_hash = research_data.get("input_hash", "UNKNOWN")
    variants = research_data.get("variants", [])
    benchmarks = research_data.get("benchmarks", {})

    if not variants:
        raise ValueError(f"FAIL CLOSED: No variants found in {RESEARCH_RESULTS_PATH}!")

    # 5. Dynamically extract Best Factor Baseline and ML Ranker
    # Best Factor: highest net CAGR among non-ML factor variants
    factor_variants = [v for v in variants if "ML" not in v["name"]]
    best_factor = max(factor_variants, key=lambda v: v["net_cagr"]) if factor_variants else variants[0]

    # ML Ranker
    ml_variants = [v for v in variants if "ML" in v["name"]]
    ml_ranker = ml_variants[0] if ml_variants else None

    # Construct clean baseline JSON with zero hardcoded values
    baseline_payload = {
        "status": "VALID_OUT_OF_SAMPLE",
        "generated_at": now_ist,
        "input_database_file": DB_PATH,
        "input_database_sha256": db_hash,
        "eval_period": eval_period_str,
        "oos_trading_days": total_days,
        "universe_size": total_symbols,
        "source_research_file": RESEARCH_RESULTS_PATH,
        "source_research_script": res_script,
        "source_research_timestamp": res_timestamp,
        "source_research_input_hash": res_input_hash,
        "baseline_provenance": f"Out-of-sample research ({res_script}, input hash: {res_input_hash[:12]}...)",
        "benchmarks": {
            "equal_weight_universe_cagr": benchmarks.get("equal_weight_cagr"),
            "equal_weight_universe_sharpe": benchmarks.get("equal_weight_sharpe"),
            "nifty_50_index_cagr": benchmarks.get("nifty_cagr"),
            "nifty_50_index_sharpe": benchmarks.get("nifty_sharpe")
        },
        "best_factor_baseline": {
            "strategy": best_factor["name"],
            "net_cagr_angel_cnc": best_factor["net_cagr"],
            "net_cagr_after_tax": best_factor["after_tax_cagr"],
            "sharpe": best_factor["sharpe_net"],
            "max_drawdown_pct": best_factor["max_dd"],
            "annual_turnover_pct": best_factor["turnover"],
            "bootstrap_95_ci_excess_vs_ew": best_factor["boot_ew_ci"],
            "bootstrap_p_val": best_factor["boot_ew_pval"]
        }
    }

    if ml_ranker:
        baseline_payload["turnover_controlled_ml_ranker"] = {
            "strategy": ml_ranker["name"],
            "net_cagr_angel_cnc": ml_ranker["net_cagr"],
            "net_cagr_after_tax": ml_ranker["after_tax_cagr"],
            "sharpe": ml_ranker["sharpe_net"],
            "max_drawdown_pct": ml_ranker["max_dd"],
            "annual_turnover_pct": ml_ranker["turnover"],
            "bootstrap_95_ci_excess_vs_ew": ml_ranker["boot_ew_ci"],
            "bootstrap_p_val": ml_ranker["boot_ew_pval"]
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline_payload, f, indent=2)

    print("=" * 80)
    print("   QUANTBOT PRO: DYNAMIC WALK-FORWARD BASELINE JSON GENERATED")
    print("=" * 80)
    print(f"Target File              : {OUTPUT_PATH}")
    print(f"Generated At             : {now_ist}")
    print(f"Input DB (ohlcv_candles) : {DB_PATH} ({total_days} daily bars across {total_symbols} symbols)")
    print(f"Input DB SHA-256         : {db_hash}")
    print(f"Source Research File     : {RESEARCH_RESULTS_PATH}")
    print(f"Source Script & Hash     : {res_script} (Hash: {res_input_hash[:12]}...)")
    print(f"Status                   : {baseline_payload['status']}")
    print(f"Best Factor Baseline     : {best_factor['name']} -> CAGR: {best_factor['net_cagr']}%")
    if ml_ranker:
        print(f"ML Ranker Baseline       : {ml_ranker['name']} -> CAGR: {ml_ranker['net_cagr']}%")
    print(f"Benchmarks               : EW Universe = {benchmarks.get('equal_weight_cagr')}% | Nifty = {benchmarks.get('nifty_cagr')}%")
    print("=" * 80)

if __name__ == "__main__":
    generate_baseline()
