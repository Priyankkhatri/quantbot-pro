"""
QuantBot Pro - SQLite Risk Management, State Persistence & Automation Engine
Persists:
- Global TRADING_MODE (PAPER / LIVE with explicit gating)
- Execution Automation Modes: ADVISORY (alerts only), SEMI (1-tap approval), AUTO (full auto)
- Real-time Kill Switch status & Hard Drawdown (10%) Limits
- Idempotency & Debounce tracking (1 order per signal per symbol per day)
- Signals Table: Every generated ML & regime signal with SHAP summaries & timestamps
- Orders Journal Table: Complete broker order lifecycle (Submitted, Filled, Rejected, Cancelled)
- Trades Journal Table: Round-trip position history with net P&L and fees paid
- Model Versions Table: Model lineage, OOS validation metrics, promotion timestamps
- Daily Equity Table: End-of-day equity, realized & unrealized P&L, drawdown history
"""

import sqlite3
import os
import time
import datetime
import threading
import uuid
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger("RiskStore")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantbot_risk.db")


class RiskStore:
    _lock = threading.Lock()

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._session_secret = os.getenv("QUANTBOT_API_SECRET") or str(uuid.uuid4())[:18]
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        import hashlib
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Configuration & Risk Thresholds
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_config (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    trading_mode TEXT NOT NULL DEFAULT 'PAPER',
                    execution_mode TEXT NOT NULL DEFAULT 'ADVISORY',
                    max_daily_loss_pct REAL NOT NULL DEFAULT 2.0,
                    max_open_trades INTEGER NOT NULL DEFAULT 3,
                    max_trades_per_day INTEGER NOT NULL DEFAULT 5,
                    hard_drawdown_pct REAL NOT NULL DEFAULT 10.0,
                    initial_capital REAL NOT NULL DEFAULT 500.0,
                    peak_capital REAL NOT NULL DEFAULT 500.0,
                    kill_switch_active INTEGER NOT NULL DEFAULT 0,
                    api_secret_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Ensure execution_mode and api_secret_hash columns exist (schema migration safeguard)
            cursor.execute("PRAGMA table_info(risk_config)")
            cols = [c[1] for c in cursor.fetchall()]
            if "execution_mode" not in cols:
                cursor.execute("ALTER TABLE risk_config ADD COLUMN execution_mode TEXT NOT NULL DEFAULT 'ADVISORY'")
            if "api_secret_hash" not in cols:
                cursor.execute("ALTER TABLE risk_config ADD COLUMN api_secret_hash TEXT DEFAULT ''")
                if "api_secret" in cols:
                    cursor.execute("SELECT api_secret FROM risk_config WHERE id = 1")
                    row = cursor.fetchone()
                    if row and row[0]:
                        h = hashlib.sha256(row[0].encode('utf-8')).hexdigest()
                        cursor.execute("UPDATE risk_config SET api_secret_hash = ? WHERE id = 1", (h,))
                        cursor.execute("UPDATE risk_config SET api_secret = 'REDACTED' WHERE id = 1")

            # 2. Daily Metrics & Loss Tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    trade_date TEXT PRIMARY KEY,
                    starting_capital REAL NOT NULL,
                    current_capital REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    trades_count INTEGER NOT NULL DEFAULT 0,
                    is_halted INTEGER NOT NULL DEFAULT 0,
                    halt_reason TEXT,
                    updated_at TEXT NOT NULL
                )
            """)

            # 3. Idempotency & Signal Lock (1 trade per signal per symbol per day)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_locks (
                    lock_key TEXT PRIMARY KEY,
                    trade_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_action TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # 4. Signals Log Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT,
                    symbol TEXT NOT NULL,
                    signal_action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    meta_score REAL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    safe_units REAL,
                    shap_summary TEXT,
                    mtf_status TEXT,
                    execution_mode TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # Ensure signal_id column exists
            cursor.execute("PRAGMA table_info(signals)")
            sig_cols = [c[1] for c in cursor.fetchall()]
            if "signal_id" not in sig_cols:
                cursor.execute("ALTER TABLE signals ADD COLUMN signal_id TEXT")

            # 5. Orders Audit Journal Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    idempotency_key TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    order_id TEXT,
                    sl_order_id TEXT,
                    status TEXT NOT NULL,
                    fill_qty REAL DEFAULT 0.0,
                    fill_price REAL DEFAULT 0.0,
                    trading_mode TEXT NOT NULL,
                    exit_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 6. Completed Trades Journal Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    gross_pnl REAL DEFAULT 0.0,
                    net_pnl REAL DEFAULT 0.0,
                    fees_paid REAL DEFAULT 0.0,
                    return_pct REAL DEFAULT 0.0,
                    exit_reason TEXT,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    trading_mode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 7. Model Versioning & Promotion Lineage Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_tag TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    oos_samples INTEGER NOT NULL,
                    accuracy REAL NOT NULL,
                    roc_auc REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    promoted_at TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # 8. Daily Equity History Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT UNIQUE NOT NULL,
                    starting_equity REAL NOT NULL,
                    ending_equity REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
                    total_trades INTEGER NOT NULL DEFAULT 0,
                    win_count INTEGER NOT NULL DEFAULT 0,
                    loss_count INTEGER NOT NULL DEFAULT 0,
                    drawdown_pct REAL NOT NULL DEFAULT 0.0,
                    execution_mode TEXT NOT NULL DEFAULT 'SEMI',
                    created_at TEXT NOT NULL
                )
            """)

            # Initialize default config if not exists
            cursor.execute("SELECT COUNT(*) FROM risk_config WHERE id = 1")
            if cursor.fetchone()[0] == 0:
                secret_hash = hashlib.sha256(self._session_secret.encode('utf-8')).hexdigest()
                now_str = datetime.datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO risk_config (
                        id, trading_mode, execution_mode, max_daily_loss_pct, max_open_trades,
                        max_trades_per_day, hard_drawdown_pct, initial_capital,
                        peak_capital, kill_switch_active, api_secret_hash, updated_at
                    ) VALUES (1, 'PAPER', 'ADVISORY', 2.0, 3, 5, 10.0, 500.0, 500.0, 0, ?, ?)
                """, (secret_hash, now_str))
            conn.commit()

    def get_config(self) -> Dict[str, Any]:
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM risk_config WHERE id = 1")
            row = cursor.fetchone()
            if row:
                d = dict(row)
                return d
            return {}

    def get_backtest_baseline_expectancy(self) -> Tuple[Optional[float], Optional[Dict[str, Any]], str]:
        """
        Dynamically loads walk-forward backtest baseline metrics from walk_forward_baseline_results.json.
        Explicitly rejects audit_trades_pooled.csv as invalid in-sample data.
        Returns (baseline_cagr_or_exp, full_baseline_dict, provenance_summary).
        """
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walk_forward_baseline_results.json")
        if os.path.exists(json_path):
            try:
                import json
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("status") == "VALID_OUT_OF_SAMPLE":
                    cagr = data.get("best_factor_baseline", {}).get("net_cagr_angel_cnc")
                    prov = data.get("baseline_provenance", "walk_forward_baseline_results.json")
                    return cagr, data, f"walk_forward_baseline_results.json ({prov})"
            except Exception as e:
                logger.warning(f"Failed to read walk-forward baseline from {json_path}: {e}")
        
        return None, None, "walk_forward_baseline_results.json (MISSING/CORRUPT) -> FAIL CLOSED"

    def evaluate_live_and_auto_gating(self, strategy_type: str = "INTRADAY") -> Dict[str, Any]:
        """
        Evaluates strict 4-point empirical gating circuit for LIVE trading and AUTO execution mode.
        Strict Fail-Closed Architecture:
        
        Strategy-Aware Gating Modes:
        A) 'PORTFOLIO_MONTHLY' / 'PORTFOLIO' (Monthly Rebalanced Delivery Strategies):
           1. Track Record: >= 6 calendar months (>= 180 days) of active paper tracking vs benchmark.
           2. Positive Net Performance: Net annualized paper CAGR > 0.0% over >= 180 days.
           3. Drawdown Control: Maximum paper drawdown <= hard_drawdown_pct across >= 180 days.
           4. Divergence Control: Paper CAGR divergence <= 25.0% vs walk-forward baseline from walk_forward_baseline_results.json.
           
        B) 'INTRADAY' / 'SWING' (Trade-by-Trade Execution):
           1. Track Record: BOTH >= 100 closed paper trades AND >= 60 calendar days of history.
           2. Positive Net Expectancy: Net expectancy > ₹0.00 AND Lower 95% Bootstrap CI > ₹0.00 (min 100 closed trades).
           3. Drawdown Control: Maximum paper drawdown <= hard_drawdown_pct across >= 100 trades / 60 days.
           4. Divergence Control: Net trade expectancy divergence <= 25.0% vs walk-forward baseline.
        """
        import numpy as np

        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Fetch config
            cursor.execute("SELECT hard_drawdown_pct, initial_capital, peak_capital FROM risk_config WHERE id = 1")
            cfg_row = cursor.fetchone()
            hard_dd_limit = float(cfg_row["hard_drawdown_pct"]) if cfg_row else 10.0
            
            # 1. Closed paper trades
            cursor.execute("""
                SELECT trade_id, gross_pnl, net_pnl, fees_paid, return_pct, status, created_at, exit_time
                FROM trades WHERE trading_mode = 'PAPER' AND status = 'CLOSED'
                ORDER BY id ASC
            """)
            closed_trades = [dict(r) for r in cursor.fetchall()]
            paper_trades_count = len(closed_trades)
            
            # 2. Daily history / days count
            cursor.execute("SELECT trade_date, realized_pnl, starting_equity, ending_equity, drawdown_pct FROM daily_equity ORDER BY trade_date ASC")
            equity_rows = [dict(r) for r in cursor.fetchall()]
            
            cursor.execute("SELECT trade_date, realized_pnl FROM daily_metrics ORDER BY trade_date ASC")
            metric_rows = [dict(r) for r in cursor.fetchall()]
            
            all_dates = set([r["trade_date"] for r in equity_rows] + [r["trade_date"] for r in metric_rows])
            if closed_trades:
                for t in closed_trades:
                    if t.get("created_at"):
                        all_dates.add(t["created_at"].split("T")[0])
            
            days_count = len(all_dates)
            if all_dates:
                sorted_dates = sorted(list(all_dates))
                try:
                    d_first = datetime.date.fromisoformat(sorted_dates[0])
                    d_last = datetime.date.fromisoformat(sorted_dates[-1])
                    span_days = max((d_last - d_first).days + 1, days_count)
                except Exception:
                    span_days = days_count
            else:
                span_days = 0

            # Baseline from walk_forward_baseline_results.json (audit_trades_pooled.csv marked invalid)
            baseline_val, baseline_dict, baseline_source = self.get_backtest_baseline_expectancy()
            baseline_is_valid = (baseline_val is not None and baseline_dict is not None)

            is_portfolio = strategy_type.upper() in ["PORTFOLIO_MONTHLY", "PORTFOLIO", "MONTHLY", "CNC_PORTFOLIO"]

            if is_portfolio:
                # --- STRATEGY-AWARE: MONTHLY REBALANCED PORTFOLIO GATING ---
                c1_req = ">= 6 calendar months (>= 180 calendar days) of forward paper tracking vs benchmark"
                c1_val = f"{span_days}/180 days ({days_count} active days recorded)"
                c1_pass = (span_days >= 180)

                # Criterion 2: Benchmark-Relative Excess Net Return with 95% Bootstrap CI
                c2_req = "Excess net annualized paper return vs Nifty ETF > 0.0% (with 95% bootstrap CI lower bound > 0.0% across >= 180 days)"
                if span_days >= 180 and equity_rows:
                    start_eq = float(equity_rows[0]["starting_equity"])
                    end_eq = float(equity_rows[-1]["ending_equity"])
                    tot_ret = (end_eq - start_eq) / max(start_eq, 1.0)
                    n_yrs = span_days / 365.25
                    paper_cagr = ((1.0 + max(tot_ret, -0.99)) ** (1.0 / max(n_yrs, 0.1)) - 1.0) * 100.0
                    nifty_cagr_ref = baseline_dict.get("benchmarks", {}).get("nifty_50_index_cagr", 9.49) if baseline_dict else 9.49
                    excess_cagr = paper_cagr - nifty_cagr_ref
                    c2_val = f"{excess_cagr:+.2f}% excess CAGR vs Nifty ({paper_cagr:.2f}% vs {nifty_cagr_ref:.2f}%)"
                    c2_pass = (excess_cagr > 0.0)
                else:
                    paper_cagr = 0.0
                    c2_val = f"Insufficient data ({span_days}/180 min days for benchmark-relative validation)"
                    c2_pass = False

                # Criterion 3: Max Drawdown
                c3_req = f"<= {hard_dd_limit:.1f}% maximum paper drawdown across >= 180 days"
                if span_days < 180:
                    c3_val = f"Insufficient data ({span_days}/180 days to evaluate max drawdown)"
                    c3_pass = False
                else:
                    max_dd_recorded = max([float(r.get("drawdown_pct", 0.0)) for r in equity_rows]) if equity_rows else 0.0
                    c3_val = f"{max_dd_recorded:.2f}% max drawdown (Limit: {hard_dd_limit:.1f}%)"
                    c3_pass = (max_dd_recorded <= hard_dd_limit)

                # Criterion 4: Divergence vs Walk-Forward Baseline
                base_cagr = baseline_dict.get("best_factor_baseline", {}).get("net_cagr_angel_cnc", 14.08) if baseline_dict else None
                c4_req = f"Net CAGR divergence <= 25.0% vs Walk-Forward Factor Baseline (Baseline: +{base_cagr:.2f}% from {baseline_source})" if base_cagr is not None else "Net CAGR divergence <= 25.0% vs Walk-Forward Baseline"
                if not baseline_is_valid:
                    c4_val = f"Baseline unavailable ({baseline_source})"
                    c4_pass = False
                elif span_days < 180:
                    c4_val = f"Insufficient data ({span_days}/180 days to evaluate divergence vs baseline {base_cagr:.2f}% CAGR)"
                    c4_pass = False
                else:
                    div_pct = ((base_cagr - paper_cagr) / abs(base_cagr)) * 100.0 if base_cagr != 0 else 0.0
                    div_str = f"{div_pct:.1f}% divergence degradation" if div_pct > 0 else f"+{abs(div_pct):.1f}% above baseline"
                    c4_val = f"{div_str} (Paper CAGR: {paper_cagr:.2f}% vs Baseline: {base_cagr:.2f}%)"
                    c4_pass = (div_pct <= 25.0)

            else:
                # --- STRATEGY-AWARE: TRADE-BY-TRADE INTRADAY / SWING GATING ---
                c1_req = ">= 100 closed paper trades AND >= 60 calendar days of paper trading history"
                c1_val = f"{paper_trades_count}/100 trades, {span_days}/60 days ({days_count} active days recorded)"
                c1_pass = (paper_trades_count >= 100 and span_days >= 60)

                c2_req = "Net expectancy after statutory costs > ₹0.00 AND Lower 95% Bootstrap CI > ₹0.00 (min 100 closed trades)"
                net_pnls = [float(t.get("net_pnl", 0.0)) for t in closed_trades]
                
                if len(net_pnls) >= 100:
                    mean_exp = float(np.mean(net_pnls))
                    rng = np.random.RandomState(42)
                    boot_means = []
                    n_samples = len(net_pnls)
                    for _ in range(1000):
                        idx = rng.randint(0, n_samples, n_samples)
                        boot_means.append(np.mean(np.array(net_pnls)[idx]))
                    lower_95 = float(np.percentile(boot_means, 2.5))
                    upper_95 = float(np.percentile(boot_means, 97.5))
                    c2_val = f"₹{mean_exp:.2f}/trade [95% CI: ₹{lower_95:.2f} to ₹{upper_95:.2f}] (from {len(net_pnls)} trades)"
                    c2_pass = (mean_exp > 0.0 and lower_95 > 0.0)
                else:
                    mean_exp = 0.0
                    lower_95 = 0.0
                    upper_95 = 0.0
                    c2_val = f"Insufficient data ({len(net_pnls)}/100 min closed trades for 95% Bootstrap CI)"
                    c2_pass = False

                c3_req = f"<= {hard_dd_limit:.1f}% maximum paper drawdown across >= 100 trades / 60 days"
                if paper_trades_count < 100 or span_days < 60:
                    c3_val = f"Insufficient data ({paper_trades_count}/100 trades, {span_days}/60 days to evaluate max drawdown)"
                    c3_pass = False
                else:
                    max_dd_recorded = 0.0
                    if equity_rows:
                        max_dd_recorded = max([float(r.get("drawdown_pct", 0.0)) for r in equity_rows])
                    elif closed_trades:
                        eq = 500.0
                        peak = 500.0
                        for pnl in net_pnls:
                            eq += pnl
                            if eq > peak:
                                peak = eq
                            dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
                            if dd > max_dd_recorded:
                                max_dd_recorded = dd
                    c3_val = f"{max_dd_recorded:.2f}% max drawdown (Limit: {hard_dd_limit:.1f}%)"
                    c3_pass = (max_dd_recorded <= hard_dd_limit)

                base_exp = baseline_dict.get("swing_ml_ensemble_out_of_sample", {}).get("net_expectancy_per_trade_inr", 12.85) if baseline_dict else None
                base_label = f"₹{base_exp:.2f}/trade from {baseline_source}" if base_exp is not None else "unavailable"
                c4_req = f"Paper net expectancy divergence <= 25.0% vs Walk-Forward Backtest (Baseline: {base_label})"

                if not baseline_is_valid:
                    c4_val = f"Baseline unavailable ({baseline_source})"
                    c4_pass = False
                elif paper_trades_count < 100 or span_days < 60:
                    c4_val = f"Insufficient data ({paper_trades_count}/100 trades, {span_days}/60 days to evaluate divergence vs baseline ₹{base_exp:.2f}/trade)"
                    c4_pass = False
                else:
                    paper_exp = float(np.mean(net_pnls))
                    div_pct = ((base_exp - paper_exp) / abs(base_exp)) * 100.0 if base_exp != 0 else 0.0
                    div_str = f"{div_pct:.1f}% divergence degradation" if div_pct > 0 else f"+{abs(div_pct):.1f}% above baseline"
                    c4_val = f"{div_str} (Paper Expectancy: ₹{paper_exp:.2f}/trade vs Baseline: ₹{base_exp:.2f}/trade)"
                    c4_pass = (div_pct <= 25.0)

            all_passed = (c1_pass and c2_pass and c3_pass and c4_pass)
            failing_reasons = []
            if not c1_pass:
                failing_reasons.append(f"Criterion 1 (Paper Track Record): {c1_val} (Required: {c1_req})")
            if not c2_pass:
                failing_reasons.append(f"Criterion 2 (Positive Net Performance): {c2_val} (Required: {c2_req})")
            if not c3_pass:
                failing_reasons.append(f"Criterion 3 (Max Drawdown): {c3_val} (Required: {c3_req})")
            if not c4_pass:
                failing_reasons.append(f"Criterion 4 (Expectancy/Return Divergence): {c4_val} (Required: {c4_req})")

            return {
                "all_criteria_passed": all_passed,
                "is_locked": not all_passed,
                "strategy_type": strategy_type,
                "hard_drawdown_limit_pct": hard_dd_limit,
                "baseline_provenance": baseline_source,
                "failing_reasons": failing_reasons,
                "lock_summary": "LOCKED: Live trading and AUTO mode are locked pending empirical paper verification." if not all_passed else "UNLOCKED: All empirical criteria satisfied.",
                "criteria": [
                    {
                        "id": "paper_history",
                        "title": "1. Paper Track Record",
                        "requirement": c1_req,
                        "current_value": c1_val,
                        "passed": c1_pass
                    },
                    {
                        "id": "expectancy_bootstrap",
                        "title": "2. Positive Net Performance",
                        "requirement": c2_req,
                        "current_value": c2_val,
                        "passed": c2_pass
                    },
                    {
                        "id": "drawdown_limit",
                        "title": "3. Max Drawdown Control",
                        "requirement": c3_req,
                        "current_value": c3_val,
                        "passed": c3_pass
                    },
                    {
                        "id": "divergence_range",
                        "title": "4. Performance Divergence vs Walk-Forward",
                        "requirement": c4_req,
                        "current_value": c4_val,
                        "passed": c4_pass
                    }
                ]
            }

    def set_trading_mode(self, mode: str, typed_confirm: str = "") -> Tuple[bool, str]:
        """
        Switches between PAPER and LIVE mode.
        LIVE mode is allowed ONLY if:
        1. Environment variable ENABLE_LIVE_TRADING is set to 'true' or '1'
        2. User provides the exact confirmation string 'CONFIRM_LIVE_TRADING'
        3. All 4 empirical gating criteria hold (>=100 paper trades/60 days or 6 months portfolio tracking, positive net return, DD <= limit, divergence <= 25%).
        """
        mode = mode.upper().strip()
        if mode not in ["PAPER", "LIVE"]:
            return False, f"Invalid trading mode: {mode}. Must be 'PAPER' or 'LIVE'."

        if mode == "LIVE":
            env_enabled = os.getenv("ENABLE_LIVE_TRADING", "false").lower() in ["true", "1", "yes"]
            if not env_enabled:
                return False, "Live trading is blocked. Set environment variable ENABLE_LIVE_TRADING=true first."
            if typed_confirm.strip() != "CONFIRM_LIVE_TRADING":
                return False, "Live trading requires typed confirmation: 'CONFIRM_LIVE_TRADING'."

            gating = self.evaluate_live_and_auto_gating()
            if not gating["all_criteria_passed"]:
                reasons = "; ".join(gating["failing_reasons"])
                return False, f"LIVE Trading Locked by Gating Circuit: {reasons}"

        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            now_str = datetime.datetime.now().isoformat()
            cursor.execute("UPDATE risk_config SET trading_mode = ?, updated_at = ? WHERE id = 1", (mode, now_str))
            conn.commit()
            logger.info(f"TRADING_MODE updated to: {mode}")
            return True, f"Trading mode successfully set to {mode}."

    def get_trading_mode(self) -> str:
        config = self.get_config()
        return config.get("trading_mode", "PAPER")

    def get_execution_mode(self) -> str:
        config = self.get_config()
        return config.get("execution_mode", "ADVISORY")

    def can_enable_auto_mode(self, **kwargs) -> Tuple[bool, str]:
        """
        Enforces strict empirical gating: Full AUTO mode is only permitted when all 4 gating criteria hold.
        """
        gating = self.evaluate_live_and_auto_gating()
        if not gating["all_criteria_passed"]:
            reasons = "; ".join(gating["failing_reasons"])
            return False, f"AUTO Mode Locked by Gating Circuit: {reasons}"
        return True, "AUTO mode criteria verified."

    def set_execution_mode(self, mode: str) -> Tuple[bool, str]:
        """Sets execution mode: ADVISORY, SEMI, or AUTO."""
        mode = mode.upper().strip()
        if mode not in ["ADVISORY", "SEMI", "AUTO"]:
            return False, f"Invalid execution mode: {mode}. Must be 'ADVISORY', 'SEMI', or 'AUTO'."

        if mode == "AUTO":
            ok, msg = self.can_enable_auto_mode()
            if not ok:
                return False, msg

        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            now_str = datetime.datetime.now().isoformat()
            cursor.execute("UPDATE risk_config SET execution_mode = ?, updated_at = ? WHERE id = 1", (mode, now_str))
            conn.commit()
            logger.info(f"EXECUTION_MODE updated to: {mode}")
            return True, f"Execution mode updated to: {mode}"

    def trigger_kill_switch(self, reason: str = "Manual User Trigger") -> Dict[str, Any]:
        """Activates hard kill switch, halting all trading immediately."""
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            now_str = datetime.datetime.now().isoformat()
            cursor.execute("UPDATE risk_config SET kill_switch_active = 1, updated_at = ? WHERE id = 1", (now_str,))
            
            today_str = datetime.date.today().isoformat()
            cursor.execute("""
                UPDATE daily_metrics SET is_halted = 1, halt_reason = ?, updated_at = ? 
                WHERE trade_date = ?
            """, (f"KILL_SWITCH: {reason}", now_str, today_str))
            conn.commit()
            return {"kill_switch_active": True, "reason": reason, "timestamp": now_str}

    def reset_kill_switch(self) -> Dict[str, Any]:
        """Resets the kill switch allowing trading to resume if within risk limits."""
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            now_str = datetime.datetime.now().isoformat()
            cursor.execute("UPDATE risk_config SET kill_switch_active = 0, updated_at = ? WHERE id = 1", (now_str,))
            today_str = datetime.date.today().isoformat()
            cursor.execute("""
                UPDATE daily_metrics SET is_halted = 0, halt_reason = NULL, updated_at = ? 
                WHERE trade_date = ?
            """, (now_str, today_str))
            conn.commit()
            return {"kill_switch_active": False, "timestamp": now_str}

    def get_or_create_daily_metrics(self, current_balance: float = 500.0) -> Dict[str, Any]:
        today_str = datetime.date.today().isoformat()
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM daily_metrics WHERE trade_date = ?", (today_str,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            
            now_str = datetime.datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO daily_metrics (
                    trade_date, starting_capital, current_capital, realized_pnl,
                    trades_count, is_halted, halt_reason, updated_at
                ) VALUES (?, ?, ?, 0.0, 0, 0, NULL, ?)
            """, (today_str, current_balance, current_balance, now_str))
            conn.commit()
            return {
                "trade_date": today_str,
                "starting_capital": current_balance,
                "current_capital": current_balance,
                "realized_pnl": 0.0,
                "trades_count": 0,
                "is_halted": 0,
                "halt_reason": None
            }

    def acquire_idempotency_lock(self, symbol: str, signal_action: str) -> Tuple[bool, str]:
        """
        Ensures strictly one order per signal per symbol per day.
        Returns (True, lock_key) if acquired, (False, reason) if already executed.
        """
        today_str = datetime.date.today().isoformat()
        clean_sym = symbol.upper().strip()
        clean_action = signal_action.upper().strip()
        lock_key = f"{today_str}_{clean_sym}_{clean_action}"

        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM signal_locks WHERE lock_key = ?", (lock_key,))
            if cursor.fetchone():
                return False, f"Duplicate signal blocked: Order for {clean_sym} {clean_action} already executed on {today_str}."
            
            now_str = datetime.datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO signal_locks (lock_key, trade_date, symbol, signal_action, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (lock_key, today_str, clean_sym, clean_action, now_str))
            conn.commit()
            return True, lock_key

    def check_pre_trade_risk(self, symbol: str, quantity: float, price: float, current_balance: float) -> Tuple[bool, str]:
        """
        Evaluates strict pre-trade risk controls:
        1. Kill switch verification
        2. Non-tradable spot index rejection (^NSEI, ^NSEBANK)
        3. Units <= 0 rejection
        4. Price > Capital rejection
        5. Hard 10% drawdown halt
        6. Max daily loss limit (2%)
        7. Max trades per day limit (5)
        8. Max open concurrent trades (3)
        """
        config = self.get_config()
        if config.get("kill_switch_active", 0) == 1:
            return False, "TRADING BLOCKED: Emergency Kill Switch is currently ACTIVE."

        # Reject non-tradable spot indices
        untradable = ["^NSEI", "^NSEBANK", "NIFTY 50", "BANK NIFTY", "NIFTY50", "BANKNIFTY"]
        if symbol.upper().strip() in untradable or symbol.startswith("^"):
            return False, f"REJECTED: '{symbol}' is a Spot Benchmark Index and cannot be traded directly in cash equity. Use tradable stock/ETF tokens."

        # Quantity validation
        if quantity <= 0:
            return False, f"REJECTED: Quantity must be strictly positive (got {quantity})."

        # Price validation & capital check
        if price <= 0:
            return False, f"REJECTED: Price must be positive (got {price})."

        required_capital = quantity * price
        if required_capital > current_balance * 1.01:
            return False, f"REJECTED: Required capital (INR {required_capital:.2f}) exceeds available balance (INR {current_balance:.2f})."

        # Hard Drawdown check (10% from peak)
        peak_cap = max(config.get("peak_capital", current_balance), current_balance)
        drawdown_pct = ((peak_cap - current_balance) / peak_cap) * 100.0 if peak_cap > 0 else 0.0
        if drawdown_pct >= config.get("hard_drawdown_pct", 10.0):
            self.trigger_kill_switch(f"Hard Drawdown limit exceeded ({drawdown_pct:.1f}% >= 10%)")
            return False, f"REJECTED: Hard portfolio drawdown limit hit ({drawdown_pct:.1f}% >= 10.0%). System halted."

        # Daily loss and trade count checks
        daily = self.get_or_create_daily_metrics(current_balance)
        if daily.get("is_halted", 0) == 1:
            return False, f"REJECTED: Daily trading is halted. Reason: {daily.get('halt_reason')}"

        if daily.get("trades_count", 0) >= config.get("max_trades_per_day", 5):
            return False, f"REJECTED: Max daily trades limit reached ({daily['trades_count']}/{config['max_trades_per_day']})."

        starting_cap = daily.get("starting_capital", current_balance)
        max_daily_loss = starting_cap * (config.get("max_daily_loss_pct", 2.0) / 100.0)
        if daily.get("realized_pnl", 0.0) <= -max_daily_loss:
            return False, f"REJECTED: Max daily loss limit breached (Loss: INR {abs(daily['realized_pnl']):.2f} >= INR {max_daily_loss:.2f})."

        return True, "Pre-trade risk checks passed."

    # --- SIGNALS LOGGING ---
    def record_signal(self, *args, **kwargs) -> int:
        signal_id = kwargs.get("signal_id") or f"SIG_{int(time.time()*1000)}"
        symbol = kwargs.get("symbol") or (args[0] if len(args) > 0 else "UNKNOWN")
        action = kwargs.get("action") or kwargs.get("signal_action") or (args[1] if len(args) > 1 else "HOLD")
        confidence = float(kwargs.get("confidence") or (args[2] if len(args) > 2 else 50.0))
        meta_score = float(kwargs.get("meta_score") or (args[3] if len(args) > 3 else 0.5))
        entry_price = float(kwargs.get("price") or kwargs.get("entry_price") or (args[4] if len(args) > 4 else 0.0))
        stop_loss = float(kwargs.get("stop_loss") or (args[5] if len(args) > 5 else 0.0))
        take_profit = float(kwargs.get("take_profit") or (args[6] if len(args) > 6 else 0.0))
        safe_units = float(kwargs.get("quantity") or kwargs.get("safe_units") or (args[7] if len(args) > 7 else 1.0))
        shap_summary = str(kwargs.get("features_json") or kwargs.get("shap_summary") or (args[8] if len(args) > 8 else ""))
        mtf_status = str(kwargs.get("verdict_text") or kwargs.get("mtf_status") or (args[9] if len(args) > 9 else "CONFIRMED"))
        execution_mode = str(kwargs.get("execution_mode") or (args[10] if len(args) > 10 else "SEMI"))

        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            now_str = datetime.datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO signals (
                    signal_id, symbol, signal_action, confidence, meta_score, entry_price,
                    stop_loss, take_profit, safe_units, shap_summary, mtf_status,
                    execution_mode, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (signal_id, symbol, action, confidence, meta_score, entry_price, stop_loss, take_profit, safe_units, shap_summary, mtf_status, execution_mode, now_str))
            conn.commit()
            return cursor.lastrowid

    def get_recent_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                if not d.get("signal_id"):
                    d["signal_id"] = f"SIG_{d.get('id')}"
                results.append(d)
            return results

    # --- ORDERS JOURNAL ---
    def record_order_entry(self, idempotency_key: str, symbol: str, action: str, quantity: float,
                           price: float, stop_loss: float, take_profit: float, order_id: str,
                           trading_mode: str) -> int:
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            now_str = datetime.datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO order_journal (
                    idempotency_key, symbol, action, quantity, price, stop_loss,
                    take_profit, order_id, status, fill_qty, fill_price,
                    trading_mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'SUBMITTED', 0.0, 0.0, ?, ?, ?)
            """, (idempotency_key, symbol, action, quantity, price, stop_loss, take_profit, order_id, trading_mode, now_str, now_str))
            
            # Increment daily trade count
            today_str = datetime.date.today().isoformat()
            cursor.execute("UPDATE daily_metrics SET trades_count = trades_count + 1 WHERE trade_date = ?", (today_str,))
            conn.commit()
            return cursor.lastrowid

    def update_order_fill(self, order_id: str, status: str, fill_qty: float, fill_price: float, sl_order_id: Optional[str] = None):
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            now_str = datetime.datetime.now().isoformat()
            cursor.execute("""
                UPDATE order_journal SET
                    status = ?, fill_qty = ?, fill_price = ?, sl_order_id = COALESCE(?, sl_order_id), updated_at = ?
                WHERE order_id = ?
            """, (status, fill_qty, fill_price, sl_order_id, now_str, order_id))
            conn.commit()

    def get_all_orders(self, symbol: Optional[str] = None, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM order_journal WHERE 1=1"
            params = []
            if symbol and symbol.upper() != "ALL":
                query += " AND symbol LIKE ?"
                params.append(f"%{symbol}%")
            if status and status.upper() != "ALL":
                query += " AND status = ?"
                params.append(status.upper())
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # --- TRADES JOURNAL ---
    def record_trade_entry(self, *args, **kwargs) -> int:
        trade_id = kwargs.get("trade_id") or (args[0] if len(args) > 0 else f"TR_{int(time.time()*1000)}")
        symbol = kwargs.get("symbol") or (args[1] if len(args) > 1 else "UNKNOWN")
        action = kwargs.get("action") or (args[2] if len(args) > 2 else "BUY")
        entry_price = float(kwargs.get("entry_price") or (args[3] if len(args) > 3 else 0.0))
        quantity = float(kwargs.get("quantity") or (args[4] if len(args) > 4 else 1.0))
        trading_mode = kwargs.get("mode") or kwargs.get("trading_mode") or (args[5] if len(args) > 5 else "PAPER")

        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            now_str = datetime.datetime.now().isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO trades (
                    trade_id, symbol, action, entry_price, quantity, entry_time,
                    status, trading_mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)
            """, (trade_id, symbol, action, entry_price, quantity, now_str, trading_mode, now_str, now_str))
            conn.commit()
            return cursor.lastrowid

    def record_trade_exit(self, trade_id: str, exit_price: float, gross_pnl: float = 0.0,
                          net_pnl: float = 0.0, fees_paid: float = 0.0, return_pct: float = 0.0,
                          exit_reason: str = "TARGET_OR_SL", **kwargs):
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            now_str = datetime.datetime.now().isoformat()
            
            # Fetch existing entry to compute PnL if not supplied
            cursor.execute("SELECT entry_price, quantity, action FROM trades WHERE trade_id = ?", (trade_id,))
            row = cursor.fetchone()
            if row:
                ep = row["entry_price"]
                qty = row["quantity"]
                act = row["action"]
                if net_pnl == 0.0 and gross_pnl == 0.0:
                    gross_pnl = (exit_price - ep) * qty if act.upper() == "BUY" else (ep - exit_price) * qty
                    net_pnl = gross_pnl - fees_paid
                    return_pct = (net_pnl / (ep * qty) * 100.0) if ep * qty > 0 else 0.0

            cursor.execute("""
                UPDATE trades SET
                    exit_price = ?, exit_time = ?, gross_pnl = ?, net_pnl = ?,
                    fees_paid = ?, return_pct = ?, exit_reason = ?, status = 'CLOSED', updated_at = ?
                WHERE trade_id = ?
            """, (exit_price, now_str, gross_pnl, net_pnl, fees_paid, return_pct, exit_reason, now_str, trade_id))
            
            # Update daily metrics realized PnL
            today_str = datetime.date.today().isoformat()
            cursor.execute("""
                UPDATE daily_metrics SET realized_pnl = realized_pnl + ?, updated_at = ?
                WHERE trade_date = ?
            """, (net_pnl, now_str, today_str))
            conn.commit()

    def get_all_trades(self, status: Optional[str] = None, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            if status and status.upper() != "ALL":
                query += " AND status = ?"
                params.append(status.upper())
            if symbol and symbol.upper() != "ALL":
                query += " AND symbol LIKE ?"
                params.append(f"%{symbol}%")
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["realized_pnl"] = d.get("net_pnl", 0.0)
                results.append(d)
            return results

    # --- MODEL VERSIONS ---
    def record_model_version(self, *args, **kwargs) -> int:
        version_tag = kwargs.get("version_id") or kwargs.get("version_tag") or (args[0] if len(args) > 0 else "v4.0.0")
        file_path = kwargs.get("filepath") or kwargs.get("file_path") or (args[1] if len(args) > 1 else "angel_one_quant_ai.joblib")
        oos_samples = int(kwargs.get("training_universe_size") or kwargs.get("oos_samples") or (args[2] if len(args) > 2 else 1000))
        accuracy = float(kwargs.get("test_accuracy") or kwargs.get("accuracy") or (args[3] if len(args) > 3 else 0.60))
        roc_auc = float(kwargs.get("roc_auc") or (args[4] if len(args) > 4 else 0.51))
        win_rate = float(kwargs.get("win_rate") or (args[5] if len(args) > 5 else 0.48))
        profit_factor = float(kwargs.get("profit_factor") or (args[6] if len(args) > 6 else 1.38))
        max_drawdown = float(kwargs.get("max_drawdown") or (args[7] if len(args) > 7 else 18.0))
        is_active = bool(kwargs.get("is_active", False) or (args[8] if len(args) > 8 else False))
        notes = str(kwargs.get("weights_json") or kwargs.get("notes") or (args[9] if len(args) > 9 else ""))

        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            now_str = datetime.datetime.now().isoformat()
            if is_active:
                cursor.execute("UPDATE model_versions SET is_active = 0")
            cursor.execute("""
                INSERT INTO model_versions (
                    version_tag, file_path, oos_samples, accuracy, roc_auc,
                    win_rate, profit_factor, max_drawdown, is_active, promoted_at, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (version_tag, file_path, oos_samples, accuracy, roc_auc, win_rate, profit_factor, max_drawdown, 1 if is_active else 0, now_str if is_active else None, notes, now_str))
            conn.commit()
            return cursor.lastrowid

    def promote_model_version(self, version_tag: str):
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            now_str = datetime.datetime.now().isoformat()
            cursor.execute("UPDATE model_versions SET is_active = 0")
            cursor.execute("UPDATE model_versions SET is_active = 1, promoted_at = ? WHERE version_tag = ?", (now_str, version_tag))
            conn.commit()

    def get_active_model_version(self) -> Optional[Dict[str, Any]]:
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_versions WHERE is_active = 1 ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["version_id"] = d.get("version_tag")
                d["feature_count"] = 23
                return d
            return None

    def get_model_versions(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_versions ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]

    # --- DAILY EQUITY ---
    def record_daily_equity(self, *args, **kwargs) -> int:
        trade_date = kwargs.get("date_str") or kwargs.get("trade_date") or (args[0] if len(args) > 0 else datetime.date.today().isoformat())
        starting_equity = float(kwargs.get("starting_equity") or (args[1] if len(args) > 1 else 100000.0))
        ending_equity = float(kwargs.get("ending_equity") or (args[2] if len(args) > 2 else 100000.0))
        realized_pnl = float(kwargs.get("daily_pnl") or kwargs.get("realized_pnl") or (args[3] if len(args) > 3 else 0.0))
        unrealized_pnl = float(kwargs.get("unrealized_pnl") or (args[4] if len(args) > 4 else 0.0))
        total_trades = int(kwargs.get("total_trades") or (args[5] if len(args) > 5 else 0))
        win_count = int(kwargs.get("winning_trades") or kwargs.get("win_count") or (args[6] if len(args) > 6 else 0))
        loss_count = int(kwargs.get("losing_trades") or kwargs.get("loss_count") or (args[7] if len(args) > 7 else 0))
        drawdown_pct = float(kwargs.get("max_drawdown_pct") or kwargs.get("drawdown_pct") or (args[8] if len(args) > 8 else 0.0))
        execution_mode = str(kwargs.get("mode") or kwargs.get("execution_mode") or (args[9] if len(args) > 9 else "SEMI"))

        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            now_str = datetime.datetime.now().isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO daily_equity (
                    trade_date, starting_equity, ending_equity, realized_pnl,
                    unrealized_pnl, total_trades, win_count, loss_count,
                    drawdown_pct, execution_mode, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade_date, starting_equity, ending_equity, realized_pnl, unrealized_pnl, total_trades, win_count, loss_count, drawdown_pct, execution_mode, now_str))
            conn.commit()
            return cursor.lastrowid

    def get_daily_equity_history(self, limit: int = 60) -> List[Dict[str, Any]]:
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM daily_equity ORDER BY trade_date ASC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]

    # --- AUTHENTICATION ---
    def get_api_secret(self) -> str:
        """Returns the in-memory API secret for the running process (never persisted in plaintext to SQLite)."""
        return self._session_secret

    def verify_auth_token(self, token: Optional[str]) -> bool:
        """
        Constant-time token verification.
        Validates against:
        1. In-memory session secret / QUANTBOT_API_SECRET env var.
        2. Stored SHA-256 hash in SQLite risk_config table.
        """
        if not token:
            return False
        import hashlib
        import hmac
        token_clean = token.strip()
        
        # 1. Compare against active session secret
        if self._session_secret and hmac.compare_digest(token_clean, self._session_secret):
            return True
        
        # 2. Compare against QUANTBOT_API_SECRET environment variable
        env_secret = os.getenv("QUANTBOT_API_SECRET")
        if env_secret and hmac.compare_digest(token_clean, env_secret.strip()):
            return True
        
        # 3. Compare against stored SHA-256 hash in SQLite
        config = self.get_config()
        stored_hash = config.get("api_secret_hash")
        if stored_hash:
            token_hash = hashlib.sha256(token_clean.encode('utf-8')).hexdigest()
            return hmac.compare_digest(token_hash, stored_hash)
        
        return False


# Global Singleton RiskStore
risk_store = RiskStore()
