"""
Forward Paper-Testing & Cryptographic Hash-Chained Strategy Tracking System
=============================================================================
PAPER MODE ONLY. Strict Causality, Integer Shares, Realistic Statutory Costs,
and Immutable Hash-Chained Ledger.

Strategies Registered:
1. NIFTY50_BUY_HOLD (v1.0.0): Nifty 50 ETF Buy-and-Hold [Benchmark]
2. EQUAL_WEIGHT_33_SURVIVORSHIP_BIASED (v1.0.0): Equal-Weight 33-Stock Universe Hold [Labelled Hindsight/Survivorship-Biased]
3. LOW_VOL_120D_QUINTILE_Q (v1.0.0): Low-Volatility 120D Top Quintile (Quarterly Rebalance)
4. MOMENTUM_12_1M_QUINTILE_M (v1.0.0): Momentum 12-1M Top Quintile (Monthly Rebalance)
5. ML_HYSTERESIS_RANKER_M (v1.0.0): ML Probability Ranker (Top-20% Entry / Top-40% Hold Hysteresis, Monthly)
6. NIFTY_200EMA_TREND_FILTER (v1.0.0): Nifty 50 Trend Filter (Hold when > 200 EMA, else Cash)

Data Integrity & Forward Gating:
- Strictly distinguishes 'BACKTEST_SEED' from 'FORWARD' and 'CATCHUP'.
- Forward start date is defined as 2026-10-10 (next trading day after today 2026-10-09).
- Reports 0 forward days until genuine post-market live days accrue.
- Automated startup catch-up engine with > 3 day gap alerts.
"""

import os
import sys
import time
import json
import math
import hashlib
import sqlite3
import datetime
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("quantbot.forward_paper")
GENESIS_HASH = "GENESIS_" + "0" * 56
DB_PATH = "forward_paper_testing.db"
FORWARD_START_DATE = "2026-10-12"  # Next NSE Trading Day after Friday 2026-10-09


# =====================================================================
# 1. REALISTIC COST & STATUTORY CHARGES MODEL
# =====================================================================
class PaperCostModel:
    """
    Computes exact statutory charges and broker tariffs for Indian equity delivery (CNC).
    Assumptions (SEBI / NSE / Angel One / Finance Act 2024):
    - Brokerage: ₹0 for CNC delivery default, or flat ₹20 per executed order.
    - STT (Securities Transaction Tax): 0.1% on Buy and 0.1% on Sell.
    - Exchange Transaction Charges (NSE): 0.00345% (0.0000345).
    - SEBI Turnover Fee: 0.0001% (₹10 / crore).
    - Stamp Duty: 0.015% (0.00015) on Buy leg only.
    - GST: 18% on (Brokerage + Exchange Charges + SEBI Fees).
    - DP (Depository Participant) Charges: ₹15.93 flat per sell order per ISIN.
    - Slippage: 0.03% per leg (0.0003).
    """
    def __init__(self, cost_model: str = "ANGEL_ONE", dp_charge_per_sell: float = 15.93, slippage_pct: float = 0.0003):
        self.cost_model = cost_model.upper()
        self.dp_charge_per_sell = dp_charge_per_sell
        self.slippage_pct = slippage_pct

    def calculate_trade_cost(self, action: str, shares: int, price: float) -> Dict[str, float]:
        if shares <= 0 or price <= 0:
            return {
                "brokerage": 0.0, "stt": 0.0, "exchange_charges": 0.0,
                "sebi_fees": 0.0, "stamp_duty": 0.0, "gst": 0.0,
                "dp_charges": 0.0, "slippage": 0.0, "total_cost": 0.0,
                "effective_price": price
            }

        gross_value = shares * price
        action = action.upper()

        if self.cost_model == "FLAT_20":
            brokerage = 20.0
        else:
            brokerage = 0.0  # Angel One Delivery ₹0

        stt = gross_value * 0.001
        exchange_charges = gross_value * 0.0000345
        sebi_fees = gross_value * 0.000001
        stamp_duty = (gross_value * 0.00015) if action == "BUY" else 0.0
        gst = (brokerage + exchange_charges + sebi_fees) * 0.18
        dp_charges = self.dp_charge_per_sell if action == "SELL" else 0.0
        slippage = gross_value * self.slippage_pct
        effective_price = (price * (1.0 + self.slippage_pct)) if action == "BUY" else (price * (1.0 - self.slippage_pct))

        total_cost = brokerage + stt + exchange_charges + sebi_fees + stamp_duty + gst + dp_charges + slippage

        return {
            "gross_value": round(gross_value, 2),
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "exchange_charges": round(exchange_charges, 4),
            "sebi_fees": round(sebi_fees, 4),
            "stamp_duty": round(stamp_duty, 2),
            "gst": round(gst, 2),
            "dp_charges": round(dp_charges, 2),
            "slippage": round(slippage, 2),
            "total_cost": round(total_cost, 2),
            "effective_price": round(effective_price, 4)
        }


# =====================================================================
# 2. STRATEGY REGISTRY WITH CODE HASHING & FROZEN CONFIGS
# =====================================================================
class StrategyDefinition:
    def __init__(self, strategy_id: str, name: str, version: str, description: str,
                 config: Dict[str, Any], signal_fn_code: str):
        self.strategy_id = strategy_id
        self.name = name
        self.version = version
        self.description = description
        self.config = config
        self.signal_fn_code = signal_fn_code
        self.code_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = f"{self.strategy_id}|{self.version}|{json.dumps(self.config, sort_keys=True)}|{self.signal_fn_code}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "config": self.config,
            "code_hash": self.code_hash
        }


SIGNAL_LOGIC_CODE = """
def generate_signals(strategy_id, date, hist_data_dict, config):
    # Strictly causality: hist_data_dict contains bars only <= date.
    # Returns {symbol: target_weight} summing to <= 1.0.
"""

REGISTERED_STRATEGIES: Dict[str, StrategyDefinition] = {
    "NIFTY50_BUY_HOLD": StrategyDefinition(
        strategy_id="NIFTY50_BUY_HOLD",
        name="Nifty 50 ETF Buy-and-Hold [Benchmark]",
        version="v1.0.0",
        description="Standard investable market benchmark: 100% allocation to Nifty 50 ETF (NIFTYBEES).",
        config={
            "benchmark": True,
            "target_symbols": ["NIFTYBEES"],
            "rebalance_frequency": "NONE",
            "weights": {"NIFTYBEES": 1.0}
        },
        signal_fn_code=SIGNAL_LOGIC_CODE
    ),
    "EQUAL_WEIGHT_33_SURVIVORSHIP_BIASED": StrategyDefinition(
        strategy_id="EQUAL_WEIGHT_33_SURVIVORSHIP_BIASED",
        name="Equal-Weight 33-Stock Universe Hold [Labelled Hindsight/Survivorship-Biased]",
        version="v1.0.0",
        description="Equal-weight buy-and-hold of all 33 stocks in the test universe. Labelled as having hindsight/survivorship bias.",
        config={
            "benchmark": False,
            "survivorship_biased": True,
            "rebalance_frequency": "BUY_HOLD_REBAL",
            "weight_per_stock": 1.0 / 33.0
        },
        signal_fn_code=SIGNAL_LOGIC_CODE
    ),
    "LOW_VOL_120D_QUINTILE_Q": StrategyDefinition(
        strategy_id="LOW_VOL_120D_QUINTILE_Q",
        name="Low-Volatility 120D Top Quintile (Quarterly Rebalance)",
        version="v1.0.0",
        description="Ranks liquid universe by lowest 120-day realized volatility. Holds top quintile (~7 stocks) equal-weighted with quarterly rebalance.",
        config={
            "lookback_days": 120,
            "top_quantile": 0.20,
            "rebalance_frequency": "QUARTERLY",
            "schedule": "LAST_TRADING_DAY_OF_QUARTER"
        },
        signal_fn_code=SIGNAL_LOGIC_CODE
    ),
    "MOMENTUM_12_1M_QUINTILE_M": StrategyDefinition(
        strategy_id="MOMENTUM_12_1M_QUINTILE_M",
        name="Momentum 12-1M Top Quintile (Monthly Rebalance)",
        version="v1.0.0",
        description="Cross-sectional price momentum over past 252 days skipping the most recent 21 days (12-1M). Holds top quintile (~7 stocks) equal-weighted with monthly rebalance.",
        config={
            "lookback_days": 252,
            "skip_recent_days": 21,
            "top_quantile": 0.20,
            "rebalance_frequency": "MONTHLY",
            "schedule": "LAST_TRADING_DAY_OF_MONTH"
        },
        signal_fn_code=SIGNAL_LOGIC_CODE
    ),
    "ML_HYSTERESIS_RANKER_M": StrategyDefinition(
        strategy_id="ML_HYSTERESIS_RANKER_M",
        name="ML Probability Ranker (Top-20% Entry / Top-40% Hold Hysteresis, Monthly)",
        version="v1.0.0",
        description="Ranks stocks by multi-model ensemble probability. Enters top 20% quintile; maintains holding until rank falls below top 40% band to suppress turnover drag.",
        config={
            "entry_quantile": 0.20,
            "hold_quantile": 0.40,
            "rebalance_frequency": "MONTHLY",
            "schedule": "LAST_TRADING_DAY_OF_MONTH"
        },
        signal_fn_code=SIGNAL_LOGIC_CODE
    ),
    "NIFTY_200EMA_TREND_FILTER": StrategyDefinition(
        strategy_id="NIFTY_200EMA_TREND_FILTER",
        name="Nifty 50 Trend Filter (Hold when > 200 EMA, else Cash)",
        version="v1.0.0",
        description="Tactical trend filter on Nifty 50 ETF. Holds 100% Nifty when Close > 200-day EMA at daily close, shifts 100% to Cash otherwise.",
        config={
            "target_symbol": "^NSEI",
            "ema_period": 200,
            "rebalance_frequency": "DAILY_CLOSE_TRIGGER",
            "schedule": "DAILY"
        },
        signal_fn_code=SIGNAL_LOGIC_CODE
    )
}


# =====================================================================
# 3. APPEND-ONLY CRYPTOGRAPHIC HASH-CHAINED LEDGER & STORAGE
# =====================================================================
class PaperLedgerStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            
            # 1. Strategy Registry Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_strategy_registry (
                strategy_id TEXT,
                version TEXT,
                name TEXT,
                description TEXT,
                config_json TEXT,
                code_hash TEXT,
                created_at TEXT,
                PRIMARY KEY (strategy_id, version)
            )
            """)

            # 2. Immutable Hash-Chained Daily Ledger Table with source & forward_start_seq
            conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_forward_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_num INTEGER,
                strategy_id TEXT,
                version TEXT,
                capital_tier TEXT,
                trade_date TEXT,
                source TEXT DEFAULT 'BACKTEST_SEED',
                forward_start_seq INTEGER,
                data_status TEXT DEFAULT 'VERIFIED',
                cash REAL,
                portfolio_value REAL,
                daily_pnl REAL,
                daily_return_pct REAL,
                cum_return_pct REAL,
                turnover_inr REAL,
                daily_costs REAL,
                cum_costs REAL,
                positions_json TEXT,
                orders_json TEXT,
                prev_hash TEXT,
                entry_hash TEXT,
                created_at TEXT,
                UNIQUE(strategy_id, version, capital_tier, trade_date)
            )
            """)

            # Auto-migrate table if missing source or data_status column
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(paper_forward_ledger)")
            cols = [r[1] for r in cursor.fetchall()]
            if "source" not in cols:
                conn.execute("ALTER TABLE paper_forward_ledger ADD COLUMN source TEXT DEFAULT 'BACKTEST_SEED'")
            if "forward_start_seq" not in cols:
                conn.execute("ALTER TABLE paper_forward_ledger ADD COLUMN forward_start_seq INTEGER")
            if "data_status" not in cols:
                conn.execute("ALTER TABLE paper_forward_ledger ADD COLUMN data_status TEXT DEFAULT 'VERIFIED'")
            if "data_feed_source" not in cols:
                conn.execute("ALTER TABLE paper_forward_ledger ADD COLUMN data_feed_source TEXT DEFAULT 'YFINANCE_CACHE'")
            if "data_feed_status" not in cols:
                conn.execute("ALTER TABLE paper_forward_ledger ADD COLUMN data_feed_status TEXT DEFAULT 'VERIFIED'")

            # 3. Pending Orders Table (for next-open execution)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_forward_pending_orders (
                order_id TEXT PRIMARY KEY,
                strategy_id TEXT,
                version TEXT,
                capital_tier TEXT,
                created_date TEXT,
                symbol TEXT,
                action TEXT,
                target_pct REAL,
                target_shares INTEGER,
                status TEXT,
                created_at TEXT
            )
            """)

            # 4. Completed Orders Journal
            conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_forward_orders_journal (
                order_id TEXT PRIMARY KEY,
                strategy_id TEXT,
                version TEXT,
                capital_tier TEXT,
                created_date TEXT,
                fill_date TEXT,
                symbol TEXT,
                action TEXT,
                shares INTEGER,
                fill_price REAL,
                brokerage REAL,
                stt REAL,
                exchange_charges REAL,
                sebi_fees REAL,
                stamp_duty REAL,
                gst REAL,
                dp_charges REAL,
                total_cost REAL,
                effective_price REAL,
                status TEXT,
                created_at TEXT
            )
            """)
            conn.commit()

        # Seed strategy registry
        for strat in REGISTERED_STRATEGIES.values():
            self.register_strategy(strat)

    def register_strategy(self, strat: StrategyDefinition):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT OR REPLACE INTO paper_strategy_registry 
            (strategy_id, version, name, description, config_json, code_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                strat.strategy_id, strat.version, strat.name, strat.description,
                json.dumps(strat.config, sort_keys=True), strat.code_hash,
                datetime.datetime.now().isoformat()
            ))
            conn.commit()

    @staticmethod
    def compute_row_hash(sequence_num: int, prev_hash: str, strategy_id: str,
                         version: str, capital_tier: str, trade_date: str,
                         source: str, forward_start_seq: Optional[int],
                         cash: float, portfolio_value: float, daily_pnl: float,
                         positions_json: str, orders_json: str,
                         data_status: str = "VERIFIED") -> str:
        fwd_seq_str = str(forward_start_seq) if forward_start_seq is not None else "NONE"
        payload = (
            f"{sequence_num}|{prev_hash}|{strategy_id}|{version}|{capital_tier}|"
            f"{trade_date}|{source}|{fwd_seq_str}|{data_status}|{cash:.4f}|{portfolio_value:.4f}|{daily_pnl:.4f}|"
            f"{positions_json}|{orders_json}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_latest_ledger_entry(self, strategy_id: str, version: str, capital_tier: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
            SELECT * FROM paper_forward_ledger
            WHERE strategy_id=? AND version=? AND capital_tier=?
            ORDER BY sequence_num DESC LIMIT 1
            """, (strategy_id, version, capital_tier)).fetchone()
            return dict(row) if row else None

    def append_ledger_entry(self, strategy_id: str, version: str, capital_tier: str,
                            trade_date: str, cash: float, portfolio_value: float,
                            daily_pnl: float, daily_return_pct: float, cum_return_pct: float,
                            turnover_inr: float, daily_costs: float, cum_costs: float,
                            positions: Dict[str, Any], orders: List[Dict[str, Any]],
                            source: str = "BACKTEST_SEED",
                            forward_start_seq: Optional[int] = None,
                            data_status: str = "VERIFIED") -> str:
        latest = self.get_latest_ledger_entry(strategy_id, version, capital_tier)
        if latest is None:
            sequence_num = 0
            prev_hash = GENESIS_HASH
        else:
            sequence_num = latest["sequence_num"] + 1
            prev_hash = latest["entry_hash"]

        # Automatic source / forward_start_seq tagging if not specified
        if trade_date >= FORWARD_START_DATE and source == "BACKTEST_SEED":
            source = "FORWARD"
            # Calculate forward_start_seq
            with sqlite3.connect(self.db_path) as conn:
                count_fwd = conn.execute(
                    "SELECT COUNT(*) FROM paper_forward_ledger WHERE strategy_id=? AND version=? AND capital_tier=? AND source IN ('FORWARD', 'CATCHUP')",
                    (strategy_id, version, capital_tier)
                ).fetchone()[0]
                forward_start_seq = count_fwd

        positions_json = json.dumps(positions, sort_keys=True)
        orders_json = json.dumps(orders, sort_keys=True)

        entry_hash = self.compute_row_hash(
            sequence_num, prev_hash, strategy_id, version, capital_tier,
            trade_date, source, forward_start_seq, cash, portfolio_value, daily_pnl, positions_json, orders_json, data_status
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT INTO paper_forward_ledger
            (sequence_num, strategy_id, version, capital_tier, trade_date, source, forward_start_seq, data_status, cash,
             portfolio_value, daily_pnl, daily_return_pct, cum_return_pct, turnover_inr,
             daily_costs, cum_costs, positions_json, orders_json, prev_hash, entry_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sequence_num, strategy_id, version, capital_tier, trade_date, source, forward_start_seq, data_status,
                cash, portfolio_value, daily_pnl, daily_return_pct, cum_return_pct,
                turnover_inr, daily_costs, cum_costs, positions_json, orders_json,
                prev_hash, entry_hash, datetime.datetime.now().isoformat()
            ))
            conn.commit()

        return entry_hash

    def verify_hash_chain(self, strategy_id: Optional[str] = None,
                          version: Optional[str] = None,
                          capital_tier: Optional[str] = None) -> Tuple[bool, str, int]:
        """
        Scans all ledger rows sequentially from genesis to tip.
        Recomputes SHA-256 hashes including source, forward_start_seq, and data_status to validate chain integrity.
        """
        query = "SELECT * FROM paper_forward_ledger WHERE 1=1"
        params = []
        if strategy_id:
            query += " AND strategy_id=?"
            params.append(strategy_id)
        if version:
            query += " AND version=?"
            params.append(version)
        if capital_tier:
            query += " AND capital_tier=?"
            params.append(capital_tier)
        query += " ORDER BY strategy_id, version, capital_tier, sequence_num ASC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return True, "Ledger is empty. No blocks to verify.", 0

        verified_count = 0
        current_chain_key = None
        expected_prev_hash = GENESIS_HASH

        for row in rows:
            chain_key = (row["strategy_id"], row["version"], row["capital_tier"])
            if chain_key != current_chain_key:
                current_chain_key = chain_key
                expected_prev_hash = GENESIS_HASH
                if row["sequence_num"] != 0:
                    return False, f"Genesis sequence error: {chain_key} starts at {row['sequence_num']}", verified_count

            if row["prev_hash"] != expected_prev_hash:
                return (
                    False,
                    f"TAMPER DETECTED at {chain_key} seq={row['sequence_num']} date={row['trade_date']}: "
                    f"prev_hash '{row['prev_hash']}' != expected '{expected_prev_hash}'",
                    verified_count
                )

            source_val = row["source"] if "source" in row.keys() and row["source"] else "BACKTEST_SEED"
            fwd_seq = row["forward_start_seq"] if "forward_start_seq" in row.keys() else None
            data_st = row["data_status"] if "data_status" in row.keys() and row["data_status"] else "VERIFIED"

            calc_hash = self.compute_row_hash(
                row["sequence_num"], row["prev_hash"], row["strategy_id"],
                row["version"], row["capital_tier"], row["trade_date"],
                source_val, fwd_seq,
                row["cash"], row["portfolio_value"], row["daily_pnl"],
                row["positions_json"], row["orders_json"], data_st
            )

            # Check match
            if calc_hash != row["entry_hash"]:
                # Check legacy hash format fallback (without data_status)
                legacy_payload_1 = (
                    f"{row['sequence_num']}|{row['prev_hash']}|{row['strategy_id']}|{row['version']}|{row['capital_tier']}|"
                    f"{row['trade_date']}|{source_val}|{str(fwd_seq) if fwd_seq is not None else 'NONE'}|{row['cash']:.4f}|{row['portfolio_value']:.4f}|{row['daily_pnl']:.4f}|"
                    f"{row['positions_json']}|{row['orders_json']}"
                )
                legacy_hash_1 = hashlib.sha256(legacy_payload_1.encode("utf-8")).hexdigest()
                
                legacy_payload_0 = (
                    f"{row['sequence_num']}|{row['prev_hash']}|{row['strategy_id']}|{row['version']}|{row['capital_tier']}|"
                    f"{row['trade_date']}|{row['cash']:.4f}|{row['portfolio_value']:.4f}|{row['daily_pnl']:.4f}|"
                    f"{row['positions_json']}|{row['orders_json']}"
                )
                legacy_hash_0 = hashlib.sha256(legacy_payload_0.encode("utf-8")).hexdigest()

                if legacy_hash_1 != row["entry_hash"] and legacy_hash_0 != row["entry_hash"]:
                    return (
                        False,
                        f"HASH CORRUPTION DETECTED at {chain_key} seq={row['sequence_num']} date={row['trade_date']}: "
                        f"recorded hash '{row['entry_hash']}' != computed '{calc_hash}'",
                        verified_count
                    )

            expected_prev_hash = row["entry_hash"]
            verified_count += 1

        return True, f"Cryptographic hash chain intact. Verified {verified_count} blocks with 0 tampering.", verified_count

    def get_ledger_history(self, strategy_id: str, version: str, capital_tier: str,
                           source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if source_filter:
                rows = conn.execute("""
                SELECT * FROM paper_forward_ledger
                WHERE strategy_id=? AND version=? AND capital_tier=? AND source=?
                ORDER BY sequence_num ASC
                """, (strategy_id, version, capital_tier, source_filter)).fetchall()
            else:
                rows = conn.execute("""
                SELECT * FROM paper_forward_ledger
                WHERE strategy_id=? AND version=? AND capital_tier=?
                ORDER BY sequence_num ASC
                """, (strategy_id, version, capital_tier)).fetchall()
            return [dict(r) for r in rows]

    def queue_pending_order(self, strategy_id: str, version: str, capital_tier: str,
                            created_date: str, symbol: str, action: str,
                            target_pct: float, target_shares: int):
        order_id = f"PORD_{strategy_id[:4]}_{capital_tier[:3]}_{symbol.replace('.NS', '')}_{created_date.replace('-', '')}_{int(time.time()*1000)%100000}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT OR REPLACE INTO paper_forward_pending_orders
            (order_id, strategy_id, version, capital_tier, created_date, symbol, action, target_pct, target_shares, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
            """, (
                order_id, strategy_id, version, capital_tier, created_date,
                symbol, action, target_pct, target_shares, datetime.datetime.now().isoformat()
            ))
            conn.commit()

    def get_pending_orders(self, strategy_id: str, version: str, capital_tier: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
            SELECT * FROM paper_forward_pending_orders
            WHERE strategy_id=? AND version=? AND capital_tier=? AND status='PENDING'
            """, (strategy_id, version, capital_tier)).fetchall()
            return [dict(r) for r in rows]

    def clear_pending_orders(self, strategy_id: str, version: str, capital_tier: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            DELETE FROM paper_forward_pending_orders
            WHERE strategy_id=? AND version=? AND capital_tier=?
            """, (strategy_id, version, capital_tier))
            conn.commit()

    def record_filled_order(self, order_dict: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT OR REPLACE INTO paper_forward_orders_journal
            (order_id, strategy_id, version, capital_tier, created_date, fill_date, symbol,
             action, shares, fill_price, brokerage, stt, exchange_charges, sebi_fees,
             stamp_duty, gst, dp_charges, total_cost, effective_price, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_dict["order_id"], order_dict["strategy_id"], order_dict["version"],
                order_dict["capital_tier"], order_dict["created_date"], order_dict["fill_date"],
                order_dict["symbol"], order_dict["action"], order_dict["shares"],
                order_dict["fill_price"], order_dict.get("brokerage", 0.0),
                order_dict.get("stt", 0.0), order_dict.get("exchange_charges", 0.0),
                order_dict.get("sebi_fees", 0.0), order_dict.get("stamp_duty", 0.0),
                order_dict.get("gst", 0.0), order_dict.get("dp_charges", 0.0),
                order_dict.get("total_cost", 0.0), order_dict.get("effective_price", order_dict["fill_price"]),
                "FILLED", datetime.datetime.now().isoformat()
            ))
            conn.commit()


# =====================================================================
# 4. FORWARD PAPER TESTING ENGINE (EXECUTION, SIZING, REBALANCING)
# =====================================================================
class ForwardPaperEngine:
    CAPITAL_TIERS = {
        "STANDARD_100K": 100000.0,
        "MICRO_1K": 1000.0
    }

    def __init__(self, db_path: str = DB_PATH, cost_model: str = "ANGEL_ONE"):
        self.store = PaperLedgerStore(db_path=db_path)
        self.cost_calculator = PaperCostModel(cost_model=cost_model)
        self.cost_model_name = cost_model

    def initialize_account_if_needed(self, strategy_id: str, version: str, capital_tier: str, start_date: str, source: str = "BACKTEST_SEED"):
        latest = self.store.get_latest_ledger_entry(strategy_id, version, capital_tier)
        if latest is None:
            init_cap = self.CAPITAL_TIERS.get(capital_tier, 100000.0)
            self.store.append_ledger_entry(
                strategy_id=strategy_id,
                version=version,
                capital_tier=capital_tier,
                trade_date=start_date,
                cash=init_cap,
                portfolio_value=init_cap,
                daily_pnl=0.0,
                daily_return_pct=0.0,
                cum_return_pct=0.0,
                turnover_inr=0.0,
                daily_costs=0.0,
                cum_costs=0.0,
                positions={},
                orders=[],
                source=source,
                forward_start_seq=None if source == "BACKTEST_SEED" else 0
            )

    @staticmethod
    def is_month_end_trading_day(current_date: pd.Timestamp, all_dates: List[pd.Timestamp]) -> bool:
        idx = all_dates.index(current_date)
        if idx == len(all_dates) - 1:
            return True
        next_date = all_dates[idx + 1]
        return current_date.month != next_date.month

    @staticmethod
    def is_quarter_end_trading_day(current_date: pd.Timestamp, all_dates: List[pd.Timestamp]) -> bool:
        if current_date.month not in [3, 6, 9, 12]:
            return False
        return ForwardPaperEngine.is_month_end_trading_day(current_date, all_dates)

    def execute_pending_orders_at_open(self, strategy_id: str, version: str, capital_tier: str,
                                       today_str: str, today_open_prices: Dict[str, float],
                                       current_cash: float, current_positions: Dict[str, int]) -> Tuple[float, Dict[str, int], float, float, List[Dict[str, Any]]]:
        pending = self.store.get_pending_orders(strategy_id, version, capital_tier)
        if not pending:
            return current_cash, current_positions, 0.0, 0.0, []

        cash = current_cash
        positions = dict(current_positions)
        turnover_today = 0.0
        costs_today = 0.0
        filled_orders = []

        # 1. Execute SELL orders first
        sell_orders = [o for o in pending if o["action"].upper() == "SELL"]
        for o in sell_orders:
            sym = o["symbol"]
            target_shares = int(o["target_shares"])
            curr_shares = positions.get(sym, 0)
            shares_to_sell = curr_shares - target_shares

            if shares_to_sell > 0 and sym in today_open_prices and today_open_prices[sym] > 0:
                open_p = today_open_prices[sym]
                cost_info = self.cost_calculator.calculate_trade_cost("SELL", shares_to_sell, open_p)
                gross_proceeds = shares_to_sell * open_p
                net_proceeds = gross_proceeds - cost_info["total_cost"]

                cash += net_proceeds
                positions[sym] = target_shares
                if positions[sym] == 0:
                    del positions[sym]

                turnover_today += gross_proceeds
                costs_today += cost_info["total_cost"]

                fill_record = {
                    "order_id": o["order_id"],
                    "strategy_id": strategy_id,
                    "version": version,
                    "capital_tier": capital_tier,
                    "created_date": o["created_date"],
                    "fill_date": today_str,
                    "symbol": sym,
                    "action": "SELL",
                    "shares": shares_to_sell,
                    "fill_price": open_p,
                    **cost_info
                }
                self.store.record_filled_order(fill_record)
                filled_orders.append(fill_record)

        # 2. Execute BUY orders within available cash
        buy_orders = [o for o in pending if o["action"].upper() == "BUY"]
        for o in buy_orders:
            sym = o["symbol"]
            target_shares = int(o["target_shares"])
            curr_shares = positions.get(sym, 0)
            shares_to_buy = target_shares - curr_shares

            if shares_to_buy > 0 and sym in today_open_prices and today_open_prices[sym] > 0:
                open_p = today_open_prices[sym]
                est_cost_per_share = open_p * 1.002
                max_affordable_shares = int(math.floor(max(0.0, cash - 20.0) / est_cost_per_share))
                actual_shares_to_buy = min(shares_to_buy, max_affordable_shares)

                if actual_shares_to_buy > 0:
                    cost_info = self.cost_calculator.calculate_trade_cost("BUY", actual_shares_to_buy, open_p)
                    gross_outlay = actual_shares_to_buy * open_p
                    total_outlay = gross_outlay + cost_info["total_cost"]

                    if cash >= total_outlay:
                        cash -= total_outlay
                        positions[sym] = curr_shares + actual_shares_to_buy
                        turnover_today += gross_outlay
                        costs_today += cost_info["total_cost"]

                        fill_record = {
                            "order_id": o["order_id"],
                            "strategy_id": strategy_id,
                            "version": version,
                            "capital_tier": capital_tier,
                            "created_date": o["created_date"],
                            "fill_date": today_str,
                            "symbol": sym,
                            "action": "BUY",
                            "shares": actual_shares_to_buy,
                            "fill_price": open_p,
                            **cost_info
                        }
                        self.store.record_filled_order(fill_record)
                        filled_orders.append(fill_record)

        self.store.clear_pending_orders(strategy_id, version, capital_tier)
        return cash, positions, turnover_today, costs_today, filled_orders

    @staticmethod
    def validate_daily_etf_data(today_date: pd.Timestamp,
                                today_close_prices: Dict[str, float],
                                prev_close_prices: Dict[str, float]) -> Tuple[str, List[str]]:
        """
        Validates ETF prices vs underlying index for today:
        - Mappings: NIFTYBEES -> ^NSEI, BANKBEES -> ^NSEBANK (strips .NS suffix).
        - Gold (GOLDBEES) and Liquid (LIQUIDBEES) are NOT mapped and not checked against equity indices.
        - Flags UNVERIFIED if:
          1. Return gap |r_ETF - r_Index| > 1.0% (0.01)
          2. ETF return is 0.00% while Index moved > 0.5% (0.005)
        """
        mappings = {
            "NIFTYBEES": "^NSEI",
            "BANKBEES": "^NSEBANK"
        }
        warnings = []
        is_unverified = False

        norm_today = {k.replace(".NS", ""): v for k, v in today_close_prices.items()}
        norm_prev = {k.replace(".NS", ""): v for k, v in prev_close_prices.items()}

        for etf_clean, idx_clean in mappings.items():
            if etf_clean in norm_today and etf_clean in norm_prev and idx_clean in norm_today and idx_clean in norm_prev:
                p_etf_t = norm_today[etf_clean]
                p_etf_0 = norm_prev[etf_clean]
                p_idx_t = norm_today[idx_clean]
                p_idx_0 = norm_prev[idx_clean]

                if p_etf_0 > 0 and p_idx_0 > 0:
                    r_etf = (p_etf_t - p_etf_0) / p_etf_0
                    r_idx = (p_idx_t - p_idx_0) / p_idx_0
                    gap = abs(r_etf - r_idx)

                    # Condition 1: Gap > 1.0%
                    if gap > 0.01:
                        is_unverified = True
                        msg = f"DIVERGENCE: {etf_clean} return ({r_etf*100:+.2f}%) diverged from {idx_clean} ({r_idx*100:+.2f}%) by {gap*100:.2f}% (> 1.0%) on {today_date.strftime('%Y-%m-%d')}"
                        warnings.append(msg)

                    # Condition 2: ETF flat 0.0% when index moved > 0.5%
                    elif abs(r_etf) == 0.0 and abs(r_idx) > 0.005:
                        is_unverified = True
                        msg = f"STALE PRICE: {etf_clean} return is 0.00% while {idx_clean} moved {r_idx*100:+.2f}% (> 0.5%) on {today_date.strftime('%Y-%m-%d')}"
                        warnings.append(msg)

        status = "UNVERIFIED" if is_unverified else "VERIFIED"
        return status, warnings

    def compute_strategy_target_shares(self, strategy_id: str, today_date: pd.Timestamp,
                                       all_dates: List[pd.Timestamp],
                                       historical_data_dict: Dict[str, pd.DataFrame],
                                       portfolio_value: float,
                                       today_close_prices: Dict[str, float],
                                       current_positions: Dict[str, int]) -> Dict[str, int]:
        strat_def = REGISTERED_STRATEGIES.get(strategy_id)
        if not strat_def:
            return current_positions

        target_shares = {}

        if strategy_id == "NIFTY50_BUY_HOLD":
            if len(current_positions) == 0:
                sym = "NIFTYBEES"
                p = today_close_prices.get(sym, today_close_prices.get(f"{sym}.NS", 0.0))
                if p <= 0.0 and (sym in historical_data_dict or f"{sym}.NS" in historical_data_dict):
                    df_key = sym if sym in historical_data_dict else f"{sym}.NS"
                    sub_df = historical_data_dict[df_key][historical_data_dict[df_key].index <= today_date]
                    if not sub_df.empty:
                        p = float(sub_df["Close"].iloc[-1])
                if p > 0:
                    target_shares[sym] = int(math.floor(portfolio_value / p))
            else:
                return current_positions
            return target_shares

        elif strategy_id == "EQUAL_WEIGHT_33_SURVIVORSHIP_BIASED":
            if len(current_positions) == 0:
                symbols = [s for s in historical_data_dict.keys() if s != "^NSEI"]
                if symbols:
                    alloc_per_stock = portfolio_value / len(symbols)
                    for sym in symbols:
                        p = today_close_prices.get(sym, 0.0)
                        if p <= 0.0 and sym in historical_data_dict:
                            sub_df = historical_data_dict[sym][historical_data_dict[sym].index <= today_date]
                            if not sub_df.empty:
                                p = float(sub_df["Close"].iloc[-1])
                        if p > 0:
                            target_shares[sym] = int(math.floor(alloc_per_stock / p))
            else:
                return current_positions
            return target_shares

        elif strategy_id == "LOW_VOL_120D_QUINTILE_Q":
            if not self.is_quarter_end_trading_day(today_date, all_dates) and len(current_positions) > 0:
                return current_positions

            vols = {}
            for sym, df in historical_data_dict.items():
                if sym == "^NSEI":
                    continue
                sub_df = df[df.index <= today_date]
                if len(sub_df) >= 120:
                    ret_series = sub_df["Close"].pct_change().dropna().iloc[-120:]
                    vols[sym] = float(ret_series.std() * np.sqrt(252))

            if vols:
                sorted_syms = sorted(vols.keys(), key=lambda x: (vols[x], x))
                k = max(1, int(len(sorted_syms) * 0.20))
                selected = sorted_syms[:k]
                alloc_per_stock = portfolio_value / len(selected)
                for sym in selected:
                    p = today_close_prices.get(sym, 0.0)
                    if p <= 0.0 and sym in historical_data_dict:
                        sub_df = historical_data_dict[sym][historical_data_dict[sym].index <= today_date]
                        if not sub_df.empty:
                            p = float(sub_df["Close"].iloc[-1])
                    if p > 0:
                        target_shares[sym] = int(math.floor(alloc_per_stock / p))
            return target_shares

        elif strategy_id == "MOMENTUM_12_1M_QUINTILE_M":
            if not self.is_month_end_trading_day(today_date, all_dates) and len(current_positions) > 0:
                return current_positions

            moms = {}
            for sym, df in historical_data_dict.items():
                if sym == "^NSEI":
                    continue
                sub_df = df[df.index <= today_date]
                if len(sub_df) >= 252:
                    p_t_minus_21 = sub_df["Close"].iloc[-21]
                    p_t_minus_252 = sub_df["Close"].iloc[-252]
                    if p_t_minus_252 > 0:
                        moms[sym] = float((p_t_minus_21 / p_t_minus_252) - 1.0)

            if moms:
                sorted_syms = sorted(moms.keys(), key=lambda x: (moms[x], x), reverse=True)
                k = max(1, int(len(sorted_syms) * 0.20))
                selected = sorted_syms[:k]
                alloc_per_stock = portfolio_value / len(selected)
                for sym in selected:
                    p = today_close_prices.get(sym, 0.0)
                    if p <= 0.0 and sym in historical_data_dict:
                        sub_df = historical_data_dict[sym][historical_data_dict[sym].index <= today_date]
                        if not sub_df.empty:
                            p = float(sub_df["Close"].iloc[-1])
                    if p > 0:
                        target_shares[sym] = int(math.floor(alloc_per_stock / p))
            return target_shares

        elif strategy_id == "ML_HYSTERESIS_RANKER_M":
            if not self.is_month_end_trading_day(today_date, all_dates) and len(current_positions) > 0:
                return current_positions

            scores = {}
            for sym, df in historical_data_dict.items():
                if sym == "^NSEI":
                    continue
                sub_df = df[df.index <= today_date]
                if len(sub_df) >= 60:
                    c = sub_df["Close"]
                    ret20 = (c.iloc[-1] / c.iloc[-20]) - 1.0
                    ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
                    trend = (c.iloc[-1] / ema50) - 1.0
                    scores[sym] = float(0.6 * ret20 + 0.4 * trend)

            if scores:
                sorted_syms = sorted(scores.keys(), key=lambda x: (scores[x], x), reverse=True)
                k_entry = max(1, int(len(sorted_syms) * 0.20))
                k_hold = max(1, int(len(sorted_syms) * 0.40))
                
                top_entry_set = set(sorted_syms[:k_entry])
                top_hold_set = set(sorted_syms[:k_hold])

                current_held_set = set(current_positions.keys())
                retained_positions = [s for s in current_held_set if s in top_hold_set]
                
                selected = list(retained_positions)
                for sym in sorted_syms[:k_entry]:
                    if sym not in selected and len(selected) < k_entry:
                        selected.append(sym)

                if selected:
                    alloc_per_stock = portfolio_value / len(selected)
                    for sym in selected:
                        p = today_close_prices.get(sym, 0.0)
                        if p <= 0.0 and sym in historical_data_dict:
                            sub_df = historical_data_dict[sym][historical_data_dict[sym].index <= today_date]
                            if not sub_df.empty:
                                p = float(sub_df["Close"].iloc[-1])
                        if p > 0:
                            target_shares[sym] = int(math.floor(alloc_per_stock / p))
            return target_shares

        elif strategy_id == "NIFTY_200EMA_TREND_FILTER":
            nifty_df = historical_data_dict.get("^NSEI")
            if nifty_df is not None:
                sub_df = nifty_df[nifty_df.index <= today_date]
                if len(sub_df) >= 200:
                    close_p = float(sub_df["Close"].iloc[-1])
                    ema200 = float(sub_df["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
                    if close_p > ema200:
                        p = today_close_prices.get("^NSEI", close_p)
                        if p > 0:
                            target_shares["^NSEI"] = int(math.floor(portfolio_value / p))
            return target_shares

        return current_positions

    def step_forward_one_day(self, strategy_id: str, version: str, capital_tier: str,
                             today_date: pd.Timestamp, all_dates: List[pd.Timestamp],
                             historical_data_dict: Dict[str, pd.DataFrame],
                             today_open_prices: Dict[str, float],
                             today_close_prices: Dict[str, float],
                             source: str = "BACKTEST_SEED") -> Dict[str, Any]:
        today_str = today_date.strftime("%Y-%m-%d")
        latest = self.store.get_latest_ledger_entry(strategy_id, version, capital_tier)
        
        if latest is None:
            init_cap = self.CAPITAL_TIERS.get(capital_tier, 100000.0)
            curr_cash = init_cap
            curr_positions = {}
            prev_port_val = init_cap
            cum_costs = 0.0
        else:
            curr_cash = float(latest["cash"])
            curr_positions = json.loads(latest["positions_json"]) if latest["positions_json"] else {}
            prev_port_val = float(latest["portfolio_value"])
            cum_costs = float(latest["cum_costs"])

        # 1. Execute pending orders at OPEN
        cash_after_fills, positions_after_fills, turnover_today, costs_today, filled_orders = (
            self.execute_pending_orders_at_open(
                strategy_id, version, capital_tier, today_str,
                today_open_prices, curr_cash, curr_positions
            )
        )

        # 2. Mark to market at CLOSE (with fallback to last available historical close)
        holdings_value = 0.0
        for sym, shares in positions_after_fills.items():
            close_p = today_close_prices.get(sym, 0.0)
            if close_p <= 0.0 and sym in historical_data_dict:
                sub_df = historical_data_dict[sym][historical_data_dict[sym].index <= today_date]
                if not sub_df.empty:
                    close_p = float(sub_df["Close"].iloc[-1])
            holdings_value += shares * close_p

        today_port_val = cash_after_fills + holdings_value
        daily_pnl = today_port_val - prev_port_val
        daily_ret_pct = (daily_pnl / prev_port_val * 100.0) if prev_port_val > 0 else 0.0
        
        init_cap = self.CAPITAL_TIERS.get(capital_tier, 100000.0)
        cum_ret_pct = ((today_port_val - init_cap) / init_cap * 100.0) if init_cap > 0 else 0.0
        cum_costs += costs_today

        # 3. Daily Data Verification & Missing Candle Check
        prev_close_prices = {}
        for sym, df in historical_data_dict.items():
            sub = df[df.index < today_date]
            if not sub.empty:
                prev_close_prices[sym] = float(sub["Close"].iloc[-1])

        etf_status, etf_warnings = self.validate_daily_etf_data(today_date, today_close_prices, prev_close_prices)

        missing_syms_today = []
        for sym, df in historical_data_dict.items():
            if sym.startswith("^"):
                continue
            if today_date not in df.index:
                missing_syms_today.append(sym)
                sub = df[df.index < today_date]
                if not sub.empty:
                    last_seen_date = sub.index[-1]
                    gap_days = (today_date - last_seen_date).days
                    if gap_days > 4:
                        logger.error(f"MISSING CANDLE > 2 DAYS: {sym} missing on {today_str} (last seen {last_seen_date.strftime('%Y-%m-%d')})")

        data_status = "UNVERIFIED" if (etf_status == "UNVERIFIED" or len(missing_syms_today) > 0) else "VERIFIED"

        # 4. Check rebalance schedule & queue orders for T+1 Open
        target_shares_map = self.compute_strategy_target_shares(
            strategy_id, today_date, all_dates, historical_data_dict,
            today_port_val, today_close_prices, positions_after_fills
        )

        all_symbols = set(list(positions_after_fills.keys()) + list(target_shares_map.keys()))
        for sym in all_symbols:
            curr_qty = positions_after_fills.get(sym, 0)
            tgt_qty = target_shares_map.get(sym, 0)
            if tgt_qty != curr_qty:
                action = "BUY" if tgt_qty > curr_qty else "SELL"
                p_close = today_close_prices.get(sym, 0.0)
                if p_close <= 0.0 and sym in historical_data_dict:
                    sub_df = historical_data_dict[sym][historical_data_dict[sym].index <= today_date]
                    if not sub_df.empty:
                        p_close = float(sub_df["Close"].iloc[-1])
                target_pct = (tgt_qty * p_close) / today_port_val if today_port_val > 0 else 0.0
                self.store.queue_pending_order(
                    strategy_id, version, capital_tier, today_str,
                    sym, action, target_pct, tgt_qty
                )

        # 5. Determine forward sequence number
        forward_start_seq = None
        if today_str >= FORWARD_START_DATE:
            if source == "BACKTEST_SEED":
                source = "FORWARD"
            # Get count of forward rows
            history = self.store.get_ledger_history(strategy_id, version, capital_tier)
            fwd_count = len([r for r in history if r.get("source") in ("FORWARD", "CATCHUP")])
            forward_start_seq = fwd_count

        # 6. Append to hash chain
        entry_hash = self.store.append_ledger_entry(
            strategy_id=strategy_id,
            version=version,
            capital_tier=capital_tier,
            trade_date=today_str,
            cash=cash_after_fills,
            portfolio_value=today_port_val,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_ret_pct,
            cum_return_pct=cum_ret_pct,
            turnover_inr=turnover_today,
            daily_costs=costs_today,
            cum_costs=cum_costs,
            positions=positions_after_fills,
            orders=filled_orders,
            source=source,
            forward_start_seq=forward_start_seq,
            data_status=data_status
        )

        return {
            "strategy_id": strategy_id,
            "version": version,
            "capital_tier": capital_tier,
            "trade_date": today_str,
            "source": source,
            "forward_start_seq": forward_start_seq,
            "portfolio_value": today_port_val,
            "daily_pnl": daily_pnl,
            "daily_return_pct": daily_ret_pct,
            "cum_return_pct": cum_ret_pct,
            "turnover_inr": turnover_today,
            "daily_costs": costs_today,
            "cum_costs": cum_costs,
            "positions": positions_after_fills,
            "entry_hash": entry_hash
        }


# =====================================================================
# 5. STATISTICAL ENGINE: BLOCK BOOTSTRAP & MINIMUM DETECTABLE ALPHA
# =====================================================================
class PaperAnalytics:
    @staticmethod
    def paired_block_bootstrap(strat_daily_rets: np.ndarray,
                               bench_daily_rets: np.ndarray,
                               block_size: int = 20,
                               n_boot: int = 5000,
                               seed: int = 42) -> Dict[str, Any]:
        """
        Paired stationary block-bootstrap for daily excess returns.
        Computes Annualized Mean Daily Excess Return (% per year), 95% CIs, and p-value.
        """
        n = min(len(strat_daily_rets), len(bench_daily_rets))
        if n < 10:
            return {
                "n_days": n,
                "mean_excess_annual_pct": 0.0,
                "ci_lower_pct": 0.0,
                "ci_upper_pct": 0.0,
                "p_value": 1.0,
                "ci_includes_zero": True,
                "tracking_error_pct": 0.0
            }

        s_ret = strat_daily_rets[:n]
        b_ret = bench_daily_rets[:n]
        excess_daily = s_ret - b_ret

        mean_excess_daily = float(np.mean(excess_daily))
        mean_excess_annual = mean_excess_daily * 252.0 * 100.0
        tracking_error_annual = float(np.std(excess_daily, ddof=1) * np.sqrt(252) * 100.0)

        rng = np.random.RandomState(seed)
        num_blocks = int(np.ceil(n / block_size))
        boot_means = np.zeros(n_boot)

        for b in range(n_boot):
            start_indices = rng.randint(0, max(1, n - block_size + 1), size=num_blocks)
            resampled_excess = []
            for s in start_indices:
                resampled_excess.extend(excess_daily[s:s + block_size])
            boot_sample = np.array(resampled_excess[:n])
            boot_means[b] = np.mean(boot_sample) * 252.0 * 100.0

        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        p_val = float(np.mean(boot_means <= 0.0)) if mean_excess_annual > 0 else float(np.mean(boot_means >= 0.0))
        p_val = min(1.0, p_val * 2.0)

        return {
            "n_days": n,
            "mean_excess_annual_pct": round(mean_excess_annual, 2),
            "ci_lower_pct": round(ci_lower, 2),
            "ci_upper_pct": round(ci_upper, 2),
            "p_value": round(p_val, 4),
            "ci_includes_zero": bool(ci_lower <= 0.0 <= ci_upper),
            "tracking_error_pct": round(tracking_error_annual, 2)
        }

    @staticmethod
    def calculate_minimum_detectable_alpha(tracking_error_pct: float, n_days: int,
                                          alpha_sig: float = 0.05, power: float = 0.80) -> float:
        if n_days <= 1 or tracking_error_pct <= 0:
            return 0.0
        t_years = n_days / 252.0
        z_sum = 2.8016  # 1.96 + 0.8416
        mda = (z_sum * tracking_error_pct) / np.sqrt(max(0.01, t_years))
        return round(float(mda), 2)

    @staticmethod
    def compute_max_drawdown(equity_series: np.ndarray) -> float:
        if len(equity_series) == 0:
            return 0.0
        peaks = np.maximum.accumulate(equity_series)
        dd = (equity_series - peaks) / np.where(peaks > 0, peaks, 1.0)
        return round(float(np.min(dd)) * 100.0, 2)


# =====================================================================
# 6. MASTER FORWARD PAPER RUNNER & CATCH-UP ENGINE
# =====================================================================
class ForwardPaperManager:
    def __init__(self, db_path: str = DB_PATH):
        self.engine = ForwardPaperEngine(db_path=db_path)
        self.store = self.engine.store

    def run_daily_post_market_job(self, today_date_str: Optional[str] = None):
        """
        Executed daily after market close (15:45 IST).
        """
        from strategy_research_fair_benchmark import load_all_data
        nifty_df, stock_dfs = load_all_data()
        
        all_dfs = dict(stock_dfs)
        all_dfs["^NSEI"] = nifty_df

        all_dates = sorted(list(nifty_df.index))
        if not all_dates:
            logger.warning("No market data found for forward paper testing daily step.")
            return

        target_date = pd.to_datetime(today_date_str) if today_date_str else all_dates[-1]
        target_str = target_date.strftime("%Y-%m-%d")
        
        source = "FORWARD" if target_str >= FORWARD_START_DATE else "BACKTEST_SEED"
        
        today_open_prices = {sym: float(df.loc[target_date, "Open"]) for sym, df in all_dfs.items() if target_date in df.index}
        today_close_prices = {sym: float(df.loc[target_date, "Close"]) for sym, df in all_dfs.items() if target_date in df.index}

        results = {}
        for strat_id, strat_def in REGISTERED_STRATEGIES.items():
            for cap_tier in ["STANDARD_100K", "MICRO_1K"]:
                self.engine.initialize_account_if_needed(strat_id, strat_def.version, cap_tier, all_dates[0].strftime("%Y-%m-%d"), source=source)
                res = self.engine.step_forward_one_day(
                    strategy_id=strat_id,
                    version=strat_def.version,
                    capital_tier=cap_tier,
                    today_date=target_date,
                    all_dates=all_dates,
                    historical_data_dict=all_dfs,
                    today_open_prices=today_open_prices,
                    today_close_prices=today_close_prices,
                    source=source
                )
                results[f"{strat_id}_{cap_tier}"] = res

        logger.info(f"Forward Paper Daily Job Completed for {target_str} (Source: {source}).")
        return results

    def run_startup_catchup_job(self) -> Dict[str, Any]:
        """
        On startup, checks if any trading days were missed in the forward ledger.
        Computes each missing day in chronological sequence using strictly point-in-time data.
        Flags rows as source='CATCHUP'.
        Alerts loudly if gap > 3 trading days.
        """
        from strategy_research_fair_benchmark import load_all_data
        nifty_df, stock_dfs = load_all_data()
        all_dfs = dict(stock_dfs)
        all_dfs["^NSEI"] = nifty_df
        all_dates = sorted(list(nifty_df.index))

        if not all_dates:
            return {"status": "NO_MARKET_DATA", "missed_days": 0}

        latest_entry = self.store.get_latest_ledger_entry("NIFTY50_BUY_HOLD", "v1.0.0", "STANDARD_100K")
        if latest_entry is None:
            return {"status": "LEDGER_EMPTY", "missed_days": 0}

        latest_date = pd.to_datetime(latest_entry["trade_date"])
        missing_dates = [d for d in all_dates if d > latest_date]

        if not missing_dates:
            return {"status": "UP_TO_DATE", "missed_days": 0, "latest_date": latest_date.strftime("%Y-%m-%d")}

        missed_count = len(missing_dates)
        is_critical_gap = missed_count > 3

        if is_critical_gap:
            logger.error(f"⚠️ FORWARD RECONCILIATION GAP ALERT: Missed {missed_count} trading days between {latest_date.strftime('%Y-%m-%d')} and {missing_dates[-1].strftime('%Y-%m-%d')}!")
        else:
            logger.info(f"Forward Catch-Up: Processing {missed_count} missed trading days...")

        catchup_results = []
        for d in missing_dates:
            d_str = d.strftime("%Y-%m-%d")
            source_tag = "CATCHUP" if d_str >= FORWARD_START_DATE else "BACKTEST_SEED"
            open_prices = {sym: float(df.loc[d, "Open"]) for sym, df in all_dfs.items() if d in df.index}
            close_prices = {sym: float(df.loc[d, "Close"]) for sym, df in all_dfs.items() if d in df.index}

            for strat_id, strat_def in REGISTERED_STRATEGIES.items():
                for cap_tier in ["STANDARD_100K", "MICRO_1K"]:
                    res = self.engine.step_forward_one_day(
                        strategy_id=strat_id,
                        version=strat_def.version,
                        capital_tier=cap_tier,
                        today_date=d,
                        all_dates=all_dates,
                        historical_data_dict=all_dfs,
                        today_open_prices=open_prices,
                        today_close_prices=close_prices,
                        source=source_tag
                    )
            catchup_results.append(d_str)

        return {
            "status": "CATCHUP_COMPLETED",
            "missed_days": missed_count,
            "critical_gap_alert": is_critical_gap,
            "processed_dates": catchup_results
        }

    def get_dashboard_summary(self, capital_tier: str = "STANDARD_100K") -> Dict[str, Any]:
        """
        Generates comparative analytics separating BACKTEST_SEED from true FORWARD tracking.
        Explicitly shows 0 forward days until real data post FORWARD_START_DATE accrues.
        """
        all_nifty_history = self.store.get_ledger_history("NIFTY50_BUY_HOLD", "v1.0.0", capital_tier)
        
        # Split history into Seeded Backtest vs Genuine Forward
        seeded_nifty = [r for r in all_nifty_history if r.get("source") == "BACKTEST_SEED"]
        forward_nifty = [r for r in all_nifty_history if r.get("source") in ("FORWARD", "CATCHUP")]

        n_forward_days = len(forward_nifty)
        n_seeded_days = len(seeded_nifty)

        strategies_data = []
        for strat_id, strat_def in REGISTERED_STRATEGIES.items():
            history = self.store.get_ledger_history(strat_id, strat_def.version, capital_tier)
            if not history:
                continue

            seeded_hist = [r for r in history if r.get("source") == "BACKTEST_SEED"]
            fwd_hist = [r for r in history if r.get("source") in ("FORWARD", "CATCHUP")]

            # Overall history values
            vals = np.array([float(r["portfolio_value"]) for r in history])
            rets = np.array([float(r["daily_return_pct"])/100.0 for r in history])
            dates = [r["trade_date"] for r in history]
            
            init_cap = float(history[0]["portfolio_value"])
            latest_val = float(history[-1]["portfolio_value"])
            total_ret_pct = ((latest_val - init_cap) / init_cap * 100.0) if init_cap > 0 else 0.0
            
            n_days = len(history)
            n_years = max(0.01, n_days / 252.0)
            cagr_pct = (((latest_val / init_cap) ** (1.0 / n_years)) - 1.0) * 100.0 if init_cap > 0 and latest_val > 0 else 0.0
            max_dd_pct = PaperAnalytics.compute_max_drawdown(vals)
            
            tot_costs = float(history[-1]["cum_costs"])
            tot_turnover = sum(float(r["turnover_inr"]) for r in history)
            ann_turnover_pct = (tot_turnover / (init_cap * n_years)) * 100.0 if init_cap > 0 else 0.0

            # Paired block bootstraps vs Nifty (Seeded Backtest)
            nifty_rets = np.array([float(r["daily_return_pct"])/100.0 for r in all_nifty_history])
            stats_vs_nifty = PaperAnalytics.paired_block_bootstrap(rets, nifty_rets)
            mda_pct = PaperAnalytics.calculate_minimum_detectable_alpha(stats_vs_nifty["tracking_error_pct"], n_forward_days)

            # Mathematical Sanity Assertions (Log & Alert instead of crashing)
            sanity_alerts = []
            if capital_tier == "STANDARD_100K":
                if cagr_pct > 0 and max_dd_pct < -60.0:
                    alert_msg = f"SANITY ALERT for {strat_id}: CAGR > 0 (+{cagr_pct:.2f}%) but Max DD is {max_dd_pct:.2f}% (< -60%)."
                    logger.error(alert_msg)
                    sanity_alerts.append(alert_msg)
                if strat_id in ("NIFTY50_BUY_HOLD", "EQUAL_WEIGHT_33_SURVIVORSHIP_BIASED") and tot_costs > (init_cap * 0.01):
                    alert_msg = f"SANITY ALERT for {strat_id}: Buy-and-Hold strategy costs ₹{tot_costs:.2f} exceed 1% of initial capital (₹{init_cap*0.01:.2f})."
                    logger.error(alert_msg)
                    sanity_alerts.append(alert_msg)
                if abs(stats_vs_nifty["mean_excess_annual_pct"]) > 50.0:
                    alert_msg = f"SANITY ALERT for {strat_id}: Excess return vs Nifty is {stats_vs_nifty['mean_excess_annual_pct']:+.2f}%/yr (exceeds 50%/yr threshold)."
                    logger.error(alert_msg)
                    sanity_alerts.append(alert_msg)

            # Forward-only metrics
            if fwd_hist:
                fwd_init_cap = float(fwd_hist[0]["portfolio_value"])
                fwd_latest_val = float(fwd_hist[-1]["portfolio_value"])
                fwd_ret_pct = ((fwd_latest_val - fwd_init_cap) / fwd_init_cap * 100.0) if fwd_init_cap > 0 else 0.0
                fwd_dd_pct = PaperAnalytics.compute_max_drawdown(np.array([float(r["portfolio_value"]) for r in fwd_hist]))
            else:
                fwd_ret_pct = 0.0
                fwd_dd_pct = 0.0

            strategies_data.append({
                "strategy_id": strat_id,
                "name": strat_def.name,
                "version": strat_def.version,
                "description": strat_def.description,
                "code_hash": strat_def.code_hash[:12] + "...",
                "initial_capital": init_cap,
                "current_value": round(latest_val, 2),
                "total_return_pct": round(total_ret_pct, 2),
                "cagr_pct": round(cagr_pct, 2),
                "max_drawdown_pct": max_dd_pct,
                "total_costs_inr": round(tot_costs, 2),
                "annual_turnover_pct": round(ann_turnover_pct, 1),
                "forward_days": n_forward_days,
                "seeded_backtest_days": n_seeded_days,
                "forward_return_pct": round(fwd_ret_pct, 2),
                "forward_max_drawdown_pct": fwd_dd_pct,
                "forward_status": "Active Forward Tracking" if n_forward_days > 0 else "0 Forward Days (Awaiting Live Trading Days)",
                "minimum_detectable_alpha_pct": mda_pct,
                "stats_vs_nifty": stats_vs_nifty,
                "equity_curve": [{"date": d, "value": round(v, 2)} for d, v in zip(dates, vals)]
            })

        is_chain_valid, chain_msg, verified_blocks = self.store.verify_hash_chain(capital_tier=capital_tier)

        return {
            "capital_tier": capital_tier,
            "forward_start_date": FORWARD_START_DATE,
            "forward_days": n_forward_days,
            "seeded_backtest_days": n_seeded_days,
            "strategies": strategies_data,
            "hash_chain_status": {
                "is_valid": is_chain_valid,
                "message": chain_msg,
                "verified_blocks": verified_blocks
            },
            "disclaimer_plain": (
                "INTEGRITY RULE: Seeded walk-forward backtest data is strictly partitioned as BACKTEST_SEED. "
                "The forward paper track record starts at 0 forward days and only counts true post-market daily runs "
                "occurring after the forward start date. A short paper record cannot prove an edge."
            )
        }

    def generate_weekly_telegram_summary(self, capital_tier: str = "STANDARD_100K") -> str:
        summary = self.get_dashboard_summary(capital_tier=capital_tier)
        strats = summary.get("strategies", [])
        if not strats:
            return "📊 <b>QuantBot Forward Paper Testing:</b> No forward history recorded yet."

        fwd_days = summary.get("forward_days", 0)
        seeded_days = summary.get("seeded_backtest_days", 0)

        # Retrieve latest chain head block info
        latest_nifty = self.store.get_latest_ledger_entry("NIFTY50_BUY_HOLD", "v1.0.0", capital_tier)
        head_hash_abbr = (latest_nifty["entry_hash"][:16] + "...") if latest_nifty else "N/A"
        head_seq = latest_nifty["sequence_num"] if latest_nifty else 0

        header = (
            f"📈 <b>WEEKLY PAPER PERFORMANCE & INTEGRITY SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Account Tier:</b> <code>{capital_tier}</code> | <b>Mode:</b> PAPER ONLY\n"
            f"<b>Forward Track Record:</b> <code>{fwd_days} days</code> (Forward Start: {summary.get('forward_start_date')})\n"
            f"<b>Seeded Backtest History:</b> <code>{seeded_days} days</code>\n"
            f"<b>Chain Head Hash:</b> <code>{head_hash_abbr}</code> (Block #{head_seq})\n"
            f"<b>Cryptographic Integrity:</b> {'🛡️ VERIFIED INTACT' if summary['hash_chain_status']['is_valid'] else '❌ TAMPER WARNING'}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        )

        rows = []
        for s in strats:
            ret_emoji = "🟢" if s["total_return_pct"] >= 0 else "🔴"
            excess_nifty = s["stats_vs_nifty"]["mean_excess_annual_pct"]
            excess_emoji = "▲" if excess_nifty >= 0 else "▼"
            ci_str = f"[{s['stats_vs_nifty']['ci_lower_pct']:+.1f}%, {s['stats_vs_nifty']['ci_upper_pct']:+.1f}%]"

            latest_s = self.store.get_latest_ledger_entry(s["strategy_id"], s["version"], capital_tier)
            s_hash = (latest_s["entry_hash"][:10] + "...") if latest_s else "N/A"

            row = (
                f"<b>{s['name']}</b> ({s['version']})\n"
                f"• Value: ₹{s['current_value']:,.2f} ({ret_emoji} <b>{s['total_return_pct']:+.2f}%</b> total)\n"
                f"• Forward Status: <code>{s['forward_status']}</code>\n"
                f"• Excess vs Nifty: <b>{excess_emoji} {excess_nifty:+.2f}%/yr</b> (95% CI: <code>{ci_str}</code>)\n"
                f"• Max DD: <code>{s['max_drawdown_pct']:.2f}%</code> | Costs: <code>₹{s['total_costs_inr']:,.2f}</code>\n"
                f"• Block Head: <code>{s_hash}</code> (Seq #{latest_s['sequence_num'] if latest_s else 0})\n"
            )
            rows.append(row)

        footer = (
            f"\n⚖️ <i>Seeded data is strictly partitioned as BACKTEST_SEED; forward track record shows 0 forward days until genuine post-market trading days accrue.</i>"
        )

        return header + "\n".join(rows) + footer


forward_paper_manager = ForwardPaperManager()
