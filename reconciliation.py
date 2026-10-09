"""
Startup Position & Order Reconciliation Module for QuantBot Pro
Safeguards trading state on bot restart by reconciling broker-side open positions
and pending orders against local SQLite persistence tables.
Detects orphan positions, ghost trades, and missing stop-loss protections.
"""

import logging
import time
from datetime import datetime
from risk_store import risk_store
from angel_one_client import angel_client
from telegram_notifier import telegram_notifier

logger = logging.getLogger("quantbot.reconciliation")

class StateReconciler:
    def __init__(self, broker=None, store=None, notifier=None):
        self.broker = broker or angel_client
        self.store = store or risk_store
        self.notifier = notifier or telegram_notifier

    def reconcile_on_startup(self, trading_mode="PAPER"):
        """
        Executes full two-way reconciliation on application startup.
        Returns a detailed reconciliation summary dictionary.
        """
        logger.info(f"Initiating Startup State Reconciliation in {trading_mode} mode...")
        report = {
            "timestamp": datetime.now().isoformat(),
            "trading_mode": trading_mode,
            "broker_connected": False,
            "broker_positions_count": 0,
            "local_open_trades_count": 0,
            "matched_positions": [],
            "orphan_broker_positions": [],
            "ghost_local_trades": [],
            "missing_stop_loss_warnings": [],
            "status": "SUCCESS"
        }

        try:
            # 1. Fetch Local Open Trades from SQLite
            local_trades = self.store.get_all_trades(status="OPEN", limit=100)
            local_map = {t["symbol"]: t for t in local_trades}
            report["local_open_trades_count"] = len(local_trades)

            # 2. Fetch Broker Positions
            broker_positions = []
            if trading_mode.upper() == "LIVE" and self.broker.is_connected:
                report["broker_connected"] = True
                pos_resp = self.broker.get_positions()
                if pos_resp.get("status") and isinstance(pos_resp.get("data"), list):
                    # Filter for net positive/negative open positions
                    broker_positions = [
                        p for p in pos_resp["data"] 
                        if int(p.get("netqty", 0) or 0) != 0
                    ]
            else:
                # In PAPER mode, local DB is the primary source of truth
                report["broker_connected"] = True
                broker_positions = []

            report["broker_positions_count"] = len(broker_positions)
            broker_map = {p.get("tradingsymbol", ""): p for p in broker_positions}

            # 3. Two-Way Cross Matching (For Live Broker Mode)
            if trading_mode.upper() == "LIVE" and self.broker.is_connected:
                # Check Local Trades against Broker Positions
                for sym, l_trade in local_map.items():
                    if sym in broker_map:
                        report["matched_positions"].append(sym)
                    else:
                        # Local trade is OPEN, but not in broker net positions -> Ghost Trade
                        logger.warning(f"Reconciliation: Ghost trade detected for {sym}. Marking CLOSED.")
                        self.store.record_trade_exit(
                            trade_id=l_trade["trade_id"],
                            exit_price=l_trade["entry_price"], # fallback
                            exit_reason="RECONCILED_CLOSED_AT_BROKER"
                        )
                        report["ghost_local_trades"].append(sym)

                # Check Broker Positions against Local Trades
                for sym, b_pos in broker_map.items():
                    if sym not in local_map:
                        qty = abs(int(b_pos.get("netqty", 0)))
                        action = "BUY" if int(b_pos.get("netqty", 0)) > 0 else "SELL"
                        avg_price = float(b_pos.get("avgprice", 0.0) or 0.0)
                        
                        logger.warning(f"Reconciliation: Orphan broker position detected for {sym} (Qty: {qty}). Adopting into DB.")
                        
                        # Adopt into SQLite
                        trade_id = f"RECON_{int(time.time())}_{sym.replace('.', '_')}"
                        self.store.record_trade_entry(
                            trade_id=trade_id,
                            symbol=sym,
                            action=action,
                            entry_price=avg_price,
                            quantity=qty,
                            stop_loss=avg_price * 0.98 if action == "BUY" else avg_price * 1.02,
                            take_profit=avg_price * 1.04 if action == "BUY" else avg_price * 0.96,
                            mode="LIVE",
                            strategy_tag="ADOPTED_ORPHAN",
                            broker_order_id=b_pos.get("orderid", "RECONCILED")
                        )
                        report["orphan_broker_positions"].append(sym)

                # Check for Protective Stop Loss Orders
                order_book_resp = self.broker.get_order_book()
                pending_orders = []
                if order_book_resp.get("status") and isinstance(order_book_resp.get("data"), list):
                    pending_orders = [
                        o for o in order_book_resp["data"]
                        if o.get("status", "").upper() in ["OPEN", "TRIGGER_PENDING", "PENDING"]
                    ]
                
                open_sl_symbols = {o.get("tradingsymbol") for o in pending_orders if "STOPLOSS" in o.get("ordertype", "").upper() or "SL" in o.get("ordertype", "").upper()}
                for sym in broker_map:
                    if sym not in open_sl_symbols:
                        report["missing_stop_loss_warnings"].append(sym)
                        logger.warning(f"Reconciliation Alert: Position in {sym} has NO active broker-side Stop-Loss order!")

            logger.info(f"State Reconciliation Finished: {len(report['matched_positions'])} matched, "
                        f"{len(report['orphan_broker_positions'])} orphans adopted, "
                        f"{len(report['ghost_local_trades'])} ghosts closed.")

        except Exception as e:
            logger.error(f"Reconciliation encountered error: {type(e).__name__}: {str(e)}", exc_info=True)
            report["status"] = "ERROR"
            report["error_message"] = str(e)
            self.notifier.send_error_alert("ReconciliationError", str(e))

        return report

# Singleton Instance
state_reconciler = StateReconciler()
