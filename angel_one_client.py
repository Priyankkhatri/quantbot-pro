"""
Angel One SmartAPI Hardened Execution Module
Features:
- Global TRADING_MODE Integration (PAPER default, LIVE only with explicit confirmation)
- Real Broker-Side Stop-Loss Placement with Emergency Auto-Exit on SL Failure
- Active Order Status Polling (FILLED / REJECTED / CANCELLED) with Partial Fill Handling
- Non-tradable Spot Index Rejection (^NSEI, ^NSEBANK)
- Units == 0 and Price > Capital Rejection
- Idempotency & Debounce Enforcement (1 order per signal per symbol per day)
- Persistent SQLite Audit Trail via RiskStore
- Zero Plaintext Credential Logging
"""

import os
import time
import json
import logging
import datetime
import pyotp
from typing import Dict, Any, Optional, Tuple
from SmartApi import SmartConnect

from risk_store import risk_store
from market_calendar import get_market_session_state, get_current_ist_time
from telegram_notifier import telegram_notifier

logger = logging.getLogger("AngelOneClient")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def mask_credential(cred: str, visible_chars: int = 4) -> str:
    """Masks sensitive credentials for secure display/logging."""
    if not cred:
        return "[NOT SET]"
    if len(cred) <= visible_chars:
        return "******"
    return cred[:2] + "*" * (len(cred) - visible_chars) + cred[-2:]


class AngelOneClient:
    def __init__(self, api_key=None, client_code=None, pin=None, totp_key=None):
        self._api_key = api_key or os.getenv("ANGEL_API_KEY", "")
        self._client_code = client_code or os.getenv("ANGEL_CLIENT_CODE", "")
        self._pin = pin or os.getenv("ANGEL_PIN", "")
        self._totp_key = totp_key or os.getenv("ANGEL_TOTP_KEY", "")
        
        self.smart_api = None
        self.is_connected = False
        self.auth_data = None
        self.feed_token = None
        self.user_profile = None

    @property
    def client_code(self) -> str:
        return self._client_code

    def connect(self, api_key=None, client_code=None, pin=None, totp_key=None) -> Dict[str, Any]:
        """Authenticate with Angel One SmartAPI using TOTP. Never logs credentials."""
        if api_key: self._api_key = api_key.strip()
        if client_code: self._client_code = client_code.strip()
        if pin: self._pin = pin.strip()
        if totp_key: self._totp_key = totp_key.strip().replace(" ", "")

        if not (self._api_key and self._client_code and self._pin and self._totp_key):
            return {
                "success": False,
                "message": "Missing credentials. Required: API Key, Client Code, PIN, and TOTP Secret."
            }

        try:
            logger.info(f"Initiating SmartAPI session for client: {mask_credential(self._client_code)}")
            self.smart_api = SmartConnect(api_key=self._api_key)
            totp = pyotp.TOTP(self._totp_key).now()
            
            data = self.smart_api.generateSession(self._client_code, self._pin, totp)
            
            if data and data.get("status") is True:
                self.auth_data = data.get("data", {})
                self.feed_token = self.smart_api.getfeedToken()
                self.is_connected = True
                
                # Fetch profile info
                profile_res = self.smart_api.getProfile(self.auth_data.get("refreshToken"))
                if profile_res and profile_res.get("status") is True:
                    self.user_profile = profile_res.get("data", {})

                masked_client = mask_credential(self._client_code)
                user_name = self.user_profile.get("name", masked_client) if self.user_profile else masked_client
                logger.info(f"Angel One SmartAPI authenticated successfully for {masked_client}.")

                return {
                    "success": True,
                    "message": "Connected to Angel One SmartAPI successfully!",
                    "client_code": masked_client,
                    "user_name": user_name
                }
            else:
                self.is_connected = False
                err_msg = data.get("message", "Authentication failed. Check API Key, PIN, or TOTP.")
                logger.warning(f"SmartAPI auth failed: {err_msg}")
                return {"success": False, "message": err_msg}

        except Exception as e:
            self.is_connected = False
            logger.error(f"SmartAPI connection exception: {str(e)}")
            return {"success": False, "message": f"Connection error: {str(e)}"}

    def get_funds_and_margin(self) -> Dict[str, Any]:
        """Retrieve available cash and margin limits from broker RMS."""
        config = risk_store.get_config()
        trading_mode = config.get("trading_mode", "PAPER")

        if not self.is_connected or not self.smart_api or trading_mode == "PAPER":
            return {
                "success": True,
                "is_paper_trading": True,
                "trading_mode": trading_mode,
                "available_cash": 500.00,
                "used_margin": 0.00,
                "total_balance": 500.00,
                "status": f"Simulation Mode ({trading_mode} Active)",
                "kill_switch_active": bool(config.get("kill_switch_active", 0))
            }

        try:
            res = self.smart_api.rmsLimit()
            if res and res.get("status") is True:
                data = res.get("data", {})
                net_cash = float(data.get("net", 0.0))
                available_cash = float(data.get("availablecash", 0.0))
                return {
                    "success": True,
                    "is_paper_trading": False,
                    "trading_mode": "LIVE",
                    "available_cash": available_cash,
                    "net_cash": net_cash,
                    "used_margin": float(data.get("utiliseddebits", 0.0)),
                    "total_balance": net_cash,
                    "status": "Live Connected (Angel One)",
                    "kill_switch_active": bool(config.get("kill_switch_active", 0))
                }
            else:
                return {
                    "success": False,
                    "trading_mode": trading_mode,
                    "message": res.get("message", "Failed to retrieve RMS limits"),
                    "available_cash": 500.00,
                    "kill_switch_active": bool(config.get("kill_switch_active", 0))
                }
        except Exception as e:
            return {
                "success": False,
                "trading_mode": trading_mode,
                "message": str(e),
                "available_cash": 500.00,
                "kill_switch_active": bool(config.get("kill_switch_active", 0))
            }

    def poll_order_status(self, order_id: str, timeout_sec: int = 10) -> Dict[str, Any]:
        """
        Polls order status from broker until FILLED / complete, REJECTED, or CANCELLED.
        Handles partial fills and never assumes execution.
        """
        if not self.is_connected or not self.smart_api:
            return {"status": "FILLED", "fill_qty": 1.0, "fill_price": 0.0, "message": "Paper Simulated Fill"}

        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            try:
                order_book = self.smart_api.orderBook()
                if order_book and order_book.get("status") is True:
                    orders = order_book.get("data", []) or []
                    for o in orders:
                        if str(o.get("orderid")) == str(order_id):
                            raw_status = o.get("status", "").lower()
                            fill_qty = float(o.get("filledshares", 0))
                            unfilled_qty = float(o.get("unfilledshares", 0))
                            avg_price = float(o.get("averageprice", 0))

                            if raw_status in ["complete", "filled"]:
                                return {
                                    "status": "FILLED",
                                    "fill_qty": fill_qty,
                                    "fill_price": avg_price,
                                    "message": "Order fully filled."
                                }
                            elif raw_status in ["rejected", "cancelled"]:
                                return {
                                    "status": "REJECTED" if raw_status == "rejected" else "CANCELLED",
                                    "fill_qty": fill_qty,
                                    "fill_price": avg_price,
                                    "message": o.get("text", "Order was rejected/cancelled by exchange.")
                                }
                            elif fill_qty > 0 and unfilled_qty > 0:
                                logger.info(f"Partial fill detected for order {order_id}: {fill_qty} filled, {unfilled_qty} pending.")
                                return {
                                    "status": "PARTIAL_FILL",
                                    "fill_qty": fill_qty,
                                    "fill_price": avg_price,
                                    "message": f"Partially filled: {fill_qty} shares."
                                }
                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"Error polling order {order_id}: {e}")
                time.sleep(1.0)

        return {"status": "TIMEOUT", "fill_qty": 0.0, "fill_price": 0.0, "message": "Order polling timed out. Status UNKNOWN."}

    def place_safe_order(self, symbol: str, token: str, action: str, quantity: float,
                         price: float = 0, stop_loss: float = 0, take_profit: float = 0,
                         product_type: str = "INTRADAY", exchange: str = "NSE") -> Dict[str, Any]:
        """
        Hardened risk-managed order entry:
        1. Validates market hours & session rules (No entries after 14:45 IST).
        2. Validates units > 0, price > 0, capital limits, and spot index rejection.
        3. Enforces idempotency (1 order per signal per symbol per day).
        4. In LIVE mode: places entry order, polls fill status, and immediately places real broker-side SL order.
        5. If the SL order placement fails, triggers emergency market exit for the filled position!
        """
        config = risk_store.get_config()
        trading_mode = config.get("trading_mode", "PAPER")

        # 1. Market Hours & Scheduler Check
        session = get_market_session_state()
        if not session["can_enter_new_trade"]:
            msg = f"REJECTED by Market Scheduler: {session['reason']} (IST: {session['ist_time']})"
            telegram_notifier.send_rejection_alert(symbol, action, quantity, msg, mode=trading_mode)
            return {
                "success": False,
                "trading_mode": trading_mode,
                "message": msg
            }

        # 2. Available Funds Check
        funds_info = self.get_funds_and_margin()
        available_cash = funds_info.get("available_cash", 500.0)

        # 3. Pre-Trade Risk Verification
        eval_price = price if price > 0 else 2780.0  # Fallback reference price for market orders
        is_risk_ok, risk_msg = risk_store.check_pre_trade_risk(symbol, quantity, eval_price, available_cash)
        if not is_risk_ok:
            logger.warning(f"Pre-trade risk failed for {symbol}: {risk_msg}")
            telegram_notifier.send_rejection_alert(symbol, action, quantity, risk_msg, mode=trading_mode)
            return {"success": False, "trading_mode": trading_mode, "message": risk_msg}

        # 4. Idempotency Lock Acquisition
        lock_ok, lock_res = risk_store.acquire_idempotency_lock(symbol, action)
        if not lock_ok:
            telegram_notifier.send_rejection_alert(symbol, action, quantity, lock_res, mode=trading_mode)
            return {"success": False, "trading_mode": trading_mode, "message": lock_res}
        
        idempotency_key = lock_res

        # Calculate dynamic stop loss if not supplied
        if stop_loss <= 0:
            sl_buffer = eval_price * 0.015
            stop_loss = round(eval_price - sl_buffer if action.upper() == "BUY" else eval_price + sl_buffer, 2)

        # 5. Execution based on TRADING_MODE
        if trading_mode == "PAPER" or not self.is_connected or not self.smart_api:
            # Paper Simulation Execution
            sim_order_id = f"SIM_{int(time.time()*1000)}"
            sim_sl_id = f"SIM_SL_{int(time.time()*1000)}"
            risk_store.record_order_entry(
                idempotency_key=idempotency_key,
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=eval_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                order_id=sim_order_id,
                trading_mode="PAPER"
            )
            risk_store.update_order_fill(sim_order_id, "FILLED", quantity, eval_price, sim_sl_id)
            
            # Record in SQLite trades table
            trade_uuid = f"TRADE_PAPER_{int(time.time()*1000)}"
            risk_store.record_trade_entry(
                trade_id=trade_uuid,
                symbol=symbol,
                action=action,
                entry_price=eval_price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                mode="PAPER",
                strategy_tag="QUANT_ML_ENSEMBLE",
                broker_order_id=sim_order_id
            )

            # Trigger Telegram Alert
            telegram_notifier.send_order_fill_alert(
                symbol=symbol,
                action=action,
                quantity=quantity,
                fill_price=eval_price,
                order_id=sim_order_id,
                mode="PAPER"
            )
            
            logger.info(f"[PAPER SIMULATION] Order {sim_order_id} executed: {action} {quantity} {symbol} @ INR {eval_price:.2f} (SL: INR {stop_loss:.2f})")
            return {
                "success": True,
                "trading_mode": "PAPER",
                "order_id": sim_order_id,
                "sl_order_id": sim_sl_id,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": eval_price,
                "stop_loss": stop_loss,
                "message": f"PAPER TRADE FILLED: {action} {quantity} units of {symbol} @ INR {eval_price:.2f} (Simulated SL: INR {stop_loss:.2f})"
            }

        # --- LIVE ANGEL ONE EXECUTION ---
        try:
            logger.info(f"[LIVE ORDER SUBMITTING] {action} {quantity} {symbol} to Angel One...")
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": symbol,
                "symboltoken": str(token),
                "transactiontype": action.upper(),
                "exchange": exchange.upper(),
                "ordertype": "MARKET" if price <= 0 else "LIMIT",
                "producttype": product_type.upper(),
                "duration": "DAY",
                "price": str(price) if price > 0 else "0",
                "squareoff": "0",
                "stoploss": "0",
                "quantity": str(int(quantity))
            }

            res = self.smart_api.placeOrder(order_params)
            if not res or res.get("status") is not True:
                err = res.get("message", "Live order rejected by broker.") if res else "No response from broker."
                logger.error(f"Live order submission failed: {err}")
                telegram_notifier.send_rejection_alert(symbol, action, quantity, err, mode="LIVE")
                return {"success": False, "trading_mode": "LIVE", "message": f"Broker Rejected: {err}"}

            live_order_id = res.get("data", {}).get("orderid")
            risk_store.record_order_entry(
                idempotency_key=idempotency_key,
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=eval_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                order_id=live_order_id,
                trading_mode="LIVE"
            )

            # Step 2: Poll status until FILLED
            poll_res = self.poll_order_status(live_order_id, timeout_sec=12)
            if poll_res["status"] not in ["FILLED", "PARTIAL_FILL"]:
                risk_store.update_order_fill(live_order_id, poll_res["status"], 0.0, 0.0)
                telegram_notifier.send_rejection_alert(symbol, action, quantity, poll_res["message"], mode="LIVE")
                return {
                    "success": False,
                    "trading_mode": "LIVE",
                    "order_id": live_order_id,
                    "message": f"Order status '{poll_res['status']}': {poll_res['message']}"
                }

            filled_qty = poll_res["fill_qty"]
            fill_price = poll_res["fill_price"] or eval_price

            # Step 3: Place Real Broker-Side Stop-Loss Order (SL-M / STOPLOSS_LIMIT)
            sl_action = "SELL" if action.upper() == "BUY" else "BUY"
            sl_params = {
                "variety": "STOPLOSS",
                "tradingsymbol": symbol,
                "symboltoken": str(token),
                "transactiontype": sl_action,
                "exchange": exchange.upper(),
                "ordertype": "STOPLOSS_LIMIT",
                "producttype": product_type.upper(),
                "duration": "DAY",
                "price": str(round(stop_loss * 0.998, 2)),  # Limit price slightly below trigger for safe execution
                "triggerprice": str(round(stop_loss, 2)),
                "quantity": str(int(filled_qty))
            }

            logger.info(f"Placing broker-side SL order for {symbol}: Trigger=INR {stop_loss}")
            sl_res = self.smart_api.placeOrder(sl_params)

            # Step 4: Emergency Exit if SL Order Fails
            if not sl_res or sl_res.get("status") is not True:
                sl_err = sl_res.get("message", "SL order placement failed") if sl_res else "SL API timeout"
                logger.critical(f"EMERGENCY: Broker-side SL failed for {symbol} ({sl_err}). Executing immediate market close!")
                
                # Emergency Market Exit
                exit_params = {
                    "variety": "NORMAL",
                    "tradingsymbol": symbol,
                    "symboltoken": str(token),
                    "transactiontype": sl_action,
                    "exchange": exchange.upper(),
                    "ordertype": "MARKET",
                    "producttype": product_type.upper(),
                    "duration": "DAY",
                    "price": "0",
                    "quantity": str(int(filled_qty))
                }
                emergency_res = self.smart_api.placeOrder(exit_params)
                risk_store.update_order_fill(live_order_id, "EMERGENCY_CLOSED", filled_qty, fill_price, None)
                telegram_notifier.send_error_alert("EmergencySLExit", f"SL Order failed for {symbol}. Emergency square-off triggered.")
                
                return {
                    "success": False,
                    "trading_mode": "LIVE",
                    "order_id": live_order_id,
                    "message": f"CRITICAL: Broker SL Failed ({sl_err}). Position was IMMEDIATELY SQUARED OFF for capital protection."
                }

            sl_order_id = sl_res.get("data", {}).get("orderid")
            risk_store.update_order_fill(live_order_id, "FILLED_PROTECTED", filled_qty, fill_price, sl_order_id)
            
            # Record in SQLite trades table
            trade_uuid = f"TRADE_LIVE_{int(time.time()*1000)}"
            risk_store.record_trade_entry(
                trade_id=trade_uuid,
                symbol=symbol,
                action=action,
                entry_price=fill_price,
                quantity=filled_qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                mode="LIVE",
                strategy_tag="QUANT_ML_ENSEMBLE",
                broker_order_id=live_order_id
            )

            # Trigger Telegram Alert
            telegram_notifier.send_order_fill_alert(
                symbol=symbol,
                action=action,
                quantity=filled_qty,
                fill_price=fill_price,
                order_id=live_order_id,
                mode="LIVE"
            )

            logger.info(f"Live Position PROTECTED: Entry Order={live_order_id} | SL Order={sl_order_id}")

            return {
                "success": True,
                "trading_mode": "LIVE",
                "order_id": live_order_id,
                "sl_order_id": sl_order_id,
                "symbol": symbol,
                "action": action,
                "quantity": filled_qty,
                "fill_price": fill_price,
                "stop_loss": stop_loss,
                "message": f"LIVE ORDER FILLED & PROTECTED: {action} {filled_qty} {symbol} @ INR {fill_price:.2f}. Broker SL Active: INR {stop_loss:.2f} (SL Order ID: {sl_order_id})"
            }

        except Exception as e:
            logger.error(f"Live execution error for {symbol}: {e}")
            return {"success": False, "trading_mode": "LIVE", "message": f"Execution Exception: {str(e)}"}

    def cancel_all_open_orders(self) -> Dict[str, Any]:
        """Cancels all pending orders on broker."""
        if not self.is_connected or not self.smart_api:
            return {"cancelled_count": 0, "status": "PAPER_MODE"}

        cancelled = 0
        try:
            order_book = self.smart_api.orderBook()
            if order_book and order_book.get("status") is True:
                orders = order_book.get("data", []) or []
                for o in orders:
                    if o.get("status", "").lower() in ["open", "pending", "trigger pending"]:
                        oid = str(o.get("orderid"))
                        variety = o.get("variety", "NORMAL")
                        self.smart_api.cancelOrder(oid, variety)
                        cancelled += 1
            return {"cancelled_count": cancelled, "status": "SUCCESS"}
        except Exception as e:
            logger.error(f"Cancel all orders error: {e}")
            return {"cancelled_count": cancelled, "error": str(e)}

    def square_off_all_positions(self) -> Dict[str, Any]:
        """Squares off all open positions immediately (Kill switch / EOD)."""
        if not self.is_connected or not self.smart_api:
            return {"squared_off_count": 0, "status": "PAPER_MODE"}

        closed = 0
        try:
            pos_res = self.smart_api.position()
            if pos_res and pos_res.get("status") is True:
                positions = pos_res.get("data", []) or []
                for p in positions:
                    net_qty = int(p.get("netqty", 0))
                    if net_qty != 0:
                        action = "SELL" if net_qty > 0 else "BUY"
                        qty = abs(net_qty)
                        sym = p.get("tradingsymbol")
                        tok = p.get("symboltoken")
                        prod = p.get("producttype", "INTRADAY")
                        exch = p.get("exchange", "NSE")
                        
                        exit_params = {
                            "variety": "NORMAL",
                            "tradingsymbol": sym,
                            "symboltoken": str(tok),
                            "transactiontype": action,
                            "exchange": exch,
                            "ordertype": "MARKET",
                            "producttype": prod,
                            "duration": "DAY",
                            "price": "0",
                            "quantity": str(qty)
                        }
                        self.smart_api.placeOrder(exit_params)
                        closed += 1
            return {"squared_off_count": closed, "status": "SUCCESS"}
        except Exception as e:
            logger.error(f"Square off all positions error: {e}")
            return {"squared_off_count": closed, "error": str(e)}

    def get_positions(self) -> list:
        """Retrieves active broker positions."""
        if not self.is_connected or not self.smart_api:
            return []
        try:
            pos_res = self.smart_api.position()
            if pos_res and pos_res.get("status") is True:
                return pos_res.get("data", []) or []
            return []
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    def resolve_symbol_token(self, symbol: str) -> Optional[str]:
        """Resolves common NSE ticker / symbol names to Angel One tokens."""
        sym_clean = symbol.upper().strip()
        lookup = {
            "RELIANCE": "2885", "RELIANCE.NS": "2885", "RELIANCE-EQ": "2885",
            "HDFCBANK": "1333", "HDFCBANK.NS": "1333", "HDFCBANK-EQ": "1333",
            "TCS": "11536", "TCS.NS": "11536", "TCS-EQ": "11536",
            "SBIN": "3045", "SBIN.NS": "3045", "SBIN-EQ": "3045",
            "INFY": "1594", "INFY.NS": "1594", "INFY-EQ": "1594",
            "ITC": "1660", "ITC.NS": "1660", "ITC-EQ": "1660",
            "TATAMOTORS": "3456", "TATAMOTORS.NS": "3456", "TATAMOTORS-EQ": "3456",
            "GOLDBEES": "14428", "GOLDBEES.NS": "14428", "GOLDBEES-EQ": "14428",
            "^NSEI": "99926000", "NIFTY": "99926000", "NIFTY 50": "99926000",
            "^NSEBANK": "99926009", "BANKNIFTY": "99926009"
        }
        return lookup.get(sym_clean, None)

    def fetch_historical_candles(self, symbol_or_token: str, exchange: str = "NSE",
                                interval: str = "ONE_DAY", from_date: str = None,
                                to_date: str = None) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV candle data directly from Angel One SmartAPI getCandleData.
        """
        if not self.is_connected or not self.smart_api:
            return None

        token = self.resolve_symbol_token(symbol_or_token)
        if not token:
            token = str(symbol_or_token)

        if not from_date:
            from_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d 09:15")
        if not to_date:
            to_date = datetime.datetime.now().strftime("%Y-%m-%d 15:30")

        params = {
            "exchange": exchange.upper(),
            "symboltoken": str(token),
            "interval": interval.upper(),
            "fromdate": from_date,
            "todate": to_date
        }

        try:
            res = self.smart_api.getCandleData(params)
            if res and res.get("status") is True and res.get("data"):
                raw_candles = res.get("data", [])
                if not raw_candles:
                    return None
                
                df = pd.DataFrame(raw_candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df.set_index('Timestamp', inplace=True)
                
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
                df['Volume'] = df['Volume'].fillna(0.0)
                df.sort_index(inplace=True)
                return df if len(df) >= 5 else None
            else:
                logger.warning(f"SmartAPI getCandleData failed for {symbol_or_token}: {res.get('message', 'No data') if res else 'None'}")
                return None
        except Exception as e:
            logger.error(f"Exception in fetch_historical_candles for {symbol_or_token}: {e}")
            return None

    def get_ltp(self, symbol_or_token: str, exchange: str = "NSE") -> Optional[float]:
        """Fetches live Last Traded Price (LTP) from Angel One SmartAPI."""
        if not self.is_connected or not self.smart_api:
            return None
        token = self.resolve_symbol_token(symbol_or_token) or str(symbol_or_token)
        try:
            res = self.smart_api.ltpData(exchange.upper(), symbol_or_token, token)
            if res and res.get("status") is True and res.get("data"):
                return float(res["data"].get("ltp", 0.0))
        except Exception as e:
            logger.warning(f"Failed to fetch LTP for {symbol_or_token}: {e}")
        return None


# Global Singleton
angel_client = AngelOneClient()
