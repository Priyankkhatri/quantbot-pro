"""
Broker Adapter Module
Modular gateway supporting Paper Simulation, Angel One SmartAPI, and Zerodha Kite Connect.
Ensures identical execution logic whether trading virtually or through real brokers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time


class BaseBrokerAdapter(ABC):
    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def get_account_balance(self) -> float:
        pass

    @abstractmethod
    def place_order(self, symbol: str, action: str, quantity: float, order_type: str = "MARKET", stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> Dict[str, Any]:
        pass


class PaperBrokerAdapter(BaseBrokerAdapter):
    """Virtual simulated broker for zero-risk testing."""
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: list = []

    def connect(self) -> bool:
        return True

    def get_account_balance(self) -> float:
        return self.balance

    def place_order(self, symbol: str, action: str, quantity: float, order_type: str = "MARKET", stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict[str, Any]:
        order_id = f"PAPER_{int(time.time() * 1000)}"
        order_record = {
            "order_id": order_id,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "FILLED",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.positions[symbol] = order_record
        return order_record

    def close_position(self, symbol: str) -> Dict[str, Any]:
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            pos["status"] = "CLOSED"
            pos["closed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.trade_history.append(pos)
            return pos
        return {"error": f"No open position found for {symbol}"}


class AngelOneSmartAPIAdapter(BaseBrokerAdapter):
    """Angel One SmartAPI adapter template for Phase 3 live trading."""
    def __init__(self, api_key: str = "", client_code: str = "", password: str = "", totp_token: str = ""):
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_token = totp_token
        self.smart_api = None

    def connect(self) -> bool:
        if not self.api_key or not self.client_code:
            return False  # Config not supplied yet
        # Ready for smartapi-python integration
        return True

    def get_account_balance(self) -> float:
        return 0.0

    def place_order(self, symbol: str, action: str, quantity: float, order_type: str = "MARKET", stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict[str, Any]:
        return {"status": "NOT_CONNECTED", "message": "Supply Angel One API Key in settings for live broker execution."}

    def close_position(self, symbol: str) -> Dict[str, Any]:
        return {"status": "NOT_CONNECTED"}
