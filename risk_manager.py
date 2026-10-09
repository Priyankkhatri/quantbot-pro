"""
Risk Management Engine
Enforces institutional risk controls:
- Maximum Risk Per Trade (e.g., 1-2% of total capital)
- Dynamic ATR-based Stop-Loss and Take-Profit calculations
- Position Sizing calculation
- Maximum Drawdown Circuit Breaker
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TradeOrder:
    symbol: str
    entry_date: str
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    capital_at_risk: float


class RiskManager:
    def __init__(self, initial_capital: float = 10000.0, risk_per_trade_pct: float = 0.02, max_drawdown_pct: float = 0.15):
        """
        :param initial_capital: Starting balance in local currency (e.g., INR or USD).
        :param risk_per_trade_pct: Fraction of capital to risk on any single trade (default: 2%).
        :param max_drawdown_pct: Halt trading if total portfolio loses this percentage from peak.
        """
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.is_circuit_broken = False

    def check_drawdown(self, current_capital: float) -> bool:
        """Checks if portfolio has hit maximum allowed drawdown limit."""
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        
        drawdown = (self.peak_capital - current_capital) / self.peak_capital
        if drawdown >= self.max_drawdown_pct:
            self.is_circuit_broken = True
            return False  # Halt trading
        return True

    def calculate_position(self, entry_price: float, atr_value: float, sl_multiplier: float = 1.5, tp_multiplier: float = 3.0) -> Tuple[int, float, float]:
        """
        Calculates exact Stop-Loss, Take-Profit, and integer whole shares (floor) to buy.
        Ensures that if Stop-Loss is hit, loss is within `risk_per_trade_pct`.
        If calculated units == 0, returns 0 shares and records a clear rejection reason.
        """
        import math
        self.last_rejection_reason = ""

        if entry_price <= 0:
            self.last_rejection_reason = "REJECTED: Entry price must be strictly positive."
            return 0, 0.0, 0.0

        if self.capital <= 0:
            self.last_rejection_reason = "REJECTED: Insufficient account capital (<= ₹0)."
            return 0, 0.0, 0.0

        risk_amount = self.capital * self.risk_per_trade_pct
        sl_distance = max(atr_value * sl_multiplier, entry_price * 0.005)  # minimum 0.5% buffer
        tp_distance = sl_distance * (tp_multiplier / sl_multiplier)       # maintain 1:2 R:R

        stop_loss = round(entry_price - sl_distance, 2)
        take_profit = round(entry_price + tp_distance, 2)

        # Number of units/shares such that (entry - SL) * units <= risk_amount
        raw_risk_units = risk_amount / max(sl_distance, 0.0001)
        
        # Cap position size so it does not exceed 100% available capital
        max_affordable_units = self.capital / entry_price
        
        # Strict Whole Shares (Floor)
        units = int(math.floor(min(raw_risk_units, max_affordable_units)))

        if units <= 0:
            self.last_rejection_reason = (
                f"REJECTED: Sizing resulted in 0 integer shares. "
                f"Account capital ₹{self.capital:,.2f} with 2% risk limit (₹{risk_amount:.2f}) is insufficient "
                f"to purchase 1 whole share of asset priced at ₹{entry_price:,.2f} with SL distance ₹{sl_distance:.2f}."
            )
            return 0, stop_loss, take_profit

        return units, stop_loss, take_profit

    def calculate_position_detailed(self, entry_price: float, atr_value: float, sl_multiplier: float = 1.5, tp_multiplier: float = 3.0) -> Tuple[int, float, float, str]:
        """Returns (integer_units, stop_loss, take_profit, reason_or_status)."""
        units, sl, tp = self.calculate_position(entry_price, atr_value, sl_multiplier, tp_multiplier)
        status = self.last_rejection_reason if units <= 0 else "APPROVED"
        return units, sl, tp, status
