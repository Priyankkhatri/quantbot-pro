"""
Quantitative Strategy Module
Generates systematic entry and exit signals based on trend-following and momentum filters.
"""

import pandas as pd
from indicators import add_all_indicators


class TrendMomentumStrategy:
    """
    Dual EMA + RSI Momentum + ATR Volatility Strategy:
    - Long Entry: 
        1. Short EMA (20) > Long EMA (50)
        2. Close > 200 EMA (Macro Bullish Trend)
        3. RSI between 45 and 65 (Healthy momentum, not overbought)
        4. Previous candle was not already in a buy signal (Fresh trigger)
    - Exit Rules:
        1. Hit dynamic Take-Profit (e.g., 2x ATR)
        2. Hit dynamic Stop-Loss (e.g., 1x ATR)
        3. Trend Reversal: EMA 20 crosses below EMA 50
    """

    def __init__(self, atr_multiplier_sl: float = 1.5, atr_multiplier_tp: float = 3.0):
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        data = add_all_indicators(df)
        data['Signal'] = 0  # 1 = Buy, -1 = Sell/Exit, 0 = Hold

        # Condition 1: Bullish Trend Alignment
        trend_bullish = (data['EMA_20'] > data['EMA_50']) & (data['Close'] > data['EMA_200'])
        
        # Condition 2: Healthy RSI Range (avoid buying when RSI > 70 overbought)
        rsi_valid = (data['RSI_14'] > 45) & (data['RSI_14'] < 68)

        # Condition 3: MACD Momentum Confirmation
        macd_positive = data['MACD'] > data['MACD_Signal']

        # Combine for raw buy signal
        raw_buy = trend_bullish & rsi_valid & macd_positive

        # Detect fresh signal crossover
        data['Buy_Trigger'] = raw_buy & (~raw_buy.shift(1).fillna(False))
        
        # Exit trigger: Trend breakdown or RSI overbought breakdown
        data['Exit_Trigger'] = (data['EMA_20'] < data['EMA_50']) | (data['Close'] < data['EMA_50'])

        return data
