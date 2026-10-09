"""
Technical Indicators Calculation Module
Mathematical computations for quantitative indicators:
- EMA (9, 20, 21, 50, 200)
- SMA (Simple Moving Average)
- RSI (Wilder Relative Strength Index)
- ATR (Average True Range)
- Bollinger Bands
- MACD (Moving Average Convergence Divergence)
"""

import pandas as pd
import numpy as np


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculates Simple Moving Average."""
    return series.rolling(window=period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates Relative Strength Index (RSI).
    Safeguarded against zero loss division.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()

    rs = gain / (loss.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates Average True Range (ATR).
    Safeguarded against NaN and empty windows.
    """
    high = df['High']
    low = df['Low']
    close = df['Close'].shift(1).fillna(df['Open'])

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr.fillna(0.01)


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculates MACD line, Signal line, and Histogram."""
    fast_ema = compute_ema(series, fast)
    slow_ema = compute_ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds full suite of indicators to the OHLCV dataframe."""
    data = df.copy()
    data['EMA_9'] = compute_ema(data['Close'], 9)
    data['EMA_20'] = compute_ema(data['Close'], 20)
    data['EMA_21'] = compute_ema(data['Close'], 21)
    data['EMA_50'] = compute_ema(data['Close'], 50)
    data['EMA_200'] = compute_ema(data['Close'], min(200, len(data) - 1))
    data['RSI_14'] = compute_rsi(data['Close'], 14)
    data['ATR_14'] = compute_atr(data, 14)
    
    macd_line, signal_line, hist = compute_macd(data['Close'])
    data['MACD'] = macd_line
    data['MACD_Signal'] = signal_line
    data['MACD_Hist'] = hist
    
    return data
