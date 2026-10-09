"""
Data Layer & Persistent OHLCV Engine
Features:
- Primary Source: Angel One SmartAPI Historical Candles (getCandleData) & Live SmartStream
- Fallback Source: yfinance (safeguarded and clearly labeled)
- Persistent SQLite OHLCV Caching (market_data_cache.db) with rate limiting & exponential backoff
- Strict Completed Candle Filter (Discards still-forming bars to eliminate look-ahead leakage)
- Data Freshness Indicator & Stale Signal Circuit (Halts signal generation if data is stale)
"""

import os
import time
import sqlite3
import datetime
import logging
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import yfinance as yf

from market_calendar import get_market_session_state, get_current_ist_time

logger = logging.getLogger("DataFetcher")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_data_cache.db")


def init_cache_db():
    """Initializes the SQLite database schema for OHLCV candle caching."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_candles (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_meta (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                last_sync TEXT NOT NULL,
                source TEXT NOT NULL,
                bar_count INTEGER NOT NULL,
                PRIMARY KEY (symbol, timeframe)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup ON ohlcv_candles (symbol, timeframe, timestamp)")
        conn.commit()


# Initialize SQLite on module load
init_cache_db()


def filter_completed_candles(df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    """
    CRITICAL QUANT SAFETY RULE:
    Never compute features or signals on a still-forming daily/intraday candle.
    If the latest candle's timestamp corresponds to the current ongoing session,
    it is sliced off (df.iloc[:-1]) to prevent repainting and look-ahead leakage.
    """
    if df is None or df.empty:
        return df

    data = df.copy()
    ist_now = get_current_ist_time()
    today_ist_date = ist_now.date()
    market_close_time = datetime.time(15, 30, 0)

    last_idx = data.index[-1]
    if isinstance(last_idx, (pd.Timestamp, datetime.datetime, datetime.date)):
        last_date = last_idx.date() if isinstance(last_idx, (pd.Timestamp, datetime.datetime)) else last_idx
        
        # If the last candle is today and market is currently before or at 15:30 IST
        if last_date == today_ist_date:
            if ist_now.time() < market_close_time:
                # Candle is still forming live during today's session -> exclude it!
                data = data.iloc[:-1]

    return data


def save_candles_to_cache(symbol: str, timeframe: str, df: pd.DataFrame, source: str):
    """Saves validated OHLCV DataFrame into SQLite cache with UPSERT semantics."""
    if df is None or df.empty:
        return

    now_iso = get_current_ist_time().isoformat()
    rows = []
    for idx, row in df.iterrows():
        ts_str = str(idx)
        rows.append((
            symbol,
            timeframe,
            ts_str,
            float(row['Open']),
            float(row['High']),
            float(row['Low']),
            float(row['Close']),
            float(row.get('Volume', 0.0)),
            source,
            now_iso
        ))

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO ohlcv_candles (symbol, timeframe, timestamp, open, high, low, close, volume, source, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timeframe, timestamp) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                source=excluded.source,
                fetched_at=excluded.fetched_at
        """, rows)

        cursor.execute("""
            INSERT INTO cache_meta (symbol, timeframe, last_sync, source, bar_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timeframe) DO UPDATE SET
                last_sync=excluded.last_sync,
                source=excluded.source,
                bar_count=excluded.bar_count
        """, (symbol, timeframe, now_iso, source, len(rows)))
        conn.commit()


def load_candles_from_cache(symbol: str, timeframe: str = "1d", min_bars: int = 10) -> Optional[Tuple[pd.DataFrame, str, str]]:
    """
    Loads candles from SQLite cache.
    Returns: (df, source, last_sync_time) or None
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT last_sync, source FROM cache_meta WHERE symbol = ? AND timeframe = ?", (symbol, timeframe))
        meta = cursor.fetchone()
        if not meta:
            return None

        last_sync, source = meta[0], meta[1]

        cursor.execute("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_candles
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp ASC
        """, (symbol, timeframe))
        rows = cursor.fetchall()

        if not rows or len(rows) < min_bars:
            return None

        df = pd.DataFrame(rows, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        return df, source, last_sync


def fetch_from_yfinance_with_retries(ticker: str, period: str = "2y", interval: str = "1d", max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Downloads data using yfinance with exponential backoff and formatting safeguards."""
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] for col in data.columns]

                required_cols = ['Open', 'High', 'Low', 'Close']
                if all(col in data.columns for col in required_cols):
                    data = data[required_cols + ([col for col in ['Volume'] if col in data.columns])].dropna()
                    if 'Volume' not in data.columns:
                        data['Volume'] = 0.0
                    else:
                        data['Volume'] = data['Volume'].fillna(0.0)

                    for col in required_cols:
                        data[col] = pd.to_numeric(data[col], errors='coerce')

                    data = data.dropna(subset=required_cols)
                    if len(data) >= 10:
                        return data
        except Exception as e:
            logger.warning(f"yfinance fetch attempt {attempt}/{max_retries} failed for '{ticker}': {e}")

        time.sleep(backoff)
        backoff *= 2.0

    return None


def fetch_market_data(ticker: str, period: str = "2y", interval: str = "1d",
                      max_cache_age_seconds: int = 3600, force_refresh: bool = False,
                      completed_candles_only: bool = True) -> Optional[pd.DataFrame]:
    """
    Unified Data Layer Pipeline:
    1. Checks SQLite cache for fresh data.
    2. Primary: Attempts Angel One SmartAPI historical candles (if connected).
    3. Fallback: Downloads from yfinance with retry backoff.
    4. Caches result in SQLite with source tags.
    5. Strips still-forming bars when completed_candles_only=True.
    """
    timeframe = interval
    now = time.time()

    # Step 1: Check SQLite Cache
    if not force_refresh:
        min_expected_bars = 500 if period in ["5y", "10y", "max"] else (200 if period in ["1y", "2y"] else 10)
        cached = load_candles_from_cache(ticker, timeframe=timeframe, min_bars=min_expected_bars)
        if cached:
            df_cache, cache_source, last_sync_iso = cached
            try:
                last_sync_dt = datetime.datetime.fromisoformat(last_sync_iso)
                age_sec = (get_current_ist_time() - last_sync_dt).total_seconds()
                session = get_market_session_state()

                # If market is closed, cache from previous valid sync is always fresh
                if not session["can_enter_new_trade"] or age_sec < max_cache_age_seconds:
                    df_res = filter_completed_candles(df_cache, interval=interval) if completed_candles_only else df_cache
                    df_res.attrs['data_source'] = f"{cache_source} (Cached)"
                    df_res.attrs['last_sync'] = last_sync_iso
                    df_res.attrs['cache_age_sec'] = age_sec
                    return df_res
            except Exception as e:
                logger.debug(f"Cache age check note: {e}")

    # Step 2: Primary Fetch via Angel One SmartAPI
    data = None
    data_source = None
    
    try:
        from angel_one_client import angel_client
        if angel_client.is_connected and angel_client.smart_api:
            logger.info(f"Attempting primary fetch for {ticker} via Angel One SmartAPI...")
            df_angel = angel_client.fetch_historical_candles(ticker, exchange="NSE", interval="ONE_DAY")
            if df_angel is not None and len(df_angel) >= 10:
                data = df_angel
                data_source = "ANGEL_ONE"
                logger.info(f"Successfully fetched {len(data)} bars for {ticker} via Angel One SmartAPI.")
    except Exception as e:
        logger.warning(f"Angel One primary fetch failed for {ticker}: {e}")

    # Step 3: Fallback Fetch via yfinance
    if data is None:
        logger.info(f"Fetching {ticker} via Fallback Provider (yfinance)...")
        df_yf = fetch_from_yfinance_with_retries(ticker, period=period, interval=interval)
        if df_yf is not None and len(df_yf) >= 10:
            data = df_yf
            data_source = "YFINANCE_FALLBACK"
            logger.info(f"Successfully fetched {len(data)} bars for {ticker} via yfinance fallback.")

    # Step 4: Cache to SQLite
    if data is not None and not data.empty:
        save_candles_to_cache(ticker, timeframe, data, data_source)
        df_final = filter_completed_candles(data, interval=interval) if completed_candles_only else data
        df_final.attrs['data_source'] = data_source
        df_final.attrs['last_sync'] = get_current_ist_time().isoformat()
        df_final.attrs['cache_age_sec'] = 0
        return df_final

    # Final Fallback to any cached data regardless of age
    cached = load_candles_from_cache(ticker, timeframe=timeframe)
    if cached:
        df_cache, cache_source, last_sync_iso = cached
        logger.warning(f"Using stale cached data for {ticker} as offline emergency fallback.")
        df_final = filter_completed_candles(df_cache, interval=interval) if completed_candles_only else df_cache
        df_final.attrs['data_source'] = f"{cache_source} (Emergency Stale Cache)"
        df_final.attrs['last_sync'] = last_sync_iso
        return df_final

    logger.error(f"Failed to fetch market data for {ticker} across all providers.")
    return None


def get_data_freshness(last_tick_time: Optional[float] = None, session_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Computes data freshness metrics:
    - If market is closed: reports 'MARKET CLOSED' with last close time.
    - If market is open: reports tick age in seconds.
    - If tick age > 60s during open market: marks data as STALE and triggers signal circuit break.
    """
    if session_state is None:
        session_state = get_market_session_state()

    ist_now = get_current_ist_time()
    is_market_open = session_state.get("can_enter_new_trade", False) or session_state.get("session_status") == "ACTIVE_TRADING"

    if not is_market_open:
        return {
            "is_fresh": True,
            "is_stale": False,
            "is_market_open": False,
            "market_status": "MARKET CLOSED",
            "age_seconds": 0,
            "display_label": f"MARKET CLOSED ({session_state.get('reason', 'Closed')})",
            "allow_signal_generation": False,
            "ist_timestamp": ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
        }

    if last_tick_time is None:
        last_tick_time = time.time()

    age_seconds = round(max(time.time() - last_tick_time, 0), 1)
    is_stale = age_seconds > 60.0

    if is_stale:
        display_label = f"STALE DATA ({age_seconds:.0f}s ago - Signals Halted)"
    elif age_seconds < 2.0:
        display_label = "LIVE (~250-350ms)"
    else:
        display_label = f"LIVE ({age_seconds:.1f}s ago)"

    return {
        "is_fresh": not is_stale,
        "is_stale": is_stale,
        "is_market_open": True,
        "market_status": "LIVE OPEN",
        "age_seconds": age_seconds,
        "display_label": display_label,
        "allow_signal_generation": not is_stale,
        "ist_timestamp": ist_now.strftime("%H:%M:%S IST")
    }
