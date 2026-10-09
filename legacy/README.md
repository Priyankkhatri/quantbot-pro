# Legacy Visual Bot & Selenium Overlays (Archived)

The Selenium-based TradingView stealth automation (`tradingview_bot.py`), JavaScript overlay injection (`ai_overlay.js`), and associated startup scripts have been moved to this `/legacy` directory.

### Modern Architecture:
- Real-time market data is ingested directly through **Angel One SmartAPI** and official brokers with persistent SQLite caching.
- Interactive charting is natively embedded directly in the web dashboard using **TradingView Lightweight Charts (v4.x)** canvas, eliminating the heavy Chromium/Selenium overhead.
