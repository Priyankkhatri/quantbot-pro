# QuantBot Pro: Institutional Quantitative Trading & ML System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Market: NSE India](https://img.shields.io/badge/Market-NSE%20India-orange.svg)](https://www.nseindia.com/)
[![Integrity: SHA--256 Hash Chain](https://img.shields.io/badge/Integrity-SHA--256%20Chained-green.svg)]()
[![Tests: 24/24 Passed](https://img.shields.io/badge/Tests-24%2F24%20Passing-brightgreen.svg)]()

**QuantBot Pro** is an institutional-grade algorithmic trading, machine learning ranking, and forward paper-testing platform designed for Indian Equities (NSE) and liquid ETFs. It combines multi-model ensemble forecasting, factor momentum, volatility targeting, hard cryptographic forward ledger auditing, and strict real-world market friction modeling.

---

## 🏛️ System Architecture

```
                                  ┌──────────────────────────────┐
                                  │   NSE / Angel One SmartAPI   │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DATA & PERSISTENCE LAYER                                │
│  • SQLite OHLCV Cache (market_data_cache.db)      • Stale-Price & Gap Detection         │
│  • Completed-Candle Enforcement                   • 10-Year Clean Adjusted History      │
└────────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               QUANTITATIVE SIGNAL ENGINES                               │
│  • Multi-Head ML Ensemble (XGBoost, LightGBM, CatBoost, Random Forest, ExtraTrees)      │
│  • Low-Volatility 120D Top-Quintile Quarterly Rebalancer                               │
│  • Momentum 12-1M Top-Quintile Monthly Rebalancer                                       │
│  • ML Hysteresis Turnover-Controlled Ranker                                             │
│  • HAR / EWMA / GARCH Realized Volatility Targeting & Risk Overlay                      │
└────────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               RISK & EXECUTION CONTROLS                                 │
│  • 4-Point Gating Circuit (Live Fail-Closed)      • Hard Drawdown Circuit Breakers      │
│  • Idempotency & Debounce Engine                  • Realistic Cost Model (STT/DP/CNC)   │
│  • Integer Whole-Share Allocator                  • Micro Tier (₹1,000) Sizing Engine   │
└────────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         FORWARD AUDIT & MONITORING LAYER                                │
│  • SHA-256 Hash-Chained Forward Ledger (paper_forward_ledger)                           │
│  • Strict BACKTEST_SEED vs FORWARD Separation (Zero Lookahead / Hindsight Bias)         │
│  • Flask Server-Sent Events (SSE) Live Web Dashboard & TradingView Lightweight Charts   │
│  • Automated Post-Market Daily Job (15:45 IST) & Telegram Broadcasts                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

### 1. Multi-Head ML & Factor Strategy Suite
* **Multi-Head ML Ensemble:** Weighted ensemble combining LightGBM ($35\%$), XGBoost ($25\%$), CatBoost ($20\%$), Random Forest ($10\%$), and ExtraTrees ($10\%$) with walk-forward out-of-sample promotion gates.
* **Low-Vol 120D:** Quarterly top-quintile volatility ranking with historical risk-adjusted outperformance.
* **Momentum 12-1M:** Monthly top-quintile rebalanced relative-momentum ranking excluding the most recent month to avoid short-term reversals.
* **ML Hysteresis Ranker:** Dual-threshold buffer ($K_{\text{entry}}=20\%$, $K_{\text{hold}}=40\%$) minimizing turnover friction.
* **HAR / EWMA Volatility Target:** Realized volatility forecasting overlay dynamically scaling risk exposure and parking remainder in liquid assets.

### 2. Cryptographic SHA-256 Forward Paper Ledger
* **Tamper-Proof Audit Chain:** Every daily portfolio state hashes the previous row (`prev_hash`), sequence number, cash balance, positions, fills, and data status.
* **Zero Hindsight Bias:** Historical backtest data is strictly tagged as `BACKTEST_SEED`. Genuine forward tracking commences at `FORWARD_START_DATE` with isolated performance metrics.
* **Automated Catch-Up Engine:** Reconciles missed trading days chronologically using only point-in-time historical data.

### 3. Institutional Cost & Friction Modeling
* **Angel One CNC Model:** Real-world transaction simulation including flat brokerage, STT ($0.1\%$), exchange turnover fees, SEBI charges, stamp duty ($0.015\%$), and depository participant (DP) charges.
* **Dual Capital Tiers:**
  * **Standard Tier (₹1,00,000):** Full institutional diversified allocation.
  * **Micro Tier (₹1,000):** Integer whole-share constraint analysis demonstrating friction and uninvested cash dynamics.

### 4. Real-Time Risk & Fail-Safe Controls
* **4-Point Gating Circuit:** Blocks live execution unless empirical criteria (OOS trade count, profit factor, drawdown ceiling, and baseline anchor) pass.
* **Hard Drawdown Kill-Switch:** Automatically halts trading upon breaching daily or lifetime portfolio drawdown thresholds.
* **Idempotency & Debounce:** Strict single-order-per-day enforcement prevents duplicate execution.
* **Market Calendar Awareness:** Automatic NSE trading session, weekend, and holiday detection.

---

## 📁 Repository Structure

```
├── app.py                                   # Live Flask Web Dashboard & SSE Streaming Server
├── backtester.py                            # Vectorized & Event-Driven Backtesting Engine
├── broker_adapter.py                        # Unified Broker Interface (Paper & Live Execution)
├── data_fetcher.py                          # Market Data Ingestion & Persistent SQLite Cache
├── forward_paper_testing.py                 # SHA-256 Hash-Chained Forward Ledger & Engine
├── generate_walk_forward_baseline.py        # Dynamic Baseline Performance Generator
├── indicators.py                            # Technical Indicators & Vectorized Math
├── interactive_menu.py                      # Interactive Terminal CLI Tester
├── market_calendar.py                       # NSE Market Holidays & Trading Hours Scheduler
├── ml_model.py                              # 5-Model Multi-Head Quantitative AI Engine
├── reconciliation.py                        # Startup Broker Position & Order State Reconciler
├── risk_manager.py                          # Integer Share Sizing & Pre-Trade Risk Checks
├── risk_store.py                            # SQLite Risk Controls, Gating & Trade Journal
├── scheduler.py                             # Post-Market Daily Jobs & EOD Accounting
├── strategy.py                              # Trading Strategy Logic & Parameter Rules
├── strategy_research_fair_benchmark.py      # Paired Block Bootstrap & Statistical Harness
├── strategy_research_etf_vol_overlay.py     # ETF Rotation & HAR Volatility Overlay Models
├── telegram_notifier.py                     # Asynchronous HTML Telegram Alert Dispatcher
├── test_suite.py                            # 24-Test Institutional Master Verification Suite
├── walk_forward_baseline_results.json       # Frozen Out-Of-Sample Baseline Records
└── README_FORWARD_SCHEDULER.md              # Windows Task Scheduler Setup Guide
```

---

## ⚡ Quick Start

### 1. Installation & Environment

Clone the repository and install dependencies:

```bash
git clone https://github.com/Priyankkhatri/quantbot-pro.git
cd quantbot-pro

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration (`.env`)

Create a `.env` file in the root directory:

```env
TRADING_MODE=PAPER
EXECUTION_MODE=SEMI

# Angel One SmartAPI (Optional for Paper Mode)
ANGEL_API_KEY=your_api_key
ANGEL_CLIENT_CODE=your_client_code
ANGEL_PASSWORD=your_password
ANGEL_TOTP_KEY=your_totp_key

# Telegram Alerts (Optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3. Launching the System

* **Web Dashboard UI:**
  ```bash
  python app.py
  ```
  Open `http://127.0.0.1:5000` in your browser.

* **Interactive CLI Terminal:**
  ```bash
  python interactive_menu.py
  ```

* **Daily Post-Market Forward Step (Manual):**
  ```bash
  python forward_paper_testing.py --post-market
  ```

---

## 🧪 Master Test Suite

QuantBot Pro includes a comprehensive 24-test institutional verification suite covering indicator math, risk controls, idempotency, gating circuits, data cache persistence, and broker adapters:

```bash
python test_suite.py
```

```
===========================================================================
   QUANTBOT PRO: MASTER INSTITUTIONAL CODE AUDIT & SAFETY VERIFICATION
===========================================================================
[Test 1] Testing Mathematical Indicators on Extreme Data...        ✓ PASS
[Test 2] Testing Data Fetcher with Indian Equities...               ✓ PASS
[Test 3] Testing 5-Model Quantitative AI Multi-Head Engine...      ✓ PASS
...
[Test 24] Testing Post-Market Promotion & Forward Hash-Chain...     ✓ PASS
===========================================================================
  🎉 ALL 24 MASTER STRESS, RISK, PERSISTENCE & AUTOMATION TESTS PASSED!
===========================================================================
```

---

## 🔒 Security & Failsafe Design

1. **Default Paper Mode:** `TRADING_MODE` defaults to `PAPER`. Live mode requires explicit configuration, token authentication, and passing the 4-point empirical gating circuit.
2. **Credential Masking:** All API keys, passwords, and tokens are masked in structured logs.
3. **Localhost Binding:** Web server strictly binds to `127.0.0.1`.

---

## 📜 License & Disclaimer

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Disclaimer:** *QuantBot Pro is designed for quantitative research, backtesting, and paper trading. Past performance is no guarantee of future results. Algorithmic trading in financial markets involves substantial risk of loss. Always test thoroughly in paper trading mode before deploying capital.*
