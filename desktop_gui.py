"""
Desktop Visual Trading Bot GUI
Native Windows Application with Visual Browser Automation.
Opens Chrome/Edge visibly on your screen and performs automated live strategy execution.
"""

import sys
import os
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By

from data_fetcher import fetch_market_data
from strategy import TrendMomentumStrategy
from risk_manager import RiskManager

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ASSETS_CONFIG = {
    "USD/INR Forex": {"ticker": "USDINR=X", "tv_symbol": "FX_IDC:USDINR"},
    "Nifty 50 Index": {"ticker": "^NSEI", "tv_symbol": "NSE:NIFTY"},
    "Bank Nifty": {"ticker": "^NSEBANK", "tv_symbol": "NSE:BANKNIFTY"},
    "Reliance Industries": {"ticker": "RELIANCE.NS", "tv_symbol": "NSE:RELIANCE"},
    "Apple Inc. (US)": {"ticker": "AAPL", "tv_symbol": "NASDAQ:AAPL"},
    "Gold (GOLDBEES)": {"ticker": "GOLDBEES.NS", "tv_symbol": "NSE:GOLDBEES"}
}


class VisualTradingApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("QuantBot Pro - Live Visual Trading Automation")
        self.geometry("900x650")
        self.minsize(800, 600)

        self.driver = None
        self.is_running = False
        self.bot_thread = None
        self.current_capital = 10000.0

        self.init_ui()

    def init_ui(self):
        # Header Frame
        header = ctk.CTkFrame(self, corner_radius=10, fg_color="#131b2e")
        header.pack(fill="x", padx=15, pady=10)

        title = ctk.CTkLabel(header, text="QUANTBOT PRO", font=ctk.CTkFont(size=20, weight="bold"), text_color="#10b981")
        title.pack(side="left", padx=15, pady=10)

        subtitle = ctk.CTkLabel(header, text="Live Visual Browser Trading Bot", font=ctk.CTkFont(size=12), text_color="#94a3b8")
        subtitle.pack(side="left", padx=5, pady=10)

        self.status_badge = ctk.CTkLabel(header, text="● BOT STOPPED", font=ctk.CTkFont(size=12, weight="bold"), text_color="#ef4444")
        self.status_badge.pack(side="right", padx=15, pady=10)

        # Control Panel Frame
        control_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#131b2e")
        control_frame.pack(fill="x", padx=15, pady=5)

        # Grid config
        control_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # 1. Asset Dropdown
        ctk.CTkLabel(control_frame, text="Target Market:", font=ctk.CTkFont(size=11, weight="bold"), text_color="#cbd5e1").grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")
        self.asset_var = ctk.StringVar(value="USD/INR Forex")
        self.asset_menu = ctk.CTkOptionMenu(control_frame, values=list(ASSETS_CONFIG.keys()), variable=self.asset_var, fg_color="#1e293b", button_color="#334155")
        self.asset_menu.grid(row=1, column=0, padx=10, pady=(0, 12), sticky="ew")

        # 2. Starting Capital
        ctk.CTkLabel(control_frame, text="Capital (₹):", font=ctk.CTkFont(size=11, weight="bold"), text_color="#cbd5e1").grid(row=0, column=1, padx=10, pady=(10, 2), sticky="w")
        self.capital_entry = ctk.CTkEntry(control_frame, placeholder_text="10000", fg_color="#1e293b")
        self.capital_entry.insert(0, "10000")
        self.capital_entry.grid(row=1, column=1, padx=10, pady=(0, 12), sticky="ew")

        # 3. Risk Per Trade
        ctk.CTkLabel(control_frame, text="Risk Per Trade (%):", font=ctk.CTkFont(size=11, weight="bold"), text_color="#cbd5e1").grid(row=0, column=2, padx=10, pady=(10, 2), sticky="w")
        self.risk_entry = ctk.CTkEntry(control_frame, placeholder_text="2.0", fg_color="#1e293b")
        self.risk_entry.insert(0, "2.0")
        self.risk_entry.grid(row=1, column=2, padx=10, pady=(0, 12), sticky="ew")

        # 4. Action Buttons
        btn_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        btn_frame.grid(row=1, column=3, padx=10, pady=(0, 12), sticky="ew")
        btn_frame.grid_columnconfigure((0, 1), weight=1)

        self.start_btn = ctk.CTkButton(btn_frame, text="▶ START BOT", font=ctk.CTkFont(weight="bold"), fg_color="#10b981", hover_color="#059669", command=self.start_bot)
        self.start_btn.grid(row=0, column=0, padx=3, sticky="ew")

        self.stop_btn = ctk.CTkButton(btn_frame, text="⏹ STOP", font=ctk.CTkFont(weight="bold"), fg_color="#ef4444", hover_color="#dc2626", state="disabled", command=self.stop_bot)
        self.stop_btn.grid(row=0, column=1, padx=3, sticky="ew")

        # Metrics Overview Row
        metrics_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#131b2e")
        metrics_frame.pack(fill="x", padx=15, pady=5)
        metrics_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.metric_price = self.create_kpi_card(metrics_frame, "Latest Price", "₹0.00", 0)
        self.metric_pnl = self.create_kpi_card(metrics_frame, "Net Realized PnL", "₹0.00", 1)
        self.metric_trades = self.create_kpi_card(metrics_frame, "Trades Executed", "0", 2)
        self.metric_status = self.create_kpi_card(metrics_frame, "Position Status", "FLAT", 3)

        # Terminal Log Output Box
        log_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#131b2e")
        log_frame.pack(fill="both", expand=True, padx=15, pady=10)

        log_header = ctk.CTkLabel(log_frame, text="LIVE BOT TERMINAL LOGS & DECISION AUDIT", font=ctk.CTkFont(size=12, weight="bold"), text_color="#cbd5e1")
        log_header.pack(anchor="w", padx=15, pady=(10, 5))

        self.log_text = ctk.CTkTextbox(log_frame, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#0b0f19", text_color="#10b981")
        self.log_text.pack(fill="both", expand=True, padx=15, pady=(0, 12))

        self.log("[SYSTEM] QuantBot Pro Desktop Engine ready.")
        self.log("[INFO] Click 'START BOT' to open a live visible browser window and begin automated trading.")

    def create_kpi_card(self, parent, title, val, col):
        card = ctk.CTkFrame(parent, fg_color="#0b0f19", corner_radius=8)
        card.grid(row=0, column=col, padx=8, pady=8, sticky="nsew")
        
        lbl_t = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=10, weight="bold"), text_color="#94a3b8")
        lbl_t.pack(pady=(6, 0))
        
        lbl_v = ctk.CTkLabel(card, text=val, font=ctk.CTkFont(size=15, weight="bold"), text_color="#f8fafc")
        lbl_v.pack(pady=(0, 6))
        return lbl_v

    def log(self, message):
        t = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{t}] {message}\n")
        self.log_text.see("end")

    def start_bot(self):
        try:
            self.current_capital = float(self.capital_entry.get())
            risk_pct = float(self.risk_entry.get()) / 100.0
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for capital and risk.")
            return

        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.asset_menu.configure(state="disabled")
        self.status_badge.configure(text="● BOT LIVE & RUNNING", text_color="#10b981")

        self.bot_thread = threading.Thread(target=self.run_visual_bot_worker, daemon=True)
        self.bot_thread.start()

    def stop_bot(self):
        self.is_running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.asset_menu.configure(state="normal")
        self.status_badge.configure(text="● BOT STOPPED", text_color="#ef4444")
        self.log("[USER] Stop command issued. Halting visual automation.")

        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def launch_browser(self, tv_symbol, display_name):
        self.log(f"[BROWSER] Launching live visual browser window for {display_name}...")
        
        url = f"http://127.0.0.1:5000/terminal?symbol={tv_symbol}&ticker={display_name}"
        
        driver = None
        # Try Chrome first
        try:
            opts = ChromeOptions()
            opts.add_argument("--start-maximized")
            opts.add_experimental_option("excludeSwitches", ["enable-automation"])
            driver = webdriver.Chrome(options=opts)
        except Exception:
            # Fallback to Edge
            try:
                edge_opts = EdgeOptions()
                edge_opts.add_argument("--start-maximized")
                driver = webdriver.Edge(options=edge_opts)
            except Exception as e:
                self.log(f"[ERROR] Could not start Chrome/Edge via Selenium: {e}")
                return None

        driver.get(url)
        self.log("[BROWSER] Real-Time Live Moving Market Terminal loaded on screen successfully.")
        return driver

    def run_visual_bot_worker(self):
        selected_name = self.asset_var.get()
        asset_info = ASSETS_CONFIG[selected_name]
        ticker = asset_info["ticker"]
        tv_symbol = asset_info["tv_symbol"]

        self.log(f"[START] Initializing visual live bot for {selected_name} ({ticker})...")
        
        self.driver = self.launch_browser(tv_symbol, selected_name)
        if not self.driver:
            self.stop_bot()
            return

        risk_mgr = RiskManager(initial_capital=self.current_capital)
        strategy = TrendMomentumStrategy()

        in_position = False
        position_units = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        total_pnl = 0.0
        trades_count = 0

        # Run live visual loop
        while self.is_running:
            try:
                # Fetch recent market data
                df = fetch_market_data(ticker, period="60d", interval="1d")
                if df is None or len(df) < 20:
                    self.log("[WARN] Waiting for market data response...")
                    time.sleep(5)
                    continue

                signals_df = strategy.generate_signals(df)
                last_bar = signals_df.iloc[-1]
                current_price = float(last_bar['Close'])
                atr = float(last_bar['ATR_14'])
                ema20 = float(last_bar['EMA_20'])
                ema50 = float(last_bar['EMA_50'])
                rsi = float(last_bar['RSI_14'])

                # Update UI KPIs
                self.metric_price.configure(text=f"₹{current_price:,.2f}")
                self.log(f"[TICK] {ticker} @ ₹{current_price:,.2f} | EMA20: {ema20:.2f} | EMA50: {ema50:.2f} | RSI: {rsi:.1f}")

                # Check In-Position Exit Rules
                if in_position:
                    exit_reason = None
                    if current_price <= stop_loss:
                        exit_reason = "Stop Loss Triggered"
                    elif current_price >= take_profit:
                        exit_reason = "Take Profit Reached"
                    elif last_bar['Exit_Trigger']:
                        exit_reason = "Trend Reversal Signal"

                    if exit_reason:
                        pnl = (current_price - entry_price) * position_units
                        total_pnl += pnl
                        self.current_capital += pnl
                        trades_count += 1

                        self.log(f"[ALERT] {exit_reason}! Executing visual CLOSE order on screen...")

                        # Trigger visual script in browser
                        try:
                            self.driver.execute_script(f"window.triggerExitVisual({current_price:.2f}, {pnl:.2f}, '{exit_reason}');")
                            self.driver.execute_script(f"window.updateBalanceVisual({self.current_capital:.2f});")
                        except Exception:
                            pass

                        self.metric_pnl.configure(text=f"₹{total_pnl:+,.2f}", text_color="#10b981" if total_pnl >= 0 else "#ef4444")
                        self.metric_trades.configure(text=str(trades_count))
                        self.metric_status.configure(text="FLAT", text_color="#f8fafc")

                        in_position = False
                        position_units = 0

                # Check New Buy Trigger
                elif not in_position and not risk_mgr.is_circuit_broken:
                    if last_bar['Buy_Trigger']:
                        position_units, stop_loss, take_profit = risk_mgr.calculate_position(current_price, atr)
                        entry_price = current_price
                        in_position = True

                        self.log(f"[SIGNAL] Strategy BUY Criteria Met! Executing visual BUY order on screen...")

                        # Trigger visual buy animation in browser
                        try:
                            self.driver.execute_script(f"window.triggerBuyVisual({current_price:.2f}, {position_units:.2f}, {stop_loss:.2f}, {take_profit:.2f});")
                        except Exception:
                            pass

                        self.metric_status.configure(text=f"LONG ({position_units:.2f} units)", text_color="#10b981")
                    else:
                        self.log("[SCAN] Strategy conditions not yet triggered. Watching next tick...")

                # Wait before next live tick scan
                for _ in range(6):
                    if not self.is_running:
                        break
                    time.sleep(1)

            except Exception as e:
                self.log(f"[ERROR] Live loop exception: {e}")
                time.sleep(5)

        self.log("[STOP] Visual Bot worker loop terminated.")


if __name__ == "__main__":
    app = VisualTradingApp()
    app.mainloop()
