"""
TradingView AI Vision Auto-Trader Pro (Trained Forex & Multi-Asset Edition)
Real-time continuous live streaming, non-obtrusive draggable HUD, and 5-model consensus engine.
"""

import sys
import os
import time
import json
import threading
import queue
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By

from data_fetcher import fetch_market_data
from ml_model import InstitutionalForexAIEngine
from risk_manager import RiskManager

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

TRADINGVIEW_SYMBOLS = {
    "Reliance Industries": {"tv_symbol": "NSE:RELIANCE", "ticker": "RELIANCE.NS"},
    "Nifty 50 Index": {"tv_symbol": "NSE:NIFTY", "ticker": "^NSEI"},
    "Bank Nifty": {"tv_symbol": "NSE:BANKNIFTY", "ticker": "^NSEBANK"},
    "HDFC Bank": {"tv_symbol": "NSE:HDFCBANK", "ticker": "HDFCBANK.NS"},
    "TCS": {"tv_symbol": "NSE:TCS", "ticker": "TCS.NS"},
    "State Bank of India": {"tv_symbol": "NSE:SBIN", "ticker": "SBIN.NS"},
    "ITC Limited": {"tv_symbol": "NSE:ITC", "ticker": "ITC.NS"},
    "Infosys": {"tv_symbol": "NSE:INFY", "ticker": "INFY.NS"},
    "Gold BeES ETF": {"tv_symbol": "NSE:GOLDBEES", "ticker": "GOLDBEES.NS"}
}


class TradingViewAngelOneAIBotApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("QuantBot Pro - Angel One & NSE Neural Scanner")
        self.geometry("1020x750")
        self.minsize(920, 660)
        self.configure(fg_color="#f8fafc")

        self.msg_queue = queue.Queue()
        self.driver = None
        self.is_running = False
        self.bot_thread = None
        self.current_capital = 500.0
        self.ai_engine = InstitutionalForexAIEngine(model_path="angel_one_quant_ai.joblib")

        self.init_ui()
        self.after(100, self.process_queue)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_ui(self):
        # 1. Header Frame
        header = ctk.CTkFrame(self, corner_radius=12, fg_color="#ffffff", border_width=1, border_color="#e2e8f0")
        header.pack(fill="x", padx=16, pady=12)

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left", padx=16, pady=12)

        ctk.CTkLabel(title_box, text="QUANTBOT PRO (ANGEL ONE & NSE)", font=ctk.CTkFont(size=20, weight="bold"), text_color="#2563eb").pack(anchor="w")
        ctk.CTkLabel(title_box, text="5-Model Institutional NSE Engine (PyTorch + XGBoost + LightGBM + RF)", font=ctk.CTkFont(size=11), text_color="#64748b").pack(anchor="w")

        self.status_pill = ctk.CTkLabel(header, text="● BOT READY", font=ctk.CTkFont(size=12, weight="bold"), text_color="#64748b", fg_color="#f1f5f9", corner_radius=8, padx=12, pady=6)
        self.status_pill.pack(side="right", padx=16, pady=12)

        # 2. Control Panel
        control = ctk.CTkFrame(self, corner_radius=12, fg_color="#ffffff", border_width=1, border_color="#e2e8f0")
        control.pack(fill="x", padx=16, pady=(0, 12))
        control.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Target Market
        ctk.CTkLabel(control, text="Target NSE Stock / Index:", font=ctk.CTkFont(size=11, weight="bold"), text_color="#334155").grid(row=0, column=0, padx=12, pady=(10, 2), sticky="w")
        self.asset_var = ctk.StringVar(value="Reliance Industries")
        self.asset_menu = ctk.CTkOptionMenu(control, values=list(TRADINGVIEW_SYMBOLS.keys()), variable=self.asset_var, fg_color="#2563eb", button_color="#1d4ed8", font=ctk.CTkFont(size=11, weight="bold"))
        self.asset_menu.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="ew")

        # Capital Input
        ctk.CTkLabel(control, text="Capital (₹):", font=ctk.CTkFont(size=11, weight="bold"), text_color="#334155").grid(row=0, column=1, padx=12, pady=(10, 2), sticky="w")
        self.capital_entry = ctk.CTkEntry(control, placeholder_text="500", font=ctk.CTkFont(size=12, weight="bold"), fg_color="#f8fafc")
        self.capital_entry.insert(0, "500")
        self.capital_entry.grid(row=1, column=1, padx=12, pady=(0, 12), sticky="ew")

        # Risk Slider
        ctk.CTkLabel(control, text="Risk Per Trade (Max 2%):", font=ctk.CTkFont(size=11, weight="bold"), text_color="#334155").grid(row=0, column=2, padx=12, pady=(10, 2), sticky="w")
        self.risk_entry = ctk.CTkEntry(control, placeholder_text="2.0", font=ctk.CTkFont(size=12, weight="bold"), fg_color="#f8fafc")
        self.risk_entry.insert(0, "2.0")
        self.risk_entry.grid(row=1, column=2, padx=12, pady=(0, 12), sticky="ew")

        # Action Buttons
        btn_box = ctk.CTkFrame(control, fg_color="transparent")
        btn_box.grid(row=1, column=3, padx=12, pady=(0, 12), sticky="ew")
        btn_box.grid_columnconfigure((0, 1), weight=1)

        self.start_btn = ctk.CTkButton(btn_box, text="▶ START SCANNER", font=ctk.CTkFont(size=11, weight="bold"), fg_color="#10b981", hover_color="#059669", command=self.start_bot)
        self.start_btn.grid(row=0, column=0, padx=2, sticky="ew")

        self.stop_btn = ctk.CTkButton(btn_box, text="⏹ STOP", font=ctk.CTkFont(size=11, weight="bold"), fg_color="#ef4444", hover_color="#dc2626", state="disabled", command=self.stop_bot)
        self.stop_btn.grid(row=0, column=1, padx=2, sticky="ew")

        # 3. Live Metrics Cards
        kpi_frame = ctk.CTkFrame(self, corner_radius=12, fg_color="#ffffff", border_width=1, border_color="#e2e8f0")
        kpi_frame.pack(fill="x", padx=16, pady=(0, 12))
        kpi_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.card_price = self.create_kpi_card(kpi_frame, "LIVE MARKET PRICE", "₹--", 0)
        self.card_confidence = self.create_kpi_card(kpi_frame, "AI CONSENSUS EDGE", "--%", 1)
        self.card_risk = self.create_kpi_card(kpi_frame, "REGIME & RISK", "SCANNING", 2)
        self.card_position = self.create_kpi_card(kpi_frame, "ACTION SIGNAL", "HOLD / SCAN", 3)

        # 4. Live Audit Log Box
        log_frame = ctk.CTkFrame(self, corner_radius=12, fg_color="#ffffff", border_width=1, border_color="#e2e8f0")
        log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        ctk.CTkLabel(log_frame, text="Live Quantitative Decision Stream & Audit Logs:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#1e293b").pack(anchor="w", padx=16, pady=(12, 6))

        self.log_text = ctk.CTkTextbox(log_frame, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#f8fafc", text_color="#0f172a", border_width=1, border_color="#e2e8f0")
        self.log_text.pack(fill="both", expand=True, padx=16, pady=(0, 12))

    def create_kpi_card(self, parent, title, val, col):
        frame = ctk.CTkFrame(parent, corner_radius=8, fg_color="#f8fafc", border_width=1, border_color="#e2e8f0")
        frame.grid(row=0, column=col, padx=8, pady=10, sticky="ew")
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=9, weight="bold"), text_color="#64748b").pack(pady=(6, 0))
        lbl = ctk.CTkLabel(frame, text=val, font=ctk.CTkFont(size=15, weight="bold"), text_color="#0f172a")
        lbl.pack(pady=(0, 6))
        return lbl

    def queue_log(self, msg):
        self.msg_queue.put(("log", msg))

    def queue_update(self, widget_name, value, color=None):
        self.msg_queue.put(("update", (widget_name, value, color)))

    def process_queue(self):
        try:
            while not self.msg_queue.empty():
                item_type, payload = self.msg_queue.get_nowait()
                if item_type == "log":
                    t = time.strftime("%H:%M:%S")
                    self.log_text.insert("end", f"[{t}] {payload}\n")
                    self.log_text.see("end")
                elif item_type == "update":
                    widget_name, value, color = payload
                    target = getattr(self, widget_name, None)
                    if target:
                        if color:
                            target.configure(text=value, text_color=color)
                        else:
                            target.configure(text=value)
        except Exception:
            pass

        self.after(100, self.process_queue)

    def create_selenium_driver(self):
        """Creates a stealth WebDriver that hides automation banners."""
        try:
            import undetected_chromedriver as uc
            opts = uc.ChromeOptions()
            opts.add_argument("--start-maximized")
            driver = uc.Chrome(options=opts, use_subprocess=True)
            return driver
        except Exception as e1:
            self.queue_log(f"[INFO] UC init note ({e1}), using stealth Edge...")

        try:
            edge_opts = EdgeOptions()
            edge_opts.add_argument("--start-maximized")
            edge_opts.add_argument("--disable-blink-features=AutomationControlled")
            edge_opts.add_experimental_option("excludeSwitches", ["enable-automation"])
            edge_opts.add_experimental_option("useAutomationExtension", False)
            driver = webdriver.Edge(options=edge_opts)
            try:
                driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                    "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
                })
            except Exception:
                pass
            return driver
        except Exception as e2:
            self.queue_log(f"[WARN] Edge start note: {e2}")
            return None

    def inject_ai_overlay(self):
        if not self.driver:
            return
        try:
            overlay_file = os.path.join(os.path.dirname(__file__), "ai_overlay.js")
            if os.path.exists(overlay_file):
                with open(overlay_file, "r", encoding="utf-8") as f:
                    js_code = f.read()
                self.driver.execute_script(js_code)
        except Exception:
            pass

    def update_browser_hud(self, ai_data):
        if not self.driver:
            return
        try:
            payload_json = json.dumps(ai_data)
            script = f"if (window.updateAIThoughtHUD) {{ window.updateAIThoughtHUD({payload_json}); }}"
            self.driver.execute_script(script)
        except Exception:
            pass

    def start_bot(self):
        try:
            self.current_capital = float(self.capital_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid numeric capital.")
            return

        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.asset_menu.configure(state="disabled")
        self.status_pill.configure(text="● AI BOT SCANNING LIVE", text_color="#059669", fg_color="#d1fae5")

        self.bot_thread = threading.Thread(target=self.run_bot_worker, daemon=True)
        self.bot_thread.start()

    def stop_bot(self):
        self.is_running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.asset_menu.configure(state="normal")
        self.status_pill.configure(text="● AI BOT STOPPED", text_color="#dc2626", fg_color="#fee2e2")
        self.queue_log("[USER] AI Bot stopped.")

        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def on_close(self):
        self.stop_bot()
        self.destroy()

    def run_bot_worker(self):
        selected_name = self.asset_var.get()
        asset_info = TRADINGVIEW_SYMBOLS[selected_name]
        ticker = asset_info["ticker"]
        tv_symbol = asset_info["tv_symbol"]

        url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"
        self.queue_log(f"[START] Opening TradingView chart: {url}")

        self.driver = self.create_selenium_driver()
        if self.driver:
            try:
                self.driver.get(url)
                time.sleep(3)
                self.inject_ai_overlay()
            except Exception as e:
                self.queue_log(f"[WARN] Navigation note: {e}")

        risk_mgr = RiskManager(initial_capital=self.current_capital)

        while self.is_running:
            try:
                # 1. Fetch live market data
                df = fetch_market_data(ticker, period="1y", interval="1d")
                if df is not None and len(df) >= 30:
                    ai_analysis = self.ai_engine.analyze_live(df, symbol_name=selected_name)
                    
                    current_price = ai_analysis["price"]
                    confidence = ai_analysis["confidence"]
                    action = ai_analysis["action"]
                    risk_level = ai_analysis["risk_level"]
                    summary = ai_analysis["simple_summary"]
                    models = ai_analysis.get("model_breakdown", {})
                    deep_info = ai_analysis.get("deep", {})
                    regime = deep_info.get("market_phase", "EXPANSION")

                    formatted_price = f"₹{current_price:,.4f}" if current_price < 100 else f"₹{current_price:,.2f}"

                    # Update Desktop KPI Cards
                    self.queue_update("card_price", formatted_price)
                    self.queue_update("card_confidence", f"{confidence}%", "#059669" if confidence >= 60 else ("#dc2626" if confidence <= 42 else "#d97706"))
                    self.queue_update("card_risk", risk_level, "#dc2626" if "HIGH" in risk_level else ("#d97706" if "MODERATE" in risk_level else "#059669"))
                    self.queue_update("card_position", action, "#059669" if "BUY" in action else ("#dc2626" if "SELL" in action else "#0f172a"))

                    # Prepare Comprehensive Blueprint HUD payload
                    hud_payload = {
                        "price": current_price,
                        "confidence": confidence,
                        "action": action,
                        "risk_level": risk_level,
                        "simple_summary": summary,
                        "regime": regime,
                        "model_breakdown": models,
                        "suggested_sl": deep_info.get("suggested_sl", current_price * 0.99),
                        "suggested_tp1": deep_info.get("suggested_tp1", current_price * 1.02),
                        "safe_units": deep_info.get("safe_units", 20.0),
                        "key_levels": deep_info.get("key_levels", {})
                    }

                    # Inject and update HUD
                    self.inject_ai_overlay()
                    self.update_browser_hud(hud_payload)

                    # Log AI Reasoning
                    self.queue_log(f"[AI TICK] {formatted_price} | Consensus: {confidence}% -> {action} | Regime: {regime}")

                # Fast 1.5s refresh cycle for sub-second responsiveness
                for _ in range(2):
                    if not self.is_running:
                        break
                    time.sleep(0.75)

            except Exception as e:
                self.queue_log(f"[AI SCAN NOTE] {e}")
                time.sleep(2)


if __name__ == "__main__":
    app = TradingViewAngelOneAIBotApp()
    app.mainloop()
