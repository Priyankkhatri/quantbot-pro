"""
Telegram Alert Notifier Module for QuantBot Pro
Provides asynchronous, non-blocking delivery of trading signals, fills, rejections,
SL/TP hits, risk events, kill-switch activations, and daily P&L summaries.
Uses urllib.request (zero external deps) with timeout and threading.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
import threading
import logging
from datetime import datetime

logger = logging.getLogger("quantbot.telegram")

class TelegramNotifier:
    def __init__(self, bot_token=None, chat_id=None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self._enabled = bool(self.bot_token and self.chat_id)

    def update_credentials(self, bot_token, chat_id):
        """Update credentials at runtime."""
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()
        self._enabled = bool(self.bot_token and self.chat_id)

    @property
    def is_configured(self):
        return bool(self.bot_token and self.chat_id)

    def _send_payload(self, text, parse_mode="HTML"):
        """Internal worker to execute HTTP POST to Telegram Bot API."""
        if not self.is_configured:
            logger.debug("Telegram not configured. Alert suppressed.")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, 
                data=data, 
                headers={"Content-Type": "application/json", "User-Agent": "QuantBotPro/4.0"}
            )
            with urllib.request.urlopen(req, timeout=5.0) as response:
                res_data = response.read().decode("utf-8")
                res_json = json.loads(res_data)
                return res_json.get("ok", False)
        except Exception as e:
            logger.warning(f"Telegram notification dispatch error: {type(e).__name__}: {str(e)}")
            return False

    def send_message_async(self, text, parse_mode="HTML"):
        """Dispatches message in a background daemon thread so trading is never blocked."""
        if not self.is_configured:
            return
        thread = threading.Thread(target=self._send_payload, args=(text, parse_mode), daemon=True)
        thread.start()

    def send_signal_alert(self, symbol, action, price, confidence, stop_loss, take_profit, verdict_reason="", mode="PAPER"):
        """Alerts when a new ML ensemble signal is generated."""
        badge = "🟢 BUY" if action.upper() == "BUY" else ("🔴 SELL" if action.upper() == "SELL" else "⚪ NEUTRAL")
        mode_badge = f"<b>[{mode.upper()} MODE]</b>"
        
        text = (
            f"⚡ <b>QUANTBOT SIGNAL ALERT</b> {mode_badge}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Instrument:</b> <code>{symbol}</code>\n"
            f"<b>Signal:</b> {badge}\n"
            f"<b>Price:</b> ₹{price:,.2f}\n"
            f"<b>Confidence:</b> {confidence:.1f}%\n"
            f"<b>Stop Loss:</b> ₹{stop_loss:,.2f}\n"
            f"<b>Target:</b> ₹{take_profit:,.2f}\n"
        )
        if verdict_reason:
            text += f"<b>Rationale:</b> <i>{verdict_reason}</i>\n"
        text += f"🕒 <i>{datetime.now().strftime('%d-%b-%Y %H:%M:%S IST')}</i>"
        
        self.send_message_async(text)

    def send_order_fill_alert(self, symbol, action, quantity, fill_price, order_id, mode="PAPER"):
        """Alerts when an order gets filled on exchange/paper."""
        side_emoji = "🟢" if action.upper() == "BUY" else "🔴"
        text = (
            f"✅ <b>ORDER FILLED</b> [{mode.upper()}]\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Symbol:</b> <code>{symbol}</code>\n"
            f"<b>Side:</b> {side_emoji} {action.upper()}\n"
            f"<b>Quantity:</b> {quantity}\n"
            f"<b>Fill Price:</b> ₹{fill_price:,.2f}\n"
            f"<b>Total Value:</b> ₹{(quantity * fill_price):,.2f}\n"
            f"<b>Order ID:</b> <code>{order_id}</code>\n"
            f"🕒 <i>{datetime.now().strftime('%d-%b-%Y %H:%M:%S IST')}</i>"
        )
        self.send_message_async(text)

    def send_rejection_alert(self, symbol, action, quantity, reason, mode="PAPER"):
        """Alerts when an order is rejected by broker or risk checks."""
        text = (
            f"⚠️ <b>ORDER REJECTED</b> [{mode.upper()}]\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Symbol:</b> <code>{symbol}</code>\n"
            f"<b>Side:</b> {action.upper()}\n"
            f"<b>Quantity:</b> {quantity}\n"
            f"<b>Reason:</b> <code>{reason}</code>\n"
            f"🕒 <i>{datetime.now().strftime('%d-%b-%Y %H:%M:%S IST')}</i>"
        )
        self.send_message_async(text)

    def send_sl_tp_alert(self, symbol, exit_type, exit_price, pnl_rupees, pnl_pct, mode="PAPER"):
        """Alerts when Stop-Loss or Take-Profit is hit."""
        emoji = "🎯 TARGET HIT" if exit_type.upper() in ["TP", "TAKE_PROFIT", "TARGET"] else "🛑 STOP-LOSS HIT"
        pnl_badge = f"+₹{pnl_rupees:,.2f} (+{pnl_pct:.2f}%)" if pnl_rupees >= 0 else f"-₹{abs(pnl_rupees):,.2f} ({pnl_pct:.2f}%)"
        
        text = (
            f"{emoji} [{mode.upper()}]\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Symbol:</b> <code>{symbol}</code>\n"
            f"<b>Exit Type:</b> {exit_type.upper()}\n"
            f"<b>Exit Price:</b> ₹{exit_price:,.2f}\n"
            f"<b>Realized P&L:</b> <b>{pnl_badge}</b>\n"
            f"🕒 <i>{datetime.now().strftime('%d-%b-%Y %H:%M:%S IST')}</i>"
        )
        self.send_message_async(text)

    def send_kill_switch_alert(self, activated=True, reason="Manual user trigger", triggered_by="Web UI"):
        """Alerts immediately upon Kill Switch trigger or reset."""
        if activated:
            text = (
                f"🚨🚨 <b>EMERGENCY KILL SWITCH ACTIVATED</b> 🚨🚨\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"<b>Status:</b> ALL TRADING HALTED\n"
                f"<b>Triggered By:</b> {triggered_by}\n"
                f"<b>Reason:</b> {reason}\n"
                f"<b>Action:</b> Open orders cancelled, positions safe-guarded.\n"
                f"🕒 <i>{datetime.now().strftime('%d-%b-%Y %H:%M:%S IST')}</i>"
            )
        else:
            text = (
                f"🛡️ <b>KILL SWITCH DEACTIVATED (RESET)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"<b>Status:</b> Trading controls restored to NORMAL.\n"
                f"🕒 <i>{datetime.now().strftime('%d-%b-%Y %H:%M:%S IST')}</i>"
            )
        self.send_message_async(text)

    def send_daily_summary(self, date_str, starting_equity, ending_equity, daily_pnl, pnl_pct, trade_count, win_rate, mode="PAPER"):
        """Alerts daily session performance summary at market close."""
        pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
        pnl_text = f"+₹{daily_pnl:,.2f}" if daily_pnl >= 0 else f"-₹{abs(daily_pnl):,.2f}"
        
        text = (
            f"📊 <b>END-OF-DAY PERFORMANCE SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Date:</b> {date_str} | <b>Mode:</b> {mode.upper()}\n"
            f"<b>Starting Equity:</b> ₹{starting_equity:,.2f}\n"
            f"<b>Ending Equity:</b> ₹{ending_equity:,.2f}\n"
            f"<b>Net Daily P&L:</b> {pnl_emoji} <b>{pnl_text} ({pnl_pct:+.2f}%)</b>\n"
            f"<b>Total Trades Today:</b> {trade_count}\n"
            f"<b>Win Rate:</b> {win_rate:.1f}%\n"
            f"🕒 <i>{datetime.now().strftime('%d-%b-%Y %H:%M:%S IST')}</i>"
        )
        self.send_message_async(text)

    def send_forward_paper_weekly_summary(self, summary_html: str):
        """Dispatches weekly forward paper-testing performance summary."""
        self.send_message_async(summary_html)

    def send_error_alert(self, error_type, message):
        """Alerts on critical system runtime exception."""
        text = (
            f"❌ <b>SYSTEM ERROR WARNING</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Type:</b> <code>{error_type}</code>\n"
            f"<b>Message:</b> {message[:300]}\n"
            f"🕒 <i>{datetime.now().strftime('%d-%b-%Y %H:%M:%S IST')}</i>"
        )
        self.send_message_async(text)

# Singleton Instance
telegram_notifier = TelegramNotifier()

