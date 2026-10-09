@echo off
chcp 65001 >nul
title TradingView Live Auto-Trader Pro
cd /d "%~dp0"
echo Launching TradingView Live Auto-Trader Desktop Software...
python tradingview_bot.py
pause
