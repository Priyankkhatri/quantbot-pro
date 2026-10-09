@echo off
chcp 65001 >nul
title Quant Trading Web Dashboard
cd /d "%~dp0"
echo Starting Quantitative Trading Live Dashboard...
echo Opening your web browser at http://127.0.0.1:5000 ...
python app.py
pause
