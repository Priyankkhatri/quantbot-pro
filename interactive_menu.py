"""
Interactive Menu for Easy Trading Backtesting
"""

import os
import sys

# Ensure UTF-8 console output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from main import run_strategy_analysis

ASSETS = {
    "1": ("USD/INR Currency (Forex)", "USDINR=X"),
    "2": ("Nifty 50 Index (India)", "^NSEI"),
    "3": ("Bank Nifty Index (India)", "^NSEBANK"),
    "4": ("Reliance Industries (NSE)", "RELIANCE.NS"),
    "5": ("Tata Motors (NSE)", "TATAMOTORS.NS"),
    "6": ("Apple Inc (US Market)", "AAPL"),
    "7": ("Custom Ticker Symbol", "CUSTOM")
}

def main():
    while True:
        print("\n" + "=" * 55)
        print("       QUANT TRADING STRATEGY TESTER")
        print("=" * 55)
        for key, (name, _) in ASSETS.items():
            print(f"  [{key}] {name}")
        print("  [0] Exit")
        print("=" * 55)
        
        choice = input("\nEnter choice number (0-7): ").strip()
        
        if choice == "0":
            print("\nExiting. Happy trading and learning!")
            break
        
        if choice in ASSETS:
            _, ticker = ASSETS[choice]
            if ticker == "CUSTOM":
                ticker = input("Enter ticker symbol (e.g. TSLA, INFY.NS, EURUSD=X): ").strip()
                if not ticker:
                    continue
            
            period = input("Enter historical time period (default: 2y, e.g., 1y, 2y, 5y): ").strip()
            if not period:
                period = "2y"
                
            capital_str = input("Enter starting capital in INR (default: 10000): ").strip()
            capital = float(capital_str) if capital_str.replace('.', '', 1).isdigit() else 10000.0
            
            run_strategy_analysis(ticker=ticker, period=period, capital=capital)
            input("\nPress Enter to return to menu...")
        else:
            print("Invalid selection! Please enter a number between 0 and 7.")

if __name__ == "__main__":
    main()
