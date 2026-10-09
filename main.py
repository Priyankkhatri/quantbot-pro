"""
Quantitative Trading System - Main CLI Runner
Allows running strategy backtests across different tickers and market regimes.
"""

import sys
import argparse
from tabulate import tabulate
from data_fetcher import fetch_market_data
from backtester import Backtester

# Ensure UTF-8 output across Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def print_banner():
    print("=" * 70)
    print("      QUANTITATIVE TRADING & RISK MANAGEMENT SYSTEM")
    print("=" * 70)


def run_strategy_analysis(ticker: str, period: str = "2y", capital: float = 10000.0):
    print_banner()
    data = fetch_market_data(ticker, period=period)
    if data is None or len(data) < 50:
        print(f"Error: Not enough data available for {ticker}.")
        return

    print(f"\nRunning Quantitative Backtest with Initial Capital: INR {capital:,.2f}")
    backtester = Backtester(data, initial_capital=capital)
    results = backtester.run()

    # Display Summary Table
    summary = [
        ["Asset Ticker", ticker],
        ["Time Period", f"{data.index[0].date()} to {data.index[-1].date()}"],
        ["Initial Capital", f"INR {results['initial_capital']:,.2f}"],
        ["Final Capital", f"INR {results['final_capital']:,.2f}"],
        ["Net PnL", f"INR {results['total_pnl']:+,.2f}"],
        ["Strategy Total Return", f"{results['strategy_return_pct']:+.2f}%"],
        ["Buy & Hold Benchmark Return", f"{results['buy_and_hold_return_pct']:+.2f}%"],
        ["Total Trades Executed", results['total_trades']],
        ["Winning Trades", results['winning_trades']],
        ["Losing Trades", results['losing_trades']],
        ["Win Rate", f"{results['win_rate_pct']:.2f}%"],
        ["Profit Factor", results['profit_factor']],
        ["Maximum Drawdown (Peak to Trough)", f"{results['max_drawdown_pct']:.2f}%"],
    ]

    print("\n" + tabulate(summary, headers=["Metric", "Value"], tablefmt="grid"))

    # Display Recent Trades
    trades = results['trades_list']
    if trades:
        print("\n--- LAST 5 TRADES EXECUTED ---")
        trade_rows = []
        for t in trades[-5:]:
            trade_rows.append([
                str(t['entry_time'].date()),
                str(t['exit_time'].date()),
                f"{t['entry_price']}",
                f"{t['exit_price']}",
                f"INR {t['net_pnl']:+,.2f}",
                f"{t['return_pct']:+,.2f}%",
                t['reason']
            ])
        print(tabulate(trade_rows, headers=["Entry", "Exit", "Buy Price", "Sell Price", "Net PnL", "Return %", "Exit Reason"], tablefmt="grid"))
    else:
        print("\nNo trades were triggered during this period under the strict risk criteria.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quant Strategy Backtest")
    parser.add_argument("--ticker", type=str, default="^NSEI", help="Ticker symbol (e.g., ^NSEI, USDINR=X, RELIANCE.NS, AAPL)")
    parser.add_argument("--period", type=str, default="2y", help="Historical data period (e.g., 1y, 2y, 5y)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital in account")
    
    args = parser.parse_args()
    run_strategy_analysis(ticker=args.ticker, period=args.period, capital=args.capital)
