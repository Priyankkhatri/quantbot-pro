"""
Equity Curve Plotter Module
Generates visual charts comparing the strategy equity curve against buy-and-hold benchmarks.
"""

import matplotlib.pyplot as plt
import os


def save_performance_chart(equity_series, data, ticker: str, output_path: str = "performance_chart.png"):
    plt.figure(figsize=(12, 6))
    
    # Normalize both to 100 base for fair percentage comparison
    norm_strategy = (equity_series / equity_series.iloc[0]) * 100
    norm_benchmark = (data['Close'] / data['Close'].iloc[0]) * 100

    plt.plot(norm_strategy.index, norm_strategy.values, label="Quant Strategy Equity", color="#00C853", linewidth=2)
    plt.plot(norm_benchmark.index, norm_benchmark.values, label=f"Buy & Hold ({ticker})", color="#B0BEC5", linestyle="--")

    plt.title(f"Quantitative Strategy Performance vs Benchmark: {ticker}", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Growth of Capital (Base 100)", fontsize=11)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Chart saved to: {output_path}")
