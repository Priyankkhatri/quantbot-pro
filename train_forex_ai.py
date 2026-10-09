"""
Forex & Multi-Market Institutional AI Model Trainer
Trains the 5-Model Multi-Head Quantitative Ensemble:
1. XGBoost Classifier
2. LightGBM Classifier
3. Random Forest Classifier
4. Extra Trees Classifier
5. Deep Neural Multi-Layer Perceptron (MLP)
Across 5 years of historical Forex, Indices, Crypto, and Equity data.
"""

import sys
import pandas as pd
from data_fetcher import fetch_market_data
from ml_model import InstitutionalForexAIEngine

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

TRAINING_ASSETS = {
    "USD/INR Forex": "USDINR=X",
    "EUR/USD Forex": "EURUSD=X",
    "GBP/USD Forex": "GBPUSD=X",
    "Nifty 50 Index": "^NSEI",
    "Bank Nifty Index": "^NSEBANK",
    "Bitcoin / USD": "BTC-USD",
    "Reliance Industries": "RELIANCE.NS",
    "Gold ETF BEES": "GOLDBEES.NS"
}


def run_training_pipeline():
    print("=" * 70)
    print("   QUANTBOT PRO: MULTI-MODEL QUANTITATIVE AI TRAINING PIPELINE")
    print("   [XGBoost + LightGBM + Random Forest + Extra Trees + Deep Neural MLP]")
    print("=" * 70)
    print("Downloading 5 years of high-resolution institutional datasets...\n")

    dataset_dict = {}
    for name, ticker in TRAINING_ASSETS.items():
        print(f"Fetching historical bars for {name} ({ticker})...")
        df = fetch_market_data(ticker, period="5y", interval="1d")
        if df is not None and len(df) > 100:
            dataset_dict[name] = df
            print(f"  ✓ Downloaded {len(df)} bars")
        else:
            print(f"  ⚠️ Warning: Could not fetch sufficient data for {name}")

    print(f"\n[Dataset Summary] Acquired {len(dataset_dict)} diverse market datasets.")
    print("Initializing Multi-Head Training & Calibration...\n")

    ai_engine = InstitutionalForexAIEngine(model_path="forex_quant_ai.joblib")
    success = ai_engine.train(dataset_dict)

    if success:
        print("\n" + "=" * 70)
        print("  🎉 MULTI-MODEL ENSEMBLE TRAINING COMPLETE!")
        print("  Checkpoint: forex_quant_ai.joblib")
        print("  Models Active: XGBoost | LightGBM | Random Forest | Extra Trees | Neural MLP")
        print("=" * 70)
    else:
        print("\n❌ Model training encountered an error.")


if __name__ == "__main__":
    run_training_pipeline()
