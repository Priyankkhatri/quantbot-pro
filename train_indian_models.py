"""
Indian Stock Market & Angel One SmartAPI ML Training Pipeline
Purged Walk-Forward Time Validation & Probability Calibration Edition
Trains 5-Model Ensemble (PyTorch ResNet, XGBoost, LightGBM, Random Forest, Extra Trees)
across Nifty 100 Multi-Sector Universe with strict zero-leakage embargo controls.
"""

import os
import sys
from rebuild_ml_pipeline import train_and_evaluate_walk_forward

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def train_and_save():
    print("Executing Purged Walk-Forward Training & Calibration Pipeline...")
    metrics, stocks = train_and_evaluate_walk_forward()
    return metrics, stocks


if __name__ == "__main__":
    train_and_save()
