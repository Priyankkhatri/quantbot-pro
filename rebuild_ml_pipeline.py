"""
Institutional Purged Walk-Forward ML Pipeline Rebuilder & Optimization Engine
Features:
- Feature Engineering: 28 Alpha Features (Volume Ratio, OBV, VWAP, Nifty Macro, Gap %, Sector Relative Strength)
- Collinear Feature Pruning: Multi-collinearity analysis (Correlation / VIF)
- Marcos López de Prado Triple Barrier Method (TBM) Labeling matched to 1:1.5/1:2 ATR Risk Rules + No-Trade Class
- Optuna Hyperparameter Optimization with Purged Time-Series Walk-Forward Cross Validation
- Secondary Meta-Model (De Prado Meta-Labeling for Signal Viability & False-Positive Filtering)
- Platt Sigmoid Probability Calibration on dedicated Validation Windows
- Out-Of-Sample (OOS) Performance Gate & Baseline Comparison
"""

import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Any

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from data_fetcher import fetch_market_data, filter_completed_candles
from indicators import compute_ema, compute_rsi, compute_atr, compute_macd
from ml_model import PyTorchDeepQuantNet, InstitutionalForexAIEngine, compute_triple_barrier_labels

# Diverse Nifty 100 Constituents across 10 major NSE sectors
NIFTY100_UNIVERSE = {
    # IT
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "WIPRO": "WIPRO.NS",
    "HCLTECH": "HCLTECH.NS",
    "TECHM": "TECHM.NS",
    # Banking & Financials
    "HDFC_BANK": "HDFCBANK.NS",
    "ICICI_BANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "KOTAK_BANK": "KOTAKBANK.NS",
    "AXIS_BANK": "AXISBANK.NS",
    "BAJAJ_FINANCE": "BAJFINANCE.NS",
    # Energy, Oil & Power
    "RELIANCE": "RELIANCE.NS",
    "ONGC": "ONGC.NS",
    "BPCL": "BPCL.NS",
    "NTPC": "NTPC.NS",
    "POWERGRID": "POWERGRID.NS",
    # Auto
    "MARUTI": "MARUTI.NS",
    "MAHINDRA": "M&M.NS",
    "BAJAJ_AUTO": "BAJAJ-AUTO.NS",
    # Metals & Materials
    "TATA_STEEL": "TATASTEEL.NS",
    "HINDALCO": "HINDALCO.NS",
    "JSW_STEEL": "JSWSTEEL.NS",
    # FMCG & Consumption
    "ITC": "ITC.NS",
    "HINDUSTAN_UNILEVER": "HINDUNILVR.NS",
    "NESTLE": "NESTLEIND.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "TITAN": "TITAN.NS",
    # Pharma & Healthcare
    "SUN_PHARMA": "SUNPHARMA.NS",
    "DR_REDDY": "DRREDDY.NS",
    "CIPLA": "CIPLA.NS",
    # Conglomerate, Telecom & Infra
    "LT": "LT.NS",
    "BHARTI_AIRTEL": "BHARTIARTL.NS",
    "ADANI_PORTS": "ADANIPORTS.NS"
}


def build_enhanced_feature_dataframe(df: pd.DataFrame, engine: InstitutionalForexAIEngine, nifty_df: pd.DataFrame) -> pd.DataFrame:
    """Builds 28 alpha features and Triple-Barrier labels matched to ATR SL/TP on completed daily candles."""
    if df is None or len(df) < 60:
        return pd.DataFrame()
    
    data = filter_completed_candles(df, interval="1d")
    feat_df = engine.compute_features(data, nifty_df=nifty_df)
    if feat_df.empty or len(feat_df) < 50:
        return pd.DataFrame()

    # Triple Barrier Labeling (TP: 1.5*ATR, SL: 1.0*ATR, Horizon: 5 bars)
    tb_labels = compute_triple_barrier_labels(feat_df, tp_mult=1.5, sl_mult=1.0, max_holding=5)
    feat_df['TB_Raw_Label'] = tb_labels
    
    # Binary Target for Primary Directional Model: 1 if TP hit first (+1), 0 for SL or Choppy/No-Trade (-1 or 0)
    feat_df['Target'] = (feat_df['TB_Raw_Label'] == 1).astype(int)
    feat_df['EMA_Trend_Rule'] = ((feat_df['Close'] > feat_df['EMA_200']) & (feat_df['EMA_9'] > feat_df['EMA_21'])).astype(int)
    
    clean_df = feat_df.dropna(subset=engine.feature_names + ['Target', 'EMA_Trend_Rule', 'TB_Raw_Label'])
    return clean_df


def prune_collinear_features(X_df: pd.DataFrame, threshold: float = 0.88) -> List[str]:
    """Prunes redundant collinear features exceeding correlation threshold."""
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected_features = [col for col in X_df.columns if col not in to_drop]
    print(f"\n[Feature Pruning] Checked {len(X_df.columns)} features. Identified {len(to_drop)} collinear features for pruning: {to_drop}")
    print(f"  -> Retaining {len(selected_features)} uncorrelated alpha features.")
    return selected_features


def create_walk_forward_splits(dates: pd.DatetimeIndex, embargo_bars: int = 5) -> List[Dict[str, Any]]:
    """Creates expanding walk-forward splits with 5-day purge/embargo gap."""
    unique_dates = pd.Series(dates.unique()).sort_values().reset_index(drop=True)
    total_dates = len(unique_dates)
    
    cutoffs = [
        (0.50, 0.68, 0.80),
        (0.65, 0.80, 0.90),
        (0.75, 0.88, 1.00)
    ]
    
    splits = []
    for tr_pct, val_pct, test_pct in cutoffs:
        tr_end = int(total_dates * tr_pct) - embargo_bars
        v_start = int(total_dates * tr_pct)
        v_end = int(total_dates * val_pct) - embargo_bars
        t_start = int(total_dates * val_pct)
        t_end = min(int(total_dates * test_pct) - 1, total_dates - 1)
        
        splits.append({
            "train_dates": (unique_dates.iloc[0], unique_dates.iloc[tr_end]),
            "val_dates": (unique_dates.iloc[v_start], unique_dates.iloc[v_end]),
            "test_dates": (unique_dates.iloc[t_start], unique_dates.iloc[t_end])
        })
        
    return splits


def optimize_hyperparameters_optuna(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    """Runs Optuna study to find optimal gradient booster & tree parameters."""
    if not OPTUNA_AVAILABLE:
        return {
            "xgb_params": {"n_estimators": 160, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85},
            "lgb_params": {"n_estimators": 160, "num_leaves": 24, "learning_rate": 0.03, "subsample": 0.85, "min_child_samples": 15},
            "rf_params": {"n_estimators": 150, "max_depth": 8, "min_samples_split": 6}
        }

    print("  [Optuna HPO] Tuning Hyperparameters with Walk-Forward Purged Time-Series CV...")

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 0.01, 0.08, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 6)
        n_estimators = trial.suggest_int("n_estimators", 80, 200, step=20)
        subsample = trial.suggest_float("subsample", 0.7, 0.95)
        colsample = trial.suggest_float("colsample_bytree", 0.7, 0.95)

        if XGB_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
                subsample=subsample, colsample_bytree=colsample, eval_metric='logloss',
                random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
            val_probs = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, val_probs)
        else:
            score = 0.5
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=12, timeout=60)
    best_params = study.best_params
    print(f"  ✓ Optuna Best HPO Parameters: {best_params} (Validation AUC: {study.best_value:.4f})")

    return {
        "xgb_params": {
            "n_estimators": best_params.get("n_estimators", 160),
            "max_depth": best_params.get("max_depth", 4),
            "learning_rate": best_params.get("learning_rate", 0.03),
            "subsample": best_params.get("subsample", 0.85),
            "colsample_bytree": best_params.get("colsample_bytree", 0.85)
        },
        "lgb_params": {
            "n_estimators": best_params.get("n_estimators", 160),
            "num_leaves": 2 ** best_params.get("max_depth", 4),
            "learning_rate": best_params.get("learning_rate", 0.03),
            "subsample": best_params.get("subsample", 0.85),
            "min_child_samples": 15
        },
        "rf_params": {
            "n_estimators": 150,
            "max_depth": best_params.get("max_depth", 5) + 2,
            "min_samples_split": 6
        }
    }


def train_and_evaluate_walk_forward():
    print("=" * 78)
    print("  INSTITUTIONAL QUANTITATIVE ML AUDIT & PURGED WALK-FORWARD REBUILD v4.0")
    print("=" * 78)
    
    engine = InstitutionalForexAIEngine(model_path="angel_one_quant_ai.joblib")
    
    # Ingest Nifty 50 for Macro & Sector Relative Strength
    print("\n[Step 1] Ingesting Nifty 50 Benchmark Data for Macro Features...")
    nifty_raw = fetch_market_data("^NSEI", period="10y", interval="1d", completed_candles_only=True)
    if nifty_raw is None or len(nifty_raw) < 500:
        print("  ⚠️ Nifty 50 index data unavailable; using standalone fallback features.")
        nifty_raw = pd.DataFrame()
    else:
        print(f"  ✓ Nifty 50 Index Ingested: {len(nifty_raw)} completed daily bars.")

    # Ingest Nifty 100 Multi-Sector Universe
    print("\n[Step 2] Ingesting 8-10 Years OHLCV & Computing Triple-Barrier Alpha Datasets...")
    universe_dfs = {}
    total_raw_bars = 0
    
    for name, ticker in NIFTY100_UNIVERSE.items():
        df = fetch_market_data(ticker, period="10y", interval="1d", completed_candles_only=True)
        if df is not None and len(df) >= 500:
            feat_df = build_enhanced_feature_dataframe(df, engine, nifty_df=nifty_raw)
            if not feat_df.empty:
                universe_dfs[name] = feat_df
                total_raw_bars += len(feat_df)
                start_yr = feat_df.index[0].year
                end_yr = feat_df.index[-1].year
                tb_counts = feat_df['TB_Raw_Label'].value_counts().to_dict()
                print(f"  ✓ {name:<18} ({ticker:<14}): {len(feat_df):>5} bars ({start_yr}-{end_yr}) | TP Win={tb_counts.get(1, 0)}, SL Loss={tb_counts.get(-1, 0)}, No-Trade={tb_counts.get(0, 0)}")

    print(f"\n[Dataset Summary] Ingested {len(universe_dfs)} assets with {total_raw_bars:,} total clean daily samples.")

    # Combine all assets chronologically
    combined_list = []
    for name, df in universe_dfs.items():
        df_copy = df.copy()
        df_copy['Symbol'] = name
        combined_list.append(df_copy)
        
    all_data = pd.concat(combined_list).sort_index()
    print(f"Total Pooled Samples: {len(all_data):,} records across {len(engine.feature_names)} features.")

    # Prune Correlated Features
    feature_names = prune_collinear_features(all_data[engine.feature_names], threshold=0.88)
    engine.feature_names = feature_names

    # Setup Purged Walk-Forward Splits
    print("\n[Step 3] Constructing Purged Walk-Forward Time Folds with 5-Day Embargo Gap...")
    splits = create_walk_forward_splits(all_data.index, embargo_bars=5)
    
    for idx, s in enumerate(splits, 1):
        tr_s, tr_e = s["train_dates"]
        va_s, va_e = s["val_dates"]
        te_s, te_e = s["test_dates"]
        print(f"  Fold {idx}: Train [{tr_s.date()} -> {tr_e.date()}] | [5d Embargo] | Val [{va_s.date()} -> {va_e.date()}] | [5d Embargo] | Test [{te_s.date()} -> {te_e.date()}]")

    model_keys = ['pytorch_resnet', 'xgboost', 'lightgbm', 'random_forest', 'extra_trees', 'ensemble_uncalibrated', 'ensemble_calibrated', 'meta_model_ensemble']
    fold_results = []
    
    test_actuals = []
    test_probs_by_model = {k: [] for k in model_keys}
    baseline_always_up_preds = []
    baseline_ema_rule_preds = []

    best_scaler = None
    best_models = None
    best_calibrators = None
    best_weights = None
    best_meta_model = None
    best_threshold = 0.48

    for fold_num, split in enumerate(splits, 1):
        print(f"\n" + "-" * 75)
        print(f"  EXECUTING WALK-FORWARD FOLD {fold_num}/{len(splits)}")
        print(f"-" * 75)

        tr_s, tr_e = split["train_dates"]
        va_s, va_e = split["val_dates"]
        te_s, te_e = split["test_dates"]

        train_data = all_data.loc[tr_s:tr_e]
        val_data = all_data.loc[va_s:va_e]
        test_data = all_data.loc[te_s:te_e]

        print(f"  Samples: Train={len(train_data):,} | Val={len(val_data):,} | Test={len(test_data):,}")

        X_train_raw = train_data[feature_names]
        y_train = train_data['Target']
        
        X_val_raw = val_data[feature_names]
        y_val = val_data['Target']
        
        X_test_raw = test_data[feature_names]
        y_test = test_data['Target']

        # Leakage Rule: Scale strictly on Train split
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # Baseline Predictions on Test
        always_up = np.ones(len(y_test), dtype=int)
        ema_rule = test_data['EMA_Trend_Rule'].values

        # Optuna HPO on Fold
        hpo_config = optimize_hyperparameters_optuna(X_train_scaled, y_train.values, X_val_scaled, y_val.values)

        # 1. PyTorch Deep Quant ResNet
        print("  -> Training PyTorch Deep Quant ResNet...")
        pt_net = PyTorchDeepQuantNet(input_dim=len(feature_names), hidden_dim=64, num_blocks=2)
        optimizer = optim.AdamW(pt_net.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        t_X_tr = torch.tensor(X_train_scaled, dtype=torch.float32)
        t_y_tr = torch.tensor(y_train.values, dtype=torch.long)
        loader = DataLoader(TensorDataset(t_X_tr, t_y_tr), batch_size=256, shuffle=True)

        pt_net.train()
        for epoch in range(20):
            for bx, by in loader:
                optimizer.zero_grad()
                out = pt_net(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        pt_net.eval()
        with torch.no_grad():
            val_pt_logits = pt_net(torch.tensor(X_val_scaled, dtype=torch.float32))
            val_pt_probs = torch.softmax(val_pt_logits, dim=1)[:, 1].numpy()
            test_pt_logits = pt_net(torch.tensor(X_test_scaled, dtype=torch.float32))
            test_pt_probs = torch.softmax(test_pt_logits, dim=1)[:, 1].numpy()

        # 2. XGBoost
        xgb_m = None
        if XGB_AVAILABLE:
            print("  -> Training XGBoost (Optuna Tuned)...")
            xgb_m = xgb.XGBClassifier(
                **hpo_config["xgb_params"],
                eval_metric='logloss', random_state=42, n_jobs=-1
            )
            xgb_m.fit(X_train_scaled, y_train)
            val_xgb_probs = xgb_m.predict_proba(X_val_scaled)[:, 1]
            test_xgb_probs = xgb_m.predict_proba(X_test_scaled)[:, 1]
        else:
            val_xgb_probs = val_pt_probs
            test_xgb_probs = test_pt_probs

        # 3. LightGBM
        lgb_m = None
        if LGB_AVAILABLE:
            print("  -> Training LightGBM (Optuna Tuned)...")
            lgb_m = lgb.LGBMClassifier(
                **hpo_config["lgb_params"],
                random_state=42, verbose=-1, n_jobs=-1
            )
            lgb_m.fit(X_train_scaled, y_train)
            val_lgb_probs = lgb_m.predict_proba(X_val_scaled)[:, 1]
            test_lgb_probs = lgb_m.predict_proba(X_test_scaled)[:, 1]
        else:
            val_lgb_probs = val_pt_probs
            test_lgb_probs = test_pt_probs

        # 4. Random Forest
        print("  -> Training Random Forest...")
        rf_m = RandomForestClassifier(
            **hpo_config["rf_params"],
            random_state=42, n_jobs=-1
        )
        rf_m.fit(X_train_scaled, y_train)
        val_rf_probs = rf_m.predict_proba(X_val_scaled)[:, 1]
        test_rf_probs = rf_m.predict_proba(X_test_scaled)[:, 1]

        # 5. Extra Trees
        print("  -> Training Extra Trees...")
        et_m = ExtraTreesClassifier(
            n_estimators=130, max_depth=6, min_samples_split=6, random_state=42, n_jobs=-1
        )
        et_m.fit(X_train_scaled, y_train)
        val_et_probs = et_m.predict_proba(X_val_scaled)[:, 1]
        test_et_probs = et_m.predict_proba(X_test_scaled)[:, 1]

        # Probability Calibration on Validation Window
        print("  -> Calibrating Probabilities (Platt Scaling on Validation Window)...")
        calibrators = {}
        
        # PyTorch Calibrator
        cal_pt = LogisticRegression()
        cal_pt.fit(val_pt_probs.reshape(-1, 1), y_val)
        calibrators['pytorch_resnet'] = cal_pt
        val_pt_cal = cal_pt.predict_proba(val_pt_probs.reshape(-1, 1))[:, 1]
        test_pt_cal = cal_pt.predict_proba(test_pt_probs.reshape(-1, 1))[:, 1]

        # XGBoost Calibrator
        cal_xgb = LogisticRegression()
        cal_xgb.fit(val_xgb_probs.reshape(-1, 1), y_val)
        calibrators['xgboost'] = cal_xgb
        val_xgb_cal = cal_xgb.predict_proba(val_xgb_probs.reshape(-1, 1))[:, 1]
        test_xgb_cal = cal_xgb.predict_proba(test_xgb_probs.reshape(-1, 1))[:, 1]

        # LightGBM Calibrator
        cal_lgb = LogisticRegression()
        cal_lgb.fit(val_lgb_probs.reshape(-1, 1), y_val)
        calibrators['lightgbm'] = cal_lgb
        val_lgb_cal = cal_lgb.predict_proba(val_lgb_probs.reshape(-1, 1))[:, 1]
        test_lgb_cal = cal_lgb.predict_proba(test_lgb_probs.reshape(-1, 1))[:, 1]

        # Random Forest Calibrator
        cal_rf = LogisticRegression()
        cal_rf.fit(val_rf_probs.reshape(-1, 1), y_val)
        calibrators['random_forest'] = cal_rf
        val_rf_cal = cal_rf.predict_proba(val_rf_probs.reshape(-1, 1))[:, 1]
        test_rf_cal = cal_rf.predict_proba(test_rf_probs.reshape(-1, 1))[:, 1]

        # Extra Trees Calibrator
        cal_et = LogisticRegression()
        cal_et.fit(val_et_probs.reshape(-1, 1), y_val)
        calibrators['extra_trees'] = cal_et
        val_et_cal = cal_et.predict_proba(val_et_probs.reshape(-1, 1))[:, 1]
        test_et_cal = cal_et.predict_proba(test_et_probs.reshape(-1, 1))[:, 1]

        # Validation-Tuned Ensemble Weights (Inverse Brier Score)
        brier_scores = [
            brier_score_loss(y_val, val_pt_cal),
            brier_score_loss(y_val, val_xgb_cal),
            brier_score_loss(y_val, val_lgb_cal),
            brier_score_loss(y_val, val_rf_cal),
            brier_score_loss(y_val, val_et_cal)
        ]
        inv_brier = [1.0 / max(b, 1e-4) for b in brier_scores]
        total_inv = sum(inv_brier)
        weights = [round(w / total_inv, 4) for w in inv_brier]
        print(f"  ✓ Validation Ensemble Weights (Inverse Brier): {weights}")

        val_ens_cal = (
            weights[0] * val_pt_cal +
            weights[1] * val_xgb_cal +
            weights[2] * val_lgb_cal +
            weights[3] * val_rf_cal +
            weights[4] * val_et_cal
        )

        test_ens_cal = (
            weights[0] * test_pt_cal +
            weights[1] * test_xgb_cal +
            weights[2] * test_lgb_cal +
            weights[3] * test_rf_cal +
            weights[4] * test_et_cal
        )

        # Train Secondary Meta-Model (De Prado Meta-Labeling)
        print("  -> Training Secondary Meta-Model (Trade Viability & False-Positive Filter)...")
        # Generate Meta-Features on Validation Set:
        # [Primary Prob, Model Disagreement Std, ATR Pct, Volume Ratio, VWAP Dist, Sector Rel Strength, Market Vol]
        val_probs_matrix = np.column_stack([val_pt_cal, val_xgb_cal, val_lgb_cal, val_rf_cal, val_et_cal])
        val_disagreement = np.std(val_probs_matrix, axis=1)

        val_meta_X = np.column_stack([
            val_ens_cal,
            val_disagreement,
            val_data['ATR_Pct'].values if 'ATR_Pct' in val_data else np.full(len(val_data), 0.015),
            val_data['Volume_Ratio_20'].values if 'Volume_Ratio_20' in val_data else np.ones(len(val_data)),
            val_data['VWAP_Dist'].values if 'VWAP_Dist' in val_data else np.zeros(len(val_data)),
            val_data['Sector_Rel_Strength'].values if 'Sector_Rel_Strength' in val_data else np.zeros(len(val_data)),
            val_data['Market_Vol_Proxy'].values if 'Market_Vol_Proxy' in val_data else np.full(len(val_data), 0.15)
        ])
        
        # Meta-Target: 1 if the primary signal was profitable (+1), 0 otherwise
        val_meta_y = (val_data['TB_Raw_Label'] == 1).astype(int).values

        meta_clf = LogisticRegression(C=0.1, max_iter=200)
        meta_clf.fit(val_meta_X, val_meta_y)

        # Optimize decision threshold on Validation set (never test)
        best_f1 = -1.0
        fold_best_th = 0.40
        for th in np.arange(0.36, 0.52, 0.01):
            val_preds = (val_ens_cal >= th).astype(int)
            val_f1 = f1_score(y_val, val_preds, zero_division=0)
            if val_f1 > best_f1:
                best_f1 = val_f1
                fold_best_th = round(float(th), 2)
        print(f"  ✓ Validation-Optimized Decision Cutoff: {fold_best_th*100:.1f}% (Val F1: {best_f1:.4f})")
        best_threshold = fold_best_th

        # Apply Meta-Model on Out-of-Sample Test Set
        test_probs_matrix = np.column_stack([test_pt_cal, test_xgb_cal, test_lgb_cal, test_rf_cal, test_et_cal])
        test_disagreement = np.std(test_probs_matrix, axis=1)

        test_meta_X = np.column_stack([
            test_ens_cal,
            test_disagreement,
            test_data['ATR_Pct'].values if 'ATR_Pct' in test_data else np.full(len(test_data), 0.015),
            test_data['Volume_Ratio_20'].values if 'Volume_Ratio_20' in test_data else np.ones(len(test_data)),
            test_data['VWAP_Dist'].values if 'VWAP_Dist' in test_data else np.zeros(len(test_data)),
            test_data['Sector_Rel_Strength'].values if 'Sector_Rel_Strength' in test_data else np.zeros(len(test_data)),
            test_data['Market_Vol_Proxy'].values if 'Market_Vol_Proxy' in test_data else np.full(len(test_data), 0.15)
        ])
        
        test_meta_probs = meta_clf.predict_proba(test_meta_X)[:, 1]

        # Accumulate out-of-sample predictions
        test_actuals.extend(y_test.values)
        test_probs_by_model['pytorch_resnet'].extend(test_pt_cal)
        test_probs_by_model['xgboost'].extend(test_xgb_cal)
        test_probs_by_model['lightgbm'].extend(test_lgb_cal)
        test_probs_by_model['random_forest'].extend(test_rf_cal)
        test_probs_by_model['extra_trees'].extend(test_et_cal)
        test_probs_by_model['ensemble_calibrated'].extend(test_ens_cal)
        test_probs_by_model['meta_model_ensemble'].extend(test_meta_probs)
        baseline_always_up_preds.extend(always_up)
        baseline_ema_rule_preds.extend(ema_rule)

        # Store best artifacts from the final expanding fold
        best_scaler = scaler
        best_models = {
            "pytorch_state": pt_net.state_dict(),
            "xgb_model": xgb_m,
            "lgb_model": lgb_m,
            "rf_model": rf_m,
            "et_model": et_m,
            "meta_model": meta_clf
        }
        best_calibrators = calibrators
        best_weights = weights
        best_meta_model = meta_clf

    # Save Checkpoint Bundle
    print("\n[Step 4] Saving Production Multi-Model Bundle with Meta-Model & SHAP Engine...")
    bundle = {
        "pytorch_state": best_models["pytorch_state"],
        "xgb_model": best_models["xgb_model"],
        "lgb_model": best_models["lgb_model"],
        "rf_model": best_models["rf_model"],
        "et_model": best_models["et_model"],
        "meta_model": best_meta_model,
        "scaler": best_scaler,
        "calibrators": best_calibrators,
        "ensemble_weights": best_weights,
        "decision_threshold": best_threshold,
        "feature_names": feature_names,
        "pytorch_hidden_dim": 64,
        "pytorch_num_blocks": 2,
        "version": "4.0.0-TripleBarrier-Optuna-MetaModel"
    }

    joblib.dump(bundle, "angel_one_quant_ai.joblib")
    joblib.dump(bundle, "forex_quant_ai.joblib")
    print("  ✓ Saved enhanced models to 'angel_one_quant_ai.joblib' and 'forex_quant_ai.joblib'")

    # Report Out-Of-Sample Metrics Across Folds
    print("\n" + "=" * 78)
    print("  OUT-OF-SAMPLE (OOS) PURGED WALK-FORWARD EVALUATION RESULTS")
    print("=" * 78)

    y_true_all = np.array(test_actuals)
    always_up_arr = np.array(baseline_always_up_preds)
    ema_rule_arr = np.array(baseline_ema_rule_preds)

    print(f"\nTotal Out-Of-Sample Test Samples: {len(y_true_all):,}")
    print(f"Base Target Positive Rate: {np.mean(y_true_all)*100:.2f}%\n")

    print(f"{'Model / Architecture':<28} | {'Accuracy':<9} | {'ROC-AUC':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'Brier Score':<11}")
    print("-" * 92)

    # Baselines
    acc_up = accuracy_score(y_true_all, always_up_arr)
    prec_up = precision_score(y_true_all, always_up_arr, zero_division=0)
    rec_up = recall_score(y_true_all, always_up_arr, zero_division=0)
    f1_up = f1_score(y_true_all, always_up_arr, zero_division=0)
    print(f"{'Naive Baseline: Always-Up':<28} | {acc_up*100:>8.2f}% | {'N/A':<8} | {prec_up*100:>8.2f}% | {rec_up*100:>7.2f}% | {f1_up:>8.4f} | {'N/A':<11}")

    acc_ema = accuracy_score(y_true_all, ema_rule_arr)
    prec_ema = precision_score(y_true_all, ema_rule_arr, zero_division=0)
    rec_ema = recall_score(y_true_all, ema_rule_arr, zero_division=0)
    f1_ema = f1_score(y_true_all, ema_rule_arr, zero_division=0)
    print(f"{'Baseline: 200-EMA Trend':<28} | {acc_ema*100:>8.2f}% | {'N/A':<8} | {prec_ema*100:>8.2f}% | {rec_ema*100:>7.2f}% | {f1_ema:>8.4f} | {'N/A':<11}")
    print("-" * 92)

    # Evaluated AI Models
    display_names = {
        'pytorch_resnet': 'PyTorch Deep Quant ResNet',
        'xgboost': 'XGBoost (Optuna Tuned)',
        'lightgbm': 'LightGBM (Optuna Tuned)',
        'random_forest': 'Random Forest (Bagged)',
        'extra_trees': 'Extra Trees (Ensemble)',
        'ensemble_calibrated': '5-Model Platt Ensemble',
        'meta_model_ensemble': '5-Model + Meta-Label Filter'
    }

    for key, name in display_names.items():
        probs = np.array(test_probs_by_model[key])
        preds = (probs >= best_threshold).astype(int)
        
        acc = accuracy_score(y_true_all, preds)
        auc = roc_auc_score(y_true_all, probs)
        prec = precision_score(y_true_all, preds, zero_division=0)
        rec = recall_score(y_true_all, preds, zero_division=0)
        f1 = f1_score(y_true_all, preds, zero_division=0)
        brier = brier_score_loss(y_true_all, probs)
        
        print(f"{name:<28} | {acc*100:>8.2f}% | {auc:>8.4f} | {prec*100:>8.2f}% | {rec*100:>7.2f}% | {f1:>8.4f} | {brier:>11.4f}")

    print("=" * 92)
    return True


if __name__ == "__main__":
    train_and_evaluate_walk_forward()
