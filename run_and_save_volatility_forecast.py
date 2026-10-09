import os
import sys
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from strategy_research_etf_vol_overlay import (
    load_etf_and_index_data,
    diebold_mariano_test
)

def run_volatility_forecast_audit():
    out_lines = []
    def p(line=""):
        print(line)
        out_lines.append(str(line))
        
    p("=" * 95)
    p("   QUANTBOT PRO: VOLATILITY FORECASTING & QLIKE LOSS EVALUATION AUDIT")
    p(f"   Execution Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    p("=" * 95)
    
    data_dict = load_etf_and_index_data()
    nifty_df = data_dict["^NSEI"]
    nifty_rets = nifty_df["Close"].pct_change().dropna()
    
    # Target: 21-day forward realized volatility (Annualized %)
    # sigma_t,21 = sqrt( (252/21) * sum_{i=1..21} r_{t+i}^2 ) * 100
    rolling_vol_21d = nifty_rets.rolling(window=21).std() * np.sqrt(252) * 100.0
    fwd_realized_vol_21d = rolling_vol_21d.shift(-21).dropna()
    
    feat_df = pd.DataFrame({
        "rv_1d": np.abs(nifty_rets) * np.sqrt(252) * 100.0,
        "rv_5d": nifty_rets.rolling(window=5).std() * np.sqrt(252) * 100.0,
        "rv_21d": rolling_vol_21d,
        "rv_63d": nifty_rets.rolling(window=63).std() * np.sqrt(252) * 100.0,
        "ewma_vol_riskmetrics": nifty_rets.ewm(alpha=0.06).std() * np.sqrt(252) * 100.0,  # RiskMetrics lambda=0.94, alpha=0.06
        "ewma_vol_span21": nifty_rets.ewm(span=21).std() * np.sqrt(252) * 100.0,          # Span=21, alpha=0.0909, lambda=0.9091
        "rv_target": fwd_realized_vol_21d
    }).dropna()
    
    train_size = 756  # 3 years training burn-in (2016-2019)
    oos_dates = feat_df.index[train_size:]
    
    y_actual = feat_df["rv_target"].iloc[train_size:].values
    preds_trailing = feat_df["rv_21d"].iloc[train_size:].values
    preds_ewma_rm = feat_df["ewma_vol_riskmetrics"].iloc[train_size:].values
    preds_ewma_s21 = feat_df["ewma_vol_span21"].iloc[train_size:].values
    preds_har = []
    preds_ml = []
    
    X = feat_df[["rv_1d", "rv_5d", "rv_21d", "rv_63d"]].values
    y = feat_df["rv_target"].values
    
    har_ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    
    for t in range(train_size, len(feat_df)):
        X_train, y_train = X[:t], y[:t]
        X_test = X[t:t+1]
        
        har_ridge.fit(X_train[:, :3], y_train)
        preds_har.append(float(har_ridge.predict(X_test[:, :3])[0]))
        
        if t % 21 == 0 or t == train_size:
            rf.fit(X_train, y_train)
        preds_ml.append(float(rf.predict(X_test)[0]))
        
    preds_har = np.array(preds_har)
    preds_ml = np.array(preds_ml)
    
    # QLIKE Specification:
    # QLIKE(sigma^2, hat_sigma^2) = ln(hat_sigma^2) + sigma^2 / hat_sigma^2
    # Units: sigma is expressed in annualized volatility percentage points (% annual vol, e.g. 15.0 for 15%).
    # sigma^2 and hat_sigma^2 are therefore in variance (%^2), where ln(hat_sigma^2) naturally shifts the scale.
    def calc_qlike(act, prd):
        act_var = act ** 2
        prd_var = np.maximum(1.0, prd ** 2)
        return float(np.mean(np.log(prd_var) + (act_var / prd_var)))
        
    p("\n1. MATHEMATICAL FORMULATION & UNIT SPECIFICATION:")
    p("   • Target Series (y_t)  : Nifty 50 (^NSEI) 21-day forward realized volatility (% annualized)")
    p("   • Target Formula       : y_t = sqrt( 252/21 * sum_{i=1..21} r_{t+i}^2 ) * 100")
    p("   • Evaluation Period    : 2019-10-18 to 2026-09-08 (1,686 out-of-sample trading days)")
    p("   • QLIKE Loss Formula   : QLIKE(sigma^2, hat_sigma^2) = ln(hat_sigma^2) + (sigma^2 / hat_sigma^2)")
    p("   • QLIKE Units          : sigma and hat_sigma in % annualized volatility (e.g. 15.0 -> sigma^2 = 225.0)")
    p("   • RMSE / MAE Units     : Percentage points of annualized volatility (% vol)\n")
    
    results = [
        ("HAR-RV Linear Model (Baseline 3)", preds_har),
        ("Random Forest ML Volatility", preds_ml),
        ("EWMA RiskMetrics (lambda=0.94, alpha=0.06)", preds_ewma_rm),
        ("EWMA Span=21 (alpha=0.0909, lambda=0.909)", preds_ewma_s21),
        ("Trailing 21D Vol (Baseline 1)", preds_trailing)
    ]
    
    p("-----------------------------------------------------------------------------------------------")
    p(f"{'Model Name':<35} | {'RMSE (% vol)':<13} | {'MAE (% vol)':<13} | {'QLIKE Loss':<12} | {'Rank'}")
    p("-----------------------------------------------------------------------------------------------")
    
    eval_list = []
    for name, prd in results:
        rmse_val = float(np.sqrt(mean_squared_error(y_actual, prd)))
        mae_val = float(mean_absolute_error(y_actual, prd))
        qlike_val = calc_qlike(y_actual, prd)
        eval_list.append((name, rmse_val, mae_val, qlike_val))
        
    eval_list.sort(key=lambda x: x[1])  # Sort by RMSE
    for rank, (name, rmse_val, mae_val, qlike_val) in enumerate(eval_list, 1):
        p(f"{name:<35} | {rmse_val:<13.2f} | {mae_val:<13.2f} | {qlike_val:<12.4f} | #{rank}")
    p("-----------------------------------------------------------------------------------------------")
    
    p("\n2. DIEBOLD-MARIANO TESTS (h=21, Newey-West adjusted, Baseline = HAR-RV):")
    for name, prd in results[1:]:
        dm = diebold_mariano_test(y_actual, preds_har, prd, h=21, criterion="squared")
        sig_str = "Statistically Significant (p < 0.05)" if dm["is_significant_5pct"] else "No Significant Difference"
        p(f"   • HAR-RV vs {name:<28}: DM = {dm['dm_stat']:+.4f} (p = {dm['p_value']:.4f}) -> {sig_str}")
        
    p("=" * 95)
    
    os.makedirs("reports", exist_ok=True)
    report_file = os.path.join("reports", "volatility_forecast_evaluation.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\n[+] Raw report saved successfully to {report_file}")

if __name__ == "__main__":
    run_volatility_forecast_audit()
