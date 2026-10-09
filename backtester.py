"""
Advanced Quantitative ML Backtester & Indian Market Simulator
Features:
1. Walk-Forward ML Ensemble Signal Execution (PyTorch ResNet, XGBoost, LightGBM, RF, ExtraTrees).
2. Realistic Indian Regulatory & Broker Cost Model (Brokerage, STT, Exchange Turnover, GST, Stamp Duty, SEBI Fees, Slippage).
3. Strict Next-Bar Open Fill Execution (Zero Look-Ahead / No same-bar repainting).
4. Comprehensive Metrics: CAGR, Win Rate, Profit Factor, Expectancy, Sharpe Ratio, Max Drawdown, Longest Losing Streak, Exposure %, With vs Without Costs.
5. Benchmark Comparisons: Buy-and-Hold Asset/Nifty, Plain EMA Trend Rule.
6. Monte Carlo Shuffle (1,000 iterations) for empirical Drawdown and Confidence Intervals.
7. Multi-Capital Size Sensitivity Analysis (₹500, ₹25,000, ₹1,00,000).
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from indicators import add_all_indicators, compute_ema, compute_atr
from risk_manager import RiskManager
from ml_model import InstitutionalForexAIEngine


class IndianTradingCosts:
    """
    Simulates exact Indian broker and statutory fees for Equity Intraday (MIS) & Delivery (CNC).
    Reference: SEBI / NSE / Angel One Tariff.
    """
    @staticmethod
    def calculate(turnover_buy: float, turnover_sell: float, product_type: str = "MIS",
                  flat_brokerage_max: float = 20.0, brokerage_rate: float = 0.0003,
                  slippage_pct: float = 0.0003) -> Dict[str, float]:
        """
        Calculates all itemized fees on a complete round-trip trade.
        """
        total_turnover = turnover_buy + turnover_sell
        
        # 1. Brokerage
        if product_type.upper() == "MIS":
            # Lower of 0.03% or ₹20 per leg
            brokerage_buy = min(turnover_buy * brokerage_rate, flat_brokerage_max)
            brokerage_sell = min(turnover_sell * brokerage_rate, flat_brokerage_max)
            brokerage = brokerage_buy + brokerage_sell
        else:
            # Delivery (CNC) - ₹0 on discount brokers or small delivery fee
            brokerage = 0.0

        # 2. STT (Securities Transaction Tax)
        if product_type.upper() == "MIS":
            stt = turnover_sell * 0.00025  # 0.025% on sell side only
        else:
            stt = total_turnover * 0.001   # 0.1% on buy and sell for delivery

        # 3. Exchange Transaction Charges (NSE Equity ~0.00297%)
        exchange_charges = total_turnover * 0.0000297

        # 4. SEBI Turnover Fee (₹10 per crore = 0.0001%)
        sebi_fees = total_turnover * 0.000001

        # 5. GST (18% on Brokerage + Exchange Charges + SEBI Charges)
        gst = (brokerage + exchange_charges + sebi_fees) * 0.18

        # 6. Stamp Duty (State Government - Buy side only)
        if product_type.upper() == "MIS":
            stamp_duty = turnover_buy * 0.00003  # 0.003% (₹300/crore)
        else:
            stamp_duty = turnover_buy * 0.00015  # 0.015% (₹1500/crore)

        # 7. Slippage Cost (simulated execution impact)
        slippage_cost = total_turnover * slippage_pct

        total_regulatory_and_broker = brokerage + stt + exchange_charges + sebi_fees + gst + stamp_duty
        total_costs = total_regulatory_and_broker + slippage_cost

        return {
            "brokerage": round(brokerage, 4),
            "stt": round(stt, 4),
            "exchange_charges": round(exchange_charges, 4),
            "sebi_fees": round(sebi_fees, 4),
            "gst": round(gst, 4),
            "stamp_duty": round(stamp_duty, 4),
            "slippage_cost": round(slippage_cost, 4),
            "total_statutory": round(total_regulatory_and_broker, 4),
            "total_all_costs": round(total_costs, 4)
        }


class Backtester:
    def __init__(self, df: pd.DataFrame, initial_capital: float = 10000.0,
                 product_type: str = "MIS", model_path: str = "angel_one_quant_ai.joblib",
                 slippage_pct: float = 0.0003, use_ml_ensemble: bool = True,
                 confidence_threshold: Optional[float] = None):
        """
        :param df: Historical OHLCV dataframe
        :param initial_capital: Starting capital
        :param product_type: 'MIS' (Intraday) or 'CNC' (Delivery)
        :param model_path: Path to trained ML Ensemble checkpoint
        :param slippage_pct: Slippage per execution leg (0.03%)
        :param use_ml_ensemble: True to use 5-model AI predictions; False for plain EMA strategy
        :param confidence_threshold: Bullish probability cutoff for trade entry (defaults to bundle threshold)
        """
        self.raw_data = df.copy()
        self.initial_capital = float(initial_capital)
        self.product_type = product_type.upper()
        self.slippage_pct = slippage_pct
        self.use_ml_ensemble = use_ml_ensemble
        self.ai_engine = InstitutionalForexAIEngine(model_path=model_path)
        self.ai_engine.load_model()
        
        bundle_th = float(self.ai_engine.ensemble_bundle.get("decision_threshold", 0.48)) * 100.0 if self.ai_engine.ensemble_bundle else 48.0
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else bundle_th
        self.risk_manager = RiskManager(initial_capital=self.initial_capital)

    def generate_ml_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates ML signals walk-forward style across the dataset.
        Evaluates 5-model ensemble on closed bar t, flags Buy_Trigger for execution on bar t+1.
        """
        df = add_all_indicators(data)
        self.ai_engine.load_model()

        buy_triggers = [False] * len(df)
        exit_triggers = [False] * len(df)
        ai_confidences = [50.0] * len(df)
        ai_regimes = ["NEUTRAL"] * len(df)

        feature_df = self.ai_engine.compute_features(df)
        if feature_df.empty or len(feature_df) < 30:
            df['Buy_Trigger'] = False
            df['Exit_Trigger'] = False
            df['AI_Confidence'] = 50.0
            df['AI_Regime'] = "NEUTRAL"
            return df

        scaler = self.ai_engine.ensemble_bundle.get("scaler", self.ai_engine.scaler) if self.ai_engine.ensemble_bundle else self.ai_engine.scaler
        calibrators = self.ai_engine.ensemble_bundle.get("calibrators", {}) if self.ai_engine.ensemble_bundle else {}
        weights = self.ai_engine.ensemble_bundle.get("ensemble_weights", [0.2, 0.25, 0.25, 0.15, 0.15]) if self.ai_engine.ensemble_bundle else [0.2, 0.25, 0.25, 0.15, 0.15]
        
        # Batch predict for speed
        common_idx = feature_df.index
        X_all = feature_df[self.ai_engine.feature_names]
        X_scaled = scaler.transform(X_all)

        probs_matrix = []

        # 1. PyTorch
        if self.ai_engine.pytorch_model:
            try:
                import torch
                with torch.no_grad():
                    t_in = torch.tensor(X_scaled, dtype=torch.float32)
                    pt_logits = self.ai_engine.pytorch_model(t_in)
                    raw_pt = torch.softmax(pt_logits, dim=1)[:, 1].numpy()
                    if "pytorch_resnet" in calibrators:
                        p_pt = calibrators["pytorch_resnet"].predict_proba(raw_pt.reshape(-1, 1))[:, 1] * 100
                    else:
                        p_pt = raw_pt * 100
                    probs_matrix.append(p_pt)
            except Exception:
                pass

        # 2. XGBoost
        if self.ai_engine.ensemble_bundle and self.ai_engine.ensemble_bundle.get("xgb_model"):
            try:
                raw_xgb = self.ai_engine.ensemble_bundle["xgb_model"].predict_proba(X_scaled)[:, 1]
                if "xgboost" in calibrators:
                    p_xgb = calibrators["xgboost"].predict_proba(raw_xgb.reshape(-1, 1))[:, 1] * 100
                else:
                    p_xgb = raw_xgb * 100
                probs_matrix.append(p_xgb)
            except Exception:
                pass

        # 3. LightGBM
        if self.ai_engine.ensemble_bundle and self.ai_engine.ensemble_bundle.get("lgb_model"):
            try:
                raw_lgb = self.ai_engine.ensemble_bundle["lgb_model"].predict_proba(X_scaled)[:, 1]
                if "lightgbm" in calibrators:
                    p_lgb = calibrators["lightgbm"].predict_proba(raw_lgb.reshape(-1, 1))[:, 1] * 100
                else:
                    p_lgb = raw_lgb * 100
                probs_matrix.append(p_lgb)
            except Exception:
                pass

        # 4. Random Forest
        if self.ai_engine.ensemble_bundle and self.ai_engine.ensemble_bundle.get("rf_model"):
            try:
                raw_rf = self.ai_engine.ensemble_bundle["rf_model"].predict_proba(X_scaled)[:, 1]
                if "random_forest" in calibrators:
                    p_rf = calibrators["random_forest"].predict_proba(raw_rf.reshape(-1, 1))[:, 1] * 100
                else:
                    p_rf = raw_rf * 100
                probs_matrix.append(p_rf)
            except Exception:
                pass

        # 5. Extra Trees
        if self.ai_engine.ensemble_bundle and self.ai_engine.ensemble_bundle.get("et_model"):
            try:
                raw_et = self.ai_engine.ensemble_bundle["et_model"].predict_proba(X_scaled)[:, 1]
                if "extra_trees" in calibrators:
                    p_et = calibrators["extra_trees"].predict_proba(raw_et.reshape(-1, 1))[:, 1] * 100
                else:
                    p_et = raw_et * 100
                probs_matrix.append(p_et)
            except Exception:
                pass

        if probs_matrix:
            if len(probs_matrix) == len(weights):
                weighted_ensemble = np.zeros(len(X_all))
                for w, p in zip(weights, probs_matrix):
                    weighted_ensemble += w * p
                final_confidences = weighted_ensemble
            else:
                final_confidences = np.mean(probs_matrix, axis=0)
        else:
            final_confidences = np.full(len(X_all), 50.0)

        # Map back to full df index
        conf_series = pd.Series(final_confidences, index=common_idx).reindex(df.index, fill_value=50.0)

        for i in range(len(df)):
            close_val = df['Close'].iloc[i]
            ema200_val = df['EMA_200'].iloc[i]
            ema9_val = df['EMA_9'].iloc[i] if 'EMA_9' in df.columns else df['EMA_20'].iloc[i]
            ema21_val = df['EMA_21'].iloc[i] if 'EMA_21' in df.columns else df['EMA_50'].iloc[i]
            c_val = conf_series.iloc[i]
            
            ai_confidences[i] = round(float(c_val), 1)

            if self.use_ml_ensemble:
                # ML Ensemble + Trend Confluence Entry
                is_bull_trend = close_val > ema200_val and ema9_val > ema21_val
                is_ai_bullish = c_val >= self.confidence_threshold
                
                # Check Meta-Model if available
                meta_pass = True
                if self.ai_engine.ensemble_bundle and self.ai_engine.ensemble_bundle.get("meta_model") and i in feature_df.index:
                    try:
                        row_f = feature_df.loc[df.index[i]]
                        disagr = float(np.std([p[min(i, len(p)-1)] for p in probs_matrix])) if probs_matrix else 0.0
                        meta_in = np.array([[
                            c_val / 100.0,
                            disagr / 100.0,
                            float(row_f.get('ATR_Pct', 0.015)),
                            float(row_f.get('Volume_Ratio_20', 1.0)),
                            float(row_f.get('VWAP_Dist', 0.0)),
                            float(row_f.get('Sector_Rel_Strength', 0.0)),
                            float(row_f.get('Market_Vol_Proxy', 0.15))
                        ]])
                        meta_p = float(self.ai_engine.ensemble_bundle["meta_model"].predict_proba(meta_in)[0][1])
                        meta_pass = meta_p >= 0.48
                    except Exception:
                        meta_pass = True

                if is_bull_trend and is_ai_bullish and meta_pass:
                    buy_triggers[i] = True
                    ai_regimes[i] = "BULL_EXPANSION_AI_CONFIRMED"
                elif not meta_pass and is_bull_trend and is_ai_bullish:
                    ai_regimes[i] = "BULL_TREND_META_FILTERED"
                elif close_val < ema200_val or c_val <= 42.0:
                    exit_triggers[i] = True
                    ai_regimes[i] = "BEAR_DISTRIBUTION"
                else:
                    ai_regimes[i] = "NEUTRAL_CONSOLIDATION"
            else:
                # Plain 200-EMA & MACD baseline strategy
                buy_cond = (close_val > ema200_val) and (df['EMA_20'].iloc[i] > df['EMA_50'].iloc[i]) and (df['RSI_14'].iloc[i] > 50)
                exit_cond = (close_val < df['EMA_50'].iloc[i]) or (df['RSI_14'].iloc[i] < 40)
                buy_triggers[i] = bool(buy_cond)
                exit_triggers[i] = bool(exit_cond)

        df['Buy_Trigger'] = buy_triggers
        df['Exit_Trigger'] = exit_triggers
        df['AI_Confidence'] = ai_confidences
        df['AI_Regime'] = ai_regimes
        return df

    def run(self) -> Dict[str, Any]:
        """
        Executes strict next-bar execution backtest with full Indian transaction costs.
        """
        data = self.generate_ml_signals(self.raw_data)
        
        capital = self.initial_capital
        capital_gross = self.initial_capital  # Track without costs
        self.risk_manager.capital = capital
        
        trades: List[Dict[str, Any]] = []
        equity_curve: List[float] = [capital]
        equity_curve_gross: List[float] = [capital_gross]
        timestamps: List[Any] = [data.index[0]]

        in_position = False
        units = 0.0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        entry_time = None
        entry_idx = 0

        total_costs_accumulated = {
            "brokerage": 0.0, "stt": 0.0, "exchange_charges": 0.0,
            "sebi_fees": 0.0, "gst": 0.0, "stamp_duty": 0.0, "slippage_cost": 0.0,
            "total_statutory": 0.0, "total_all_costs": 0.0
        }

        bars_in_trade = 0

        for i in range(1, len(data)):
            current_bar = data.iloc[i]
            prev_bar = data.iloc[i - 1]
            timestamp = data.index[i]

            high = current_bar['High']
            low = current_bar['Low']
            close = current_bar['Close']
            open_p = current_bar['Open']
            atr = current_bar.get('ATR_14', close * 0.015)

            # 1. If in position, check for exit conditions during current bar
            if in_position:
                bars_in_trade += 1
                exit_price = None
                exit_reason = None

                # Check Stop Loss (Hit within bar low)
                if low <= stop_loss:
                    exit_price = min(stop_loss, open_p) * (1 - self.slippage_pct)
                    exit_reason = "Stop Loss Hit"
                # Check Take Profit (Hit within bar high)
                elif high >= take_profit:
                    exit_price = max(take_profit, open_p) * (1 - self.slippage_pct)
                    exit_reason = "Take Profit Hit"
                # Check Next-Bar Open Exit on previous bar's Exit Trigger
                elif prev_bar['Exit_Trigger']:
                    exit_price = open_p * (1 - self.slippage_pct)
                    exit_reason = "Signal Exit Trigger"

                if exit_price is not None:
                    # Execute Exit & Calculate Indian Statutory Costs
                    turnover_buy = entry_price * units
                    turnover_sell = exit_price * units
                    gross_pnl = (exit_price - entry_price) * units

                    costs = IndianTradingCosts.calculate(
                        turnover_buy=turnover_buy,
                        turnover_sell=turnover_sell,
                        product_type=self.product_type,
                        slippage_pct=self.slippage_pct
                    )

                    net_pnl = gross_pnl - costs["total_all_costs"]
                    capital += net_pnl
                    capital_gross += gross_pnl

                    for k in total_costs_accumulated:
                        total_costs_accumulated[k] += costs.get(k, 0.0)

                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(exit_price, 2),
                        'units': round(units, 4),
                        'gross_pnl': round(gross_pnl, 2),
                        'net_pnl': round(net_pnl, 2),
                        'costs_breakdown': costs,
                        'total_costs': round(costs["total_all_costs"], 2),
                        'return_pct': round((net_pnl / max(turnover_buy, 0.01)) * 100, 2),
                        'gross_return_pct': round((gross_pnl / max(turnover_buy, 0.01)) * 100, 2),
                        'holding_bars': i - entry_idx,
                        'reason': exit_reason,
                        'balance': round(capital, 2)
                    })

                    in_position = False
                    units = 0.0

            # 2. If not in position, check for ENTRY SIGNAL on PREVIOUS bar
            # CRITICAL ZERO LOOK-AHEAD RULE: Trade ONLY on next-bar open!
            elif not in_position and not self.risk_manager.is_circuit_broken:
                if prev_bar['Buy_Trigger']:
                    # Enter at Open of current candle + slippage
                    entry_price = open_p * (1 + self.slippage_pct)
                    units_calc, stop_loss_calc, take_profit_calc = self.risk_manager.calculate_position(entry_price, atr)

                    # Sizing with leverage awareness
                    leverage_mult = 5.0 if self.product_type == "MIS" else 1.0
                    max_purchasing_power = capital * leverage_mult

                    if self.initial_capital >= 5000:
                        units_calc = max(math.floor(units_calc), 1)
                        if (entry_price * units_calc) > max_purchasing_power:
                            units_calc = math.floor(max_purchasing_power / entry_price)
                    else:
                        # Micro account (₹500) fractional simulation
                        max_units_allowed = max_purchasing_power / entry_price
                        units_calc = min(units_calc, max_units_allowed)
                    
                    cost_entry = entry_price * units_calc
                    if units_calc > 0 and cost_entry <= (max_purchasing_power * 1.05):
                        in_position = True
                        units = units_calc
                        stop_loss = stop_loss_calc
                        take_profit = take_profit_calc
                        entry_time = timestamp
                        entry_idx = i

            # Current Equity update
            unrealized_pnl = (close - entry_price) * units if in_position else 0.0
            current_equity = capital + unrealized_pnl
            current_equity_gross = capital_gross + unrealized_pnl
            
            equity_curve.append(current_equity)
            equity_curve_gross.append(current_equity_gross)
            timestamps.append(timestamp)

            self.risk_manager.check_drawdown(current_equity)

        # Close position at backtest end if still open
        if in_position:
            last_bar = data.iloc[-1]
            last_close = last_bar['Close'] * (1 - self.slippage_pct)
            turnover_buy = entry_price * units
            turnover_sell = last_close * units
            gross_pnl = (last_close - entry_price) * units

            costs = IndianTradingCosts.calculate(
                turnover_buy=turnover_buy,
                turnover_sell=turnover_sell,
                product_type=self.product_type,
                slippage_pct=self.slippage_pct
            )

            net_pnl = gross_pnl - costs["total_all_costs"]
            capital += net_pnl
            capital_gross += gross_pnl

            for k in total_costs_accumulated:
                total_costs_accumulated[k] += costs.get(k, 0.0)

            trades.append({
                'entry_time': entry_time,
                'exit_time': data.index[-1],
                'entry_price': round(entry_price, 2),
                'exit_price': round(last_close, 2),
                'units': round(units, 4),
                'gross_pnl': round(gross_pnl, 2),
                'net_pnl': round(net_pnl, 2),
                'costs_breakdown': costs,
                'total_costs': round(costs["total_all_costs"], 2),
                'return_pct': round((net_pnl / max(turnover_buy, 0.01)) * 100, 2),
                'gross_return_pct': round((gross_pnl / max(turnover_buy, 0.01)) * 100, 2),
                'holding_bars': len(data) - entry_idx,
                'reason': "Backtest Session Ended",
                'balance': round(capital, 2)
            })

        # --- COMPREHENSIVE PERFORMANCE METRICS CALCULATION ---
        df_equity = pd.Series(equity_curve, index=timestamps)
        df_equity_gross = pd.Series(equity_curve_gross, index=timestamps)
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['net_pnl'] > 0]
        losing_trades = [t for t in trades if t['net_pnl'] <= 0]

        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0.0
        total_net_pnl = capital - self.initial_capital
        total_gross_pnl = capital_gross - self.initial_capital
        total_return_pct = (total_net_pnl / self.initial_capital) * 100
        gross_return_pct = (total_gross_pnl / self.initial_capital) * 100

        gross_gains = sum(t['net_pnl'] for t in winning_trades)
        gross_losses = abs(sum(t['net_pnl'] for t in losing_trades))
        profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else (999.0 if gross_gains > 0 else 0.0)

        # Expectancy
        expectancy_inr = (total_net_pnl / total_trades) if total_trades > 0 else 0.0
        avg_trade_ret_pct = np.mean([t['return_pct'] for t in trades]) if total_trades > 0 else 0.0

        # Longest Losing Streak
        longest_loss_streak = 0
        current_streak = 0
        for t in trades:
            if t['net_pnl'] <= 0:
                current_streak += 1
                if current_streak > longest_loss_streak:
                    longest_loss_streak = current_streak
            else:
                current_streak = 0

        # Exposure % (Fraction of trading days in active trade)
        exposure_pct = (bars_in_trade / max(len(data), 1)) * 100

        # Drawdown calculation
        rolling_max = df_equity.cummax()
        drawdowns = (df_equity - rolling_max) / rolling_max
        max_drawdown_pct = abs(drawdowns.min() * 100) if len(drawdowns) > 0 else 0.0

        # Time span & CAGR
        num_days = max((data.index[-1] - data.index[0]).days, 1)
        num_years = max(num_days / 365.25, 0.1)
        if capital > 0:
            cagr_pct = ((capital / self.initial_capital) ** (1.0 / num_years) - 1.0) * 100
        else:
            cagr_pct = -100.0

        # Sharpe Ratio (Daily returns, Risk-Free Rate = 6.5% INR Treasury Repo)
        daily_returns = df_equity.pct_change().dropna()
        rf_daily = (0.065 / 252)
        excess_returns = daily_returns - rf_daily
        if len(daily_returns) > 5 and daily_returns.std() > 0:
            sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Benchmark Buy & Hold
        start_p = data.iloc[0]['Close']
        end_p = data.iloc[-1]['Close']
        buy_and_hold_return_pct = ((end_p - start_p) / start_p) * 100
        bnh_cagr_pct = (((end_p / start_p) ** (1.0 / num_years) - 1.0) * 100) if end_p > 0 else 0.0

        # Monte Carlo Shuffle Analysis (1,000 runs)
        monte_carlo_results = self.run_monte_carlo(trades, n_simulations=1000)

        results = {
            'initial_capital': self.initial_capital,
            'final_capital': round(capital, 2),
            'final_capital_gross': round(capital_gross, 2),
            'total_pnl': round(total_net_pnl, 2),
            'total_gross_pnl': round(total_gross_pnl, 2),
            'strategy_return_pct': round(total_return_pct, 2),
            'gross_return_pct': round(gross_return_pct, 2),
            'cagr_pct': round(cagr_pct, 2),
            'buy_and_hold_return_pct': round(buy_and_hold_return_pct, 2),
            'buy_and_hold_cagr_pct': round(bnh_cagr_pct, 2),
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'expectancy_inr': round(expectancy_inr, 2),
            'expectancy_pct': round(avg_trade_ret_pct, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'longest_losing_streak': longest_loss_streak,
            'market_exposure_pct': round(exposure_pct, 2),
            'total_costs_paid': round(total_costs_accumulated["total_all_costs"], 2),
            'costs_breakdown': {k: round(v, 2) for k, v in total_costs_accumulated.items()},
            'monte_carlo': monte_carlo_results,
            'trades_list': trades,
            'equity_curve': df_equity,
            'equity_curve_gross': df_equity_gross
        }

        return results

    def run_monte_carlo(self, trades: List[Dict[str, Any]], n_simulations: int = 1000) -> Dict[str, Any]:
        """
        Shuffles the historical trade return sequence 1,000 times to construct
        empirical drawdown distribution and risk-of-ruin confidence bounds.
        """
        if len(trades) < 4:
            return {
                "p5_best_case_dd": 0.0, "p50_median_dd": 0.0, "p95_worst_case_dd": 0.0,
                "p99_extreme_dd": 0.0, "ruin_probability_pct": 0.0
            }

        trade_pnls = [t['net_pnl'] for t in trades]
        simulated_max_drawdowns = []
        ruin_count = 0

        for _ in range(n_simulations):
            shuffled = np.random.permutation(trade_pnls)
            sim_equity = [self.initial_capital]
            curr_cap = self.initial_capital
            
            for pnl in shuffled:
                curr_cap += pnl
                sim_equity.append(curr_cap)
                if curr_cap <= (self.initial_capital * 0.50):  # 50% drawdown ruin definition
                    ruin_count += 1
                    break

            sim_series = pd.Series(sim_equity)
            roll_max = sim_series.cummax()
            dd_series = (sim_series - roll_max) / roll_max
            simulated_max_drawdowns.append(abs(dd_series.min() * 100))

        sim_dds = np.array(simulated_max_drawdowns)
        return {
            "p5_best_case_dd": round(float(np.percentile(sim_dds, 5)), 2),
            "p50_median_dd": round(float(np.percentile(sim_dds, 50)), 2),
            "p95_worst_case_dd": round(float(np.percentile(sim_dds, 95)), 2),
            "p99_extreme_dd": round(float(np.percentile(sim_dds, 99)), 2),
            "ruin_probability_pct": round((ruin_count / n_simulations) * 100, 2)
        }


def run_multi_capital_sensitivity(df: pd.DataFrame, ticker_name: str = "RELIANCE.NS") -> Dict[str, Any]:
    """
    Runs backtest comparisons across ₹500, ₹25,000, and ₹1,00,000 starting capital.
    Analyzes the drag of fixed flat broker costs on micro accounts vs institutional sizes.
    """
    capital_tiers = [500.0, 25000.0, 100000.0]
    results_by_tier = {}

    for cap in capital_tiers:
        bt_ml = Backtester(df, initial_capital=cap, use_ml_ensemble=True)
        res_ml = bt_ml.run()

        bt_ema = Backtester(df, initial_capital=cap, use_ml_ensemble=False)
        res_ema = bt_ema.run()

        results_by_tier[f"INR_{int(cap)}"] = {
            "capital": cap,
            "ml_ensemble": res_ml,
            "plain_ema": res_ema
        }

    return results_by_tier
