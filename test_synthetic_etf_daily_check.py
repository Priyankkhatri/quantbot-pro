"""
Synthetic Unit Tests for ETF Daily Validation Check
===================================================
Tests required cases:
1. NIFTYBEES -3% vs ^NSEI -6% -> Flag UNVERIFIED (gap = 3.0% > 1.0%)
2. GOLDBEES +1.5% vs ^NSEI +0.5% -> Must NOT flag (Gold not mapped to equity indices)
3. 0.00% ETF day with NIFTY +1.0% -> Flag UNVERIFIED (stale ETF price on moving index)
4. Normal tracking (NiftyBees +0.48% vs Nifty +0.50%) -> VERIFIED (gap = 0.02% <= 1.0%)
5. Handles '.NS' suffix gracefully
"""

import sys
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from forward_paper_testing import ForwardPaperEngine

def test_etf_synthetic_validation():
    print("=" * 80)
    print("   TESTING SYNTHETIC ETF DAILY VALIDATION CHECK")
    print("=" * 80)

    t_date = pd.Timestamp("2026-10-09")

    # Case 1: NiftyBees -3% vs Nifty -6% -> Must Flag UNVERIFIED (gap = 3%)
    prev_1 = {"NIFTYBEES.NS": 100.0, "^NSEI": 10000.0}
    today_1 = {"NIFTYBEES.NS": 97.0, "^NSEI": 9400.0} # -3% vs -6%
    status_1, msgs_1 = ForwardPaperEngine.validate_daily_etf_data(t_date, today_1, prev_1)
    print(f"[Test 1] NiftyBees -3% vs Nifty -6%: status={status_1}, msgs={msgs_1}")
    assert status_1 == "UNVERIFIED", f"Expected UNVERIFIED, got {status_1}"
    assert len(msgs_1) > 0 and "DIVERGENCE" in msgs_1[0]
    print("  ✓ Test 1 Passed: NiftyBees -3% vs Nifty -6% correctly flagged.")

    # Case 2: Gold +1.5% vs Nifty +0.5% -> Must NOT Flag (VERIFIED)
    prev_2 = {"GOLDBEES.NS": 100.0, "^NSEI": 10000.0}
    today_2 = {"GOLDBEES.NS": 101.5, "^NSEI": 10050.0} # +1.5% vs +0.5%
    status_2, msgs_2 = ForwardPaperEngine.validate_daily_etf_data(t_date, today_2, prev_2)
    print(f"[Test 2] Gold +1.5% vs Nifty +0.5%: status={status_2}, msgs={msgs_2}")
    assert status_2 == "VERIFIED", f"Expected VERIFIED for unmapped Gold, got {status_2}"
    assert len(msgs_2) == 0
    print("  ✓ Test 2 Passed: Gold divergence ignored (not mapped to equity index).")

    # Case 3: 0.00% ETF Day with Nifty +1.0% -> Must Flag UNVERIFIED (stale price)
    prev_3 = {"NIFTYBEES": 100.0, "^NSEI": 10000.0}
    today_3 = {"NIFTYBEES": 100.0, "^NSEI": 10100.0} # 0.0% vs +1.0%
    status_3, msgs_3 = ForwardPaperEngine.validate_daily_etf_data(t_date, today_3, prev_3)
    print(f"[Test 3] 0.00% ETF day with Nifty +1.0%: status={status_3}, msgs={msgs_3}")
    assert status_3 == "UNVERIFIED", f"Expected UNVERIFIED for 0% ETF day, got {status_3}"
    assert len(msgs_3) > 0 and "STALE PRICE" in msgs_3[0]
    print("  ✓ Test 3 Passed: 0.00% ETF return with moving index correctly flagged.")

    # Case 4: Normal tracking (NiftyBees +0.48% vs Nifty +0.50%) -> VERIFIED
    prev_4 = {"NIFTYBEES": 100.0, "^NSEI": 10000.0}
    today_4 = {"NIFTYBEES": 100.48, "^NSEI": 10050.0} # +0.48% vs +0.50%
    status_4, msgs_4 = ForwardPaperEngine.validate_daily_etf_data(t_date, today_4, prev_4)
    print(f"[Test 4] Normal Tracking (+0.48% vs +0.50%): status={status_4}, msgs={msgs_4}")
    assert status_4 == "VERIFIED", f"Expected VERIFIED for normal tracking, got {status_4}"
    print("  ✓ Test 4 Passed: Normal tracking accepted as VERIFIED.")

    # Case 5: BankBees -2.5% vs Bank Nifty -0.5% -> Must Flag UNVERIFIED (gap = 2.0%)
    prev_5 = {"BANKBEES.NS": 400.0, "^NSEBANK": 40000.0}
    today_5 = {"BANKBEES.NS": 390.0, "^NSEBANK": 39800.0} # -2.5% vs -0.5%
    status_5, msgs_5 = ForwardPaperEngine.validate_daily_etf_data(t_date, today_5, prev_5)
    print(f"[Test 5] BankBees -2.5% vs Bank Nifty -0.5%: status={status_5}, msgs={msgs_5}")
    assert status_5 == "UNVERIFIED", f"Expected UNVERIFIED, got {status_5}"
    print("  ✓ Test 5 Passed: BankBees divergence correctly flagged.")

    print("\n🎉 ALL 5 SYNTHETIC ETF VALIDATION TESTS PASSED!")

if __name__ == "__main__":
    test_etf_synthetic_validation()
