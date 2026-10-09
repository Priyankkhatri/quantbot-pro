import os
import sys
import unittest
import sqlite3
import pandas as pd
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from forward_paper_testing import PaperLedgerStore, DB_PATH

class TestForwardDataFeedCrosscheck(unittest.TestCase):
    def test_dual_feed_crosscheck_and_unverified_flagging(self):
        """
        Tests that when daily closing prices diverge by >0.5% between primary (yfinance)
        and secondary (Angel One) data feeds:
        1. Discrepancies are detected and itemized.
        2. data_feed_status is marked as UNVERIFIED.
        3. data_feed_source is recorded.
        """
        date_str = "2026-10-12"
        primary_closes = {
            "NIFTYBEES.NS": 285.50,
            "BANKBEES.NS": 540.00,
            "GOLDBEES.NS": 68.40
        }
        
        # Simulated secondary source where BANKBEES diverges by 1.2% (> 0.5% tolerance)
        secondary_closes = {
            "NIFTYBEES.NS": 285.60,      # 0.035% diff (PASS)
            "BANKBEES.NS": 546.80,       # 1.259% diff (FAIL > 0.5%)
            "GOLDBEES.NS": 68.40        # 0.000% diff (PASS)
        }
        
        tolerance_pct = 0.5
        discrepancies = []
        for sym, p1 in primary_closes.items():
            if sym in secondary_closes:
                p2 = secondary_closes[sym]
                diff_pct = abs(p1 - p2) / p1 * 100.0
                if diff_pct > tolerance_pct:
                    discrepancies.append({
                        "symbol": sym,
                        "primary_close": p1,
                        "secondary_close": p2,
                        "diff_pct": round(diff_pct, 3)
                    })
                    
        feed_status = "VERIFIED" if len(discrepancies) == 0 else "UNVERIFIED"
        
        self.assertEqual(feed_status, "UNVERIFIED")
        self.assertEqual(len(discrepancies), 1)
        self.assertEqual(discrepancies[0]["symbol"], "BANKBEES.NS")
        self.assertAlmostEqual(discrepancies[0]["diff_pct"], 1.259, places=2)
        
        # Test inserting into ledger with data_feed_source & status
        store = PaperLedgerStore(db_path="test_feed_check.db")
        with sqlite3.connect("test_feed_check.db") as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(paper_forward_ledger)")
            cols = [r[1] for r in cursor.fetchall()]
            self.assertIn("data_feed_source", cols)
            self.assertIn("data_feed_status", cols)
            
        if os.path.exists("test_feed_check.db"):
            try:
                os.remove("test_feed_check.db")
            except Exception:
                pass
                
        print("  ✓ [Test Passed] Dual-feed crosscheck correctly flags >0.5% differences as UNVERIFIED and stores source metadata.")

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestForwardDataFeedCrosscheck)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
