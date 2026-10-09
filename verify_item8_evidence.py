import os
import sys
import json
import sqlite3
import shutil
import tempfile
import datetime

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from forward_paper_testing import ForwardPaperManager
import backup_forward_paper_ledger
import seed_forward_paper_history

def verify_item8():
    print("=" * 95)
    print("   QUANTBOT PRO: ITEM 8 HASH ANCHORING, BACKUP & SEED ISOLATION VERIFICATION")
    print("=" * 95)
    
    # 1. Weekly Telegram Dry-Run Payload
    print("\n1. TELEGRAM WEEKLY SUMMARY PAYLOAD (INCLUDING CHAIN HEAD & SEQUENCE NUMBERS):")
    print("-" * 95)
    mgr = ForwardPaperManager(db_path="forward_paper_testing.db")
    telegram_msg = mgr.generate_weekly_telegram_summary("STANDARD_100K")
    print(telegram_msg)
    
    # 2. Run backup_forward_paper_ledger.py
    print("\n2. RUNNING BACKUP ENGINE (backup_forward_paper_ledger.py):")
    print("-" * 95)
    backup_file = backup_forward_paper_ledger.backup_ledger("backups")
    if backup_file and os.path.exists(backup_file):
        size_bytes = os.path.getsize(backup_file)
        print(f"  [+] Backup Verified Successfully!")
        print(f"  • Created File : {backup_file}")
        print(f"  • Size         : {size_bytes:,} bytes")
        print(f"  • Timestamp    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    else:
        print("  [-] Backup file creation failed.")
        
    # 3. Temp DB insert FORWARD row and test seed abort
    print("\n3. SEED ISOLATION PROTECTION TEST (INDEPENDENT TEMP DB):")
    print("-" * 95)
    temp_dir = tempfile.mkdtemp()
    temp_db = os.path.join(temp_dir, "test_abort_ledger.db")
    try:
        # Create table and insert a FORWARD row
        with sqlite3.connect(temp_db) as conn:
            conn.execute("""
                CREATE TABLE paper_forward_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    capital_tier TEXT NOT NULL,
                    sequence_num INTEGER NOT NULL,
                    trade_date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    block_hash TEXT NOT NULL
                )
            """)
            conn.execute("""
                INSERT INTO paper_forward_ledger 
                (strategy_id, version, capital_tier, sequence_num, trade_date, source, block_hash)
                VALUES ('NIFTY50_BUY_HOLD', 'v1.0.0', 'STANDARD_100K', 1, '2026-10-12', 'FORWARD', 'HASH_FORWARD_TEST_0001')
            """)
            conn.commit()
            
        print(f"  • Created temporary test ledger with 1 existing 'FORWARD' row: {temp_db}")
        print("  • Attempting to run seed_forward_paper_history on this database...")
        
        # Test abort logic
        seed_attempt_failed = False
        try:
            seed_forward_paper_history.seed_history(db_path=temp_db)
        except Exception as e:
            seed_attempt_failed = True
            print(f"\n  ✓ [EXPECTED ABORT TRIGGERED] Refused to seed: {e}")
            
        if not seed_attempt_failed:
            print("  [-] Error: Seed did not abort on existing FORWARD rows!")
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    print("=" * 95)

if __name__ == "__main__":
    verify_item8()
