"""
QuantBot Pro: Forward Paper Testing Ledger Backup & Integrity Anchor
====================================================================
Safely creates timestamped WAL-checkpointed backups of forward_paper_testing.db.
Verifies cryptographic SHA-256 hash chains on the backup copy before confirming.
"""

import os
import sys
import shutil
import sqlite3
import hashlib
import datetime

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from forward_paper_testing import PaperLedgerStore, DB_PATH

def backup_ledger(backup_dir: str = "backups"):
    os.makedirs(backup_dir, exist_ok=True)
    
    if not os.path.exists(DB_PATH):
        print(f"[-] Database {DB_PATH} does not exist. Nothing to backup.")
        return None

    # Step 1: Checkpoint SQLite WAL mode
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA wal_checkpoint(FULL);")

    # Step 2: Create timestamped backup file
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"forward_paper_testing_backup_{ts_str}.db")
    shutil.copy2(DB_PATH, backup_file)

    # Step 3: Compute SHA-256 file checksum
    hasher = hashlib.sha256()
    with open(backup_file, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    file_sha256 = hasher.hexdigest()

    # Step 4: Verify hash chains in backup copy
    backup_store = PaperLedgerStore(db_path=backup_file)
    is_valid, msg, count = backup_store.verify_hash_chain()

    print("=" * 80)
    print("   QUANTBOT PRO: FORWARD LEDGER BACKUP COMPLETE")
    print("=" * 80)
    print(f"Source Database     : {DB_PATH}")
    print(f"Backup Location     : {backup_file}")
    print(f"Backup File SHA-256 : {file_sha256}")
    print(f"Hash Chain Verified : {'✓ VALID' if is_valid else '❌ CORRUPT'} ({count} blocks)")
    print(f"Status Message      : {msg}")
    print("=" * 80)

    return backup_file

if __name__ == "__main__":
    backup_ledger()
