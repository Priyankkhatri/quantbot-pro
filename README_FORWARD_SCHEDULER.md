# QuantBot Pro: Reliable Post-Market Daily Job & Windows Scheduler Guide

## 1. Overview
QuantBot Pro runs an automated daily post-market forward-stepping routine at **15:45 IST (Mon–Fri)** after Indian markets close. This job:
1. Fetches official daily closing OHLCV candles from the broker/NSE.
2. Marks-to-market all forward paper portfolios at today's close.
3. Generates and queues pending orders for scheduled rebalances at $T+1$ Open.
4. Appends tamper-proof SHA-256 blocks to the forward ledger (`forward_paper_testing.db`).
5. Dispatches Telegram summaries.

---

## 2. Setting Up Windows Task Scheduler

To ensure the daily job runs automatically even if the main UI is closed:

### Method A: One-Line PowerShell Command (Run as Administrator)
```powershell
$Action = New-ScheduledTaskAction -Execute "python.exe" -Argument "forward_paper_testing.py --post-market" -WorkingDirectory "C:\Users\priya\Desktop\QuantTradingBot"
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 3:45PM
Register-ScheduledTask -TaskName "QuantBot_PostMarket_DailyJob" -Action $Action -Trigger $Trigger -Description "QuantBot Pro Post-Market Forward Paper Step" -User "$env:USERNAME"
```

### Method B: GUI Steps via `taskschd.msc`
1. Press `Win + R`, type `taskschd.msc`, and hit **Enter**.
2. Click **Create Basic Task...** in the right Actions pane.
3. **Name:** `QuantBot_PostMarket_DailyJob`
4. **Trigger:** Select **Weekly**, check **Monday through Friday**, set Time to `15:45:00`.
5. **Action:** Select **Start a program**.
   - **Program/script:** `C:\Users\priya\AppData\Local\Programs\Python\Python314\python.exe` (or your Python PATH)
   - **Add arguments:** `forward_paper_testing.py --post-market`
   - **Start in:** `C:\Users\priya\Desktop\QuantTradingBot`
6. Click **Finish**.

---

## 3. Automated Catch-Up Engine & Startup Safety

If the system was offline during market close or forward days were missed:
- On startup, `ForwardPaperManager.run_startup_catchup_job()` inspects the ledger's latest `trade_date`.
- It processes each missing date chronologically using **only historical data available as of that bar**.
- Each processed row is tagged with `source='CATCHUP'`.
- **Integrity Alert:** If more than 3 consecutive trading days are missed, the system logs a high-priority warning and notifies Telegram (`⚠️ FORWARD RECONCILIATION GAP ALERT: Missed >3 trading days`).
