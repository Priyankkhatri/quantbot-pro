import datetime
import sys
from market_calendar import is_market_day

def get_next_trading_day(cur_date=None):
    if cur_date is None:
        cur_date = datetime.date(2026, 10, 9)
    d = cur_date + datetime.timedelta(days=1)
    while True:
        is_valid, reason = is_market_day(d)
        print(f"Checking {d.isoformat()} ({d.strftime('%A')}): is_market_day={is_valid} [{reason}]")
        if is_valid:
            return d
        d += datetime.timedelta(days=1)

if __name__ == "__main__":
    today = datetime.date(2026, 10, 9)
    next_day = get_next_trading_day(today)
    print(f"\n==================================================================")
    print(f"  NSE CALENDAR LOOKUP RESULT")
    print(f"  Current Date (Today)    : {today.isoformat()} ({today.strftime('%A')})")
    print(f"  Next NSE Trading Day    : {next_day.isoformat()} ({next_day.strftime('%A')})")
    print(f"==================================================================")
