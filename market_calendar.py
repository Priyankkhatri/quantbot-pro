"""
NSE Market Hours & Holiday Calendar Module (IST Timings)
Enforces:
- Skipping Weekends (Saturday, Sunday) and Official NSE Holidays
- 09:15 - 14:45 IST: Active Entry Allowed
- 14:45 IST Cutoff: No New Entries Allowed
- 15:10 IST: Auto Square-off MIS Intraday Positions
- 15:30 IST: Market Closed
"""

import datetime
from typing import Tuple, Dict, Any

# Official NSE Major Trading Holidays (2026 / Standard Calendar)
NSE_HOLIDAYS_2026 = {
    "2026-01-26": "Republic Day",
    "2026-02-17": "Mahashivratri",
    "2026-03-03": "Holi",
    "2026-03-21": "Id-ul-Fitr (Ramzan Id)",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Dr. Ambedkar Jayanti",
    "2026-05-01": "Maharashtra Day",
    "2026-05-27": "Bakri Id",
    "2026-08-15": "Independence Day",
    "2026-09-04": "Janmashtami",
    "2026-10-02": "Mahatma Gandhi Jayanti",
    "2026-10-20": "Dussehra",
    "2026-11-09": "Diwali Laxmi Pujan",
    "2026-11-10": "Diwali Balipratipada",
    "2026-11-24": "Gurunanak Jayanti",
    "2026-12-25": "Christmas"
}


def get_current_ist_time() -> datetime.datetime:
    """Returns current time in Indian Standard Time (UTC+5:30)."""
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    return utc_now.astimezone(ist_tz)


def is_market_holiday(check_date=None) -> bool:
    """Checks if date is an official NSE holiday."""
    if check_date is None:
        date_str = get_current_ist_time().date().isoformat()
    elif isinstance(check_date, datetime.date):
        date_str = check_date.isoformat()
    else:
        date_str = str(check_date)
    return date_str in NSE_HOLIDAYS_2026


def is_market_day(check_date: datetime.date = None) -> Tuple[bool, str]:
    """Checks if the given date is a valid NSE trading day (not weekend and not holiday)."""
    if check_date is None:
        check_date = get_current_ist_time().date()

    # Weekend Check (5 = Saturday, 6 = Sunday)
    if check_date.weekday() >= 5:
        day_name = "Saturday" if check_date.weekday() == 5 else "Sunday"
        return False, f"Market Closed: Weekend ({day_name})"

    # Holiday Check
    date_str = check_date.isoformat()
    if date_str in NSE_HOLIDAYS_2026:
        return False, f"Market Closed: NSE Trading Holiday ({NSE_HOLIDAYS_2026[date_str]})"

    return True, "Valid NSE Trading Day"


def get_market_session_state(ist_dt: datetime.datetime = None) -> Dict[str, Any]:
    """
    Evaluates current time against NSE market sessions.
    - 09:15 - 14:45 IST: Entries Permitted
    - 14:45 - 15:10 IST: Cutoff (No New Entries)
    - 15:10 - 15:30 IST: Intraday MIS Square-off Period
    - Post 15:30 IST: Market Closed
    """
    if ist_dt is None:
        ist_dt = get_current_ist_time()

    is_day_valid, day_reason = is_market_day(ist_dt.date())
    if not is_day_valid:
        return {
            "can_enter_new_trade": False,
            "must_square_off_mis": False,
            "session_status": "CLOSED",
            "reason": day_reason,
            "ist_time": ist_dt.strftime("%H:%M:%S")
        }

    current_time = ist_dt.time()
    market_open = datetime.time(9, 15, 0)
    entry_cutoff = datetime.time(14, 45, 0)
    mis_squareoff = datetime.time(15, 10, 0)
    market_close = datetime.time(15, 30, 0)

    if current_time < market_open:
        return {
            "can_enter_new_trade": False,
            "must_square_off_mis": False,
            "session_status": "PRE_MARKET",
            "reason": "Market opens at 09:15 IST",
            "ist_time": ist_dt.strftime("%H:%M:%S")
        }
    elif market_open <= current_time < entry_cutoff:
        return {
            "can_enter_new_trade": True,
            "must_square_off_mis": False,
            "session_status": "ACTIVE_TRADING",
            "reason": "NSE Live Market Session (Entries Permitted)",
            "ist_time": ist_dt.strftime("%H:%M:%S")
        }
    elif entry_cutoff <= current_time < mis_squareoff:
        return {
            "can_enter_new_trade": False,
            "must_square_off_mis": False,
            "session_status": "ENTRY_CUTOFF",
            "reason": "Post 14:45 IST Cutoff: No New Entries Permitted",
            "ist_time": ist_dt.strftime("%H:%M:%S")
        }
    elif mis_squareoff <= current_time <= market_close:
        return {
            "can_enter_new_trade": False,
            "must_square_off_mis": True,
            "session_status": "SQUARE_OFF_WINDOW",
            "reason": "15:10 IST EOD Auto Square-off Window",
            "ist_time": ist_dt.strftime("%H:%M:%S")
        }
    else:
        return {
            "can_enter_new_trade": False,
            "must_square_off_mis": False,
            "session_status": "MARKET_CLOSED",
            "reason": "Market Closed at 15:30 IST",
            "ist_time": ist_dt.strftime("%H:%M:%S")
        }
