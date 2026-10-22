"""Date and Time context utility providing local time and date awareness online and offline."""

from datetime import datetime


def get_current_datetime_info() -> dict:
    """Return structured date, time, year, and timezone info for the local machine."""
    now = datetime.now().astimezone()
    return {
        "datetime_iso": now.isoformat(),
        "year": now.year,
        "month": now.strftime("%B"),
        "day": now.day,
        "weekday": now.strftime("%A"),
        "time_str": now.strftime("%I:%M:%S %p"),
        "timezone": now.tzname() or f"UTC{now.strftime('%z')}",
        "full_human_readable": now.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z"),
    }


def format_datetime_system_prompt() -> str:
    """Produce a concise, authoritative date/time header for LLM system prompts."""
    info = get_current_datetime_info()
    return (
        f"[CURRENT TEMPORAL CONTEXT]\n"
        f"- Current Date: {info['weekday']}, {info['month']} {info['day']}, {info['year']}\n"
        f"- Current Time: {info['time_str']} ({info['timezone']})\n"
        f"- Current Year: {info['year']}\n"
        f"- System Status: Operates offline and online with verified local system clock.\n"
        f"[END TEMPORAL CONTEXT]"
    )
