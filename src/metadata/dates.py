"""
Date normalisation with the standard library only.

`normalize_date` accepts the shapes found in EnterpriseRAG-Bench (RFC 2822, ISO 8601 incl. "Z",
Gmail-UI "Tue, Jun 3, 2025 at 9:12 AM PT", "2025-03-27", "March 27, 2025", "27 March 2025",
"03/27/2025", optionally followed by "| 60 min" style suffixes) and returns an ISO 8601 string:
"YYYY-MM-DD" when only the day is known, "YYYY-MM-DDTHH:MM:SSZ" (UTC) when a timezone is known,
"YYYY-MM-DDTHH:MM:SS" when the time is naive. All outputs sort lexicographically.
"""
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional, Tuple

TZ_OFFSETS_HOURS = {
    "UTC": 0, "GMT": 0, "Z": 0,
    "PST": -8, "PDT": -7, "PT": -7,
    "MST": -7, "MDT": -6, "MT": -6,
    "CST": -6, "CDT": -5, "CT": -5,
    "EST": -5, "EDT": -4, "ET": -4,
    "BST": 1, "CET": 1, "CEST": 2, "IST": 5.5, "SGT": 8, "JST": 9, "AEST": 10,
}

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
MONTH_NAMES = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

MIN_YEAR, MAX_YEAR = 1990, 2100

_ISO_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})"
    r"(?:[T ](?P<H>\d{2}):(?P<M>\d{2})(?::(?P<S>\d{2})(?:\.\d+)?)?)?"
    r"\s*(?P<tz>Z|[+-]\d{2}:?\d{2}|[A-Z]{2,4})?$"
)
_GMAIL_UI_RE = re.compile(
    rf"^(?:[a-z]{{3,9}},?\s+)?(?P<mon>{MONTH_NAMES})\.?\s+(?P<d>\d{{1,2}}),?\s+(?P<y>\d{{4}})"
    r"(?:\s+(?:at\s+)?(?P<H>\d{1,2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?\s*(?P<ampm>[ap]m)?)?"
    r"\s*(?P<tz>[A-Z]{2,4})?$",
    re.IGNORECASE,
)
_DMY_RE = re.compile(
    rf"^(?P<d>\d{{1,2}})\s+(?P<mon>{MONTH_NAMES})\.?,?\s+(?P<y>\d{{4}})"
    r"(?:\s+(?P<H>\d{1,2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?\s*(?P<tz>[A-Z]{2,4})?)?$",
    re.IGNORECASE,
)
_MDY_RE = re.compile(r"^(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{4})$")
_TIME_RE = re.compile(
    r"^(?P<H>\d{1,2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?\s*(?P<ampm>[AaPp][Mm])?\s*(?P<tz>Z|[+-]\d{2}:?\d{2}|[A-Z]{2,4})?$"
)

TZ_ABBR = r"(?:UTC|GMT|PST|PDT|PT|MST|MDT|MT|CST|CDT|CT|EST|EDT|ET|BST|CET|CEST|IST|SGT|JST|AEST)"
_FIND_RE = re.compile(
    r"(?<![\w.-])(?:"
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2}|\s" + TZ_ABBR + r"\b)?)?"
    rf"|(?:[A-Za-z]{{3,9}},?\s+)?{MONTH_NAMES}\.?\s+\d{{1,2}},?\s+\d{{4}}"
    rf"|\d{{1,2}}\s+{MONTH_NAMES}\.?,?\s+\d{{4}}"
    r"|\d{1,2}/\d{1,2}/\d{4}"
    r")(?![\w-])",
    re.IGNORECASE,
)


US_GENERIC_ZONES = {"PT": (-8, -7), "MT": (-7, -6), "CT": (-6, -5), "ET": (-5, -4)}


def _us_dst(y: int, m: int, d: int) -> bool:
    """US daylight saving time: second Sunday of March .. first Sunday of November."""
    if m < 3 or m > 11:
        return False
    if 3 < m < 11:
        return True
    first = datetime(y, m, 1).weekday()  # Monday=0
    first_sunday = 1 + (6 - first) % 7
    if m == 3:
        return d >= first_sunday + 7
    return d < first_sunday


def _tz_from_token(token: Optional[str], y: Optional[int] = None, m: Optional[int] = None,
                   d: Optional[int] = None) -> Optional[timezone]:
    if not token:
        return None
    token = token.strip()
    if token in ("Z", "z"):
        return timezone.utc
    mo = re.match(r"^([+-])(\d{2}):?(\d{2})$", token)
    if mo:
        sign = 1 if mo.group(1) == "+" else -1
        return timezone(sign * timedelta(hours=int(mo.group(2)), minutes=int(mo.group(3))))
    upper = token.upper()
    if upper in US_GENERIC_ZONES and y is not None:
        standard, daylight = US_GENERIC_ZONES[upper]
        return timezone(timedelta(hours=daylight if _us_dst(y, m, d) else standard))
    hours = TZ_OFFSETS_HOURS.get(upper)
    if hours is None:
        return None
    return timezone(timedelta(hours=hours))


def _format(dt: datetime, has_time: bool) -> str:
    if not has_time:
        return dt.strftime("%Y-%m-%d")
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _build(y: int, m: int, d: int, H=None, M=None, S=None, ampm=None, tz=None) -> Optional[str]:
    if not (MIN_YEAR <= y <= MAX_YEAR):
        return None
    try:
        if H is None:
            return _format(datetime(y, m, d), has_time=False)
        hour = int(H)
        if ampm:
            ampm = ampm.lower()
            if ampm == "pm" and hour < 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
        dt = datetime(y, m, d, hour, int(M or 0), int(S or 0), tzinfo=_tz_from_token(tz, y, m, d))
        return _format(dt, has_time=True)
    except ValueError:
        return None


def _clean(value: str) -> str:
    value = value.strip()
    value = value.strip("[]()")
    # "2025-03-27 | Duration: 60 min" / "2026-01-14 (approx)" -> keep the first field only
    value = re.split(r"\s*[|]\s*", value, maxsplit=1)[0]
    value = re.sub(r"\s+", " ", value).strip(" .,;")
    return value


def normalize_date(value: Optional[str]) -> Optional[str]:
    """Return an ISO 8601 date/datetime string, or None when `value` is not a recognisable date."""
    if not value or not isinstance(value, str):
        return None
    value = _clean(value)
    if not value:
        return None

    m = _ISO_RE.match(value)
    if m:
        g = m.groupdict()
        return _build(int(g["y"]), int(g["m"]), int(g["d"]), g["H"], g["M"], g["S"], None, g["tz"])

    m = _MDY_RE.match(value)
    if m:
        g = m.groupdict()
        return _build(int(g["y"]), int(g["m"]), int(g["d"]))

    m = _GMAIL_UI_RE.match(value)
    if m:
        g = m.groupdict()
        return _build(int(g["y"]), MONTHS[g["mon"].lower()[:3]], int(g["d"]),
                      g["H"], g["M"], g["S"], g["ampm"], g["tz"])

    m = _DMY_RE.match(value)
    if m:
        g = m.groupdict()
        return _build(int(g["y"]), MONTHS[g["mon"].lower()[:3]], int(g["d"]),
                      g["H"], g["M"], g["S"], None, g["tz"])

    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        dt = None
    if dt is not None and MIN_YEAR <= dt.year <= MAX_YEAR:
        return _format(dt, has_time=True)
    return None


def combine_date_time(date_iso: Optional[str], time_value: Optional[str]) -> Optional[str]:
    """Attach a "10:03 AM PT" / "15:00 UTC" time to a date-only ISO string."""
    if not date_iso:
        return None
    if not time_value or "T" in date_iso:
        return date_iso
    m = _TIME_RE.match(_clean(time_value))
    if not m:
        return date_iso
    g = m.groupdict()
    y, mo, d = (int(x) for x in date_iso[:10].split("-"))
    return _build(y, mo, d, g["H"], g["M"], g["S"], g["ampm"], g["tz"]) or date_iso


def find_dates(text: str, max_n: int = 20) -> List[str]:
    """Scan prose for dates; returns sorted, unique ISO strings (at most `max_n`)."""
    if not text:
        return []
    found = set()
    for m in _FIND_RE.finditer(text):
        iso = normalize_date(m.group(0))
        if iso:
            found.add(iso)
            if len(found) >= 4 * max_n:
                break
    return sorted(found)[:max_n]


def date_bounds(dates: List[str]) -> Tuple[Optional[str], Optional[str]]:
    dates = [d for d in dates if d]
    if not dates:
        return None, None
    return min(dates), max(dates)
