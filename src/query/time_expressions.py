"""
Date windows expressed in questions ("Feb 12, 2026", "Q4 2025", "early March 2026", "H1 2026",
"between 2026-01-10 and 2026-01-20", "in 2025"). Standard library only, reuses the explicit-date
scanner of src/metadata/dates.py. Relative expressions without a reference date ("last quarter",
"recently") are deliberately ignored.
"""
import calendar
import re
from datetime import date
from typing import List, Optional, Tuple

from ..metadata.dates import MONTHS, MONTH_NAMES, _FIND_RE, normalize_date

_QUARTER_RE = re.compile(
    r"(?<![\w-])(?:Q(?P<q>[1-4])\s*(?:of\s+)?(?P<y>\d{4})"
    r"|(?P<y2>\d{4})\s*[- ]?Q(?P<q2>[1-4])"
    r"|(?P<word>first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of\s+)?(?P<y3>\d{4}))(?![\w-])",
    re.IGNORECASE,
)
_HALF_RE = re.compile(r"(?<![\w-])H(?P<h>[12])\s*(?:of\s+)?(?P<y>\d{4})(?![\w-])", re.IGNORECASE)
_MONTH_RE = re.compile(
    rf"(?<![\w-])(?:(?P<mod>early|mid|mid-|late|end of|beginning of|start of)\s+)?"
    rf"(?P<mon>{MONTH_NAMES})\.?\s+(?P<y>\d{{4}})(?![\w-])",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(?<![\w./-])(?:in|during|of|for|since|by)\s+(?P<y>20\d{2})(?![\w./-])", re.IGNORECASE)

_QUARTER_WORDS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
MIN_YEAR, MAX_YEAR = 1990, 2100


def _month_bounds(y: int, m: int) -> Tuple[date, date]:
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


def _iso(d: date) -> str:
    return d.isoformat()


def _mask(text: str, spans: List[Tuple[int, int]]) -> str:
    chars = list(text)
    for a, b in spans:
        for i in range(a, b):
            chars[i] = " "
    return "".join(chars)


def query_time_window(text: str) -> Optional[Tuple[str, str]]:
    """
    Return one inclusive (start, end) ISO-date window implied by the question, or None.
    Several expressions are merged into their hull ("between Jan 5 and Jan 9, 2026" -> both days).
    """
    if not text:
        return None
    starts: List[str] = []
    ends: List[str] = []
    spans: List[Tuple[int, int]] = []

    # 1) explicit dates ("2026-01-15", "Feb 12, 2026", "12 March 2026", "03/27/2025")
    for m in _FIND_RE.finditer(text):
        iso = normalize_date(m.group(0))
        if iso:
            day = iso[:10]
            starts.append(day)
            ends.append(day)
            spans.append(m.span())
    masked = _mask(text, spans)

    # 2) quarters / halves
    for m in _QUARTER_RE.finditer(masked):
        if m.group("q"):
            q, y = int(m.group("q")), int(m.group("y"))
        elif m.group("q2"):
            q, y = int(m.group("q2")), int(m.group("y2"))
        else:
            q, y = _QUARTER_WORDS[m.group("word").lower()], int(m.group("y3"))
        if not (MIN_YEAR <= y <= MAX_YEAR):
            continue
        first = 3 * (q - 1) + 1
        starts.append(_iso(_month_bounds(y, first)[0]))
        ends.append(_iso(_month_bounds(y, first + 2)[1]))
        spans.append(m.span())
    for m in _HALF_RE.finditer(masked):
        h, y = int(m.group("h")), int(m.group("y"))
        if not (MIN_YEAR <= y <= MAX_YEAR):
            continue
        first = 1 if h == 1 else 7
        starts.append(_iso(_month_bounds(y, first)[0]))
        ends.append(_iso(_month_bounds(y, first + 5)[1]))
        spans.append(m.span())
    masked = _mask(text, spans)

    # 3) "early / mid / late March 2026", "March 2026"
    for m in _MONTH_RE.finditer(masked):
        mon = MONTHS.get(m.group("mon").lower()[:4]) or MONTHS.get(m.group("mon").lower()[:3])
        y = int(m.group("y"))
        if mon is None or not (MIN_YEAR <= y <= MAX_YEAR):
            continue
        lo, hi = _month_bounds(y, mon)
        mod = (m.group("mod") or "").lower().rstrip("-")
        if mod in ("early", "beginning of", "start of"):
            hi = date(y, mon, 10)
        elif mod == "mid":
            lo, hi = date(y, mon, 11), date(y, mon, 20)
        elif mod in ("late", "end of"):
            lo = date(y, mon, 21)
        starts.append(_iso(lo))
        ends.append(_iso(hi))
        spans.append(m.span())
    masked = _mask(text, spans)

    # 4) a bare year only when nothing more specific was found ("in 2025")
    if not starts:
        for m in _YEAR_RE.finditer(masked):
            y = int(m.group("y"))
            starts.append(_iso(date(y, 1, 1)))
            ends.append(_iso(date(y, 12, 31)))

    if not starts:
        return None
    return min(starts), max(ends)
