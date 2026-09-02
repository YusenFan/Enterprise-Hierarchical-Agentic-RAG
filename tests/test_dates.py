import pytest

from src.metadata.dates import combine_date_time, find_dates, normalize_date


@pytest.mark.parametrize("raw, expected", [
    ("Mon, 10 Jun 2025 09:12:00 -0700", "2025-06-10T16:12:00Z"),
    ("2025-06-10T09:12:00-07:00", "2025-06-10T16:12:00Z"),
    ("2025-06-10T09:12:00Z", "2025-06-10T09:12:00Z"),
    ("2026-03-09T14:12:03.342Z", "2026-03-09T14:12:03Z"),
    ("Tue, Jun 3, 2025 at 9:12 AM", "2025-06-03T09:12:00"),
    ("Wed, May 14, 2025 at 9:12 AM PT", "2025-05-14T16:12:00Z"),
    ("2025-03-27", "2025-03-27"),
    ("2025-03-27 | Duration: 62 minutes", "2025-03-27"),
    ("2026-01-14 15:00 UTC", "2026-01-14T15:00:00Z"),
    ("March 27, 2025", "2025-03-27"),
    ("27 March 2025", "2025-03-27"),
    ("06/03/2025", "2025-06-03"),
    ("[2026-03-01T14:12:07.342Z]", "2026-03-01T14:12:07Z"),
    ("Mon, Apr 05 2027 09:12:00 -0700", "2027-04-05T16:12:00Z"),
    ("garbage", None),
    ("", None),
    (None, None),
    ("1.18.0", None),
])
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


def test_combine_date_time():
    assert combine_date_time("2026-01-14", "10:03 AM PT") == "2026-01-14T18:03:00Z"   # January: PST
    assert combine_date_time("2026-07-14", "10:03 AM PT") == "2026-07-14T17:03:00Z"   # July: PDT
    assert combine_date_time("2026-01-14", "15:00 UTC") == "2026-01-14T15:00:00Z"
    assert combine_date_time("2026-01-14", None) == "2026-01-14"
    assert combine_date_time("2026-01-14T15:00:00Z", "10:00 AM") == "2026-01-14T15:00:00Z"


def test_find_dates_prose():
    text = ("we shipped on 2026-05-04 and again May 27, 2025; log 2026-03-09T14:12:03Z; "
            "version 1.18.0 and py:1.18.0 are not dates; 12/01/2025 is.")
    assert find_dates(text) == ["2025-05-27", "2025-12-01", "2026-03-09T14:12:03Z", "2026-05-04"]


def test_find_dates_sorted_and_capped():
    text = " ".join(f"2026-01-{d:02d}" for d in range(1, 29))
    dates = find_dates(text, max_n=5)
    assert dates == ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
