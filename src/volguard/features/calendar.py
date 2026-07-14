"""Exchange-calendar features evaluated at the daily snap."""

from __future__ import annotations

import math
from calendar import monthrange
from datetime import UTC, datetime


def _last_friday(year: int, month: int) -> datetime:
    last_day = monthrange(year, month)[1]
    candidate = datetime(year, month, last_day, 8, tzinfo=UTC)
    return candidate.replace(day=last_day - (candidate.weekday() - 4) % 7)


def _next_expiry(snap_ts: datetime, months: tuple[int, ...] | None = None) -> datetime:
    year, month = snap_ts.year, snap_ts.month
    for offset in range(25):
        absolute = year * 12 + month - 1 + offset
        candidate_year, month_zero = divmod(absolute, 12)
        candidate_month = month_zero + 1
        if months is not None and candidate_month not in months:
            continue
        expiry = _last_friday(candidate_year, candidate_month)
        if expiry > snap_ts:
            return expiry
    raise RuntimeError("could not find a future Deribit expiry")


def calendar_features(snap_ts: datetime) -> dict[str, int]:
    """Return weekday and ceil calendar-day distances to exchange expiries."""
    monthly = _next_expiry(snap_ts)
    quarterly = _next_expiry(snap_ts, (3, 6, 9, 12))
    seconds_per_day = 86_400.0
    return {
        "day_of_week": snap_ts.weekday(),
        "days_to_monthly_expiry": math.ceil((monthly - snap_ts).total_seconds() / seconds_per_day),
        "days_to_quarterly_expiry": math.ceil(
            (quarterly - snap_ts).total_seconds() / seconds_per_day
        ),
    }
