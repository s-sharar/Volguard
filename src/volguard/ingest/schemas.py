"""Raw-layer pandera schemas + Deribit instrument-name parsing.

These schemas are the frozen contract between Layer 0 (ingestion) and Layer 1
(curation): every ``data/raw/`` table is validated against one of them before
it is written to Parquet, so downstream stages can trust column names and
dtypes without re-checking.  Timestamps are stored as UTC-aware millisecond
``Datetime`` columns (Deribit's native epoch-ms resolution, made explicit).

The instrument-name parser lives here (rather than in ``curate/``) because both
the trades backfill and the expired-instrument reference table need it, and it
is a pure string function with no data dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

import pandera.polars as pa
import polars as pl

# Deribit option/future names look like ``BTC-27JUN25-100000-C`` (option),
# ``BTC-27JUN25`` (dated future) or ``BTC-PERPETUAL`` (perp).
_DATE_RE = re.compile(r"^(\d{1,2})([A-Z]{3})(\d{2})$")
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}  # fmt: skip
# Deribit options settle at 08:00 UTC on their expiry date.
_EXPIRY_HOUR_UTC = 8
_YEAR_2000 = 2000
_OPTION_PARTS = 4
_FUTURE_PARTS = 2


@dataclass(frozen=True, slots=True)
class ParsedInstrument:
    """Structured view of a Deribit instrument name."""

    currency: str
    kind: str  # "option" | "future" | "perpetual"
    expiry: datetime | None  # None for perpetuals
    strike: float | None  # None for futures/perps
    cp: str | None  # "C" | "P" for options, else None


def _parse_expiry(token: str) -> datetime | None:
    """Parse a ``27JUN25`` expiry token into an 08:00-UTC datetime."""
    m = _DATE_RE.match(token)
    if m is None:
        return None
    day, mon, yy = m.groups()
    month = _MONTHS.get(mon)
    if month is None:
        return None
    return datetime(_YEAR_2000 + int(yy), month, int(day), _EXPIRY_HOUR_UTC, 0, tzinfo=UTC)


def parse_instrument(name: str) -> ParsedInstrument:
    """Parse any Deribit BTC instrument name into its components.

    Raises ``ValueError`` on names that do not match a known layout so callers
    fail loudly rather than silently mislabel a row.
    """
    parts = name.split("-")
    currency = parts[0]
    if len(parts) == _FUTURE_PARTS:
        if parts[1] == "PERPETUAL":
            return ParsedInstrument(currency, "perpetual", None, None, None)
        expiry = _parse_expiry(parts[1])
        if expiry is not None:
            return ParsedInstrument(currency, "future", expiry, None, None)
    elif len(parts) == _OPTION_PARTS:
        expiry = _parse_expiry(parts[1])
        cp = parts[3].upper()
        if expiry is not None and cp in ("C", "P"):
            return ParsedInstrument(currency, "option", expiry, float(parts[2]), cp)
    raise ValueError(f"unrecognized Deribit instrument name: {name!r}")


# --- Parquet dtypes -------------------------------------------------------

_TS = pl.Datetime(time_unit="ms", time_zone="UTC")


def _ts_col(*, nullable: bool = False) -> pa.Column:
    return pa.Column(_TS, nullable=nullable)


# --- Raw-table schemas (plan section 6) -----------------------------------

TRADES_OPTIONS = pa.DataFrameSchema(
    {
        "ts": _ts_col(),
        "instrument": pa.Column(pl.String),
        "expiry": _ts_col(nullable=True),
        "strike": pa.Column(pl.Float64, nullable=True),
        "cp": pa.Column(pl.String, pa.Check.isin(["C", "P"]), nullable=True),
        "price_btc": pa.Column(pl.Float64, nullable=True),
        "iv": pa.Column(pl.Float64, nullable=True),
        "amount": pa.Column(pl.Float64, nullable=True),
        "index_price": pa.Column(pl.Float64, nullable=True),
        "trade_id": pa.Column(pl.String),
        "block_flag": pa.Column(pl.Boolean),
        "source": pa.Column(pl.String),
    },
    strict=True,
    coerce=True,
)

TRADES_FUTURES = pa.DataFrameSchema(
    {
        "ts": _ts_col(),
        "instrument": pa.Column(pl.String),
        "price": pa.Column(pl.Float64, nullable=True),
        "amount": pa.Column(pl.Float64, nullable=True),
        "index_price": pa.Column(pl.Float64, nullable=True),
        "trade_id": pa.Column(pl.String),
        "block_flag": pa.Column(pl.Boolean),
        "source": pa.Column(pl.String),
    },
    strict=True,
    coerce=True,
)

OHLC = pa.DataFrameSchema(
    {
        "ts": _ts_col(),
        "instrument": pa.Column(pl.String),
        "open": pa.Column(pl.Float64, nullable=True),
        "high": pa.Column(pl.Float64, nullable=True),
        "low": pa.Column(pl.Float64, nullable=True),
        "close": pa.Column(pl.Float64, nullable=True),
        "volume": pa.Column(pl.Float64, nullable=True),
    },
    strict=True,
    coerce=True,
)

DVOL = pa.DataFrameSchema(
    {
        "ts": _ts_col(),
        "currency": pa.Column(pl.String),
        "open": pa.Column(pl.Float64, nullable=True),
        "high": pa.Column(pl.Float64, nullable=True),
        "low": pa.Column(pl.Float64, nullable=True),
        "close": pa.Column(pl.Float64, nullable=True),
    },
    strict=True,
    coerce=True,
)

FUNDING = pa.DataFrameSchema(
    {
        "ts": _ts_col(),
        "instrument": pa.Column(pl.String),
        "interest_1h": pa.Column(pl.Float64, nullable=True),
        "interest_8h": pa.Column(pl.Float64, nullable=True),
        "index_price": pa.Column(pl.Float64, nullable=True),
    },
    strict=True,
    coerce=True,
)

DELIVERY_PRICES = pa.DataFrameSchema(
    {
        "date": pa.Column(pl.String),
        "index": pa.Column(pl.String),
        "delivery_price": pa.Column(pl.Float64, nullable=True),
    },
    strict=True,
    coerce=True,
)

INSTRUMENTS = pa.DataFrameSchema(
    {
        "instrument": pa.Column(pl.String),
        "kind": pa.Column(pl.String),
        "expiry": _ts_col(nullable=True),
        "strike": pa.Column(pl.Float64, nullable=True),
        "cp": pa.Column(pl.String, pa.Check.isin(["C", "P"]), nullable=True),
        "creation_ts": _ts_col(nullable=True),
    },
    strict=True,
    coerce=True,
)

# Tardis free-day option chains carry many analytics columns; validate the ones
# we rely on downstream and allow the rest through (strict=False) so a Tardis
# schema tweak upstream does not break ingestion.
TARDIS_CHAIN = pa.DataFrameSchema(
    {
        "symbol": pa.Column(pl.String),
        "timestamp": pa.Column(pl.Int64, nullable=True),
        "type": pa.Column(pl.String, nullable=True),
        "strike_price": pa.Column(pl.Float64, nullable=True),
        "expiration": pa.Column(pl.Int64, nullable=True),
        "mark_iv": pa.Column(pl.Float64, nullable=True),
        "underlying_price": pa.Column(pl.Float64, nullable=True),
    },
    strict=False,
    coerce=True,
)


def validate[T: (pl.DataFrame, pl.LazyFrame)](df: T, schema: pa.DataFrameSchema) -> T:
    """Validate a (Lazy)Frame against ``schema``, returning the coerced frame."""
    return schema.validate(df)
