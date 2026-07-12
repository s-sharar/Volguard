"""Unit tests for ``curate/normalize.py`` (design Component 1, R1.1-R1.8).

Covers both canonical sources:

- :func:`normalize_trades` — instrument-name parsing fallback, ``tau``
  arithmetic, ``usd_premium``, percent->fraction IV, ``source_ts``
  propagation, ``cp_sign`` bridge, and the ``source_ts <= snap_ts`` leakage
  filter.
- :func:`canonical_from_tardis` — the committed ``tardis_sample.csv`` free-day
  chain normalized into the identical canonical shape (epoch-us ``timestamp``,
  epoch-ms ``expiration``).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from volguard.config import CurateConfig
from volguard.curate.normalize import (
    CANONICAL_COLUMNS,
    canonical_from_tardis,
    normalize_trades,
)
from volguard.ingest.schemas import TARDIS_CHAIN, TRADES_OPTIONS, validate

FIXTURE = Path(__file__).parent / "fixtures" / "tardis_sample.csv"

_SNAP = datetime(2022, 4, 1, 8, 5, tzinfo=UTC)
_EXPIRY = datetime(2022, 4, 8, 8, 0, tzinfo=UTC)  # BTC-8APR22 settles 08:00 UTC
_TS = pl.Datetime(time_unit="ms", time_zone="UTC")
_CFG = CurateConfig()


def _instruments() -> pl.LazyFrame:
    """A minimal (unused-by-logic) instrument reference frame."""
    return pl.DataFrame(
        {
            "instrument": ["BTC-8APR22-40000-C"],
            "kind": ["option"],
            "expiry": pl.Series([_EXPIRY], dtype=_TS),
            "strike": [40000.0],
            "cp": ["C"],
            "creation_ts": pl.Series([_SNAP], dtype=_TS),
        }
    ).lazy()


def _raw_option_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ts": datetime(2022, 4, 1, 8, 0, tzinfo=UTC),
        "instrument": "BTC-8APR22-40000-C",
        "expiry": None,  # force the parse_instrument fallback
        "strike": None,
        "cp": None,
        "price_btc": 0.05,
        "iv": 65.0,  # percent
        "amount": 2.5,
        "index_price": 45000.0,
        "trade_id": "t1",
        "block_flag": False,
        "source": "deribit_history",
    }
    row.update(overrides)
    return row


def _raw_options(rows: list[dict[str, object]]) -> pl.LazyFrame:
    df = pl.DataFrame(rows)
    return validate(df, TRADES_OPTIONS).lazy()


def test_normalize_trades_emits_canonical_columns() -> None:
    out = normalize_trades(_raw_options([_raw_option_row()]), _instruments(), _SNAP, _CFG)
    assert out.columns == list(CANONICAL_COLUMNS)
    assert out.height == 1


def test_normalize_trades_parses_instrument_when_fields_missing() -> None:
    """R1.2: missing expiry/strike/cp derived via parse_instrument."""
    out = normalize_trades(_raw_options([_raw_option_row()]), _instruments(), _SNAP, _CFG)
    assert out["strike"][0] == 40000.0
    assert out["cp"][0] == "C"
    assert out["expiry"][0] == _EXPIRY


def test_normalize_trades_snap_ts_pinned_to_0805() -> None:
    """R1.3: snap_ts is 08:05 UTC of the snap day regardless of the input time."""
    out = normalize_trades(
        _raw_options([_raw_option_row()]),
        _instruments(),
        datetime(2022, 4, 1, 23, 59, tzinfo=UTC),  # arbitrary time on the day
        _CFG,
    )
    assert out["snap_ts"][0] == datetime(2022, 4, 1, 8, 5, tzinfo=UTC)


def test_normalize_trades_tau_arithmetic() -> None:
    """R1.3: tau = (expiry - snap_ts) in years (365-day convention)."""
    out = normalize_trades(_raw_options([_raw_option_row()]), _instruments(), _SNAP, _CFG)
    expected = (_EXPIRY - _SNAP).total_seconds() / (365.0 * 24.0 * 3600.0)
    assert math.isclose(out["tau"][0], expected, rel_tol=1e-12)


def test_normalize_trades_usd_premium() -> None:
    """R1.4: usd_premium = price_btc * index_price."""
    out = normalize_trades(_raw_options([_raw_option_row()]), _instruments(), _SNAP, _CFG)
    assert math.isclose(out["usd_premium"][0], 0.05 * 45000.0)


def test_normalize_trades_iv_percent_to_fraction() -> None:
    """R1.5: Deribit percent IV (65.0) carried as fraction (0.65)."""
    out = normalize_trades(_raw_options([_raw_option_row()]), _instruments(), _SNAP, _CFG)
    assert math.isclose(out["iv_trade"][0], 0.65)


def test_normalize_trades_carries_size_block_source_ts() -> None:
    """R1.6: size from amount, block_flag, source_ts from ts."""
    src = datetime(2022, 4, 1, 7, 30, tzinfo=UTC)
    out = normalize_trades(
        _raw_options([_raw_option_row(amount=3.0, block_flag=True, ts=src)]),
        _instruments(),
        _SNAP,
        _CFG,
    )
    assert out["size"][0] == 3.0
    assert out["block_flag"][0] is True
    assert out["source_ts"][0] == src


def test_normalize_trades_cp_sign_bridge() -> None:
    """cp_sign is +1 for calls, -1 for puts (blackiv CallPut bridge)."""
    rows = [
        _raw_option_row(instrument="BTC-8APR22-40000-C", trade_id="c"),
        _raw_option_row(instrument="BTC-8APR22-40000-P", trade_id="p"),
    ]
    out = normalize_trades(_raw_options(rows), _instruments(), _SNAP, _CFG).sort("cp")
    by_cp = dict(zip(out["cp"], out["cp_sign"], strict=True))
    assert by_cp == {"C": 1, "P": -1}


def test_normalize_trades_leakage_filter() -> None:
    """R1.8: rows with source_ts > snap_ts are dropped."""
    rows = [
        _raw_option_row(ts=datetime(2022, 4, 1, 8, 0, tzinfo=UTC), trade_id="keep"),
        _raw_option_row(ts=datetime(2022, 4, 1, 8, 6, tzinfo=UTC), trade_id="leak"),
    ]
    out = normalize_trades(_raw_options(rows), _instruments(), _SNAP, _CFG)
    assert out.height == 1
    assert out.select((pl.col("source_ts") <= _SNAP).all()).item()


def test_normalize_trades_boundary_source_ts_equal_snap_kept() -> None:
    """R1.8 boundary: source_ts == snap_ts is kept (<=, not <)."""
    rows = [_raw_option_row(ts=datetime(2022, 4, 1, 8, 5, tzinfo=UTC))]
    out = normalize_trades(_raw_options(rows), _instruments(), _SNAP, _CFG)
    assert out.height == 1


# --- Tardis path ----------------------------------------------------------


def _tardis_chain() -> pl.LazyFrame:
    df = pl.read_csv(FIXTURE)
    return validate(df, TARDIS_CHAIN).lazy()


def test_canonical_from_tardis_same_shape() -> None:
    """R1.7: Tardis chain normalized into the identical canonical shape."""
    out = canonical_from_tardis(_tardis_chain(), _SNAP, _CFG)
    assert out.columns == list(CANONICAL_COLUMNS)
    assert out.height == 8  # all fixture rows have source_ts <= snap_ts


def test_canonical_from_tardis_parses_symbol_and_units() -> None:
    """Symbol parsed for strike/cp; expiration epoch-ms -> 08:00-UTC expiry."""
    out = canonical_from_tardis(_tardis_chain(), _SNAP, _CFG).sort("strike", "cp")
    first = out.row(0, named=True)
    assert first["strike"] == 40000.0
    assert first["cp"] in {"C", "P"}
    # 1648800000000 ms == 2022-04-01 08:00:00 UTC
    assert first["expiry"] == datetime(2022, 4, 1, 8, 0, tzinfo=UTC)


def test_canonical_from_tardis_iv_percent_to_fraction() -> None:
    """mark_iv percent (e.g. 59.3) carried as fraction (0.593)."""
    out = canonical_from_tardis(_tardis_chain(), _SNAP, _CFG).sort("strike", "cp")
    # BTC-1APR22-40000-C mark_iv == 59.3 in the fixture.
    call_40k = out.filter((pl.col("strike") == 40000.0) & (pl.col("cp") == "C"))
    assert math.isclose(call_40k["iv_trade"][0], 0.593)


def test_canonical_from_tardis_source_ts_from_timestamp_us() -> None:
    """timestamp is epoch-microseconds (1648771200000000 -> 2022-04-01 00:00 UTC)."""
    out = canonical_from_tardis(_tardis_chain(), _SNAP, _CFG)
    assert out["source_ts"][0] == datetime(2022, 4, 1, 0, 0, tzinfo=UTC)
    assert out.select((pl.col("source_ts") <= _SNAP).all()).item()


def test_canonical_from_tardis_leakage_filter() -> None:
    """R1.8 on the Tardis path: a post-snap quote is dropped."""
    df = pl.read_csv(FIXTURE)
    # Push one row's timestamp past the snap (epoch-us for 2022-04-01 09:00 UTC).
    leak_us = int(datetime(2022, 4, 1, 9, 0, tzinfo=UTC).timestamp() * 1_000_000)
    df = df.with_columns(
        pl.when(pl.col("symbol") == "BTC-1APR22-50000-C")
        .then(pl.lit(leak_us))
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    )
    chain = validate(df, TARDIS_CHAIN).lazy()
    out = canonical_from_tardis(chain, _SNAP, _CFG)
    assert out.height == 7  # the leaking row is dropped
