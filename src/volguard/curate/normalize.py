"""Layer-1 normalization: raw sources -> canonical per-trade observations.

This module turns two raw Layer-0 sources into one canonical, per-trade long
frame keyed by ``(snap_ts, expiry, strike, cp)`` (design Component 1):

- :func:`normalize_trades` — Deribit option trades (``TRADES_OPTIONS``).
- :func:`canonical_from_tardis` — Tardis free-day option chains
  (``TARDIS_CHAIN``), emitted in the *same* canonical shape so the golden
  validation pass and future quote-based sources share one code path.

Both return exactly :data:`CANONICAL_COLUMNS` (same names, dtypes, and order),
so downstream stages (forwards, IV cross-check, filters) do not care which
source a row came from.

Conventions match the trusted M1 core and the M2 raw schemas:

- Timestamps are UTC-aware millisecond ``Datetime`` (``_TS``), matching
  :mod:`volguard.ingest.schemas`.
- ``tau`` is time-to-expiry in years, ``(expiry - snap_ts)`` measured in
  seconds and divided by a 365-day year.
- ``usd_premium = price_btc * index_price`` (Deribit inverse-contract: premium
  is quoted in BTC, the Black-76 solver wants USD alongside a USD forward).
- Deribit / Tardis report IV in *percent* (e.g. ``65.0``); it is carried as a
  fraction ``iv_trade`` (``0.65``).
- ``cp_sign`` (+1 for calls, -1 for puts) bridges the canonical ``"C"/"P"``
  strings to the ``blackiv`` solver's ``CallPut`` (+1/-1) convention.
- Leakage rule: only rows with ``source_ts <= snap_ts`` survive.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from volguard.config import CurateConfig
from volguard.ingest.schemas import parse_instrument

__all__ = ["CANONICAL_COLUMNS", "canonical_from_tardis", "normalize_trades"]

# UTC-millisecond datetimes, identical to the raw-layer schemas (M2).
_TS = pl.Datetime(time_unit="ms", time_zone="UTC")

# 365-day year in seconds; ``tau`` is expressed against this fixed convention.
_SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0

# The frozen canonical per-trade shape shared by every normalization source.
# forwards.py / filters.py consume exactly these columns; both normalizers emit
# them in this order with these dtypes.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "snap_ts",
    "expiry",
    "tau",
    "strike",
    "cp",
    "price_btc",
    "index_price",
    "usd_premium",
    "iv_trade",
    "size",
    "block_flag",
    "source_ts",
    "cp_sign",
)


def _snap_instant(snap_ts: datetime, cfg: CurateConfig) -> datetime:
    """Pin ``snap_ts`` to the configured snap time (08:05 UTC) of its own day.

    The snap *day* comes from the caller-supplied instant; the *time* is fixed
    from ``cfg.snap_hour_utc`` / ``cfg.snap_minute_utc`` so every row on a snap
    day shares one canonical, tz-aware UTC ``snap_ts`` (design R1.3).
    """
    aware = snap_ts.replace(tzinfo=UTC) if snap_ts.tzinfo is None else snap_ts.astimezone(UTC)
    return aware.replace(
        hour=cfg.snap_hour_utc, minute=cfg.snap_minute_utc, second=0, microsecond=0
    )


def _parsed_instrument_columns(names: list[str]) -> tuple[pl.Series, pl.Series, pl.Series]:
    """Parse Deribit instrument names into (expiry, strike, cp) Series.

    Reuses the M2 :func:`parse_instrument` (a pure string function) so the
    instrument-name grammar lives in exactly one place. Dtypes are pinned so
    empty inputs and all-null columns still carry the canonical types.
    """
    parsed = [parse_instrument(name) for name in names]
    expiry = pl.Series("expiry", [p.expiry for p in parsed], dtype=_TS)
    strike = pl.Series("strike", [p.strike for p in parsed], dtype=pl.Float64)
    cp = pl.Series("cp", [p.cp for p in parsed], dtype=pl.String)
    return expiry, strike, cp


def _tau_years(expiry: pl.Expr, snap_ts: datetime) -> pl.Expr:
    """``(expiry - snap_ts)`` as a fraction of a 365-day year (design R1.3)."""
    delta_us = (expiry - pl.lit(snap_ts).cast(_TS)).dt.total_microseconds()
    return delta_us.cast(pl.Float64) / 1_000_000.0 / _SECONDS_PER_YEAR


def _cp_sign(cp: pl.Expr) -> pl.Expr:
    """Bridge canonical ``"C"/"P"`` to the blackiv ``CallPut`` (+1/-1)."""
    return pl.when(cp == "C").then(pl.lit(1)).otherwise(pl.lit(-1)).cast(pl.Int64)


def normalize_trades(
    raw_options: pl.LazyFrame,
    instruments: pl.LazyFrame,  # reserved: ref-instruments join is a later-stage concern
    snap_ts: datetime,
    cfg: CurateConfig,
) -> pl.DataFrame:
    """Normalize raw Deribit option trades into canonical per-trade rows.

    Keyed by ``(snap_ts, expiry, strike, cp)``; missing ``expiry``/``strike``/
    ``cp`` are derived from the instrument name via M2 ``parse_instrument``
    (design R1.1, R1.2). Assigns the 08:05-UTC ``snap_ts`` and ``tau``
    (R1.3), computes ``usd_premium`` (R1.4), converts the percent IV to a
    fraction ``iv_trade`` (R1.5), carries ``size``/``block_flag``/``source_ts``
    (R1.6) plus a ``cp_sign`` bridge, and drops any row with
    ``source_ts > snap_ts`` (the leakage rule, R1.8).

    ``instruments`` is accepted to match the design signature (the reference
    join is a later-stage concern); the instrument name on each trade already
    carries everything normalization needs.
    """
    snap = _snap_instant(snap_ts, cfg)
    df = raw_options.lazy().collect()

    exp_p, strike_p, cp_p = _parsed_instrument_columns(df["instrument"].to_list())
    df = df.with_columns(
        exp_p.alias("_expiry_parsed"),
        strike_p.alias("_strike_parsed"),
        cp_p.alias("_cp_parsed"),
    )

    # coalesce(pre-parsed, name-derived): trust an explicit column when present,
    # otherwise fall back to the instrument-name parse (R1.2).
    expiry = pl.coalesce([pl.col("expiry"), pl.col("_expiry_parsed")]).cast(_TS)
    strike = pl.coalesce([pl.col("strike"), pl.col("_strike_parsed")]).cast(pl.Float64)
    cp = pl.coalesce([pl.col("cp"), pl.col("_cp_parsed")]).cast(pl.String)

    out = df.with_columns(
        expiry.alias("expiry"),
        strike.alias("strike"),
        cp.alias("cp"),
    ).with_columns(
        pl.lit(snap).cast(_TS).alias("snap_ts"),
        _tau_years(pl.col("expiry"), snap).alias("tau"),
        (pl.col("price_btc") * pl.col("index_price")).alias("usd_premium"),
        (pl.col("iv") / 100.0).alias("iv_trade"),
        pl.col("amount").cast(pl.Float64).alias("size"),
        pl.col("ts").cast(_TS).alias("source_ts"),
        _cp_sign(pl.col("cp")).alias("cp_sign"),
    )

    return (
        out.filter(pl.col("source_ts") <= pl.lit(snap).cast(_TS))
        .select(CANONICAL_COLUMNS)
    )


def canonical_from_tardis(
    chain: pl.LazyFrame,
    snap_ts: datetime,
    cfg: CurateConfig,
) -> pl.DataFrame:
    """Normalize a Tardis free-day chain into the canonical shape (design R1.7).

    Tardis columns (``TARDIS_CHAIN``): ``symbol``, ``timestamp`` (epoch
    *microseconds*), ``type``, ``strike_price``, ``expiration`` (epoch
    *milliseconds*), ``mark_iv``, ``underlying_price``. The instrument
    ``symbol`` is parsed via M2 ``parse_instrument`` for ``strike``/``cp``;
    ``expiry`` prefers the ``expiration`` epoch (ms) and falls back to the
    parsed expiry; ``source_ts`` comes from ``timestamp`` (us).

    IV is carried from ``mark_iv`` (percent -> fraction). The Tardis path is
    quote-based: it has no traded premium, so ``usd_premium`` is derived from
    ``mark_price * underlying_price`` when a mark price column is present, else
    left null (design Component 1 / error-handling notes). Emits only rows with
    ``source_ts <= snap_ts`` (the leakage rule).
    """
    snap = _snap_instant(snap_ts, cfg)
    df = chain.lazy().collect()

    exp_p, strike_p, cp_p = _parsed_instrument_columns(df["symbol"].to_list())
    df = df.with_columns(
        exp_p.alias("_expiry_parsed"),
        strike_p.alias("_strike_parsed"),
        cp_p.alias("_cp_parsed"),
    )

    # expiration is epoch-ms, timestamp is epoch-us (verified against the
    # committed fixture); attach UTC and normalize both to _TS (ms).
    expiry_epoch = pl.from_epoch(pl.col("expiration"), time_unit="ms").dt.replace_time_zone("UTC")
    source_ts = (
        pl.from_epoch(pl.col("timestamp"), time_unit="us")
        .dt.replace_time_zone("UTC")
        .cast(_TS)
    )
    expiry = pl.coalesce([expiry_epoch, pl.col("_expiry_parsed")]).cast(_TS)
    cp = pl.coalesce([pl.col("_cp_parsed")]).cast(pl.String)
    strike = pl.coalesce([pl.col("strike_price"), pl.col("_strike_parsed")]).cast(pl.Float64)

    # No traded premium in a quote chain; derive from mark_price if it survived
    # the (strict=False) Tardis validation, otherwise leave null.
    if "mark_price" in df.columns:
        price_btc = pl.col("mark_price").cast(pl.Float64)
        usd_premium = (pl.col("mark_price") * pl.col("underlying_price")).cast(pl.Float64)
    else:
        price_btc = pl.lit(None, dtype=pl.Float64)
        usd_premium = pl.lit(None, dtype=pl.Float64)

    out = df.with_columns(
        expiry.alias("expiry"),
        strike.alias("strike"),
        cp.alias("cp"),
        source_ts.alias("source_ts"),
    ).with_columns(
        pl.lit(snap).cast(_TS).alias("snap_ts"),
        _tau_years(pl.col("expiry"), snap).alias("tau"),
        price_btc.alias("price_btc"),
        pl.col("underlying_price").cast(pl.Float64).alias("index_price"),
        usd_premium.alias("usd_premium"),
        (pl.col("mark_iv") / 100.0).alias("iv_trade"),
        pl.lit(None, dtype=pl.Float64).alias("size"),
        pl.lit(value=False).alias("block_flag"),
        _cp_sign(pl.col("cp")).alias("cp_sign"),
    )

    return (
        out.filter(pl.col("source_ts") <= pl.lit(snap).cast(_TS))
        .select(CANONICAL_COLUMNS)
    )
