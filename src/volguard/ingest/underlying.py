"""Underlying market-data backfill: OHLC, DVOL, funding, deliveries, instruments.

Pulls the non-options context the surface pipeline needs, all via the same
throttled :class:`~volguard.ingest.client.HistoryClient`:

* ``futures_ohlc`` -- perpetual (+ any configured dated futures) candles,
* ``index_ohlc``   -- underlying-index proxy candles (the perp by default),
* ``dvol``         -- Deribit DVOL volatility-index candles,
* ``funding``      -- perpetual hourly funding-rate history,
* ``delivery_prices`` -- daily settlement prices,
* ``ref/instruments`` -- the expired + live instrument reference table.

Each writes a schema-validated Parquet table under ``data/raw`` (``ref`` under
``data/ref``). Requests are chunked by time so a multi-year pull stays within
Deribit's per-request bar cap; re-running overwrites the (small) tables.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from volguard.config import DataConfig
from volguard.ingest.client import HistoryClient
from volguard.ingest.schemas import (
    DELIVERY_PRICES,
    DVOL,
    FUNDING,
    INSTRUMENTS,
    OHLC,
    validate,
)

log = logging.getLogger("volguard.ingest")

_MS_PER_S = 1000
_SEC_PER_MIN = 60
_MS_PER_MIN = _SEC_PER_MIN * _MS_PER_S
_MAX_BARS = 5000  # Deribit caps chart/vol-index responses; chunk under this.
_FUNDING_CHUNK_DAYS = 30
_MS_PER_DAY = 24 * 60 * _MS_PER_MIN
_DELIVERY_PAGE = 100
_CP_FROM_TYPE = {"call": "C", "put": "P"}
_OHLC_COLS = ["ts", "instrument", "open", "high", "low", "close", "volume"]


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * _MS_PER_S)


def _f(value: Any) -> float | None:
    return None if value is None else float(value)


def _utc_ms(col: str) -> pl.Expr:
    return pl.from_epoch(col, time_unit="ms").dt.replace_time_zone("UTC").alias(col)


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd")


# --- OHLC -----------------------------------------------------------------


def _ohlc_frame(result: dict[str, Any], instrument: str) -> pl.DataFrame | None:
    ticks = result.get("ticks") or []
    if not ticks:
        return None
    n = len(ticks)
    vol = result.get("volume") or [None] * n
    df = pl.DataFrame(
        {
            "ts": [int(t) for t in ticks],
            "open": [_f(x) for x in result.get("open") or [None] * n],
            "high": [_f(x) for x in result.get("high") or [None] * n],
            "low": [_f(x) for x in result.get("low") or [None] * n],
            "close": [_f(x) for x in result.get("close") or [None] * n],
            "volume": [_f(x) for x in vol],
        }
    ).with_columns(_utc_ms("ts"), pl.lit(instrument).alias("instrument"))
    return validate(df.select(_OHLC_COLS), OHLC)


async def fetch_ohlc(
    client: HistoryClient, instrument: str, start_ms: int, end_ms: int, *, resolution: str
) -> pl.DataFrame | None:
    """Fetch OHLC candles for one instrument, chunked under the bar cap."""
    res_min = int(resolution) if resolution.isdigit() else 1
    # A chunk spanning N intervals covers N+1 inclusive candle timestamps, so
    # request one interval short of the cap; otherwise the boundary candle could
    # be trimmed by the server's ~5000-bar limit and then skipped forever by the
    # cur = chunk_end + 1 advance, leaving holes in long backfills.
    chunk_ms = (_MAX_BARS - 1) * res_min * _MS_PER_MIN
    frames: list[pl.DataFrame] = []
    cur = start_ms
    while cur <= end_ms:
        chunk_end = min(cur + chunk_ms, end_ms)
        result = await client.get_tradingview_chart_data(
            instrument, cur, chunk_end, resolution=resolution
        )
        frame = _ohlc_frame(result, instrument)
        if frame is not None:
            frames.append(frame)
        cur = chunk_end + 1
    if not frames:
        return None
    return pl.concat(frames).unique(subset=["ts"]).sort("ts")


# --- DVOL -----------------------------------------------------------------


def _dvol_frame(data: list[list[float]], currency: str) -> pl.DataFrame | None:
    if not data:
        return None
    df = pl.DataFrame(
        {
            "ts": [int(r[0]) for r in data],
            "open": [_f(r[1]) for r in data],
            "high": [_f(r[2]) for r in data],
            "low": [_f(r[3]) for r in data],
            "close": [_f(r[4]) for r in data],
        }
    ).with_columns(_utc_ms("ts"), pl.lit(currency).alias("currency"))
    return validate(df.select(["ts", "currency", "open", "high", "low", "close"]), DVOL)


async def fetch_dvol(
    client: HistoryClient, currency: str, start_ms: int, end_ms: int, *, resolution: str
) -> pl.DataFrame | None:
    """Fetch DVOL candles across the full range, following continuation pages.

    The volatility-index endpoint caps each response at ~1000 candles (the most
    recent within the range) and returns a ``continuation`` timestamp to fetch
    the older remainder. We page backward via ``continuation`` (using it as the
    next ``end_timestamp``) until it is absent, so long ranges are not silently
    truncated to the first page. ``unique`` drops any boundary overlap.
    """
    frames: list[pl.DataFrame] = []
    cursor_end = end_ms
    while True:
        result = await client.get_volatility_index_data(
            currency, start_ms, cursor_end, resolution=resolution
        )
        data = result.get("data") or []
        frame = _dvol_frame(data, currency)
        if frame is not None:
            frames.append(frame)
        cont = result.get("continuation")
        # Stop when the API signals no more pages (empty data or null
        # continuation) or the cursor would not advance backward (guards
        # against an unexpected non-decreasing continuation).
        if not data or cont is None or int(cont) >= cursor_end:
            break
        cursor_end = int(cont)
    if not frames:
        return None
    return pl.concat(frames).unique(subset=["ts"]).sort("ts")


# --- Funding --------------------------------------------------------------


def _funding_frame(records: list[dict[str, Any]], instrument: str) -> pl.DataFrame | None:
    if not records:
        return None
    df = pl.DataFrame(
        {
            "ts": [int(r["timestamp"]) for r in records],
            "interest_1h": [_f(r.get("interest_1h")) for r in records],
            "interest_8h": [_f(r.get("interest_8h")) for r in records],
            "index_price": [_f(r.get("index_price")) for r in records],
        }
    ).with_columns(_utc_ms("ts"), pl.lit(instrument).alias("instrument"))
    return validate(
        df.select(["ts", "instrument", "interest_1h", "interest_8h", "index_price"]),
        FUNDING,
    )


async def fetch_funding(
    client: HistoryClient, instrument: str, start_ms: int, end_ms: int
) -> pl.DataFrame | None:
    """Fetch perpetual funding-rate history, chunked by a fixed day span."""
    chunk_ms = _FUNDING_CHUNK_DAYS * _MS_PER_DAY
    records: list[dict[str, Any]] = []
    cur = start_ms
    while cur <= end_ms:
        chunk_end = min(cur + chunk_ms, end_ms)
        page = await client.get_funding_rate_history(instrument, cur, chunk_end)
        records.extend(page)
        cur = chunk_end + 1
    frame = _funding_frame(records, instrument)
    if frame is None:
        return None
    return frame.unique(subset=["ts"]).sort("ts")


# --- Delivery prices ------------------------------------------------------


async def fetch_delivery(client: HistoryClient, index_name: str) -> pl.DataFrame | None:
    """Fetch all historical daily delivery (settlement) prices for an index."""
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        res = await client.get_delivery_prices(index_name, offset=offset, count=_DELIVERY_PAGE)
        data = res.get("data") or []
        rows.extend(data)
        total = int(res.get("records_total", 0))
        offset += _DELIVERY_PAGE
        if not data or offset >= total:
            break
    if not rows:
        return None
    df = pl.DataFrame(
        {
            "date": [str(r["date"]) for r in rows],
            "delivery_price": [_f(r.get("delivery_price")) for r in rows],
        }
    ).with_columns(pl.lit(index_name).alias("index"))
    return validate(df.select(["date", "index", "delivery_price"]), DELIVERY_PRICES)


# --- Instrument reference -------------------------------------------------


def _instruments_frame(records: list[dict[str, Any]]) -> pl.DataFrame:
    rows = [
        {
            "instrument": r["instrument_name"],
            "kind": r.get("kind"),
            "expiry": r.get("expiration_timestamp"),
            "strike": _f(r.get("strike")),
            "cp": _CP_FROM_TYPE.get(r.get("option_type") or ""),
            "creation_ts": r.get("creation_timestamp"),
        }
        for r in records
    ]
    df = pl.DataFrame(rows).unique(subset=["instrument"], keep="first")
    df = df.with_columns(_utc_ms("expiry"), _utc_ms("creation_ts"))
    return validate(
        df.select(["instrument", "kind", "expiry", "strike", "cp", "creation_ts"]),
        INSTRUMENTS,
    )


async def fetch_instruments(client: HistoryClient, currency: str) -> pl.DataFrame:
    """Fetch the live + expired option and future instrument reference table."""
    records: list[dict[str, Any]] = []
    for kind in ("option", "future"):
        for expired in (True, False):
            records.extend(await client.get_instruments(currency, kind=kind, expired=expired))
    return _instruments_frame(records)


# --- Orchestration --------------------------------------------------------


async def run_underlying(
    cfg: DataConfig, *, start: str | None = None, end: str | None = None
) -> None:
    """Backfill every underlying dataset for the configured currency."""
    start_date = date.fromisoformat(start) if start else date.fromisoformat(cfg.history_start)
    end_date = date.fromisoformat(end) if end else datetime.now(UTC).date()
    start_ms = _ms(datetime(start_date.year, start_date.month, start_date.day, tzinfo=UTC))
    end_ms = _ms(datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=UTC))
    index_name = f"{cfg.currency.lower()}_usd"

    async with httpx.AsyncClient(timeout=cfg.request_timeout_s) as http:
        client = HistoryClient(
            http,
            base_url=cfg.underlying_base_url,
            rate_limit_rps=cfg.rate_limit_rps,
            max_retries=cfg.max_retries,
            retry_backoff_s=cfg.retry_backoff_s,
        )

        instruments = await fetch_instruments(client, cfg.currency)
        _write(instruments, cfg.ref_dir / "instruments" / "part.parquet")
        log.info("instruments: %d rows", instruments.height)

        for inst in [cfg.perpetual, *cfg.future_instruments]:
            frame = await fetch_ohlc(client, inst, start_ms, end_ms, resolution=cfg.ohlc_resolution)
            if frame is not None:
                path = cfg.raw_table_dir("futures_ohlc") / f"instrument={inst}" / "part.parquet"
                _write(frame, path)
                log.info("futures_ohlc %s: %d bars", inst, frame.height)

        idx = await fetch_ohlc(
            client, cfg.index_instrument, start_ms, end_ms, resolution=cfg.ohlc_resolution
        )
        if idx is not None:
            idx_dir = cfg.raw_table_dir("index_ohlc") / f"instrument={cfg.index_instrument}"
            _write(idx, idx_dir / "part.parquet")
            log.info("index_ohlc %s: %d bars", cfg.index_instrument, idx.height)

        dvol = await fetch_dvol(
            client, cfg.currency, start_ms, end_ms, resolution=cfg.dvol_resolution
        )
        if dvol is not None:
            _write(dvol, cfg.raw_table_dir("dvol") / "part.parquet")
            log.info("dvol: %d bars", dvol.height)

        funding = await fetch_funding(client, cfg.perpetual, start_ms, end_ms)
        if funding is not None:
            fund_dir = cfg.raw_table_dir("funding") / f"instrument={cfg.perpetual}"
            _write(funding, fund_dir / "part.parquet")
            log.info("funding %s: %d rows", cfg.perpetual, funding.height)

        delivery = await fetch_delivery(client, index_name)
        if delivery is not None:
            _write(delivery, cfg.raw_table_dir("delivery_prices") / "part.parquet")
            log.info("delivery_prices: %d rows", delivery.height)
