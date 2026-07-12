"""Resumable Deribit history-API trades backfill (options + futures).

Walks every BTC options and futures trade since ``history_start`` (default
2021-01-01) using ``get_last_trades_by_currency_and_time`` on
``history.deribit.com``, and writes month-partitioned, schema-validated Parquet:

    data/raw/trades_options/month=YYYY-MM/part.parquet
    data/raw/trades_futures/month=YYYY-MM/part.parquet

Resumability is at month granularity: a JSON checkpoint records which months are
fully written.  Re-running skips completed months and always re-pulls the final
(still-growing) month, overwriting its partition, so an interrupted multi-hour
run continues cleanly and a repeat run picks up newly printed trades.  Trades
are deduped on ``trade_id`` within each month, so overlap at pagination or month
boundaries never double-counts.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from volguard.config import DataConfig
from volguard.ingest.client import HistoryClient
from volguard.ingest.schemas import (
    TRADES_FUTURES,
    TRADES_OPTIONS,
    ParsedInstrument,
    parse_instrument,
    validate,
)

log = logging.getLogger("volguard.ingest")

_MS_PER_S = 1000
_DECEMBER = 12
# Options carry an option-specific column set; futures are the plainer stream.
_STREAM = {"option": "trades_options", "future": "trades_futures"}


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * _MS_PER_S)


def _month_start(d: date) -> datetime:
    return datetime(d.year, d.month, 1, tzinfo=UTC)


def _next_month(dt: datetime) -> datetime:
    if dt.month == _DECEMBER:
        return datetime(dt.year + 1, 1, 1, tzinfo=UTC)
    return datetime(dt.year, dt.month + 1, 1, tzinfo=UTC)


def iter_months(start: date, end: date) -> Iterator[tuple[datetime, datetime, str]]:
    """Yield ``(month_start, next_month_start, "YYYY-MM")`` covering ``[start, end]``."""
    cur = _month_start(start)
    last = _month_start(end)
    while cur <= last:
        nxt = _next_month(cur)
        yield cur, nxt, f"{cur.year:04d}-{cur.month:02d}"
        cur = nxt


def _checkpoint_path(cfg: DataConfig, stream: str) -> Path:
    return cfg.checkpoint_dir / f"{stream}.json"


def load_checkpoint(cfg: DataConfig, stream: str) -> dict[str, Any]:
    path = _checkpoint_path(cfg, stream)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_checkpoint(cfg: DataConfig, stream: str, data: dict[str, Any]) -> None:
    path = _checkpoint_path(cfg, stream)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


async def fetch_trades_window(
    client: HistoryClient,
    currency: str,
    kind: str,
    start_ms: int,
    end_ms: int,
    *,
    count: int,
) -> list[dict[str, Any]]:
    """Fetch every trade in ``[start_ms, end_ms]`` by paginating ascending in time.

    Deribit's only forward cursor is the trade timestamp (milliseconds). When a
    page boundary lands inside a millisecond that holds more trades than fit in
    one page, we re-query *from* that millisecond (not ``+1``) so the remaining
    trades in it are not silently dropped; the ``seen`` set removes the
    re-fetched overlap. We only step past a millisecond once it yields no new
    trades, which also guards against an infinite loop in the (practically
    impossible) case of more than ``count`` trades in a single millisecond.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor = start_ms
    while cursor <= end_ms:
        res = await client.get_last_trades_by_currency_and_time(
            currency, cursor, end_ms, kind=kind, count=count, sorting="asc"
        )
        trades = res.get("trades", [])
        if not trades:
            break
        new = [t for t in trades if str(t["trade_id"]) not in seen]
        seen.update(str(t["trade_id"]) for t in new)
        out.extend(new)
        if not res.get("has_more", False):
            break
        last_ts = int(trades[-1]["timestamp"])
        cursor = last_ts if last_ts > cursor else last_ts + 1
    return out


def _f(value: Any) -> float | None:
    """Coerce an API numeric field to float, tolerating ``None``."""
    return None if value is None else float(value)


def _build_frame(trades: list[dict[str, Any]], kind: str) -> pl.DataFrame:
    """Turn raw trade dicts into a validated, deduped Parquet-ready frame."""
    parsed: dict[str, ParsedInstrument] = {}

    def _parse(name: str) -> ParsedInstrument:
        p = parsed.get(name)
        if p is None:
            p = parse_instrument(name)
            parsed[name] = p
        return p

    if kind == "option":
        rows = [
            {
                "ts": int(t["timestamp"]),
                "instrument": t["instrument_name"],
                "expiry": _ms(p.expiry) if (p := _parse(t["instrument_name"])).expiry else None,
                "strike": p.strike,
                "cp": p.cp,
                "price_btc": _f(t.get("price")),
                "iv": _f(t.get("iv")),
                "amount": _f(t.get("amount")),
                "index_price": _f(t.get("index_price")),
                "trade_id": str(t["trade_id"]),
                "block_flag": "block_trade_id" in t,
                "source": "deribit_history",
            }
            for t in trades
        ]
        df = pl.DataFrame(rows).with_columns(
            pl.from_epoch("ts", time_unit="ms").dt.replace_time_zone("UTC"),
            pl.from_epoch("expiry", time_unit="ms").dt.replace_time_zone("UTC"),
        )
        df = df.unique(subset=["trade_id"], keep="first").sort("ts")
        return validate(df, TRADES_OPTIONS)

    rows = [
        {
            "ts": int(t["timestamp"]),
            "instrument": t["instrument_name"],
            "price": _f(t.get("price")),
            "amount": _f(t.get("amount")),
            "index_price": _f(t.get("index_price")),
            "trade_id": str(t["trade_id"]),
            "block_flag": "block_trade_id" in t,
            "source": "deribit_history",
        }
        for t in trades
    ]
    df = pl.DataFrame(rows).with_columns(
        pl.from_epoch("ts", time_unit="ms").dt.replace_time_zone("UTC")
    )
    df = df.unique(subset=["trade_id"], keep="first").sort("ts")
    return validate(df, TRADES_FUTURES)


def _write_partition(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd")


async def backfill_kind(
    client: HistoryClient,
    cfg: DataConfig,
    kind: str,
    start_date: date,
    end_date: date,
) -> int:
    """Backfill one trade kind month by month. Returns total trades written."""
    stream = _STREAM[kind]
    out_dir = cfg.raw_table_dir(stream)
    ckpt = load_checkpoint(cfg, stream)
    completed: set[str] = set(ckpt.get("completed_months", []))

    months = list(iter_months(start_date, end_date))
    total = 0
    for i, (m_start, m_next, label) in enumerate(months):
        is_last = i == len(months) - 1
        if label in completed and not is_last:
            continue
        trades = await fetch_trades_window(
            client, cfg.currency, kind, _ms(m_start), _ms(m_next) - 1, count=cfg.page_count
        )
        if trades:
            df = _build_frame(trades, kind)
            _write_partition(df, out_dir / f"month={label}" / "part.parquet")
            total += df.height
            log.info("%s %s: %d trades", stream, label, df.height)
        else:
            log.info("%s %s: no trades", stream, label)
        if not is_last:
            completed.add(label)
            save_checkpoint(cfg, stream, {"completed_months": sorted(completed)})
    return total


async def run_backfill(
    cfg: DataConfig,
    *,
    kinds: tuple[str, ...] = ("option", "future"),
    start: str | None = None,
    end: str | None = None,
) -> None:
    """Run the trades backfill for the requested kinds (default options+futures)."""
    start_date = date.fromisoformat(start) if start else date.fromisoformat(cfg.history_start)
    end_date = date.fromisoformat(end) if end else datetime.now(UTC).date()

    async with httpx.AsyncClient(timeout=cfg.request_timeout_s) as http:
        client = HistoryClient(
            http,
            base_url=cfg.history_base_url,
            rate_limit_rps=cfg.rate_limit_rps,
            max_retries=cfg.max_retries,
            retry_backoff_s=cfg.retry_backoff_s,
        )
        for kind in kinds:
            log.info("backfilling %s trades %s -> %s", kind, start_date, end_date)
            total = await backfill_kind(client, cfg, kind, start_date, end_date)
            log.info("done %s: %d trades total", kind, total)
