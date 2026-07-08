"""Live collector: poll Deribit every N seconds and write NDJSON snapshots.

One snapshot = one book-summary call for all option instruments (cheap, single
request) plus per-instrument tickers for the most liquid subset (greeks + mark
IV that book_summary omits), plus the underlying index price. Each snapshot is
appended as one JSON line to a date-partitioned file under ``out_dir``; a daily
systemd timer runs ``rclone`` to sync completed files to R2/B2.

Designed to be crash-tolerant: a failed poll is logged and skipped, the loop
continues, and files are append-only so a restart never loses prior snapshots.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from volguard.collector.deribit_client import DeribitClient
from volguard.config import CollectorConfig

log = logging.getLogger("volguard.collector")


def _liquid_instruments(book_summary: Iterable[dict[str, Any]], limit: int) -> list[str]:
    """Pick the ``limit`` most-liquid instruments by 24h volume."""
    ranked = sorted(
        book_summary,
        key=lambda row: row.get("volume") or 0.0,
        reverse=True,
    )
    return [row["instrument_name"] for row in ranked[:limit]]


async def collect_snapshot(client: DeribitClient, cfg: CollectorConfig) -> dict[str, Any]:
    """Build a single point-in-time snapshot dict (no disk I/O)."""
    ts = datetime.now(UTC).isoformat()
    book = await client.get_book_summary_by_currency(cfg.currency, kind="option")
    names = _liquid_instruments(book, cfg.max_ticker_instruments)

    # Fetch tickers concurrently but bounded, so we don't hammer the API.
    sem = asyncio.Semaphore(10)

    async def _one(name: str) -> dict[str, Any] | None:
        async with sem:
            try:
                return await client.ticker(name)
            except Exception:  # one bad instrument must not kill the poll
                log.warning("ticker failed for %s", name, exc_info=True)
                return None

    tickers = [t for t in await asyncio.gather(*(_one(n) for n in names)) if t is not None]
    index = await client.get_index_price(f"{cfg.currency.lower()}_usd")

    return {
        "snap_ts": ts,
        "currency": cfg.currency,
        "index_price": index.get("index_price"),
        "book_summary": book,
        "tickers": tickers,
    }


def write_snapshot(snapshot: dict[str, Any], out_dir: Path) -> Path:
    """Append a snapshot as one NDJSON line to a UTC-date-partitioned file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    day = snapshot["snap_ts"][:10]  # YYYY-MM-DD
    path = out_dir / f"tickers_{day}.ndjson"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
    return path


async def run_forever(cfg: CollectorConfig) -> None:
    """Poll on a fixed cadence until interrupted. Errors are logged, not fatal."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    async with httpx.AsyncClient(timeout=cfg.request_timeout_s) as http:
        client = DeribitClient(
            http,
            base_url=cfg.base_url,
            max_retries=cfg.max_retries,
            retry_backoff_s=cfg.retry_backoff_s,
        )
        log.info("collector started: every %ss -> %s", cfg.poll_seconds, cfg.out_dir)
        while True:
            started = asyncio.get_event_loop().time()
            try:
                snap = await collect_snapshot(client, cfg)
                path = write_snapshot(snap, cfg.out_dir)
                log.info(
                    "snapshot ok: %d instruments, %d tickers -> %s",
                    len(snap["book_summary"]),
                    len(snap["tickers"]),
                    path.name,
                )
            except Exception:  # keep the loop alive across failures
                log.exception("poll failed; will retry next cycle")
            # Sleep the remainder of the cadence, accounting for poll duration.
            elapsed = asyncio.get_event_loop().time() - started
            await asyncio.sleep(max(1.0, cfg.poll_seconds - elapsed))
