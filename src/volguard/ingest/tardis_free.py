"""Tardis.dev free-sample option-chain downloader (no API key).

Tardis publishes the full tick-level Deribit options chain for the **first day
of every month** for free at

    https://datasets.tardis.dev/v1/deribit/options_chain/YYYY/MM/01/OPTIONS.csv.gz

These ~85 days (since 2019-04) are used downstream to (a) validate our
trade-based surfaces against quote-based marks and (b) fit the bid/ask spread
model behind the transaction-cost simulator.  We fetch every available month
from ``tardis_start`` onward, tolerate months that 404, skip already-downloaded
files, and convert each gzip CSV to a schema-validated Parquet partition:

    data/raw/tardis_chain/date=YYYY-MM-01/part.parquet
"""

from __future__ import annotations

import gzip
import logging
import shutil
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import polars as pl

from volguard.config import DataConfig
from volguard.ingest.schemas import TARDIS_CHAIN, validate

log = logging.getLogger("volguard.ingest")

_HTTP_NOT_FOUND = 404
_DECEMBER = 12
# Recent free days are ~1 GB gzipped / tens of millions of rows, so convert by
# streaming rather than materializing the whole CSV in memory.
_CONTRACT_SAMPLE_ROWS = 1000
# Read the numeric columns as fixed dtypes (via schema_overrides) so partitions
# are consistent across days regardless of what CSV inference would pick per day.
# Amount/price/greek columns are integer-valued in the header sample of some
# days (e.g. `bid_amount` = 1, 5, 100) but carry fractional values deeper in the
# file, so inference locks them to i64 and then fails on the first float. Pinning
# every numeric column to its true dtype avoids that per-day drift entirely.
_CONTRACT_CASTS = {
    "timestamp": pl.Int64,
    "local_timestamp": pl.Int64,
    "expiration": pl.Int64,
    "strike_price": pl.Float64,
    "open_interest": pl.Float64,
    "last_price": pl.Float64,
    "bid_price": pl.Float64,
    "bid_amount": pl.Float64,
    "bid_iv": pl.Float64,
    "ask_price": pl.Float64,
    "ask_amount": pl.Float64,
    "ask_iv": pl.Float64,
    "mark_price": pl.Float64,
    "mark_iv": pl.Float64,
    "underlying_price": pl.Float64,
    "delta": pl.Float64,
    "gamma": pl.Float64,
    "vega": pl.Float64,
    "theta": pl.Float64,
    "rho": pl.Float64,
}


def iter_first_days(start: date, end: date) -> Iterator[date]:
    """Yield the first-of-month date for every month in ``[start, end]``."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield date(y, m, 1)
        y, m = (y + 1, 1) if m == _DECEMBER else (y, m + 1)


def _url(cfg: DataConfig, d: date) -> str:
    return f"{cfg.tardis_base_url.rstrip('/')}/{d.year:04d}/{d.month:02d}/01/OPTIONS.csv.gz"


def _day_dir(cfg: DataConfig, d: date) -> Path:
    return cfg.raw_table_dir("tardis_chain") / f"date={d.isoformat()}"


def download_free_day(client: httpx.Client, cfg: DataConfig, d: date) -> Path | None:
    """Download one free day's gzip chain. Returns its path, or ``None`` on 404.

    Streams the response to disk in chunks: recent free days are ~1 GB gzipped,
    so buffering the whole body in memory would risk exhausting RAM.
    """
    gz_path = _day_dir(cfg, d) / "OPTIONS.csv.gz"
    if gz_path.exists():
        return gz_path
    gz_path.parent.mkdir(parents=True, exist_ok=True)
    # Download to a temp path and atomically rename only on success, so a
    # stream that fails mid-write never leaves a truncated .gz that the
    # exists() fast-path would later feed to the (corrupt) gzip converter.
    tmp_path = gz_path.with_suffix(".gz.part")
    try:
        with client.stream("GET", _url(cfg, d)) as resp:
            if resp.status_code == _HTTP_NOT_FOUND:
                return None
            resp.raise_for_status()
            with tmp_path.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
        tmp_path.replace(gz_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return gz_path


def gz_to_parquet(gz_path: Path, out_path: Path) -> int:
    """Convert a gzip Tardis CSV to a Parquet partition (streamed); return rows.

    The contract is checked cheaply on a header sample (the full daily file can
    be tens of millions of rows), then the conversion streams via
    ``scan_csv -> sink_parquet`` so peak memory stays bounded regardless of day.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_path.parent / "_decompressed.csv"
    # Sink to a temp path and atomically replace, so an interrupted/failed
    # sink_parquet never leaves a partial part.parquet that the exists() check
    # in run_tardis would treat as a successfully converted (but corrupt) day.
    tmp_parquet = out_path.with_suffix(".parquet.part")
    try:
        with gzip.open(gz_path, "rb") as fin, tmp_csv.open("wb") as fout:
            shutil.copyfileobj(fin, fout)
        # Force the contract columns to stable dtypes at *read* time via
        # schema_overrides: a column that is empty/integer-only in the inference
        # sample but float later would otherwise lock the scan schema to a type
        # that fails (or silently shifts) partway through the file, and a
        # post-scan cast cannot recover from that.
        validate(
            pl.read_csv(tmp_csv, n_rows=_CONTRACT_SAMPLE_ROWS, schema_overrides=_CONTRACT_CASTS),
            TARDIS_CHAIN,
        )
        pl.scan_csv(tmp_csv, schema_overrides=_CONTRACT_CASTS).sink_parquet(
            tmp_parquet, compression="zstd"
        )
        tmp_parquet.replace(out_path)
    finally:
        tmp_csv.unlink(missing_ok=True)
        tmp_parquet.unlink(missing_ok=True)
    return pl.scan_parquet(out_path).select(pl.len()).collect().item()


def run_tardis(cfg: DataConfig, *, start: str | None = None, end: str | None = None) -> None:
    """Download + convert every available Tardis free day in the range."""
    start_date = date.fromisoformat(start) if start else date.fromisoformat(cfg.tardis_start)
    end_date = date.fromisoformat(end) if end else datetime.now(UTC).date()

    downloaded = 0
    with httpx.Client(timeout=cfg.request_timeout_s, follow_redirects=True) as client:
        for d in iter_first_days(start_date, end_date):
            parquet_path = _day_dir(cfg, d) / "part.parquet"
            # The Parquet part is the only artifact downstream stages read, so a
            # day that is already converted needs neither a re-download nor a
            # re-convert -- skip it before spending bandwidth on the ~1 GB gzip.
            if parquet_path.exists():
                log.info("tardis %s: already converted", d)
                downloaded += 1
                continue
            gz_path = download_free_day(client, cfg, d)
            if gz_path is None:
                log.info("tardis %s: unavailable (404)", d)
                continue
            rows = gz_to_parquet(gz_path, parquet_path)
            # The gzip is a transient download artifact; once the Parquet part is
            # written it is pure dead weight (~1 GB/day), so drop it to keep the
            # raw layer Parquet-only. It is re-downloadable from Tardis if needed.
            gz_path.unlink(missing_ok=True)
            log.info("tardis %s: %d rows", d, rows)
            downloaded += 1
    log.info("tardis done: %d free days available", downloaded)
