"""Tardis free-day downloader tests: gz->parquet, skip-if-exists, 404 — no network."""

from __future__ import annotations

import gzip
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from volguard.config import DataConfig
from volguard.ingest.tardis_free import (
    download_free_day,
    gz_to_parquet,
    iter_first_days,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tardis_sample.csv"


def _gz_bytes() -> bytes:
    return gzip.compress(FIXTURE.read_bytes())


def test_iter_first_days() -> None:
    days = list(iter_first_days(date(2019, 4, 10), date(2019, 7, 1)))
    assert days == [date(2019, 4, 1), date(2019, 5, 1), date(2019, 6, 1), date(2019, 7, 1)]


def test_gz_to_parquet_roundtrip(tmp_path) -> None:
    gz_path = tmp_path / "OPTIONS.csv.gz"
    gz_path.write_bytes(_gz_bytes())
    out = tmp_path / "part.parquet"
    rows = gz_to_parquet(gz_path, out)

    assert rows == 8
    df = pl.read_parquet(out)
    assert {"symbol", "strike_price", "mark_iv", "underlying_price"} <= set(df.columns)
    assert df["strike_price"].dtype == pl.Float64


def test_download_free_day_writes_and_skips(tmp_path) -> None:
    requests = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        requests["n"] += 1
        return httpx.Response(200, content=_gz_bytes())

    cfg = DataConfig(raw_dir=tmp_path)
    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    p1 = download_free_day(client, cfg, date(2022, 4, 1))
    assert p1 is not None
    assert p1.exists()
    # Second call short-circuits on the existing file (no new request).
    p2 = download_free_day(client, cfg, date(2022, 4, 1))
    assert p2 == p1
    assert requests["n"] == 1


def test_gz_to_parquet_handles_late_float_column(tmp_path) -> None:
    # A contract column (underlying_price) is empty/integer-only in the early
    # rows and float later. Without read-time schema overrides the scan would
    # lock it to an int/null type and fail (or shift) mid-file.
    header = "symbol,timestamp,type,strike_price,expiration,mark_iv,underlying_price"
    early = [
        f"BTC-1APR22-4{i}000-C,1648771200000,call,4{i}000,1648800000000,50.0," for i in range(3)
    ]
    late = ["BTC-1APR22-50000-C,1648771200000,call,50000,1648800000000,55.0,45123.75"]
    csv = "\n".join([header, *early, *late]) + "\n"
    gz_path = tmp_path / "OPTIONS.csv.gz"
    gz_path.write_bytes(gzip.compress(csv.encode()))

    out = tmp_path / "part.parquet"
    rows = gz_to_parquet(gz_path, out)
    assert rows == 4
    df = pl.read_parquet(out)
    assert df["underlying_price"].dtype == pl.Float64
    assert df["strike_price"].dtype == pl.Float64
    assert df["underlying_price"].to_list()[-1] == 45123.75


def test_download_free_day_tolerates_404(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    cfg = DataConfig(raw_dir=tmp_path)
    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    assert download_free_day(client, cfg, date(2018, 1, 1)) is None
