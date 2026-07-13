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
    run_tardis,
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


def _mock_client_factory(handler):
    """A `httpx.Client(**kwargs)` replacement bound to a MockTransport.

    Captures the real ``httpx.Client`` up front so the returned factory does not
    re-enter the patched attribute (which would recurse infinitely).
    """
    real_client = httpx.Client

    def factory(**_kwargs):
        return real_client(transport=httpx.MockTransport(handler), timeout=5.0)

    return factory


def test_run_tardis_keeps_only_parquet(tmp_path, monkeypatch) -> None:
    # After a successful convert only the Parquet part should remain on disk —
    # the ~1 GB gzip is transient and must be deleted to keep the raw layer
    # Parquet-only (the sole artifact downstream stages read).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_gz_bytes())

    monkeypatch.setattr("volguard.ingest.tardis_free.httpx.Client", _mock_client_factory(handler))
    cfg = DataConfig(raw_dir=tmp_path)
    run_tardis(cfg, start="2022-04-01", end="2022-04-01")

    day_dir = tmp_path / "tardis_chain" / "date=2022-04-01"
    assert (day_dir / "part.parquet").exists()
    assert list(day_dir.glob("*.gz")) == []  # gzip cleaned up after conversion


def test_run_tardis_skips_converted_day_without_download(tmp_path, monkeypatch) -> None:
    # A day that already has a Parquet part must not trigger a re-download of the
    # gzip (bandwidth + disk waste); the parquet is the source of truth.
    requests = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        requests["n"] += 1
        return httpx.Response(200, content=_gz_bytes())

    monkeypatch.setattr("volguard.ingest.tardis_free.httpx.Client", _mock_client_factory(handler))
    cfg = DataConfig(raw_dir=tmp_path)
    # Seed an already-converted day.
    part = tmp_path / "tardis_chain" / "date=2022-04-01" / "part.parquet"
    part.parent.mkdir(parents=True)
    seed_gz = tmp_path / "seed.csv.gz"
    seed_gz.write_bytes(_gz_bytes())
    gz_to_parquet(seed_gz, part)

    run_tardis(cfg, start="2022-04-01", end="2022-04-01")
    assert requests["n"] == 0  # no download attempted for the converted day
