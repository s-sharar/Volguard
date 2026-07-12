"""Underlying-data tests: OHLC/DVOL/funding/instruments frames + delivery pagination."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import polars as pl
import pytest

import volguard.ingest.underlying as underlying_mod
from volguard.ingest.client import HistoryClient
from volguard.ingest.underlying import (
    _dvol_frame,
    _funding_frame,
    _instruments_frame,
    _ohlc_frame,
    fetch_delivery,
    fetch_dvol,
    fetch_ohlc,
)

BASE = "https://history.test/api/v2"
_TS = pl.Datetime(time_unit="ms", time_zone="UTC")


def _client(handler: httpx.MockTransport) -> HistoryClient:
    http = httpx.AsyncClient(transport=handler, timeout=5.0)
    return HistoryClient(http, base_url=BASE, rate_limit_rps=0.0, retry_backoff_s=0.0)


def test_ohlc_frame_maps_arrays() -> None:
    result = {
        "ticks": [1_648_771_200_000, 1_648_774_800_000],
        "open": [45000.0, 45100.0],
        "high": [45200.0, 45300.0],
        "low": [44900.0, 45000.0],
        "close": [45100.0, 45250.0],
        "volume": [12.0, 8.0],
        "status": "ok",
    }
    df = _ohlc_frame(result, "BTC-PERPETUAL")
    assert df is not None
    assert df.height == 2
    assert df.schema["ts"] == _TS
    assert df["instrument"][0] == "BTC-PERPETUAL"
    assert df["close"][1] == 45250.0


def test_ohlc_frame_empty_returns_none() -> None:
    assert _ohlc_frame({"ticks": [], "status": "no_data"}, "BTC-PERPETUAL") is None


@pytest.mark.asyncio
async def test_fetch_ohlc_multi_chunk_has_no_gaps(monkeypatch) -> None:
    # Shrink the cap so a small range spans several chunks, and have the handler
    # enforce it strictly (returning at most `cap` inclusive bars per request).
    monkeypatch.setattr(underlying_mod, "_MAX_BARS", 4)
    hour = 3_600_000
    start = 1_609_459_200_000
    end = start + 20 * hour  # 21 candles

    def handler(request: httpx.Request) -> httpx.Response:
        s = int(request.url.params["start_timestamp"])
        e = int(request.url.params["end_timestamp"])
        grid = [t for t in range(start, end + 1, hour) if s <= t <= e]
        capped = grid[: underlying_mod._MAX_BARS]  # server trims to the cap
        return httpx.Response(
            200,
            json={
                "result": {
                    "ticks": capped,
                    "open": [1.0] * len(capped),
                    "high": [1.0] * len(capped),
                    "low": [1.0] * len(capped),
                    "close": [1.0] * len(capped),
                    "volume": [1.0] * len(capped),
                    "status": "ok",
                }
            },
        )

    client = _client(httpx.MockTransport(handler))
    df = await fetch_ohlc(client, "BTC-PERPETUAL", start, end, resolution="60")
    assert df is not None
    expected = list(range(start, end + 1, hour))
    got = [int(v.timestamp() * 1000) for v in df["ts"].to_list()]
    assert got == expected  # every candle present, no gaps, sorted, deduped


def test_dvol_frame() -> None:
    data = [
        [1_648_771_200_000, 60.0, 62.0, 59.0, 61.0],
        [1_648_774_800_000, 61.0, 63.0, 60.0, 62.5],
    ]
    df = _dvol_frame(data, "BTC")
    assert df is not None
    assert df.height == 2
    assert df["currency"][0] == "BTC"
    assert df["close"][1] == 62.5


def test_funding_frame() -> None:
    records = [
        {
            "timestamp": 1_648_771_200_000,
            "interest_1h": 0.00001,
            "interest_8h": 0.00008,
            "index_price": 45000.0,
        }
    ]
    df = _funding_frame(records, "BTC-PERPETUAL")
    assert df is not None
    assert df["instrument"][0] == "BTC-PERPETUAL"
    assert df.schema["ts"] == _TS


def test_instruments_frame_option_and_future() -> None:
    records = [
        {
            "instrument_name": "BTC-1APR22-45000-C",
            "kind": "option",
            "expiration_timestamp": 1_648_800_000_000,
            "strike": 45000.0,
            "option_type": "call",
            "creation_timestamp": 1_640_000_000_000,
        },
        {
            "instrument_name": "BTC-PERPETUAL",
            "kind": "future",
            "expiration_timestamp": 32_503_708_800_000,
            "creation_timestamp": 1_500_000_000_000,
        },
    ]
    df = _instruments_frame(records)
    assert df.height == 2
    opt = df.filter(pl.col("instrument") == "BTC-1APR22-45000-C")
    assert opt["cp"][0] == "C"
    assert opt["strike"][0] == 45000.0
    perp = df.filter(pl.col("instrument") == "BTC-PERPETUAL")
    assert perp["cp"][0] is None


@pytest.mark.asyncio
async def test_fetch_dvol_follows_continuation() -> None:
    # The endpoint caps each page and returns the most-recent bars first with a
    # `continuation` (next end_timestamp) for the older remainder. A single-page
    # reader would drop everything before the last page.
    hour = 3_600_000
    start = 1_609_459_200_000
    all_ts = [start + i * hour for i in range(5)]  # 5 hourly bars

    def handler(request: httpx.Request) -> httpx.Response:
        end = int(request.url.params["end_timestamp"])
        # Return up to 2 most-recent bars <= end, oldest-first within the page.
        window = [t for t in all_ts if t <= end]
        page_ts = window[-2:]
        rows = [[t, 60.0, 61.0, 59.0, 60.5] for t in page_ts]
        remaining = window[:-2]
        cont = remaining[-1] if remaining else None
        return httpx.Response(200, json={"result": {"data": rows, "continuation": cont}})

    client = _client(httpx.MockTransport(handler))
    df = await fetch_dvol(client, "BTC", start, all_ts[-1], resolution="3600")
    assert df is not None
    assert df.height == 5  # all bars recovered across pages, deduped + sorted
    assert df["ts"].is_sorted()


@pytest.mark.asyncio
async def test_fetch_delivery_paginates() -> None:
    rows = [{"date": f"2022-01-{d:02d}", "delivery_price": 40000.0 + d} for d in range(1, 6)]

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        count = int(request.url.params["count"])
        page = rows[offset : offset + count]
        return httpx.Response(200, json={"result": {"data": page, "records_total": len(rows)}})

    client = _client(httpx.MockTransport(handler))
    df = await fetch_delivery(client, "btc_usd")
    assert df is not None
    assert df.height == 5
    assert df["index"][0] == "btc_usd"


def test_instruments_expiry_is_utc_datetime() -> None:
    records = [
        {
            "instrument_name": "BTC-1APR22-45000-C",
            "kind": "option",
            "expiration_timestamp": 1_648_800_000_000,
            "strike": 45000.0,
            "option_type": "call",
            "creation_timestamp": 1_640_000_000_000,
        }
    ]
    df = _instruments_frame(records)
    assert df.schema["expiry"] == _TS
    assert df["expiry"][0] == datetime(2022, 4, 1, 8, 0, tzinfo=UTC)
