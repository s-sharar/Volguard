"""Collector tests: client parsing/retry and snapshot assembly, no network."""

from __future__ import annotations

import json

import httpx
import pytest

from volguard.collector.deribit_client import (
    DeribitClient,
    DeribitError,
    DeribitRateLimitError,
)
from volguard.collector.poller import (
    _liquid_instruments,
    collect_snapshot,
    write_snapshot,
)
from volguard.config import CollectorConfig

BASE = "https://test.deribit/api/v2"


def _make_client(handler: httpx.MockTransport, **kwargs) -> DeribitClient:
    http = httpx.AsyncClient(transport=handler, timeout=5.0)
    return DeribitClient(http, base_url=BASE, retry_backoff_s=0.0, **kwargs)


@pytest.mark.asyncio
async def test_get_unwraps_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": [{"instrument_name": "BTC-1"}]})

    client = _make_client(httpx.MockTransport(handler))
    out = await client.get_instruments("BTC")
    assert out == [{"instrument_name": "BTC-1"}]


@pytest.mark.asyncio
async def test_api_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "bad"}})

    client = _make_client(httpx.MockTransport(handler))
    with pytest.raises(DeribitError):
        await client.ticker("BTC-1")


@pytest.mark.asyncio
async def test_rate_limit_error_retries_then_succeeds() -> None:
    # 10028 too_many_requests is transient: retry, don't drop the call.
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(
                200, json={"error": {"code": 10028, "message": "too_many_requests"}}
            )
        return httpx.Response(200, json={"result": {"mark_iv": 55.0}})

    client = _make_client(httpx.MockTransport(handler), max_retries=5)
    out = await client.ticker("BTC-1")
    assert out["mark_iv"] == 55.0
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_rate_limit_exhausted_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"error": {"code": 10066, "message": "too_many_concurrent_requests"}}
        )

    client = _make_client(httpx.MockTransport(handler), max_retries=2)
    with pytest.raises(DeribitError):
        await client.ticker("BTC-1")


@pytest.mark.asyncio
async def test_http_429_retries() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(429)
        return httpx.Response(200, json={"result": {"index_price": 1.0}})

    client = _make_client(httpx.MockTransport(handler), max_retries=3)
    out = await client.get_index_price("btc_usd")
    assert out["index_price"] == 1.0
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_terminal_error_does_not_retry() -> None:
    # A non-retryable code (invalid instrument) must fail fast, not loop.
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"error": {"code": 10020, "message": "invalid_instrument"}})

    client = _make_client(httpx.MockTransport(handler), max_retries=5)
    with pytest.raises(DeribitError) as exc_info:
        await client.ticker("BOGUS")
    assert not isinstance(exc_info.value, DeribitRateLimitError)
    assert calls["n"] == 1  # no retries on a terminal error


@pytest.mark.asyncio
async def test_retry_then_succeed() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"result": {"index_price": 50000.0}})

    client = _make_client(httpx.MockTransport(handler), max_retries=5)
    out = await client.get_index_price("btc_usd")
    assert out["index_price"] == 50000.0
    assert calls["n"] == 3


def test_liquid_instruments_ranks_by_volume() -> None:
    book = [
        {"instrument_name": "A", "volume": 1.0},
        {"instrument_name": "B", "volume": 9.0},
        {"instrument_name": "C", "volume": None},
    ]
    assert _liquid_instruments(book, limit=2) == ["B", "A"]


@pytest.mark.asyncio
async def test_collect_snapshot_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("get_book_summary_by_currency"):
            return httpx.Response(
                200,
                json={"result": [{"instrument_name": "BTC-X", "volume": 5.0}]},
            )
        if request.url.path.endswith("ticker"):
            return httpx.Response(200, json={"result": {"mark_iv": 60.0}})
        if request.url.path.endswith("get_index_price"):
            return httpx.Response(200, json={"result": {"index_price": 50000.0}})
        return httpx.Response(404)

    client = _make_client(httpx.MockTransport(handler))
    cfg = CollectorConfig(base_url=BASE, max_ticker_instruments=10)
    snap = await collect_snapshot(client, cfg)

    assert snap["currency"] == "BTC"
    assert snap["index_price"] == 50000.0
    assert len(snap["book_summary"]) == 1
    assert len(snap["tickers"]) == 1
    assert "snap_ts" in snap


def test_write_snapshot_appends_ndjson(tmp_path) -> None:
    snap = {"snap_ts": "2026-07-07T08:05:00+00:00", "book_summary": [], "tickers": []}
    p1 = write_snapshot(snap, tmp_path)
    p2 = write_snapshot(snap, tmp_path)
    assert p1 == p2  # same UTC day -> same file
    lines = p1.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["snap_ts"] == snap["snap_ts"]
