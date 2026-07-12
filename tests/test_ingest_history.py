"""Deribit trades-backfill tests: parsing, pagination, resume, schema — no network."""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from volguard.config import DataConfig
from volguard.ingest.client import HistoryClient
from volguard.ingest.deribit_history import (
    _build_frame,
    backfill_kind,
    fetch_trades_window,
    iter_months,
    load_checkpoint,
    save_checkpoint,
)
from volguard.ingest.schemas import parse_instrument

BASE = "https://history.test/api/v2"
_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _client(handler: httpx.MockTransport) -> HistoryClient:
    http = httpx.AsyncClient(transport=handler, timeout=5.0)
    return HistoryClient(http, base_url=BASE, rate_limit_rps=0.0, retry_backoff_s=0.0)


# --- instrument parser ----------------------------------------------------


def test_parse_option_instrument() -> None:
    p = parse_instrument("BTC-27JUN25-100000-C")
    assert p.kind == "option"
    assert p.cp == "C"
    assert p.strike == 100000.0
    assert p.expiry == datetime(2025, 6, 27, 8, 0, tzinfo=UTC)


def test_parse_future_and_perpetual() -> None:
    fut = parse_instrument("BTC-27JUN25")
    assert fut.kind == "future"
    assert fut.strike is None
    perp = parse_instrument("BTC-PERPETUAL")
    assert perp.kind == "perpetual"
    assert perp.expiry is None


def test_parse_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="unrecognized"):
        parse_instrument("BTC-NOTADATE-1-C")


@given(
    day=st.integers(min_value=1, max_value=28),
    month=st.sampled_from(_MONTHS),
    year=st.integers(min_value=19, max_value=30),
    strike=st.integers(min_value=1000, max_value=200000),
    cp=st.sampled_from(["C", "P"]),
)
def test_parse_option_roundtrip(day: int, month: str, year: int, strike: int, cp: str) -> None:
    name = f"BTC-{day}{month}{year}-{strike}-{cp}"
    p = parse_instrument(name)
    assert p.cp == cp
    assert p.strike == float(strike)
    assert p.expiry is not None
    assert p.expiry.year == 2000 + year


# --- month iteration ------------------------------------------------------


def test_iter_months_spans_inclusive() -> None:
    months = [label for _, _, label in iter_months(date(2021, 11, 15), date(2022, 2, 1))]
    assert months == ["2021-11", "2021-12", "2022-01", "2022-02"]


# --- frame building -------------------------------------------------------


def _opt_trade(ts: int, tid: str, *, block: bool = False) -> dict:
    t = {
        "timestamp": ts,
        "instrument_name": "BTC-1APR22-45000-C",
        "price": 0.05,
        "iv": 56.0,
        "amount": 10.0,
        "index_price": 45000.0,
        "trade_id": tid,
    }
    if block:
        t["block_trade_id"] = "blk1"
    return t


def test_build_options_frame_dedupes_and_types() -> None:
    trades = [_opt_trade(1_648_771_200_000, "a"), _opt_trade(1_648_771_200_001, "a")]
    df = _build_frame(trades, "option")
    assert df.height == 1  # deduped on trade_id
    assert df.schema["ts"] == pl.Datetime(time_unit="ms", time_zone="UTC")
    assert df["cp"][0] == "C"
    assert df["strike"][0] == 45000.0


def test_build_options_frame_block_flag() -> None:
    df = _build_frame([_opt_trade(1_648_771_200_000, "a", block=True)], "option")
    assert df["block_flag"][0] is True


def test_build_futures_frame() -> None:
    trades = [
        {
            "timestamp": 1_648_771_200_000,
            "instrument_name": "BTC-PERPETUAL",
            "price": 45000.0,
            "amount": 100.0,
            "index_price": 45010.0,
            "trade_id": "f1",
        }
    ]
    df = _build_frame(trades, "future")
    assert df.height == 1
    assert "price" in df.columns
    assert "iv" not in df.columns


# --- pagination -----------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_trades_window_paginates() -> None:
    all_trades = [_opt_trade(1_648_771_200_000 + i, str(i)) for i in range(5)]

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["start_timestamp"])
        end = int(request.url.params["end_timestamp"])
        count = int(request.url.params["count"])
        window = [t for t in all_trades if start <= t["timestamp"] <= end]
        page = window[:count]
        return httpx.Response(
            200, json={"result": {"trades": page, "has_more": len(window) > count}}
        )

    client = _client(httpx.MockTransport(handler))
    out = await fetch_trades_window(
        client, "BTC", "option", 1_648_771_200_000, 1_648_771_200_010, count=2
    )
    assert len(out) == 5
    assert [t["trade_id"] for t in out] == ["0", "1", "2", "3", "4"]


@pytest.mark.asyncio
async def test_fetch_trades_window_preserves_same_ms_trades() -> None:
    # A page boundary splits a millisecond (three trades share ts T1 but count=3
    # only returns T0 + two of them). The extra T1 trade must not be skipped.
    t0, t1, t2 = 1_648_771_200_000, 1_648_771_200_001, 1_648_771_200_002
    all_trades = [
        _opt_trade(t0, "a"),
        _opt_trade(t1, "b"),
        _opt_trade(t1, "c"),
        _opt_trade(t1, "d"),
        _opt_trade(t2, "e"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["start_timestamp"])
        end = int(request.url.params["end_timestamp"])
        count = int(request.url.params["count"])
        window = [t for t in all_trades if start <= t["timestamp"] <= end]
        page = window[:count]
        return httpx.Response(
            200, json={"result": {"trades": page, "has_more": len(window) > count}}
        )

    client = _client(httpx.MockTransport(handler))
    out = await fetch_trades_window(client, "BTC", "option", t0, t2, count=3)
    assert sorted(t["trade_id"] for t in out) == ["a", "b", "c", "d", "e"]


@pytest.mark.asyncio
async def test_fetch_trades_window_empty_terminates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": {"trades": [], "has_more": False}})

    client = _client(httpx.MockTransport(handler))
    out = await fetch_trades_window(client, "BTC", "option", 0, 10, count=100)
    assert out == []


# --- backfill orchestration + resume --------------------------------------


@pytest.mark.asyncio
async def test_backfill_kind_writes_and_checkpoints(tmp_path) -> None:
    # One trade in Jan 2022, one in Feb 2022; range ends in Feb (the last month).
    jan = int(datetime(2022, 1, 15, tzinfo=UTC).timestamp() * 1000)
    feb = int(datetime(2022, 2, 15, tzinfo=UTC).timestamp() * 1000)
    trades = [_opt_trade(jan, "jan"), _opt_trade(feb, "feb")]

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["start_timestamp"])
        end = int(request.url.params["end_timestamp"])
        window = [t for t in trades if start <= t["timestamp"] <= end]
        return httpx.Response(200, json={"result": {"trades": window, "has_more": False}})

    cfg = DataConfig(raw_dir=tmp_path, rate_limit_rps=0.0)
    client = _client(httpx.MockTransport(handler))
    total = await backfill_kind(client, cfg, "option", date(2022, 1, 1), date(2022, 2, 28))

    assert total == 2
    assert (tmp_path / "trades_options" / "month=2022-01" / "part.parquet").exists()
    assert (tmp_path / "trades_options" / "month=2022-02" / "part.parquet").exists()
    # Only the non-final month is marked complete; the last month is always re-pulled.
    ckpt = load_checkpoint(cfg, "trades_options")
    assert ckpt["completed_months"] == ["2022-01"]


@pytest.mark.asyncio
async def test_backfill_kind_skips_completed_months(tmp_path) -> None:
    calls: list[tuple[int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["start_timestamp"])
        end = int(request.url.params["end_timestamp"])
        calls.append((start, end))
        return httpx.Response(200, json={"result": {"trades": [], "has_more": False}})

    cfg = DataConfig(raw_dir=tmp_path, rate_limit_rps=0.0)
    # Pre-seed a checkpoint marking Jan complete.
    save_checkpoint(cfg, "trades_options", {"completed_months": ["2022-01"]})
    client = _client(httpx.MockTransport(handler))
    await backfill_kind(client, cfg, "option", date(2022, 1, 1), date(2022, 2, 28))

    # Jan is skipped: the only fetched window starts in Feb.
    assert len(calls) == 1
    feb_start = int(datetime(2022, 2, 1, tzinfo=UTC).timestamp() * 1000)
    assert calls[0][0] == feb_start
