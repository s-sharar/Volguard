"""Throttled Deribit *history* API client for backfill pulls.

The live collector's :class:`~volguard.collector.deribit_client.DeribitClient`
already implements the retry / backoff / rate-limit-aware JSON-RPC loop we want.
Rather than fork that logic (and risk drift with the deployed VPS code) we
subclass it here to add two backfill-only concerns:

* a client-side request throttle (Deribit's public history endpoints tolerate
  only a handful of requests per second before returning ``10028``), and
* the read-only history/underlying methods the collector never needs.

``history.deribit.com`` mirrors the trading API but serves the full trade
history since platform launch, which is what the 2021->now backfill walks.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from volguard.collector.deribit_client import DeribitClient

HISTORY_BASE_URL = "https://history.deribit.com/api/v2"


class _Throttle:
    """Simple async rate limiter enforcing a minimum spacing between calls.

    A single monotonic "next allowed time" cursor guarded by a lock keeps
    concurrent callers globally under ``rate_per_s`` without a background task.
    """

    def __init__(self, rate_per_s: float) -> None:
        self._min_interval = 1.0 / rate_per_s if rate_per_s > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        if self._min_interval <= 0.0:
            return
        async with self._lock:
            loop = asyncio.get_event_loop()
            now = loop.time()
            wait = self._next_at - now
            if wait > 0.0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_at = now + self._min_interval


class HistoryClient(DeribitClient):
    """Deribit history-API client: throttled + backfill read methods."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str = HISTORY_BASE_URL,
        rate_limit_rps: float = 5.0,
        max_retries: int = 5,
        retry_backoff_s: float = 1.0,
    ) -> None:
        super().__init__(
            client,
            base_url=base_url,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        self._throttle = _Throttle(rate_limit_rps)

    async def _get(self, method: str, params: dict[str, Any]) -> Any:
        # Space every request (including retries) under the configured rate so a
        # long backfill never trips Deribit's per-connection limiter.
        await self._throttle.acquire()
        return await super()._get(method, params)

    async def get_last_trades_by_currency_and_time(
        self,
        currency: str,
        start_timestamp: int,
        end_timestamp: int,
        *,
        kind: str = "option",
        count: int = 1000,
        sorting: str = "asc",
    ) -> dict[str, Any]:
        """One page of trades in ``[start, end]`` ms, ascending by default.

        Returns the raw result dict with ``trades`` and ``has_more`` keys.
        """
        return await self._get(
            "get_last_trades_by_currency_and_time",
            {
                "currency": currency,
                "kind": kind,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "count": count,
                "sorting": sorting,
            },
        )

    async def get_tradingview_chart_data(
        self,
        instrument_name: str,
        start_timestamp: int,
        end_timestamp: int,
        *,
        resolution: str = "60",
    ) -> dict[str, Any]:
        """OHLCV candles for an instrument (resolution in minutes, ``"1D"`` etc)."""
        return await self._get(
            "get_tradingview_chart_data",
            {
                "instrument_name": instrument_name,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "resolution": resolution,
            },
        )

    async def get_volatility_index_data(
        self,
        currency: str,
        start_timestamp: int,
        end_timestamp: int,
        *,
        resolution: str = "3600",
    ) -> dict[str, Any]:
        """DVOL candles for a currency (resolution in seconds, ``"1D"`` etc)."""
        return await self._get(
            "get_volatility_index_data",
            {
                "currency": currency,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "resolution": resolution,
            },
        )

    async def get_funding_rate_history(
        self,
        instrument_name: str,
        start_timestamp: int,
        end_timestamp: int,
    ) -> list[dict[str, Any]]:
        """Hourly perpetual funding-rate history."""
        return await self._get(
            "get_funding_rate_history",
            {
                "instrument_name": instrument_name,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
            },
        )

    async def get_delivery_prices(
        self, index_name: str, *, offset: int = 0, count: int = 100
    ) -> dict[str, Any]:
        """Historical daily settlement (delivery) prices for an index."""
        return await self._get(
            "get_delivery_prices",
            {"index_name": index_name, "offset": offset, "count": count},
        )
