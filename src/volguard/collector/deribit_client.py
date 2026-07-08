"""Minimal async Deribit public-API client used by the live collector.

Only the read-only public endpoints needed for surface snapshots are wrapped.
Uses ``httpx.AsyncClient`` (a core dep) so the collector needs no extra install
and is testable with ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

_SERVER_ERROR_MIN = 500
_HTTP_TOO_MANY_REQUESTS = 429

# Deribit JSON-RPC error codes that are transient and safe to retry on a
# read-only public request (see https://docs.deribit.com/articles/errors):
#   10028 too_many_requests          — request rate exceeded
#   10066 too_many_concurrent_requests — too many un-executed public requests
#   10040 retry                      — can't be processed now, should be retried
#   13028 temporarily_unavailable    — service not responding / slow
#   13503 unavailable                — method temporarily unavailable
_RETRYABLE_RPC_CODES = frozenset({10028, 10066, 10040, 13028, 13503})


class DeribitError(RuntimeError):
    """Raised when the Deribit API returns a JSON-RPC error payload."""


class DeribitRateLimitError(DeribitError):
    """A transient/rate-limit JSON-RPC error that should be retried."""

    def __init__(self, message: str, *, wait_s: float | None = None) -> None:
        super().__init__(message)
        self.wait_s = wait_s


class DeribitClient:
    """Thin wrapper over Deribit's public JSON-RPC-over-REST endpoints.

    Args:
        client: an ``httpx.AsyncClient`` (injected for testability).
        base_url: API base, e.g. ``https://www.deribit.com/api/v2``.
        max_retries: attempts per call before giving up.
        retry_backoff_s: base for exponential backoff between retries.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str = "https://www.deribit.com/api/v2",
        max_retries: int = 4,
        retry_backoff_s: float = 2.0,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s

    def _parse_rpc_error(self, method: str, error: Any) -> DeribitError:
        """Map a JSON-RPC ``error`` payload to a (possibly retryable) exception.

        Deribit sometimes attaches a ``data.wait`` (seconds) hint on rate-limit
        errors; honor it when present so we back off for the advised duration.
        """
        code = error.get("code") if isinstance(error, dict) else None
        msg = f"{method}: {error}"
        if code in _RETRYABLE_RPC_CODES:
            wait_s: float | None = None
            data = error.get("data") if isinstance(error, dict) else None
            if isinstance(data, dict) and isinstance(data.get("wait"), int | float):
                wait_s = float(data["wait"])
            return DeribitRateLimitError(msg, wait_s=wait_s)
        return DeribitError(msg)

    async def _get(self, method: str, params: dict[str, Any]) -> Any:
        """Call a public method, retrying transport, 5xx/429, and rate-limit errors.

        Terminal JSON-RPC errors (bad params, unknown instrument, etc.) are
        raised immediately; only transient failures enter the backoff loop so a
        rate-limited ticker is not silently dropped from a snapshot.
        """
        url = f"{self._base_url}/public/{method}"
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            wait_hint: float | None = None
            try:
                resp = await self._client.get(url, params=params)
                if resp.status_code >= _SERVER_ERROR_MIN or (
                    resp.status_code == _HTTP_TOO_MANY_REQUESTS
                ):
                    resp.raise_for_status()
                payload = resp.json()
                if "error" in payload:
                    exc = self._parse_rpc_error(method, payload["error"])
                    if isinstance(exc, DeribitRateLimitError):
                        wait_hint = exc.wait_s
                        raise exc
                    raise exc  # terminal — propagate without retry
                return payload["result"]
            except (httpx.TransportError, httpx.HTTPStatusError, DeribitRateLimitError) as e:
                last_exc = e
                if attempt < self._max_retries - 1:
                    backoff = self._retry_backoff_s * (2**attempt)
                    await asyncio.sleep(wait_hint if wait_hint is not None else backoff)
        raise DeribitError(f"{method} failed after {self._max_retries} attempts") from last_exc

    async def get_instruments(
        self, currency: str, kind: str = "option", expired: bool = False
    ) -> list[dict[str, Any]]:
        """List instruments for a currency (default: live options)."""
        return await self._get(
            "get_instruments",
            {"currency": currency, "kind": kind, "expired": str(expired).lower()},
        )

    async def get_book_summary_by_currency(
        self, currency: str, kind: str = "option"
    ) -> list[dict[str, Any]]:
        """One-shot book summary for every instrument of a kind (bid/ask/OI/vol)."""
        return await self._get("get_book_summary_by_currency", {"currency": currency, "kind": kind})

    async def ticker(self, instrument_name: str) -> dict[str, Any]:
        """Full ticker for one instrument (mark IV, greeks, best bid/ask)."""
        return await self._get("ticker", {"instrument_name": instrument_name})

    async def get_index_price(self, index_name: str) -> dict[str, Any]:
        """Spot index price, e.g. ``btc_usd``."""
        return await self._get("get_index_price", {"index_name": index_name})
