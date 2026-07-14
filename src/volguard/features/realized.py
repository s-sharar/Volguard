"""Leakage-safe daily OHLC aggregation and realized-volatility features."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import cast

import numpy as np
import polars as pl

from volguard.config import FeatureConfig

_SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True, slots=True)
class _DailyBar:
    open: float
    high: float
    low: float
    close: float
    source_ts: datetime


def _snap(day: date, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


def _daily_bar(
    ohlc: pl.DataFrame,
    resolution_minutes: int,
) -> _DailyBar | None:
    resolution = timedelta(minutes=resolution_minutes)
    usable = (
        ohlc.drop_nulls(["open", "high", "low", "close"])
        .filter(
            pl.all_horizontal(
                pl.col("open").is_finite(),
                pl.col("high").is_finite(),
                pl.col("low").is_finite(),
                pl.col("close").is_finite(),
                pl.col("open") > 0.0,
                pl.col("high") > 0.0,
                pl.col("low") > 0.0,
                pl.col("close") > 0.0,
                pl.col("high") >= pl.col("low"),
            )
        )
        .unique("ts", keep="last")
        .sort("ts")
    )
    expected_bars = round(_SECONDS_PER_DAY / resolution.total_seconds())
    if usable["ts"].n_unique() != expected_bars:
        return None
    return _DailyBar(
        open=float(cast(float, usable["open"][0])),
        high=float(cast(float, usable["high"].max())),
        low=float(cast(float, usable["low"].min())),
        close=float(cast(float, usable["close"][-1])),
        source_ts=cast(datetime, usable["available_ts"].max()),
    )


def _candles_by_snap_date(
    ohlc: pl.DataFrame,
    resolution_minutes: int,
    snap_hour_utc: int,
    snap_minute_utc: int,
) -> dict[date, pl.DataFrame]:
    available = pl.col("ts") + timedelta(minutes=resolution_minutes)
    snap_minutes = snap_hour_utc * 60 + snap_minute_utc
    prepared = ohlc.with_columns(available.alias("available_ts")).with_columns(
        pl.when(
            pl.col("available_ts").dt.hour().cast(pl.Int32) * 60
            + pl.col("available_ts").dt.minute().cast(pl.Int32)
            <= snap_minutes
        )
        .then(pl.col("available_ts").dt.date())
        .otherwise(pl.col("available_ts").dt.date() + timedelta(days=1))
        .alias("feature_date")
    )
    partitions = prepared.partition_by("feature_date", as_dict=True, maintain_order=False)
    return {cast(date, key[0]): frame.drop("feature_date") for key, frame in partitions.items()}


def _rv(bars: list[_DailyBar], horizon: int, method: str) -> float | None:
    if len(bars) < horizon:
        return None
    window = bars[-horizon:]
    hl = np.asarray([math.log(bar.high / bar.low) for bar in window])
    if method == "parkinson":
        variance = 365.0 * float(np.mean(hl**2)) / (4.0 * math.log(2.0))
    else:
        co = np.asarray([math.log(bar.close / bar.open) for bar in window])
        terms = 0.5 * hl**2 - (2.0 * math.log(2.0) - 1.0) * co**2
        variance = 365.0 * max(float(np.mean(terms)), 0.0)
    return math.sqrt(max(variance, 0.0))


def build_realized_features(
    ohlc: pl.DataFrame,
    snap_dates: Sequence[date],
    cfg: FeatureConfig,
    *,
    resolution_minutes: int = 60,
    snap_hour_utc: int = 8,
    snap_minute_utc: int = 5,
) -> dict[date, dict[str, object]]:
    """Build rolling features using only candles complete by each daily snap."""
    candles_by_day = _candles_by_snap_date(ohlc, resolution_minutes, snap_hour_utc, snap_minute_utc)
    bars: dict[date, _DailyBar] = {}
    returns: dict[date, float | None] = {}
    output: dict[date, dict[str, object]] = {}
    for day in sorted(set(snap_dates)):
        snap_ts = _snap(day, snap_hour_utc, snap_minute_utc)
        candles = candles_by_day.get(day)
        bar = None if candles is None else _daily_bar(candles, resolution_minutes)
        if bar is None:
            returns[day] = None
            output[day] = {
                "jump_flag": None,
                "underlying_source_ts": None,
                "underlying_age_s": None,
                "underlying_available": False,
            }
            continue
        source_ts = bar.source_ts
        age_s = (snap_ts - source_ts).total_seconds()
        if age_s > cfg.ohlc_max_age_s:
            returns[day] = None
            output[day] = {
                "jump_flag": None,
                "underlying_source_ts": None,
                "underlying_age_s": None,
                "underlying_available": False,
            }
            continue
        previous = bars.get(day - timedelta(days=1))
        previous_close = previous.close if previous is not None else None
        current_return = None if previous_close is None else math.log(bar.close / previous_close)
        bars[day] = bar
        returns[day] = current_return
        row: dict[str, object] = {
            "underlying_return_1d": current_return,
            "underlying_log_range": math.log(bar.high / bar.low),
            "underlying_source_ts": source_ts,
            "underlying_age_s": age_s,
            "underlying_available": True,
        }
        for horizon in cfg.realized_horizons_days:
            horizon_bars = [
                bars.get(day - timedelta(days=offset)) for offset in reversed(range(horizon))
            ]
            complete = all(item is not None for item in horizon_bars)
            typed_bars = cast(list[_DailyBar], horizon_bars)
            row[f"rv_parkinson_{horizon}d"] = (
                _rv(typed_bars, horizon, "parkinson") if complete else None
            )
            row[f"rv_garman_klass_{horizon}d"] = (
                _rv(typed_bars, horizon, "garman_klass") if complete else None
            )
        prior = [
            returns.get(day - timedelta(days=offset))
            for offset in reversed(range(1, cfg.jump_lookback_days + 1))
        ]
        if current_return is None or any(value is None for value in prior):
            row["jump_flag"] = None
        else:
            lookback = cast(list[float], prior)
            sigma = float(np.std(lookback, ddof=1 if len(lookback) > 1 else 0))
            row["jump_flag"] = abs(current_return) > cfg.jump_sigma_threshold * sigma
        output[day] = row
    return output
