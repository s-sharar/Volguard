"""As-of market features with explicit availability and lineage."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import cast

import polars as pl

from volguard.config import FeatureConfig
from volguard.ingest.schemas import parse_instrument

_YEAR_SECONDS = 365.0 * 86_400.0
logger = logging.getLogger("volguard.features.market")
_MARKET_INPUT_ERRORS = (
    KeyError,
    TypeError,
    ValueError,
    ZeroDivisionError,
    pl.exceptions.PolarsError,
)


def _safe_feature(
    name: str, builder: Callable[[], dict[str, object]], missing: dict[str, object]
) -> dict[str, object]:
    try:
        return builder()
    except _MARKET_INPUT_ERRORS as exc:
        logger.warning("ignoring invalid optional %s input: %s", name, exc)
        return missing


def _lineage(prefix: str, source_ts: datetime | None, snap_ts: datetime) -> dict[str, object]:
    return {
        f"{prefix}_source_ts": source_ts,
        f"{prefix}_age_s": None if source_ts is None else (snap_ts - source_ts).total_seconds(),
        f"{prefix}_available": source_ts is not None,
    }


def _dvol_features(
    snap_ts: datetime,
    cfg: FeatureConfig,
    frame: pl.DataFrame | None,
    resolution_seconds: int,
) -> dict[str, object]:
    if frame is None or frame.is_empty():
        return {"dvol": None, "dvol_change_5d": None, **_lineage("dvol", None, snap_ts)}
    bars = frame.with_columns(
        (pl.col("ts") + timedelta(seconds=resolution_seconds)).alias("available_ts")
    )
    usable = (
        bars.filter(
            (pl.col("available_ts") <= snap_ts)
            & pl.col("close").is_finite()
            & (pl.col("close") >= 0.0)
        )
        .drop_nulls("close")
        .sort("available_ts")
    )
    if usable.is_empty():
        return {"dvol": None, "dvol_change_5d": None, **_lineage("dvol", None, snap_ts)}
    source_ts = cast(datetime, usable["available_ts"][-1])
    if (snap_ts - source_ts).total_seconds() > cfg.dvol_max_age_s:
        return {"dvol": None, "dvol_change_5d": None, **_lineage("dvol", None, snap_ts)}
    current = float(usable["close"][-1]) / 100.0
    target = snap_ts - timedelta(days=cfg.dvol_change_days)
    old = usable.filter(pl.col("available_ts") <= target)
    old_source_ts = None if old.is_empty() else cast(datetime, old["available_ts"][-1])
    change = (
        None
        if old_source_ts is None or (target - old_source_ts).total_seconds() > cfg.dvol_max_age_s
        else current - float(old["close"][-1]) / 100.0
    )
    return {"dvol": current, "dvol_change_5d": change, **_lineage("dvol", source_ts, snap_ts)}


def _funding_features(
    snap_ts: datetime, cfg: FeatureConfig, frame: pl.DataFrame | None
) -> dict[str, object]:
    if frame is None or frame.is_empty():
        return {"funding_8h": None, **_lineage("funding", None, snap_ts)}
    usable = (
        frame.filter(pl.col("ts") <= snap_ts)
        .drop_nulls("interest_8h")
        .filter(pl.col("interest_8h").is_finite())
        .sort("ts")
    )
    if usable.is_empty():
        return {"funding_8h": None, **_lineage("funding", None, snap_ts)}
    source_ts = cast(datetime, usable["ts"][-1])
    if (snap_ts - source_ts).total_seconds() > cfg.funding_max_age_s:
        return {"funding_8h": None, **_lineage("funding", None, snap_ts)}
    return {
        "funding_8h": float(usable["interest_8h"][-1]),
        **_lineage("funding", source_ts, snap_ts),
    }


def _volume_features(snap_ts: datetime, frame: pl.DataFrame | None) -> dict[str, object]:
    if frame is None or frame.is_empty():
        return {
            "options_volume_btc": None,
            "put_call_volume_ratio": None,
            **_lineage("volume", None, snap_ts),
        }
    usable = frame.filter(
        (pl.col("ts") > snap_ts - timedelta(days=1)) & (pl.col("ts") <= snap_ts)
    ).drop_nulls(["amount", "cp"])
    usable = usable.filter(pl.col("amount").is_finite() & (pl.col("amount") >= 0.0))
    if usable.is_empty():
        return {
            "options_volume_btc": None,
            "put_call_volume_ratio": None,
            **_lineage("volume", None, snap_ts),
        }
    calls = float(usable.filter(pl.col("cp") == "C")["amount"].sum())
    puts = float(usable.filter(pl.col("cp") == "P")["amount"].sum())
    return {
        "options_volume_btc": calls + puts,
        "put_call_volume_ratio": None if calls == 0.0 else puts / calls,
        **_lineage("volume", cast(datetime, usable["ts"].max()), snap_ts),
    }


def _basis_features(
    snap_ts: datetime, cfg: FeatureConfig, frame: pl.DataFrame | None
) -> dict[str, object]:
    missing = {"futures_basis_annualized": None, **_lineage("basis", None, snap_ts)}
    if frame is None or frame.is_empty():
        return missing
    usable = frame.filter(pl.col("ts") <= snap_ts).drop_nulls(["price", "index_price"])
    usable = usable.filter(
        pl.col("price").is_finite()
        & pl.col("index_price").is_finite()
        & (pl.col("price") > 0.0)
        & (pl.col("index_price") > 0.0)
    )
    candidates: list[tuple[float, float, datetime]] = []
    for instrument in usable["instrument"].unique().to_list():
        latest = usable.filter(pl.col("instrument") == instrument).sort("ts").tail(1)
        source_ts = cast(datetime, latest["ts"][0])
        if (snap_ts - source_ts).total_seconds() > cfg.basis_max_age_s:
            continue
        try:
            expiry = parse_instrument(instrument).expiry
        except ValueError:
            continue
        if expiry is None:
            continue
        tau_seconds = (expiry - snap_ts).total_seconds()
        tau_days = tau_seconds / 86_400.0
        if tau_days < cfg.basis_min_days:
            continue
        basis = (float(latest["price"][0]) / float(latest["index_price"][0]) - 1.0) / (
            tau_seconds / _YEAR_SECONDS
        )
        candidates.append((abs(tau_days - cfg.basis_target_days), basis, source_ts))
    if not candidates:
        return missing
    _, value, source_ts = min(candidates, key=lambda item: item[0])
    return {"futures_basis_annualized": value, **_lineage("basis", source_ts, snap_ts)}


def _oi_features(
    snap_ts: datetime, cfg: FeatureConfig, frame: pl.DataFrame | None
) -> dict[str, object]:
    missing = {"aggregate_oi": None, **_lineage("oi", None, snap_ts)}
    if frame is None or frame.is_empty():
        return missing
    usable = frame.filter(
        (pl.col("source_ts") <= snap_ts)
        & (pl.col("source_ts") >= snap_ts - timedelta(seconds=cfg.oi_max_age_s))
    ).drop_nulls("open_interest")
    usable = usable.filter(pl.col("open_interest").is_finite() & (pl.col("open_interest") >= 0.0))
    for source in ("collector", "tardis"):
        selected = usable.filter(pl.col("source") == source)
        if source == "tardis":
            selected = selected.filter(pl.col("source_ts").dt.date() == snap_ts.date())
        if selected.is_empty():
            continue
        latest = selected.sort("source_ts").unique("instrument", keep="last")
        source_ts = cast(datetime, latest["source_ts"].max())
        return {
            "aggregate_oi": float(latest["open_interest"].sum()),
            **_lineage("oi", source_ts, snap_ts),
        }
    return missing


def build_market_features(
    snap_ts: datetime,
    cfg: FeatureConfig,
    *,
    dvol: pl.DataFrame | None = None,
    funding: pl.DataFrame | None = None,
    option_trades: pl.DataFrame | None = None,
    future_trades: pl.DataFrame | None = None,
    oi_snapshots: pl.DataFrame | None = None,
    dvol_resolution_seconds: int = 3_600,
) -> dict[str, object]:
    """Build all nullable market features for one snap."""
    if dvol_resolution_seconds <= 0:
        raise ValueError("dvol_resolution_seconds must be positive")
    dvol_missing = {"dvol": None, "dvol_change_5d": None, **_lineage("dvol", None, snap_ts)}
    funding_missing = {"funding_8h": None, **_lineage("funding", None, snap_ts)}
    volume_missing = {
        "options_volume_btc": None,
        "put_call_volume_ratio": None,
        **_lineage("volume", None, snap_ts),
    }
    basis_missing = {"futures_basis_annualized": None, **_lineage("basis", None, snap_ts)}
    oi_missing = {"aggregate_oi": None, **_lineage("oi", None, snap_ts)}
    return {
        **_safe_feature(
            "DVOL",
            lambda: _dvol_features(snap_ts, cfg, dvol, dvol_resolution_seconds),
            dvol_missing,
        ),
        **_safe_feature(
            "funding", lambda: _funding_features(snap_ts, cfg, funding), funding_missing
        ),
        **_safe_feature(
            "option-volume", lambda: _volume_features(snap_ts, option_trades), volume_missing
        ),
        **_safe_feature(
            "basis", lambda: _basis_features(snap_ts, cfg, future_trades), basis_missing
        ),
        **_safe_feature("OI", lambda: _oi_features(snap_ts, cfg, oi_snapshots), oi_missing),
    }
