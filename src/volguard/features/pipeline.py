"""M5 daily feature orchestration, validation, audit, and atomic writes."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

import polars as pl

from volguard.config import DataConfig, EvalConfig, FeatureConfig, SurfaceConfig
from volguard.datasets.splits import (
    build_split_manifest,
    generate_walk_forward_folds,
    write_split_manifest,
)
from volguard.features.calendar import calendar_features
from volguard.features.market import build_market_features
from volguard.features.realized import build_realized_features
from volguard.features.schemas import (
    DAILY_FEATURES,
    FEATURE_QC,
    SOURCE_GROUPS,
    validate,
    validate_daily_features,
)
from volguard.features.surface_factors import SurfaceQualityError, build_surface_features
from volguard.features.types import FeatureRunSummary

logger = logging.getLogger("volguard.features")
_TS = pl.Datetime("ms", "UTC")
_RAW_INPUT_ERRORS = (
    ArithmeticError,
    KeyError,
    OSError,
    TypeError,
    ValueError,
    pl.exceptions.PolarsError,
)


@dataclass(frozen=True, slots=True)
class _ParquetRangeSource:
    files: tuple[str, ...]

    def collect(self, start_ts: datetime, end_ts: datetime) -> pl.DataFrame:
        return (
            pl.scan_parquet(list(self.files))
            .filter((pl.col("ts") >= start_ts) & (pl.col("ts") <= end_ts))
            .collect()
        )


@dataclass(frozen=True, slots=True)
class _RawInputs:
    underlying: pl.DataFrame | None
    dvol: _ParquetRangeSource | None
    funding: _ParquetRangeSource | None
    option_trades: _ParquetRangeSource | None
    future_trades: _ParquetRangeSource | None
    oi: pl.DataFrame | None


def _snap(day: date, cfg: SurfaceConfig) -> datetime:
    return datetime(
        day.year,
        day.month,
        day.day,
        cfg.snap_hour_utc,
        cfg.snap_minute_utc,
        tzinfo=UTC,
    )


def _partition_days(root: Path) -> dict[date, Path]:
    result: dict[date, Path] = {}
    for part in root.glob("date=*/part.parquet"):
        try:
            result[date.fromisoformat(part.parent.name.removeprefix("date="))] = part
        except ValueError:
            logger.warning("ignoring malformed surface partition path: %s", part)
    return result


def _iter_days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _bounded_days(available_days: list[date], start: date | None, end: date | None) -> list[date]:
    requested_first = start or available_days[0]
    requested_last = end or available_days[-1]
    if requested_first > requested_last:
        raise ValueError("start must not be after end")
    first = max(requested_first, available_days[0])
    last = min(requested_last, available_days[-1])
    if first > last:
        raise ValueError("requested range does not overlap available surfaces")
    return _iter_days(first, last)


def _load_ts_table(root: Path, start_ts: datetime, end_ts: datetime) -> pl.DataFrame | None:
    files = [str(path) for path in root.rglob("*.parquet")]
    if not files:
        return None
    try:
        return (
            pl.scan_parquet(files)
            .filter((pl.col("ts") >= start_ts) & (pl.col("ts") <= end_ts))
            .collect()
        )
    except _RAW_INPUT_ERRORS as exc:
        logger.warning("ignoring invalid optional raw table %s: %s", root, exc)
        return None


def _range_source(root: Path) -> _ParquetRangeSource | None:
    files = tuple(str(path) for path in root.rglob("*.parquet"))
    return None if not files else _ParquetRangeSource(files)


def _collector_oi(raw_dir: Path, days: list[date]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for day in days:
        path = raw_dir / "ticker_snapshots" / f"tickers_{day.isoformat()}.ndjson"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    snapshot = json.loads(line)
                    source_ts = datetime.fromisoformat(snapshot["snap_ts"])
                    summaries = snapshot.get("book_summary", [])
                    if not isinstance(summaries, list):
                        raise TypeError("book_summary must be a list")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    logger.warning("ignoring malformed collector snapshot in %s: %s", path, exc)
                    continue
                for item in summaries:
                    try:
                        instrument = item["instrument_name"]
                        open_interest = float(item["open_interest"])
                    except (KeyError, TypeError, ValueError) as exc:
                        logger.warning("ignoring malformed collector OI item in %s: %s", path, exc)
                        continue
                    if (
                        not isinstance(instrument, str)
                        or not instrument
                        or not math.isfinite(open_interest)
                        or open_interest < 0.0
                    ):
                        continue
                    rows.append(
                        {
                            "source_ts": source_ts,
                            "instrument": instrument,
                            "open_interest": open_interest,
                            "source": "collector",
                        }
                    )
    return rows


def _tardis_oi(
    raw_dir: Path, days: list[date], surface_cfg: SurfaceConfig
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for day in days:
        path = raw_dir / "tardis_chain" / f"date={day.isoformat()}" / "part.parquet"
        if not path.exists():
            continue
        snap_ts = _snap(day, surface_cfg)
        try:
            latest = (
                pl.scan_parquet(path)
                .select(
                    pl.from_epoch(pl.col("timestamp"), time_unit="us")
                    .dt.replace_time_zone("UTC")
                    .alias("source_ts"),
                    pl.col("symbol").alias("instrument"),
                    pl.col("open_interest"),
                )
                .filter(pl.col("source_ts") <= snap_ts)
                .sort("source_ts")
                .group_by("instrument")
                .agg(pl.col("source_ts").last(), pl.col("open_interest").last())
                .drop_nulls("open_interest")
                .collect()
            )
        except _RAW_INPUT_ERRORS as exc:
            logger.warning("ignoring invalid optional Tardis OI table %s: %s", path, exc)
            continue
        rows.extend(latest.with_columns(pl.lit("tardis").alias("source")).to_dicts())
    return rows


def _load_oi(raw_dir: Path, days: list[date], surface_cfg: SurfaceConfig) -> pl.DataFrame | None:
    try:
        rows = _collector_oi(raw_dir, days) + _tardis_oi(raw_dir, days, surface_cfg)
    except _RAW_INPUT_ERRORS as exc:
        logger.warning("ignoring invalid optional OI inputs under %s: %s", raw_dir, exc)
        return None
    if not rows:
        return None
    return pl.DataFrame(rows).with_columns(pl.col("source_ts").cast(_TS))


def _empty_row() -> dict[str, object]:
    row: dict[str, object] = {str(name): None for name in DAILY_FEATURES.columns}
    for source in SOURCE_GROUPS:
        row[f"{source}_available"] = False
    return row


def _assert_safe_output_path(path: Path, root: Path) -> None:
    root_resolved = root.resolve()
    parent_resolved = path.parent.resolve()
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError(f"refusing symlinked output path: {path}")
    if not parent_resolved.is_relative_to(root_resolved):
        raise ValueError(f"output path is outside expected root {root}: {path}")


def _atomic_write(frame: pl.DataFrame, path: Path, root: Path) -> None:
    _assert_safe_output_path(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_safe_output_path(path, root)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.is_symlink():
        raise ValueError(f"refusing symlinked temporary output path: {temporary}")
    frame.write_parquet(temporary, compression="zstd")
    os.replace(temporary, path)


def _remove_exact_feature_part(path: Path, daily_dir: Path) -> None:
    if not path.exists():
        return
    _assert_safe_output_path(path, daily_dir)
    if path.name != "part.parquet" or path.parent.parent.resolve() != daily_dir.resolve():
        raise ValueError(f"refusing to remove unexpected feature path: {path}")
    path.unlink()
    if not any(path.parent.iterdir()):
        path.parent.rmdir()


def _qc_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={name: column.dtype for name, column in FEATURE_QC.columns.items()}
        )
    frame = pl.DataFrame(rows)
    timestamp_columns = [name for name in ("snap_ts", "max_source_ts") if name in frame]
    return frame.with_columns(pl.col(timestamp_columns).cast(_TS))


def _merged_qc(current: pl.DataFrame, qc_path: Path, rerun_days: list[date]) -> pl.DataFrame:
    if not qc_path.exists():
        return validate(current.sort("snap_date"), FEATURE_QC)
    existing = validate(pl.read_parquet(qc_path), FEATURE_QC).filter(
        ~pl.col("snap_date").is_in(rerun_days)
    )
    return validate(pl.concat([existing, current]).sort("snap_date"), FEATURE_QC)


def _accepted_qc_row(
    day: date,
    snap_ts: datetime,
    daily: pl.DataFrame,
    out_part: Path,
    input_rows: int,
) -> dict[str, object]:
    quality_flags = cast(list[str], daily["surface_quality_flags"].to_list()[0])
    return {
        "snap_date": day,
        "snap_ts": snap_ts,
        "status": "accepted",
        "reason_code": "surface_quality_warning" if quality_flags else "ok",
        "detail": ",".join(quality_flags) if quality_flags else "feature row validated",
        "input_rows": input_rows,
        "grid_cells": len(daily["grid_w"][0]),
        "output_path": str(out_part),
        "max_source_ts": daily["max_source_ts"][0],
    }


def _raw_inputs(
    data_cfg: DataConfig, days: list[date], cfg: FeatureConfig, surface_cfg: SurfaceConfig
) -> _RawInputs:
    lookback = max(*cfg.realized_horizons_days, cfg.jump_lookback_days) + 2
    start_ts = _snap(days[0] - timedelta(days=lookback), surface_cfg) - timedelta(
        minutes=data_cfg.ohlc_resolution_minutes
    )
    end_ts = _snap(days[-1], surface_cfg)
    return _RawInputs(
        underlying=_load_ts_table(data_cfg.raw_table_dir("index_ohlc"), start_ts, end_ts),
        dvol=_range_source(data_cfg.raw_table_dir("dvol")),
        funding=_range_source(data_cfg.raw_table_dir("funding")),
        option_trades=_range_source(data_cfg.raw_table_dir("trades_options")),
        future_trades=_range_source(data_cfg.raw_table_dir("trades_futures")),
        oi=_load_oi(
            data_cfg.raw_dir,
            _iter_days(
                days[0] - timedelta(days=math.ceil(cfg.oi_max_age_s / 86_400.0)),
                days[-1],
            ),
            surface_cfg,
        ),
    )


def _collect_range(
    source_name: str,
    source: _ParquetRangeSource | None,
    start_ts: datetime,
    end_ts: datetime,
) -> pl.DataFrame | None:
    if source is None:
        return None
    try:
        return source.collect(start_ts, end_ts)
    except _RAW_INPUT_ERRORS as exc:
        logger.warning("ignoring invalid optional %s input: %s", source_name, exc)
        return None


def _realized(
    raw: pl.DataFrame | None,
    days: list[date],
    cfg: FeatureConfig,
    surface_cfg: SurfaceConfig,
    resolution_minutes: int,
) -> dict[date, dict[str, object]]:
    lookback = max(*cfg.realized_horizons_days, cfg.jump_lookback_days) + 2
    all_days = _iter_days(days[0] - timedelta(days=lookback), days[-1])
    if raw is None:
        return {
            day: {
                "underlying_source_ts": None,
                "underlying_age_s": None,
                "underlying_available": False,
            }
            for day in all_days
        }
    try:
        return build_realized_features(
            raw,
            all_days,
            cfg,
            resolution_minutes=resolution_minutes,
            snap_hour_utc=surface_cfg.snap_hour_utc,
            snap_minute_utc=surface_cfg.snap_minute_utc,
        )
    except _RAW_INPUT_ERRORS as exc:
        logger.warning("ignoring invalid optional underlying input: %s", exc)
        return {
            day: {
                "underlying_source_ts": None,
                "underlying_age_s": None,
                "underlying_available": False,
            }
            for day in all_days
        }


def _prepare_inputs(
    data_cfg: DataConfig,
    days: list[date],
    feature_cfg: FeatureConfig,
    surface_cfg: SurfaceConfig,
) -> tuple[_RawInputs, dict[date, dict[str, object]]]:
    raw = _raw_inputs(data_cfg, days, feature_cfg, surface_cfg)
    return raw, _realized(
        raw.underlying,
        days,
        feature_cfg,
        surface_cfg,
        data_cfg.ohlc_resolution_minutes,
    )


def _build_row(
    surface_rows: pl.DataFrame,
    day: date,
    feature_cfg: FeatureConfig,
    surface_cfg: SurfaceConfig,
    realized: dict[str, object],
    raw: _RawInputs,
    dvol_resolution_seconds: int,
) -> pl.DataFrame:
    surface = build_surface_features(surface_rows, day, surface_cfg, feature_cfg=feature_cfg)
    snap_ts = cast(datetime, surface["snap_ts"])
    market = build_market_features(
        snap_ts,
        feature_cfg,
        dvol=_collect_range(
            "DVOL",
            raw.dvol,
            snap_ts
            - timedelta(
                days=feature_cfg.dvol_change_days,
                seconds=feature_cfg.dvol_max_age_s + dvol_resolution_seconds,
            ),
            snap_ts,
        ),
        funding=_collect_range(
            "funding",
            raw.funding,
            snap_ts - timedelta(seconds=feature_cfg.funding_max_age_s),
            snap_ts,
        ),
        option_trades=_collect_range(
            "option trades", raw.option_trades, snap_ts - timedelta(days=1), snap_ts
        ),
        future_trades=_collect_range(
            "future trades",
            raw.future_trades,
            snap_ts - timedelta(seconds=feature_cfg.basis_max_age_s),
            snap_ts,
        ),
        oi_snapshots=raw.oi,
        dvol_resolution_seconds=dvol_resolution_seconds,
    )
    row = _empty_row()
    row.update(surface)
    row.update(realized)
    row.update(market)
    row.update(calendar_features(snap_ts))
    rv22 = row.get("rv_parkinson_22d")
    for tenor in surface_cfg.tenor_grid_days:
        name = f"iv_rv_spread_{int(tenor)}d"
        atm = row.get(f"atm_iv_{int(tenor)}d")
        row[name] = None if atm is None or rv22 is None else cast(float, atm) - cast(float, rv22)
    source_times = [
        value
        for source in SOURCE_GROUPS
        if isinstance((value := row[f"{source}_source_ts"]), datetime)
    ]
    row["max_source_ts"] = max(source_times)
    frame = pl.DataFrame([row])
    ts_columns = [name for name in frame.columns if name.endswith("_ts")]
    return validate_daily_features(frame.with_columns(pl.col(ts_columns).cast(_TS)))


def run_features(
    feature_cfg: FeatureConfig,
    data_cfg: DataConfig,
    surface_cfg: SurfaceConfig,
    eval_cfg: EvalConfig,
    *,
    start: date | None = None,
    end: date | None = None,
) -> FeatureRunSummary:
    """Build bounded daily feature partitions and an auditable QC table."""
    surface_parts = _partition_days(data_cfg.curated_dir / "surfaces_daily")
    available_days = sorted(surface_parts)
    daily_dir = data_cfg.features_dir / "daily"
    qc_path = data_cfg.features_dir / "qc" / "part.parquet"
    if not available_days:
        raise FileNotFoundError("no curated/surfaces_daily partitions found")
    days = _bounded_days(available_days, start, end)
    raw, realized = _prepare_inputs(data_cfg, days, feature_cfg, surface_cfg)
    accepted: list[date] = []
    rejected: list[date] = []
    qc_rows: list[dict[str, object]] = []
    pending_writes: dict[Path, pl.DataFrame] = {}
    pending_removals: list[Path] = []
    for day in days:
        snap_ts = _snap(day, surface_cfg)
        out_part = daily_dir / f"date={day.isoformat()}" / "part.parquet"
        part = surface_parts.get(day)
        if part is None:
            rejected.append(day)
            pending_removals.append(out_part)
            qc_rows.append(
                {
                    "snap_date": day,
                    "snap_ts": snap_ts,
                    "status": "rejected",
                    "reason_code": "surface_missing",
                    "detail": "no surface partition",
                    "input_rows": 0,
                    "grid_cells": 0,
                    "output_path": None,
                    "max_source_ts": None,
                }
            )
            continue
        surface_rows: pl.DataFrame | None = None
        try:
            surface_rows = pl.read_parquet(part)
            daily = _build_row(
                surface_rows,
                day,
                feature_cfg,
                surface_cfg,
                realized.get(day, {}),
                raw,
                data_cfg.dvol_resolution_seconds,
            )
        except (
            OSError,
            TypeError,
            ValueError,
            pl.exceptions.PolarsError,
        ) as exc:
            if isinstance(exc, SurfaceQualityError):
                quality_error = exc
            elif surface_rows is None:
                quality_error = SurfaceQualityError(
                    "surface_input_invalid", f"surface input could not be read: {exc}"
                )
            else:
                quality_error = SurfaceQualityError(
                    "feature_row_invalid", f"feature row could not be built: {exc}"
                )
            rejected.append(day)
            pending_removals.append(out_part)
            grid_cells = (
                0
                if surface_rows is None or "record_kind" not in surface_rows.columns
                else surface_rows.filter(pl.col("record_kind") == "grid").height
            )
            qc_rows.append(
                {
                    "snap_date": day,
                    "snap_ts": snap_ts,
                    "status": "rejected",
                    "reason_code": quality_error.reason_code,
                    "detail": quality_error.detail,
                    "input_rows": 0 if surface_rows is None else surface_rows.height,
                    "grid_cells": grid_cells,
                    "output_path": None,
                    "max_source_ts": None,
                }
            )
            continue
        pending_writes[out_part] = daily
        accepted.append(day)
        qc_rows.append(_accepted_qc_row(day, snap_ts, daily, out_part, surface_rows.height))
    qc = _merged_qc(validate(_qc_frame(qc_rows), FEATURE_QC), qc_path, days)
    for path in pending_removals:
        _remove_exact_feature_part(path, daily_dir)
    for path, daily in pending_writes.items():
        _atomic_write(daily, path, daily_dir)
    _atomic_write(qc, qc_path, data_cfg.features_dir)
    canonical_dates = sorted(_partition_days(daily_dir))
    folds = generate_walk_forward_folds(canonical_dates, eval_cfg)
    split_manifest = build_split_manifest(canonical_dates, folds)
    split_manifest_path = data_cfg.features_dir / "splits" / "part.parquet"
    write_split_manifest(split_manifest, split_manifest_path, root=data_cfg.features_dir)
    return FeatureRunSummary(
        accepted_dates=tuple(accepted),
        rejected_dates=tuple(rejected),
        daily_dir=daily_dir,
        qc_path=qc_path,
        split_manifest_path=split_manifest_path,
    )
