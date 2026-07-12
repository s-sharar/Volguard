"""Unit + property tests for ``curate/pipeline.py`` (design Component 5).

Covers the orchestration driver that wires the four pure stages into one
per-snap transform and drives it over a date range:

- Property **CP7** — ``quotes_norm`` schema invariants: synthetic well-formed
  ``RawInputs`` (Black-76-priced C/P trades at a few strikes/expiries) run
  through :func:`curate_one_snap` yield a frame that passes ``QUOTES_NORM``
  validation and satisfies every invariant (``tau > 0``, ``F > 0``,
  ``strike > 0``, finite ``k``, banded ``iv_obs``, enumerated
  ``cp``/``iv_source``/``fwd_method``).
- Driver guards (Task 9.4): a leakage breach fails the snap and writes nothing
  (R8.2/R8.3); a bad computed column raises a pandera error naming the column
  (R10.2); an empty ``(snap, expiry)`` cell logs a coverage warning (R10.3);
  the ``index_carry`` fallback count is logged (R10.4).
- A thin :func:`run_curate` round-trip over a tmp-path raw fixture (R7.7/R11.1).
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from volguard.config import CurateConfig, DataConfig
from volguard.curate import pipeline
from volguard.curate.blackiv import black76_price
from volguard.curate.pipeline import RawInputs, curate_one_snap, run_curate
from volguard.curate.schemas import QUOTES_NORM, validate

_TS = pl.Datetime(time_unit="ms", time_zone="UTC")
_SNAP = datetime(2022, 4, 1, 8, 5, tzinfo=UTC)
_EXPIRY = datetime(2022, 5, 27, 8, 0, tzinfo=UTC)
_EXPIRY2 = datetime(2022, 6, 24, 8, 0, tzinfo=UTC)
_TAU = (_EXPIRY - _SNAP).total_seconds() / (365.0 * 24.0 * 3600.0)
_INDEX = 45_000.0
_CFG = CurateConfig()


# --- raw-frame builders ---------------------------------------------------


def _instrument_name(expiry: datetime, strike: float, cp: str) -> str:
    """A Deribit option name like ``BTC-27MAY22-45000-C`` for ``parse_instrument``."""
    mon = expiry.strftime("%b").upper()
    return f"BTC-{expiry.day}{mon}{expiry:%y}-{int(strike)}-{cp}"


def _option_row(
    ts: datetime,
    expiry: datetime,
    strike: float,
    cp: str,
    price_btc: float,
    iv_pct: float,
    amount: float,
) -> dict[str, object]:
    """One ``TRADES_OPTIONS``-shaped raw row (iv in *percent*, size in BTC)."""
    return {
        "ts": ts,
        "instrument": _instrument_name(expiry, strike, cp),
        "expiry": None,
        "strike": None,
        "cp": None,
        "price_btc": price_btc,
        "iv": iv_pct,
        "amount": amount,
        "index_price": _INDEX,
        "trade_id": f"{_instrument_name(expiry, strike, cp)}-{ts.isoformat()}",
        "block_flag": False,
        "source": "test",
    }


def _options_lf(rows: list[dict[str, object]]) -> pl.LazyFrame:
    """Assemble raw option rows into a ``TRADES_OPTIONS``-typed lazy frame."""
    schema = {
        "ts": _TS,
        "instrument": pl.String,
        "expiry": _TS,
        "strike": pl.Float64,
        "cp": pl.String,
        "price_btc": pl.Float64,
        "iv": pl.Float64,
        "amount": pl.Float64,
        "index_price": pl.Float64,
        "trade_id": pl.String,
        "block_flag": pl.Boolean,
        "source": pl.String,
    }
    return pl.DataFrame(rows, schema=schema).lazy()


def _empty_futures_lf() -> pl.LazyFrame:
    return pl.LazyFrame(schema={"ts": _TS, "instrument": pl.String, "price": pl.Float64})


def _empty_funding_lf() -> pl.LazyFrame:
    return pl.LazyFrame(
        schema={"ts": _TS, "interest_1h": pl.Float64, "interest_8h": pl.Float64}
    )


def _empty_instruments_lf() -> pl.LazyFrame:
    return pl.LazyFrame(schema={"instrument": pl.String})


def _well_formed_raw(
    f_star: float,
    sigma: float,
    strikes: list[float],
    expiries: list[datetime],
    amount: float = 5.0,
) -> RawInputs:
    """Build a no-arb ``RawInputs``: Black-76 C/P premiums at each strike/expiry.

    Premiums are priced in *BTC* (``price_btc = usd_premium / index``) so
    ``normalize`` recovers the USD premium via ``price_btc * index_price``, and
    the per-trade ``iv`` is exactly ``sigma`` (percent) so the trade IV lands in
    the ``QUOTES_NORM`` band. Near-simultaneous same-strike C/P pairs let the
    forward inference use PCP and recover ``f_star``.
    """
    ts = _SNAP.replace(minute=0)  # inside the base window, before the snap
    rows: list[dict[str, object]] = []
    for expiry in expiries:
        tau = (expiry - _SNAP).total_seconds() / (365.0 * 24.0 * 3600.0)
        for strike in strikes:
            call = black76_price(f_star, strike, tau, sigma, cp=1)
            put = black76_price(f_star, strike, tau, sigma, cp=-1)
            rows.append(
                _option_row(ts, expiry, strike, "C", call / _INDEX, sigma * 100.0, amount)
            )
            rows.append(
                _option_row(ts, expiry, strike, "P", put / _INDEX, sigma * 100.0, amount)
            )
    return RawInputs(
        options=_options_lf(rows),
        futures=_empty_futures_lf(),
        funding=_empty_funding_lf(),
        instruments=_empty_instruments_lf(),
    )


# --- CP7: quotes_norm schema invariants ------------------------------------


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    f_star=st.floats(min_value=20_000.0, max_value=80_000.0),
    sigma=st.floats(min_value=0.3, max_value=1.2),
    n_strikes=st.integers(min_value=2, max_value=4),
)
def test_cp7_quotes_norm_invariants(f_star: float, sigma: float, n_strikes: int) -> None:
    """CP7: curated rows satisfy every ``QUOTES_NORM`` invariant.

    **Validates: Requirements 7.2, 7.3, 7.4**
    """
    # Strikes bracketing the forward, near the money so delta lands in-band and
    # the trade IV is inside [iv_min, iv_max].
    strikes = [f_star * (0.9 + 0.05 * i) for i in range(n_strikes)]
    raw = _well_formed_raw(f_star, sigma, strikes, [_EXPIRY])
    out = curate_one_snap(_SNAP, raw, _CFG)

    # Passing curate_one_snap already ran QUOTES_NORM.validate; re-validate to be
    # explicit and assert the frame is non-empty for these well-formed inputs.
    validate(out, QUOTES_NORM)
    assert out.height > 0
    assert list(out.columns) == list(QUOTES_NORM.columns.keys())

    assert bool((out["tau"] > 0.0).all())
    assert bool((out["F"] > 0.0).all())
    assert bool((out["strike"] > 0.0).all())
    assert bool(out["k"].is_finite().all())
    assert bool(out["iv_obs"].is_between(_CFG.iv_min, _CFG.iv_max).all())
    assert set(out["cp"].unique().to_list()) <= {"C", "P"}
    assert set(out["iv_source"].unique().to_list()) <= {"trade", "mark", "mid"}
    assert set(out["fwd_method"].unique().to_list()) <= {"pcp", "future", "index_carry"}
    # Log-moneyness consistency (CP8) holds on the curated frame too.
    for strike, forward, k in zip(
        out["strike"].to_list(), out["F"].to_list(), out["k"].to_list(), strict=True
    ):
        assert k == pytest.approx(math.log(strike / forward), rel=1e-9, abs=1e-9)


def test_curate_one_snap_uses_pcp_forward() -> None:
    """A well-formed snap infers the PCP forward and produces banded IVs."""
    strikes = [42_000.0, 45_000.0, 48_000.0]
    raw = _well_formed_raw(45_000.0, 0.7, strikes, [_EXPIRY])
    out = curate_one_snap(_SNAP, raw, _CFG)
    assert out.height > 0
    assert set(out["fwd_method"].unique().to_list()) == {"pcp"}
    assert out["iv_source"].unique().to_list() == ["trade"]


# --- Task 9.4: driver guards ----------------------------------------------


def test_leakage_breach_fails_snap(monkeypatch: pytest.MonkeyPatch) -> None:
    """R8.2/R8.3: a surviving row with source_ts > snap_ts fails the snap.

    The window builder guarantees ``source_ts <= snap_ts``, so to exercise the
    driver's independent leakage assertion we patch ``apply_filters`` to inject
    a post-snap ``source_ts`` into an otherwise-kept row.
    """
    strikes = [45_000.0, 46_000.0]
    raw = _well_formed_raw(45_000.0, 0.7, strikes, [_EXPIRY])
    real_apply = pipeline.filters.apply_filters

    def _leaky_apply(rows: pl.DataFrame, snap_ts: datetime, cfg: CurateConfig) -> pl.DataFrame:
        out = real_apply(rows, snap_ts, cfg)
        after = snap_ts.replace(hour=snap_ts.hour + 1)
        return out.with_columns(pl.lit(after).cast(_TS).alias("source_ts"))

    monkeypatch.setattr(pipeline.filters, "apply_filters", _leaky_apply)
    with pytest.raises(ValueError, match="leakage"):
        curate_one_snap(_SNAP, raw, _CFG)


def test_bad_computed_column_raises_named_pandera_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R10.2: a QUOTES_NORM violation raises and names the offending column.

    Force a non-positive ``F`` (violating ``F > 0``) into the kept rows and
    assert pandera fails loudly, mentioning ``F``.
    """
    strikes = [45_000.0, 46_000.0]
    raw = _well_formed_raw(45_000.0, 0.7, strikes, [_EXPIRY])
    real_apply = pipeline.filters.apply_filters

    def _bad_forward(rows: pl.DataFrame, snap_ts: datetime, cfg: CurateConfig) -> pl.DataFrame:
        out = real_apply(rows, snap_ts, cfg)
        return out.with_columns(pl.lit(-1.0, dtype=pl.Float64).alias("F"))

    monkeypatch.setattr(pipeline.filters, "apply_filters", _bad_forward)
    with pytest.raises(Exception, match="F") as excinfo:
        curate_one_snap(_SNAP, raw, _CFG)
    assert "F" in str(excinfo.value)


def test_empty_expiry_cell_logs_coverage_warning(caplog: pytest.LogCaptureFixture) -> None:
    """R6.5/R10.3: an expiry with zero surviving rows logs a coverage warning.

    A second expiry is priced with a sub-``min_size_btc`` amount so every one of
    its rows is rejected by the size filter, leaving that ``(snap, expiry)`` cell
    empty after filtering while the first expiry survives.
    """
    strikes = [45_000.0, 46_000.0]
    good = _well_formed_raw(45_000.0, 0.7, strikes, [_EXPIRY], amount=5.0)
    # Tiny-size rows for a second expiry: rejected by BELOW_MIN_SIZE.
    small = _well_formed_raw(45_000.0, 0.7, strikes, [_EXPIRY2], amount=1e-6)
    good_rows = good.options.collect()
    small_rows = small.options.collect()
    combined = RawInputs(
        options=pl.concat([good_rows, small_rows]).lazy(),
        futures=_empty_futures_lf(),
        funding=_empty_funding_lf(),
        instruments=_empty_instruments_lf(),
    )
    with caplog.at_level(logging.WARNING, logger=pipeline.logger.name):
        out = curate_one_snap(_SNAP, combined, _CFG)
    # The small-size expiry survives in the window but not the curated output.
    assert _EXPIRY2 not in set(out["expiry"].unique().to_list())
    assert any(
        "coverage gap" in rec.message and rec.levelno == logging.WARNING
        for rec in caplog.records
    )


def test_index_carry_fallback_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """R10.4: the index_carry forward-fallback count is logged for QC.

    With only calls (no put to pair) and no futures/funding, forward inference
    falls through to ``index_carry`` for the expiry, which the driver reports.
    """
    ts = _SNAP.replace(minute=0)
    rows = [
        _option_row(ts, _EXPIRY, 45_000.0, "C", 0.1, 70.0, 5.0),
        _option_row(ts, _EXPIRY, 46_000.0, "C", 0.09, 70.0, 5.0),
        _option_row(ts, _EXPIRY, 47_000.0, "C", 0.08, 70.0, 5.0),
        _option_row(ts, _EXPIRY, 48_000.0, "C", 0.07, 70.0, 5.0),
    ]
    raw = RawInputs(
        options=_options_lf(rows),
        futures=_empty_futures_lf(),
        funding=_empty_funding_lf(),
        instruments=_empty_instruments_lf(),
    )
    with caplog.at_level(logging.INFO, logger=pipeline.logger.name):
        curate_one_snap(_SNAP, raw, _CFG)
    assert any("index_carry forward fallback" in rec.message for rec in caplog.records)


# --- run_curate thin round-trip -------------------------------------------


def test_run_curate_writes_partition(tmp_path: Path) -> None:
    """R7.7/R11.1: run_curate reads raw parquet, loops days, writes partitions."""
    raw_dir = tmp_path / "raw"
    curated_dir = tmp_path / "curated"
    opt_dir = raw_dir / "trades_options" / "month=2022-04"
    opt_dir.mkdir(parents=True)

    strikes = [42_000.0, 45_000.0, 48_000.0]
    raw = _well_formed_raw(45_000.0, 0.7, strikes, [_EXPIRY])
    raw.options.collect().write_parquet(opt_dir / "part.parquet", compression="zstd")

    data_cfg = DataConfig(raw_dir=raw_dir, curated_dir=curated_dir)
    run_curate(_CFG, data_cfg, start="2022-04-01", end="2022-04-01")

    part = curated_dir / "quotes_norm" / "date=2022-04-01" / "part.parquet"
    assert part.exists()
    frame = pl.read_parquet(part)
    validate(frame, QUOTES_NORM)
    assert frame.height > 0


def test_run_curate_skips_missing_raw(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """No raw trades_options -> logged and skipped, no crash (graceful degrade)."""
    data_cfg = DataConfig(raw_dir=tmp_path / "raw", curated_dir=tmp_path / "curated")
    with caplog.at_level(logging.WARNING, logger=pipeline.logger.name):
        run_curate(_CFG, data_cfg, start="2022-04-01", end="2022-04-01")
    assert any("no raw trades_options" in rec.message for rec in caplog.records)
    assert not (tmp_path / "curated").exists()
