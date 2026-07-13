"""Golden Tardis free-day validation tests (design CP10 / requirements R9).

Covers the ``Golden_Validator`` (:mod:`volguard.curate.validate_tardis`):

- **CP10 golden snapshot regression** (:func:`test_cp10_golden_tardis_free_day`):
  run curation on the committed ``tests/fixtures/tardis_sample.csv`` free day,
  assert the produced ``quotes_norm`` passes ``QUOTES_NORM`` validation, that its
  per-``(expiry, moneyness-bucket)`` IVs match the fixture's ``mark_iv`` within
  the documented tolerance, that two runs on the same input are byte-identical
  (R9.5), and that the snapshot matches the committed golden artifact (R9.4).
- **Graceful degradation** (:func:`test_missing_tardis_day_is_skipped_not_failed`):
  a requested date with no landed Tardis data is skipped with a logged coverage
  message rather than raising (R9.3), and the full backfill is never required
  (R9.2).

The golden artifact ``tests/fixtures/tardis_golden_quotes_norm.parquet`` is
written on first run only if absent, then compared against on every subsequent
run. The committed-artifact comparison is structural-within-tolerance (exact on
keys/enums, ``iv_obs``/``F``/``k`` within a tight numeric tolerance) so it is
robust to cross-platform float noise, while byte-stability across two runs of the
same input is asserted directly with ``DataFrame.equals`` (design R9.5).
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

import volguard.curate.validate_tardis as vt
from volguard.config import CurateConfig
from volguard.curate.schemas import QUOTES_NORM, validate
from volguard.curate.validate_tardis import (
    compare_curated_vs_tardis,
    curate_tardis_chain,
    validate_free_days,
)
from volguard.ingest import tardis_free as tf
from volguard.ingest.schemas import TARDIS_CHAIN
from volguard.ingest.schemas import validate as validate_raw

_FIXTURE = Path(__file__).parent / "fixtures" / "tardis_sample.csv"
_GOLDEN = Path(__file__).parent / "fixtures" / "tardis_golden_quotes_norm.parquet"

# The Tardis fixture is the 2022-04-01 free day; the snap is that day at 08:05 UTC.
_DAY = date(2022, 4, 1)
_SNAP = datetime(2022, 4, 1, 8, 5, tzinfo=UTC)
_CFG = CurateConfig()


def _load_fixture_chain() -> pl.LazyFrame:
    """Load the committed Tardis CSV fixture as a validated ``TARDIS_CHAIN`` frame."""
    df = pl.read_csv(_FIXTURE, schema_overrides=tf._CONTRACT_CASTS)
    return validate_raw(df, TARDIS_CHAIN).lazy()


def test_cp10_golden_tardis_free_day() -> None:
    """CP10: curated Tardis snapshot is valid, matches marks, and is byte-stable.

    **Validates: Requirements 9.4, 9.5**
    """
    chain = _load_fixture_chain()

    # Run curation through the shared code path; must satisfy the frozen contract.
    curated = curate_tardis_chain(chain, _SNAP, _CFG)
    validate(curated, QUOTES_NORM)
    assert curated.height > 0
    assert list(curated.columns) == list(QUOTES_NORM.columns.keys())

    # Per-(expiry, moneyness-bucket) curated IVs match the Tardis mark IVs within
    # the documented cross-source tolerance (design CP10 / R9.1, R9.4).
    result = compare_curated_vs_tardis(curated, chain, _SNAP, _CFG)
    assert result.buckets, "expected at least one overlapping (expiry, k-bucket) to compare"
    assert result.passed, (
        f"curated vs Tardis mark IV disagreement exceeds tol={result.tolerance}: "
        f"max abs diff {result.max_abs_diff:.4f}\n{result.to_frame()}"
    )

    # R9.5: two runs on the same committed fixture are byte-identical.
    curated_again = curate_tardis_chain(chain, _SNAP, _CFG)
    assert curated.equals(curated_again)

    # R9.4: compare against the committed golden artifact (write-if-absent first).
    if not _GOLDEN.exists():
        curated.write_parquet(_GOLDEN, compression="zstd")
    golden = pl.read_parquet(_GOLDEN)
    _assert_matches_golden(curated, golden)


def _assert_matches_golden(curated: pl.DataFrame, golden: pl.DataFrame) -> None:
    """Structural-within-tolerance comparison against the committed golden frame.

    Exact on shape, columns, keys (``expiry``/``strike``/``cp``) and enums
    (``iv_source``/``fwd_method``); numeric IV/forward/moneyness columns within a
    tight tolerance so the check is robust to cross-platform float noise while
    still catching any real regression in the curated surface.
    """
    assert list(curated.columns) == list(golden.columns)
    assert curated.height == golden.height

    keyed_cur = curated.sort(["expiry", "strike", "cp"])
    keyed_gold = golden.sort(["expiry", "strike", "cp"])

    # Keys and enumerated provenance columns must match exactly.
    for col in ("expiry", "strike", "cp", "iv_source", "fwd_method"):
        assert keyed_cur[col].to_list() == keyed_gold[col].to_list(), f"golden mismatch in {col}"

    # Numeric columns within a tight absolute tolerance.
    for col in ("tau", "F", "k", "iv_obs", "usd_premium", "size", "staleness_s"):
        for got, want in zip(keyed_cur[col].to_list(), keyed_gold[col].to_list(), strict=True):
            assert got == pytest.approx(want, rel=1e-9, abs=1e-9), f"golden mismatch in {col}"


def test_curated_tardis_snapshot_rejects_expired_and_outliers() -> None:
    """The curated Tardis snapshot keeps only in-band, non-outlier live options.

    The fixture's ``1APR22`` options expire 08:00 that day (before the 08:05 snap,
    so ``tau <= 0`` — rejected), and the ``29APR22`` 50000 call is a per-expiry
    MAD outlier; only the two 45000 ``29APR22`` C/P rows survive. This pins the
    documented golden shape so a future normalize/filter change is caught.
    """
    curated = curate_tardis_chain(_load_fixture_chain(), _SNAP, _CFG)
    assert curated.height == 2
    expiry = datetime(2022, 4, 29, 8, 0, tzinfo=UTC)
    assert set(curated["expiry"].unique().to_list()) == {expiry}
    assert curated["strike"].unique().to_list() == [45000.0]
    assert set(curated["cp"].to_list()) == {"C", "P"}
    # iv_obs is carried from the Tardis mark_iv (percent -> fraction).
    assert set(curated["iv_source"].unique().to_list()) == {"trade"}


def test_missing_tardis_day_is_skipped_not_failed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """R9.3: a requested date with no landed Tardis data is skipped and logged.

    No Parquet part exists under ``tardis_dir`` for the requested date, so the
    non-blocking driver must skip it with a logged coverage message rather than
    raising or requiring the full backfill (R9.2).
    """
    empty_tardis_dir = tmp_path / "tardis_chain"  # deliberately not created

    def _curated_for_date(_day: date) -> pl.DataFrame | None:
        # Would supply curated output, but the date has no Tardis data to compare.
        return curate_tardis_chain(_load_fixture_chain(), _SNAP, _CFG)

    with caplog.at_level(logging.INFO, logger=vt.logger.name):
        results = validate_free_days(
            [_DAY],
            empty_tardis_dir,
            _curated_for_date,
            lambda _day: _SNAP,
            _CFG,
        )

    assert results == {}  # nothing validated, nothing raised
    assert any(
        "no free-day data landed" in rec.message and _DAY.isoformat() in rec.message
        for rec in caplog.records
    )


def test_validate_free_days_validates_landed_day(tmp_path: Path) -> None:
    """R9.1/R9.2: a landed Tardis day with curated output is compared and passes.

    Writes the fixture as a landed Tardis Parquet part, supplies the curated
    snapshot for that date, and asserts the driver returns a passing
    :class:`~volguard.curate.validate_tardis.ValidationResult` — validating only
    what has landed, without the full backfill.
    """
    tardis_dir = tmp_path / "tardis_chain"
    part = tardis_dir / f"date={_DAY.isoformat()}" / "part.parquet"
    part.parent.mkdir(parents=True)
    _load_fixture_chain().collect().write_parquet(part, compression="zstd")

    curated = curate_tardis_chain(_load_fixture_chain(), _SNAP, _CFG)
    results = validate_free_days(
        [_DAY],
        tardis_dir,
        lambda _day: curated,
        lambda _day: _SNAP,
        _CFG,
    )

    assert set(results) == {_DAY}
    assert results[_DAY].passed
    assert results[_DAY].buckets
