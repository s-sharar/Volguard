"""Golden-day regression for the M4 surface stack (design CP10 / requirements R10).

Covers the ``Golden_Validator`` for Layer 2: run the full M4 surface build
(``load_snap -> fit_surface -> sample_grid -> surface_arb_metrics -> assemble ->
SURFACES_DAILY validate``) on a committed, real-grounded ``quotes_norm`` fixture
and lock the produced ``surfaces_daily`` snapshot against a committed golden
artifact.

Fixtures
--------
``tests/fixtures/surface_quotes_norm.parquet`` is a small, committed,
``QUOTES_NORM``-valid frame carved deterministically from the real M3 curated
partition ``data/curated/quotes_norm/date=2021-05-20/`` (which is gitignored and
therefore cannot itself be the committed golden source). It keeps the three
expiries with the most observations and an evenly-spaced strike subset within
each (36 rows across 3 expiries at the single 2021-05-20 08:05-UTC snap), so a
surface with >= ``cfg.min_expiries_per_snap`` distinct expiries can be fit.

``tests/fixtures/surface_golden.parquet`` is the committed golden
``SURFACES_DAILY`` snapshot, written on first run when absent (R10.3) and
compared against on every subsequent run (R10.1).

Tolerances
----------
The per-tenor SVI params and the grid tensor are produced by a ``scipy``
``least_squares`` fit, which is deterministic run-to-run in a single process but
is not guaranteed bit-reproducible across platforms / BLAS builds. The
committed-artifact comparison is therefore *structural-within-tolerance* with
two tiers: exact on keys/enums (``snap_date``, ``record_kind``, ``expiry``,
``moneyness``, ``fit_method``, ``interp_flag``); a tight tolerance
(``rel=1e-3``, ``abs=1e-6``) on the *observable* / derived floats (``grid_k``,
``grid_w``, ``rmse``, ``vega_sum``, ``tau``) — the true regression anchor, sized
to absorb cross-platform ``scipy`` optimizer noise (~2e-4 relative observed
between Windows and Linux CI) while still catching a genuine surface regression
(orders of magnitude larger); and a
deliberately loose tolerance (``rel=5e-2``, ``abs=1e-3``) on the raw-SVI params
(``svi_*``). The raw params are an ill-conditioned reparameterization — many
``(a, b, rho, m, sigma)`` tuples yield a near-identical smile — so a tiny
cross-platform solver difference can move a single param by O(1e-3) relative
while the observable smile is unchanged; pinning the observable grid tightly
still catches any genuine surface regression. Byte-stability across two runs
*in the same process* (R10.2) is asserted directly with ``DataFrame.equals``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from volguard.config import SurfaceConfig
from volguard.surface import pipeline
from volguard.surface.schemas import SURFACES_DAILY, validate

_FIXTURE = Path(__file__).parent / "fixtures" / "surface_quotes_norm.parquet"
_GOLDEN = Path(__file__).parent / "fixtures" / "surface_golden.parquet"

# The fixture is the 2021-05-20 08:05-UTC snap (single snap day).
_SNAP_DATE = date(2021, 5, 20)
_CFG = SurfaceConfig()

# Two tolerance tiers (see module docstring). Raw-SVI params are an
# ill-conditioned reparameterization: many (a, b, rho, m, sigma) tuples yield a
# near-identical smile, so a tiny cross-platform ``scipy`` solver difference can
# move a single param by O(1e-3) relative while the *observable* smile is
# unchanged. We therefore compare the observable/derived quantities tightly and
# the raw params loosely — a real regression still shifts the observable grid.
_OBS_REL_TOL = 1e-3  # observable/derived floats (grid_w, grid_k, rmse, vega_sum, tau)
_OBS_ABS_TOL = 1e-6
_PARAM_REL_TOL = 5e-2  # raw-SVI params (ill-conditioned; observable is the real anchor)
_PARAM_ABS_TOL = 1e-3

# Columns compared exactly (keys / enums / discrete provenance).
_EXACT_COLS = ("snap_date", "record_kind", "expiry", "moneyness", "fit_method", "interp_flag")
# Observable / derived floats — the real regression anchor, compared tightly.
_OBS_COLS = ("grid_k", "grid_w", "rmse", "vega_sum", "tau")
# Raw-SVI params — ill-conditioned, compared loosely.
_PARAM_COLS = ("svi_a", "svi_b", "svi_rho", "svi_m", "svi_sigma")

# Deterministic sort key that fully orders both param and grid rows.
_SORT_KEY = ["record_kind", "expiry", "moneyness", "tau"]


def _build_rows() -> pl.DataFrame:
    """Run the full M4 build on the committed fixture, returning the long frame."""
    frame = pl.scan_parquet(_FIXTURE)
    obs = pipeline.load_snap(frame, _SNAP_DATE, _CFG)
    result = pipeline.build_one_surface(_SNAP_DATE, obs, _CFG)
    assert result is not None, "expected a productive snap from the committed fixture"
    return result.rows


def test_cp10_surface_golden_snapshot() -> None:
    """CP10: the built surface is valid, well-shaped, and matches the golden snapshot.

    **Validates: Requirements 10.1, 10.2**
    """
    rows = _build_rows()

    # Boundary contract holds (build_one_surface already validates; re-assert here
    # so a schema regression fails in the golden test too).
    validate(rows, SURFACES_DAILY)
    assert rows.height > 0

    # Shape: one param row per fitted slice + exactly n_tenor * n_money grid rows.
    n_grid = len(_CFG.tenor_grid_days) * len(_CFG.moneyness_grid)
    n_param = rows.filter(pl.col("record_kind") == "param").height
    assert rows.filter(pl.col("record_kind") == "grid").height == n_grid
    assert rows.height == n_param + n_grid

    # R10.3: generate-and-commit the golden snapshot on first run rather than
    # failing silently; subsequent runs compare against it.
    if not _GOLDEN.exists():
        rows.write_parquet(_GOLDEN, compression="zstd")

    golden = pl.read_parquet(_GOLDEN)
    _assert_matches_golden(rows, golden)


def test_cp10_surface_build_is_byte_stable() -> None:
    """R10.2: two builds on the same committed fixture are byte-identical.

    **Validates: Requirements 10.2**
    """
    first = _build_rows()
    second = _build_rows()
    assert first.equals(second)


def _assert_matches_golden(rows: pl.DataFrame, golden: pl.DataFrame) -> None:
    """Structural-within-tolerance comparison against the committed golden frame.

    Exact on shape, columns, and the key/enum/provenance columns; numeric
    scipy-fit / derived columns within the documented ``rel``/``abs`` tolerance
    so the check is robust to cross-platform float noise while still catching a
    real regression in the fitted surface.
    """
    assert list(rows.columns) == list(golden.columns)
    assert rows.height == golden.height

    keyed = rows.sort(_SORT_KEY)
    keyed_gold = golden.sort(_SORT_KEY)

    # Keys / enums / discrete provenance must match exactly (nulls compare equal).
    for col in _EXACT_COLS:
        assert keyed[col].to_list() == keyed_gold[col].to_list(), f"golden mismatch in {col}"

    # Observable / derived floats: tight tolerance (the true regression anchor).
    for col in _OBS_COLS:
        _assert_col_close(keyed, keyed_gold, col, _OBS_REL_TOL, _OBS_ABS_TOL)
    # Raw-SVI params: loose tolerance (ill-conditioned reparameterization).
    for col in _PARAM_COLS:
        _assert_col_close(keyed, keyed_gold, col, _PARAM_REL_TOL, _PARAM_ABS_TOL)


def _assert_col_close(
    got_frame: pl.DataFrame, want_frame: pl.DataFrame, col: str, rel: float, abs_: float
) -> None:
    """Assert one column matches within tolerance, treating aligned nulls as equal."""
    for got, want in zip(got_frame[col].to_list(), want_frame[col].to_list(), strict=True):
        if got is None or want is None:
            assert got is want, f"golden null mismatch in {col}"
            continue
        assert got == pytest.approx(want, rel=rel, abs=abs_), (
            f"golden mismatch in {col}: got {got}, want {want}"
        )
