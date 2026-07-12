"""Layer-1 IV cross-check, quality flags, filter cascade, and staleness.

This module owns the Layer-1 *quality* concerns (design Component 4):

- The :class:`QualityFlag` bitmask — the single vocabulary of rejection/flag
  reasons recorded in the ``quality_flags`` column so no row is ever silently
  dropped. Every stage that flags a row ORs its reason into this bitmask.
- :func:`cross_check_iv` — recompute IV with the trusted M1 Black-76 solver
  (:func:`volguard.curate.blackiv.implied_vol`, reused, never reimplemented),
  pick the observed IV (Deribit's per-trade ``iv_trade`` when finite, else the
  recomputed "mark"), and flag divergence / unsolvable rows (design R3).

The band/MAD/size/block filter cascade and staleness weighting (Task 6, design
R4/R5) extend this file and reuse :class:`QualityFlag`; the enum is defined here
in full now so those later stages slot in without churn.

Conventions match the trusted M1 core and the canonical frame emitted by
:mod:`volguard.curate.normalize` + :func:`volguard.curate.forwards.attach_forward`:
premiums are USD (``usd_premium``), ``tau`` is in years, ``F``/``strike`` are the
Black-76 forward/strike, ``cp_sign`` is +1/-1 (bridging canonical ``"C"/"P"`` to
the solver's ``CallPut``), and ``iv_trade`` is a fraction (``nan`` when absent).
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import IntFlag

import polars as pl

from volguard.config import CurateConfig
from volguard.curate.blackiv import CallPut, black76_greeks, implied_vol

__all__ = [
    "QualityFlag",
    "apply_filters",
    "cross_check_iv",
    "mad_outlier_mask",
    "staleness_weight",
]

# UTC-millisecond datetimes, identical to the canonical/raw-layer schemas so
# ``snap_ts`` / ``source_ts`` arithmetic stays in one dtype (design Model 2).
_TS = pl.Datetime(time_unit="ms", time_zone="UTC")

# ``ln(2)`` for the recency half-life discount (design R5.2).
_LN2 = math.log(2.0)


class QualityFlag(IntFlag):
    """Bitmask of every rejection/flag reason for a curated row (design Model 2).

    Recorded in the ``quality_flags`` column so nothing is dropped silently: a
    row can accumulate several reasons at once (e.g. ``IV_DIVERGENCE`` plus a
    later band rejection). ``OK`` is the empty mask. Values are stable powers of
    two — downstream QC counts rely on the exact bit positions.

    Task 5 (IV cross-check) sets :attr:`IV_DIVERGENCE` and :attr:`IV_UNSOLVABLE`;
    the remaining members are the vocabulary the Task 6 filter cascade
    (band/MAD/size/block/staleness) ORs into the same mask.
    """

    OK = 0
    TAU_TOO_SHORT = 1
    DELTA_OUT_OF_BAND = 2
    IV_OUT_OF_BAND = 4
    IV_DIVERGENCE = 8  # |iv_trade - iv_recomputed| > cfg.iv_divergence_tol (retained)
    MAD_OUTLIER = 16
    BELOW_MIN_SIZE = 32
    BLOCK_TRADE = 64  # informative-but-wide; flagged, retained
    STALE = 128
    IV_UNSOLVABLE = 256  # implied_vol returned nan (no usable recomputed IV)


def _as_float(value: object) -> float:
    """Coerce a Polars scalar (``PythonLiteral | None``) to ``float``.

    ``None`` (a null cell) maps to ``nan`` so the finiteness guards below take
    the documented "no usable value" path rather than raising.
    """
    if value is None:
        return math.nan
    return float(value)  # type: ignore[arg-type]


def _cross_check_row(
    usd_premium: float,
    forward: float,
    strike: float,
    tau: float,
    cp_sign: int,
    iv_trade: float,
    tol: float,
) -> tuple[float, float, str, int]:
    """Row-level IV cross-check (design "IV recompute + cross-check" algorithm).

    Returns ``(iv_recomputed, iv_obs, iv_source, flags)`` where ``flags`` is the
    integer value of the :class:`QualityFlag` reasons raised for this row:

    - ``iv_recomputed = implied_vol(usd_premium, F, K, tau, cp_sign)`` — ``nan``
      when the premium is outside the no-arb bounds / inputs are non-finite.
    - ``iv_trade`` finite  -> ``iv_obs = iv_trade``,      ``iv_source = "trade"``.
    - else recomputed finite -> ``iv_obs = iv_recomputed``, ``iv_source = "mark"``.
    - else ``iv_obs = nan``, ``iv_source = "mark"`` (dropped later by the band).

    Flags: :attr:`QualityFlag.IV_UNSOLVABLE` when the solver returned ``nan``;
    :attr:`QualityFlag.IV_DIVERGENCE` (row retained) when both IVs are finite and
    ``|iv_trade - iv_recomputed| > tol``.
    """
    # cp_sign is the canonical +1/-1 bridge; narrow it to the solver's CallPut.
    cp: CallPut = 1 if cp_sign >= 0 else -1
    iv_recomputed = implied_vol(usd_premium, forward, strike, tau, cp)

    flags = QualityFlag.OK
    if math.isnan(iv_recomputed):
        flags |= QualityFlag.IV_UNSOLVABLE

    # Trust Deribit's per-trade iv as primary; else fall back to the recompute.
    if math.isfinite(iv_trade):
        iv_obs, iv_source = iv_trade, "trade"
        if math.isfinite(iv_recomputed) and abs(iv_trade - iv_recomputed) > tol:
            flags |= QualityFlag.IV_DIVERGENCE  # expected in deep wings; retained
    elif math.isfinite(iv_recomputed):
        iv_obs, iv_source = iv_recomputed, "mark"
    else:
        iv_obs, iv_source = math.nan, "mark"

    return iv_recomputed, iv_obs, iv_source, int(flags)


def cross_check_iv(rows: pl.DataFrame, cfg: CurateConfig) -> pl.DataFrame:
    """Recompute IV and cross-check it against Deribit's per-trade IV (design R3).

    Preconditions: ``rows`` carries the canonical + forward columns ``F``,
    ``strike``, ``tau``, ``cp_sign``, ``usd_premium``, and ``iv_trade`` (from
    :func:`~volguard.curate.forwards.attach_forward`). ``F > 0``, ``strike > 0``,
    ``tau > 0`` for well-formed rows (out-of-bound inputs yield a ``nan``
    recomputed IV and an :attr:`QualityFlag.IV_UNSOLVABLE` flag rather than a
    crash — :func:`implied_vol` is documented to return ``nan`` there).

    Adds four columns and returns a new frame:

    - ``iv_recomputed`` — the M1 Black-76 solve on ``usd_premium`` (may be ``nan``).
    - ``iv_obs``        — ``iv_trade`` when finite, else ``iv_recomputed``.
    - ``iv_source``     — ``"trade"`` / ``"mark"`` (always in ``{trade, mark, mid}``).
    - ``quality_flags`` — :class:`QualityFlag` bitmask; ORed into any existing
      ``quality_flags`` column so this step composes with later filter stages.

    ``implied_vol`` is a scalar Brent solve (the hot path), so this applies it
    per row; only well-formed rows reach here and the row counts are per-snap,
    so a readable Python pass over the rows is acceptable (design perf notes).
    """
    tol = cfg.iv_divergence_tol
    had_flags = "quality_flags" in rows.columns

    iv_recomputed: list[float] = []
    iv_obs: list[float] = []
    iv_source: list[str] = []
    flags: list[int] = []

    for row in rows.iter_rows(named=True):
        cp_sign = int(row["cp_sign"])
        rec, obs, src, flag = _cross_check_row(
            _as_float(row["usd_premium"]),
            _as_float(row["F"]),
            _as_float(row["strike"]),
            _as_float(row["tau"]),
            cp_sign,
            _as_float(row["iv_trade"]),
            tol,
        )
        prior = int(row["quality_flags"]) if had_flags else 0
        iv_recomputed.append(rec)
        iv_obs.append(obs)
        iv_source.append(src)
        flags.append(prior | flag)

    # Empty lists with an explicit dtype keep the columns well-typed on an
    # empty input frame (no rows to iterate), so no special-casing is needed.
    return rows.with_columns(
        pl.Series("iv_recomputed", iv_recomputed, dtype=pl.Float64),
        pl.Series("iv_obs", iv_obs, dtype=pl.Float64),
        pl.Series("iv_source", iv_source, dtype=pl.String),
        pl.Series("quality_flags", flags, dtype=pl.Int64),
    )


# --- Task 6.3: staleness weighting (design R5) ----------------------------


def staleness_weight(staleness_s: float, half_life_s: float) -> float:
    """Exponential recency weight for an observation ``staleness_s`` old (R5.2).

    ``weight = exp(-ln(2) * staleness_s / half_life_s)`` so the weight is
    ``1.0`` at zero staleness (R5.3), ``0.5`` at exactly one half-life (R5.4),
    strictly decreasing in ``staleness_s`` and bounded in ``(0, 1]`` for
    non-negative staleness (R5.5).

    Preconditions: ``staleness_s >= 0``, ``half_life_s > 0`` (guaranteed by the
    :class:`~volguard.config.CurateConfig` validator).
    """
    return math.exp(-_LN2 * staleness_s / half_life_s)


# --- Task 6.2: MAD outlier rejection on total variance (design R4.6) -------


def mad_outlier_mask(w: pl.Series, expiry: pl.Series, multiplier: float) -> pl.Series:
    """Boolean mask of per-expiry MAD outliers on total variance ``w`` (R4.6).

    For each expiry, a row is an outlier when its total variance ``w`` deviates
    from the expiry's median by more than ``multiplier`` times the median
    absolute deviation (MAD): ``|w - median_expiry(w)| > multiplier * MAD``.

    The median and MAD are computed over the finite ``w`` values of the expiry
    (independent of any band rejection), so the mask is a pure function of the
    input ``w``/``expiry`` values: it is stable across repeated application
    (supporting filter idempotence, CP3) and unaffected by band tightening
    (supporting filter monotonicity, CP4). Non-finite ``w`` (e.g. an unsolvable
    IV) is never flagged here — such rows are rejected by the IV band instead.

    Returns a :class:`polars.Series` of dtype ``Boolean`` aligned to the input
    row order (same length as ``w``).
    """
    n = w.len()
    if n == 0:
        return pl.Series("mad_outlier", [], dtype=pl.Boolean)

    frame = pl.DataFrame({"_w": w, "_expiry": expiry}).with_row_index("_idx")
    finite = frame.filter(pl.col("_w").is_finite())

    # Per-expiry median of w, then per-expiry MAD = median(|w - median|).
    median = finite.group_by("_expiry").agg(pl.col("_w").median().alias("_med"))
    with_dev = frame.join(median, on="_expiry", how="left").with_columns(
        (pl.col("_w") - pl.col("_med")).abs().alias("_absdev")
    )
    mad = (
        with_dev.filter(pl.col("_w").is_finite())
        .group_by("_expiry")
        .agg(pl.col("_absdev").median().alias("_mad"))
    )

    scored = (
        with_dev.join(mad, on="_expiry", how="left")
        .with_columns(
            (
                pl.col("_w").is_finite()
                & (pl.col("_absdev") > (multiplier * pl.col("_mad")))
            ).alias("_outlier")
        )
        .sort("_idx")
    )
    return scored["_outlier"].fill_null(value=False).rename("mad_outlier")


# --- Task 6.1: the band/size/block/MAD filter cascade (design R4) ----------

# Reasons that reject a row (removed from the curated surface). ``BLOCK_TRADE``
# and ``IV_DIVERGENCE`` are informative-but-retained (design R3.4/R4.5); a plain
# ``IV_UNSOLVABLE`` propagates to a NaN ``iv_obs`` which the IV band rejects.
_HARD_REJECT_MASK: int = (
    int(QualityFlag.TAU_TOO_SHORT)
    | int(QualityFlag.DELTA_OUT_OF_BAND)
    | int(QualityFlag.IV_OUT_OF_BAND)
    | int(QualityFlag.BELOW_MIN_SIZE)
    | int(QualityFlag.MAD_OUTLIER)
)


def _flag_when(pred: pl.Expr, flag: QualityFlag) -> pl.Expr:
    """A per-row Int64 expr equal to ``int(flag)`` where ``pred`` holds, else 0."""
    return (
        pl.when(pred)
        .then(pl.lit(int(flag), dtype=pl.Int64))
        .otherwise(pl.lit(0, dtype=pl.Int64))
    )


def _row_deltas(rows: pl.DataFrame) -> list[float]:
    """Black-76 delta per row via the trusted M1 pricer (design R4.2).

    ``implied_vol``/greeks are scalar solves, so this mirrors
    :func:`cross_check_iv` with a readable per-row pass. Rows with non-positive
    ``F``/``strike``/``tau`` or a non-finite ``iv_obs`` yield ``nan`` (the delta
    band then rejects them via ``is_between``); no crash on bad inputs.
    """
    deltas: list[float] = []
    for row in rows.iter_rows(named=True):
        forward = _as_float(row["F"])
        strike = _as_float(row["strike"])
        tau = _as_float(row["tau"])
        iv_obs = _as_float(row["iv_obs"])
        cp: CallPut = 1 if int(row["cp_sign"]) >= 0 else -1
        if forward > 0.0 and strike > 0.0 and tau > 0.0 and math.isfinite(iv_obs):
            deltas.append(black76_greeks(forward, strike, tau, iv_obs, cp)["delta"])
        else:
            deltas.append(math.nan)
    return deltas


def apply_filters(rows: pl.DataFrame, snap_ts: datetime, cfg: CurateConfig) -> pl.DataFrame:
    """Apply the liquidity/quality filter cascade and staleness weights (R4/R5).

    Preconditions: ``rows`` carries the canonical + forward + cross-check columns
    ``F``, ``strike``, ``tau``, ``iv_obs``, ``cp_sign``, ``size``, ``block_flag``,
    ``expiry``, ``source_ts`` (and usually ``quality_flags`` from
    :func:`cross_check_iv`; a missing column is treated as ``OK``/0).

    Adds/overwrites these columns and returns a new frame with **every input row
    preserved** (nothing is silently dropped — R4.7):

    - ``delta``       — Black-76 delta (design R4.2).
    - ``w``           — total variance ``iv_obs^2 * tau`` (the MAD statistic).
    - ``staleness_s`` — ``(snap_ts - source_ts)`` in seconds (design R5.1).
    - ``weight``      — recency weight ``exp(-ln2 * staleness_s / half_life)`` (R5.2).
    - ``quality_flags`` — the :class:`QualityFlag` bitmask, ORed with any prior
      flags: ``TAU_TOO_SHORT`` / ``DELTA_OUT_OF_BAND`` / ``IV_OUT_OF_BAND`` /
      ``BELOW_MIN_SIZE`` / ``MAD_OUTLIER`` (reject) and ``BLOCK_TRADE`` (retain).
    - ``rejected``    — ``True`` iff any hard-reject bit is set. The driver keeps
      only ``~rejected`` rows for the ``QUOTES_NORM`` output; retaining the
      rejected rows here keeps the stage fully auditable (R4.7).

    The cascade is **idempotent** (CP3) and **band-monotone** (CP4): flags are
    pure functions of each row's own values, and the MAD statistic is computed
    over all rows per expiry (independent of band rejection), so re-running the
    cascade — or tightening any band — never re-admits a previously rejected row.
    """
    had_flags = "quality_flags" in rows.columns
    base_flags = (
        pl.col("quality_flags").cast(pl.Int64) if had_flags else pl.lit(0, dtype=pl.Int64)
    )

    staleness_s = (
        (pl.lit(snap_ts).cast(_TS) - pl.col("source_ts")).dt.total_microseconds().cast(pl.Float64)
        / 1_000_000.0
    )

    # Delta needs the scalar M1 pricer; compute it first so the band predicate
    # below is a plain vectorized column comparison.
    stage = rows.with_columns(
        pl.Series("delta", _row_deltas(rows), dtype=pl.Float64),
        (pl.col("iv_obs") ** 2 * pl.col("tau")).alias("w"),
        staleness_s.alias("staleness_s"),
    ).with_columns(
        (-_LN2 * pl.col("staleness_s") / cfg.recency_half_life_s).exp().alias("weight"),
    )

    # Band / size / block flags (all pure per-row; NaN delta/iv fail is_between
    # and are correctly rejected by the delta/IV band).
    flag_expr = (
        base_flags
        | _flag_when(pl.col("tau") < cfg.tau_min_years, QualityFlag.TAU_TOO_SHORT)
        | _flag_when(
            ~pl.col("delta").abs().is_between(cfg.delta_min, cfg.delta_max),
            QualityFlag.DELTA_OUT_OF_BAND,
        )
        | _flag_when(
            ~pl.col("iv_obs").is_between(cfg.iv_min, cfg.iv_max),
            QualityFlag.IV_OUT_OF_BAND,
        )
        | _flag_when(
            pl.col("size").is_not_null() & (pl.col("size") < cfg.min_size_btc),
            QualityFlag.BELOW_MIN_SIZE,
        )
        | _flag_when(pl.col("block_flag").fill_null(value=False), QualityFlag.BLOCK_TRADE)
    )
    stage = stage.with_columns(flag_expr.alias("quality_flags"))

    # MAD outlier rejection on total variance, per expiry (design R4.6). Attach
    # the mask as a column so the flag update stays a vectorized column expr.
    mad_mask = mad_outlier_mask(stage["w"], stage["expiry"], cfg.mad_multiplier)
    stage = stage.with_columns(mad_mask.alias("_mad_outlier")).with_columns(
        (
            pl.col("quality_flags")
            | _flag_when(pl.col("_mad_outlier"), QualityFlag.MAD_OUTLIER)
        ).alias("quality_flags")
    )

    return stage.drop("_mad_outlier").with_columns(
        ((pl.col("quality_flags") & _HARD_REJECT_MASK) != 0).alias("rejected")
    )
