"""Tests for the rank/inverse-normal normalisation switch and the panel upgrade.

Offline, no network, no saved-panel dependency (a saved-panel test would skip on the
CEO's other machine, which is exactly the class of bare `return` bail-out the ship-gate
flagged -- so these are all built from synthetic frames instead).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import postBoRank as pbr


def _frame(n=200, seed=0):
    """A pool with the shapes the real metric columns have: a lognormal (fat right tail),
    a symmetric one, a discrete ordinal, and a column with real missingness."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "source": ["S%03d" % i for i in range(n)],
        "earnYield": rng.lognormal(0, 1.2, n),              # skewed
        "RoA": rng.normal(0, 1, n),                          # symmetric
        "Piotroski": rng.integers(0, 10, n).astype(float),   # discrete ordinal
        "currentRatio": np.where(rng.random(n) < 0.25, np.nan,
                                 rng.lognormal(0, 0.8, n)),  # 25% missing
    })


# --------------------------------------------------------------------------- #
#  _rank_normal properties                                                    #
# --------------------------------------------------------------------------- #
def test_rank_normal_is_centred_and_order_preserving():
    x = pd.Series([5.0, 1.0, 3.0, 100.0, 2.0])
    z = pbr._rank_normal(x)
    assert abs(float(z.mean())) < 1e-12, "the map is symmetric about p=0.5 -> mean 0"
    # order preserved exactly
    assert list(x.rank()) == list(z.rank())
    # and it is a NORMAL scale, not a percentile scale
    assert 0.5 < float(z.std(ddof=1)) < 1.5


def test_rank_normal_is_invariant_to_any_strictly_monotone_transform():
    """THE defining property: the output depends on the column's ORDER and nothing else,
    so no amount of tail-reshaping changes it."""
    x = pd.Series(np.random.default_rng(1).lognormal(0, 2, 300))
    a = pbr._rank_normal(x)
    for f in (np.log, np.sqrt, lambda v: v ** 3, lambda v: 1000 * v + 7):
        b = pbr._rank_normal(pd.Series(f(x)))
        assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_rank_normal_clipping_only_merges_the_clipped_tail():
    """The exact reason winsorization is INAPPLICABLE under this method (and the limit of
    the claim -- clipping is only WEAKLY monotone, so it does not leave the map untouched:
    it collapses the clipped names onto one tied score).  What matters is that it can never
    REORDER anyone and never touches an unclipped name."""
    x = pd.Series(np.random.default_rng(1).lognormal(0, 2, 300))
    cap = x.quantile(0.9)
    a = pbr._rank_normal(x)
    b = pbr._rank_normal(x.clip(upper=cap))
    unclipped = (x <= cap).to_numpy()
    assert np.allclose(a[unclipped].to_numpy(), b[unclipped].to_numpy())
    assert b[~unclipped].nunique() == 1, "the clipped tail becomes one tied score"
    # no inversions anywhere
    assert (a.rank().to_numpy() == b.rank(method="min").to_numpy()).sum() >= unclipped.sum()


def test_rank_normal_keeps_nan_and_does_not_let_it_consume_a_rank():
    x = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
    z = pbr._rank_normal(x)
    assert z.isna().tolist() == [False, True, False, True, False]
    # the 3 observed values occupy the 3 plotting positions of an n=3 sample
    assert abs(float(z.dropna().mean())) < 1e-12
    assert float(z.iloc[0]) < 0 < float(z.iloc[4])


def test_rank_normal_is_exactly_centred_ONLY_when_values_are_distinct():
    """Guards the docstring claim, which used to overstate this as "EXACTLY 0 by
    construction".  Distinct -> exact; tied -> displaced, and by a material amount."""
    distinct = pbr._rank_normal(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert abs(float(distinct.mean())) < 1e-12
    assert abs(float(distinct.median())) < 1e-12

    tied = pbr._rank_normal(pd.Series([1.0, 1.0, 2.0]))
    assert abs(float(tied.mean()) - 0.0260) < 1e-3, float(tied.mean())
    assert float(tied.mean()) != 0.0


def test_rank_normal_fill_of_zero_can_sit_far_off_median_on_a_discrete_column():
    """The residual N1 mechanism, as a test: a lopsided discrete column leaves the fill
    value 0 above most of the pool even after the rank map."""
    binary = pd.Series([0.0] * 60 + [1.0] * 40)
    z = pbr._rank_normal(binary)
    frac_below_fill = float((z < 0).mean())
    assert frac_below_fill >= 0.55, frac_below_fill      # ~60%, NOT 50%
    # and a 7-level column (the marketCapRevQuants shape: the SIX absolute market-cap
    # bands of stage2_metrics.MCAP_BAND_SCORES plus the 0.0 missing-cap sentinel -- it was
    # 5 while the metric was a pool quartile, register D-5 2026-08-06) is off-centre too
    seven = pd.Series(([-0.5] * 25) + ([-0.3] * 20) + ([-0.1] * 15) + ([0.0] * 10)
                      + ([0.1] * 15) + ([0.3] * 10) + ([0.5] * 5))
    zf = pbr._rank_normal(seven)
    assert abs(float((zf < 0).mean()) - 0.50) > 0.02


def test_rank_normal_ties_get_one_value():
    z = pbr._rank_normal(pd.Series([1.0, 1.0, 1.0, 2.0, 2.0]))
    assert z.iloc[0] == z.iloc[1] == z.iloc[2]
    assert z.iloc[3] == z.iloc[4]
    assert z.iloc[0] < z.iloc[3]


# --------------------------------------------------------------------------- #
#  The switch                                                                 #
# --------------------------------------------------------------------------- #
def test_zscore_is_the_default_and_is_unchanged_by_the_switch():
    """Regression guard: the shipped path must be bit-for-bit whether or not `method` is
    passed, because the deployed mu weights were tuned on it."""
    a, oa = pbr.normalizeAndDropNA(_frame().copy())
    b, ob = pbr.normalizeAndDropNA(_frame().copy(), method=pbr.NORM_ZSCORE)
    assert oa == ob
    num = [c for c in a.columns if c != "source"]
    assert np.allclose(a[num].to_numpy(dtype=float), b[num].to_numpy(dtype=float),
                       equal_nan=True)


def test_bad_method_is_refused_loudly():
    with pytest.raises(ValueError):
        pbr.normalizeAndDropNA(_frame().copy(), method="quantile")


def test_BOTH_modes_now_put_the_missing_data_fill_on_the_median():
    """Finding N1, as a test -- AND THE Z-PATH NOW WINS IT, which is why this test was
    inverted on 2026-08-03 rather than deleted.

    IT USED TO ASSERT THE DEFECT.  The old body was `assert pct_z > 0.55` -- i.e. it REQUIRED
    that under z-scoring a missing `currentRatio` scored above the typical name, because 0 was
    the winsorized MEAN of a right-skewed column, and that was the argument for the rank map.
    The E-1 change (postBoRank.HUBER_C / normalizeAndDropNA) centres every column on its
    observed MEDIAN, so 0 IS the median and the defect is gone from the z-path.  A test pinning
    a fixed defect passes only until the defect is fixed; leaving it would have made a genuine
    improvement look like a regression.

    THE Z-PATH IS NOW THE STRICTER OF THE TWO, and that is the substance here.  z = (x - median)
    / sigma has median exactly 0, and the squash is monotone with zeta(0) = 0, so it survives.
    The RANK map centres exactly only when a column's values are DISTINCT -- ties displace the
    centre (see `_rank_normal`'s docstring) -- so it is approximate wherever it matters most."""
    zs, _ = pbr.normalizeAndDropNA(_frame().copy(), method=pbr.NORM_ZSCORE)
    rk, _ = pbr.normalizeAndDropNA(_frame().copy(), method=pbr.NORM_RANK)
    col = "currentRatio"
    # percentile the fill value occupies among the rows that HAVE the metric
    have = _frame()[col].notna().to_numpy()
    pct_z = float((zs.loc[have, col].astype(float) < 0).mean())
    pct_r = float((rk.loc[have, col].astype(float) < 0).mean())
    assert abs(pct_z - 0.50) < 0.02, \
        "robust z: the fill must sit AT the observed median (%.3f)" % pct_z
    assert abs(pct_r - 0.50) < 0.02, "rank: the fill IS the median (%.3f)" % pct_r
    # exact on the z-path: the observed median of the normalised column is 0 itself
    assert abs(float(zs.loc[have, col].astype(float).median())) < 1e-12


def test_rank_mode_never_drops_or_reorders_rows():
    f = _frame()
    rk, dropped = pbr.normalizeAndDropNA(f.copy(), method=pbr.NORM_RANK)
    assert dropped == []
    assert list(rk["source"]) == list(f["source"])


def test_rank_mode_leaves_bounded_columns_raw_when_asked():
    f = _frame()
    on, _ = pbr.normalizeAndDropNA(f.copy(), method=pbr.NORM_RANK, rank_bounded=True)
    off, _ = pbr.normalizeAndDropNA(f.copy(), method=pbr.NORM_RANK, rank_bounded=False)
    #  Piotroski is in BOUNDED_DISCRETE_COLUMNS (which until the 2026-08-03 E-1 change was
    #  `WINSOR_EXEMPT_BOUNDED`; it is no longer a NORMALISATION exemption -- the z-path now
    #  applies the same robust ruler to every column -- and `rank_bounded=False` is the only
    #  thing left that reads it).
    assert "Piotroski" in pbr.BOUNDED_DISCRETE_COLUMNS
    assert not hasattr(pbr, "WINSOR_EXEMPT_BOUNDED"), \
        "the old exemption name is back: a name list must not gate the z-path again"
    assert np.allclose(off["Piotroski"].astype(float), f["Piotroski"].astype(float))
    assert not np.allclose(on["Piotroski"].astype(float), f["Piotroski"].astype(float))
    # a NON-exempt column is mapped either way
    assert np.allclose(on["RoA"].astype(float), off["RoA"].astype(float))


def test_all_nan_row_is_still_dropped_under_rank():
    f = _frame(n=50)
    f.loc[7, ["earnYield", "RoA", "Piotroski", "currentRatio"]] = np.nan
    rk, dropped = pbr.normalizeAndDropNA(f.copy(), method=pbr.NORM_RANK)
    assert dropped == ["S007"]
    assert "S007" not in set(rk["source"])


def test_rank_mode_is_scored_end_to_end_by_getAggScore():
    f = _frame()
    rk, _ = pbr.normalizeAndDropNA(f.copy(), method=pbr.NORM_RANK)
    ranked = pbr.getAggScore(rk.copy())
    assert "AggScore" in ranked.columns
    assert ranked["AggScore"].notna().all()
    # descending, as getAggScore promises
    assert ranked["AggScore"].is_monotonic_decreasing


# --------------------------------------------------------------------------- #
#  Extracted production helpers still behave as the fetch loop needs          #
# --------------------------------------------------------------------------- #
def test_build_bometric_rows_trims_the_oldest_rpy_rows():
    import getData_fmp as gdf
    import createDicts as cdic
    d = cdic.getDicts()
    packed = (d[2], d[3], d[5], d[4], d[6])
    n = 12
    # NEWEST-FIRST, as the contract requires
    dates = pd.date_range("2024-10-01", periods=n, freq="-3MS")
    tf = pd.DataFrame({"date": dates, "source": "X"})
    # every column the Stage-1 construction can read, taken from the SAME dict the ingest
    # uses -- so the test cannot silently rot when a criterion gains an input
    inputs = set()
    for k, cols in d[0].items():
        inputs.update(cols)
    for spec in (d[2], d[3], d[4], d[5]):
        for v in spec.values():
            inputs.update([v["Upper"], v["Lower"]])
    inputs.discard("Identity")
    inputs.discard("date")
    for c in sorted(inputs):
        tf[c] = np.linspace(10, 100, n)
    for rpy in (2, 4):
        tmp = pd.DataFrame(columns=["date", "source"])
        tmp["date"] = tf["date"].values
        tmp["source"] = "X"
        out = gdf.build_bometric_rows(tf.copy(), tmp.copy(), rpy, dicts=packed)
        assert len(out) == n - rpy, "must drop exactly rpy rows, got %d" % (n - len(out))
        # the dropped rows are the OLDEST (the frame is newest-first, so the tail)
        assert out["date"].min() == dates[n - rpy - 1]


def test_stamp_frequency_and_graham_stamps_both_fields():
    import getData_fmp as gdf
    import reporting_period as rp
    n = 8
    tf = pd.DataFrame({
        "date": pd.date_range("2024-10-01", periods=n, freq="-3MS"),
        "netIncome": np.full(n, 100.0),
        "weightedAverageShsOut": np.full(n, 50.0),
        "bookValuePerShare": np.full(n, 9.0),
    })
    out = gdf.stamp_frequency_and_graham(tf.copy())
    assert rp.FREQ_COLUMN in out.columns and out[rp.FREQ_COLUMN].iloc[0] == rp.QUARTERLY
    assert "grahamUndefinedReason" in out.columns
    # EPS_ttm = 4*100/50 = 8; graham = sqrt(22.5*8*9) = 40.249...
    assert abs(float(out["grahamNumber"].iloc[0]) - np.sqrt(22.5 * 8.0 * 9.0)) < 1e-9
    # the 3 oldest rows have no full trailing year -> undefined for missing inputs
    assert out["grahamNumber"].isna().sum() == 3
    assert (out["grahamUndefinedReason"].iloc[-3:] == "graham_missing_inputs").all()


def test_stamp_frequency_and_graham_refuses_negative_earnings_and_book_value():
    import getData_fmp as gdf
    n = 8
    base = dict(date=pd.date_range("2024-10-01", periods=n, freq="-3MS"),
                weightedAverageShsOut=np.full(n, 50.0))
    neg_eps = pd.DataFrame({**base, "netIncome": np.full(n, -100.0),
                            "bookValuePerShare": np.full(n, 9.0)})
    neg_bv = pd.DataFrame({**base, "netIncome": np.full(n, 100.0),
                           "bookValuePerShare": np.full(n, -9.0)})
    a = gdf.stamp_frequency_and_graham(neg_eps)
    b = gdf.stamp_frequency_and_graham(neg_bv)
    assert a["grahamNumber"].isna().all() and b["grahamNumber"].isna().all()
    assert (a["grahamUndefinedReason"].iloc[:5] == "graham_undefined_negative_eps").all()
    assert (b["grahamUndefinedReason"].iloc[:5] == "graham_undefined_negative_bv").all()


def test_panel_upgrade_puts_a_synthetic_old_basis_panel_onto_the_new_basis():
    """The whole point of panel_upgrade, on a frame small enough to check by hand:
    an OLD-basis panel reads marketCap/(price*shares) ~ 4; after the upgrade it reads 1."""
    import panel_upgrade as pu
    import run_target_test as rtt
    rng = np.random.default_rng(3)
    rows = []
    for s in range(40):
        n = 10
        mc = rng.uniform(1e8, 1e9)
        sh = rng.uniform(1e6, 1e7)
        for i in range(n):
            rows.append({"source": "S%02d" % s,
                         "date": pd.Timestamp("2020-01-01") + pd.DateOffset(months=3 * i),
                         "marketCap": mc, "weightedAverageShsOut": sh,
                         "price": (mc / sh) / 4.0,          # the OLD quarterly-PE basis
                         "netIncome": rng.uniform(1e6, 1e7),
                         "bookValuePerShare": rng.uniform(1, 20)})
    cdx = pd.DataFrame(rows)
    assert rtt.detect_price_basis(cdx)[0] == "old"
    up = pu.upgrade_cdx(cdx, verbose=False)
    basis, med = rtt.detect_price_basis(up)
    assert basis == "new" and abs(med - 1.0) < 1e-9
    # order and length preserved; the new fields are stamped
    assert list(up["source"]) == list(cdx["source"])
    assert len(up) == len(cdx)
    assert "grahamNumber" in up.columns and "reportingFrequency" in up.columns
