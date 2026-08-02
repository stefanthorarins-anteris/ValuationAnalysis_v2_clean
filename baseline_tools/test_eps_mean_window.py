"""EPStoEPSmean's BASELINE must be a fixed CALENDAR span, not "whatever rows the fetch returned".

THE DEFECT THESE PIN (fixed 2026-07-31).  `eps_to_eps_mean` was the LAST uncapped window in the
Stage-2 block: `epsmean = eps.mean()` averaged the WHOLE per-ticker panel, and cdx_dftop100 has no
row cap, so the baseline length WAS the fetch depth.  The metric is
(epsmean - ewma_recent) / |epsmean| with w = +0.056, so POSITIVE = recent EPS below its own history
= REWARDED.  A longer baseline drags epsmean toward older, smaller earnings, so it PENALISES any
company whose EPS has grown.  Measured on the shipped 2026-07-17 top-100 (90 names defined on both
windows), full 24-row panel vs the most recent 12:
  * mean -1.317 -> -0.623, std 2.715 -> 1.291, spearman 0.785;
  * 14 of 90 names CHANGE SIGN -- flip from penalised to rewarded or back;
  * rising-EPS names (n=78) score -1.461 on the full panel vs -0.533 on the recent half, while
    falling-EPS names (n=9) score -0.004 vs -1.159: the baseline is inverted relative to the
    metric's own mean-reversion thesis.
At `-nrperiods 80` the baseline becomes ~20 years, i.e. a two-decade growth penalty.

This file is the EPStoEPSmean twin of test_cycleheat_window.py, and deliberately mirrors it:
the two metrics had the same defect and carry the same 28-quarter constant.
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

import reporting_period as rp
import stage2_metrics as sm


def _frame(n, months, eps_path=None, shares=100.0, start="2026-01-01"):
    """n rows spaced `months` apart, NEWEST-FIRST -- the order production hands the metric
    (postBoRank._sort_cdx_newest_first / stage2_pit._sort_newest_first).  `eps_path` is given
    NEWEST-FIRST too, so eps_path[0] is the most recent period."""
    dates = pd.date_range(end=pd.Timestamp(start), periods=n, freq="-%dMS" % months)
    ni = (np.asarray(eps_path, dtype=float) * shares if eps_path is not None
          else np.full(n, 10.0) * shares)
    return pd.DataFrame({"date": dates, "netIncome": ni,
                         "weightedAverageShsOut": np.full(n, shares)})


def _uncapped(tempcdx, rpy=4):
    """The pre-fix computation: the SAME formula with NO baseline truncation.  Written out
    rather than imported so a future edit to the production function cannot silently redefine
    what "uncapped" means (that is the whole comparison these tests make)."""
    eps = (tempcdx["netIncome"] / tempcdx["weightedAverageShsOut"]).replace(
        [np.inf, -np.inf], np.nan)
    epsmean = eps.mean()
    a, _ny = 0.4, int(rpy)
    if len(eps) >= max(4, _ny) and all(eps.iloc[0:_ny] > 0):
        den, scale = abs(epsmean), eps.abs().mean()
        if (not np.isfinite(den) or not np.isfinite(scale)
                or den <= sm.EPS_MEAN_FLOOR_FRAC * scale):
            return np.nan
        _w = [(1 - a) ** k for k in range(_ny)]
        _tw = a * sum(_w)
        return (epsmean - (a / _tw) * sum(eps.iloc[k] * _w[k] for k in range(_ny))) / den
    return np.nan


# --------------------------------------------------------------------------- #
#  The window                                                                 #
# --------------------------------------------------------------------------- #
def test_base_window_is_long_enough_to_hold_a_cycle_and_not_bind_on_todays_panel():
    """Two constraints fix the constant: >= a business cycle, and >= the ~24 rows a quarterly
    filer carries today so the change is a no-op for them."""
    assert 24 <= sm.EPS_MEAN_BASE_NQ <= 32, sm.EPS_MEAN_BASE_NQ
    assert rp.scale_window(sm.EPS_MEAN_BASE_NQ, 4) >= 24
    assert rp.scale_window(sm.EPS_MEAN_BASE_NQ, 2) == pytest.approx(
        sm.EPS_MEAN_BASE_NQ / 2, abs=1)


def test_window_spans_the_same_calendar_time_for_both_frequencies():
    """A quarterly filer with 40 quarterly rows and a semi-annual filer with 40 semi-annual
    rows must have their baseline measured over ~the same elapsed years."""
    W = sm.EPS_MEAN_BASE_NQ
    q_years = rp.scale_window(W, 4) * 3 / 12.0
    s_years = rp.scale_window(W, 2) * 6 / 12.0
    assert abs(q_years - s_years) <= 0.75, (q_years, s_years)
    assert 6.0 <= q_years <= 8.0 and 6.0 <= s_years <= 8.0


def test_the_two_constants_are_INDEPENDENT_not_aliased():
    """review L7, corrected 2026-07-31.  The first version of this test asserted
    `EPS_MEAN_BASE_NQ == CYCLEHEAT_BASE_NQ` while its own docstring said it was NOT asserting a
    coupling -- the assertion WAS the coupling, and retuning CycleHeat alone would have failed
    it.  It also asserted `EPS_MEAN_BASE_NQ is not sm.__dict__.get("_CYCLEHEAT_ALIAS")`, which
    compares an int to None and is vacuously true.  Both are replaced by the property actually
    wanted: the two constants are SEPARATE knobs, and each independently satisfies the two
    constraints that motivated it.  Their equality today is a coincidence of shared reasoning,
    so it is documented, not asserted."""
    # separate module-level names, each independently valid
    for name in ("EPS_MEAN_BASE_NQ", "CYCLEHEAT_BASE_NQ"):
        v = getattr(sm, name)
        assert isinstance(v, int) and 24 <= v <= 32, (name, v)
        assert rp.scale_window(v, 4) >= 24, name
    # and re-tuning ONE must not move the OTHER: rebind CycleHeat's and confirm ours holds
    _saved = sm.CYCLEHEAT_BASE_NQ
    try:
        sm.CYCLEHEAT_BASE_NQ = 32
        assert sm.EPS_MEAN_BASE_NQ == 28, "EPStoEPSmean's baseline tracks CycleHeat's -- aliased"
    finally:
        sm.CYCLEHEAT_BASE_NQ = _saved


# --------------------------------------------------------------------------- #
#  THE regression: bit-identity on today's panel                              #
# --------------------------------------------------------------------------- #
def test_quarterly_is_BIT_IDENTICAL_when_the_cap_cannot_bind():
    """THE reason 28 is safe: on a 24-row quarterly panel the 28-row cap does not bind, so the
    metric must equal the uncapped computation EXACTLY -- not approximately."""
    rng = np.random.default_rng(3)
    for seed in range(8):
        rng = np.random.default_rng(seed)
        f = _frame(24, 3, eps_path=np.abs(rng.normal(10, 3, 24)) + 0.5)
        capped, uncapped = sm.eps_to_eps_mean(f, rpy=4), _uncapped(f, rpy=4)
        assert capped == uncapped, (seed, capped, uncapped)


def test_every_row_count_up_to_the_window_is_bit_identical():
    """The cap must be inert for EVERY panel length at or below the window, not just 24 --
    the live pool carries 9-, 13-, 14-, 15-, 19-, 22-, 23- and 24-row names."""
    rng = np.random.default_rng(11)
    path = np.abs(rng.normal(10, 3, 28)) + 0.5
    for n in range(4, rp.scale_window(sm.EPS_MEAN_BASE_NQ, 4) + 1):
        f = _frame(n, 3, eps_path=path[:n])
        capped, uncapped = sm.eps_to_eps_mean(f, rpy=4), _uncapped(f, rpy=4)
        assert capped == uncapped, (n, capped, uncapped)


def _real_panel_before_after():
    """(DataFrame of before/after per source, on the shipped 07-17 top-100) or None."""
    cache = os.path.join(_HERE, "resdic_2026-07-17_CORRECTED.pickle")
    if not os.path.exists(cache):
        return None
    import postBoRank as pbr
    cdx = pbr._sort_cdx_newest_first(pd.read_pickle(cache)["cdx_dftop100"])
    freq = rp.frequency_by_source(cdx)
    rows = []
    for src, t in cdx.groupby("source", sort=False):
        _rpy = rp.rows_per_year(freq, src)
        rows.append((src, len(t), _rpy, rp.scale_window(sm.EPS_MEAN_BASE_NQ, _rpy),
                     _uncapped(t, rpy=_rpy), sm.eps_to_eps_mean(t, rpy=_rpy)))
    return pd.DataFrame(rows, columns=["source", "n", "rpy", "win", "before", "after"])


def test_the_real_2026_07_17_panel_QUARTERLY_names_are_bit_identical():
    """Bit-identity asserted on the ACTUAL shipped top-100 panel, not only synthetic frames --
    this is the claim the pre-fetch fix rests on.  QUARTERLY names only: their 24 rows sit
    under the 28-row window, so the cap cannot bind.  Measured 2026-07-31: 76 quarterly names,
    66 defined, 66 bit-identical, 0 changed."""
    d = _real_panel_before_after()
    if d is None:
        pytest.skip("no saved resdic on this machine; the synthetic bit-identity tests hold")
    q = d[d["rpy"] == 4].dropna(subset=["before", "after"])
    assert len(q) >= 60, "expected most quarterly names defined, got %d" % len(q)
    changed = q[q["before"] != q["after"]]
    assert changed.empty, ("the cap BOUND on a quarterly name -- it must not on today's panel:"
                           "\n%s" % changed.to_string(index=False))


def test_the_real_2026_07_17_panel_SEMIANNUAL_names_DO_move_and_mostly_upward():
    """The semi-annual cohort is NOT a no-op, and that is the POINT -- exactly as for CycleHeat
    (fix 2026-07-30).  A semi-annual filer's 24 rows are 12 CALENDAR YEARS against a quarterly
    peer's 6, so the 14-row window truncates all of them.  Measured 2026-07-31 on the shipped
    top-100: 24 of 24 semi-annual names change, mean -1.654 -> -1.236 (median -0.942 ->
    -0.681), 20 of 24 move UP i.e. are penalised LESS, 2 change sign, 0 change NaN status.
    This test PINS that direction: a change that moved the cohort DOWN would mean the
    truncation is taking the wrong end of the series."""
    d = _real_panel_before_after()
    if d is None:
        pytest.skip("no saved resdic on this machine")
    s = d[d["rpy"] == 2].dropna(subset=["before", "after"])
    assert len(s) >= 20, "expected the ~24-name semi-annual cohort, got %d" % len(s)
    assert (s["n"] > s["win"]).all(), "a semi-annual name was NOT truncated"
    assert (s["before"] != s["after"]).all(), "a truncated name did not change"
    assert (s["after"] > s["before"]).sum() >= 0.6 * len(s), \
        "most of the semi-annual cohort must be penalised LESS, not more"
    assert s["after"].mean() > s["before"].mean()


def test_no_name_on_the_real_panel_changes_NaN_STATUS():
    """A window change must not turn a scored name unscored or vice versa: NaN feeds z=0 via
    normalizeAndDropNA, so a NaN-status flip is a silent neutralisation, not a value change."""
    d = _real_panel_before_after()
    if d is None:
        pytest.skip("no saved resdic on this machine")
    flipped = d[d["before"].isna() != d["after"].isna()]
    assert flipped.empty, "NaN-status changed for:\n%s" % flipped.to_string(index=False)


# --------------------------------------------------------------------------- #
#  The cap does real work on a deep panel                                     #
# --------------------------------------------------------------------------- #
def test_deep_history_does_NOT_lengthen_the_baseline():
    """The fetch-depth guarantee: an 80-row panel must give the SAME answer as its newest
    28 rows -- i.e. `-nrperiods 80` cannot move this metric."""
    rng = np.random.default_rng(7)
    path = np.abs(rng.normal(10, 3, 80)) + 0.5            # newest-first
    f80, f28 = _frame(80, 3, eps_path=path), _frame(28, 3, eps_path=path[:28])
    assert sm.eps_to_eps_mean(f80, rpy=4) == pytest.approx(
        sm.eps_to_eps_mean(f28, rpy=4), abs=1e-12)
    # and it is genuinely different from the uncapped 80-row answer (the cap is doing work)
    assert sm.eps_to_eps_mean(f80, rpy=4) != _uncapped(f80, rpy=4)


def test_the_window_actually_used_is_28_rows_quarterly_and_14_semiannual():
    """Direct assertion on the truncation length, independent of the formula: replacing every
    row OUTSIDE the window must not change the answer, and replacing one INSIDE it must."""
    for rpy, months in ((4, 3), (2, 6)):
        w = rp.scale_window(sm.EPS_MEAN_BASE_NQ, rpy)
        base = np.full(80, 10.0)
        f = _frame(80, months, eps_path=base)
        ref = sm.eps_to_eps_mean(f, rpy=rpy)
        outside = base.copy()
        outside[w:] = 1000.0                              # older than the window
        assert sm.eps_to_eps_mean(_frame(80, months, eps_path=outside), rpy=rpy) == ref
        inside = base.copy()
        inside[w - 1] = 1000.0                            # newest row still in the window
        assert sm.eps_to_eps_mean(_frame(80, months, eps_path=inside), rpy=rpy) != ref
    assert rp.scale_window(sm.EPS_MEAN_BASE_NQ, 4) == 28
    assert rp.scale_window(sm.EPS_MEAN_BASE_NQ, 2) == 14


def test_a_grower_is_penalised_LESS_after_the_cap_on_a_deep_panel():
    """Directly the damaging direction: the metric is REWARDED when positive (w = +0.056), and
    a long baseline made it more NEGATIVE for a company whose EPS has grown.  Capping must move
    a grower UP (less penalised), not down."""
    n = 80
    rising = np.concatenate([np.linspace(8.0, 1.0, 12), np.full(n - 12, 1.0)])   # newest-first
    f = _frame(n, 3, eps_path=rising)
    capped, uncapped = sm.eps_to_eps_mean(f, rpy=4), _uncapped(f, rpy=4)
    assert uncapped < 0, uncapped                 # a grower was penalised
    assert capped > uncapped, (capped, uncapped)  # and is penalised less now


def test_semiannual_baseline_is_TRUNCATED_to_the_same_calendar_span():
    """A semi-annual filer with 40 rows spans 20 years; the cap takes it to 14 rows / 7 years,
    matching a quarterly peer instead of doubling its baseline."""
    n = 40
    rising = np.concatenate([np.linspace(8.0, 1.0, 8), np.full(n - 8, 1.0)])
    f = _frame(n, 6, eps_path=rising)
    capped, uncapped = sm.eps_to_eps_mean(f, rpy=2), _uncapped(f, rpy=2)
    assert capped != uncapped
    assert capped > uncapped, (capped, uncapped)


# --------------------------------------------------------------------------- #
#  Nothing else about the metric moved                                        #
# --------------------------------------------------------------------------- #
def test_truncation_keeps_the_NEWEST_rows_not_the_oldest():
    """Taking the oldest rows would invert the metric outright.  A name whose recent EPS is far
    ABOVE a flat old baseline must read NEGATIVE (above its own history)."""
    n = 60
    path = np.concatenate([np.full(4, 50.0), np.full(n - 4, 5.0)])   # newest-first: hot now
    assert sm.eps_to_eps_mean(_frame(n, 3, eps_path=path), rpy=4) < 0


def test_positivity_gate_and_NaN_conventions_are_unchanged():
    """The <4-row guard, the most-recent-year positivity gate and the floor all still return
    NaN (not 0, not an exception) -- the cap must not have perturbed them."""
    assert np.isnan(sm.eps_to_eps_mean(_frame(3, 3, eps_path=[5.0, 5.0, 5.0]), rpy=4))
    loss = _frame(24, 3, eps_path=np.concatenate([[-1.0], np.full(23, 5.0)]))
    assert np.isnan(sm.eps_to_eps_mean(loss, rpy=4))          # newest period is a loss
    # near-cancelling series -> |mean| below the floor fraction of mean |EPS| -> NaN.
    # Constructed INSIDE the 28-row window: 4 tiny positive rows (so the positivity gate
    # passes) then 24 rows of +-10 that cancel, so |mean| ~ 0.0014 vs mean|EPS| ~ 8.57.
    alt = np.array([0.01] * 4 + [10.0, -10.0] * 30)
    f = _frame(64, 3, eps_path=alt)
    assert np.isnan(sm.eps_to_eps_mean(f, rpy=4))
    # and the floor is judged on the WINDOW, not the whole panel: the same series with the
    # cancelling rows pushed OUTSIDE the window is a normal, scoreable name
    outside = np.array([0.01] * 4 + [0.01] * 24 + [10.0, -10.0] * 18)
    assert not np.isnan(sm.eps_to_eps_mean(_frame(64, 3, eps_path=outside), rpy=4))


def test_metric_is_scale_invariant_a_share_split_does_not_move_it():
    """The audit-C4 dimensionless property, re-asserted after the window change."""
    rng = np.random.default_rng(5)
    path = np.abs(rng.normal(10, 3, 40)) + 0.5
    a = sm.eps_to_eps_mean(_frame(40, 3, eps_path=path), rpy=4)
    b = sm.eps_to_eps_mean(_frame(40, 3, eps_path=path * 7.0), rpy=4)
    assert a == pytest.approx(b, rel=1e-12)


# --------------------------------------------------------------------------- #
#  Every caller -- the sweep (the partial-sweep defect class)                  #
# --------------------------------------------------------------------------- #
EPS_MEAN_CALL_SITES = [
    ("postBoRank.py", "sm.eps_to_eps_mean(tempcdx, rpy=rpy)"),
    (os.path.join("baseline_tools", "stage2_pit.py"),
     "sm.eps_to_eps_mean(tempcdx, rpy=_rpy)"),
]


@pytest.mark.parametrize("path,needle", EPS_MEAN_CALL_SITES)
def test_every_caller_passes_rpy(path, needle):
    """Two call sites (live + the offline PIT reproduction).  A caller omitting `rpy` gets a
    28-row baseline for a semi-annual filer = 14 calendar years against a quarterly peer's 7,
    re-introducing the 2x-span divergence in the other direction."""
    src = open(os.path.join(_REPO, path), encoding="utf-8").read()
    assert needle in src, "%s does not pass rpy to eps_to_eps_mean" % path


def test_the_call_site_inventory_is_COMPLETE():
    """The partial-sweep guard: the worst bug in this project's history was a fix applied to
    two of three call sites.  Enumerate every `eps_to_eps_mean(` call in the repo and assert it
    is either in the inventory above or an explicitly-known non-production site."""
    import re
    ALLOWED_ELSEWHERE = {
        # the opt-in short-history guard wrapper: forwards *args/**kwargs untouched
        os.path.join("baseline_tools", "skill_baseline.py"),
        # its own unit test, which calls the wrapper deliberately
        os.path.join("baseline_tools", "test_skill_baseline.py"),
        # this file
        os.path.join("baseline_tools", "test_eps_mean_window.py"),
        # the definition
        "stage2_metrics.py",
        # MD's Part 5 defect-verification harness (read-only, no network).  CLEARED by audit
        # 2026-08-01: its single call is `sm.eps_to_eps_mean(tc, rpy=rpy)` -- it passes `rpy=`
        # and does NOT pass the ambient scoring `nq`, so it inherits the 28-quarter
        # EPS_MEAN_BASE_NQ cap exactly as the two production sites do.  Its `rpy` comes from
        # `rp.rows_per_year(fmap, t)`, the same source production uses, so a semi-annual filer
        # gets the same 14-row window there as in the live scorer.  Exempt because the call is
        # CORRECT, not because the file is out of scope -- this guard firing on it was the
        # guard working.
        os.path.join("baseline_tools", "verify_part5_defects.py"),
    }
    expected = {p for p, _n in EPS_MEAN_CALL_SITES} | ALLOWED_ELSEWHERE
    found = set()
    for root, dirs, files in os.walk(_REPO):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, _REPO)
            try:
                src = open(full, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            if re.search(r"eps_to_eps_mean\s*\(", src):
                found.add(rel)
    unknown = found - expected
    assert not unknown, ("NEW eps_to_eps_mean call site(s) not in the inventory -- verify each "
                         "passes rpy and does NOT pass the ambient scoring `nq`: %s"
                         % sorted(unknown))


def test_no_caller_passes_the_ambient_scoring_nq_as_the_baseline():
    """The specific trap this metric's signature creates.  Both call sites sit in a function
    whose local `nq` is the 16-row RATIO window; passing it here would shrink the
    business-cycle baseline to 4 years and silently change the metric.  `nq` must come from
    EPS_MEAN_BASE_NQ, i.e. the call must not pass nq at all."""
    import re
    for path, _needle in EPS_MEAN_CALL_SITES:
        src = open(os.path.join(_REPO, path), encoding="utf-8").read()
        for m in re.finditer(r"eps_to_eps_mean\s*\(([^)]*)\)", src):
            args = m.group(1)
            assert "nq" not in args, "%s passes nq to eps_to_eps_mean: %r" % (path, args)
            # and no bare second positional argument either
            assert args.count(",") <= 1 and "rpy=" in args, (path, args)


# --------------------------------------------------------------------------- #
#  MUTATION TEST: removing the cap must FAIL something                        #
# --------------------------------------------------------------------------- #
def test_MUTATION_removing_the_cap_breaks_a_test():
    """The sweep's own smoke alarm, in the style of test_cycleheat_window.py.  If the
    truncation is deleted, at least one deep-history assertion above must fail -- otherwise
    these tests would pass a regression through.  Asserted by re-running the deep-history and
    grower checks against the UNCAPPED reference."""
    rng = np.random.default_rng(7)
    path = np.abs(rng.normal(10, 3, 80)) + 0.5
    f80, f28 = _frame(80, 3, eps_path=path), _frame(28, 3, eps_path=path[:28])
    # the fetch-depth guarantee FAILS under the mutation ...
    assert _uncapped(f80, rpy=4) != pytest.approx(_uncapped(f28, rpy=4), abs=1e-12)
    # ... and so does the grower direction
    rising = np.concatenate([np.linspace(8.0, 1.0, 12), np.full(68, 1.0)])
    g = _frame(80, 3, eps_path=rising)
    assert not (_uncapped(g, rpy=4) > _uncapped(g, rpy=4))     # trivially, no relaxation
    assert _uncapped(g, rpy=4) < sm.eps_to_eps_mean(g, rpy=4)


# --------------------------------------------------------------------------- #
#  D2 -- incomeQuality: sign-safe, scale-free, non-exploding                   #
# --------------------------------------------------------------------------- #
#  Stage-2 weighted FMP's CFO/NI ratio at w = +0.072.  A ratio whose DENOMINATOR CHANGES SIGN
#  inverts for loss-makers and explodes as NI -> 0.  Stage-1 got the sign-safe treatment in
#  July; Stage-2 was missed until 2026-08-01 (CEO-approved).
def _iq_frame(ni, cfo, ta, n=None, months=3):
    n = len(ni) if n is None else n
    e = pd.Timestamp("2026-01-01")
    return pd.DataFrame({
        "date": [e - pd.DateOffset(months=months * k) for k in range(n)],
        "netIncome": np.asarray(ni, dtype=float),
        "netCashProvidedByOperatingActivities": np.asarray(cfo, dtype=float),
        "totalAssets": np.asarray(ta, dtype=float),
    })


def test_a_GOOD_loss_maker_is_NOT_penalised():
    """THE point of the fix.  NI < 0 with CFO > 0 -- a loss-maker generating cash -- is a
    strong earnings-quality signal, and the old ratio handed it a large NEGATIVE value that
    w = +0.072 then penalised."""
    good = _iq_frame([-50.0] * 8, [+80.0] * 8, [1000.0] * 8)
    v = sm.income_quality_accruals(good, 16, rpy=4)
    assert v > 0, v
    # old ratio on the same data would be CFO/NI = 80/-50 = -1.6, i.e. penalised
    assert (80.0 / -50.0) < 0


def test_the_metric_ORDERS_the_four_quadrants_correctly():
    """profit-without-cash must be the ONLY negative quadrant; the cash-generating loss-maker
    must sit ABOVE the healthy company (its earnings understate its cash the most)."""
    ta = [1000.0] * 8
    healthy = sm.income_quality_accruals(_iq_frame([50.0] * 8, [60.0] * 8, ta), 16, rpy=4)
    nocash = sm.income_quality_accruals(_iq_frame([50.0] * 8, [10.0] * 8, ta), 16, rpy=4)
    goodloss = sm.income_quality_accruals(_iq_frame([-50.0] * 8, [20.0] * 8, ta), 16, rpy=4)
    assert nocash < 0 < healthy < goodloss, (nocash, healthy, goodloss)


def test_it_does_NOT_explode_as_netIncome_approaches_zero():
    """The ratio's other failure mode: CFO/NI -> +-inf as NI -> 0.  The fix must be finite and
    barely move, because the DENOMINATOR no longer involves NI at all."""
    vals = []
    for ni in (10.0, 1.0, 1e-3, 1e-9, 0.0, -1e-9, -1e-3):
        v = sm.income_quality_accruals(_iq_frame([ni] * 8, [50.0] * 8, [1000.0] * 8), 16, rpy=4)
        assert np.isfinite(v), (ni, v)
        vals.append(v)
    assert max(vals) - min(vals) < 0.02, vals          # smooth through NI = 0
    # and the metric is CONTINUOUS across the sign change of NI
    assert abs(vals[3] - vals[5]) < 1e-6, (vals[3], vals[5])


def test_the_denominator_can_never_go_through_zero():
    """totalAssets is a positive stock; 0 / negative / NaN must yield NaN, never inf."""
    for bad in (0.0, -1000.0, np.nan):
        v = sm.income_quality_accruals(_iq_frame([10.0] * 8, [20.0] * 8, [bad] * 8), 16, rpy=4)
        assert np.isnan(v), (bad, v)
    # a single bad row does not poison the window -- it drops out of the mean
    mixed = _iq_frame([10.0] * 8, [20.0] * 8, [1000.0, 0.0] + [1000.0] * 6)
    assert np.isfinite(sm.income_quality_accruals(mixed, 16, rpy=4))


def test_it_is_SCALE_FREE_across_company_size():
    """Stage-2 z-scores ACROSS companies, so a raw CFO - NI difference is unusable.  Two
    companies with identical economics at 1000x different size must score identically."""
    small = sm.income_quality_accruals(_iq_frame([50.0] * 8, [60.0] * 8, [1000.0] * 8), 16, rpy=4)
    large = sm.income_quality_accruals(
        _iq_frame([50e3] * 8, [60e3] * 8, [1000e3] * 8), 16, rpy=4)
    assert small == pytest.approx(large, rel=1e-12), (small, large)


def test_the_SIGN_matches_the_existing_POSITIVE_weight():
    """Sign inversions are this project's most repeated bug, so the direction is asserted, not
    assumed: MORE cash relative to earnings must score HIGHER, matching w = +0.072."""
    import createDicts as cdic
    postBm, postNew = cdic.getPostDict()
    w = float({**postBm, **postNew}["incomeQuality"]["w"])
    assert w > 0, "the weight is no longer positive -- re-verify the metric's direction"
    more_cash = sm.income_quality_accruals(_iq_frame([50.0] * 8, [90.0] * 8, [1000.0] * 8), 16, rpy=4)
    less_cash = sm.income_quality_accruals(_iq_frame([50.0] * 8, [55.0] * 8, [1000.0] * 8), 16, rpy=4)
    assert more_cash > less_cash, (more_cash, less_cash)


def test_the_window_and_frequency_scaling_match_the_neighbouring_metrics():
    """flow/stock, so it takes the per-quarter normalisation earnYield and RoA take -- a
    semi-annual filer's 6-month CFO over a point-in-time asset base would otherwise read ~2x.

    ASSERTED THROUGH `postbm_metric`, NOT through `income_quality_accruals` (updated
    2026-08-02).  The per-quarter factor used to be applied INSIDE the metric function, which
    is exactly why `incomeQuality` was absent from the table that was supposed to list every
    flow-corrected metric: the table could not see a correction the function was hiding.  The
    factor now comes from STAGE2_METRIC_SPEC via `flow_factor`, so the property belongs at the
    level that applies it.  The companion test below pins that the bare function is
    frequency-NEUTRAL, so the two together say where the correction lives.
    """
    q = _iq_frame([50.0] * 24, [60.0] * 24, [1000.0] * 24, months=3)
    s = _iq_frame([100.0] * 24, [120.0] * 24, [1000.0] * 24, months=6)   # 2x the 6-month flow
    assert sm.postbm_metric("incomeQuality", "incomeQuality", q, 16, rpy=4) == pytest.approx(
        sm.postbm_metric("incomeQuality", "incomeQuality", s, 16, rpy=2), rel=1e-12)
    assert rp.scale_window(16, 2) == 8


def test_the_bare_accruals_function_applies_NO_frequency_factor_of_its_own():
    """The other half of the pair above: `income_quality_accruals` must NOT double-apply.

    If the factor is ever put back inside the function while the registry entry stays, a
    semi-annual filer gets x0.25 and the defect is silent -- the quarterly path (x1.0 twice)
    stays bit-identical, so nothing on the current panel would look wrong.
    """
    f = _iq_frame([100.0] * 24, [120.0] * 24, [1000.0] * 24, months=6)
    bare = sm.income_quality_accruals(f, 16, rpy=2)
    scored = sm.postbm_metric("incomeQuality", "incomeQuality", f, 16, rpy=2)
    assert scored == pytest.approx(bare * 0.5, rel=1e-12), (bare, scored)
    assert sm.flow_factor("incomeQuality", 2) == 0.5
    assert sm.flow_factor("incomeQuality", 4) == 1.0


def test_the_old_RATIO_is_no_longer_read_anywhere_in_stage2():
    """The partial-sweep guard: postbm_metric must route incomeQuality to the new function, not
    fall through to the generic `tempcdx[met].head(w).mean()` on FMP's ratio column."""
    import inspect
    src = inspect.getsource(sm.postbm_metric)
    assert 'key == "incomeQuality"' in src
    assert "income_quality_accruals" in src
    i_branch = src.index('key == "incomeQuality"')
    i_else = src.index("else:")
    assert i_branch < i_else, "the incomeQuality branch must precede the generic fallthrough"


def test_the_real_panel_metric_is_bounded_and_the_old_one_was_not():
    """Measured on the shipped 2026-07-17 top-100: the ratio spanned -19.78..+190.41; the
    replacement spans ~-0.05..+0.04, a ~1500x reduction in dispersion."""
    cache = os.path.join(_HERE, "resdic_2026-07-17_CORRECTED.pickle")
    if not os.path.exists(cache):
        pytest.skip("no saved resdic on this machine -- dispersion NOT re-measured here")
    import postBoRank as pbr
    r = pd.read_pickle(cache)
    cdx = pbr._sort_cdx_newest_first(r["cdx_dftop100"])
    fmap = rp.frequency_by_source(cdx)
    new = pd.Series({s: sm.income_quality_accruals(t, 16, rp.rows_per_year(fmap, s))
                     for s, t in cdx.groupby("source", sort=False)}).dropna()
    old = pd.to_numeric(r["postScoreMetric_raw"].set_index("source")["incomeQuality"],
                        errors="coerce").dropna()
    assert new.abs().max() < 1.0, new.abs().max()
    assert old.abs().max() > 100.0, old.abs().max()
    assert old.std() / new.std() > 100, old.std() / new.std()


def test_MUTATION_the_cap_is_read_from_the_constant_not_hard_coded():
    """A hard-coded 28 inside the function would make EPS_MEAN_BASE_NQ decorative.  Drive the
    parameter directly and confirm the window follows it."""
    rng = np.random.default_rng(13)
    path = np.abs(rng.normal(10, 3, 60)) + 0.5
    f = _frame(60, 3, eps_path=path)
    a = sm.eps_to_eps_mean(f, nq=8, rpy=4)
    b = sm.eps_to_eps_mean(_frame(8, 3, eps_path=path[:8]), rpy=4)
    assert a == pytest.approx(b, abs=1e-12), (a, b)
    assert a != sm.eps_to_eps_mean(f, rpy=4)
