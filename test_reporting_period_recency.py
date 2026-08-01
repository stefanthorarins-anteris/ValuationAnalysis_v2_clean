"""Reporting-frequency classification must read RECENT history, and its watchdog must be ALIVE.

TWO COUPLED DEFECTS, both fixed 2026-07-31, both made worse by a deep-history fetch.

(a) THE CLASSIFIER WAS A WHOLE-HISTORY REDUCTION.  `classify_from_period` is a SET UNION, so a
    single `Q1` or `Q3` label ANYWHERE in the fetched history stamped a source QUARTERLY for its
    entire panel; `classify_from_cadence` took the median gap over the whole window.  Neither
    has a notion of "now", so the verdict's meaning was a function of FETCH DEPTH -- ~6 years at
    `-nrperiods 24`, ~20 years at `-nrperiods 80`.  UK/LSE issuers largely dropped quarterly
    reporting after the FCA removed the interim-management-statement requirement (~2014-15), so
    on a deep fetch a currently-semi-annual LSE filer with a pre-2015 quarterly era gets stamped
    QUARTERLY -- silently reverting the ENTIRE semi-annual fix wave for that name (Graham's
    EPS_ttm sums 4 halves = 24 months; Piotroski's lag becomes 2 years; the flow factors run at
    1.0 instead of 0.5; every scale_window goes unscaled).  ~14% of the universe reports
    semi-annually.

    MEASURED on the 07-17 panel (7,729 sources, cadence path -- the saved cdx_df carries no
    `period` column, so the LABEL path is only testable synthetically, which is what this file
    does): 8 sources (0.10%) change classification, and the semi-annual share moves
    14.37% -> 14.34%, i.e. the cohort is preserved.  TAM.L and SMIN.L -- the two live-verified
    semi-annual filers -- stay semi-annual.

(b) THE CONFLICT WATCHDOG REPORTED NOTHING, BY CONSTRUCTION.  `classify_source` only records a
    conflict when handed a `conflicts` list, and the ingest site -- the only place holding both
    the raw dates and the authoritative `period` -- called it without one.  The universe-wide
    banner site DOES pass a list, but `frequency_by_source` short-circuits on the stored
    `reportingFrequency` and so never calls `classify_source`.  Reproduced before the fix: a
    frame with a genuine conflict AND a stored verdict emitted no banner and no CSV; dropping
    the stored column made both appear.  "Zero conflicts" meant "nothing looked".
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import reporting_period as rp

_WHOLE = 999999          # a `recent_days` wide enough to reproduce the OLD whole-history behaviour


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _semi_then_quarterly(n_semi=22, n_quart=40, switch="2015-06-30"):
    """A currently SEMI-ANNUAL filer with a pre-switch QUARTERLY era.  Returns (dates, periods)
    NEWEST-FIRST, which is the raw FMP statement order the ingest site sees."""
    sw = pd.Timestamp(switch)
    d_semi = [sw + pd.DateOffset(months=6 * k) for k in range(n_semi)][::-1]
    d_quart = [sw - pd.DateOffset(months=3 * (k + 1)) for k in range(n_quart)]
    p_semi = (["Q4", "Q2"] * ((n_semi + 1) // 2))[:n_semi]
    p_quart = (["Q4", "Q3", "Q2", "Q1"] * ((n_quart + 3) // 4))[:n_quart]
    return list(d_semi) + list(d_quart), list(p_semi) + list(p_quart)


def _dates(n, months, end="2026-06-30"):
    """n period-end dates `months` apart, NEWEST-FIRST."""
    e = pd.Timestamp(end)
    return [e - pd.DateOffset(months=months * k) for k in range(n)]


# --------------------------------------------------------------------------- #
#  (a) THE DEFECT, and that recency fixes it                                  #
# --------------------------------------------------------------------------- #
def test_the_whole_history_set_union_defect_is_REAL_and_recency_fixes_it():
    """THE headline case.  A pre-2015 quarterly era must not decide today's frequency."""
    dates, periods = _semi_then_quarterly()
    assert {"Q1", "Q3"} & set(periods), "fixture must contain the pre-2015 quarterly labels"
    # BEFORE: the set union sees the old Q1/Q3 and stamps the name quarterly
    assert rp.classify_source(dates=dates, period_values=periods,
                              recent_days=_WHOLE) == rp.QUARTERLY
    # AFTER: the recent window sees only halves
    assert rp.classify_source(dates=dates, period_values=periods) == rp.SEMIANNUAL


def test_recency_applies_to_the_CADENCE_FALLBACK_too_not_just_the_labels():
    """The brief's 'same recency logic to both paths'.  A panel with NO `period` column must be
    fixed as well -- every pickle saved before the field existed takes this path, and it is the
    only path the real 07-17 panel can exercise."""
    dates, _p = _semi_then_quarterly(n_semi=22, n_quart=60)
    # whole-history median gap is dominated by the 60 quarterly rows -> quarterly
    assert rp.classify_source(dates=dates, recent_days=_WHOLE) == rp.QUARTERLY
    assert rp.classify_from_cadence(dates) == rp.QUARTERLY, "the primitive stays whole-input"
    # the recent window sees only ~183-day gaps
    assert rp.classify_source(dates=dates) == rp.SEMIANNUAL


def test_a_genuinely_quarterly_filer_is_UNCHANGED():
    """The expensive error is calling a quarterly filer semi-annual (it halves every window on a
    name whose rows really are quarters).  Recency must not create that error."""
    d = _dates(80, 3)
    p = (["Q4", "Q3", "Q2", "Q1"] * 20)
    assert rp.classify_source(dates=d, period_values=p) == rp.QUARTERLY
    assert rp.classify_source(dates=d) == rp.QUARTERLY
    assert rp.classify_source(dates=d, period_values=p,
                              recent_days=_WHOLE) == rp.QUARTERLY


def test_a_filer_that_went_the_OTHER_way_is_read_conservatively():
    """Semi-annual era first, quarterly now.  The recent window sees Q1/Q3 -> QUARTERLY, which
    is both correct AND the module's documented cheap error, so no special handling is needed."""
    sw = pd.Timestamp("2023-06-30")
    d = [sw + pd.DateOffset(months=3 * k) for k in range(12)][::-1] + \
        [sw - pd.DateOffset(months=6 * (k + 1)) for k in range(20)]
    p = (["Q4", "Q3", "Q2", "Q1"] * 3) + (["Q4", "Q2"] * 10)
    assert rp.classify_source(dates=d, period_values=p) == rp.QUARTERLY


# --------------------------------------------------------------------------- #
#  The window: why 4 years, asserted as the two bounds that pick it           #
# --------------------------------------------------------------------------- #
def test_the_window_is_long_enough_for_SEMIANNUAL_to_be_REACHABLE():
    """THE LOWER BOUND, and the failure mode that is easy to miss: too SHORT a window makes
    SEMIANNUAL unreachable (the label path needs PERIOD_MIN_LABELLED_ROWS rows, the cadence path
    CADENCE_MIN_GAPS gaps), UNKNOWN is treated as quarterly, and the fix is reverted just as
    thoroughly as by too LONG a window."""
    semi_rows_in_window = rp.CLASSIFY_RECENT_DAYS / 182.6
    assert semi_rows_in_window >= rp.PERIOD_MIN_LABELLED_ROWS, semi_rows_in_window
    assert semi_rows_in_window >= rp.CADENCE_MIN_GAPS + 1, semi_rows_in_window
    # and demonstrate it: a plain semi-annual filer must classify from the window alone
    d = _dates(int(semi_rows_in_window), 6)
    p = (["Q4", "Q2"] * len(d))[:len(d)]
    assert rp.classify_source(dates=d, period_values=p) == rp.SEMIANNUAL
    assert rp.classify_source(dates=d) == rp.SEMIANNUAL


def test_the_window_is_short_enough_to_clear_the_2014_15_regime_change():
    """THE UPPER BOUND.  The window must not reach back to the FCA interim-management-statement
    change (~2014-15) from a present-day run."""
    assert rp.CLASSIFY_RECENT_DAYS / 365.25 <= 8.0
    newest = pd.Timestamp("2026-06-30")
    oldest_seen = newest - pd.Timedelta(days=rp.CLASSIFY_RECENT_DAYS)
    assert oldest_seen > pd.Timestamp("2016-01-01"), oldest_seen


def test_the_window_matches_the_pipelines_own_history_gate():
    """failTests demands rows_per_year x 4 = 4 CALENDAR years of history.  The classifier now
    looks at that same span rather than at whatever depth was fetched, so the fetch gate and
    the classifier reason about the same window."""
    assert rp.CLASSIFY_RECENT_DAYS == pytest.approx(4 * 365.25, abs=1.0)


def test_verdict_is_INDEPENDENT_OF_FETCH_DEPTH():
    """The property the whole fix exists to establish: 24, 44 and 80 fetched periods must give
    the SAME verdict for the same company."""
    for months, want in ((3, rp.QUARTERLY), (6, rp.SEMIANNUAL)):
        labels = ["Q4", "Q3", "Q2", "Q1"] if months == 3 else ["Q4", "Q2"]
        verdicts = set()
        for depth in (24, 44, 80):
            d = _dates(depth, months)
            p = (labels * depth)[:depth]
            verdicts.add(rp.classify_source(dates=d, period_values=p))
        assert verdicts == {want}, (months, verdicts)


# --------------------------------------------------------------------------- #
#  Recency must never DESTROY a signal (the min-rows floor)                    #
# --------------------------------------------------------------------------- #
def test_a_SPARSE_source_falls_back_to_the_most_recent_rows_not_to_UNKNOWN():
    """A gappy semi-annual filer with too few rows inside the calendar window must NOT collapse
    to UNKNOWN (-> the quarterly path), which would revert the fix for exactly the cohort the
    module protects.  The floor extends to the most recent CLASSIFY_RECENT_MIN_ROWS rows."""
    # halves every 6 months but starting 6 years back, so only ~2 rows fall in a 4-year window
    d = _dates(10, 6, end="2020-06-30")
    p = (["Q4", "Q2"] * 5)
    kept_dates, kept_periods = rp._recent_slice(d, p)
    assert len(kept_dates) >= rp.CLASSIFY_RECENT_MIN_ROWS
    assert rp.classify_source(dates=d, period_values=p) == rp.SEMIANNUAL
    assert rp.classify_source(dates=d) == rp.SEMIANNUAL


def test_the_floor_is_BOUNDED_so_fetch_depth_still_cannot_leak_in():
    """The sparse fallback takes N ROWS, not the whole panel -- otherwise it would reintroduce
    the very fetch-depth dependence being removed.  Constructed so the calendar window really
    is under-filled: 3 recent halves, then a 10-year gap, then a deep quarterly block."""
    recent = _dates(3, 6, end="2026-06-30")
    old = _dates(70, 3, end="2012-06-30")
    d = list(recent) + list(old)
    p = ["Q4", "Q2", "Q4"] + (["Q4", "Q3", "Q2", "Q1"] * 18)[:70]
    # the 4-year calendar window holds only the 3 recent rows, so the floor must engage ...
    inside = sum(1 for x in d if (pd.Timestamp(recent[0]) - pd.Timestamp(x)).days
                 <= rp.CLASSIFY_RECENT_DAYS)
    assert inside < rp.CLASSIFY_RECENT_MIN_ROWS, inside
    kept, kept_p = rp._recent_slice(d, p)
    # ... and it must take EXACTLY the floor, not the 73-row panel
    assert len(kept) == rp.CLASSIFY_RECENT_MIN_ROWS, len(kept)
    assert len(kept_p) == rp.CLASSIFY_RECENT_MIN_ROWS
    assert len(kept) < len(d)


def test_the_window_is_anchored_to_the_SOURCES_OWN_newest_row_not_to_today():
    """A dead/delisted name, and every point-in-time as_of reproduction, has a newest row that
    is years old.  A wall-clock anchor would select ZERO rows and hand back UNKNOWN -> quarterly
    for the entire offline PIT path."""
    d = _dates(12, 6, end="2009-12-31")                   # ended 17 years ago
    p = ["Q4", "Q2"] * 6
    assert rp.classify_source(dates=d, period_values=p) == rp.SEMIANNUAL
    kept, _ = rp._recent_slice(d, p)
    assert len(kept) >= 4


# --------------------------------------------------------------------------- #
#  _recent_slice degrades to OLD behaviour on every unsafe input               #
# --------------------------------------------------------------------------- #
def test_recent_slice_falls_back_to_whole_history_on_unsafe_input():
    d = _dates(12, 3)
    p = ["Q4", "Q3", "Q2", "Q1"] * 3
    # no dates at all -> nothing to be recent about
    assert rp._recent_slice(None, p) == (None, p)
    # all-NaT dates carry no recency information
    nat = [pd.NaT] * 12
    got_d, got_p = rp._recent_slice(nat, p)
    assert got_d is nat and got_p is p
    # a length mismatch means alignment cannot be assumed -- do not guess
    got_d, got_p = rp._recent_slice(d, p[:5])
    assert got_d is d and got_p == p[:5]
    # and a labels-only caller still classifies (no dates -> no restriction)
    assert rp.classify_from_period(p) == rp.QUARTERLY


def test_recent_slice_keeps_dates_and_labels_POSITIONALLY_ALIGNED():
    """The slice applies one mask to two containers.  If it ever misaligned them, a semi-annual
    filer's halves would be paired with a quarterly filer's labels -- silent and undetectable."""
    d, p = _semi_then_quarterly()
    kept_d, kept_p = rp._recent_slice(d, p)
    assert len(kept_d) == len(kept_p)
    lookup = {pd.Timestamp(x): y for x, y in zip(d, p)}
    for x, y in zip(list(kept_d), kept_p):
        assert lookup[pd.Timestamp(x)] == y


def test_recent_slice_is_order_agnostic():
    """The three call sites hand over frames in DIFFERENT row orders (raw FMP is newest-first,
    some panels are date-ascending), so the slice must select by DATE, not by position."""
    d, p = _semi_then_quarterly()
    a_d, a_p = rp._recent_slice(d, p)
    b_d, b_p = rp._recent_slice(d[::-1], p[::-1])
    assert sorted(pd.Timestamp(x) for x in a_d) == sorted(pd.Timestamp(x) for x in b_d)
    assert sorted(a_p) == sorted(b_p)


def test_an_over_wide_window_degrades_rather_than_raising():
    """A caller disabling the cap must not blow up: `newest - Timedelta(days=huge)` raises
    OutOfBoundsTimedelta, so the comparison is done as a day difference instead."""
    d = _dates(12, 3)
    assert rp.classify_source(dates=d, recent_days=10 ** 9) == rp.QUARTERLY


# --------------------------------------------------------------------------- #
#  ONE place applies the window -> the sweep is complete by construction       #
# --------------------------------------------------------------------------- #
def test_the_recency_window_is_applied_in_EXACTLY_ONE_place():
    """This project's signature defect is a fix applied at two of three call sites.  The window
    lives inside `classify_source`, which is the sole funnel for all three production sites, so
    no site can be missed -- asserted here rather than trusted."""
    import inspect
    src = inspect.getsource(rp.classify_source)
    assert "_recent_slice(" in src
    # and the primitives stay PURE whole-input reductions (tests call them directly)
    for fn in (rp.classify_from_period, rp.classify_from_cadence):
        assert "_recent_slice" not in inspect.getsource(fn), fn.__name__


@pytest.mark.parametrize("path", ["getData_fmp.py", "failTests.py"])
def test_every_production_classify_site_funnels_through_classify_source(path):
    """The two OUTSIDE sites -- the ingest stamp and the fetch history gate.  Neither may call
    the primitives directly; that would bypass the recency window.

    review L7, corrected 2026-07-31: this was parametrised over THREE files with a body of
    `if path != "reporting_period.py":`, so the third case asserted NOTHING and reported a pass.
    `reporting_period.py` legitimately calls the primitives (it defines them), so it does not
    belong in this parametrisation at all -- it gets its own assertion below."""
    src = open(os.path.join(_HERE, path), encoding="utf-8").read()
    assert "classify_source(" in src
    assert "classify_from_period(" not in src, "%s bypasses classify_source" % path
    assert "classify_from_cadence(" not in src, "%s bypasses classify_source" % path


def test_no_module_ANYWHERE_calls_the_primitives_except_reporting_period_itself():
    """The funnel claim, swept repo-wide rather than over a hand-listed trio -- a hand-listed
    inventory cannot notice a NEW bypassing caller, which is the failure it exists to catch.
    Test files may call the primitives directly (that is what they are for)."""
    import re
    offenders = []
    for root, dirs, files in os.walk(_HERE):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        for fn in files:
            if not fn.endswith(".py") or fn.startswith("test_"):
                continue
            rel = os.path.relpath(os.path.join(root, fn), _HERE)
            if rel == "reporting_period.py":
                continue                      # defines them
            try:
                src = open(os.path.join(root, fn), encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            for prim in ("classify_from_period", "classify_from_cadence"):
                if re.search(r"\b%s\s*\(" % prim, src):
                    offenders.append((rel, prim))
    assert not offenders, ("module(s) bypass classify_source and so skip the recency window: %s"
                          % offenders)


# --------------------------------------------------------------------------- #
#  The REAL 07-17 panel: the required proof                                   #
# --------------------------------------------------------------------------- #
def _real_panel():
    p = os.path.join(_HERE, "baseline_tools", "resdic_2026-07-17_CORRECTED.pickle")
    return pd.read_pickle(p)["cdx_df"] if os.path.exists(p) else None


def test_real_panel_classification_swing_is_SMALL_and_the_semiannual_cohort_SURVIVES():
    """THE required proof.  Measured 2026-07-31 on all 7,729 sources of the 07-17 panel (cadence
    path only -- the saved cdx_df has no `period` column): 8 sources (0.10%) change, and the
    semi-annual share moves 14.37% -> 14.34%.  A LARGE swing would mean the change is broken,
    not that it is working, so the bound is asserted in BOTH directions."""
    cdx = _real_panel()
    if cdx is None:
        pytest.skip("no saved resdic on this machine")
    groups = list(cdx[["source", "date"]].groupby("source", sort=False))
    before = {s: rp.classify_source(dates=g["date"], recent_days=_WHOLE) for s, g in groups}
    after = {s: rp.classify_source(dates=g["date"]) for s, g in groups}
    n = len(before)
    assert n > 7000, n
    changed = [s for s in before if before[s] != after[s]]
    assert len(changed) <= 0.01 * n, \
        "%d of %d sources changed -- far too many; the window is wrong" % (len(changed), n)
    share_before = sum(1 for v in before.values() if v == rp.SEMIANNUAL) / n
    share_after = sum(1 for v in after.values() if v == rp.SEMIANNUAL) / n
    assert 0.13 <= share_after <= 0.16, share_after
    assert abs(share_after - share_before) <= 0.01, (share_before, share_after)


def test_real_panel_TAM_L_and_SMIN_L_stay_SEMIANNUAL():
    """The two filers whose semi-annual reporting was verified LIVE against FMP (2026-07-25).
    They are the ground truth this classifier is calibrated against."""
    cdx = _real_panel()
    if cdx is None:
        pytest.skip("no saved resdic on this machine")
    for t in ("TAM.L", "SMIN.L"):
        g = cdx.loc[cdx["source"] == t, "date"]
        assert len(g), "%s missing from the panel" % t
        assert rp.classify_source(dates=g) == rp.SEMIANNUAL, t


def test_real_panel_stored_ingest_verdict_still_wins_where_present():
    """frequency_by_source must keep preferring the stored verdict; recency changes what gets
    COMPUTED, never which signal wins."""
    cdx = _real_panel()
    if cdx is None:
        pytest.skip("no saved resdic on this machine")
    assert rp.FREQ_COLUMN in cdx.columns
    got = rp.frequency_by_source(cdx)
    stored = cdx.groupby("source")[rp.FREQ_COLUMN].first().to_dict()
    assert got == stored


# --------------------------------------------------------------------------- #
#  (b) THE WATCHDOG                                                           #
# --------------------------------------------------------------------------- #
def _conflict_frame(with_stored=True, with_stamp=True):
    """A frame with a GENUINE period-vs-cadence conflict: ~183-day gaps (cadence -> semiannual)
    carrying Q1/Q3 labels (period -> quarterly)."""
    d = _dates(12, 6)
    df = pd.DataFrame({"source": ["CONFLICT.L"] * 12, "date": d, "period": ["Q1", "Q3"] * 6})
    if with_stored:
        df[rp.FREQ_COLUMN] = "quarterly"
    if with_stamp:
        df[rp.FREQ_CONFLICT_COLUMN] = "quarterly|semiannual"
    return df


def test_the_conflict_is_DETECTED_even_though_the_stored_verdict_short_circuits():
    """THE defect.  Before the fix this returned an EMPTY conflicts list on any panel carrying
    the ingest verdict -- i.e. on every panel from the next fetch onward."""
    got = []
    df = _conflict_frame()
    # the short-circuit path is genuinely taken (the verdict comes from the stored column)
    assert rp.frequency_by_source(df) == {"CONFLICT.L": "quarterly"}
    # and the conflict is still reported, decoded from the stamped column
    rp.log_conflicts([], verbose=False, csv=False)          # no-op, proves signature
    conflicts = _collect_conflicts(df)
    assert conflicts == [("CONFLICT.L", "quarterly", "semiannual")], conflicts
    # with the stamp absent there is nothing to decode -- which is exactly the old blindness,
    # and why the stamp has to be written at ingest
    assert _collect_conflicts(_conflict_frame(with_stamp=False)) == []


def _collect_conflicts(df):
    """Re-run frequency_by_source and capture what it hands to log_conflicts."""
    seen = []
    orig = rp.log_conflicts
    rp.log_conflicts = lambda c, **kw: (seen.extend(c), c)[1]
    try:
        rp.frequency_by_source(df)
    finally:
        rp.log_conflicts = orig
    return seen


def test_the_ingest_site_RECORDS_and_STAMPS_the_conflict():
    """The ingest site is the only place holding both raw dates and the authoritative `period`.
    It must both append to the run-level list and stamp the frame so the verdict survives the
    pickle to postBo."""
    import getData_fmp as gdf
    tf = pd.DataFrame({"source": ["CONFLICT.L"] * 12, "date": _dates(12, 6),
                       "period": ["Q1", "Q3"] * 6,
                       "netIncome": np.full(12, 100.0),
                       "weightedAverageShsOut": np.full(12, 50.0),
                       "bookValuePerShare": np.full(12, 10.0)})
    acc = []
    out = gdf.stamp_frequency_and_graham(tf.copy(), conflicts=acc)
    assert acc == [("CONFLICT.L", "quarterly", "semiannual")], acc
    assert out[rp.FREQ_CONFLICT_COLUMN].iloc[0] == "quarterly|semiannual"
    assert rp.decode_conflict(out[rp.FREQ_CONFLICT_COLUMN].iloc[0]) == \
        ("quarterly", "semiannual")


def test_a_clean_source_is_stamped_EMPTY_not_NaN():
    """An empty string, so the column is a real string column and decode_conflict is total."""
    import getData_fmp as gdf
    tf = pd.DataFrame({"source": ["CLEAN"] * 12, "date": _dates(12, 3),
                       "period": ["Q4", "Q3", "Q2", "Q1"] * 3,
                       "netIncome": np.full(12, 100.0),
                       "weightedAverageShsOut": np.full(12, 50.0),
                       "bookValuePerShare": np.full(12, 10.0)})
    acc = []
    out = gdf.stamp_frequency_and_graham(tf.copy(), conflicts=acc)
    assert acc == []
    assert out[rp.FREQ_CONFLICT_COLUMN].iloc[0] == ""
    assert rp.decode_conflict("") is None
    assert rp.decode_conflict(np.nan) is None
    assert rp.decode_conflict("garbage|nonsense") is None


def test_the_stamped_column_SURVIVES_forceNumOnDf():
    """It is a string column in a frame that gets blanket-coerced to numeric.  Left off the
    passthrough list it becomes all-NaN and the watchdog goes dark again -- which is precisely
    how reportedCurrency was lost once already."""
    import getData_gen as gdg
    df = _conflict_frame()
    kept = gdg.forceNumOnDf(df)
    assert kept[rp.FREQ_CONFLICT_COLUMN].iloc[0] == "quarterly|semiannual"


def test_ZERO_conflicts_is_an_AFFIRMATIVE_report_not_silence(capsys):
    """The core observability defect: a guard that is silent when healthy cannot be told apart
    from a guard that is dead.  Zero must print, and must say how many sources it examined."""
    rp.log_conflicts([], verbose=True, csv=False, n_examined=7729,
                     detected_via="unit test")
    out = capsys.readouterr().out
    assert "0 conflict(s)" in out
    assert "7729 source(s) examined" in out
    assert "watchdog RAN" in out


def test_the_CSV_is_written_even_when_EMPTY_so_its_ABSENCE_is_the_signal(tmp_path,
                                                                        monkeypatch):
    monkeypatch.chdir(tmp_path)
    rp.log_conflicts([], verbose=False, csv=True, n_examined=3)
    files = glob.glob(str(tmp_path / "ReportingFrequencyConflicts_*.csv"))
    assert len(files) == 1, files
    df = pd.read_csv(files[0])
    assert list(df.columns) == ["source", "by_period", "by_cadence"]
    assert df.empty


def test_the_CSV_contains_the_conflicting_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rp.log_conflicts([("A.L", "quarterly", "semiannual"),
                      ("B.L", "semiannual", "quarterly")], verbose=False, csv=True)
    files = glob.glob(str(tmp_path / "ReportingFrequencyConflicts_*.csv"))
    df = pd.read_csv(files[0])
    assert sorted(df["source"]) == ["A.L", "B.L"]
    assert set(df["by_period"]) == {"quarterly", "semiannual"}


def test_only_the_UNIVERSE_WIDE_site_writes_the_CSV(tmp_path, monkeypatch):
    """Four sites call frequency_by_source(verbose=True) on DIFFERENT pools.  If each wrote the
    shared filename, the last to run -- usually a narrow pool with nothing to report -- would
    clobber the universe-wide list.  csv defaults to False and is enabled at one site."""
    monkeypatch.chdir(tmp_path)
    df = _conflict_frame()
    rp.frequency_by_source(df, verbose=True)                        # narrow-pool default
    assert glob.glob(str(tmp_path / "ReportingFrequencyConflicts_*.csv")) == []
    rp.frequency_by_source(df, verbose=True, csv=True)              # the entitled site
    assert len(glob.glob(str(tmp_path / "ReportingFrequencyConflicts_*.csv"))) == 1
    # and the entitled site is postBo's universe-wide read, asserted in source.
    # review L7, corrected 2026-07-31: this counted the bare substring "csv=True" over the whole
    # of postBo.py, so any unrelated `csv=True` elsewhere in the file would break it (and a
    # mention in a COMMENT already did once).  Count only real keyword arguments named `csv`,
    # via the AST.
    import ast
    src = open(os.path.join(_HERE, "postBo.py"), encoding="utf-8").read()
    assert "rp.frequency_by_source(dmdic.get('cdx_df'), verbose=True, csv=True)" in src
    enabled = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call):
            for kw in n.keywords:
                if (kw.arg == "csv" and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True):
                    enabled.append((ast.unparse(n.func), n.lineno))
    assert len(enabled) == 1, "a second CSV writer would clobber the artifact: %s" % enabled
    assert enabled[0][0] == "rp.frequency_by_source", enabled


def test_the_FETCH_itself_emits_the_conflict_report():
    """The fetch must report even if the ranking stage is run separately or never reached -- and
    it is the only place that sees the RAW, un-snapped dates."""
    import inspect
    import getData_fmp as gdf
    src = inspect.getsource(gdf.get_fundamentals_fmp)
    assert "freq_conflicts = []" in src
    assert "conflicts=freq_conflicts" in src
    assert "rp.log_conflicts(freq_conflicts" in src
    assert "'freqConflicts': freq_conflicts" in src


def test_the_conflict_report_can_never_abort_a_twelve_hour_fetch():
    """A diagnostic that can raise is worse than no diagnostic on an unattended 12-hour run."""
    import inspect
    import getData_fmp as gdf
    src = inspect.getsource(gdf.get_fundamentals_fmp)
    i = src.index("rp.log_conflicts(freq_conflicts")
    assert "try:" in src[max(0, i - 400):i], "the fetch-level conflict report is unguarded"
    # and the CSV write itself swallows everything
    assert "except Exception" in inspect.getsource(rp._write_conflict_csv)


def test_conflict_detection_is_taken_on_the_SAME_recent_window_for_both_signals():
    """A conflict must be a real disagreement, not an artifact of the two signals reading
    different spans.  Both are sliced once, together, before either is computed."""
    import inspect
    src = inspect.getsource(rp.classify_source)
    i_slice = src.index("_recent_slice(")
    i_period = src.index("classify_from_period(")
    i_cadence = src.index("classify_from_cadence(")
    assert i_slice < i_period and i_slice < i_cadence
