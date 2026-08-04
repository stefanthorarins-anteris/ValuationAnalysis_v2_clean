"""Offline unit tests for the reporting_period primitives + the two 2026-07-26 rulings.

Run:  python test_reporting_period.py

WHY THIS FILE EXISTS (review S8): every consumer in the pipeline shares these four
primitives and NOTHING tested them.  The H1 defect that reached review was a single
default (`minimum=2`) on `scale_window` that silently imposed a 12-month smoothing on all
18 d* columns -- 44.5% of Stage-1 summed weight -- for 14.4% of the universe.  A table
test over `scale_window(n, rpy)` is the cheapest guard in the repo, so it is first.
"""
import os
import sys
import io
import contextlib

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import reporting_period as rp
import calcScore as cs


# --------------------------------------------------------------------------- #
#  scale_window -- the contract, exhaustively                                 #
# --------------------------------------------------------------------------- #
def test_scale_window_contract():
    """For every n and rpy: 1 <= result <= max(1, n), rpy=4 is an exact no-op, and the
    result is the calendar-equivalent row count (half-up rounded)."""
    for n in range(0, 33):
        for rpy in (2, 4, None):
            got = rp.scale_window(n, rpy)
            assert isinstance(got, int) or float(got).is_integer(), (n, rpy, got)
            assert got >= 1, ('never below 1 row', n, rpy, got)
            assert got <= max(1, n), ('NEVER larger than the quarterly window', n, rpy, got)
            if rpy in (4, None):
                assert got == n or (n == 0 and got == 1), ('rpy=4 must be a no-op', n, got)
    # the exact H1 case: fsMAnumber=1 must stay a 1-row window for a semi-annual source
    assert rp.scale_window(1, 2) == 1, 'H1 REGRESSION: scale_window(1, rpy=2) must be 1'
    # half-up rounding, not bankers'
    assert rp.scale_window(3, 2) == 2, 'half-up: 1.5 -> 2'
    assert rp.scale_window(2, 2) == 1
    assert rp.scale_window(16, 2) == 8
    assert rp.scale_window(20, 2) == 10
    # explicit minimum is honoured but still cannot exceed n
    assert rp.scale_window(2, 2, minimum=1) == 1
    assert rp.scale_window(0, 2) == 1, 'degenerate n=0 clamps to 1, never 0'
    print('PASS test_scale_window_contract')


def test_rows_per_year_and_factors():
    assert rp.rows_per_year(rp.QUARTERLY) == 4
    assert rp.rows_per_year(rp.SEMIANNUAL) == 2
    assert rp.rows_per_year(rp.UNKNOWN) == rp.DEFAULT_ROWS_PER_YEAR == 4
    assert rp.rows_per_year(None) == 4
    assert rp.rows_per_year('nonsense') == 4
    m = {'A': rp.SEMIANNUAL, 'B': rp.QUARTERLY}
    assert rp.rows_per_year(m, 'A') == 2
    assert rp.rows_per_year(m, 'B') == 4
    assert rp.rows_per_year(m, 'MISSING') == 4, 'unknown source -> quarterly path'
    assert rp.rows_per_year(m, None) == 4

    # per_quarter_factor: exact no-op for quarterly, 0.5 for semi-annual
    assert rp.per_quarter_factor(4) == 1.0
    assert rp.per_quarter_factor(2) == 0.5
    # annualize_factor: a genuine full-year multiplier
    assert rp.annualize_factor(4) == 4.0
    assert rp.annualize_factor(2) == 2.0

    # Stage-1 flow factors: direction matters and is asserted, not assumed.
    assert rp.stage1_flow_factor('earningsYield', 4) == 1.0
    assert rp.stage1_flow_factor('earningsYield', 2) == 0.5      # flow/stock -> scale down
    # `freeCashFlowToMarketCap` REPLACED `pfcfRatio` when the criterion was inverted to yield
    # form (2026-08-04), and the flow moved from the DENOMINATOR to the NUMERATOR with it -- so
    # the factor flipped from x2.0 to x0.5 on a semi-annual filer.  Both are asserted, and the
    # retired key is asserted to be GONE: a stale 'pfcfRatio' entry would still resolve and
    # still return a plausible-looking 2.0, applying the correction backwards in silence.
    assert rp.stage1_flow_factor('freeCashFlowToMarketCap', 4) == 1.0
    assert rp.stage1_flow_factor('freeCashFlowToMarketCap', 2) == 0.5   # flow/stock -> down
    assert 'pfcfRatio' not in rp.STAGE1_FLOW_CORRECTION, \
        'the inverted metric is freeCashFlowToMarketCap; a leftover pfcfRatio entry would ' \
        'scale semi-annual names by 2.0 instead of 0.5 -- a 4x error, silently'
    assert rp.stage1_flow_factor('netDebtToEBITDA', 4) == 0.25   # absolute unity threshold
    assert rp.stage1_flow_factor('netDebtToEBITDA', 2) == 0.5
    assert rp.stage1_flow_factor('grossProfitMargin', 2) == 1.0  # flow/flow -> untouched
    assert rp.stage1_flow_factor('not_a_metric', 2) == 1.0
    # EVERY key in the correction table must be a live Stage-1 criterion.  This is the general
    # form of the assertion above: renaming or retiring a metric without touching this table
    # leaves an orphan that silently corrects nothing, or a missing entry that silently
    # corrects nothing -- neither raises.
    import createDicts as _cdic
    _keys = set()
    for _d in _cdic.getBaseMeanDiffUnitySpecialDicts():
        _keys |= set(_d)
    _orphans = sorted(set(rp.STAGE1_FLOW_CORRECTION) - _keys)
    assert not _orphans, ('STAGE1_FLOW_CORRECTION names metrics that are not Stage-1 criteria: '
                          '%s -- the correction is dead for those keys' % _orphans)
    print('PASS test_rows_per_year_and_factors')


# --------------------------------------------------------------------------- #
#  classification                                                            #
# --------------------------------------------------------------------------- #
def test_classify_from_period():
    assert rp.classify_from_period(['Q1', 'Q2', 'Q3', 'Q4'] * 4) == rp.QUARTERLY
    assert rp.classify_from_period(['Q3', 'Q4']) == rp.QUARTERLY, 'any Q1/Q3 => quarterly'
    assert rp.classify_from_period(['Q2', 'Q4', 'Q2', 'Q4']) == rp.SEMIANNUAL
    assert rp.classify_from_period(['H1', 'H2']) == rp.SEMIANNUAL, 'explicit halves'
    assert rp.classify_from_period(['Q2', 'Q4']) == rp.UNKNOWN, 'too short to conclude'
    assert rp.classify_from_period([]) == rp.UNKNOWN
    assert rp.classify_from_period([None, np.nan, '']) == rp.UNKNOWN
    assert rp.classify_from_period(['FY'] * 6) == rp.UNKNOWN, 'annual-only is not handled'
    print('PASS test_classify_from_period')


def test_classify_from_cadence():
    q = pd.date_range('2020-03-31', periods=16, freq='QE')
    sa = pd.date_range('2020-06-30', periods=10, freq='2QE')
    assert rp.classify_from_cadence(q) == rp.QUARTERLY
    assert rp.classify_from_cadence(sa) == rp.SEMIANNUAL
    assert rp.classify_from_cadence(q[:2]) == rp.UNKNOWN, 'too few gaps'
    assert rp.classify_from_cadence([]) == rp.UNKNOWN
    # the ambiguous band resolves to UNKNOWN, never to a guess
    amb = pd.to_datetime(['2020-01-01', '2020-05-15', '2020-09-27', '2021-02-08',
                          '2021-06-22'])
    assert rp.classify_from_cadence(amb) == rp.UNKNOWN
    print('PASS test_classify_from_cadence')


def test_period_vs_cadence_conflict_is_logged_and_flippable():
    """A period/cadence DISAGREEMENT must be recorded, and the tie-break must be one flag."""
    sa_dates = pd.date_range('2020-06-30', periods=8, freq='2QE').strftime('%Y-%m-%d')
    df = pd.DataFrame({'source': ['X'] * 8, 'date': sa_dates,
                       'period': ['Q1', 'Q2', 'Q3', 'Q4'] * 2})   # calendar-quarter stamps
    conflicts = []
    got = rp.classify_source(dates=df['date'], period_values=list(df['period']),
                             conflicts=conflicts, source='X')
    assert conflicts == [('X', rp.QUARTERLY, rp.SEMIANNUAL)], conflicts
    assert got == rp.QUARTERLY, 'default priority is `period`'

    saved = rp.CLASSIFIER_PRIORITY
    try:
        rp.CLASSIFIER_PRIORITY = 'cadence'
        assert rp.classify_source(dates=df['date'],
                                  period_values=list(df['period'])) == rp.SEMIANNUAL
        rp.CLASSIFIER_PRIORITY = 'unknown'
        assert rp.classify_source(dates=df['date'],
                                  period_values=list(df['period'])) == rp.UNKNOWN
    finally:
        rp.CLASSIFIER_PRIORITY = saved

    # agreement must NOT be logged as a conflict
    conflicts2 = []
    rp.classify_source(dates=sa_dates, period_values=['Q2', 'Q4'] * 4,
                       conflicts=conflicts2, source='Y')
    assert conflicts2 == [], conflicts2
    print('PASS test_period_vs_cadence_conflict_is_logged_and_flippable')


def test_frequency_by_source_no_period_column():
    """No `period` column => NO SIGNAL from it => cadence fallback.  A period label must
    never be reconstructed from the calendar-snapped date (the 279-name trap)."""
    sa = pd.date_range('2020-06-30', periods=8, freq='2QE').strftime('%Y-%m-%d')
    q = pd.date_range('2020-03-31', periods=16, freq='QE').strftime('%Y-%m-%d')
    df = pd.DataFrame({'source': ['SA'] * 8 + ['Q'] * 16, 'date': list(sa) + list(q)})
    with contextlib.redirect_stdout(io.StringIO()):
        fm = rp.frequency_by_source(df, verbose=False)
    assert fm == {'SA': rp.SEMIANNUAL, 'Q': rp.QUARTERLY}, fm
    print('PASS test_frequency_by_source_no_period_column')


# --------------------------------------------------------------------------- #
#  RULING Q2 (2026-07-26): Stage-1 window is NOT frequency-scaled              #
# --------------------------------------------------------------------------- #
def test_stage1_window_is_identical_for_both_frequencies():
    """PINS RULING Q2.  calcByTier must receive the SAME window length for a semi-annual
    and a quarterly source.  Stage-1 scores an estimated PROBABILITY (a mean of Bernoulli
    pass indicators), so shrinking the window only degrades the estimator and top-tail
    selection converts that noise into gate share.  If a future change re-introduces
    `scale_window` here, this fails."""
    import utils
    # Build the frame on the REAL BoMetric schema so every criterion's column exists --
    # a hand-listed subset silently skips most of the tests we want to observe.
    schema = list(utils.initBoMetric_fromDict()['BoMetric_df'].columns)
    sa_dates = pd.date_range('2020-06-30', periods=10, freq='2QE')
    q_dates = pd.date_range('2020-03-31', periods=20, freq='QE')
    rows = []
    for src, dts in (('SA', sa_dates), ('QQ', q_dates)):
        for d in dts:
            r = {c: 0.01 for c in schema}
            r['source'] = src
            r['date'] = d
            rows.append(r)
    bm = pd.DataFrame(rows)[schema]
    freq_map = {'SA': rp.SEMIANNUAL, 'QQ': rp.QUARTERLY}

    seen = {}
    orig = cs.calcByTier

    # *args/**kwargs is deliberate: calcByTier gained a `nan_sink=` kwarg, and a
    # fixed-signature spy would break on it -- the same wrapper-signature trap that took
    # down skill_baseline's eps guard (review R-N1).  Forward everything.
    def spy(dct, Tier, Sign, metvec, avec, met, n, *args, **kwargs):
        seen.setdefault(met, set()).add(n)
        return orig(dct, Tier, Sign, metvec, avec, met, n, *args, **kwargs)

    cs.calcByTier = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ave = cs.getAves2(bm.copy())
            cs.simpleScore_fromDict(bm, ave['BoMetric_ave'], ave['BoMetric_dateAve'],
                                    8, freq_map=freq_map)
    finally:
        cs.calcByTier = orig

    windows = set()
    for v in seen.values():
        windows |= v
    assert windows == {8}, ('RULING Q2 REGRESSION: Stage-1 used more than one window '
                           'length across frequencies', windows)
    print('PASS test_stage1_window_is_identical_for_both_frequencies')


if __name__ == '__main__':
    test_scale_window_contract()
    test_rows_per_year_and_factors()
    test_classify_from_period()
    test_classify_from_cadence()
    test_period_vs_cadence_conflict_is_logged_and_flippable()
    test_frequency_by_source_no_period_column()
    test_stage1_window_is_identical_for_both_frequencies()
    print('\nALL reporting_period / ruling tests PASSED')
