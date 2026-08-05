"""THE PINS FOR THE TWO-TIER NaN POLICY AND THE LOCAL PEG (2026-08-05).

WHAT THIS FILE EXISTS TO PREVENT.  Every mechanism in `nan_policy` decides whether a number is
a MEASUREMENT or an ABSENCE, and every one of them fails SILENTLY when it is wrong: a
mis-specified coverage denominator penalises short histories, a fixed-cadence gap rule flags
every semi-annual filer in the universe, a boundary imputed on the wrong side of a metric hands
an adverse company the maximum reward, and a primary limb keyed on the wrong row either ejects
thousands of names or none.  None of those raise.  So each one is pinned behaviourally, and the
frequency-relative ones are pinned by asserting the QUARTERLY and SEMI-ANNUAL answers DIFFER on
the same dates -- which is the only shape of assertion a fixed-cadence regression cannot pass.

No network, no pickles, no API key.  `pytest . --ignore=baseline_tools`.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

import calcMetrics as cm
import calcScore as cs
import createDicts as cdic
import data_quality as dq
import getData_fmp as gdf
import nan_policy as npol
import reporting_period as rp
import stage2_metrics as sm
import utils


# --------------------------------------------------------------------------- #
#  helpers                                                                    #
# --------------------------------------------------------------------------- #
def _dates(n, months=3, end='2026-03-31'):
    """`n` period-end dates, NEWEST FIRST, `months` apart."""
    return [pd.Timestamp(end) - pd.DateOffset(months=months * i) for i in range(n)]


def _cdx(n=20, months=3, freq=rp.QUARTERLY, **cols):
    """A minimal per-source cdx frame, NEWEST-FIRST, with `cols` overriding per row."""
    #  price x shares == marketCap, and the share count is above the 1,000 floor
    #  `data_quality.check_price_sanity` uses -- otherwise every row of the fixture is flagged as
    #  an impossible price/market-cap pair and the fixture measures that instead.
    base = dict(price=10.0, marketCap=1_000_000.0, weightedAverageShsOut=100_000.0,
                netIncome=50.0, netCashProvidedByOperatingActivities=60.0,
                totalStockholdersEquity=500.0, totalAssets=1000.0, revenue=800.0,
                #  the remaining Piotroski / Graham inputs, at bland in-domain values, so a
                #  fixture built for one metric does not accidentally make another NaN
                longTermDebt=100.0, currentRatio=2.0, grossProfitMargin=0.25,
                bookValuePerShare=5.0, grahamNumber=8.0, grahamUndefinedReason='',
                #  every `eqMet` the postBm block reads, so the registry-derived seam test can
                #  compute EVERY windowed metric on this ONE fixture rather than on a per-metric
                #  frame (a per-metric frame is how a metric gets left out of a sweep)
                returnOnAssets=0.05, earningsYield=0.05, pbRatio=2.0,
                returnOnCapitalEmployed=0.08, returnOnEquity=0.10, incomeQuality=1.5,
                tangibleBookValuePerShare=4.0, freeCashFlow=60.0, netIncomePerShare=0.5)
    d = {k: [v] * n for k, v in base.items()}
    d['date'] = _dates(n, months)
    d[rp.FREQ_COLUMN] = [freq] * n
    d['source'] = ['TEST'] * n
    for k, v in cols.items():
        d[k] = list(v) if isinstance(v, (list, tuple)) else [v] * n
    return pd.DataFrame(d)


# =========================================================================== #
#  1. PRIMARY PRESENCE -- the CEO's first tier                                 #
# =========================================================================== #
def test_the_primary_set_is_FIVE_raw_inputs_plus_two_impossibility_checks():
    """The list itself, pinned.  `revenue` and `totalAssets` were DROPPED from primary
    (ADDENDUM C1) and kept only as arithmetic-impossibility checks -- their entire measured
    eject was an impossible SIGN, not a NaN, so 'primary column' was the wrong reason for it."""
    assert set(npol.PRIMARY_POSITIVE) == {'price', 'marketCap', 'weightedAverageShsOut'}
    assert set(npol.PRIMARY_PRESENT) == {'netIncome',
                                         'netCashProvidedByOperatingActivities',
                                         'totalStockholdersEquity'}
    assert len(npol.PRIMARY_POSITIVE) + len(npol.PRIMARY_PRESENT) == 6, (
        'the primary set is (a) three inputs without which there is no valuation question + '
        '(b) three that ARE the income-quality / indebtedness test')
    assert set(npol.SANITY_IMPOSSIBLE) == {'revenue', 'totalAssets'}
    #  a DERIVED metric must never be primary (section 1c: uGrahamNumberToPrice is all-NaN on
    #  24.97% of the universe and dSalesToInventory on 40.76%, both BECAUSE of what the company
    #  is -- ejecting on them would disqualify companies for being cheap-and-troubled)
    for derived in ('uGrahamNumberToPrice', 'grahamNumberToPrice', 'earnYield', 'Piotroski'):
        assert derived not in npol.PRIMARY_POSITIVE + npol.PRIMARY_PRESENT


def test_a_legitimate_NEGATIVE_or_ZERO_value_never_ejects():
    """"A negative net income is the answer the filter is built to read" -- only ABSENCE and
    arithmetic impossibility disqualify."""
    for col, val in (('netIncome', -50.0), ('netCashProvidedByOperatingActivities', -60.0),
                     ('totalStockholdersEquity', -500.0), ('revenue', 0.0)):
        ej = npol.primary_eject(_cdx(**{col: val}))
        assert len(ej) == 0, '%s = %g ejected, and it is a real answer about the company' % (
            col, val)


def test_an_ABSENT_primary_input_on_the_NEWEST_row_ejects():
    for col in npol.PRIMARY_PRESENT:
        ej = npol.primary_eject(_cdx(**{col: [np.nan] + [50.0] * 19}))
        assert list(ej['field']) == [col], col
    for col in npol.PRIMARY_POSITIVE:
        for bad in (np.nan, 0.0, -1.0):
            ej = npol.primary_eject(_cdx(**{col: [bad] + [10.0] * 19}))
            assert list(ej['field']) == [col], (col, bad)


def test_an_absent_primary_input_on_an_OLD_row_does_NOT_eject():
    """THE SELECTIVITY OF THE WHOLE LIMB TURNS ON THIS ROW CHOICE.  The filter values the
    company AS OF NOW: a missing 2018 cash-flow statement does not make today's valuation
    unanswerable.  Keyed on ANY row instead, the CFO limb alone goes from 117 sources (1.51%)
    to 792 (10.25%) on the measured panel, and the two impossibility checks from 49 to 842."""
    old = [60.0] * 19 + [np.nan]        # newest-first, so the LAST element is the oldest
    assert len(npol.primary_eject(_cdx(netCashProvidedByOperatingActivities=old))) == 0


def test_the_verdict_row_is_found_by_DATE_and_not_by_position():
    """A caller handing over an OLDEST-FIRST frame must get the same verdict.  `filter_invalid_data`
    sorts ascending by date, `stage2_metrics` works newest-first, and the panel on disk is
    oldest-first -- a positional read would silently invert on two of the three."""
    nf = _cdx(netCashProvidedByOperatingActivities=[np.nan] + [60.0] * 19)
    of = nf.iloc[::-1].reset_index(drop=True)
    assert list(npol.primary_eject(nf)['field']) == list(npol.primary_eject(of)['field'])
    assert len(npol.primary_eject(of)) == 1


def test_the_eject_is_wired_into_the_EXISTING_gate_and_is_idempotent():
    """NOT A NEW GATE (section 2): `data_quality.filter_invalid_data` is the source-level
    exclusion that already exists and already runs twice on the live path.  A second gate is
    worse than either, and the second pass must remove nothing."""
    good = _cdx(n=12)
    bad = _cdx(n=12, netCashProvidedByOperatingActivities=[np.nan] + [60.0] * 11)
    bad['source'] = 'BAD'
    df = pd.concat([good, bad], ignore_index=True)
    clean, removed = dq.filter_invalid_data(df, verbose=False)
    assert set(clean['source']) == {'TEST'}
    assert removed['removal_reason'].str.startswith('primary_input_absent').all()
    assert 'netCashProvidedByOperatingActivities' in removed['removal_reason'].iloc[0]
    clean2, removed2 = dq.filter_invalid_data(clean, verbose=False)
    assert len(clean2) == len(clean) and removed2.empty, 'the second pass is not idempotent'


def test_a_frame_missing_a_primary_input_is_REFUSED_and_bannered_not_silently_passed(capsys):
    """Two failure modes are both worse than the third.  RAISING costs a ~12-hour run; SILENTLY
    reporting "0 ejected" for a missing column is a false negative that ships unfiltered names.
    So `primary_eject` refuses, and `filter_invalid_data` catches it and banners -- the same
    LOUD FALLBACK shape postBo's carve-out already uses."""
    with pytest.raises(KeyError, match='missing'):
        npol.primary_eject(pd.DataFrame({'source': ['A'], 'date': ['2026-01-01'],
                                         'price': [10.0]}))
    thin = _cdx(n=10).drop(columns=['netCashProvidedByOperatingActivities'])
    clean, _removed = dq.filter_invalid_data(thin, verbose=False)
    out = capsys.readouterr().out
    assert 'PRIMARY-PRESENCE EJECT DID NOT RUN' in out
    assert 'DO NOT treat this output' in out
    assert len(clean) == 10, 'the run must survive; only the tier is skipped'


def test_the_eject_runs_even_when_NOTHING_ELSE_is_corrupt():
    """`filter_invalid_data` used to RETURN EARLY on 'no corrupt data found', which would have
    made the primary eject conditional on unrelated arithmetic corruption existing."""
    df = _cdx(n=12, netIncome=[np.nan] + [50.0] * 11)
    clean, removed = dq.filter_invalid_data(df, verbose=False)
    assert clean.empty and len(removed) == 12, (
        'the primary eject was skipped because no price/mcap check happened to fire')


# =========================================================================== #
#  2. COVERAGE                                                                 #
# =========================================================================== #
def test_the_coverage_denominator_is_ROWS_PRESENT_not_the_nominal_window():
    """390 of 7,729 sources (5.05%) carry fewer rows than their scaled window.  Against a
    NOMINAL denominator they fail coverage on every metric by construction -- a covert
    history-depth filter that also double-counts data_quality's min_periods_required = 8."""
    #  6 rows, all computable, against a 16-row window: coverage is 6/6 = 1.0, not 6/16.
    assert npol.window_verdict([1.0] * 6, 16, 'RoA', 4) == pytest.approx(1.0)
    #  and a genuinely thin measurement inside a short panel still fails
    assert np.isnan(npol.window_verdict([1.0, 1.0] + [np.nan] * 4, 16, 'RoA', 4))


def test_the_threshold_is_a_STRICT_below_at_exactly_one_half():
    v = [1.0] * 4 + [np.nan] * 4
    assert npol.window_verdict(v, 8, 'RoA', 4) == pytest.approx(1.0), 'coverage 0.50 is adequate'
    assert np.isnan(npol.window_verdict([1.0] * 3 + [np.nan] * 5, 8, 'RoA', 4))
    assert npol.COVERAGE_MIN == 0.50


def test_the_STRUCTURAL_LAG_comes_off_the_coverage_denominator():
    """A YoY metric's oldest `rpy` rows are NaN BY ARITHMETIC, not by absence.  On a panel
    shorter than window+lag they sit INSIDE the window; counting them as gaps would penalise a
    short history for a reason that is not about the data -- exactly what "rows present" was
    chosen to avoid, one level down.  Measured: it would flag an extra 54 `revenueGrowth` and
    28 `freeCashFlowPerShareGrowth` sources."""
    assert sm.structural_lag('revenueGrowth', 4) == 4
    assert sm.structural_lag('revenueGrowth', 2) == 2, 'the lag is rpy, never a hard-coded 4'
    assert sm.structural_lag('freeCashFlowPerShareGrowth', 4) == 4
    assert sm.structural_lag('priceGrowth', 4) == 1
    assert sm.structural_lag('RoA', 4) == 0
    #  8-row panel, YoY: 4 computable + the 4 structural NaNs -> coverage 4/4, NOT 4/8
    v = [0.1] * 4 + [np.nan] * 4
    assert npol.window_verdict(v, 8, 'revenueGrowth', 4, structural_lag=4) == pytest.approx(0.1)
    #  and the same series WITHOUT a lag is 4/8 = 0.50, i.e. only just adequate -- so one more
    #  missing row would refuse it.  That is the margin the lag correction is protecting.
    assert npol.window_verdict(v, 8, 'RoA', 4) == pytest.approx(0.1)
    assert np.isnan(npol.window_verdict([0.1] * 3 + [np.nan] * 5, 8, 'RoA', 4))


def test_the_structural_lag_is_only_credited_where_it_lands_INSIDE_the_window():
    """A long panel's structural NaNs sit OUTSIDE the window, so nothing may be subtracted --
    otherwise every long-history name gets a free pass on coverage."""
    #  20-row series, 16-row window, 8 computable inside it: coverage 8/16 = 0.50, adequate.
    v = [0.1] * 8 + [np.nan] * 12
    assert npol.window_verdict(v, 16, 'revenueGrowth', 4,
                               structural_lag=4) == pytest.approx(0.1)
    #  7 computable -> 7/16 = 0.4375 -> refused, and the lag must NOT rescue it
    v2 = [0.1] * 7 + [np.nan] * 13
    assert np.isnan(npol.window_verdict(v2, 16, 'revenueGrowth', 4, structural_lag=4))


# =========================================================================== #
#  3. GAPPINESS                                                                #
# =========================================================================== #
def test_calendar_gappiness_is_FREQUENCY_NORMALISED_and_this_is_the_load_bearing_pin():
    """THE SHAPE OF THIS ASSERTION IS THE POINT: the SAME dates must give DIFFERENT answers for
    the two frequencies.  A regression to any fixed-cadence rule fails it.  Measured on the
    panel: cadence-relative flags 24 of 1,108 semi-annual filers; a fixed 3-month expectation
    flags all 1,108."""
    six_monthly = _dates(8, months=6)
    assert npol.calendar_gap_count(six_monthly, rpy=2) == 0, (
        'a semi-annual filer reporting exactly every six months has NO gaps')
    assert npol.calendar_gap_count(six_monthly, rpy=4) >= 2, (
        'the same dates read as gappy for a QUARTERLY filer -- if these two agree, the rule is '
        'not cadence-relative and every semi-annual filer in the universe gets flagged')
    assert npol.expected_month_spacing(2) == 6.0
    assert npol.expected_month_spacing(4) == 3.0


def test_a_real_stoppage_IS_flagged_and_ONE_late_filing_is_not():
    """TWO, not one: one gap is routinely a late filing or a fiscal-year change; two is a
    stoppage.  A threshold of 1 would cost 432 sources (5.59%) and 7 pool names."""
    assert npol.MAX_CALENDAR_GAPS == 1          # refuse at >= 2
    d = _dates(12)
    one_gap = [d[0]] + [x - pd.DateOffset(months=9) for x in d[1:]]
    assert npol.calendar_gap_count(one_gap, rpy=4) == 1
    assert npol.calendar_gap_refused(pd.DataFrame({'date': one_gap}), rpy=4) is False
    two_gaps = ([d[0]] + [x - pd.DateOffset(months=9) for x in d[1:6]]
                + [x - pd.DateOffset(months=21) for x in d[6:]])
    assert npol.calendar_gap_count(two_gaps, rpy=4) >= 2
    assert npol.calendar_gap_refused(pd.DataFrame({'date': two_gaps}), rpy=4) is True


def test_calendar_refusal_hits_EVERY_windowed_metric_and_NO_point_in_time_one():
    """A name that stopped filing has no trustworthy window; its point-in-time metrics are
    untouched, which is also why this rule NEUTRALISES rather than ejecting -- the row still has
    observed metrics, so `normalizeAndDropNA` never drops it."""
    d = _dates(12)
    stopped = ([d[0]] + [x - pd.DateOffset(months=9) for x in d[1:6]]
               + [x - pd.DateOffset(months=21) for x in d[6:]])
    frame = _cdx(n=12)
    frame['date'] = stopped
    assert np.isnan(sm.postbm_metric('RoA', 'netIncome', frame, 16, rpy=4))
    #  point-in-time: Piotroski reads rows 0 and rpy only, and is NOT refused by this rule
    assert not np.isnan(sm.piotroski(frame, rpy=4))


def test_interior_missing_runs_excludes_the_LEADING_and_TRAILING_stretches():
    """A trailing stretch is the structural lag; a leading one is a metric that has stopped
    being computable, which coverage already prices.  Only INTERIOR runs are gaps."""
    assert npol.interior_missing_runs([True] * 8) == 0
    assert npol.interior_missing_runs([np.nan] * 8) == 0
    assert npol.interior_missing_runs([True, True, False, False, True, True]) == 1
    assert npol.interior_missing_runs([True, False, True, False, True]) == 2
    assert npol.interior_missing_runs([False, False, True, True, False, False]) == 0
    assert npol.MAX_INTERIOR_RUNS == 1
    assert np.isnan(npol.window_verdict([1.0, np.nan, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0],
                                        8, 'RoA', 4))
    assert npol.window_verdict([1.0, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0],
                               8, 'RoA', 4) == pytest.approx(1.0)


# =========================================================================== #
#  4. BOUNDARY IMPUTATION AND REFUSAL                                          #
# =========================================================================== #
def test_the_boundary_is_admitted_only_where_the_limit_IS_the_metric_floor():
    """ADDENDUM A1's admissibility test, pinned as a data statement so a future metric cannot be
    added to the boundary path without its own derived limit."""
    assert set(npol.BOUNDARY_LIMIT) == {'grahamNumberToPrice', 'uGrahamNumberToPrice'}
    assert npol.BOUNDARY_LIMIT['grahamNumberToPrice'][0] == 0.0
    assert npol.BOUNDARY_LIMIT['uGrahamNumberToPrice'][0] == 0.0
    for k, (_lim, why) in npol.BOUNDARY_LIMIT.items():
        assert 'lim' in why or 'limit' in why, (
            '%s carries no derivation. ADDENDUM A1 is not automatable, so a boundary without '
            'a written limit is a tuned constant in disguise.' % k)


def test_EPStoEPSmean_is_REFUSED_and_the_reason_is_that_the_limit_is_the_metric_MAXIMUM():
    """THE ONE PLACE THIS BUILD DEPARTS FROM THE SPEC'S NUMBERS, so it is pinned behaviourally
    rather than by comment.

    ADDENDUM A2 assigns this metric a boundary of -1.0.  The shipped formula is
    (epsmean - ewma_recent)/|epsmean| -- mean MINUS recent -- so POSITIVE means the recent year
    sits BELOW the name's own history, which is the mean-reversion side the +0.0516 weight bets
    on.  Driving the non-positive recent EPS to 0+ therefore sends the metric UP, to
    epsmean/|epsmean| = +1, which is the column's MAXIMUM.  A1 admits a boundary only where the
    limit is the metric's WORST admissible value, so its own escape clause fires.

    Asserted the only way that cannot rot: compute the metric on a frame that is AT the boundary
    (recent EPS tiny and positive, so the gate passes) and show it lands ABOVE a healthy peer.
    If that inequality ever reverses, the refusal should be revisited."""
    assert 'EPStoEPSmean' in npol.REFUSED_NOT_IMPUTED
    assert 'EPStoEPSmean' not in npol.BOUNDARY_LIMIT

    n = 24
    healthy = _cdx(n=n, netIncome=[50.0] * n)
    at_boundary = _cdx(n=n, netIncome=[1e-6] * 4 + [50.0] * (n - 4))
    adverse = _cdx(n=n, netIncome=[-50.0] * 4 + [50.0] * (n - 4))

    v_healthy = sm.eps_to_eps_mean(healthy, rpy=4)
    v_boundary = sm.eps_to_eps_mean(at_boundary, rpy=4)
    assert np.isfinite(v_boundary)
    assert v_boundary > v_healthy, (
        'the boundary of EPStoEPSmean lands on the REWARDED side of the metric, which is why it '
        'must be refused rather than imputed -- imputing it would hand an adverse company the '
        'mean-reversion reward')
    #  and the genuinely adverse frame is REFUSED (NaN -> column median), never imputed
    assert np.isnan(sm.eps_to_eps_mean(adverse, rpy=4))


def test_graham_takes_the_boundary_when_ADVERSE_and_is_REFUSED_when_MISSING():
    """`grahamUndefinedReason` is the discriminator, and it is 99.1% adverse / 0.9% gap on the
    measured panel.  Imputing a real value on a genuine gap would put an answer where there is
    none."""
    n = 20
    adverse = _cdx(n=n, grahamNumber=[np.nan] * n,
                   grahamUndefinedReason=['graham_undefined_negative_eps'] * n)
    missing = _cdx(n=n, grahamNumber=[np.nan] * n,
                   grahamUndefinedReason=['graham_missing_inputs'] * n)
    assert sm.postbm_metric('grahamNumberToPrice', 'grahamNumberToPrice',
                            adverse, 16, rpy=4) == 0.0
    assert np.isnan(sm.postbm_metric('grahamNumberToPrice', 'grahamNumberToPrice',
                                     missing, 16, rpy=4))
    #  no reason column at all -> cannot tell -> refuse, never guess
    blind = _cdx(n=n, grahamNumber=[np.nan] * n)
    assert np.isnan(sm.postbm_metric('grahamNumberToPrice', 'grahamNumberToPrice',
                                     blind, 16, rpy=4))


def test_PARTIAL_TypeD_coverage_keeps_its_own_observations_and_is_NOT_imputed():
    """ADDENDUM A's closing clause: a name with 4 of 16 computable Graham quarters WAS profitable
    four times, and imputing "as if earnings were 0" would discard its own observations.  So the
    coverage-to-NaN collapse does not apply to the two Type-D columns either -- that inversion is
    the one the spec calls out by name ("a correctly-computed coverage feeding a median fill on a
    Type-D column CREATES the reward the whole scheme exists to prevent")."""
    n = 16
    g = [20.0] * 4 + [np.nan] * 12          # coverage 4/16 = 0.25, far below 0.50
    frame = _cdx(n=n, grahamNumber=g,
                 grahamUndefinedReason=[''] * 4 + ['graham_undefined_negative_eps'] * 12)
    v = sm.postbm_metric('grahamNumberToPrice', 'grahamNumberToPrice', frame, 16, rpy=4)
    assert v == pytest.approx(2.0), (
        'partial Type-D coverage was collapsed (to NaN) or imputed (to the boundary 0.0); it '
        'must keep the mean of its 4 observed quarters, 20/10 = 2.0')
    assert set(npol.TYPE_D) == {'grahamNumberToPrice', 'EPStoEPSmean'}


def test_stage1_uGraham_boundary_is_BEHAVIOUR_IDENTICAL_to_the_NaN_it_replaces():
    """The strongest available confirmation that the CEO's rule is right, and the reason the
    change is safe: on Stage-1 the criterion is `value - 1 > 0` with Sign +1, so the boundary
    0.0 tests as -1.0 and FAILS -- exactly what `calcByTier` does with the NaN today.  Measured
    behaviour-identical on every one of the 23,212 adverse rows of the panel; the fail is now
    DERIVED rather than incidental."""
    n = 16
    tf = _cdx(n=n, grahamNumber=[np.nan] * n, bookValuePerShare=5.0,
              grahamUndefinedReason=['graham_undefined_negative_eps'] * n)
    vals = cm.apply_boundary_imputation(tf, [np.nan] * n, 'graham_adverse')
    assert all(v == 0.0 for v in vals)
    #  through the REAL scorer: a unity criterion at 0.0 scores exactly what a NaN scores
    scored_boundary = cs.calcByTier('unity', 'S', 1, pd.Series(vals), 0.0, 'u', 8)
    scored_nan = cs.calcByTier('unity', 'S', 1, pd.Series([np.nan] * n), 0.0, 'u', 8)
    assert scored_boundary == scored_nan == 0.0

    #  a MISSING-input row is left NaN, not filled
    tf2 = _cdx(n=n, grahamNumber=[np.nan] * n,
               grahamUndefinedReason=['graham_missing_inputs'] * n)
    assert all(np.isnan(v) for v in cm.apply_boundary_imputation(tf2, [np.nan] * n,
                                                                 'graham_adverse'))
    #  an OBSERVED value is never overwritten
    tf3 = _cdx(n=n, grahamNumber=[8.0] * n,
               grahamUndefinedReason=['graham_undefined_negative_eps'] * n)
    assert cm.apply_boundary_imputation(tf3, [0.8] * n, 'graham_adverse') == [0.8] * n


def test_every_declared_Boundary_has_a_predicate_and_an_unknown_one_raises():
    dicts = cdic.getBaseMeanDiffUnitySpecialDicts()
    declared = {k: v['Boundary'] for d in dicts for k, v in d.items() if v.get('Boundary')}
    assert declared, 'the boundary-imputation change declares a Boundary; none found'
    for metric, b in declared.items():
        assert b in cm.STAGE1_BOUNDARY_IMPUTATIONS, (metric, b)
    with pytest.raises(KeyError, match='no boundary named'):
        cm.apply_boundary_imputation(_cdx(n=2), [1.0, 2.0], 'not_a_boundary')
    with pytest.raises(ValueError, match='flags for'):
        cm.apply_boundary_imputation(_cdx(n=3), [1.0, 2.0], 'graham_adverse')


def test_a_GUARD_refused_row_is_never_REFILLED_by_a_boundary():
    """ENFORCED, not merely ordered (review finding).  Both mechanisms express refusal as NaN, so
    "the boundary only fills rows that are still NaN" is exactly what would refill a row the guard
    had just refused -- the guard-then-boundary ORDERING causes that, it does not prevent it.

    Not live today (one Boundary, six Guards, no criterion declares both), which is why it is
    closed now rather than after the first criterion that declares both."""
    n = 8
    tf = _cdx(n=n, grahamNumber=[np.nan] * n,
              grahamUndefinedReason=['graham_undefined_negative_eps'] * n)
    #  no guard -> the boundary fills, as it must
    assert cm.apply_boundary_imputation(tf, [np.nan] * n, 'graham_adverse') == [0.0] * n
    #  a guard that refused every row -> the boundary must fill NOTHING
    refused = pd.Series([False] * n)
    assert all(np.isnan(v) for v in cm.apply_boundary_imputation(
        tf, [np.nan] * n, 'graham_adverse', admissible=refused))
    #  half-and-half: only the admitted rows take the limit
    half = pd.Series([True] * 4 + [False] * 4)
    got = cm.apply_boundary_imputation(tf, [np.nan] * n, 'graham_adverse', admissible=half)
    assert got[:4] == [0.0] * 4 and all(np.isnan(v) for v in got[4:])
    #  and the production seam passes the guard mask rather than trusting the order
    src = open('getData_fmp.py', encoding='utf-8').read()
    assert 'admissible=adm' in src, (
        'build_bometric_rows no longer tells the boundary which rows the guard admitted -- a '
        'criterion declaring BOTH keys would have its guard silently undone')


# =========================================================================== #
#  5. WHAT WAS DELIBERATELY NOT BUILT                                          #
# =========================================================================== #
def test_no_stage2_returnOnEquity_FLOOR_was_built():
    """CLOSED BY THE CEO (2026-08-05): "Such a company (negative equity with high earnings)
    could be surging due to good investments. But that would pop up as positive somewhere else.
    We just need to make sure it is not treated positively here."  Refusing the metric puts the
    name at the column MEDIAN -- neutral, not positive -- so the requirement is already met and
    a floor would be a second punishment for one fact.

    Pinned so a later reader does not re-open it from the note in `postbm_metric`: a fully
    refused name comes out NaN, and NOT at any sentinel or squash floor."""
    n = 8
    dn = pd.DataFrame({'returnOnEquity': [0.25] * n,
                       'totalStockholdersEquity': [-500.0] * n,
                       'date': _dates(n)})
    v = sm.postbm_metric('returnOnEquity', 'returnOnEquity', dn, n, rpy=4)
    assert np.isnan(v), 'a refused ROE must be NaN (-> column median), not a floor value'
    assert not any('floor' in str(k).lower() for k in vars(npol)
                   if 'ROE' in str(k) or 'EQUITY' in str(k).upper())


def test_no_refused_vs_missing_channel_was_built():
    """Also ruled out.  Observability is a run-level COUNT instead, so no consumer of
    `postScoreMetric` has to learn a second identity for a cell."""
    assert isinstance(npol.POLICY_COUNTS, list)
    npol.POLICY_COUNTS.clear()
    npol.window_verdict([1.0] * 2 + [np.nan] * 6, 8, 'RoA', 4)
    f = npol.counts_frame()
    assert list(f.columns) == ['pool', 'column', 'rule', 'n']
    assert f.iloc[0]['rule'] == npol.RULE_COVERAGE
    #  and the counter is the ONLY output: the reduced value carries no sentinel
    assert np.isnan(npol.window_verdict([1.0] * 2 + [np.nan] * 6, 8, 'RoA', 4))
    npol.POLICY_COUNTS.clear()


# =========================================================================== #
#  6. THE CALL SITES -- a fix applied to some of them is this repo's signature  #
# =========================================================================== #
@pytest.mark.parametrize('path,needle', [
    ('postBoRank.py', 'tempcdx=tempcdx'),
    ('baseline_tools/stage2_pit.py', 'tempcdx=tempcdx'),
])
def test_the_two_SCORING_paths_pass_the_frame_to_the_two_series_only_metrics(path, needle):
    """`free_cash_flow_yield` and `free_cash_flow_per_share_growth` are the only windowed
    metrics that do not otherwise receive `tempcdx`, so they are the only two that can silently
    skip the NAME-level calendar-gap test.  Production and the certified PIT reproduction must
    both pass it; `tempcdx` is optional purely so the three diagnostic call sites keep working."""
    src = open(path, encoding='utf-8').read()
    i = src.index('free_cash_flow_yield(')
    assert needle in src[i:i + 400], '%s does not pass the frame to free_cash_flow_yield' % path
    j = src.index('free_cash_flow_per_share_growth(')
    assert needle in src[j:j + 400], '%s does not pass it to the per-share growth metric' % path


#  key -> how to compute it from ONE source's frame.  The KEY LIST is NOT written here: it is
#  derived from the registry by `sm.windowed_metric_keys()`, and the completeness assertion below
#  is what makes an omission impossible.  That is the whole point -- the previous version of this
#  test hand-listed SEVEN functions and omitted `eps_to_eps_mean`, which was the one metric that
#  had silently opted out of the seam.  A test named for enumerating *every* windowed metric that
#  can miss one is decorative.
_WINDOWED_DISPATCH = {
    'freeCashFlowYield': lambda f: sm.free_cash_flow_yield(
        f['freeCashFlow'], f['marketCap'], 16, rpy=4, tempcdx=f),
    'freeCashFlowPerShareGrowth': lambda f: sm.free_cash_flow_per_share_growth(
        f['freeCashFlow'], f['weightedAverageShsOut'], 16, rpy=4, tempcdx=f),
    'tbVpRatio': lambda f: sm.tbv_p_ratio(f, 16, rpy=4),
    'priceGrowth': lambda f: sm.price_growth(f, 16, rpy=4),
    'EPStoEPSmean': lambda f: sm.eps_to_eps_mean(f, rpy=4),
    'CycleHeat': lambda f: sm.cycleheat(f, rpy=4),
}


def _windowed_value(key, frame):
    """`key`'s value on `frame`, via whichever entry point production uses for it."""
    import scoringWeights as sw
    if key in sw.POSTBM_EQMET:
        return sm.postbm_metric(key, sw.POSTBM_EQMET[key], frame, 16, rpy=4)
    return _WINDOWED_DISPATCH[key](frame)


def test_the_windowed_metric_enumeration_is_DERIVED_and_covers_every_registry_key():
    """The guard on the guard.  If a metric is added to STAGE2_METRIC_SPEC with a windowed basis
    and nobody adds it here, THIS fails -- rather than the seam test quietly checking one metric
    fewer."""
    import scoringWeights as sw
    keys = set(sm.windowed_metric_keys())
    assert keys, 'the registry declares no windowed metric, which cannot be right'
    covered = set(sw.POSTBM_EQMET) | set(_WINDOWED_DISPATCH)
    missing = keys - covered
    assert not missing, (
        'windowed registry key(s) with no way to compute them in this test: %s. Add them to '
        '_WINDOWED_DISPATCH -- do NOT relax this assertion, it is what stops the seam test '
        'below from silently covering one metric fewer.' % sorted(missing))
    #  and the derived set must actually contain the two non-postBm window bases, so a future
    #  edit to `windowed_metric_keys` cannot narrow it to the postBm block by accident
    assert {'EPStoEPSmean', 'CycleHeat'} <= keys
    #  DcfToPrice is WINDOW_SCORING but FREQ_NOT_A_TIME_SERIES -- its window runs over a DCF
    #  frame with its own cadence, so it is excluded by a DERIVED rule, not a carve-out list
    assert 'DcfToPrice' not in keys


def test_every_windowed_metric_goes_through_the_ONE_seam():
    """BEHAVIOURAL, and enumerated FROM THE REGISTRY.

    Asserted on a frame with TWO filing stoppages: every windowed metric must come back NaN,
    because a company that stopped filing twice has no trustworthy window for any of them.  This
    is the assertion that caught `eps_to_eps_mean` -- on this exact shape of frame it used to
    return a real value while `RoA` and `CycleHeat` returned NaN.
    """
    clean = _cdx(n=24)
    d = _dates(24)
    stopped = ([d[0]] + [x - pd.DateOffset(months=9) for x in d[1:8]]
               + [x - pd.DateOffset(months=21) for x in d[8:]])
    gappy = _cdx(n=24)
    gappy['date'] = stopped
    assert npol.calendar_gap_count(stopped, rpy=4) >= 2, 'the fixture stopped being gappy'

    keys = sorted(sm.windowed_metric_keys())
    finite_on_clean = 0
    for k in keys:
        assert np.isnan(_windowed_value(k, gappy)), (
            '%s survived TWO filing stoppages -- it is a registered WINDOWED metric and is not '
            'going through nan_policy.window_verdict. Route it through stage2_metrics._reduce.'
            % k)
        if np.isfinite(pd.to_numeric(pd.Series([_windowed_value(k, clean)]),
                                     errors='coerce').iloc[0]):
            finite_on_clean += 1
    assert finite_on_clean >= len(keys) - 3, (
        'only %d of %d windowed metrics are computable on the CLEAN fixture, so the NaN '
        'assertions above are near-vacuous -- fix the fixture, not the assertion'
        % (finite_on_clean, len(keys)))

    #  the POINT-IN-TIME metrics must NOT be refused by a name-level window rule
    assert not np.isnan(sm.piotroski(gappy, rpy=4))
    assert not np.isnan(sm.share_count_change(gappy, rpy=4))

    #  and no windowed metric may still end in a bare reduction
    body = inspect.getsource(sm)
    body = body[body.index('def postbm_metric'):]
    assert '.head(w).mean()' not in body, (
        'a windowed metric still reduces with a bare head(w).mean() -- route it through '
        '`_reduce` so it inherits the NaN policy')


def test_the_scoring_window_fallback_is_a_NAMED_constant():
    """`CycleHeat` and `EPStoEPSmean` carry a 28-quarter BASELINE, not the scoring window, so
    they cannot pass the ambient `nq` to the calendar test and fall back to this.  A named
    constant is what makes a future divergence visible."""
    assert npol.SCORING_WINDOW_NQ == 16
    assert 'scoring_nq=nq' in inspect.getsource(sm.postbm_metric), (
        'postbm_metric HAS the ambient nq and must pass it rather than take the fallback')


def test_a_STALE_saved_panel_is_detected_because_no_column_was_renamed():
    """THE SILENT-BASIS-MIX HAZARD, closed.  This change altered two Stage-1 CRITERION COLUMNS
    (`uGrahamNumberToPrice` takes a boundary value; `PEG` is computed locally) and renamed
    NEITHER, so `calcScore`'s schema gate -- which is column-EXACT on names -- would pass a
    pre-change panel on a `-loadbometric` run and score old columns with new code.
    `utils.check_panel_basis` cannot see it: cdx is unchanged.

    The detector is an EXACT ZERO, which is why it is reliable and not a heuristic:
    grahamNumber/price is a ratio of two continuous positive quantities and lands on exactly 0.0
    with probability zero, so any exact zeros mean the boundary ran."""
    n = 2000
    new = pd.DataFrame({'uGrahamNumberToPrice': [0.0] * 700 + [1.2] * 1300})
    old = pd.DataFrame({'uGrahamNumberToPrice': [np.nan] * 700 + [1.2] * 1300})
    assert utils.check_bometric_basis({'BoMetric_df': new}, verbose=False) == 'new'
    assert utils.check_bometric_basis({'BoMetric_df': old}, verbose=False) == 'old'
    #  a small panel, or one with no adverse row at all, must be UNKNOWN rather than a false alarm
    assert utils.check_bometric_basis(
        {'BoMetric_df': pd.DataFrame({'uGrahamNumberToPrice': [1.2] * 50})},
        verbose=False) == 'unknown'
    assert utils.check_bometric_basis(
        {'BoMetric_df': pd.DataFrame({'uGrahamNumberToPrice': [1.2] * n})},
        verbose=False) == 'unknown'
    assert utils.check_bometric_basis({}, verbose=False) == 'unknown'
    #  and it is WIRED into the load path, not merely available
    assert 'check_bometric_basis' in open('Sbocker.py', encoding='utf-8').read()


# =========================================================================== #
#  7. PEG -- computed locally                                                  #
# =========================================================================== #
def test_peg_eps_basis_is_pinned_and_cannot_switch_silently():
    """`eps` / `epsdiluted` are captured at ingest but ABSENT from every saved panel; they
    populate on the next full fetch.  The switch must be a DELIBERATE edit, never an
    `eps if present else proxy` fallback -- which would change a scored criterion's basis on the
    first fetch that carried the column, with nothing in the run to say it had."""
    assert cm._PEG_EPS_FIELD == 'netIncomePerShare'
    src = inspect.getsource(cm.peg_local)
    assert "_PEG_EPS_FIELD" in src
    assert "'eps'" not in src and '"eps"' not in src, (
        'the local PEG must not reference `eps` directly -- it reads the pinned constant, so '
        'switching basis is one visible edit')
    #  and the ingest still captures them, so the deliberate switch stays available at no cost
    prereq = cdic.getPreReqDict()
    assert 'eps' in prereq['inc'] and 'epsdiluted' in prereq['inc']


def test_peg_is_TTM_on_BOTH_legs_and_annual_on_the_growth_leg():
    """The vendor divided an ANNUALISED (x4) quarterly P/E by a QUARTERLY growth percentage --
    dimensionally incoherent.  Both legs are now trailing-year, and the growth rate is per YEAR."""
    eps = [1.0] * 4 + [0.5] * 8
    df = pd.DataFrame({'netIncomePerShare': eps, 'price': [40.0] * len(eps)})
    peg, now, prev = cm.peg_local(df, rpy=4)
    assert now.iloc[0] == pytest.approx(4.0), 'the P/E leg must be a TRAILING YEAR, not 4 x a quarter'
    assert prev.iloc[0] == pytest.approx(2.0)
    assert peg.iloc[0] == pytest.approx(0.10)      # PE 10 / growth 100%
    #  SEMI-ANNUAL: two rows are a trailing year, and the lag is two rows -- so the same
    #  economics give the same PEG.  A hard-coded 4 would read two years of growth as one.
    eps_sa = [2.0] * 2 + [1.0] * 4
    df_sa = pd.DataFrame({'netIncomePerShare': eps_sa, 'price': [40.0] * len(eps_sa)})
    peg_sa, now_sa, prev_sa = cm.peg_local(df_sa, rpy=2)
    assert now_sa.iloc[0] == pytest.approx(4.0)
    assert prev_sa.iloc[0] == pytest.approx(2.0)
    assert peg_sa.iloc[0] == pytest.approx(0.10)


def _peg_of(now, prev, price=10.0, crossing_growth=None):
    eps = [now] * 4 + [prev] * 8
    df = pd.DataFrame({'netIncomePerShare': eps, 'price': [price] * len(eps)})
    return cm.peg_local(df, rpy=4, crossing_growth=crossing_growth)[0].iloc[0]


def test_peg_growth_denominator_is_ABS_of_the_base_ONLY_where_the_base_is_POSITIVE():
    """The four sign states through `peg_local` alone -- the BUILD stage.

    `(E_now - E_prev)/|E_prev|` reduces to the ordinary growth rate when the base is positive.
    When the base is NON-POSITIVE it is not a growth rate at all -- it is an artifact of the
    base's sign -- so the row comes out NaN at build time and is filled from the POOL later.
    See calcMetrics.PEG_CROSSING_SUBSTITUTION."""
    #  no positive current earnings -> no P/E -> refused, so the old false-pass cells cannot
    #  reappear through the sign cancellation that produced them
    assert np.isnan(_peg_of(-0.5, 0.5))
    assert np.isnan(_peg_of(-0.5, -0.5))
    #  declining earnings -> negative PEG -> the criterion fails it (1/PEG - 1 < 0)
    assert _peg_of(0.5, 1.0) < 0
    #  a POSITIVE base is an ordinary growth rate and needs no pool
    assert _peg_of(1.0, 0.5) > 0
    #  THE CROSSING ROW IS NaN AT BUILD TIME, deliberately: the substitution value is the POOL's
    #  median growth, and `calc_special` sees one source (and on the fetch path the panel does not
    #  yet exist).  A build-time value here would have to be a tuned constant.
    assert np.isnan(_peg_of(0.5, -0.5))


def test_the_crossing_NERF_removes_the_DEPTH_OF_THE_PRIOR_LOSS_from_the_answer():
    """THE CEO'S RULING (2026-08-05): the criterion must not treat an unassessable state
    positively, and the crossing is nerfed by SUBSTITUTION rather than by a chosen number.

    THE DEFECT, stated as the property it violates: under `|E_prev|` the growth rate GROWS WITH
    HOW BAD THE PRIOR YEAR WAS, so a deeper prior loss bought a cheaper PEG.  Pinned by showing
    that two companies with the SAME current earnings and the SAME price but DIFFERENT prior
    losses used to get different answers, and now get the same one."""
    #  the two |base| growth rates the old form would have produced -- they differ, which is the
    #  defect: nothing about the CURRENT company differs between these two rows
    shallow = 100.0 * (0.5 * 4 - (-0.5 * 4)) / abs(-0.5 * 4)        # +200%/yr
    deep = 100.0 * (0.5 * 4 - (-4.0 * 4)) / abs(-4.0 * 4)           # +112.5%/yr
    assert shallow != deep, 'the fixture must actually differ in prior-loss depth'

    #  AFTER: both take the SAME pool median, so the prior loss no longer enters the answer and
    #  the row is decided by its own P/E -- neither credit nor penalty for the crossing.
    med = 25.0
    a = _peg_of(0.5, -0.5, crossing_growth=med)
    b = _peg_of(0.5, -4.0, crossing_growth=med)
    assert a == pytest.approx(b), (
        'two crossing rows with identical current earnings and price still get different PEGs -- '
        'the depth of the prior loss is still leaking into the answer')
    #  and the value is exactly PE / median: PE = 10 / (4 x 0.5) = 5, so 5/25 = 0.20
    assert a == pytest.approx(0.20)
    #  P/E still decides: double the price, double the PEG
    assert _peg_of(0.5, -0.5, price=20.0, crossing_growth=med) == pytest.approx(0.40)
    #  NO TUNED CONSTANT: the substitution value is the pool median and the constant says so by
    #  name.  A hard-coded number would be the only tuned constant on this path -- the same ground
    #  on which a relative |PEG| floor was refused.
    assert cm.PEG_CROSSING_SUBSTITUTION == 'pool_median_growth'


def test_the_pool_median_growth_is_taken_over_the_rows_the_criterion_can_SCORE():
    """Not over "every row where the arithmetic works".  The crossing rows are excluded BY
    CONSTRUCTION -- they are the rows being substituted, so including them would let the artifact
    define its own replacement."""
    def src(name, eps):
        n = len(eps)
        return pd.DataFrame({
            'source': [name] * n, 'date': _dates(n),
            'netIncomePerShare': eps, 'price': [10.0] * n,
            rp.FREQ_COLUMN: [rp.QUARTERLY] * n})
    #  three growers at +100% / +50% / +25%, plus a crossing name whose |base| growth would be a
    #  huge outlier if it were (wrongly) included in the population
    pool = pd.concat([src('A', [1.0] * 4 + [0.5] * 8),
                      src('B', [1.5] * 4 + [1.0] * 8),
                      src('C', [1.25] * 4 + [1.0] * 8),
                      src('T', [0.5] * 4 + [-0.01] * 8)], ignore_index=True)
    med, n = cm.peg_pool_median_growth(pool)
    assert n > 0 and np.isfinite(med)
    med_without_T, n_without = cm.peg_pool_median_growth(
        pool[pool['source'] != 'T'].reset_index(drop=True))
    assert med == pytest.approx(med_without_T), (
        'a sign-crossing row leaked into the median it is going to be substituted WITH')
    assert n == n_without

    #  no in-domain row -> no median -> the crossing row stays REFUSED (the pre-change answer)
    #  rather than being filled with something invented
    assert not np.isfinite(cm.peg_pool_median_growth(
        pool[pool['source'] == 'ZZZ'])[0])


def test_the_substitution_is_APPLIED_at_exactly_one_seam_and_does_not_mutate_the_panel():
    """`postBo.postBoWrapper`, immediately before Stage-1 scoring -- the same position, and for
    the same reason, as `calcScore.getAves2`'s `BoMetric_ave` (audit H-1: a cross-sectional
    baseline is recomputed on the frame ACTUALLY SCORED, never carried stale, and never frozen
    into the saved panel).

    `build_bometric_rows` has four call sites and `fixAfterGetData` four; a fix applied to three
    of four is this project's signature defect, which is why the substitution is at the ONE seam
    every Stage-1 path passes through instead."""
    src = open('postBo.py', encoding='utf-8').read()
    i = src.index('substitute_peg_crossing')
    j = src.index('simpleScore_fromDict(bmdf')
    assert i < j, 'the substitution must run BEFORE Stage-1 scores the panel'
    assert src.count('substitute_peg_crossing') == 1, (
        'the crossing substitution is applied at more than one place -- it is a cross-sectional '
        'baseline and must have exactly one seam')
    #  and it must NOT be wired into the per-source BUILDERS, which cannot hold a pool quantity
    import os as _os
    callers = []
    for root, dirs, files in _os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', '_quarantine')]
        for fn in files:
            if not fn.endswith('.py'):
                continue
            full = _os.path.join(root, fn)
            body = open(full, encoding='utf-8', errors='ignore').read()
            #  the ATTRIBUTE form, so the scan cannot match the `def` line -- and so this file's
            #  own scan string cannot exclude this file, which is exactly what the first version
            #  of this assertion did to itself.
            if '.substitute_peg_crossing(' in body:
                callers.append(_os.path.relpath(full, '.').replace(chr(92), '/'))
    #  The property is about PRODUCTION modules -- a test may call it as often as it likes, and
    #  pinning the test list too would make this assertion churn on every new pin.
    prod = sorted(c for c in callers
                  if not _os.path.basename(c).startswith('test_')
                  and 'baseline_tools/' not in c)
    assert prod == ['postBo.py'], (
        'substitute_peg_crossing is called from %s. It is a CROSS-SECTIONAL baseline and belongs '
        'at exactly ONE production seam -- `postBo.postBoWrapper`, immediately before Stage-1. '
        'The per-source builders cannot hold a pool quantity, and a second production seam would '
        'apply the substitution twice or inconsistently.' % prod)

    #  NO MUTATION: the caller replaces its own local, so the artifact on disk keeps the honest
    #  per-source pre-substitution column.  The pool needs a GROWER as well as the crossing name,
    #  or there is no in-domain row for a median and the function correctly refuses -- which is
    #  itself worth stating, because the first version of this fixture had only the crossing name
    #  and the test passed for the wrong reason.
    bm = pd.DataFrame({'source': ['T'] * 4, 'date': _dates(4), 'PEG': [np.nan] * 4})
    cdx = pd.concat([
        pd.DataFrame({'source': ['T'] * 12, 'date': _dates(12),
                      'netIncomePerShare': [0.5] * 4 + [-0.5] * 8,
                      'price': [10.0] * 12, rp.FREQ_COLUMN: [rp.QUARTERLY] * 12}),
        pd.DataFrame({'source': ['G'] * 12, 'date': _dates(12),
                      'netIncomePerShare': [1.0] * 4 + [0.5] * 8,
                      'price': [10.0] * 12, rp.FREQ_COLUMN: [rp.QUARTERLY] * 12}),
    ], ignore_index=True)
    before = bm['PEG'].copy()
    out, stats = cm.substitute_peg_crossing(bm, cdx, verbose=False)
    assert bm['PEG'].equals(before), 'substitute_peg_crossing mutated its input frame'
    assert out is not bm
    assert stats['n_crossing_rows'] >= 1 and stats['n_filled'] >= 1
    assert np.isfinite(pd.to_numeric(out['PEG'], errors='coerce').iloc[0])


def test_pegs_domain_is_stated_ONCE_and_nowhere_else():
    """The reason PEG carries no `Guard`: a guard predicate has no `rpy` parameter, so it would
    have to re-derive the filer's frequency from the stamp while `peg_local` receives it from the
    caller.  Two statements of one domain, resolved from two different places, is the pair this
    repo keeps getting bitten by.  Pinned so a well-meaning tidy-up cannot re-create it."""
    assert 'PEG' not in {k: v for d in cdic.getBaseMeanDiffUnitySpecialDicts()
                         for k, v in d.items() if v.get('Guard')}
    assert not any('peg' in g.lower() for g in cm.STAGE1_DOMAIN_GUARDS), (
        'a PEG guard is back in the registry -- see calcMetrics.peg_local')
    #  and the single statement really is the one that gates the criterion, at either frequency
    for rpy, n_now in ((4, 4), (2, 2)):
        eps = [-1.0] * n_now + [1.0] * 8            # no positive trailing year NOW
        df = pd.DataFrame({'netIncomePerShare': eps, 'price': [10.0] * len(eps)})
        assert np.isnan(cm.peg_local(df, rpy=rpy)[0].iloc[0]), rpy
