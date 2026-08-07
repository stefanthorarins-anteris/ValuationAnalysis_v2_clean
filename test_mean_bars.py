"""Targeted tests for the Stage-1 absolute mean bars (meanBars, register C-12, 2026-08-06).

WHAT IS PINNED HERE: that the bar table's key set is EXACTLY the live weight-bearing
`BoMetric_mean_dict` (the same single-source shape as
`test_scoring_weights_single_source.py`), that the two YIELD bars are on the per-quarter
basis the flow correction actually produces, that the scorer reads the constant rather than
the pooled median, and the failsafe band's report-only contract.

NOT pinned: the LEVELS.  A test asserting 0.50 or 0.0075 would just restate the table, and
the levels are a CEO decision that is meant to be a one-constant edit.
"""

import numpy as np
import pandas as pd
import pytest

import calcScore as cs
import createDicts as cdic
import meanBars as mb
import reporting_period as rp


def _mcol(key):
    return 'm' + key[0].upper() + key[1:]


def _live_mean_dict():
    return cdic.getDicts()[3]


def _tier_weight(tier):
    """Probed from calcByTier ITSELF, not mirrored -- same device as
    test_scoring_weights_single_source._tier_weight."""
    return float(cs.calcByTier('diff', tier, 1, pd.Series([1.0]), 0.0, 'probe', 1))


# --------------------------------------------------------------------------- #
#  the single-source pin
# --------------------------------------------------------------------------- #
def test_every_weight_bearing_mean_criterion_has_a_bar_and_nothing_else_does():
    """*** THE PIN. ***  If these drift apart, a criterion is being scored against the POOLED
    MEDIAN again -- sample-dependent and a lookahead channel in any backtest -- with no
    visible symptom in the output.  Tier-N (w = 0) criteria are excluded by DERIVATION from
    calcByTier's ladder, not by a hard-coded name."""
    live = _live_mean_dict()
    weighted = {_mcol(k) for k, spec in live.items() if _tier_weight(spec['Tier']) > 0}
    assert set(mb.BARS) == weighted, (
        'meanBars.BARS no longer matches the weight-bearing mean criteria.\n'
        '  missing a bar : %s\n  bar with no live weighted criterion : %s'
        % (sorted(weighted - set(mb.BARS)), sorted(set(mb.BARS) - weighted)))
    #  and every declared exception must be a real, still-weightless criterion
    for mcol in mb.NO_BAR:
        key = [k for k in live if _mcol(k) == mcol]
        assert key, 'meanBars.NO_BAR names %r, which is not a mean criterion' % mcol
        assert _tier_weight(live[key[0]]['Tier']) == 0.0, (
            '%s is in NO_BAR but now CARRIES WEIGHT -- it is being scored against the pooled '
            'median. Give it a bar in BARS.' % mcol)


def test_an_undeclared_mean_criterion_raises_rather_than_silently_using_the_median():
    with pytest.raises(KeyError, match='POOLED MEDIAN'):
        mb.mean_bar('mSomethingNobodyRuledOn', 0.123)


def test_a_declared_exception_still_gets_the_pooled_median():
    assert mb.mean_bar('mSalesToMarketCap', 0.777) == 0.777


def test_every_bar_carries_its_provenance():
    """The table is the record of a DECISION, not just a number: a bar with no rationale is a
    magic constant and the next person cannot tell whether it may be changed."""
    for mcol, spec in mb.BARS.items():
        for field in ('value', 'units', 'rationale', 'pass_rate_at_set', 'panel_at_set',
                      'date_set', 'round_to', 'annual_basis'):
            assert field in spec and spec[field] is not None, (mcol, field)
        assert np.isfinite(spec['value'])


# --------------------------------------------------------------------------- #
#  the basis -- the one way this change could silently be 4x wrong
# --------------------------------------------------------------------------- #
def test_the_yield_bars_are_on_the_PER_QUARTER_basis_the_flow_correction_produces():
    """*** THE 4x TRAP. ***  `earningsYield` and `freeCashFlowToMarketCap` are corrected
    'per_quarter', so the panel column is an annual rate over %g.  The bars are stated on that
    basis.  Switching either to 'annualize' would multiply the column by %g against an
    unchanged bar and QUADRUPLE the effective threshold, silently, because the key still
    resolves and still returns a plausible factor.
    """ % (rp.DEFAULT_ROWS_PER_YEAR, rp.DEFAULT_ROWS_PER_YEAR)
    for key, mcol in (('earningsYield', 'mEarningsYield'),
                      ('freeCashFlowToMarketCap', 'mFreeCashFlowToMarketCap')):
        leg, mode = rp.STAGE1_FLOW_CORRECTION[key]
        assert mode == 'per_quarter', (
            '%s moved to %r. The bar in meanBars is stated PER QUARTER; on an annualised '
            'column it becomes a %gx harder threshold. Change both in the same edit.'
            % (key, mode, rp.DEFAULT_ROWS_PER_YEAR))
        assert mb.BARS[mcol]['annual_basis'] is True
        #  the stored value IS <annual>/Q -- reconstructing the annual rate must be a round
        #  number of basis points, which a bare per-quarter decimal would not be
        annual = mb.BARS[mcol]['value'] * rp.DEFAULT_ROWS_PER_YEAR
        assert abs(annual * 10000 - round(annual * 10000)) < 1e-9, (
            '%s does not reconstruct to a clean annual rate -- store it as '
            '<annual> / rp.DEFAULT_ROWS_PER_YEAR so the basis stays legible' % mcol)


def test_the_two_yield_bars_are_DELIBERATELY_DIFFERENT():
    """*** ANTI-TIDYING PIN (CEO, 2026-08-06). ***  The two yield bars look like a matched pair
    and are not: `mEarningsYield` is set to a CLASSICAL ANCHOR (P/E <= 25, Graham's 2x-AAA) and
    carries the gate's value stance; `mFreeCashFlowToMarketCap` is a floor on cash reality with
    no such anchor.  Equalising them would be a silent design change wearing a cleanup's name,
    so the asymmetry is asserted rather than merely commented.

    This pins the RELATION, not the levels -- either bar may still be re-set on its own.
    """
    ey = mb.BARS['mEarningsYield']['value']
    fcf = mb.BARS['mFreeCashFlowToMarketCap']['value']
    assert ey != fcf, (
        'the earnings and FCF yield bars were harmonised. They are deliberately different: '
        'read both `rationale` fields before changing this.')
    assert ey > fcf, (
        'the earnings bar must stay the STRICTER of the two -- it is the one carrying the '
        "gate's value stance, and the FCF bar is only a floor")


def test_the_earnings_bar_pass_rate_at_set_is_recorded_BELOW_the_others():
    """43.5% is the CHOSEN outcome, not a calibration miss: the CEO took the classically
    defensible bar over the ~50%-pass alternative.  Recorded so a later reader does not
    "correct" it back toward the pack -- and it must still sit inside the failsafe band, or
    the very first run would warn about a bar we deliberately set."""
    r = mb.BARS['mEarningsYield']['pass_rate_at_set']
    assert r < min(s['pass_rate_at_set'] for m, s in mb.BARS.items()
                   if m != 'mEarningsYield')
    assert mb.BAND_LOW < r < mb.BAND_HIGH, (
        'the earnings bar is set outside its own failsafe band -- it would breach on the '
        'first full run, which makes the watchdog noise instead of signal')


def test_a_quarterly_source_is_still_a_no_op_on_the_flow_factor():
    """The per-quarter factor is 1.0 for a quarterly filer, so stating the bar per quarter
    costs the quarterly path nothing -- which is why the basis choice is free."""
    assert rp.stage1_flow_factor('earningsYield', rp.DEFAULT_ROWS_PER_YEAR) == 1.0
    assert rp.stage1_flow_factor('freeCashFlowToMarketCap', 2) == 0.5


# --------------------------------------------------------------------------- #
#  the scorer actually reads it
# --------------------------------------------------------------------------- #
def test_calcByTier_scores_the_mean_family_against_the_CONSTANT_not_the_median():
    """End-to-end on the real scorer: a value just above the bar passes and one just below
    fails, whatever median is passed in.  Pinning it through `calcByTier` rather than by
    re-implementing the comparison is deliberate -- the direction is only realised there."""
    for mcol, spec in mb.BARS.items():
        key = [k for k in _live_mean_dict() if _mcol(k) == mcol][0]
        spec_live = _live_mean_dict()[key]
        bar = mb.mean_bar(mcol, 999.0)          # an absurd "median" that must be ignored
        assert bar == spec['value']
        eps = max(abs(bar), 1.0) * 1e-6
        better = bar + eps * spec_live['Sign']
        worse = bar - eps * spec_live['Sign']
        w = _tier_weight(spec_live['Tier'])
        got_better = cs.calcByTier('mean', spec_live['Tier'], spec_live['Sign'],
                                   pd.Series([better]), bar, key, 1)
        got_worse = cs.calcByTier('mean', spec_live['Tier'], spec_live['Sign'],
                                  pd.Series([worse]), bar, key, 1)
        assert got_better == w and got_worse == 0.0, (mcol, got_better, got_worse, w)


# --------------------------------------------------------------------------- #
#  the failsafe band
# --------------------------------------------------------------------------- #
def _panel(mcol, values, n_sources=1):
    rows = []
    for s in range(n_sources):
        for i, v in enumerate(values):
            rows.append({'source': 'S%d' % s,
                         'date': pd.Timestamp('2026-03-31') - pd.DateOffset(months=3 * i),
                         mcol: v})
    df = pd.DataFrame(rows)
    for other in mb.BARS:
        if other not in df.columns:
            df[other] = np.nan
    return df


def _signs():
    return {_mcol(k): v['Sign'] for k, v in _live_mean_dict().items()}


def test_the_denominator_is_OBSERVED_cells_only():
    """*** The one that keeps the band from moving when COVERAGE moves. ***  Guard-refused and
    non-computable cells are NaN and are excluded from BOTH numerator and denominator.
    `mDebtEquityRatio` is guard-refused on ~6% of cells today, so counting them as failures
    would report ~6pp low and could trip the band on a data-coverage change."""
    bar = mb.BARS['mDebtEquityRatio']['value']
    #  4 passing (below the bar, Sign -1), 4 failing, 8 refused
    vals = [bar - 0.1] * 4 + [bar + 0.1] * 4 + [np.nan] * 8
    cal = mb.calibrate(_panel('mDebtEquityRatio', vals), _signs(), window_rows=16)
    row = cal[cal['criterion'] == 'mDebtEquityRatio'].iloc[0]
    assert row['n_observed'] == 8 and row['n_pass'] == 4 and row['pass_rate'] == 0.5, (
        'refused cells reached the denominator -- the reported rate now moves with data '
        'coverage, which is exactly what a stored bar exists to stop')


def test_a_breach_is_reported_and_the_bar_is_NEVER_changed():
    """*** REPORT AND RECORD, NEVER AUTO-ADJUST. ***  A bar that re-fits itself to hold a pass
    rate IS the pooled median with a longer time constant, and re-opens the lookahead."""
    bar = mb.BARS['mBookToPrice']['value']
    before = mb.BARS['mBookToPrice']['value']
    vals = [bar - 0.1] * 20                      # 0% pass -- far below BAND_LOW
    cal = mb.calibrate(_panel('mBookToPrice', vals, n_sources=1), _signs(),
                       window_rows=20, n_sources=mb.MIN_FULL_UNIVERSE_SOURCES)
    row = cal[cal['criterion'] == 'mBookToPrice'].iloc[0]
    assert row['pass_rate'] == 0.0 and row['breach'] == 1
    assert mb.BARS['mBookToPrice']['value'] == before, (
        'calibration MUTATED a bar. It may only report.')


def test_hysteresis_no_proposal_until_the_breach_is_the_SECOND_consecutive_run():
    """One bad fetch must not be able to move a bar."""
    bar = mb.BARS['mBookToPrice']['value']
    panel = _panel('mBookToPrice', [bar - 0.1] * 20)
    first = mb.calibrate(panel, _signs(), window_rows=20,
                         n_sources=mb.MIN_FULL_UNIVERSE_SOURCES)
    r1 = first[first['criterion'] == 'mBookToPrice'].iloc[0]
    assert r1['breach'] == 1 and r1['breach_streak'] == 1 and r1['proposed_constant'] is None
    second = mb.calibrate(panel, _signs(), window_rows=20,
                          n_sources=mb.MIN_FULL_UNIVERSE_SOURCES,
                          prior_streaks={'mBookToPrice': 1})
    r2 = second[second['criterion'] == 'mBookToPrice'].iloc[0]
    assert r2['breach_streak'] == 2 and r2['proposed_constant'] is not None
    #  the proposal is the empirical MEDIAN, rounded to the criterion's own quantum
    assert abs(r2['proposed_constant'] - round((bar - 0.1) / 0.05) * 0.05) < 1e-9


def test_a_streak_is_BROKEN_by_a_clean_run():
    bar = mb.BARS['mBookToPrice']['value']
    clean = _panel('mBookToPrice', [bar + 0.1] * 10 + [bar - 0.1] * 10)
    cal = mb.calibrate(clean, _signs(), window_rows=20,
                       n_sources=mb.MIN_FULL_UNIVERSE_SOURCES,
                       prior_streaks={'mBookToPrice': 1})
    row = cal[cal['criterion'] == 'mBookToPrice'].iloc[0]
    assert row['breach'] == 0 and row['breach_streak'] == 0


def test_a_TEST_universe_REPORTS_the_breach_but_can_never_move_a_bar():
    """*** THE GUARD RAIL, AND THE VERDICT/ACTION SPLIT (2026-08-07). ***

    The defect: `breach` used to be FORCED TO 0 whenever `advisory` was 1, so the 2,613-source
    run reported `breach=0` on all seven bars and that was read upward as "all seven bars held"
    -- a test no run under 5,000 sources could fail.  The verdict is now always truthful; only
    the CONSEQUENCE (streak, proposal) is gated.  Both halves are pinned here, because either
    one alone is the bug: a truthful breach that could move a bar, or a guard rail that lies.
    """
    bar = mb.BARS['mBookToPrice']['value']
    cal = mb.calibrate(_panel('mBookToPrice', [bar - 0.1] * 20), _signs(), window_rows=20,
                       n_sources=10)
    row = cal[cal['criterion'] == 'mBookToPrice'].iloc[0]
    assert row['advisory'] == 1
    assert row['pass_rate'] == 0.0 and row['breach'] == 1, (
        'an advisory run suppressed a REAL breach -- `breach` is a statement about the panel '
        'in hand and is true at any n; suppressing it makes the column mean something other '
        'than its label')
    assert row['breach_streak'] == 0 and row['proposed_constant'] is None, (
        'a TEST universe reached the hysteresis ledger -- a thin sample can now move a bar')


def test_an_advisory_run_can_neither_ADVANCE_nor_RESET_a_standing_streak():
    """*** THE INVARIANT THE SPLIT MUST NOT BREAK. ***  Reporting the breach truthfully must not
    give a thin universe a route into the ledger in EITHER direction -- not one more step toward
    a proposal, and not a wipe of a streak a full run recorded."""
    bar = mb.BARS['mBookToPrice']['value']
    #  breaching panel, advisory, with a streak of 1 already standing from a full run
    cal = mb.calibrate(_panel('mBookToPrice', [bar - 0.1] * 20), _signs(), window_rows=20,
                       n_sources=10, prior_streaks={'mBookToPrice': 1},
                       streak_participant=True)
    row = cal[cal['criterion'] == 'mBookToPrice'].iloc[0]
    assert row['breach'] == 1 and row['breach_streak'] == 0 and row['proposed_constant'] is None, (
        'an advisory run chained a standing streak to 2 and proposed a re-set off a thin panel')
    #  ...and the RESET direction: a CLEAN advisory run must not wipe the standing streak
    #  either.  `_prior_streaks` skips advisory reports on read, so this row's 0 is inert.
    clean = mb.calibrate(_panel('mBookToPrice', [bar + 0.1] * 10 + [bar - 0.1] * 10), _signs(),
                         window_rows=20, n_sources=10, prior_streaks={'mBookToPrice': 1},
                         streak_participant=True)
    assert int(clean[clean['criterion'] == 'mBookToPrice'].iloc[0]['advisory']) == 1


def test_an_advisory_report_on_disk_is_invisible_to_the_hysteresis(tmp_path):
    """The end-to-end of the reset half: an advisory report now WRITES `breach=1, streak=0`, so
    the only thing keeping it from resetting a full run's streak is `_prior_streaks` skipping
    advisory files.  Pinned end-to-end rather than by inspection, because the split made this
    file's contents look like a clean run for the first time."""
    bar = mb.BARS['mBookToPrice']['value']
    panel = _panel('mBookToPrice', [bar - 0.1] * 20)
    signs = _signs()
    orig = mb.MIN_FULL_UNIVERSE_SOURCES
    try:
        mb.MIN_FULL_UNIVERSE_SOURCES = 0        # this panel counts as full-universe
        full = mb.emit_calibration(panel, signs, universe='fmp_stock_NA1_EU1', window_rows=20,
                                   directory=str(tmp_path), verbose=False,
                                   streak_participant=True)
        mb.MIN_FULL_UNIVERSE_SOURCES = 10 ** 9  # ...and now everything is advisory
        adv = mb.emit_calibration(panel, signs, universe='TESTUNIVERSE', window_rows=20,
                                  directory=str(tmp_path), verbose=False,
                                  streak_participant=True)
    finally:
        mb.MIN_FULL_UNIVERSE_SOURCES = orig
    assert int(full[full['criterion'] == 'mBookToPrice'].iloc[0]['breach_streak']) == 1
    a = adv[adv['criterion'] == 'mBookToPrice'].iloc[0]
    assert a['advisory'] == 1 and a['breach'] == 1 and a['breach_streak'] == 0
    #  the standing streak of 1 must still be what the NEXT full run reads
    assert mb._prior_streaks(str(tmp_path)).get('mBookToPrice') == 1, (
        'the advisory report reset the standing streak -- a TEST run just erased a full run\'s '
        'evidence')


def test_the_yield_proposal_is_rounded_on_the_ANNUAL_rate():
    """Rounding the per-quarter number to the same quantum would be %gx coarser than
    intended.""" % rp.DEFAULT_ROWS_PER_YEAR
    spec = mb.BARS['mEarningsYield']
    #  an annual median of 4.3% must round to 4.5% (0.5pp quantum) and come back /Q
    q_median = 0.043 / rp.DEFAULT_ROWS_PER_YEAR
    assert abs(mb._round_proposal(spec, q_median)
               - 0.045 / rp.DEFAULT_ROWS_PER_YEAR) < 1e-12


def test_a_second_call_in_the_SAME_run_cannot_manufacture_a_streak(tmp_path):
    """*** `postBoWrapper` IS RE-ENTERED. ***  `baseline_tools/nan_policy_report` calls it twice
    in one process (two arms) and five other offline tools call it on full panels.  If the
    second call read the first's report it would reach `breach_streak = 2` -- a bar-change
    proposal out of ONE panel, defeating the hysteresis entirely."""
    bar = mb.BARS['mBookToPrice']['value']
    panel = _panel('mBookToPrice', [bar - 0.1] * 20)
    signs = _signs()
    orig = mb.MIN_FULL_UNIVERSE_SOURCES
    mb.MIN_FULL_UNIVERSE_SOURCES = 0            # make this thin panel count as full-universe
    try:
        #  `streak_participant=True` because this test IS the production seam being re-entered
        #  (2026-08-06) -- a non-participant is pinned at streak 0 and would pass this test
        #  vacuously, proving nothing about the basename guard it exists to cover.
        a = mb.emit_calibration(panel, signs, universe='U', window_rows=20,
                                directory=str(tmp_path), verbose=False,
                                streak_participant=True)
        b = mb.emit_calibration(panel, signs, universe='U', window_rows=20,
                                directory=str(tmp_path), verbose=False,
                                streak_participant=True)
    finally:
        mb.MIN_FULL_UNIVERSE_SOURCES = orig
    for cal in (a, b):
        row = cal[cal['criterion'] == 'mBookToPrice'].iloc[0]
        assert row['breach'] == 1 and row['breach_streak'] == 1, (
            'a repeated call within one run chained the streak -- one panel can now propose '
            'a bar change')
        assert row['proposed_constant'] is None


def test_a_research_run_under_a_DIFFERENT_universe_label_cannot_seed_a_streak(tmp_path):
    """The hole one door along from the basename guard (2026-08-06).

    `backtest_ols_analysis` builds a `temp_dmdic` with NO `universe` key, so its report is
    `MeanBarCalibration-<today>_unknown.csv` -- a different BASENAME from the day's production
    report, and non-advisory whenever its PIT-sliced panel clears MIN_FULL_UNIVERSE_SOURCES.
    Under the filename rule the production run read it and reached `breach_streak = 2` off a
    research panel: a bar-change proposal out of a panel that never scored anything.

    The research arm here is the DEFAULT call (no `streak_participant`), which is exactly how
    every offline tool reaches this seam -- so the test fails if the default polarity is ever
    flipped, not merely if the flag is mis-plumbed.
    """
    bar = mb.BARS['mBookToPrice']['value']
    panel = _panel('mBookToPrice', [bar - 0.1] * 20)
    signs = _signs()
    orig = mb.MIN_FULL_UNIVERSE_SOURCES
    mb.MIN_FULL_UNIVERSE_SOURCES = 0            # make this thin panel count as full-universe
    try:
        research = mb.emit_calibration(panel, signs, universe='unknown', window_rows=20,
                                       directory=str(tmp_path), verbose=False)
        prod = mb.emit_calibration(panel, signs, universe='fmp_stock_NA1_EU1',
                                   window_rows=20, directory=str(tmp_path), verbose=False,
                                   streak_participant=True)
    finally:
        mb.MIN_FULL_UNIVERSE_SOURCES = orig

    r = research[research['criterion'] == 'mBookToPrice'].iloc[0]
    assert r['breach'] == 1, 'the research report should still RECORD the breach as evidence'
    assert r['streak_participant'] == 0
    assert r['breach_streak'] == 0 and r['proposed_constant'] is None, (
        'a research run seeded the hysteresis ledger')

    p = prod[prod['criterion'] == 'mBookToPrice'].iloc[0]
    assert p['streak_participant'] == 1
    assert p['breach_streak'] == 1, (
        'the production run chained a streak off a research report written the same day under '
        'a different universe label -- the basename guard cannot see it')
    assert p['proposed_constant'] is None


def test_no_calibration_path_can_mutate_a_bar(tmp_path):
    """The property the reviewer verified, re-asserted against the new opt-in: participation
    decides whether a PROPOSAL is reachable, never whether a CONSTANT moves."""
    before = {k: v['value'] for k, v in mb.BARS.items()}
    bar = mb.BARS['mBookToPrice']['value']
    panel = _panel('mBookToPrice', [bar - 0.1] * 20)
    signs = _signs()
    orig = mb.MIN_FULL_UNIVERSE_SOURCES
    mb.MIN_FULL_UNIVERSE_SOURCES = 0
    try:
        for _ in range(mb.BREACH_RUNS_TO_PROPOSE + 2):
            mb.emit_calibration(panel, signs, universe='fmp_stock_NA1_EU1', window_rows=20,
                                directory=str(tmp_path), verbose=False,
                                streak_participant=True)
    finally:
        mb.MIN_FULL_UNIVERSE_SOURCES = orig
    assert {k: v['value'] for k, v in mb.BARS.items()} == before, (
        'a calibration run mutated a constant -- the whole design forbids this')


def test_the_report_carries_the_agreed_columns():
    cal = mb.calibrate(_panel('mBookToPrice', [0.6] * 8), _signs())
    for col in ('criterion', 'constant', 'n_observed', 'n_pass', 'pass_rate', 'breach',
                'proposed_constant'):
        assert col in cal.columns
    assert len(cal) == len(mb.BARS)
