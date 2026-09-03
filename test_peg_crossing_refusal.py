"""PEG CROSSING REFUSAL (CEO ruling, 2026-09-03).

A loss-to-profit crossing produces an undefined growth rate (the denominator crosses zero and the
artifact dominates), so PEG's crossing rows now produce NaN and are refused at Stage-1 scoring,
exactly like every other undefined criterion.

This test verifies that the substitution rule is DISABLED and crossing rows are REJECTED.
Before the ruling, crossing rows were filled with the pool-median growth rate (nerfed but not
refused). After the ruling, they produce NaN and flow to the existing FAIL path.

Run: pytest test_peg_crossing_refusal.py -v
"""

import numpy as np
import pandas as pd
import pytest

import calcMetrics as cm
import calcScore as cs
import createDicts as cdic
import reporting_period as rp


_ROWS = 16  # same as _PEG_ROWS in test_sign_conventions.py


def _peg_local_crossing(now, prev):
    """Shorthand: call peg_local on a loss-to-profit crossing."""
    eps = [now] * 4 + [prev] * (_ROWS - 4)
    df = pd.DataFrame({'netIncomePerShare': eps, 'price': [10.0] * len(eps)})
    return cm.peg_local(df, rpy=4, crossing_growth=None)[0].iloc[0]


def _peg_local_crossing_with_growth(now, prev, growth):
    """Call peg_local with a crossing_growth value."""
    eps = [now] * 4 + [prev] * (_ROWS - 4)
    df = pd.DataFrame({'netIncomePerShare': eps, 'price': [10.0] * len(eps)})
    return cm.peg_local(df, rpy=4, crossing_growth=growth)[0].iloc[0]


def test_peg_crossing_produces_nan_without_substitution():
    """A loss-to-profit crossing produces NaN, even if crossing_growth is ignored.

    This is the CORE of the ruling: crossing rows produce NaN at build time and stay NaN
    (substitute_peg_crossing no longer fills them). The crossing_growth parameter is accepted
    for backward compatibility but is now ignored.
    """
    # A turnaround: positive now, negative before
    crossing_nan = _peg_local_crossing(1.0, -0.8)
    assert not np.isfinite(crossing_nan), (
        'loss-to-profit crossing must produce NaN (CEO ruling 2026-09-03): the growth rate is '
        'undefined when the base crosses zero')

    # Even if crossing_growth is supplied, it is IGNORED under the new ruling
    crossing_still_nan = _peg_local_crossing_with_growth(1.0, -0.8, 25.0)
    assert not np.isfinite(crossing_still_nan), (
        'crossing rows must produce NaN even when crossing_growth is supplied -- the parameter '
        'is accepted for backward compatibility but is now ignored')


def test_peg_crossing_row_ignores_crossing_growth_parameter():
    """Crossing rows produce NaN even when crossing_growth parameter is supplied.

    The ruling explicitly states: "Refuse it — score as fail like every other undefined criterion."
    The crossing_growth parameter is now ignored (CEO ruling 2026-09-03). This test calls peg_local
    with crossing_growth supplied and verifies it is ignored.

    On HEAD (unfixed), crossing_growth=25.0 would make the crossing row compute to a finite value
    (0.20). On fixed code, it is ignored and the row is NaN.
    """
    # Call peg_local directly with a crossing row (loss->profit)
    eps = [0.5] * 4 + [-0.5] * (_ROWS - 4)     # newest trailing year +, prior year -
    df = pd.DataFrame({
        'netIncomePerShare': eps,
        'price': [10.0] * len(eps)
    })

    # Compute PEG with crossing_growth supplied (the substitution value)
    crossing_growth_value = 25.0  # Arbitrary pool median growth rate
    peg, eps_ttm, eps_prev = cm.peg_local(df, rpy=4, crossing_growth=crossing_growth_value)

    # The newest row (index 0) is a crossing row: eps_ttm > 0, eps_prev < 0
    assert eps_ttm.iloc[0] > 0, 'current eps must be positive'
    assert eps_prev.iloc[0] < 0, 'prior eps must be negative (loss-to-profit crossing)'

    # FIXED CODE: The crossing row must produce NaN (crossing_growth parameter is ignored)
    peg_val = peg.iloc[0]
    assert np.isnan(peg_val), (
        'crossing row (loss->profit) must produce NaN even when crossing_growth is supplied; '
        'the parameter is now ignored (CEO ruling 2026-09-03)')

    # When scored with calcByTier, NaN fails
    base, mean, diff, unity, special = cdic.getBaseMeanDiffUnitySpecialDicts()
    spec = special['PEG']

    score = cs.calcByTier('special', spec['Tier'], spec['Sign'],
                         pd.Series([peg_val]), 0.0, 'PEG', 8)
    assert score == 0.0, (
        'a crossing row with NaN PEG must produce zero score -- NaN is treated as a fail')


def test_substitute_peg_crossing_is_now_noop_with_in_domain_rows():
    """substitute_peg_crossing is now a no-op even when there are in-domain crossing rows
    and an available pool median (CEO ruling, 2026-09-03).

    Previously (HEAD) it would fill crossing rows with the pool median. Now (fixed) it returns
    the frame unchanged. This test uses a fixture with BOTH a crossing source AND a grower
    source so the pool median would exist (HEAD function would fill; fixed function does not).

    On HEAD, the function RUNS and FILLS crossing rows (n_filled >= 1), causing the test to
    FAIL on the assertion that crossing row must be NaN.
    On fixed code, the function is a no-op and the test PASSES.
    """
    def _dates(n, months=3, end='2026-03-31'):
        """n period-end dates, NEWEST FIRST."""
        return [pd.Timestamp(end) - pd.DateOffset(months=months * i) for i in range(n)]

    # Two sources: one with crossing, one with growth (to provide pool median)
    n = 12  # 12 periods, 3 years of quarterly data
    dates = _dates(n)

    # Crossing source: loss->profit (eps_t > 0, eps_{t-4} < 0)
    eps_turn = [0.5] * 4 + [-0.5] * 8   # 4 quarters of +0.5, then 8 quarters of -0.5

    # Grower source: steady growth
    eps_grow = [1.0 * (0.95 ** i) for i in range(n)]

    turn_df = pd.DataFrame({
        'netIncomePerShare': eps_turn,
        'price': [10.0] * n,
        'date': dates,
        'source': ['TURN'] * n,
    })
    grow_df = pd.DataFrame({
        'netIncomePerShare': eps_grow,
        'price': [10.0] * n,
        'date': dates,
        'source': ['GROW'] * n,
    })

    # Compute PEG columns using peg_local (fixed code produces NaN for crossings)
    peg_turn_full, _, _ = cm.peg_local(turn_df, rpy=4)
    peg_turn_crit = cm.peg_criterion(peg_turn_full)

    peg_grow_full, _, _ = cm.peg_local(grow_df, rpy=4)
    peg_grow_crit = cm.peg_criterion(peg_grow_full)

    # Build BoMetric frame with proper columns (source, date required)
    bm = pd.DataFrame({
        'PEG': list(peg_turn_crit) + list(peg_grow_crit),
        'source': ['TURN'] * n + ['GROW'] * n,
        'date': dates + dates
    })

    # cdx has both sources, so pool median IS computable
    cdx = pd.concat([turn_df, grow_df], ignore_index=True)
    cdx[rp.FREQ_COLUMN] = rp.QUARTERLY

    # The crossing row (TURN, index 0) starts as NaN
    assert np.isnan(bm['PEG'].iloc[0]), (
        'crossing row starts as NaN from peg_local')

    # Call substitute_peg_crossing
    bm_before = bm.copy()
    out, stats = cm.substitute_peg_crossing(bm_before, cdx, verbose=False)

    # FIXED CODE: Frame is unchanged (no-op)
    # HEAD CODE: Function would fill crossing rows with pool median
    assert out['PEG'].equals(bm['PEG']), (
        'fixed code: frame unchanged (no-op); HEAD would have filled crossing rows')

    # The crossing row must STILL be NaN (not filled by the function)
    # On HEAD, this would FAIL: the function would have filled it with pool median (~0.20)
    assert np.isnan(out['PEG'].iloc[0]), (
        'crossing row must remain NaN after calling substitute_peg_crossing')


def test_crossing_row_in_realistic_panel_is_refused():
    """End-to-end: a crossing row in a realistic frame is built as NaN and fails scoring.

    This is the mutation test: it fails if the substitution is active (the old code would fill
    the crossing row with the pool median), and passes with the new code (crossing rows are NaN).
    """
    # Two sources: one crossing, one growing (for the pool median, if substitution were active)
    eps_turn = [0.5] * 4 + [-0.5] * (_ROWS - 4)
    eps_grow = [0.5 * (0.95 ** i) for i in range(_ROWS)]

    turn_fund = pd.DataFrame({
        'netIncomePerShare': eps_turn,
        'price': [10.0] * _ROWS,
        'date': [pd.Timestamp('2026-03-31') - pd.DateOffset(months=3 * i) for i in range(_ROWS)],
        'source': ['TURN'] * _ROWS,
        'totalStockholdersEquity': [500.0] * _ROWS,
        'marketCap': [1000.0] * _ROWS,
    })
    grow_fund = pd.DataFrame({
        'netIncomePerShare': eps_grow,
        'price': [10.0] * _ROWS,
        'date': [pd.Timestamp('2026-03-31') - pd.DateOffset(months=3 * i) for i in range(_ROWS)],
        'source': ['GROW'] * _ROWS,
        'totalStockholdersEquity': [500.0] * _ROWS,
        'marketCap': [1000.0] * _ROWS,
    })

    # Build PEG columns (on the full frame, not incrementally, so we have full history)
    peg_turn_full, _, _ = cm.peg_local(turn_fund, rpy=4)
    peg_grow_full, _, _ = cm.peg_local(grow_fund, rpy=4)

    # The crossing row (index 0 of TURN frame, newest) must be NaN
    assert np.isnan(peg_turn_full.iloc[0]), (
        'the crossing row must be NaN at build time (no growth rate is defined)')

    # The growing row (index 0 of GROW frame, newest) must be finite
    assert np.isfinite(peg_grow_full.iloc[0]), (
        'the grower must have a finite PEG at build time')

    # After calling substitute_peg_crossing, the crossing row must STILL be NaN
    bm_turn_peg_crit = cm.peg_criterion(peg_turn_full)
    bm_grow_peg_crit = cm.peg_criterion(peg_grow_full)
    bm = pd.DataFrame({
        'PEG': list(bm_turn_peg_crit) + list(bm_grow_peg_crit),
        'source': ['TURN'] * len(bm_turn_peg_crit) + ['GROW'] * len(bm_grow_peg_crit)
    })
    cdx = pd.concat([turn_fund, grow_fund], ignore_index=True)
    cdx[rp.FREQ_COLUMN] = rp.QUARTERLY

    out, stats = cm.substitute_peg_crossing(bm.copy(), cdx, verbose=False)

    # The crossing row is still NaN (substitute_peg_crossing did not fill it)
    assert np.isnan(out['PEG'].iloc[0]), (
        'after substitute_peg_crossing, the crossing row must STILL be NaN (the function is now '
        'a no-op)')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
