"""Two CEO rulings of 2026-08-05, pinned.

  C-7  the Stage-1 HISTORY BONUS -- concave, saturating at 40 rows, 0.05 there
  B-8  `long_term_debt_change` treats a SIBLING-CONTRADICTED zero as non-disclosure

Both are pinned on SHAPE and on DEGRADATION, not on a panel number: B-8's sibling fields
(`totalDebt` / `shortTermDebt`) are absent from every saved pickle, so the only thing a
saved panel can demonstrate is that the change is INERT there -- which is itself the
property worth a test.
"""
import math

import numpy as np
import pandas as pd
import pytest

import calcScore as cs
import stage2_metrics as sm


# --------------------------------------------------------------------------- #
#  C-7  history bonus                                                          #
# --------------------------------------------------------------------------- #
def _bonus(rows):
    return cs.HISTORY_BONUS_MAX * math.sqrt(
        min(rows, cs.HISTORY_BONUS_SATURATION_ROWS) / cs.HISTORY_BONUS_SATURATION_ROWS)


def test_history_bonus_constants_are_the_ruling():
    assert cs.HISTORY_BONUS_MAX == 0.05
    assert cs.HISTORY_BONUS_SATURATION_ROWS == 40


def test_history_bonus_saturates_at_40_rows():
    assert _bonus(40) == pytest.approx(0.05)
    #  MORE history than saturation buys NOTHING -- 80 is not better than 40
    assert _bonus(80) == pytest.approx(_bonus(40))
    assert _bonus(400) == pytest.approx(_bonus(40))


def test_history_bonus_is_monotone_and_strictly_concave_up_to_saturation():
    xs = list(range(1, 41))
    vals = [_bonus(x) for x in xs]
    assert all(b > a for a, b in zip(vals, vals[1:])), 'must be strictly increasing'
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    assert all(d2 < d1 for d1, d2 in zip(diffs, diffs[1:])), 'increments must shrink'


def test_history_bonus_shape_points_the_ruling_named():
    #  "40 gives slightly more than 30, and the 30->40 increment is smaller than 20->30"
    assert _bonus(40) > _bonus(30)
    assert (_bonus(40) - _bonus(30)) < (_bonus(30) - _bonus(20))


def test_history_bonus_can_never_outweigh_one_tier_D_criterion():
    """THE SCALE GUARANTEE, and the reason the magnitude is 0.05.

    The smallest possible Stage-1 criterion difference is 0.1 (a Tier-D criterion passing
    over a full window vs failing it).  The bonus spans at most 0.05 across the ENTIRE
    range of history lengths, so it can break a tie and flip a genuinely close call but
    can never overturn one criterion fully differing."""
    smallest_criterion_difference = 0.1
    assert _bonus(40) - _bonus(0) <= smallest_criterion_difference / 2 + 1e-12


def test_history_bonus_is_actually_applied_by_stage1_and_is_the_only_difference():
    """End-to-end on a synthetic panel: two sources identical on every criterion, differing
    ONLY in panel length, must differ by exactly the bonus difference."""
    dict_base, dict_mean, dict_diff, dict_unity, dict_special = \
        cs.cdic.getBaseMeanDiffUnitySpecialDicts()
    cols = (list(dict_base) + ['m' + k[0].upper() + k[1:] for k in dict_mean]
            + ['d' + k[0].upper() + k[1:] for k in dict_diff]
            + ['u' + k[0].upper() + k[1:] for k in dict_unity] + list(dict_special))

    short_rows, long_rows = 8, 40
    frames = []
    for src, nrows in (('SHORT', short_rows), ('LONG', long_rows)):
        f = pd.DataFrame({c: np.ones(nrows) for c in cols})
        f['source'] = src
        f['date'] = pd.date_range('2016-03-31', periods=nrows, freq='QE')[::-1]
        frames.append(f)
    bm_df = pd.concat(frames, ignore_index=True)
    bm_ave = {c: 1.0 for c in cols}

    out = cs.simpleScore_fromDict(bm_df, bm_ave, None, n=8)
    got = dict(zip(out['source'], out['score']))
    assert got['LONG'] - got['SHORT'] == pytest.approx(_bonus(long_rows) - _bonus(short_rows))
    #  and the bonus uses ROWS AVAILABLE, not the head(8) window -- both scored 8 rows
    assert got['LONG'] > got['SHORT']


# --------------------------------------------------------------------------- #
#  B-8  non-disclosed long-term debt                                           #
# --------------------------------------------------------------------------- #
def _row(**kw):
    return pd.Series(kw)


def test_b8_zero_ltd_with_positive_totalDebt_is_non_disclosure():
    assert sm._long_term_debt_undisclosed(_row(longTermDebt=0.0, totalDebt=500.0)) is True


def test_b8_zero_ltd_with_positive_shortTermDebt_is_non_disclosure():
    assert sm._long_term_debt_undisclosed(_row(longTermDebt=0.0, shortTermDebt=12.0)) is True


def test_b8_all_debt_zero_is_a_genuinely_unlevered_balance_sheet():
    """ABSENCE IS NOT A PASS, but a DISCLOSED ZERO still is a real observation."""
    assert sm._long_term_debt_undisclosed(
        _row(longTermDebt=0.0, totalDebt=0.0, shortTermDebt=0.0)) is False


def test_b8_degrades_to_todays_behaviour_when_the_siblings_are_absent():
    """THE CASE THAT DESCRIBES EVERY SAVED PICKLE.  `totalDebt` / `shortTermDebt` were
    captured on 2026-08-05 and a saved pickle can never gain a column, so on existing data
    the discriminator must be unable to fire."""
    assert sm._long_term_debt_undisclosed(_row(longTermDebt=0.0)) is False


def test_b8_nonzero_ltd_is_never_non_disclosure():
    assert sm._long_term_debt_undisclosed(_row(longTermDebt=1.0, totalDebt=500.0)) is False


def _panel(rows):
    return pd.DataFrame(rows)


def test_long_term_debt_change_returns_nan_on_a_contradicted_zero():
    cdx = _panel([
        dict(totalAssets=1000.0, longTermDebt=0.0, totalDebt=300.0),
        dict(totalAssets=900.0, longTermDebt=50.0, totalDebt=300.0),
        dict(totalAssets=900.0, longTermDebt=50.0, totalDebt=300.0),
        dict(totalAssets=900.0, longTermDebt=50.0, totalDebt=300.0),
        dict(totalAssets=900.0, longTermDebt=50.0, totalDebt=300.0),
    ])
    assert np.isnan(sm.long_term_debt_change(cdx, rpy=4))


def test_long_term_debt_change_unchanged_when_siblings_absent():
    """Same panel MINUS the sibling column -> the pre-B-8 answer, exactly."""
    rows = [
        dict(totalAssets=1000.0, longTermDebt=0.0),
        dict(totalAssets=900.0, longTermDebt=50.0),
        dict(totalAssets=900.0, longTermDebt=50.0),
        dict(totalAssets=900.0, longTermDebt=50.0),
        dict(totalAssets=900.0, longTermDebt=50.0),
    ]
    got = sm.long_term_debt_change(_panel(rows), rpy=4)
    assert got == pytest.approx(0.0 / 1000.0 - 50.0 / 900.0)


def test_long_term_debt_change_still_returns_zero_for_a_genuinely_unlevered_name():
    cdx = _panel([dict(totalAssets=1000.0, longTermDebt=0.0, totalDebt=0.0,
                       shortTermDebt=0.0)] * 5)
    assert sm.long_term_debt_change(cdx, rpy=4) == pytest.approx(0.0)


# =========================================================================== #
#  PIOTROSKI p5 -- THE SAME DISCRIMINATOR, APPLIED (CEO ruling 2026-08-06)       #
#                                                                               #
#  The earlier reading was that touching p5 would be REDESIGNING Piotroski, which  #
#  the D-9 ruling forbids.  THE CEO REVERSED IT: this is CONFORMANCE.  p5 asks       #
#  "did long-term debt fall".  On a row whose zero is CONTRADICTED by a sibling debt  #
#  field, `0 < 0` does not ANSWER that question -- it fails it by default, on 476     #
#  sources (6.17%).  Letting the metric say UNAVAILABLE instead does not change what   #
#  Piotroski asks; it stops the metric answering a question the data cannot support.   #
# =========================================================================== #

def _p_rows(ltd_now, ltd_then, **extra):
    """A 5-row newest-first panel that scores a DEFINED Piotroski composite, so the only
    thing under test is p5's zero.  Every other component's inputs are held constant
    across the two rows, which pins p1..p9 to values that do not move between cases."""
    base = dict(totalAssets=1000.0, netIncome=100.0,
                netCashProvidedByOperatingActivities=150.0, currentRatio=2.0,
                weightedAverageShsOut=50.0, grossProfitMargin=0.4, revenue=800.0)
    curr = dict(base, longTermDebt=ltd_now, **extra)
    prev = dict(base, longTermDebt=ltd_then, **extra)
    return pd.DataFrame([curr, prev, prev, prev, prev])


def test_piotroski_nans_on_a_CONTRADICTED_zero_long_term_debt():
    """*** THE FIX.  longTermDebt == 0 while totalDebt > 0: the entity is levered
    somewhere, so its zero is non-disclosure and p5 is UNANSWERABLE -> the composite is
    NaN, which takes the column-median treatment and neither credits nor docks. ***"""
    cdx = _p_rows(0.0, 0.0, totalDebt=300.0)
    assert np.isnan(sm.piotroski(cdx, rpy=4))


def test_piotroski_nans_when_only_the_PRIOR_row_is_contradicted():
    """p5 reads BOTH periods, so either row's contradicted zero makes the comparison
    unsupportable -- same as `long_term_debt_change`, which checks both."""
    curr = dict(totalAssets=1000.0, netIncome=100.0,
                netCashProvidedByOperatingActivities=150.0, currentRatio=2.0,
                weightedAverageShsOut=50.0, grossProfitMargin=0.4, revenue=800.0,
                longTermDebt=40.0, totalDebt=300.0)
    prev = dict(curr, longTermDebt=0.0)
    assert np.isnan(sm.piotroski(pd.DataFrame([curr, prev, prev, prev, prev]), rpy=4))


def test_piotroski_shortTermDebt_is_the_discriminator_too_not_just_totalDebt():
    """The brief named `shortTermDebt` specifically. Both siblings must reach p5, because
    `_long_term_debt_undisclosed` reads `_B8_DEBT_SIBLINGS`, not one field."""
    assert 'shortTermDebt' in sm._B8_DEBT_SIBLINGS
    assert np.isnan(sm.piotroski(_p_rows(0.0, 0.0, shortTermDebt=12.0), rpy=4))


def test_piotroski_is_BYTE_IDENTICAL_when_the_siblings_are_ABSENT():
    """*** THE CASE THAT DESCRIBES EVERY SAVED PICKLE, and the reason this change is
    unexercisable on saved data.  `totalDebt` / `shortTermDebt` were captured on
    2026-08-05 and a saved pickle can never gain a column, so on existing panels the
    discriminator CANNOT fire and p5 must fail `0 < 0` exactly as it did before. ***"""
    cdx = _p_rows(0.0, 0.0)
    got = sm.piotroski(cdx, rpy=4)
    assert not np.isnan(got), 'the no-sibling path must still produce a composite'
    #  p5 fails (0 < 0 is False) and p3/p6/p8/p9 fail on unchanged inputs; p1, p2, p4 and
    #  p7 pass. The VALUE is pinned so a later edit cannot quietly change the no-data path.
    assert got == 4


def test_piotroski_a_DISCLOSED_zero_still_fails_p5_and_that_is_correct():
    """A genuinely unlevered balance sheet reports 0 on every debt field. Its leverage did
    NOT fall, so p5 SHOULD fail -- and the composite must stay computable. This is the
    boundary that makes the fix conformance rather than a free pass: it is why the 476
    affected sources are an UPPER bound on what the discriminator will reach."""
    cdx = _p_rows(0.0, 0.0, totalDebt=0.0, shortTermDebt=0.0)
    got = sm.piotroski(cdx, rpy=4)
    assert not np.isnan(got)
    assert got == 4


def test_piotroski_p5_still_PASSES_on_genuine_deleveraging_with_siblings_present():
    """The discriminator must not fire on a NON-zero longTermDebt, so a real
    deleveraging still scores its point with the sibling columns on the frame."""
    cdx = _p_rows(10.0, 90.0, totalDebt=300.0)
    got = sm.piotroski(cdx, rpy=4)
    assert not np.isnan(got)
    assert got == 5, 'p5 did not score on a genuine fall in the leverage ratio'


def test_piotroski_and_long_term_debt_change_use_the_SAME_discriminator():
    """Not "two rules that agree" -- ONE rule, called twice. If they ever diverge, the
    composite and the extracted component would disagree about whether the same row's zero
    is disclosed, which is the drift `_B8_DEBT_SIBLINGS` was named once to prevent."""
    cdx = _p_rows(0.0, 0.0, totalDebt=300.0)
    assert np.isnan(sm.piotroski(cdx, rpy=4))
    assert np.isnan(sm.long_term_debt_change(cdx, rpy=4))
    clean = _p_rows(0.0, 0.0, totalDebt=0.0, shortTermDebt=0.0)
    assert not np.isnan(sm.piotroski(clean, rpy=4))
    assert sm.long_term_debt_change(clean, rpy=4) == pytest.approx(0.0)
