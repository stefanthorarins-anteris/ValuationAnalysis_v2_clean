"""test_metric_share_cap.py  --  the single-metric share cap and the thesis transfer.

WHAT THESE TESTS ARE WRITTEN AGAINST, because this project has now filed FIFTEEN instances
of a test that pins the very defect it covers.  The rule applied here:

  * the PROPERTIES are asserted as properties, in a form that survives a change of cap
    level, a change of delta, or a change of metric -- `test_post_condition_*`,
    `test_transfer_preserves_sum_abs`, `test_only_two_metrics_move`.  These must not be
    edited when a number moves.
  * the two DELIBERATE VALUE PINS (`CAP == 0.25`, `THESIS_TRANSFER_DELTA == 0.006`) are
    isolated in `test_the_two_ceo_numbers` and are labelled as CEO decisions.
  * `test_mutation_*` verify the guards actually fire: a cap replaced by a no-op, and the
    one-pass cap that looks right and is not, must both FAIL a test here.  A guard nobody
    has watched fail is a guard nobody knows works.

WHERE THE CEO'S TWO NUMBERS REALLY REACH -- CORRECTED 2026-08-31 AFTER REVIEW
-----------------------------------------------------------------------------
An earlier version of this docstring said "nothing else in this file encodes 0.25 or 0.006"
and, worse, that moving one is "a one-line edit here plus the note in the module it lives
in, and nothing else in the suite has to be touched".  Both were false, and the second was
the dangerous one -- it is an instruction to a future editor that would have left four
assertions failing in a file they were told not to look at.

  * `0.25` appears throughout this file as an explicit `cap=` argument.  That is harmless
    (each test states the level it is testing), but it is not "nowhere else".
  * `THESIS_TRANSFER_DELTA` moves THREE pinned numbers in `test_e2_weight_vector.py` --
    `1.1532` (`:260`, ey/iq on the SHIPPED vector), `1.3365` (`:267`, ey/|CycleHeat|), and
    the post-D3 margin `-0.008830` (`:318`).  Four assertions in total fail if the delta
    moves, across two files: those three plus `test_the_two_ceo_numbers` here.  (The `1.2928`
    pinned two lines above `1.1532` is on `DEPLOYED_DERIVED`, the PRE-transfer vector, and
    does NOT move -- which is the point of asserting both.)  The margin at `:318` moves at
    TWICE the delta, because the transfer moves `earnYield` down and `incomeQuality` up.
    Verified numerically at delta = 0.003 / 0.006 / 0.008: only 0.006 leaves all four green.
    `test_the_delta_is_what_moves_the_e2_pins` below re-derives the dependency from `sw` so
    the claim cannot go stale silently.

So: MOVING THE CAP is a one-line edit in `test_the_two_ceo_numbers` plus the note in
`metric_share_cap`.  MOVING THE DELTA is that, plus three numbers in `test_e2_weight_vector`
and the B.5b/B.6 notes in `scoringWeights`.
"""
import ast
import os

import numpy as np
import pandas as pd
import pytest

import metric_share_cap as msc
import scoringWeights as sw

_REPO = os.path.dirname(os.path.abspath(__file__))


def _frame(rows):
    return pd.DataFrame(rows, columns=['a', 'b', 'c', 'd', 'e'], dtype='float64')


#  --- THE POST-CONDITION: the property the cap exists to establish -------------------- #

@pytest.mark.parametrize('cap', [0.5, 0.4, 0.3, 0.25, 0.2, 0.15])
def test_post_condition_holds_at_every_cap_level(cap):
    """After the cap, NO metric exceeds `cap` of its name's absolute total -- for any name
    and any cap level.  This is the whole claim; if it fails the module is decoration."""
    rng = np.random.default_rng(11)
    df = pd.DataFrame(rng.normal(size=(400, 9)) * rng.uniform(0.01, 0.4, size=9),
                      columns=list('abcdefghi'))
    capped, report = msc.apply_share_cap(df, cap=cap)
    A = capped.abs()
    share = A.div(A.sum(axis=1), axis=0)
    assert share.to_numpy().max() <= cap + 1e-9
    #  and the report must agree with a share recomputed from the returned frame, not with
    #  its own internal bookkeeping
    assert np.allclose(report['share_after'].to_numpy(),
                       share.max(axis=1).to_numpy(), atol=1e-12)


def test_capped_metrics_land_EXACTLY_on_the_cap_not_merely_under_it():
    """The fixed point, stated as an equality.  A one-pass cut lands UNDER the pre-cap base
    and OVER the post-cap one; only the fixed point puts the capped metric exactly at `cap`
    of the total that actually ships."""
    #  a well-spread tail, which is what makes k = 1 the consistent answer (see the cascade
    #  test below for what a THIN tail does instead)
    df = _frame([[0.4, 0.16, 0.16, 0.16, 0.12]])
    capped, report = msc.apply_share_cap(df, cap=0.25)
    A = capped.abs().iloc[0]
    assert report['n_capped'].iloc[0] == 1
    assert A['a'] / A.sum() == pytest.approx(0.25, abs=1e-12)
    #  and only the offending metric moved
    assert (capped.iloc[0][['b', 'c', 'd', 'e']] == df.iloc[0][['b', 'c', 'd', 'e']]).all()


def test_the_SHIPPED_DEFAULT_cap_is_the_one_apply_share_cap_actually_applies():
    """`msc.CAP` ITSELF, exercised.  Every other test in this file passes `cap=` explicitly,
    so until this one existed the shipped default had no functional test at all: a typo in
    the module constant would have left the whole suite green and changed the live ranking.

    Deliberately called with NO `cap=` argument.  The row binds at 0.25 and does not bind at
    0.40, so the assertion is sensitive to the constant in both directions."""
    df = _frame([[0.4, 0.16, 0.16, 0.16, 0.12]])
    capped, report = msc.apply_share_cap(df)             # NO cap= -- that is the point
    assert report['n_capped'].iloc[0] == 1, \
        'the shipped default did not bind on a row whose top metric is 40% of the base'
    A = capped.abs().iloc[0]
    assert A['a'] / A.sum() == pytest.approx(msc.CAP, abs=1e-12)
    assert A['a'] / A.sum() == pytest.approx(0.25, abs=1e-12)
    #  and the same row is untouched at a level above its top share, so this cannot pass by
    #  capping everything
    untouched, rep2 = msc.apply_share_cap(df, cap=0.45)
    assert rep2['n_capped'].iloc[0] == 0


def test_two_metrics_can_bind_at_once_and_both_land_on_the_cap():
    """k > 1 is reachable and handled: two dominant metrics both truncate to the same value
    and both sit exactly at the cap afterwards."""
    df = _frame([[0.45, 0.45, 0.05, 0.03, 0.02]])
    capped, report = msc.apply_share_cap(df, cap=0.25)
    A = capped.abs().iloc[0]
    assert report['n_capped'].iloc[0] == 2
    assert A['a'] / A.sum() == pytest.approx(0.25, abs=1e-12)
    assert A['b'] / A.sum() == pytest.approx(0.25, abs=1e-12)


def test_a_thin_tail_CASCADES_and_that_is_the_fixed_point_not_a_defect():
    """The behaviour that looks like a bug and is not, pinned so nobody "fixes" it.

    [0.9, 0.05, 0.03, 0.01, 0.01] does NOT resolve at k = 1.  Truncating `a` to 25% of the
    post-cap total shrinks that total so far that `b` and `c` then breach the cap themselves,
    and the consistent solution is k = 3.  The name's score collapses from 1.00 to 0.08 --
    which is the honest reading: a name that is 90% one metric carries almost no other
    information, so once that metric is refused the right to decide it, there is almost
    nothing left to score.

    NOT reached on the 2026-08-31 panel (all 13 bound names there resolve at k = 1, max
    share 0.4572) -- but that is a fact about one panel, not a property.  An independent
    review simulated the shipped weight vector and the real squash at the live 20% imputed
    rate and got k >= 2 on ~1.77% of names, i.e. about 1.7 per 97-name panel, worst case
    erasing 62% of a name's mass.  So this is a live shape, not a curiosity, and the log
    prints `n_capped` because of it (see `test_the_log_PRINTS_n_capped_...`)."""
    df = _frame([[0.9, 0.05, 0.03, 0.01, 0.01]])
    capped, report = msc.apply_share_cap(df, cap=0.25)
    assert report['n_capped'].iloc[0] == 3
    A = capped.abs().iloc[0]
    assert A.sum() == pytest.approx(0.08, abs=1e-12)
    assert report['share_after'].iloc[0] == pytest.approx(0.25, abs=1e-12)


def test_sign_is_preserved_and_a_negative_dominant_metric_is_also_truncated():
    """The cap is stated on |contribution|, so it is SYMMETRIC: a large negative dominant
    contribution is truncated too, which RAISES that name.  That is the shipped ruling (see
    the module docstring); the test exists so the behaviour cannot change silently."""
    df = _frame([[-0.9, 0.05, 0.03, 0.01, 0.01]])
    capped, report = msc.apply_share_cap(df, cap=0.25)
    assert capped.iloc[0]['a'] < 0                       # sign kept
    assert report['agg_delta'].iloc[0] > 0               # the name's score ROSE
    assert report['contrib_before'].iloc[0] < 0          # ...and the log's reason for it
    A = capped.abs().iloc[0]
    assert A['a'] / A.sum() == pytest.approx(0.25, abs=1e-12)


def test_untouched_when_nothing_exceeds_the_cap():
    df = _frame([[0.2, 0.2, 0.2, 0.2, 0.2]])
    capped, report = msc.apply_share_cap(df, cap=0.25)
    pd.testing.assert_frame_equal(capped, df)
    assert report['n_capped'].iloc[0] == 0
    assert report['agg_delta'].iloc[0] == 0.0
    assert report['status'].iloc[0] == msc.STATUS_OK


def test_all_zero_row_is_skipped_not_divided_by_zero():
    """A name with no contribution at all has no total to be a share of.  It must survive
    unchanged rather than raise or produce NaN in the returned frame."""
    df = _frame([[0.0] * 5, [0.9, 0.05, 0.03, 0.01, 0.01]])
    capped, report = msc.apply_share_cap(df, cap=0.25)
    assert (capped.iloc[0] == 0.0).all()
    assert report['n_capped'].iloc[0] == 0
    assert np.isfinite(capped.to_numpy()).all()
    assert report['status'].iloc[0] == msc.STATUS_ALL_ZERO


def test_the_absolute_base_is_never_negative_so_no_small_total_special_case_is_needed():
    """The reason the module uses |c|/sum|c| and not c/AggScore.  Build a row whose SIGNED
    total is ~0 while its absolute total is large: the share must stay well-defined and in
    [0, 1], where a signed-total share would diverge."""
    df = _frame([[0.5, -0.5, 0.3, -0.3, 0.0]])
    signed_total = df.iloc[0].sum()
    assert abs(signed_total) < 1e-12                     # the pathological case for c/AggScore
    capped, report = msc.apply_share_cap(df, cap=0.25)
    assert 0.0 <= report['share_before'].iloc[0] <= 1.0
    assert report['share_after'].iloc[0] <= 0.25 + 1e-9


def test_cap_never_increases_a_metrics_absolute_contribution():
    """Monotonicity in the only direction that matters: a cap TRUNCATES. If any |c| grows,
    the fixed-point search has selected an inconsistent k."""
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(size=(300, 7)), columns=list('abcdefg'))
    capped, _ = msc.apply_share_cap(df, cap=0.25)
    assert (capped.abs().to_numpy() <= df.abs().to_numpy() + 1e-12).all()


#  --- THE ROWS THE CAP CANNOT HONOUR --------------------------------------------------- #
#
#  THE DEFECT THESE PIN (review 1, 2026-08-31).  A row the cap cannot satisfy had the whole
#  row zeroed -- `share_after` reported as the sentinel `-1.0`, and `-1.0 > 0.25` is False,
#  so `_assert_post_condition` passed it.  A name whose AggScore was strongly NEGATIVE came
#  out at exactly 0.0, which on a pool where 37 of 97 names score at or below zero is a large
#  PROMOTION invented by the guard.
#
#  AND WHAT THE FIRST FIX GOT WRONG (review 2, same day).  The first fix made those rows
#  RAISE.  That is worse, not better: `postBo.py:697` does not exception-guard the
#  general-pool call, so the run dies with no top-20 at all, and the condition is
#  deterministic on a panel so a re-run hits it again.  It also FIRED on two saved panels
#  from the current universe.  The behaviour pinned below is ship-UNCAPPED-and-SAY-SO: the
#  name keeps the score it would have had if this module did not exist, and the log names it.

@pytest.mark.parametrize('row,n_nonzero', [
    ([0.30, 0.10, 0.0, 0.0, 0.0], 2),
    ([0.30, 0.10, 0.05, 0.0, 0.0], 3),
    ([0.30, 0.0, 0.0, 0.0, 0.0], 1),
    ([-0.30, -0.10, 0.0, 0.0, 0.0], 2),
])
def test_a_row_the_cap_would_ANNIHILATE_ships_UNCAPPED_and_is_NOT_zeroed(row, n_nonzero):
    """The four rows the pre-fix cap turned into all-zeros. Each must now come back EXACTLY
    as it went in -- not zeroed, not ejected, and not raising."""
    assert n_nonzero <= msc._k_max(0.25)                 # the row really is infeasible
    df = _frame([row])
    capped, report = msc.apply_share_cap(df, cap=0.25, sources=['SPARSE'])
    pd.testing.assert_frame_equal(capped, df)            # byte-for-byte unchanged
    assert report['status'].iloc[0] == msc.STATUS_INFEASIBLE
    assert report['n_capped'].iloc[0] == 0
    assert report['agg_delta'].iloc[0] == 0.0, 'an uncappable row must cost the name nothing'
    assert 'NOT CAPPED and shipped with their UNCAPPED score' in \
        msc.format_report(report, 'general', cap=0.25)


def test_the_negative_sparse_row_is_NOT_promoted_to_zero():
    """The S1 harm, stated as the number that moves. `[-0.30, -0.10, 0, 0, 0]` sums to -0.40.
    The pre-fix cap made it exactly 0.0 -- a jump over every name scoring below zero. Its
    AggScore must be untouched."""
    df = _frame([[-0.30, -0.10, 0.0, 0.0, 0.0]])
    capped, report = msc.apply_share_cap(df, cap=0.25)
    assert float(capped.iloc[0].sum()) == pytest.approx(-0.40, abs=1e-15)
    assert report['agg_delta'].iloc[0] == 0.0


def test_an_uncappable_name_does_NOT_stop_the_pool_and_the_others_are_STILL_capped():
    """THE SHIP BLOCKER FROM REVIEW 2, pinned. One sparse name used to raise, and because
    `postBo.py:697` is not exception-guarded that killed the whole run -- no postRank, no
    AggScoreTop CSV, no top-20. The pool must complete, the normal name must still be capped,
    and the sparse one must be named in the log."""
    df = _frame([[0.4, 0.16, 0.16, 0.16, 0.12],          # a perfectly normal name
                 [0.30, 0.10, 0.0, 0.0, 0.0]])           # ...and one infeasible one
    capped, report = msc.apply_share_cap(df, cap=0.25, sources=['GOOD', 'SPARSE'])
    assert report['n_capped'].iloc[0] == 1, 'the normal name stopped being capped'
    A = capped.abs().iloc[0]
    assert A['a'] / A.sum() == pytest.approx(0.25, abs=1e-12)
    assert report['status'].iloc[1] == msc.STATUS_INFEASIBLE
    assert (capped.iloc[1] == df.iloc[1]).all()
    text = msc.format_report(report, 'general', cap=0.25)
    assert 'SPARSE' in text and 'NOT CAPPED' in text


def test_the_feasibility_boundary_is_k_max_and_it_MOVES_with_the_cap():
    """Stated as a property so it survives a re-level, and derived from the SAME `_k_max` the
    fixed-point scan uses -- see `test_the_feasibility_test_and_the_scan_CANNOT_disagree`."""
    four = _frame([[0.25, 0.25, 0.25, 0.25, 0.0]])
    _, report = msc.apply_share_cap(four, cap=0.25)
    assert report['status'].iloc[0] == msc.STATUS_OK
    assert report['base_after'].iloc[0] > 0

    three = _frame([[0.30, 0.10, 0.05, 0.0, 0.0]])
    _, r3 = msc.apply_share_cap(three, cap=0.25)
    assert r3['status'].iloc[0] == msc.STATUS_INFEASIBLE
    #  ...and the SAME row at a cap it can satisfy is scored normally
    _, r34 = msc.apply_share_cap(three, cap=0.34)
    assert r34['status'].iloc[0] == msc.STATUS_OK
    assert r34['base_after'].iloc[0] > 0


#  --- THE MASS, WHICH IS THE AXIS THE COUNT MISSED ------------------------------------- #
#
#  THE DEFECT THESE PIN (review 2). The first fix asked "does this row have more than
#  `_k_max` non-zero contributions?". The harm is "did the cap replace a real AggScore with a
#  fabricated ~0?". Those are different questions and rows exist that pass the first and fail
#  the second: `[0.30, 0.10, 0.05, 1e-9]` has four non-zero metrics, so it was FEASIBLE, and
#  all four were truncated to 1e-9 -- AggScore 0.450 -> 4e-09, `status = 'ok'`,
#  `share_after = 0.2500`, post-condition clean.

@pytest.mark.parametrize('tail', [1e-3, 1e-9, 1e-15])
def test_a_FEASIBLE_row_whose_cap_would_ERASE_it_ships_UNCAPPED(tail):
    """One epsilon outside the case the count could see, and the same invented promotion."""
    row = [0.30, 0.10, 0.05, tail]
    df = pd.DataFrame([row], columns=list('abcd'), dtype='float64')
    assert (np.abs(row) > 0).sum() > msc._k_max(0.25), 'the row must pass the COUNT test'
    capped, report = msc.apply_share_cap(df, cap=0.25, sources=['ERASED'])
    assert report['status'].iloc[0] == msc.STATUS_WOULD_ERASE
    pd.testing.assert_frame_equal(capped, df)
    assert report['agg_delta'].iloc[0] == 0.0, \
        'the name lost score to a truncation that was never committed'
    assert 'NOT CAPPED and shipped with their UNCAPPED score' in \
        msc.format_report(report, 'general', cap=0.25)


def test_the_mass_floor_does_NOT_touch_the_cascade_this_module_PINS_as_correct():
    """The floor is a fabrication detector, not a scoring level, and the line matters: the
    documented cascade [0.9, 0.05, 0.03, 0.01, 0.01] leaves 8% of the mass and is declared
    correct. If the floor ever rises above that it silently repeals a documented behaviour."""
    assert msc._MASS_FLOOR < 0.08, \
        'the mass floor now swallows the pinned k=3 cascade -- that is a behaviour change'
    df = _frame([[0.9, 0.05, 0.03, 0.01, 0.01]])
    capped, report = msc.apply_share_cap(df, cap=0.25)
    assert report['status'].iloc[0] == msc.STATUS_OK, 'the pinned cascade was declined'
    assert report['n_capped'].iloc[0] == 3
    assert float(capped.abs().iloc[0].sum()) == pytest.approx(0.08, abs=1e-12)
    surviving = report['base_after'].iloc[0] / report['base_before'].iloc[0]
    assert surviving == pytest.approx(0.08, abs=1e-9)
    assert surviving > msc._MASS_FLOOR


#  --- THE POST-CONDITION, DRIVEN THROUGH THE REAL ENTRY POINT -------------------------- #
#
#  Both of these used to hand-build a `pd.DataFrame({...})` and pass it straight to
#  `_assert_post_condition`. That is the exact shape this file's own docstring names as the
#  house defect -- a guard verified only against a frame the module cannot produce. They now
#  mutate `_cap_value` so `apply_share_cap` genuinely produces the bad state and the guard
#  fires from the real entry point.

def test_the_collapse_guard_FIRES_from_apply_share_cap(monkeypatch):
    """The backstop, WATCHED TO FAIL. A committed truncation must never destroy the name.

    Reached by forcing the fixed point to 0 AND standing the mass floor down, because the
    mass floor is what normally declines such a row. That mutation is the point: the guard is
    unreachable in the shipped configuration, so the only honest way to certify it is to
    weaken the check in front of it and watch this one catch what gets through. The previous
    version of this test hand-built `pd.DataFrame({'base_after': [0.0, 0.0], ...})` and called
    `_assert_post_condition` directly -- it never ran the module, which is the exact shape
    this file's docstring names as the house defect."""
    monkeypatch.setattr(msc, '_cap_value', lambda abs_row, cap: 0.0)
    monkeypatch.setattr(msc, '_MASS_FLOOR', 0.0)
    with pytest.raises(AssertionError, match='destroyed the name'):
        msc.apply_share_cap(_frame([[0.4, 0.16, 0.16, 0.16, 0.12]]), cap=0.25)


def test_a_base_that_OVERFLOWS_is_a_data_condition_not_an_algebra_assertion():
    """Every cell finite, their SUM infinite. This row used to keep `status = 'ok'` -- the
    non-finite check looked only at the cells -- and then trip the post-condition, so a data
    condition surfaced as an assertion that the algebra was broken. It must be declined like
    any other non-finite row: shipped uncapped, named in the log, no raise."""
    df = _frame([[1e308, 1e308, 1e308, 1e308, 1e308],
                 [0.4, 0.16, 0.16, 0.16, 0.12]])
    assert np.isfinite(df.iloc[0].to_numpy()).all()      # every CELL is finite...
    assert not np.isfinite(df.iloc[0].abs().sum())       # ...and the BASE is not
    capped, report = msc.apply_share_cap(df, cap=0.25, sources=['OVERFLOW', 'GOOD'])
    assert report['status'].iloc[0] == msc.STATUS_NON_FINITE
    assert report['n_capped'].iloc[1] == 1               # the pool still completed
    assert 'OVERFLOW' in msc.format_report(report, 'general', cap=0.25)


def test_the_post_condition_ignores_rows_the_module_DECLINED_to_cap():
    """The other half of the same rule: a row that shipped uncapped is over the cap BY
    DESIGN, and must not be reported as the module violating its own post-condition. Without
    the `status == ok` scoping, every infeasible name would raise the overshoot assertion."""
    df = _frame([[0.30, 0.10, 0.0, 0.0, 0.0]])
    _, report = msc.apply_share_cap(df, cap=0.25)
    assert report['share_after'].iloc[0] > 0.25          # it really is over the cap
    assert report['status'].iloc[0] != msc.STATUS_OK     # ...and that is why it is allowed


def test_NO_share_is_ever_reported_as_the_MINUS_ONE_sentinel():
    """`-1.0` used to mean "undefined" in `share_before` / `share_after`.  It printed to the
    run log as a share (`0.7500 -> -1.0000`), shipped inside `rankdic['share_cap_report']`,
    and -- being a number below the cap -- satisfied the post-condition.  A share is in
    [0, 1] or it is NaN; there is no third value."""
    df = _frame([[0.0] * 5,                              # all-zero
                 [0.9, 0.05, np.nan, 0.03, 0.02],        # non-finite
                 [0.4, 0.16, 0.16, 0.16, 0.12]])         # normal
    _, report = msc.apply_share_cap(df, cap=0.25)
    for col in ('share_before', 'share_after'):
        v = report[col].to_numpy(dtype='float64')
        defined = v[~np.isnan(v)]
        assert ((defined >= 0.0) & (defined <= 1.0)).all(), \
            '%s carries a value outside [0, 1] -- the sentinel is back' % col
    assert np.isnan(report['share_before'].iloc[0])
    assert np.isnan(report['share_after'].iloc[0])
    assert np.isnan(report['share_before'].iloc[1])


def test_metric_before_is_None_when_there_is_no_dominant_metric_to_name():
    """`nanargmax` over an all-sentinel row returns index 0, so a name with no defined share
    was reported as dominated by whatever the frame's FIRST column happens to be -- a real
    metric name, printed beside a real-looking share, about a name the module never
    assessed."""
    df = _frame([[0.0] * 5, [0.9, 0.05, np.nan, 0.03, 0.02]])
    _, report = msc.apply_share_cap(df, cap=0.25)
    assert report['metric_before'].iloc[0] is None, \
        'an all-zero row still names a dominant metric'
    assert report['metric_before'].iloc[1] is None, \
        'a non-finite row still names a dominant metric'


#  --- NaN ROWS: uncapped is survivable, SILENT is not --------------------------------- #

def test_a_NON_FINITE_row_ships_UNCAPPED_and_is_COUNTED_and_NAMED():
    """`_cap_value` returns None for any row containing a NaN, so a name at 90% on one
    metric goes through untouched.  That is defensible -- a share cannot be formed when one
    addend of the base is NaN.  What is not defensible is that it used to produce
    `n_capped = 0` and NO line in the run log, which reads exactly like "the cap looked at
    this name and nothing bound"."""
    df = _frame([[0.9, 0.05, np.nan, 0.03, 0.02],
                 [0.4, 0.16, 0.16, 0.16, 0.12]])
    capped, report = msc.apply_share_cap(df, cap=0.25, sources=['NANROW', 'GOOD'])
    #  the row really does ship uncapped
    assert report['n_capped'].iloc[0] == 0
    assert capped.iloc[0]['a'] == 0.9
    #  ...and it is declared, both in the frame and in the log
    assert report['status'].iloc[0] == msc.STATUS_NON_FINITE
    text = msc.format_report(report, 'general', cap=0.25)
    assert 'NOT CAPPED and shipped with their UNCAPPED score' in text, \
        'a name the cap could not assess left no trace in the run log'
    assert 'NANROW' in text, 'the skipped name is not named'


def test_a_pool_where_the_cap_binds_on_NOBODY_still_reports_its_skips():
    """The panel-level and per-name confusions are the same confusion.  Even with zero hits
    the block must say that a name was not assessed."""
    df = _frame([[0.2, 0.2, 0.2, 0.2, 0.2],
                 [0.9, 0.05, np.nan, 0.03, 0.02]])
    _, report = msc.apply_share_cap(df, cap=0.25, sources=['FLAT', 'NANROW'])
    text = msc.format_report(report, 'general', cap=0.25)
    assert 'NOTHING was truncated' in text
    assert 'NOT CAPPED and shipped with their UNCAPPED score' in text
    assert 'NANROW' in text


def test_an_INF_contribution_does_what_the_docstring_SAYS_and_does_not_blame_a_lost_share():
    """`inf` used to RAISE with "had a defined share before the cap and NONE after it" --
    which it never had. `base_before` is `inf`, so `defined_before` is False and every share
    is NaN, but the guard tested `base_before > 0` and `inf > 0` is True. Two costs: the
    documented behaviour did not happen, and the reader was sent to look for a vanished share
    instead of an infinite contribution."""
    df = _frame([[0.9, 0.05, np.inf, 0.03, 0.02],
                 [0.4, 0.16, 0.16, 0.16, 0.12]])
    capped, report = msc.apply_share_cap(df, cap=0.25, sources=['INFROW', 'GOOD'])
    assert report['status'].iloc[0] == msc.STATUS_NON_FINITE
    assert np.isinf(capped.iloc[0]['c'])                 # shipped exactly as it arrived
    assert report['n_capped'].iloc[1] == 1               # ...and the pool still completed
    text = msc.format_report(report, 'general', cap=0.25)
    assert 'INFROW' in text and 'NaN/inf' in text


def test_a_NON_FINITE_row_emits_NO_numpy_warning():
    """`C.sum(axis=1) - C0.sum(axis=1)` on a non-finite row printed `RuntimeWarning: invalid
    value encountered in subtract` to stderr -- unexplained noise landing right beside the log
    block that has just explained the skip in prose. The NaN result is correct and intended;
    only the warning is not."""
    import warnings
    np.seterr(all='warn')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        msc.apply_share_cap(_frame([[0.9, 0.05, np.inf, 0.03, 0.02],
                                    [0.9, 0.05, np.nan, 0.03, 0.02],
                                    [1e308, 1e308, 1e308, 1e308, 1e308],
                                    [0.4, 0.16, 0.16, 0.16, 0.12]]), cap=0.25)
    noisy = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not noisy, 'the cap emits unexplained numpy warnings into the run log: %s' % noisy


#  --- THE TWO TOLERANCES THAT MUST NOT DISAGREE ---------------------------------------- #

def test_the_feasibility_test_reads_k_max_and_carries_NO_SECOND_TOLERANCE():
    """WHAT WAS ACTUALLY WRONG, and it is narrower than it first looked.

    The feasibility test used to be spelled `n * cap >= 1 - 1e-12` while the fixed-point scan
    bounded itself with `floor(1/cap - 1e-12)`. Two independent spellings of one bound is the
    defect -- an editor who re-levels the cap and adjusts one has silently changed the
    relationship between them.

    *** BUT THE TWO SPELLINGS AGREE NUMERICALLY EVERYWHERE, AND THAT IS MEASURED, NOT
    ASSUMED: over 500+ cap levels (every 1/k for k in 2..39, every 0.001 step from 0.020 to
    0.500, and the six levels this suite parametrizes) x n in 1..39, there is NOT ONE (cap, n)
    pair where they differ. So this fix has no behavioural consequence today and no
    behavioural test can prove it -- which is why it is asserted STRUCTURALLY instead of being
    dressed up as a behaviour test that would pass either way. ***

    (The `[1.0, 0.001, 0.001]` at `cap = 1/3` reproduction that prompted this is NOT the two
    tolerances disagreeing -- both call n=3 feasible there. It is the exact-tie knife-edge in
    `_cap_value`, surfacing through a misleading assertion message. The message is fixed and
    tested below; the knife-edge is deliberately deferred, see the module docstring.)"""
    import ast as _ast
    with open(os.path.join(_REPO, 'metric_share_cap.py'), encoding='utf-8') as f:
        tree = _ast.parse(f.read())
    fn = next(n for n in tree.body
              if isinstance(n, _ast.FunctionDef) and n.name == 'apply_share_cap')
    calls_k_max = any(isinstance(n, _ast.Name) and n.id == '_k_max' for n in _ast.walk(fn))
    assert calls_k_max, 'apply_share_cap no longer derives its bound from _k_max'
    literals = [n.value for n in _ast.walk(fn)
                if isinstance(n, _ast.Constant) and isinstance(n.value, float)]
    assert 1e-12 not in literals, (
        'apply_share_cap has grown its own tolerance literal again -- the feasibility test '
        'and the fixed-point scan must share ONE derivation, or a re-level changes the '
        'relationship between them silently. Literals found: %s' % literals)


def test_the_k_max_bound_is_exactly_what_the_algebra_admits():
    """`c = cap * R / (1 - k*cap)` needs `1 - k*cap > 0`. Asserted at the boundary from both
    sides across every level the suite exercises plus the awkward ones."""
    for cap in (0.5, 0.4, 1.0 / 3.0, 0.3, 0.25, 0.2, 0.15, 0.125, 1.0 / 7.0, 0.1):
        k_max = msc._k_max(cap)
        assert 1.0 - k_max * cap > 0.0, 'k_max=%d admits a non-positive denominator at %r' % (k_max, cap)
        #  one more non-zero metric than k_max always leaves a tail, which is what makes
        #  "feasible" and "the scan can reach it" the same statement
        assert (k_max + 1) * cap >= 1.0 - 1e-9, \
            'cap=%r: %d metrics declared feasible but cannot each sit at or below the cap' % (cap, k_max + 1)


def test_the_UNRESOLVED_row_assertion_names_BOTH_causes_and_says_the_row_SHIPPED():
    """THE HALF OF S3-A THAT IS REAL AND OBSERVABLE. When `_cap_value` finds no consistent k
    for a row that needs one, the message used to read "this is a defect in `_cap_value`, NOT
    a data condition" -- flatly telling a reader the algebra was broken. There is a second,
    known cause: two contributions tying exactly at the fixed point, where the strict and
    non-strict comparisons can both fail by one ulp. Whoever takes this module's own advice to
    "re-level the cap" can reach it, and pointing them at the wrong thing costs them the
    night."""
    monkey = _frame([[1.0, 0.001, 0.001, 0.0, 0.0]])
    with pytest.raises(AssertionError) as exc:
        msc.apply_share_cap(monkey.iloc[:, :3], cap=1.0 / 3.0)
    msg = str(exc.value)
    assert 'shipped UNCAPPED' in msg, 'the message does not say what happened to the row'
    assert 'tie EXACTLY at the fixed point' in msg, 'the second known cause is not named'
    assert 'pass-through' in msg, 'the primary cause is not named'
    assert 'not a data condition' not in msg, \
        'the message still asserts the algebra is broken as if it were the only possibility'


def test_the_shipped_cap_level_is_on_the_SAFE_side_of_both_tolerances():
    """CAP = 0.25 specifically: `1/cap` is exactly representable, so no epsilon is doing any
    work. This is why the tolerance defect was latent rather than live."""
    assert msc._k_max(0.25) == 3
    assert 1.0 / 0.25 == 4.0                             # exact in binary
    assert 4 * 0.25 == 1.0


#  --- THE RUN LOG: it is read by the CEO, so it must be true --------------------------- #

def test_the_RAISED_label_follows_the_DOMINANT_metrics_SIGN_not_the_sign_of_agg_delta():
    """`agg_delta` is the sum of the signed changes across ALL capped metrics.  On a k > 1
    row with mixed signs it can be POSITIVE while the dominant contribution is positive too
    -- and the log then told the reader that a POSITIVE dominant metric "was NEGATIVE".

    `[+0.30, -0.29, -0.28, +0.10, +0.03]` resolves at k = 3 with `agg_delta = +0.14`, and
    its dominant contribution is `+0.30`.  The docstring cites BOSS.DE and JEN.DE as exactly
    this mechanism, so the label is load-bearing for how a mover is read."""
    df = _frame([[0.30, -0.29, -0.28, 0.10, 0.03]])
    _, report = msc.apply_share_cap(df, cap=0.25, sources=['ACME'])
    assert report['n_capped'].iloc[0] > 1
    assert report['agg_delta'].iloc[0] > 0               # the score rose...
    assert report['contrib_before'].iloc[0] > 0          # ...on a POSITIVE dominant metric
    text = msc.format_report(report, 'general', cap=0.25)
    assert 'dominant contribution was NEGATIVE' not in text, \
        'the log claims a positive dominant contribution was negative'
    assert 'POSITIVE' in text, 'the log does not say what actually happened'

    #  ...and the genuine case still gets the genuine label
    neg = _frame([[-0.9, 0.05, 0.03, 0.01, 0.01]])
    _, rep2 = msc.apply_share_cap(neg, cap=0.25, sources=['BOSS.DE'])
    assert 'RAISED: its dominant contribution was NEGATIVE' in \
        msc.format_report(rep2, 'general', cap=0.25)


def test_the_log_PRINTS_n_capped_so_a_CASCADE_is_not_read_as_a_ONE_METRIC_TRIM():
    """A k = 3 name printed one metric name and one share pair, so it was indistinguishable
    from a single-metric trim -- and the AggScore delta beside it could not be reconciled
    with the `0.3000 -> 0.2500` shown.  That is what made the cascade invisible in the log
    at a rate of roughly 1.7 names per panel."""
    df = _frame([[0.9, 0.05, 0.03, 0.01, 0.01],
                 [0.4, 0.16, 0.16, 0.16, 0.12]])
    _, report = msc.apply_share_cap(df, cap=0.25, sources=['CASCADE', 'TRIM'])
    assert report['n_capped'].iloc[0] == 3
    assert report['n_capped'].iloc[1] == 1
    text = msc.format_report(report, 'general', cap=0.25)
    assert 'n_capped=3' in text, 'a three-metric cascade does not say so'
    assert 'n_capped=1' in text
    assert 'bound on MORE THAN ONE metric' in text


def test_the_report_carries_SOURCE_so_the_documented_JOIN_actually_works():
    """`postBoRank` tells a consumer to "join on `source`, never on position".  The report
    had no `source` column, and position is unusable: `postRank` and `psmdf_normalized` are
    one object and `getAggScore` does `reset_index(drop=True)`.  So the frame shipped in
    `rankdic['share_cap_report']` could not be joined to the ranked artifact at all."""
    df = _frame([[0.4, 0.16, 0.16, 0.16, 0.12], [0.2, 0.2, 0.2, 0.2, 0.2]])
    _, report = msc.apply_share_cap(df, cap=0.25, sources=['AAA.L', 'BBB.DE'])
    assert 'source' in report.columns, 'the documented join key is missing'
    assert list(report['source']) == ['AAA.L', 'BBB.DE']
    assert list(report.columns)[0] == 'source'
    #  the log names the same thing the frame does
    assert 'AAA.L' in msc.format_report(report, 'general', cap=0.25)


def test_a_MISALIGNED_source_list_is_REFUSED_rather_than_joined_to_the_wrong_names():
    """The failure this key introduces: a `sources` list of the wrong length would silently
    label every row after the mismatch with another company's ticker."""
    df = _frame([[0.4, 0.16, 0.16, 0.16, 0.12], [0.2, 0.2, 0.2, 0.2, 0.2]])
    with pytest.raises(ValueError, match='SAME row order'):
        msc.apply_share_cap(df, cap=0.25, sources=['AAA.L'])


#  --- MUTATION: the guards must actually fire ----------------------------------------- #
#
#  BOTH of these now run the REAL `apply_share_cap`.  They did not before: one hand-built a
#  `pd.DataFrame({'share_after': [...]})` and fed it to `_assert_post_condition`, and the
#  other never called `msc` AT ALL -- it asserted `0.9 / 1.0 > 0.25`, which is arithmetic,
#  and would have passed with `metric_share_cap.py` deleted from the repo.

def test_mutation_one_pass_cap_is_REJECTED_by_the_post_condition(monkeypatch):
    """The trap this module was built to avoid, exercised THROUGH the module.  The naive
    one-pass cut -- truncate to `cap x base_before` -- passes a casual read and leaves the
    capped metric above the cap of the base that ships.  Substituting it for the fixed point
    must make `apply_share_cap` itself raise."""
    def one_pass(abs_row, cap):
        return cap * float(np.sum(abs_row))              # the one-pass form

    monkeypatch.setattr(msc, '_cap_value', one_pass)
    df = _frame([[0.9, 0.05, 0.03, 0.01, 0.01]])
    with pytest.raises(AssertionError, match='ABOVE it'):
        msc.apply_share_cap(df, cap=0.25)


def test_mutation_a_NO_OP_cap_is_REJECTED_by_the_post_condition(monkeypatch):
    """If someone replaces the cap body with a pass-through, the module must fail LOUDLY
    rather than return an uncapped frame and a clean report.  Verified by substituting the
    pass-through and watching the real call raise -- not by asserting arithmetic about a
    frame the module never saw."""
    monkeypatch.setattr(msc, '_cap_value', lambda abs_row, cap: None)   # a no-op cap
    df = _frame([[0.9, 0.05, 0.03, 0.01, 0.01]])
    with pytest.raises(AssertionError, match='ABOVE it'):
        msc.apply_share_cap(df, cap=0.25)
    #  THE SENSITIVITY HALF, and it has to come AFTER `monkeypatch.undo()`. The previous
    #  version ran it under the still-active patch -- `monkeypatch` unwinds at teardown, not
    #  mid-test -- so the assertion labelled "sensitive to the cap being disabled" passed
    #  identically with the real `_cap_value` and with the no-op, and proved nothing.
    monkeypatch.undo()
    capped, report = msc.apply_share_cap(df, cap=0.25)
    assert report['n_capped'].iloc[0] == 3, \
        'the REAL cap no longer binds on this row, so the raise above proves nothing'
    assert report['share_after'].iloc[0] == pytest.approx(0.25, abs=1e-12)


#  --- THE SAVED PANELS: the regression that would have caught the ship blocker --------- #
#
#  THE DEFECT THIS PINS (review 2). Every other test here is synthetic, and the first fix
#  passed all of them while raising on the two saved panels drawn from the CURRENT CUR3K
#  universe -- `STRT` and `PET.TO` on 2026-08-11, `STRT` again on 2026-08-07. Because
#  `postBo.py:697` does not exception-guard the general-pool call, that is the whole run:
#  no postRank, no AggScoreTop CSV, no top-20. The condition is deterministic on a panel, so
#  a `-loadbometric` re-run hits it again and recovery needs a code edit.
#
#  Read-only, offline, no pipeline: these are pickles already on disk. The test SKIPS rather
#  than fails when a panel is absent, because the panels are run artifacts and not every
#  checkout has them -- which is a real weakness, stated rather than hidden: on a machine
#  without them this test proves nothing.

_PANELS = [('postRank_2026-08-11_fmp_stock_CUR3K.pickle', ['STRT', 'PET.TO']),
           (os.path.join('panels_2026-08-07', 'postRank_2026-08-07_fmp_stock_CUR3K.pickle'),
            ['STRT'])]


def _panel_contributions(relpath):
    """The weighted-contribution block of a saved panel, in the shape the cap sees."""
    path = os.path.join(_REPO, relpath)
    if not os.path.exists(path):
        pytest.skip('saved panel not present in this checkout: %s' % relpath)
    pr = pd.read_pickle(path)['postRank']
    metrics = [m for m, w in sw.DEPLOYED.items() if w != 0 and m in pr.columns]
    return pr, pr[metrics].apply(pd.to_numeric, errors='coerce')


@pytest.mark.parametrize('relpath,sparse_names', _PANELS)
def test_a_SAVED_PRODUCTION_PANEL_COMPLETES_and_does_not_kill_the_run(relpath, sparse_names):
    """The pool must come out the other side. Not "mostly" -- the call site has no except."""
    pr, X = _panel_contributions(relpath)
    capped, report = msc.apply_share_cap(X, cap=msc.CAP, sources=list(pr['source']))
    assert len(capped) == len(X)
    assert set(report['status']) <= {msc.STATUS_OK, msc.STATUS_ALL_ZERO,
                                     msc.STATUS_INFEASIBLE, msc.STATUS_WOULD_ERASE,
                                     msc.STATUS_NON_FINITE}


@pytest.mark.parametrize('relpath,sparse_names', _PANELS)
def test_the_SPARSE_NAMES_on_a_saved_panel_keep_their_score_and_are_NAMED(relpath, sparse_names):
    """The names that used to be zeroed, and then used to kill the run. Each must ship with
    the score it had, and each must appear in the log block -- shipping uncapped is only
    acceptable because it is disclosed."""
    pr, X = _panel_contributions(relpath)
    capped, report = msc.apply_share_cap(X, cap=msc.CAP, sources=list(pr['source']))
    text = msc.format_report(report, 'general', cap=msc.CAP)
    for name in sparse_names:
        hit = report.index[report['source'] == name]
        assert len(hit) == 1, '%s not on panel %s' % (name, relpath)
        i = hit[0]
        assert report.loc[i, 'status'] in (msc.STATUS_INFEASIBLE, msc.STATUS_WOULD_ERASE), (
            '%s is no longer declined -- if the cap now COMMITS a truncation on it, check '
            'what its AggScore became before assuming that is an improvement' % name)
        assert report.loc[i, 'agg_delta'] == 0.0, '%s lost score to a declined cap' % name
        assert (capped.loc[i] == X.loc[i]).all(), '%s was modified' % name
        assert name in text, '%s ships uncapped and is NOT named in the run log' % name


@pytest.mark.parametrize('relpath,sparse_names', _PANELS)
def test_the_mass_floor_does_not_fire_on_any_NORMAL_name_of_a_saved_panel(relpath, sparse_names):
    """The floor is meant to be orders of magnitude clear of anything real. Asserted against
    real panels so that "clear of anything real" is measured, not assumed -- if a future
    weight vector or squash brings a genuine name near it, this is what says so."""
    pr, X = _panel_contributions(relpath)
    _, report = msc.apply_share_cap(X, cap=msc.CAP, sources=list(pr['source']))
    ok = report[report['status'] == msc.STATUS_OK]
    surviving = (ok['base_after'] / ok['base_before']).min()
    assert surviving > 10 * msc._MASS_FLOOR, (
        'the thinnest surviving mass on %s is %.4f, within 10x of the floor %.4f -- the '
        'floor is no longer safely clear of real data' % (relpath, surviving, msc._MASS_FLOOR))


#  --- THE TWO CEO-FACING FIELDS, AT VALUE LEVEL ---------------------------------------- #
#
#  THE COVERAGE HOLE THESE CLOSE (third review, 2026-08-31).  `metric_before` and
#  `share_before` are what the run log and the evidence CSV put in front of the CEO -- "this
#  name was dominated by THIS metric, at THIS share".  63 tests, and not one asserted either
#  at value level: mutating `metric_before` to always return `cols[0]` -- literally the defect
#  this module's docstring says it fixed -- survived the entire suite, and so did reporting
#  the POST-cap share as `share_before`.  The shipped code was correct both times; the tests
#  simply could not have told anyone.
#
#  Seventeenth instance of the house shape, and the instructive part is WHERE it was: not in
#  the algebra, which is covered to death, but in the two fields whose only job is to be read
#  by a human.

def test_metric_before_NAMES_the_dominant_metric_and_not_the_first_column():
    """Value-level, and deliberately built so the dominant metric is NEVER column 0 -- the
    `cols[0]` fallback the pre-fix code used would otherwise look correct by luck."""
    #  dominant in position 3, then 1, then 4: no row here is dominated by 'a'
    df = _frame([[0.05, 0.20, 0.03, 0.60, 0.12],
                 [0.10, 0.55, 0.05, 0.20, 0.10],
                 [0.02, 0.04, 0.06, 0.08, 0.80]])
    _, report = msc.apply_share_cap(df, cap=0.25, sources=['R1', 'R2', 'R3'])
    assert list(report['metric_before']) == ['d', 'b', 'e'], (
        'metric_before does not name the largest |contribution| -- a reader is being told '
        'the wrong driver for this name')
    #  ...and it agrees with an INDEPENDENT recompute off the input frame, not off the report
    assert list(report['metric_before']) == list(df.abs().idxmax(axis=1))


def test_share_before_is_the_PRE_cap_share_not_the_post_cap_one():
    """`share_before` and `share_after` are a BEFORE/AFTER pair, and on a bound row they must
    differ.  Reporting the post-cap share as `share_before` would print `0.2500 -> 0.2500` and
    quietly erase the evidence that anything happened."""
    df = _frame([[0.60, 0.20, 0.10, 0.06, 0.04]])
    _, report = msc.apply_share_cap(df, cap=0.25)
    A = df.abs().iloc[0]
    assert report['share_before'].iloc[0] == pytest.approx(float(A.max() / A.sum()), abs=1e-12)
    assert report['share_before'].iloc[0] == pytest.approx(0.60, abs=1e-12)
    assert report['share_after'].iloc[0] == pytest.approx(0.25, abs=1e-12)
    assert report['share_before'].iloc[0] > report['share_after'].iloc[0] + 0.01, (
        'share_before equals share_after on a row the cap bound -- the before/after pair is '
        'no longer a before/after pair')


def test_contrib_before_is_the_SIGNED_pre_cap_value_of_that_same_metric():
    """The third field the log reads, and the one the RAISED label is derived from.  It must
    be the SIGNED contribution of `metric_before`, taken BEFORE the truncation."""
    df = _frame([[0.05, -0.60, 0.03, 0.20, 0.12]])
    _, report = msc.apply_share_cap(df, cap=0.25)
    assert report['metric_before'].iloc[0] == 'b'
    assert report['contrib_before'].iloc[0] == pytest.approx(-0.60, abs=1e-12), (
        'contrib_before is not the pre-cap signed value of the dominant metric')


@pytest.mark.parametrize('relpath,sparse_names', _PANELS)
def test_the_CEO_FACING_fields_match_an_INDEPENDENT_recompute_on_a_REAL_panel(
        relpath, sparse_names):
    """The synthetic tests above pin the rule; this pins it against every row of a real panel,
    recomputed from the input frame without touching the module's own bookkeeping."""
    pr, X = _panel_contributions(relpath)
    _, report = msc.apply_share_cap(X, cap=msc.CAP, sources=list(pr['source']))
    A = X.abs()
    base = A.sum(axis=1)
    defined = np.isfinite(base) & (base > 0)
    exp_metric = A.idxmax(axis=1)
    exp_share = A.max(axis=1) / base

    got_metric = report['metric_before']
    bad = [(report['source'].iloc[i], got_metric.iloc[i], exp_metric.iloc[i])
           for i in range(len(report))
           if bool(defined.iloc[i]) and got_metric.iloc[i] != exp_metric.iloc[i]]
    assert not bad, 'metric_before disagrees with an independent recompute: %s' % bad[:5]

    m = defined.to_numpy()
    assert np.allclose(report['share_before'].to_numpy(dtype='float64')[m],
                       exp_share.to_numpy(dtype='float64')[m], atol=1e-12), (
        'share_before disagrees with an independent recompute on %s' % relpath)


#  --- THE SHIPPED ARTIFACT -------------------------------------------------------------- #
#
#  THE GAP THIS CLOSES (third review).  The cap disclosed itself only in `rankdic` -- which no
#  consumer in this repo reads -- and in stdout.  The entire justification for shipping an
#  uncappable name UNCAPPED is that we said so, and a disclosure living only in a console log
#  is one nobody is obliged to have read.
#
#  Every test here passes an explicit `path` under `tmp_path`: this writer targets the repo
#  root in production, and a test that forgot would drop a run-evidence-shaped file into the
#  working tree, which is the damage class conftest's RULE E exists to stop.

def test_the_evidence_CSV_is_WRITTEN_and_carries_every_name_in_the_pool(tmp_path):
    """One row per name, capped or not, declined or not -- because "the cap bound on nobody"
    and "the cap did not run" have to stay distinguishable in the ARTIFACT too."""
    df = _frame([[0.4, 0.16, 0.16, 0.16, 0.12],          # bound
                 [0.2, 0.2, 0.2, 0.2, 0.2],              # untouched
                 [0.30, 0.10, 0.0, 0.0, 0.0]])           # declined: infeasible
    _, report = msc.apply_share_cap(df, cap=0.25, sources=['TRIM', 'FLAT', 'SPARSE'])
    out = tmp_path / 'ShareCapReport_TEST.csv'
    got = msc.write_evidence_csv(report, 'general', path=str(out))
    assert got == str(out) and out.exists(), 'no evidence CSV was written'
    back = pd.read_csv(out)
    assert len(back) == 3, 'the CSV drops names -- it must carry the whole pool'
    assert list(back['source']) == ['TRIM', 'FLAT', 'SPARSE']
    assert list(back['status']) == ['ok', 'ok', msc.STATUS_INFEASIBLE]
    #  the declined name is findable by a reader who never saw stdout
    row = back[back['source'] == 'SPARSE'].iloc[0]
    assert row['n_capped'] == 0
    assert row['agg_delta'] == 0.0


def test_the_evidence_CSV_leads_with_the_columns_a_READER_needs(tmp_path):
    """Column ORDER is part of the artifact: who, what happened to them, then the numbers."""
    df = _frame([[0.4, 0.16, 0.16, 0.16, 0.12]])
    _, report = msc.apply_share_cap(df, cap=0.25, sources=['TRIM'])
    out = tmp_path / 'x.csv'
    msc.write_evidence_csv(report, 'general', path=str(out))
    cols = list(pd.read_csv(out).columns)
    assert cols[:4] == ['pool', 'source', 'status', 'n_capped'], cols
    for needed in msc.CSV_COLUMNS:
        assert needed in cols, 'the CSV lost the %r column' % needed


def test_the_evidence_CSV_APPENDS_so_a_SECOND_POOL_cannot_clobber_the_first(tmp_path):
    """`postBoScoreRanking` runs once per pool.  A per-call overwrite would leave only the
    last one -- the same single-writer clobber `_write_missing_csv` documents."""
    df = _frame([[0.4, 0.16, 0.16, 0.16, 0.12]])
    _, report = msc.apply_share_cap(df, cap=0.25, sources=['AAA'])
    out = str(tmp_path / 'y.csv')
    msc._CSV_STARTED.discard(out)
    msc.write_evidence_csv(report, 'general', path=out)
    msc.write_evidence_csv(report, 'FinManager', path=out)
    back = pd.read_csv(out)
    assert len(back) == 2, 'the second pool overwrote the first'
    assert list(back['pool']) == ['general', 'FinManager']


def test_the_evidence_writer_NEVER_costs_the_run(tmp_path):
    """An evidence file must not be able to kill a scored run -- the contract
    `adhoc_penalty.write_evidence_csv` carries.  It returns None and prints instead."""
    df = _frame([[0.4, 0.16, 0.16, 0.16, 0.12]])
    _, report = msc.apply_share_cap(df, cap=0.25, sources=['AAA'])
    #  a directory standing where the file must go: unwritable by construction
    blocked = tmp_path / 'blocked'
    blocked.mkdir()
    assert msc.write_evidence_csv(report, 'general', path=str(blocked)) is None
    assert msc.write_evidence_csv(None, 'general', path=str(tmp_path / 'z.csv')) is None


def test_the_UNCAPPABLE_log_block_does_NOT_TRUNCATE():
    """The block that IS the mitigation must not silently truncate itself.  Twenty-five
    declined names: a `[:20]` would ship five of them undisclosed while the log still read as
    a complete disclosure."""
    rows = [[0.30, 0.10, 0.0, 0.0, 0.0] for _ in range(25)]
    names = ['SPARSE%02d' % i for i in range(25)]
    _, report = msc.apply_share_cap(_frame(rows), cap=0.25, sources=names)
    text = msc.format_report(report, 'general', cap=0.25)
    missing = [n for n in names if n not in text]
    assert not missing, ('the uncappable block truncated and hid %d name(s): %s'
                         % (len(missing), missing))
    assert 'ALL 25 are named here' in text


#  --- THE CALL SITE: ordering and pool scoping ---------------------------------------- #
#
#  Claim 3 of the review -- the cap runs AFTER the weighting loop and BEFORE the ad-hoc
#  penalty column is attached -- was verified only by READING `postBoRank`.  Nothing failed
#  if someone moved the `psmdf_normalized` concat above the cap block, or attached the
#  penalty earlier: both would silently change what is capped (the penalty would enter the
#  share denominator and be truncated as if it were a metric).
#
#  Asserted against `postBoRank.py`'s SOURCE, parsed with `ast` -- the file is NOT imported,
#  because importing it pulls in the whole live data layer.
#
#  WHAT THIS CANNOT DETECT, stated so it is not mistaken for more than it is: it checks
#  lexical order and nesting, not runtime identity.  A reassignment of
#  `temp_normpsmdf_weighted` between the cap and the concat, a caller passing
#  `pool_label='general'` for a cohort, or a change in what `weight_series` contains would
#  all pass this.  Only a live run covers those.

def _postbo_rank_function():
    with open(os.path.join(_REPO, 'postBoRank.py'), encoding='utf-8') as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'postBoScoreRanking':
            return node
    raise AssertionError('postBoScoreRanking not found in postBoRank.py')


def _first_lineno(node, pred, what):
    """The EARLIEST source line in `postBoScoreRanking` at which `pred` matches a node."""
    lines = [n.lineno for n in ast.walk(node) if hasattr(n, 'lineno') and pred(n)]
    assert lines, ('%s not found in postBoScoreRanking -- the call site has been '
                   'restructured' % what)
    return min(lines)


def _is_name(nm):
    return lambda n: isinstance(n, ast.Name) and n.id == nm


def _is_attr(a):
    return lambda n: isinstance(n, ast.Attribute) and n.attr == a


def _calls_the_cap(stmt):
    return any(_is_attr('apply_share_cap')(n) for n in ast.walk(stmt))


def test_the_call_site_runs_the_cap_AFTER_weighting_and_BEFORE_the_penalty_attach():
    """The ordering is forced from both sides and neither side was tested.

    BEFORE the cap: the weighting loop, because the cap is defined on WEIGHTED contributions.
    AFTER the cap: the `psmdf_normalized` concat and the ad-hoc penalty column, because the
    penalty is a CEO decision and not a metric -- it must be neither in the share denominator
    nor itself truncated."""
    fn = _postbo_rank_function()
    weighting = _first_lineno(fn, _is_name('temp_normpsmdf_weighted'), 'the weighting block')
    cap_call = _first_lineno(fn, _is_attr('apply_share_cap'), 'the share-cap call')
    concat = _first_lineno(fn, _is_name('psmdf_normalized'), 'the psmdf_normalized concat')
    penalty = _first_lineno(fn, _is_name('ADHOC_PENALTY_COLUMN'), 'the ad-hoc penalty attach')
    assert weighting < cap_call, 'the cap runs before the weighting loop -- it would cap raw z'
    assert cap_call < concat, 'the cap runs after psmdf_normalized is assembled'
    assert cap_call < penalty, \
        'the ad-hoc penalty is attached before the cap -- it would be capped as a metric'


def test_postBoScoreRanking_still_has_a_SINGLE_return_so_the_ordering_holds_on_every_path():
    """The ordering claim above is only as good as there being one exit. An early return
    added between the weighting loop and the cap would ship an UNCAPPED general pool with a
    green suite and no banner in the log."""
    fn = _postbo_rank_function()
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert len(returns) == 1, \
        ('postBoScoreRanking now has %d return statements. Every one of them must be checked '
         'against the cap/penalty ordering above.' % len(returns))


def test_the_cap_is_INSIDE_the_CAPPED_POOLS_branch_so_a_cohort_is_SKIPPED():
    """Scoping, asserted at the call site rather than only on the constant.
    `test_the_two_ceo_numbers` pins `CAPPED_POOLS == {'general'}`, which says nothing about
    whether the branch is still wired to it.  The cohort vectors concentrate weight BY
    DESIGN -- FIN-1 puts 0.275 on `bVpRatio` -- so a cap applied to them would bind on
    almost every member for a structural reason."""
    fn = _postbo_rank_function()
    guarded = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        if 'CAPPED_POOLS' not in ast.dump(node.test):
            continue
        in_body = any(_calls_the_cap(s) for s in node.body)
        in_else = any(_calls_the_cap(s) for s in node.orelse)
        guarded.append((in_body, in_else, node))
    assert guarded, 'the cap call is not guarded by a CAPPED_POOLS test at all'
    in_body, in_else, node = guarded[0]
    assert in_body and not in_else, \
        'apply_share_cap is not inside the CAPPED_POOLS branch -- every pool would be capped'
    #  ...and the pool that is skipped SAYS SO, rather than reading like a pool that had
    #  nothing to cap
    assert any(isinstance(n, ast.Constant) and isinstance(n.value, str)
               and 'NOT APPLIED' in n.value
               for s in node.orelse for n in ast.walk(s)), \
        'a skipped pool prints no banner -- silence reads as "the cap found nothing"'


def test_the_call_site_WRITES_the_evidence_CSV_inside_the_capped_pool_branch():
    """A writer nothing calls is not a disclosure.  Asserted at the call site, in the same
    branch as the cap itself, so a pool that is capped is always a pool that is recorded."""
    fn = _postbo_rank_function()
    guarded = [n for n in ast.walk(fn)
               if isinstance(n, ast.If) and 'CAPPED_POOLS' in ast.dump(n.test)]
    assert guarded, 'the cap call is not guarded by a CAPPED_POOLS test at all'
    body = guarded[0].body
    assert any(_is_attr('write_evidence_csv')(n) for s in body for n in ast.walk(s)), (
        'the capped-pool branch runs the cap but never writes the evidence CSV -- the '
        'per-name disclosure would exist only in stdout')
    #  ...and it runs AFTER the cap, since it writes the cap's own report
    cap_line = _first_lineno(fn, _is_attr('apply_share_cap'), 'the share-cap call')
    csv_line = _first_lineno(fn, _is_attr('write_evidence_csv'), 'the evidence-CSV call')
    assert cap_line < csv_line


#  --- CLAIMS IN THE PROSE THAT TURNED OUT TO BE FALSE ----------------------------------- #
#
#  Two sentences in this module were load-bearing and wrong, and both were believed by a
#  later reader and acted on.  Prose is not normally test material; these two are, because
#  each one is the reason somebody made a decision.

def _module_source():
    with open(os.path.join(_REPO, 'metric_share_cap.py'), encoding='utf-8') as f:
        return f.read()


def test_the_module_does_NOT_claim_it_never_raises_on_a_data_condition():
    """It does raise on one: an EXACT tie at the fixed point leaves `_cap_value` with no
    consistent k, and the post-condition fires.  Reproduced at the SHIPPED cap on
    `[0.78, 0.72, 0.65, 0.63, 0.09, 0.07]`.  Two versions of this file asserted the opposite,
    and the second of them also called the case "harmless at CAP = 0.25"."""
    src = _module_source()
    assert 'NEVER RAISES ON A DATA CONDITION' not in src, (
        'the module claims it never raises on a data condition; an exact tie is a data '
        'condition and it raises')
    assert 'A DATA condition never raises' not in src
    assert 'harmless at the shipped CAP' not in src and 'Harmless at the shipped' not in src
    #  ...and the true statement is present, with its measured scope
    assert 'EXACT TIE' in src or 'tie EXACTLY at the fixed point' in src
    assert '2,000,000' in src, 'the measured-zero scope is not stated'


def test_the_module_does_NOT_call_the_CUR3K_panels_the_current_universe():
    """Live is CUR6K.  Every sparse-name and mass-floor number here is from CUR3K panels, so
    it is a lower bound on the shape's incidence and not a prediction of tonight's rate."""
    src = _module_source()
    assert 'CURRENT CUR3K universe' not in src, (
        'the module still calls CUR3K the current universe; the live universe is CUR6K and '
        'the panel evidence is off-universe')
    assert 'CUR6K' in src, 'the live universe is not named'
    assert 'LOWER BOUND' in src, 'the off-universe evidence is not qualified'


def test_the_evidence_CSV_records_what_it_needs_from_OUTSIDE_this_module():
    """The writer is useless if the file never leaves the run machine, and the two additions
    that make it leave are in files this change could not touch.  Pinned so the gap is a
    known open item rather than something discovered when the CEO asks where the CSV is."""
    src = _module_source()
    assert 'Sbocker.allowlist_patterns' in src, (
        'the module does not record that the transfer allow-list still needs '
        "'ShareCapReport_*.csv' -- written-but-unshipped is the same as unwritten")
    assert '_EVIDENCE_GLOBS' in src


#  --- THE WEIGHT TRANSFER -------------------------------------------------------------- #

def test_transfer_preserves_sum_abs_exactly():
    """Sigma|w| = 1.000000 must survive the transfer EXACTLY, not to a tolerance: every
    published AggScore range rests on it and `_validate()` asserts it at import."""
    assert sw.sum_abs(sw.DEPLOYED) == 1.0
    assert sw.sum_abs(sw.DEPLOYED_DERIVED) == 1.0


def test_only_two_metrics_move_and_they_move_by_equal_and_opposite_amounts():
    """The point of a targeted transfer over a block-budget move: no third metric shifts as
    a side effect.  Asserted structurally, so it holds whatever delta is chosen."""
    moved = {k: sw.DEPLOYED[k] - sw.DEPLOYED_DERIVED[k]
             for k in sw.METRIC_KEYS if sw.DEPLOYED[k] != sw.DEPLOYED_DERIVED[k]}
    assert set(moved) == set(sw.THESIS_TRANSFER)
    assert len(moved) == 2
    assert sum(moved.values()) == pytest.approx(0.0, abs=1e-15)
    assert moved['earnYield'] < 0 < moved['incomeQuality']


def test_transfer_holds_the_thesis_margin_today():
    """`earnYield` must remain the largest single |w| in the general vector.  The transfer
    narrows this margin deliberately; the invariant is that it does not cross it."""
    largest = max(sw.DEPLOYED, key=lambda k: abs(sw.DEPLOYED[k]))
    assert largest == sw.THESIS_METRIC
    assert abs(sw.DEPLOYED['earnYield']) > abs(sw.DEPLOYED['incomeQuality'])


def test_the_post_D3_margin_is_now_NEGATIVE_and_that_is_recorded_not_asserted_away():
    """The priced cost of the CEO's ruling (scoringWeights B.5b / B.6).  Fixing audit D3
    drops `earnYield` to 0.39*W_P; after the transfer that lands BELOW `incomeQuality`.

    This test asserts the CONSEQUENCE so that a future D3 fix cannot land quietly: whoever
    promotes `grahamNumberToPrice` must deal with W_P in the same change.  It is not
    asserting that the current vector is broken -- `test_transfer_holds_the_thesis_margin_today`
    covers that and passes."""
    post_d3_earn_yield = 0.39 * sw.GENERAL_BUDGETS['P'] + sw.THESIS_TRANSFER['earnYield']
    assert post_d3_earn_yield < sw.DEPLOYED['incomeQuality'], (
        'the post-D3 margin has become positive again -- B.5b/B.6 say it is negative. If a '
        'budget moved to fix that, update BOTH notes; do not delete this test.')


def test_the_transfer_does_not_reach_the_cohort_vectors():
    """Scoping, asserted.  The five cohorts derive from GENERAL_BUDGETS via Rule PROP, not
    from DEPLOYED, so a post-derivation transfer must leave them untouched."""
    for label in sw.COHORT_LABELS:
        vec = sw.COHORT_WEIGHTS_RAW[label]
        expected = sw._block_vector(sw._cohort_budgets(label),
                                    sw._cohort_assignment(label))
        for metric in sw.THESIS_TRANSFER:
            assert vec.get(metric, 0.0) == expected.get(metric, 0.0), \
                '%s: the general-pool transfer leaked into a cohort vector' % label


def test_a_transfer_that_does_not_sum_to_zero_is_REFUSED():
    with pytest.raises(RuntimeError, match='pure TRANSFER'):
        sw._apply_transfer(dict(sw.DEPLOYED_DERIVED),
                           {'earnYield': -0.006, 'incomeQuality': +0.005})


def test_a_transfer_that_would_flip_a_sign_is_REFUSED():
    """A magnitude edit must never become a meaning edit.  `CycleHeat` is negative by design
    (hot late cycle is WORSE); a transfer large enough to invert it would silently reverse
    what the metric says."""
    with pytest.raises(RuntimeError, match='crossing zero'):
        sw._apply_transfer(dict(sw.DEPLOYED_DERIVED),
                           {'CycleHeat': +0.2, 'incomeQuality': -0.2})


#  --- THE TWO CEO NUMBERS -- the only deliberate value pins in this file --------------- #

def test_the_two_ceo_numbers():
    """DELIBERATE VALUE PINS.  Both are CEO decisions of 2026-08-31, not derived quantities.

    MOVING THE CAP is a one-line edit here plus the note in `metric_share_cap`.  MOVING THE
    DELTA is NOT -- see this file's docstring and `test_the_delta_is_what_moves_the_e2_pins`
    below: three pinned numbers in `test_e2_weight_vector.py` move with it, and the earlier
    claim that "nothing else in the suite has to be touched" was false.

    THE CAP LEVEL CARRIES A CAVEAT THE CEO HAS NOT YET RULED ON (metric_share_cap, "(i) vs
    (ii)"): 0.25 was chosen against the SIGNED-AggScore share distribution, where 18 of the
    2026-08-31 top-20 sit at or above it, and is applied to the ABSOLUTE-contribution one,
    where only 4 of 20 do.  If he re-levels it after seeing that, this is the line to edit."""
    assert msc.CAP == 0.25
    assert sw.THESIS_TRANSFER_DELTA == 0.006
    assert msc.CAPPED_POOLS == frozenset({'general'})


def test_the_delta_is_what_moves_the_e2_pins_and_this_file_no_longer_claims_otherwise():
    """THE CROSS-FILE DEPENDENCY, asserted rather than described.

    `test_e2_weight_vector.py` hard-codes three numbers that are all functions of
    `THESIS_TRANSFER_DELTA`.  Re-derived here from `sw` so that moving the delta breaks THIS
    test -- which names the other file -- instead of only breaking three assertions in a file
    the editor was told they would not have to touch."""
    delta = sw.THESIS_TRANSFER_DELTA
    ey = abs(sw.DEPLOYED['earnYield'])
    iq = abs(sw.DEPLOYED['incomeQuality'])
    ch = abs(sw.DEPLOYED['CycleHeat'])

    #  the three pins, in the order they appear in test_e2_weight_vector.py
    assert ey / iq == pytest.approx(1.1532, abs=5e-4), \
        'test_e2_weight_vector.py:260 pins ey/iq at 1.1532 -- update it with the delta'
    assert ey / ch == pytest.approx(1.3365, abs=5e-4), \
        'test_e2_weight_vector.py:267 pins ey/|CycleHeat| at 1.3365 -- update it too'
    post_d3 = (0.39 * sw.GENERAL_BUDGETS['P'] + sw.THESIS_TRANSFER['earnYield']) - iq
    assert post_d3 == pytest.approx(-0.008830, abs=5e-6), \
        'test_e2_weight_vector.py:318 pins the post-D3 margin at -0.008830 -- it is the ' \
        'pre-transfer margin MINUS the delta, so it moves one-for-one with it'

    #  ...and that last pin moves at TWICE the delta, which is the reason it cannot be left
    #  out of a re-level: the transfer takes `delta` OFF `earnYield` and puts it ON
    #  `incomeQuality`, so the MARGIN between them moves by 2 x delta.  That is what the
    #  reviewer measured -- 0.008 -> -0.012830 and 0.003 -> -0.002830, both exactly
    #  -0.008830 -/+ 2 x the change in delta.
    margin_before_transfer = 0.39 * sw.GENERAL_BUDGETS['P'] - (iq - delta)
    assert post_d3 == pytest.approx(margin_before_transfer - 2 * delta, abs=1e-15)
