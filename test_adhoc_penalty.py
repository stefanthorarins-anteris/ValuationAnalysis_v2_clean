"""THE AD-HOC PENALTY BUCKET (adhoc_penalty, CEO 2026-08-10).

WHAT IS PINNED HERE, and nothing else: the FIXED weight and its non-proportionality, the sign
(points positive, penalty negative), the `Sigma|w| = 1` invariant the bucket must NOT perturb,
the fact that it is applied BEFORE the sort, its exclusion from the rank-of-ranks diagnostic,
and that every contribution is self-describing.  The behaviour of the CHECKS that feed it lives
in `test_stage1_veto.py`, beside the layer that raises them.

No network, no pickles, no API key.
"""
import numpy as np
import pandas as pd
import pytest

import adhoc_penalty as ap
import postBoRank as pbr
import scoringWeights as sw


def _book(*items):
    b = ap.PenaltyBook()
    for src, check, reason, pts in items:
        b.add(src, check, reason, pts, pool='general')
    return b


# --------------------------------------------------------------------------- #
#  A.  the weight -- a RULING, not a default                                  #
# --------------------------------------------------------------------------- #
def test_the_weight_is_FIXED_at_0_01_and_the_scaling_lives_in_the_AMOUNT():
    """The CEO was explicit about WHY the weight is not proportional: *"we will add to this
    bucket every time data is missing and thus scaling is there ... Just set the weight to
    0.01. We will do the scaling in the amount mostly."*

    So the property to hold is LINEARITY IN THE POINTS AT A CONSTANT RATE: five separate data
    problems must cost five times one, with nobody touching the weight.  A proportional or
    saturating weight would pass a bare `WEIGHT == 0.01` assertion and fail this one.
    """
    assert ap.WEIGHT == 0.01
    one = _book(('X', 'c', 'r', 1.0)).penalty_by_source()['X']
    five = _book(*[('X', 'c%d' % i, 'r', 1.0) for i in range(5)]).penalty_by_source()['X']
    assert one == pytest.approx(-0.01, abs=1e-15)
    assert five == pytest.approx(5 * one, abs=1e-15)
    #  and one contribution OF five points costs the same as five OF one -- the bucket adds
    #  amounts, it does not count events.
    assert _book(('X', 'c', 'r', 5.0)).penalty_by_source()['X'] == pytest.approx(five,
                                                                                abs=1e-15)


def test_points_are_POSITIVE_and_the_penalty_is_NEGATIVE_and_neither_can_flip():
    """The one sign in the layer.  A negative amount would REWARD missing data, which is the
    entire defect the bucket exists to remove, so it is refused LOUDLY rather than clamped --
    a caller computing a negative amount has a sign error it needs to see."""
    b = _book(('X', 'c', 'r', 3.0))
    assert b.points_by_source()['X'] > 0
    assert b.penalty_by_source()['X'] < 0
    for bad in (0, 0.0, -1, -0.5, np.nan):
        with pytest.raises(ValueError):
            ap.PenaltyBook().add('X', 'c', 'r', bad)


# --------------------------------------------------------------------------- #
#  B.  THE CRITICAL INVARIANT: the bucket is NOT in the weight vector         #
# --------------------------------------------------------------------------- #
def test_the_bucket_is_NOT_a_metric_and_does_NOT_perturb_sigma_abs_w():
    """*** THE INVARIANT THE CEO NAMED. ***  `scoringWeights._validate()` enforces
    `Sigma|w| = 1.000000` at import and every published AggScore range rests on it.  The
    penalty is an ABSOLUTE additive adjustment applied OUTSIDE the normalised weighted-z sum,
    so it must not be a member of the vector -- and importing this module must not have made
    it one."""
    assert sw.sum_abs(sw.DEPLOYED) == 1.0
    assert ap.PENALTY_COLUMN not in sw.METRIC_KEYS
    assert ap.PENALTY_COLUMN not in sw.DEPLOYED
    for label in sw.COHORT_LABELS:
        assert ap.PENALTY_COLUMN not in sw.COHORT_WEIGHTS[label]
        assert sw.sum_abs(sw.COHORT_WEIGHTS[label]) == pytest.approx(1.0, abs=1e-12)


def test_the_penalty_column_is_EXCLUDED_from_the_rank_of_ranks_diagnostic():
    """`rankOfRanks_diag` weights every column it ranks EQUALLY, so a 0.01-per-point penalty
    would carry the same say there as a whole metric -- and it is not a metric at all."""
    assert ap.PENALTY_COLUMN in pbr.ROR_EXCLUDE
    df = pd.DataFrame({'source': ['A', 'B'], 'earnYield': [1.0, 2.0],
                       ap.PENALTY_COLUMN: [-0.07, 0.0]})
    out = pbr.getRankOfRanks(df.copy())
    #  B beats A on the only ranked column, and the penalty (which favours B) must not have
    #  been counted a second time -- the diagnostic is unchanged by its presence.
    ref = pbr.getRankOfRanks(df.drop(columns=[ap.PENALTY_COLUMN]).copy())
    assert list(out[pbr.ROR_COLUMN]) == list(ref[pbr.ROR_COLUMN])


# --------------------------------------------------------------------------- #
#  C.  it is applied BEFORE the sort, which is the position the CEO named     #
# --------------------------------------------------------------------------- #
def test_the_penalty_is_summed_into_AggScore_and_therefore_REORDERS():
    """*"which is then added (which in effect lowers the score) before sorting/ranking takes
    place"*.  `getAggScore` sums every column and then sorts, so a penalty column in the frame
    IS applied before the sort -- pinned by making it decide the order."""
    df = pd.DataFrame({'source': ['A', 'B'], 'earnYield': [0.10, 0.09],
                       ap.PENALTY_COLUMN: [-0.05, 0.0]})
    out = pbr.getAggScore(df.copy())
    assert list(out['source']) == ['B', 'A'], (
        'the penalty must be inside the score BEFORE the sort; A leads on the metric and '
        'must lose the lead to its bucket')
    assert out.set_index('source')['AggScore']['A'] == pytest.approx(0.05, abs=1e-12)


def test_penalty_series_is_POSITIONAL_and_defaults_to_zero():
    """The Stage-2 frames are row-aligned by construction and an INDEX join is what has
    silently mis-paired frames in this pipeline before (see the missing-data fill report).  An
    unlisted source scores exactly 0.0 -- never NaN, which would poison the AggScore sum."""
    b = _book(('B', 'c', 'r', 2.0))
    s = b.penalty_series(['A', 'B', 'C'])
    assert list(s) == [0.0, pytest.approx(-0.02, abs=1e-15), 0.0]
    assert s.notna().all() and list(s.index) == [0, 1, 2]


# --------------------------------------------------------------------------- #
#  D.  self-describing, because that is half the design                       #
# --------------------------------------------------------------------------- #
def test_every_contribution_NAMES_its_check_and_its_reason():
    """A penalised name must be explainable from the shipped CSV without archaeology -- that
    is the stated requirement, and it is what makes this ONE bucket instead of a new gate per
    data defect."""
    b = _book(('X', 'stage1_veto:refused_rows', 'uInterestCoverage: 3 of 8 rows', 3.0),
              ('X', 'stage1_veto:short_panel', 'panel carries 6 of 8 rows', 2.0),
              ('Y', 'stage1_veto:refused_rows', 'uInterestCoverage: 1 of 8 rows', 1.0))
    df = b.itemised()
    assert set(df.columns) == set(ap.ITEM_COLUMNS)
    assert len(df) == 3
    assert (df['reason'].str.len() > 0).all() and (df['check'].str.len() > 0).all()
    #  worst-affected source first, so the file reads top-down
    assert list(df['source'])[:2] == ['X', 'X']
    assert b.points_by_source() == {'X': 5.0, 'Y': 1.0}
    assert b.penalty_by_source()['X'] == pytest.approx(-0.05, abs=1e-15)


def test_the_evidence_CSV_carries_BOTH_the_per_source_total_and_the_items(tmp_path):
    """One dated file at the repo ROOT with both sections -- the CEO reads "who and how much"
    and "why" together, and two files is how the second goes unread.  Root, not `output/`,
    for the reason recorded in `transfer_utils.EVIDENCE_DIR`."""
    b = _book(('X', 'chk', 'because', 3.0), ('Y', 'chk', 'because', 1.0))
    p = ap.write_evidence_csv(b, path=str(tmp_path / 'AdHocPenaltyBucket_2026-08-10.csv'))
    got = pd.read_csv(p)
    per_src = got[got['section'] == 'per_source'].set_index('source')
    assert per_src.loc['X', 'points'] == 3.0
    assert per_src.loc['X', 'penalty'] == pytest.approx(-0.03, abs=1e-9)
    assert set(got[got['section'] == 'per_item']['source']) == {'X', 'Y'}
    #  an EMPTY book still writes a file: "the check ran and charged nobody" and "the check
    #  did not run" are different facts, the same rule the other evidence CSVs follow.
    p2 = ap.write_evidence_csv(ap.PenaltyBook(),
                               path=str(tmp_path / 'AdHocPenaltyBucket_empty.csv'))
    assert p2 and pd.read_csv(p2).empty is not None


def test_the_evidence_file_is_MATCHED_by_the_transfer_manifest_and_by_NO_denylist():
    """The artifact only counts if it SHIPS -- and it is a new file, so nothing carried it
    before.  Both halves: a manifest pattern that matches, and none of the three denylist
    patterns (`*key*`, `*pem`, `fmpAPIkey.txt`)."""
    import fnmatch
    import os
    import transfer_utils as tu
    name = 'AdHocPenaltyBucket_2026-08-10.csv'
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Sbocker.py'),
              encoding='utf-8', errors='ignore') as f:
        src = f.read()
    assert "'AdHocPenaltyBucket_*.csv'" in src, 'no transfer pattern for the bucket evidence'
    assert fnmatch.fnmatch(name, 'AdHocPenaltyBucket_*.csv')
    assert not tu.is_denied(name)
    #  ...and it is written where `glob.glob` looks, which is the whole point of the move.
    assert tu.EVIDENCE_DIR == '.'


# --------------------------------------------------------------------------- #
#  E.  the corroborator reaches NO verdict                                    #
# --------------------------------------------------------------------------- #
def test_the_corroborator_changes_the_BUCKET_and_never_the_EJECTION_SET():
    """THE SAFETY PROPERTY OF THE S1 FIX.  `cdx_df` was threaded into a layer whose job is to
    REMOVE names, so the one thing that must be true is that it cannot move an ejection: the
    veto reads `bm_df` alone, and the raw panel only decides whether a refused row is billed.

    A join that silently dropped or duplicated a window row WOULD move an ejection, which is
    why this is asserted on both arms rather than argued from the call graph.
    """
    import stage1_veto as sv
    scores = pd.DataFrame({'source': ['X', 'Y'], 'score': [1.0, 2.0]})
    dates = [pd.Timestamp('2026-03-31') - pd.DateOffset(months=3 * i) for i in range(8)]
    healthy = {'returnOnAssets': 0.05, 'CFOlessEarnings': 10.0, 'uCurrentRatio': 2.0,
               'netDebtToEBITDA': 0.75}
    rows = []
    for src, ic in (('X', [np.nan] * 8), ('Y', [0.5] * 8)):
        for i in range(8):
            rows.append({'source': src, 'date': dates[i],
                         'uInterestCoverage': ic[i], **healthy})
    bm = pd.DataFrame(rows)
    #  a cdx carrying DUPLICATE (source, date) keys -- restated quarters are real in this
    #  panel, and an un-deduplicated join would multiply the veto window's rows
    cdx = pd.concat([pd.DataFrame({'source': [s] * 8, 'date': dates,
                                   'totalDebt': [500.0] * 8, 'revenue': [1.0] * 8})
                     for s in ('X', 'Y')] * 2, ignore_index=True)

    kept_a, rep_a = sv.apply_veto(scores, bm, enabled=True, verbose=False)
    kept_b, rep_b = sv.apply_veto(scores, bm, enabled=True, verbose=False, cdx_df=cdx)
    assert rep_a['ejected'] == rep_b['ejected']
    assert rep_a['by_flag'] == rep_b['by_flag']
    assert rep_a['n_short_window'] == rep_b['n_short_window']
    assert list(kept_a['source']) == list(kept_b['source'])
    #  ...and the bucket DOES differ, or the test would be proving nothing
    assert rep_b['adhoc_penalty_points'] == {'X': 8.0}
    assert rep_a['adhoc_penalty_points'] == {}
    assert rep_a['adhoc_penalty_uncorroborated'] == ['uInterestCoverage'], (
        'without the raw panel the bucket must report a MISSING MEASUREMENT, not silence')


def test_the_NEGATIVE_revenue_row_is_charged_and_the_ZERO_one_is_not():
    """*** reviewer R1, 2026-08-10: a regression I introduced, and the general lesson. ***

    `producerEbitdaPositive`'s guard is `revenue > 0`, so a refused row is NaN, ZERO, or
    NEGATIVE.  An intermediate version of the corroborator tested `missing` only, which
    silently let the negative rows through: `078130.KQ` runs 16.7bn, 15.0bn, 15.8bn, 18.4bn,
    18.7bn, **-1.62bn**, 20.1bn, 12.8bn, and `OCI.AS` has two interior negatives among
    positives.  A negative quarter is contra-revenue or a restatement -- a company with no
    operations does not report NEGATIVE sales -- so it is not a pre-production fact.

    THE STRUCTURAL LESSON, pinned because it governs every corroborator added later:
    `totalDebt` is INDEPENDENT of the channel it corroborates (`interestExpense <= 0`), so a
    single `> 0` test is complete.  `revenue` IS its guard's own input, so it can only ever
    witness one side of that guard and every outcome must be enumerated.
    """
    import stage1_veto as sv
    col, pred = sv.REFUSAL_CORROBORATOR['producerEbitdaPositive']
    assert (col, pred) == ('revenue', sv._CORROBORATE_MISSING_OR_NEGATIVE)
    win = pd.DataFrame({sv._CORROBORATOR_PREFIX + 'revenue':
                        [np.nan, 0.0, -1.62e9, 18.7e9]})
    got = list(sv._row_is_a_data_problem(win, 'producerEbitdaPositive'))
    assert got == [True, False, True, False], (
        'NaN and NEGATIVE are data problems; an exact ZERO is the pre-production fact the '
        'flag abstains for and must never be charged: %s' % got)
    #  The POSITIVE row reads False and that is right twice over: the guard PASSES it, so it
    #  is never a refused row in the first place and `_evaluate` masks it out anyway.  Pinned
    #  so nobody "fixes" the predicate into charging rows the flag computed successfully.


def test_an_unreachable_corroborator_reaches_the_SHIPPED_CSV_not_just_stdout(tmp_path):
    """*** reviewer R3. ***  `adhoc_penalty_uncorroborated` reached stdout and the saved
    `resdic`, and NOT the dated CSV -- the artifact a reader actually opens.  So a run whose
    corroborating column was unreachable produced a CSV byte-indistinguishable from a run on a
    clean panel, which is the same "charged nothing == found nothing" confusion the whole S1
    finding was about, one artifact downstream."""
    b = ap.PenaltyBook()
    b.declare_unmeasured('stage1_veto:corroborator_unavailable',
                         'could not judge refusals on uInterestCoverage: totalDebt absent',
                         pool='general')
    p = ap.write_evidence_csv(b, path=str(tmp_path / 'AdHocPenaltyBucket_x.csv'))
    got = pd.read_csv(p)
    row = got[got['section'] == 'NOT_MEASURED']
    assert len(row) == 1, 'the unmeasured state is missing from the CSV entirely'
    assert row['points'].iloc[0] == 0.0 and row['penalty'].iloc[0] == 0.0
    assert 'uInterestCoverage' in row['reason'].iloc[0]
    #  ...and it is written even though the book charged NOTHING -- which is the case that
    #  matters, because that is exactly when a reader would otherwise infer a clean panel.
    assert len(b) == 0
