"""Targeted tests for the Stage-1 red-flag veto (stage1_veto, CEO 2026-08-05).

WHAT IS PINNED HERE, and nothing else: the flag's DEFAULT-OFF no-op, the GENERAL-POOL-ONLY scope
(CEO 2026-08-07), the `<=1 of 8` fail definition at both boundaries, `k >= 1` ejection, the
per-pool report, and the refusal to run on a panel that lacks a flag column.  No pass rate and no panel measurement -- the veto has never
been run on real data (it cannot be: no saved pickle carries `uInterestCoverage`), and a test
that asserted a rate would be asserting a number nobody has measured.
"""

import numpy as np
import pandas as pd
import pytest

import stage1_veto as sv


def _panel(sources):
    """A minimal BoMetric-shaped panel.  `sources` = {ticker: {column: [8 newest-first values]}}.

    Any flag column not given is filled with a healthy passing value, so each test only states
    the flag it is about.
    """
    healthy = {'returnOnAssets': 0.05, 'CFOlessEarnings': 10.0, 'uCurrentRatio': 2.0,
               'netDebtToEBITDA': 0.75, 'uInterestCoverage': 5.0}
    rows = []
    for src, cols in sources.items():
        for i in range(sv.WINDOW_ROWS):
            r = {'source': src,
                 'date': pd.Timestamp('2026-03-31') - pd.DateOffset(months=3 * i)}
            for c, v in healthy.items():
                seq = cols.get(c, v)
                r[c] = seq[i] if isinstance(seq, (list, tuple)) else seq
            rows.append(r)
    return pd.DataFrame(rows)


def _scores(sources):
    return pd.DataFrame({'source': list(sources), 'score': range(len(sources), 0, -1)})


def test_flag_defaults_off_and_off_is_a_bit_identical_no_op():
    """DEFAULT OFF IS NON-NEGOTIABLE -- with the flag off nothing may change."""
    assert sv.ENABLED is False, (
        'stage1_veto.ENABLED must ship False. The veto must never enter the gate silently; '
        'turning it on is a visible one-line event.')
    src = {'BAD': {'uCurrentRatio': 0.5}, 'GOOD': {}}
    scores = _scores(src)
    kept, rep = sv.apply_veto(scores, _panel(src))
    assert kept is scores, 'with the flag off the input frame must be returned UNCHANGED'
    assert rep['enabled'] is False and rep['n_ejected'] == 0


def test_one_persistent_flag_ejects_and_a_clean_name_survives():
    src = {'BAD': {'uCurrentRatio': 0.5}, 'GOOD': {}}
    kept, rep = sv.apply_veto(_scores(src), _panel(src), enabled=True)
    assert list(kept['source']) == ['GOOD'], 'k >= 1: ONE persistent red flag ejects'
    assert rep['n_in'] == 2 and rep['n_ejected'] == 1 and rep['n_out'] == 1
    assert rep['by_flag'] == {'uCurrentRatio': 1}
    assert rep['ejected'] == ['BAD']


def test_fail_is_at_most_one_pass_of_eight_at_both_boundaries():
    """`<=1 of 8`, NOT `0 of 8` -- one bad vendor print must not eject a name on its own."""
    # 1 pass of 8 -> FAILS the flag -> ejected.
    one = {'X': {'returnOnAssets': [0.05] + [-0.01] * 7}}
    kept, _ = sv.apply_veto(_scores(one), _panel(one), enabled=True)
    assert list(kept['source']) == [], '1 of 8 is at the fail threshold and must eject'
    # 2 passes of 8 -> the flag holds -> survives.  This is the tolerance, stated at its edge.
    two = {'X': {'returnOnAssets': [0.05] * 2 + [-0.01] * 6}}
    kept, _ = sv.apply_veto(_scores(two), _panel(two), enabled=True)
    assert list(kept['source']) == ['X'], (
        '2 of 8 must SURVIVE -- the `<=1` definition exists to tolerate a single bad print, so '
        'tightening it to `<=2` here would silently change the veto\'s severity')


def test_every_flag_can_eject_on_its_own():
    for col, bad in (('returnOnAssets', -0.01), ('CFOlessEarnings', -10.0),
                     ('uCurrentRatio', 0.5), ('netDebtToEBITDA', -0.25),
                     ('uInterestCoverage', 0.5)):
        src = {'X': {col: bad}}
        kept, rep = sv.apply_veto(_scores(src), _panel(src), enabled=True)
        assert list(kept['source']) == [], '%s must be able to eject on its own' % col
        assert rep['by_flag'] == {col: 1}


def test_a_refused_row_is_a_non_pass_on_every_ADVERSE_or_MOOT_field():
    """*** BIT-IDENTICAL GUARD (C-15). ***  Four of the five fields rule a refused row as
    EVIDENCE, so `NaN is not a pass` still holds for them exactly as before -- the per-field
    evidence floor must not have softened the veto anywhere except the one BENIGN field."""
    for col in ('returnOnAssets', 'CFOlessEarnings', 'uCurrentRatio', 'netDebtToEBITDA'):
        assert sv.FIELD_EVIDENCE[col] == 'counts', (
            '%s was re-ruled as BENIGN. Each of these four was source-verified: its refusal '
            'channel is adverse or gated upstream, so abstaining would let the adverse case '
            'dodge the flag.' % col)
        nan_src = {'X': {col: np.nan}}
        kept, rep = sv.apply_veto(_scores(nan_src), _panel(nan_src), enabled=True)
        assert list(kept['source']) == [], 'NaN must not count as a pass for %s' % col
        assert rep['n_short_window'] == {}, (
            '%s must not ABSTAIN on refused rows -- a full 8-row window is 8 rows of '
            'evidence for an adverse/moot field' % col)


def test_netDebtToEBITDA_flag_reads_the_verdict_column_not_a_ratio():
    """The flag is `> 0` on the THREE-BRANCH VERDICT column -- the rule is not restated here.

    A net-cash admission emits +1.0 (passes) and a refused row emits NaN (fails), so the veto
    inherits the three-branch rule instead of re-implementing it.  If someone re-wrote this flag
    as `< 1` on a leverage ratio, the net-cash sentinel of +1.0 would start FAILING.
    """
    admitted = {'X': {'netDebtToEBITDA': 1.0}}        # the net-cash sentinel
    assert list(sv.apply_veto(_scores(admitted), _panel(admitted),
                             enabled=True)[0]['source']) == ['X']
    refused = {'X': {'netDebtToEBITDA': np.nan}}
    assert list(sv.apply_veto(_scores(refused), _panel(refused),
                              enabled=True)[0]['source']) == []


def test_missing_flag_column_raises_rather_than_vetoing_on_a_subset():
    src = {'X': {}}
    panel = _panel(src).drop(columns=['uInterestCoverage'])
    with pytest.raises(KeyError, match='missing'):
        sv.apply_veto(_scores(src), panel, enabled=True)


#  ---- A SHORT WINDOW MUST NEVER EJECT  (reviewer, 2026-08-05) ------------------------- #
def _short_panel(sources, n_rows):
    """The same panel shape but with only `n_rows` rows per source."""
    full = _panel(sources)
    return (full.sort_values(['source', 'date'], ascending=[True, False])
                .groupby('source', sort=False).head(n_rows).reset_index(drop=True))


def test_a_one_row_window_that_passes_its_only_row_is_not_ejected():
    """*** THE DEFECT.  `FAIL_MAX_PASSES = 1` is an ABSOLUTE count, so a source with ONE row
    that PASSED it scores `passes = 1 <= 1` and used to fail ALL FIVE flags at once -- a 100%
    pass rate read as five persistent red flags. ***"""
    src = {'YOUNG': {}}                      # one row, every flag healthy
    kept, rep = sv.apply_veto(_scores(src), _short_panel(src, 1), enabled=True)
    assert list(kept['source']) == ['YOUNG'], (
        'a source that passed 100% of the rows it has was ejected -- the fail threshold is '
        'being applied to a window too short to support it')
    assert rep['n_ejected'] == 0 and rep['by_flag'] == {}


def test_a_short_window_abstains_rather_than_failing_even_when_every_row_is_bad():
    """ABSTAIN IS NOT FAIL, AND IT IS NOT PASS-BECAUSE-CLEAN EITHER.  A 1-row source that
    FAILED its only row is still not ejected -- 3 months is not evidence of "essentially never
    passes over two years" -- and the abstention is REPORTED so nobody reads it as clean."""
    src = {'BAD1ROW': {'uCurrentRatio': 0.5}}
    kept, rep = sv.apply_veto(_scores(src), _short_panel(src, 1), enabled=True)
    assert list(kept['source']) == ['BAD1ROW'], (
        'a source must never be ejected BECAUSE its window is short')
    assert rep['n_short_window'] == {c: 1 for c in sv.FLAGS}, (
        'the abstention must be reported PER FLAG: "found clean" and "never evaluated" are '
        'different facts and a bare ejection count cannot tell them apart')
    assert rep['short_window'] == {'BAD1ROW': {c: 1 for c in sv.FLAGS}}
    assert sv.failed_flags(_short_panel(src, 1)) == {'BAD1ROW': []}, (
        'a short window must produce NO failed flags, not five')


def test_the_per_source_row_floor_is_DERIVED_and_bites_at_the_same_boundary():
    """*** C-15. ***  `MIN_WINDOW_ROWS` is DELETED: a short source has too few EVIDENCE rows on
    every flag, so it abstains on all of them -- the old per-source floor as a consequence of
    the per-flag one rather than a second rule sitting beside it.  Pinned at both sides of the
    boundary, so the deletion cannot have disabled the veto for everyone."""
    assert not hasattr(sv, 'MIN_WINDOW_ROWS'), (
        'MIN_WINDOW_ROWS was re-added. It is now DERIVED from the per-flag evidence floor; a '
        'second rule that merely agrees with the first is the pair that diverges silently.')
    src = {'X': {'uCurrentRatio': 0.5}}
    kept, rep = sv.apply_veto(_scores(src), _short_panel(src, sv.WINDOW_ROWS - 1),
                              enabled=True)
    assert list(kept['source']) == ['X'] and rep['n_short_window']['uCurrentRatio'] == 1, (
        'one row under the window must abstain')
    kept, rep = sv.apply_veto(_scores(src), _short_panel(src, sv.WINDOW_ROWS), enabled=True)
    assert list(kept['source']) == [] and rep['n_short_window'] == {}, (
        'AT the full window the veto must evaluate normally -- the change must not have '
        'disabled the veto for every source')


def test_abstention_is_reported_per_pool_not_panel_wide():
    """The report is per pool, so its abstentions must be too -- otherwise every one of the
    six pools repeats the whole panel's short-window names and the counts mean nothing.

    BOTH ARMS RUN ON `general` (2026-08-07): the veto is now general-pool-only, so an arm run
    under a cohort label would pass VACUOUSLY on the out-of-scope no-op and prove nothing about
    the pool-membership restriction this test exists for.  The two arms differ in the SCORE
    FRAME's membership, which is the actual variable."""
    src = {'A': {}, 'B': {}}
    panel = pd.concat([_short_panel({'A': {}}, 2), _panel({'B': {}})], ignore_index=True)
    _kept, rep = sv.apply_veto(_scores({'B': {}}), panel, pool_label='general', enabled=True)
    assert rep['short_window'] == {} and rep['n_short_window'] == {}, (
        'this pool does not contain A, so A\'s abstention must not appear in its report'
    )
    _kept, rep = sv.apply_veto(_scores(src), panel, pool_label='general', enabled=True)
    assert rep['short_window'] == {'A': {c: 2 for c in sv.FLAGS}}
    assert rep['n_short_window'] == {c: 1 for c in sv.FLAGS}


#  ---- THE BENIGN FIELD: A DEBT-FREE NAME MUST NOT BE EJECTED  (C-15, 2026-08-06) ------- #
def test_a_debt_free_name_abstains_on_interest_coverage_instead_of_being_ejected():
    """*** THE MEASURED DEFECT C-15 CLOSES. ***  `interestExpense == 0` is a DEBT-FREE name and
    `calcMetrics.interest_expense_positive` refuses the row, so the column is NaN.  Under the
    old blanket rule that read as "never covers its interest" and, at `EJECT_MIN_FLAGS = 1`,
    ejected 1,668 sources -- 21.5% of the universe -- FOR HAVING NO DEBT."""
    src = {'DEBTFREE': {'uInterestCoverage': np.nan}}
    kept, rep = sv.apply_veto(_scores(src), _panel(src), enabled=True)
    assert list(kept['source']) == ['DEBTFREE'], (
        'a name with no interest expense was ejected for not covering interest')
    assert rep['by_flag'] == {}
    assert rep['n_short_window'] == {'uInterestCoverage': 1}, (
        'the abstention must be VISIBLE and attributed to the flag -- otherwise a debt-free '
        'name is indistinguishable from one the veto found clean')


def test_the_benign_field_still_fails_on_a_full_window_of_real_evidence():
    """ABSTAIN IS NOT AMNESTY.  With 8 admissible rows the flag evaluates exactly as before,
    so a genuinely uncovered borrower is still ejected."""
    src = {'LEVERED': {'uInterestCoverage': 0.5}}
    assert list(sv.apply_veto(_scores(src), _panel(src), enabled=True)[0]['source']) == []


def test_the_benign_field_needs_a_FULL_window_of_admissible_rows_to_fail():
    """The floor is on EVIDENCE, not rows: 7 admissible rows of failure plus one refusal is 7
    rows of evidence, which is under the window, so the flag abstains.  This is the tension the
    old module docstring flagged as unresolved ("the floor counts ROWS, not NON-NaN rows")."""
    seven = [0.5] * 7 + [np.nan]
    src = {'X': {'uInterestCoverage': seven}}
    kept, rep = sv.apply_veto(_scores(src), _panel(src), enabled=True)
    assert list(kept['source']) == ['X'] and rep['n_short_window'] == {'uInterestCoverage': 1}
    #  ...and the same 7-of-8 shape on an ADVERSE field still FAILS, because there the refused
    #  row IS evidence.  The two branches must not be collapsed.
    src = {'X': {'uCurrentRatio': [0.5] * 7 + [np.nan]}}
    assert list(sv.apply_veto(_scores(src), _panel(src), enabled=True)[0]['source']) == []


def test_every_flag_has_an_explicit_evidence_ruling():
    """A flag added without a ruling would KeyError at evaluation time; caught here instead,
    and it forces the ruling to be a decision rather than a default."""
    assert set(sv.FIELD_EVIDENCE) == set(sv.FLAGS)
    assert set(sv.FIELD_EVIDENCE.values()) <= {'counts', 'not_evidence'}


def test_report_is_per_pool_and_counts_pre_veto_input():
    src = {'BAD': {'uCurrentRatio': 0.5}, 'GOOD': {}}
    _kept, rep = sv.apply_veto(_scores(src), _panel(src), pool_label='general', enabled=True)
    assert rep['pool'] == 'general'
    assert rep['n_in'] == 2, 'n_in must be the PRE-veto count, or a pool looks like it ejected 0'


#  ---- GENERAL POOL ONLY  (CEO, 2026-08-07) -------------------------------------------- #
def test_the_veto_runs_on_the_GENERAL_POOL_ALONE():
    """*** THE SCOPE RULING. ***  `uCurrentRatio > 1` and `netDebtToEBITDA` are STRUCTURALLY
    UNDEFINED on the leveraged-vehicle and bank cohorts -- a REIT carries mortgage debt at 5-8x
    EBITDA by design and holds no current assets -- so the veto asks the WRONG QUESTION there
    rather than a strict one.  Measured on the 2026-08-07 run: REIT 47 of 49 ejected (95.9%),
    BalanceSheetFin 71.9%, against 58.4% on the general pool.  NOT a threshold problem: no level
    makes a leverage bar a solvency reading on a mortgage vehicle."""
    assert tuple(sv.VETO_POOLS) == ('general',)
    src = {'BAD': {'uCurrentRatio': 0.5}, 'GOOD': {}}
    scores = _scores(src)
    for cohort in ('REIT', 'BalanceSheetFin', 'Mining', 'FinManager', 'InvestmentVehicle'):
        kept, rep = sv.apply_veto(scores, _panel(src), pool_label=cohort, enabled=True)
        assert kept is scores, (
            '%s was vetoed. Out of scope must be a BIT-IDENTICAL no-op -- the same object, not '
            'a filtered copy' % cohort)
        assert rep['applies'] is False and rep['n_ejected'] == 0
        assert rep['enabled'] is True, (
            'the flag was ON; `applies` and `enabled` are different facts and collapsing them '
            'would hide that the veto ran this run and DECLINED this pool')
        assert rep['not_applicable_reason'], (
            'a cohort with n_ejected == 0 and no reason reads as "the veto found it clean". '
            'The reason is what stops that')
    #  ...and the general pool is unaffected by the scoping.
    kept, rep = sv.apply_veto(scores, _panel(src), pool_label='general', enabled=True)
    assert list(kept['source']) == ['GOOD'] and rep['applies'] is True


def test_scope_is_reported_even_when_the_flag_is_OFF():
    """`applies` must be present on every report shape, or a reader has to know which of two
    keys to look for depending on a flag they cannot see from the CSV."""
    src = {'BAD': {'uCurrentRatio': 0.5}}
    _kept, rep = sv.apply_veto(_scores(src), _panel(src), pool_label='REIT')
    assert rep['enabled'] is False and rep['applies'] is False


def test_the_pools_override_lets_an_offline_AB_measure_a_cohort_without_mutating_globals():
    """How the per-cohort ejection rates in the module docstring were measured, and the same
    polarity as `enabled=`: research never mutates module state."""
    src = {'BAD': {'uCurrentRatio': 0.5}, 'GOOD': {}}
    kept, rep = sv.apply_veto(_scores(src), _panel(src), pool_label='REIT', enabled=True,
                              pools=('general', 'REIT'))
    assert list(kept['source']) == ['GOOD'] and rep['applies'] is True
    assert tuple(sv.VETO_POOLS) == ('general',), 'the override mutated module state'
