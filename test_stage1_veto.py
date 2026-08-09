"""Targeted tests for the Stage-1 red-flag veto (stage1_veto, CEO 2026-08-05).

WHAT IS PINNED HERE, and nothing else: the flag's ON default, the FIVE-POOL scope with each
cohort's FLAG SET (CEO 2026-08-07, second ruling), the `<=1 of 8` fail definition at both
boundaries, `k >= 1` ejection, the per-pool report, the per-flag evidence floor, and the way a
panel that lacks a cohort's column degrades.  No pass rate and no panel measurement -- a test
that asserted a rate would be asserting a number nobody has re-measured since the flag sets.

*** THE SCOPE ASSERTIONS ARE PINS, AND THEY WERE RE-AUTHORED, NOT DELETED (2026-08-07). ***
Seven tests here asserted the PREVIOUS contract -- `VETO_POOLS == ('general',)`, the cohorts as
no-ops, a missing column RAISING -- and they failed the moment the CEO widened the scope.  That
is those tests doing their job: they exist so scope cannot widen without a visible decision.
They now pin the NEW contract, in the SAME shape and with the SAME purpose.  Each replacement was
verified to FAIL against the pre-widening module (`git show HEAD:stage1_veto.py`), because a pin
that passes under both contracts is not a pin -- and two tests in this repo have already been
caught pinning the defect they covered.
"""

import numpy as np
import pandas as pd
import pytest

import stage1_veto as sv

#  THE FIVE COHORT LABELS the carve-out emits, and the ONE that is out of scope.  Named once so a
#  pool added to `VETO_POOLS` without a decision here shows up as a failure rather than as a
#  cohort nobody remembered to test.
_IN_SCOPE = ('general', 'REIT', 'Mining', 'FinManager', 'BalanceSheetFin')
_OUT_OF_SCOPE = ('InvestmentVehicle',)


def _panel(sources):
    """A minimal BoMetric-shaped panel.  `sources` = {ticker: {column: [8 newest-first values]}}.

    Any flag column not given is filled with a healthy passing value, so each test only states
    the flag it is about.

    CARRIES THE FOUR COHORT COLUMNS TOO (2026-08-07).  No saved panel does yet -- they are built
    from `ebitda` / `cashAndCashEquivalents`, captured only from the 2026-08-05 fetch change on --
    so a cohort test run on a five-column panel would pass VACUOUSLY down the `missing_columns`
    no-op and prove nothing about the flag set it claims to test.  The degradation ITSELF is
    pinned separately, on a panel with the column explicitly dropped.
    """
    healthy = {'returnOnAssets': 0.05, 'CFOlessEarnings': 10.0, 'uCurrentRatio': 2.0,
               'netDebtToEBITDA': 0.75, 'uInterestCoverage': 5.0,
               'reitEbitdaInterestCoverage': 3.0, 'producerEbitdaPositive': 100.0,
               'cashRunwayOneYear': 250.0, 'equityPositive': 500.0}
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


def test_the_flag_ships_ON_and_the_scope_is_pinned_as_POOL_PLUS_FLAG_SET_PAIRS():
    """*** THE SCOPE PIN, RE-AUTHORED FOR THE FIVE-POOL CONTRACT (CEO, 2026-08-07). ***

    This test used to assert `VETO_POOLS == ('general',)` and it FAILED when the CEO widened the
    scope -- which is the test working, not breaking.  What it must keep doing is stop the scope
    widening SILENTLY, and after the flag sets a pool name alone no longer says what the veto
    does there: the same label with a different flag set is a different gate.  So the pin is now
    the PAIR, pool by pool, exhaustively.

    THE PAIRING IS THE POINT.  Asserting `VETO_POOLS` alone would let someone add
    `uCurrentRatio` back to REIT -- the flag measured at 97.0% ejected on that cohort -- without
    a single test moving.  Asserting the flag sets alone would let a pool be added to
    `VETO_POOLS` and gated by `pool_flags`'s general-set default.  Both directions are closed
    here, and the last assertion closes the third: a pool no longer in either list.
    """
    assert sv.ENABLED is True
    assert tuple(sv.VETO_POOLS) == ('general', 'REIT', 'Mining', 'FinManager',
                                    'BalanceSheetFin')
    #  Each cohort's set is the flags claimed DEFINED on its balance sheet -- every one a unity
    #  or a sign test, no cohort-specific threshold anywhere.
    assert {p: sorted(sv.pool_flags(p)) for p in sv.VETO_POOLS} == {
        'general': ['CFOlessEarnings', 'netDebtToEBITDA', 'returnOnAssets',
                    'uCurrentRatio', 'uInterestCoverage'],
        #  Gone: uCurrentRatio + netDebtToEBITDA (structurally undefined on a mortgage vehicle)
        #  AND returnOnAssets + CFOlessEarnings (depreciation of appreciating buildings /
        #  unrealised revaluation gains inside net income).  What replaces them is the one
        #  solvency question a rent-collector answers.
        'REIT': ['reitEbitdaInterestCoverage'],
        #  THE THREE DESIGNED FLAGS ONLY (reverted 2026-08-08, CEO).  `returnOnAssets`,
        #  `CFOlessEarnings` and `uInterestCoverage` were added by a dispatch brief, not by
        #  the design, and MEASURED on the 2026-08-07 CUR3K panel those three ALONE eject
        #  89 of 277 Basic-Materials sources (32.1%) -- returnOnAssets 80, uInterestCoverage
        #  45, CFOlessEarnings 2 -- against a design predicting 22 of 218 for the whole set.
        #  An exploration-stage miner has no earnings and no interest cover BY DEFINITION,
        #  so those two are structurally undefined on the pre-production half of this
        #  cohort: the REIT failure mode, reproduced.  The three that remain PARTITION the
        #  cohort instead (producer / explorer / balance-sheet floor).
        'Mining': ['cashRunwayOneYear', 'equityPositive', 'producerEbitdaPositive'],
        'FinManager': ['CFOlessEarnings', 'returnOnAssets', 'uInterestCoverage'],
        #  ONE flag: a bank that cannot earn on its own asset base fails at the only thing its
        #  asset base is for.  `ebitda / interestExpense` is deliberately NOT copied from REIT --
        #  it fails 40 of 125 here (RBC, TD, Scotiabank, BMO, ...) because interest expense is a
        #  bank's COST OF GOODS, not its debt service.
        'BalanceSheetFin': ['returnOnAssets'],
    }
    #  `general` must be the SAME OBJECT as `FLAGS`, not a copy, so the general pool cannot drift
    #  from the five-flag set the module docstring describes.
    assert sv.POOL_FLAGS['general'] is sv.FLAGS
    #  And the six carve-out cohorts are exhaustively accounted for: in scope, or ruled out BY
    #  NAME with a reason.  A seventh pool appearing in neither list fails here.
    assert set(_IN_SCOPE) | set(_OUT_OF_SCOPE) == set(sv.VETO_POOLS) | set(
        sv.NOT_APPLICABLE_REASONS)


def test_off_is_still_a_bit_identical_no_op():
    """The off path is no longer the default, so it is no longer exercised by accident -- which
    is exactly why it keeps a test.  `enabled=False` must return THE SAME OBJECT: an A/B arm and
    the fallback both depend on off meaning bit-identical, not merely equal."""
    src = {'BAD': {'uCurrentRatio': 0.5}, 'GOOD': {}}
    scores = _scores(src)
    kept, rep = sv.apply_veto(scores, _panel(src), enabled=False)
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


def test_a_missing_flag_column_DECLINES_THE_POOL_rather_than_raising():
    """*** RE-AUTHORED (2026-08-07).  This test used to assert `pytest.raises(KeyError)`. ***

    RAISING WAS RIGHT WHEN EVERY FLAG COLUMN EXISTED ON EVERY PANEL, and it stopped the veto
    running on a SUBSET of its flags -- a weaker gate shipping under the same name.  The cohort
    flag sets broke that premise: they name columns (`reitEbitdaInterestCoverage`,
    `producerEbitdaPositive`, `cashRunwayOneYear`, `equityPositive`) that only a fetch made after
    the 2026-08-05 capture change can build, so on today's panel REIT and Mining CANNOT be
    evaluated.  `postBo` wraps the veto in ONE guard, so a raise there would have taken every
    pool down together -- one stale column would have produced an entirely UN-VETOED run.

    THE SUBSTANTIVE HALF IS UNCHANGED AND IS STILL WHAT THIS TEST IS FOR: the veto still never
    gates on a subset of its flags.  It declines the POOL instead, loudly, and the decline must
    not be readable as a clean cohort -- which is the failure mode `n_ejected == 0` invites.
    """
    for pool, dropped in (('general', 'uInterestCoverage'),
                          ('REIT', 'reitEbitdaInterestCoverage')):
        src = {'BAD': {'uCurrentRatio': 0.5, 'reitEbitdaInterestCoverage': 0.5}, 'GOOD': {}}
        scores = _scores(src)
        panel = _panel(src).drop(columns=[dropped])
        kept, rep = sv.apply_veto(scores, panel, pool_label=pool, enabled=True)
        assert kept is scores, (
            '%s must be a BIT-IDENTICAL no-op when the panel cannot carry its gate -- the same '
            'object, not a filtered copy' % pool)
        assert rep['applies'] is False and rep['enabled'] is True, (
            'the flag was ON and the pool IS in scope; collapsing `applies` into `enabled` would '
            'hide that the veto ran this run and could not evaluate this pool')
        assert rep['missing_columns'] == [dropped], (
            'the MISSING COLUMN must be named. "we chose not to gate this cohort" and "this '
            'panel cannot carry the gate" are different facts and only one is fixed by '
            're-fetching -- a bare `applies=False` cannot tell them apart')
        assert rep['n_ejected'] == 0 and rep['by_flag'] == {}
        #  ...and the report must SAY it is not a clean bill of health.
        assert 'NOT thereby certified clean' in rep['not_applicable_reason']
        assert 'RE-FETCH' in rep['not_applicable_reason']


def test_an_out_of_scope_pool_and_a_stale_panel_are_DIFFERENT_reports():
    """The two `applies=False` channels must stay distinguishable, or "we declined this cohort"
    and "re-fetch and this cohort works" collapse into one unreadable state.  `missing_columns`
    is the discriminator and it is EMPTY on the out-of-scope path."""
    src = {'X': {}}
    _kept, out_of_scope = sv.apply_veto(_scores(src), _panel(src),
                                        pool_label='InvestmentVehicle', enabled=True)
    _kept, stale = sv.apply_veto(_scores(src),
                                 _panel(src).drop(columns=['reitEbitdaInterestCoverage']),
                                 pool_label='REIT', enabled=True)
    assert out_of_scope['applies'] is False and stale['applies'] is False
    assert out_of_scope['missing_columns'] == [], (
        'an out-of-scope pool has no missing column -- re-fetching would not put it in scope')
    assert stale['missing_columns'] == ['reitEbitdaInterestCoverage']
    assert out_of_scope['not_applicable_reason'] != stale['not_applicable_reason']


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


def test_every_flag_IN_EVERY_POOL_has_an_explicit_evidence_ruling():
    """*** RE-AUTHORED (2026-08-07).  This used to be `set(FIELD_EVIDENCE) == set(FLAGS)`. ***

    That was the whole flag universe when `FLAGS` was the whole flag universe.  With per-cohort
    sets the ruled set must cover EVERY pool's flags -- a cohort column added with no ruling
    would KeyError deep inside `_evaluate`, which `postBo`'s single guard turns into an entirely
    un-vetoed run: a missing RULING would present as a missing VETO.

    BOTH DIRECTIONS.  An UNRULED flag is the defect; an ORPHAN ruling is a ruling for a field
    nothing evaluates, which is how a stale rationale outlives the flag it described.
    """
    used = {c for f in sv.POOL_FLAGS.values() for c in f}
    assert used >= set(sv.FLAGS), 'the general set must be part of the used universe'
    assert set(sv.FIELD_EVIDENCE) == used, (
        'FIELD_EVIDENCE and the union of POOL_FLAGS have drifted: %s' %
        sorted(set(sv.FIELD_EVIDENCE) ^ used))
    assert set(sv.FIELD_EVIDENCE.values()) <= {'counts', 'not_evidence'}
    #  The two BENIGN rulings, pinned by name.  Both are the same measured defect in two fields
    #  (a refusal read as an adverse verdict), and re-ruling either to `counts` would eject a
    #  population for a property that is not a red flag -- debt-free names, and pre-production
    #  explorers.  The other seven are `counts` and must stay so.
    assert {k for k, v in sv.FIELD_EVIDENCE.items() if v == 'not_evidence'} == {
        'uInterestCoverage', 'reitEbitdaInterestCoverage', 'producerEbitdaPositive'}


def test_the_evidence_ruling_check_fires_AT_IMPORT_and_names_the_unruled_flag():
    """*** THE GUARD ITSELF, not just its current result. ***  The assertion above says today's
    dicts agree; this says the MODULE REFUSES TO LOAD if they ever stop agreeing.  That
    distinction is the whole value of an import-time check -- it is what stops an unruled flag
    reaching a 12-hour fetch.

    Exercised by loading the module's own source with ONE unruled flag injected into `FLAGS`
    (which `POOL_FLAGS['general']` IS), rather than by trusting the code to be there.
    """
    src = open(sv.__file__, encoding='utf-8', errors='replace').read()
    injected = src.replace("FLAGS = {\n",
                           "FLAGS = {\n    'aFlagNobodyRuledOn': lambda s: s > 0,\n", 1)
    assert injected != src, 'the FLAGS literal moved -- this test is no longer injecting anything'
    with pytest.raises(KeyError, match='aFlagNobodyRuledOn'):
        exec(compile(injected, sv.__file__, 'exec'), {'__name__': 'stage1_veto_injected'})


def test_report_is_per_pool_and_counts_pre_veto_input():
    src = {'BAD': {'uCurrentRatio': 0.5}, 'GOOD': {}}
    _kept, rep = sv.apply_veto(_scores(src), _panel(src), pool_label='general', enabled=True)
    assert rep['pool'] == 'general'
    assert rep['n_in'] == 2, 'n_in must be the PRE-veto count, or a pool looks like it ejected 0'


#  ---- FIVE POOLS, EACH ON ITS OWN FLAG SET  (CEO, 2026-08-07, second ruling) ----------- #
def test_the_veto_GATES_the_five_pools_and_InvestmentVehicle_ALONE_is_out_of_scope():
    """*** RE-AUTHORED (2026-08-07).  This used to assert every cohort was a no-op. ***

    The previous ruling scoped the veto to `general` because `uCurrentRatio > 1` and
    `netDebtToEBITDA` are STRUCTURALLY UNDEFINED on leveraged-vehicle and bank cohorts (REIT
    ejected 95.9%).  It was right about the DEFECT and wrong about the REMEDY: the flags were the
    problem, not the idea of vetoing a cohort.  Removing those two ALONE took REIT from 97.0% to
    23.9% and FinManager from 44.2% to 3.8%.

    So the assertion inverts -- the cohorts GATE now -- and the property it protects does not: a
    pool the veto declines must never be readable as a pool it found clean.  That is what the
    `InvestmentVehicle` arm is for, and it is the only cohort still on it.
    """
    #  Each cohort is ejected by a flag from ITS OWN set, which is what proves the per-pool
    #  lookup ran rather than the general set being applied five times.
    for cohort, bad_col in (('general', 'uCurrentRatio'),
                            ('REIT', 'reitEbitdaInterestCoverage'),
                            ('Mining', 'cashRunwayOneYear'),
                            ('FinManager', 'uInterestCoverage'),
                            ('BalanceSheetFin', 'returnOnAssets')):
        src = {'BAD': {bad_col: -0.5}, 'GOOD': {}}
        kept, rep = sv.apply_veto(_scores(src), _panel(src), pool_label=cohort, enabled=True)
        assert rep['applies'] is True and rep['missing_columns'] == []
        assert list(kept['source']) == ['GOOD'], (
            '%s must gate on %s -- it is in that pool\'s ruled flag set' % (cohort, bad_col))
        assert rep['by_flag'] == {bad_col: 1}

    #  ...and a flag that is NOT in a cohort's set must not gate it.  This is the subtractive
    #  half of the fix, and without it the test above passes on a veto that runs the general set
    #  everywhere.
    for cohort, dropped_col in (('REIT', 'uCurrentRatio'),
                                ('REIT', 'returnOnAssets'),
                                ('BalanceSheetFin', 'netDebtToEBITDA'),
                                ('FinManager', 'uCurrentRatio'),
                                ('Mining', 'netDebtToEBITDA')):
        src = {'X': {dropped_col: -0.5}}
        kept, rep = sv.apply_veto(_scores(src), _panel(src), pool_label=cohort, enabled=True)
        assert list(kept['source']) == ['X'], (
            '%s was ejected from %s by %s, which is NOT in that cohort\'s ruled flag set -- the '
            'general set is being applied to a cohort' % ('X', cohort, dropped_col))

    #  THE ONE POOL STILL OUT OF SCOPE.  Not for the structurally-undefined reason: n = 15 and
    #  nothing in it fails the statutory asset-coverage test, so there is no ejection to make.
    src = {'BAD': {'uCurrentRatio': 0.5}, 'GOOD': {}}
    scores = _scores(src)
    for cohort in _OUT_OF_SCOPE:
        kept, rep = sv.apply_veto(scores, _panel(src), pool_label=cohort, enabled=True)
        assert kept is scores, (
            '%s was vetoed. Out of scope must be a BIT-IDENTICAL no-op -- the same object, not '
            'a filtered copy' % cohort)
        assert rep['applies'] is False and rep['n_ejected'] == 0
        assert rep['enabled'] is True, (
            'the flag was ON; `applies` and `enabled` are different facts and collapsing them '
            'would hide that the veto ran this run and DECLINED this pool')
        assert 'NOT thereby certified clean' in (rep['not_applicable_reason'] or ''), (
            'a cohort with n_ejected == 0 and no reason reads as "the veto found it clean". '
            'The reason is what stops that')
        assert 'ASC 946' in rep['not_applicable_reason'], (
            'the reason must be THIS pool\'s, not a borrowed REIT/bank rationale -- that '
            'substitution is the failure the per-pool split exists to stop')


def test_InvestmentVehicle_is_a_no_op_ON_THE_LIVE_CALL_with_no_enabled_argument():
    """*** RE-AUTHORED (2026-08-07).  This used to run all five cohorts down the no-op path. ***

    THE CALL SHAPE IS THE POINT AND IT IS UNCHANGED.  Every other scope test passes
    `enabled=True` explicitly; `postBo` does NOT -- it calls `apply_veto(cs, bmdf,
    pool_label=lab)` and takes the module default.  So the module default is the only thing
    standing between a cohort and the gate on the LIVE path, and this test runs that exact shape.

    What changed is which cohorts must come back untouched: `InvestmentVehicle` alone.  The other
    four are asserted here to GATE on the same bare call, because a scope widening that somehow
    only took effect when `enabled=` was passed explicitly would be a live/offline divergence.
    """
    src = {'BAD': {'uCurrentRatio': 0.5, 'reitEbitdaInterestCoverage': 0.5}, 'GOOD': {}}
    scores = _scores(src)
    for cohort in _OUT_OF_SCOPE:
        kept, rep = sv.apply_veto(scores, _panel(src), pool_label=cohort)
        assert kept is scores, (
            '%s was gated on the LIVE call shape. With ENABLED=True the module default reaches '
            'every pool, and only VETO_POOLS stops it -- the same object must come back' % cohort)
        assert rep['enabled'] is True and rep['applies'] is False and rep['n_ejected'] == 0
    for cohort in _IN_SCOPE:
        kept, rep = sv.apply_veto(scores, _panel(src), pool_label=cohort)
        assert rep['enabled'] is True and rep['applies'] is True, (
            '%s did not gate on the bare live call -- the scope must not depend on an explicit '
            '`enabled=` argument' % cohort)
        assert kept is not scores


def test_scope_is_reported_even_when_the_flag_is_OFF():
    """`applies` must be present on every report shape, or a reader has to know which of two
    keys to look for depending on a flag they cannot see from the CSV.

    RE-AUTHORED ONLY IN ITS ARMS (2026-08-07): `REIT` is now IN scope, so it no longer
    demonstrates the out-of-scope shape.  BOTH shapes are now pinned -- an in-scope pool with the
    flag off (`enabled=False`, `applies=True`) and an out-of-scope one (both False) -- because
    the two fields being INDEPENDENT is the property, and one arm cannot show that.
    """
    src = {'BAD': {'uCurrentRatio': 0.5}}
    _kept, rep = sv.apply_veto(_scores(src), _panel(src), pool_label='REIT', enabled=False)
    assert rep['enabled'] is False and rep['applies'] is True, (
        'REIT is in scope; with the flag OFF that must read as "would have applied, did not run"')
    _kept, rep = sv.apply_veto(_scores(src), _panel(src), pool_label='InvestmentVehicle',
                               enabled=False)
    assert rep['enabled'] is False and rep['applies'] is False


def test_the_pools_override_lets_an_offline_AB_measure_a_cohort_without_mutating_globals():
    """How the per-cohort ejection rates in the module docstring were measured, and the same
    polarity as `enabled=`: research never mutates module state.

    RE-AUTHORED (2026-08-07) TO USE THE POOL THAT IS STILL OUT OF SCOPE.  With `REIT` now in
    `VETO_POOLS` the old arm proved nothing -- the override would have been a no-op and the test
    would have passed without exercising it.  `InvestmentVehicle` is the only cohort that can
    still show an override putting a pool IN scope.
    """
    src = {'BAD': {'uCurrentRatio': 0.5}, 'GOOD': {}}
    kept, rep = sv.apply_veto(_scores(src), _panel(src), pool_label='InvestmentVehicle',
                              enabled=True, pools=('general', 'InvestmentVehicle'))
    assert list(kept['source']) == ['GOOD'] and rep['applies'] is True
    assert tuple(sv.VETO_POOLS) == ('general', 'REIT', 'Mining', 'FinManager',
                                    'BalanceSheetFin'), 'the override mutated module state'


def test_an_UNLISTED_pool_defaults_to_the_FULL_general_set_never_an_empty_one():
    """*** THE `pool_flags` DEFAULT, CONFIRMED AS A PIN (CEO review question, 2026-08-07). ***

    An unlisted pool must never be able to look CLEAN by accident.  `{}` would be the tidier
    default and it is the wrong one: `_evaluate` over an empty flag set fails nobody, so the pool
    would report `n_ejected = 0` with `applies=True` and no missing columns -- indistinguishable
    from a cohort the veto gated on five flags and found healthy.  The general set is the
    conservative default: a pool with no ruling is gated by SOMETHING, and if the general flags
    are the wrong question there, the ejections are loud enough to be noticed and the pool
    belongs OUT of `VETO_POOLS` rather than in it with no flags.

    The pool STILL has to be in `VETO_POOLS` to be gated at all -- this default decides what
    happens to a pool someone put in scope and forgot to rule on, which is the realistic mistake.
    """
    assert sv.pool_flags('APoolNobodyRuledOn') is sv.FLAGS, (
        'an unruled pool must fall back to the FULL general set. An empty dict ejects nobody and '
        'reports a clean cohort -- silence that reads as a pass')
    src = {'BAD': {'uCurrentRatio': 0.5}, 'GOOD': {}}
    kept, rep = sv.apply_veto(_scores(src), _panel(src), pool_label='APoolNobodyRuledOn',
                              enabled=True, pools=('APoolNobodyRuledOn',))
    assert list(kept['source']) == ['GOOD'] and rep['n_ejected'] == 1, (
        'an unruled pool that was put in scope was gated by an empty flag set and came back '
        'clean')


#  ---- THE VETO COLUMN CHANNEL: COMPUTED, CARRIED, NEVER SCORED  (CEO, 2026-08-07) ------ #
def test_a_veto_column_can_NEVER_become_a_scoring_criterion():
    """*** THE PROPERTY THE SEPARATE CHANNEL EXISTS FOR. ***

    The four cohort columns were nearly declared in `BoMetric_special_dict`, which is where a
    formula-based Stage-1 column belongs.  Every entry there carries a `Tier` and a `Sign` and
    `calcScore.simpleScore_fromDict` hands it to `calcByTier` -- so that would have added FOUR
    WEIGHTED CRITERIA TO EVERY POOL'S STAGE-1 SCORE, the general pool included.  A veto is not a
    scoring change and nobody ruled for one.

    Pinned three ways, because one is not enough: the KEY SETS are disjoint (against the PREFIXED
    column names the base/mean/unity/diff forms actually emit), no veto entry carries a Tier or a
    Sign, and the veto dict is NOT in `getDicts`'s return tuple -- which is what every scoring
    caller unpacks.
    """
    import createDicts as cdic

    veto = set(cdic.getVetoDict())
    assert veto == {'reitEbitdaInterestCoverage', 'producerEbitdaPositive',
                    'cashRunwayOneYear', 'equityPositive'}
    (_pre, calc, base, mean, diff, unity, special) = cdic.getDicts()
    scored = set(special)
    for key, spec in calc.items():
        for o in spec['Operation']:
            scored.add(key if o == 'n' else o + key[0].upper() + key[1:])
    assert veto & scored == set(), (
        'veto column(s) %s are also Stage-1 scoring criteria -- they would be weighted into '
        "every pool's score" % sorted(veto & scored))
    for k, spec in cdic.getVetoDict().items():
        assert 'Tier' not in spec and 'Sign' not in spec, (
            '%s declares a Tier/Sign. Veto columns are never scored, so that is either dead '
            'weight or an attempt to score one' % k)
    for d in (base, mean, diff, unity, special):
        assert set(d) & veto == set()
    #  The dict is not reachable through the tuple every scoring caller unpacks, so there is no
    #  unpack site at which it could be mistaken for `BoMetric_special_dict`.
    assert all(cdic.BoMetric_veto_dict is not d for d in cdic.getDicts()), (
        "the veto dict is inside getDicts()'s tuple")


def test_the_panel_SCHEMA_carries_every_veto_column():
    """`stage1_veto.missing_columns` reads the panel's SCHEMA to decide whether a cohort can be
    gated at all, so a column computed at fetch time but absent from `initBoMetric_fromDict`'s
    column list would make every cohort permanently un-vetoable -- and it would report as a
    stale panel, i.e. as something a re-fetch fixes, forever."""
    import utils
    import createDicts as cdic

    cols = list(utils.initBoMetric_fromDict()['BoMetric_df'].columns)
    assert set(cdic.getVetoDict()) <= set(cols)
    assert cols[-1] == 'source', 'the veto columns must be appended BEFORE `source`'
    #  Every flag any pool evaluates must be a real panel column, or that pool is dead on arrival.
    used = {c for f in sv.POOL_FLAGS.values() for c in f}
    assert used <= set(cols), 'flag(s) with no panel column: %s' % sorted(used - set(cols))


def test_each_veto_columns_ADMISSIBILITY_GATE_lives_in_the_column():
    """*** SIGN-SAFETY, AND WHY THE GATE IS NOT IN `stage1_veto`. ***

    This project fixed eight criteria where a threshold written for a positive quantity was
    AUTO-SATISFIED once the quantity went negative.  The defence is that an inadmissible row
    arrives at the veto as NaN, so no flag condition can invert -- which requires the gate to run
    where the column is BUILT, not where it is tested.

    Exercised on the arithmetic, per column: the refused row must be NaN, and the admitted row
    must carry the quantity the flag's bar is stated on.
    """
    import calcMetrics as cm
    import createDicts as cdic

    #  row 0 admissible, row 1 inadmissible for whichever column is under test.
    raw = pd.DataFrame({
        'ebitda':                               [100.0, -5.0],
        'interestExpense':                      [50.0, 0.0],    # 0 == debt-free -> refused
        'revenue':                              [1000.0, 0.0],  # 0 == pre-production -> refused
        'cashAndCashEquivalents':               [400.0, 400.0],
        'netCashProvidedByOperatingActivities': [-50.0, -200.0],
        'totalStockholdersEquity':              [500.0, -1.0],
    })
    got = {}
    for key, spec in cdic.getVetoDict().items():
        got[key] = list(cm.calc_veto(raw, key, rpy=4, guard=spec.get('Guard'))[key])

    #  ebitda / interestExpense; the ONE ratio, so the ONE column that could invert.  The
    #  debt-free row is REFUSED rather than reading as "cannot cover its interest".
    assert got['reitEbitdaInterestCoverage'][0] == 2.0
    assert np.isnan(got['reitEbitdaInterestCoverage'][1]), (
        'a name with no interest expense produced a coverage NUMBER -- this is the measured '
        'defect that ejected 1,668 sources for having no debt, in a new field')
    #  EBITDA itself, tested `> 0`.  The pre-revenue explorer is REFUSED even though its EBITDA
    #  is negative -- the gate must fire BEFORE the sign is read, or the whole exploration half
    #  of the Mining cohort is ejected for being explorers.
    assert got['producerEbitdaPositive'][0] == 100.0
    assert np.isnan(got['producerEbitdaPositive'][1]), (
        'a zero-revenue explorer was handed a negative EBITDA verdict rather than being refused')
    #  cash + CFO x rpy.  A sum: no denominator, nothing to invert, no guard.
    assert got['cashRunwayOneYear'] == [400 - 50 * 4, 400 - 200 * 4]
    #  ...and `rpy` is what makes the horizon TWELVE MONTHS for a semi-annual filer too, rather
    #  than twelve for one filer and six for another.  This is the statutory IAS 1.25 /
    #  ASC 205-40 going-concern horizon, so the frequency correction is part of the bar.
    semi = list(cm.calc_veto(raw, 'cashRunwayOneYear', rpy=2)['cashRunwayOneYear'])
    assert semi == [400 - 50 * 2, 400 - 200 * 2], (
        'the runway was not frequency-corrected -- a semi-annual filer would be assessed over '
        'SIX months against a twelve-month statutory bar')
    #  A level, always admissible -- `totalStockholdersEquity` is never absent and a degenerate
    #  one is adverse on any reading.
    assert got['equityPositive'] == [500.0, -1.0]
    assert cdic.getVetoDict()['equityPositive'].get('Guard') is None
    assert cdic.getVetoDict()['cashRunwayOneYear'].get('Guard') is None


def test_an_unknown_veto_key_RAISES_rather_than_becoming_an_all_NaN_column():
    """The `calc_special` failure mode, in the veto channel and worse.  An unrecognised key used
    to fall through every branch and return an EMPTY frame, which the caller wrote into the panel
    as an all-NaN column.  For a SCORING column that is pool-neutral; for a VETO column it is
    worse -- an all-NaN column does not present as MISSING (so `missing_columns` stays empty and
    the pool reports `applies=True`), it presents as a cohort that ABSTAINED on everything, i.e.
    as a veto that ran and found nothing."""
    import calcMetrics as cm

    with pytest.raises(KeyError, match='calc_veto'):
        cm.calc_veto(pd.DataFrame({'ebitda': [1.0]}), 'notAVetoColumn')


def test_an_absent_RAW_INPUT_omits_the_veto_column_rather_than_emitting_an_all_NaN_one():
    """*** THE QUIET FAILURE THIS CHANNEL MUST NOT HAVE. ***

    `ebitda` and `cashAndCashEquivalents` are capture-only additions from 2026-08-05, so the
    OFFLINE rebuild paths (`baseline_tools/panel_upgrade`, `dead_merge`) can legitimately be
    handed a saved `cdx_df` that predates them.

    AN ALL-NaN COLUMN WOULD BE THE WORST AVAILABLE ANSWER, and it is the one a naive
    `pd.to_numeric(df['ebitda'])` gives.  The column would be PRESENT, so
    `stage1_veto.missing_columns` finds nothing missing, the pool reports `applies = True`, every
    flag abstains for want of evidence, and the cohort comes back with ZERO EJECTIONS -- a veto
    that COULD NOT RUN, presenting as one that ran and found the cohort clean.  Omitting the
    column instead routes it to `_STALE_PANEL_NOT_APPLICABLE`, which declines that pool by name
    and says RE-FETCH.

    Asserted end to end, because the property is a JOINT one: the builder must drop the column
    AND the veto must read that absence as a decline.
    """
    import calcMetrics as cm

    assert cm.veto_missing_inputs(pd.DataFrame({'ebitda': [1.0]}),
                                  'reitEbitdaInterestCoverage') == ['interestExpense']
    assert cm.veto_missing_inputs(pd.DataFrame({'totalStockholdersEquity': [1.0]}),
                                  'equityPositive') == []
    #  ...and the downstream half: a panel WITHOUT the column declines the pool, a panel WITH it
    #  all-NaN would NOT -- which is exactly why the builder must not emit the latter.
    src = {'X': {}}
    _kept, absent = sv.apply_veto(_scores(src),
                                  _panel(src).drop(columns=['reitEbitdaInterestCoverage']),
                                  pool_label='REIT', enabled=True)
    _kept, all_nan = sv.apply_veto(_scores(src),
                                   _panel(src).assign(reitEbitdaInterestCoverage=np.nan),
                                   pool_label='REIT', enabled=True)
    assert absent['applies'] is False and absent['missing_columns'] == [
        'reitEbitdaInterestCoverage']
    assert all_nan['applies'] is True and all_nan['n_ejected'] == 0, (
        'this is the SILENT state the builder must never produce: the pool ran, ejected nobody, '
        'and reported no missing column')


# --------------------------------------------------------------------------- #
#  THE DESIGNED FLAGS *ARE* BACKTESTABLE, AND THIS IS THE BACKTEST (2026-08-09) #
#                                                                              #
#  POOL_FLAGS used to carry "NOT BACKTESTABLE ... NO SAVED PANEL CARRIES THEM   #
#  ... the 22-of-218 prediction CANNOT be verified offline".  It conflated the  #
#  DERIVED columns (genuinely absent from every saved BoMetric_df) with the RAW #
#  INPUTS (all six present on the 2026-08-07 CUR3K panel's cdx_df).  A missing   #
#  derived column is a REBUILD, not a re-fetch.  The claim survived because      #
#  nobody tried it.                                                              #
# --------------------------------------------------------------------------- #
_CUR3K_PANEL = ('panels_2026-08-07/Bometric_dic-fmp_stock_CUR3K_all_2026-08-07_'
                'len2613_manelim0_fails445.pickle')

#  MEASURED 2026-08-09 by the rebuild below.  Hardcoded, NOT recomputed from the code under
#  test: a test that derives its expectation from the thing it is testing cannot detect that
#  thing changing.
_EXPECTED = {
    'Mining': {'sector': 'Basic Materials', 'n_sources': 277, 'n_failed': 30,
               'per_flag': {'producerEbitdaPositive': 14, 'cashRunwayOneYear': 11,
                            'equityPositive': 6},
               'n_sources_with_abstention': 63},
    'REIT':   {'sector': 'Real Estate', 'n_sources': 76, 'n_failed': 3,
               'per_flag': {'reitEbitdaInterestCoverage': 3},
               'n_sources_with_abstention': 21},
}


def _load_cur3k():
    """The panel and the sector map, or an explicit SKIP naming what is missing.

    Both are .gitignore'd, so they exist on the machine the measurement was taken on and
    not necessarily elsewhere.  An EXPLICIT skip, never a bare `return` -- register C4: a
    bail-out that reports PASS having asserted nothing is how a gate stops being one.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    panel, sectors = os.path.join(here, _CUR3K_PANEL), os.path.join(
        here, 'sectorsdic_fmp.pickle')
    for p in (panel, sectors):
        if not os.path.exists(p):
            pytest.skip('offline backtest input absent on this machine: %s' % p)
    d = pd.read_pickle(panel)
    return d['cdx_df'], d['BoMetric_df'], pd.read_pickle(sectors)


def _rebuild_veto_panel(cdx, bm, sources):
    """The veto columns, rebuilt with the PRODUCTION formulas over the EXACT row set the
    saved panel kept.

    Row set: `getData_fmp` drops the OLDEST `rpy` rows of each source's metric frame
    (verified on this panel -- 2,592 of 2,613 sources have a later `min(date)` in
    BoMetric_df than in cdx_df, and NOT ONE has an earlier `max(date)`), so taking the
    newest `len(BoMetric_df[source])` rows of `cdx_df[source]` reproduces it without
    assuming a stored row order.  Every veto formula is row-local, so order cannot change a
    value -- only which rows survive.
    """
    import calcMetrics as cm
    import createDicts as cdic
    import reporting_period as rp
    rpymap = rp.rows_per_year_by_source(cdx)
    nbm = bm.groupby('source').size().to_dict()
    vetodict = cdic.getVetoDict()
    frames = []
    for s in sources:
        g = cdx[cdx['source'] == s].sort_values('date', ascending=False)
        g = g.head(nbm.get(s, max(len(g) - rpymap.get(s, 4), 0)))
        if not len(g):
            continue
        out = pd.DataFrame({'source': s, 'date': g['date'].values})
        for key, spec in vetodict.items():
            if cm.veto_missing_inputs(g, key):
                continue
            out[key] = cm.calc_veto(g, key, rpy=rpymap.get(s, 4),
                                    guard=spec.get('Guard'))[key].values
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def test_every_RAW_INPUT_the_designed_flags_need_is_ON_the_saved_panel():
    """The premise the whole correction rests on, checked directly: the DERIVED columns are
    absent (that part of the old comment was right) and every RAW input is present (that
    part was wrong)."""
    import calcMetrics as cm
    cdx, bm, _sec = _load_cur3k()
    for key in cm._VETO_KEYS:
        assert key not in bm.columns and key not in cdx.columns, (
            '%r is on the saved panel -- the "derived columns are absent" half of the '
            'correction no longer holds' % key)
        assert cm.veto_missing_inputs(cdx, key) == [], (
            '%r is missing raw inputs %r from cdx_df -- it really would not be '
            'backtestable' % (key, cm.veto_missing_inputs(cdx, key)))


@pytest.mark.parametrize('pool', sorted(_EXPECTED))
def test_the_designed_cohort_flags_reproduce_their_MEASURED_ejection_rate(pool):
    """The measurement recorded in `POOL_FLAGS`, made re-runnable.

    Mining lands at 10.8% against a design that predicted 10.1%, so the 32.1% overshoot
    really was the three reverted additions and not the designed set -- that is now a
    measurement rather than an inference from the arithmetic.
    """
    exp = _EXPECTED[pool]
    cdx, bm, sec = _load_cur3k()
    sources = sorted(set(cdx['source'].unique()) & set(sec[exp['sector']]))
    assert len(sources) == exp['n_sources'], (
        '%s cohort is %d sources, expected %d -- the denominator moved, so the recorded '
        'rate no longer describes this panel' % (pool, len(sources), exp['n_sources']))

    panel = _rebuild_veto_panel(cdx, bm, sources)
    failed, abstained = sv._evaluate(panel, sv.POOL_FLAGS[pool])
    bad = {s: f for s, f in failed.items() if f}
    per_flag = {}
    for flags in bad.values():
        for c in flags:
            per_flag[c] = per_flag.get(c, 0) + 1

    assert len(bad) == exp['n_failed'], (
        '%s ejects %d of %d, recorded as %d' % (pool, len(bad), len(sources),
                                                exp['n_failed']))
    assert per_flag == exp['per_flag']
    assert len(abstained) == exp['n_sources_with_abstention'], (
        'abstention count moved -- the designed flags PARTITION the cohort by construction, '
        'so who abstains is part of the claim, not a detail')


def test_the_RECORDED_numbers_and_the_MEASURED_numbers_cannot_drift_apart():
    """The comment in `POOL_FLAGS` is where a reader looks; `_EXPECTED` above is what the
    backtest asserts.  If those two disagree, the comment is a stale claim again -- which
    is the entire defect class this correction belongs to.

    NOT a "the sentence was deleted" test: the corrected comment deliberately QUOTES the
    refuted claim so the correction is legible, so a substring ban would fail on the fix
    itself.  What is pinned is that the NUMBERS are present and agree.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, 'stage1_veto.py'), encoding='utf-8') as fh:
        src = fh.read()
    m, r = _EXPECTED['Mining'], _EXPECTED['REIT']
    for token in ('%d of %d' % (m['n_failed'], m['n_sources']),
                  '%d of %d' % (r['n_failed'], r['n_sources']),
                  '10.8%', '3.9%',
                  '`producerEbitdaPositive` 14', '`cashRunwayOneYear` 11',
                  '`equityPositive` 6'):
        assert token in src, (
            'POOL_FLAGS does not record %r -- the measured result and the comment a '
            'reader trusts have drifted apart' % token)
    #  and the file it shares the defect with carries the corrected framing too
    with open(os.path.join(here, 'calcMetrics.py'), encoding='utf-8') as fh:
        cm_src = fh.read()
    assert 'REBUILD, not a re-fetch' in cm_src, (
        'calcMetrics still frames the missing derived columns as a re-fetch -- correcting '
        'one of the two files and leaving the other is how a stale claim survives a pass')
