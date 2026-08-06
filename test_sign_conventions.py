"""SIGN CONVENTIONS AND DOMAIN GUARDS -- the pins for the 2026-08-04 sign-inversion fix.

WHAT THIS FILE EXISTS TO PREVENT, stated first because it is the reason the tests look the way
they do.  A Stage-1 criterion's SIGN lives in TWO registries (`BoMetric_Calc_dict`, which builds
the column schema, and the four operational dicts, which drive the arithmetic and the score),
and its DIRECTION is only realised at a THIRD place -- `calcScore.calcByTier`, which evaluates
`Sign * value > 0`.  A sign fix applied at two of those three places changes the score with no
error anywhere, and inverting a consumer that was already correct by double negation is this
project's worst historical bug.

So the tests below come in three layers, deliberately:

  1. REGISTRY AGREEMENT -- the two declarations cannot drift apart.
  2. DECLARED SPEC -- each fixed criterion's (Upper, Lower, Sign, Guard) is pinned literally,
     so a silent re-edit fails loudly and a deliberate one has to update a test that says why.
  3. BEHAVIOUR AT THE CONSUMER -- for each criterion, a synthetic row carrying the ADVERSE
     condition is pushed through the REAL `calcByTier` and must FAIL, and (where there is one)
     a row carrying the GENUINE-but-superficially-similar condition must still PASS.  This is
     the layer that actually catches a wrong sign: layers 1 and 2 only check that the code says
     what it says, layer 3 checks that what it says is right.

Run: pytest test_sign_conventions.py -v
"""

import numpy as np
import pandas as pd
import pytest

import calcMetrics as cm
import calcScore as cs
import createDicts as cdic
import getData_fmp as gdf
import reporting_period as rp
import stage2_metrics as sm
import utils


# --------------------------------------------------------------------------- #
#  1. THE TWO SIGN REGISTRIES CANNOT DRIFT                                    #
# --------------------------------------------------------------------------- #
_OP_FOR_KIND = {'base': 'n', 'mean': 'm', 'diff': 'd', 'unity': 'u'}


def _operational_signs():
    """{metric key: {operation letter: Sign}} over the four operational dicts."""
    base, mean, diff, unity, _special = cdic.getBaseMeanDiffUnitySpecialDicts()
    out = {}
    for kind, d in (('base', base), ('mean', mean), ('diff', diff), ('unity', unity)):
        for k, v in d.items():
            out.setdefault(k, {})[_OP_FOR_KIND[kind]] = v['Sign']
    return out


def test_sign_registries_agree():
    """`BoMetric_Calc_dict` (schema + a second copy of Sign) vs the operational dicts.

    THE FAILURE THIS CATCHES IS SILENT.  `utils.initBoMetric_fromDict` builds BoMetric_df's
    COLUMNS from BoMetric_Calc_dict, while `calcScore.simpleScore_fromDict` reads the SIGN from
    the operational dicts.  Flip one and not the other and the pipeline runs, the columns exist,
    and the score is wrong -- there is no exception to catch.
    """
    _pre, calc, *_rest = cdic.getDicts()
    op = _operational_signs()

    missing_from_calc = sorted(set(op) - set(calc))
    assert not missing_from_calc, (
        'these metrics are scored but have no BoMetric_Calc_dict entry, so their COLUMNS are '
        'never created: %s' % missing_from_calc)
    orphan_in_calc = sorted(set(calc) - set(op))
    assert not orphan_in_calc, (
        'these metrics create columns but are in no operational dict, so nothing scores them: '
        '%s' % orphan_in_calc)

    for k in sorted(op):
        assert set(op[k]) == set(calc[k]['Operation']), (
            "%s: operational dicts define operations %s but BoMetric_Calc_dict declares %s -- "
            "the column set and the scored set disagree"
            % (k, sorted(op[k]), sorted(calc[k]['Operation'])))
        for letter, sign in op[k].items():
            assert sign == calc[k]['Sign'], (
                "%s ('%s' form): operational Sign %+d but BoMetric_Calc_dict Sign %+d. A sign "
                "fix was applied to one registry and not the other -- exactly the silent "
                "failure this test exists for." % (k, letter, sign, calc[k]['Sign']))


def test_every_declared_guard_has_a_predicate():
    """A `Guard` naming a predicate that does not exist must not be ignorable."""
    dicts = cdic.getBaseMeanDiffUnitySpecialDicts()
    declared = {k: v['Guard'] for d in dicts for k, v in d.items() if v.get('Guard')}
    assert declared, 'the sign-inversion fix declares guards; none found'
    for metric, guard in declared.items():
        assert guard in cm.STAGE1_DOMAIN_GUARDS, (
            '%s declares Guard %r, which is not in calcMetrics.STAGE1_DOMAIN_GUARDS. The '
            'criterion would score its out-of-domain rows with nothing to say so.'
            % (metric, guard))


def test_unknown_guard_raises_rather_than_passing_through():
    df = pd.DataFrame({'totalStockholdersEquity': [1.0, 2.0]})
    with pytest.raises(KeyError, match='no guard named'):
        cm.apply_domain_guard(df, [1.0, 2.0], 'not_a_guard')


def test_guard_length_mismatch_raises():
    df = pd.DataFrame({'totalStockholdersEquity': [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match='flags for'):
        cm.apply_domain_guard(df, [1.0, 2.0], 'equity_positive')


# --------------------------------------------------------------------------- #
#  2. THE DECLARED SPEC OF EVERY FIXED CRITERION, PINNED LITERALLY            #
# --------------------------------------------------------------------------- #
#  (which dict, key, Upper, Lower, Tier, Sign, Guard)
#  Sign is the column that matters here: the two INVERTED metrics flipped -1 -> +1 because
#  their good direction reversed; the four GUARDED ones did NOT flip, because a guard shrinks
#  the domain without reversing anything.  Mixing those two up is the bug.
_EXPECTED = [
    # --- INVERTED to yield form: Sign FLIPPED from -1 to +1 --------------------
    ('mean',  'freeCashFlowToMarketCap', 'freeCashFlow',            'marketCap', 'S', +1, None),
    #  The two `bookToPrice` forms carry DIFFERENT guards, deliberately: the inversion fixes the
    #  LEVEL test outright, but on the DIFF form both legs invert on negative equity so a rising
    #  market cap manufactures a pass (review blocker 1).  Guarding the mean form would drop real
    #  negative observations from its pool median and move the bar for everyone.
    ('mean',  'bookToPrice',        'totalStockholdersEquity',      'marketCap', 'B', +1, None),
    ('diff',  'bookToPrice',        'totalStockholdersEquity',      'marketCap', 'B', +1,
     'equity_positive'),
    # --- GUARDED: Sign UNCHANGED ----------------------------------------------
    ('mean',  'debtEquityRatio',    'debtEquityRatio',   'Identity', 'C', -1, 'equity_positive'),
    ('diff',  'freeCashFlowToEquity', 'freeCashFlow',
     'totalStockholdersEquity', 'B', +1, 'equity_positive'),
    #  `netDebtToEBITDA` LEFT THIS TABLE on 2026-08-05: it is no longer a `unity` criterion with
    #  a guard but a `special` three-branch rule (Sign +1, no Guard).  Pinned by
    #  test_netDebtToEBITDA_three_branch_rule below instead.
    #  `effectiveTaxRate` is now Tier 'N' (w = 0) -- REMOVED FROM THE GATE by CEO decision
    #  2026-08-05 -- but the GUARD is still declared and still correct, so the spec stays pinned
    #  here with the new tier.
    ('diff',  'effectiveTaxRate',   'effectiveTaxRate',  'Identity', 'N', -1,
     'tax_rate_nonnegative'),
    #  NEW 2026-08-05.  The guard is the substantive half: FMP reports interestExpense == 0 for a
    #  debt-free name, and a debt-free company has no coverage ratio.
    ('unity', 'interestCoverage',   'operatingIncome',   'interestExpense', 'B', +1,
     'interest_expense_positive'),
]


def _dict_by_kind(kind):
    base, mean, diff, unity, special = cdic.getBaseMeanDiffUnitySpecialDicts()
    return {'base': base, 'mean': mean, 'diff': diff, 'unity': unity,
            'special': special}[kind]


@pytest.mark.parametrize('kind,key,upper,lower,tier,sign,guard', _EXPECTED,
                         ids=[f'{k}:{n}' for k, n, *_ in _EXPECTED])
def test_fixed_criterion_spec(kind, key, upper, lower, tier, sign, guard):
    spec = _dict_by_kind(kind)[key]
    assert spec['Upper'] == upper, '%s %s Upper' % (kind, key)
    assert spec['Lower'] == lower, '%s %s Lower' % (kind, key)
    assert spec['Tier'] == tier, (
        '%s %s Tier changed -- tiers and weights are a CEO decision, not part of this fix'
        % (kind, key))
    assert spec['Sign'] == sign, (
        '%s %s Sign is %+d, expected %+d. An INVERTED metric must be +1 (its direction '
        'reversed); a GUARDED metric keeps its original sign (a guard shrinks the domain, it '
        'does not reverse anything).' % (kind, key, spec['Sign'], sign))
    assert spec.get('Guard') == guard, '%s %s Guard' % (kind, key)


def test_PEG_special_spec():
    special = _dict_by_kind('special')['PEG']
    assert special['Tier'] == 'C'
    assert special['Sign'] == +1, (
        'PEG is GUARDED, not inverted -- the criterion is 1/PEG - 1 > 0 and its direction is '
        'unchanged')
    assert 'Guard' not in special, (
        "PEG carries NO `Guard` key since it started being computed LOCALLY (2026-08-05), and "
        "restoring one is a REGRESSION, not a tightening. A guard is a predicate on the raw "
        "frame whose signature carries no `rpy`, so it would have to re-derive the filer's "
        "frequency from the stamp while calcMetrics.peg_local receives it from the caller -- two "
        "statements of one domain resolved from two different places. PEG's domain is intrinsic "
        "to its formula and lives in peg_local, once. The criterion still REFUSES every state "
        "the old guard refused, plus non-positive current trailing EPS.")


def test_PEG_horizon_and_eps_basis_are_pinned():
    """The two constants a silent edit would move without any error anywhere."""
    assert cm.PEG_GROWTH_YEARS == 1, (
        'the PEG growth horizon is ONE YEAR (trailing-year vs trailing-year). Changing it '
        'changes the criterion pass rate -- measured 0.2149 at 1y, 0.1754 at 2y, 0.1491 at 3y '
        'on the 61,832 newest-8 rows of the 2026-07-17 CORRECTED panel -- so it is a decision, '
        'not a tuning knob.')
    assert cm._PEG_EPS_FIELD == 'netIncomePerShare', (
        "PEG's EPS basis is the `netIncomePerShare` PROXY, not `eps`. `eps` / `epsdiluted` are "
        "captured at ingest but absent from every saved panel; they populate on the next full "
        "fetch. Switching MUST be a deliberate edit -- see calcMetrics._PEG_EPS_FIELD -- never "
        "an `eps if present else proxy` fallback, which would change a scored criterion's basis "
        "silently on the first fetch that carried the column.")


def test_PEG_does_not_read_the_vendor_field_at_all():
    """The whole point of computing it locally: the vendor column must be inert.

    Asserted by CORRUPTING it.  If `priceEarningsToGrowthRatio` still fed the criterion, a value
    inside (0,1) on every row would make the criterion pass and a garbage value would move it.
    """
    growing = [0.5 * (0.95 ** i) for i in range(_PEG_ROWS)]
    ref = _score(_stage1(_frame(_PEG_ROWS, netIncomePerShare=growing)), 'PEG', 'special', 'PEG')
    for vendor in (0.5, -3.0, np.nan, 1e9):
        alt = _stage1(_frame(_PEG_ROWS, netIncomePerShare=growing,
                             priceEarningsToGrowthRatio=vendor))
        assert _score(alt, 'PEG', 'special', 'PEG') == ref, (
            'the vendor PEG field (=%r) still influences the criterion' % vendor)


def test_PEG_domain_is_a_POSITIVE_TRAILING_EPS_and_the_crossing_is_DEFERRED():
    """The four sign states, asserted directly on `peg_local`.

    NEWEST-FIRST frames.  The legs are TRAILING-YEAR sums, so each state is built by holding a
    whole year of EPS at one sign -- the single-period frames the old guard used cannot express a
    TTM condition.

    THE CROSSING STATE IS IN DOMAIN BUT NOT YET ANSWERABLE.  A growth rate computed from a
    NEGATIVE base is not a growth rate, so the row cannot take `|E_prev|` growth (that is the
    2026-08-05 nerf) and it cannot take a build-time constant either -- it takes the POOL's median
    growth, which `calc_special` cannot see.  So it comes out NaN here and is filled by
    `calcMetrics.substitute_peg_crossing`.  Both halves are asserted, because "NaN at build" alone
    is indistinguishable from the pre-fix refusal.
    """
    def peg(now, prev, crossing_growth=None):
        #  4 rows at `now` (this trailing year) then 8 at `prev`, so the one-year lag lands.
        eps = [now] * 4 + [prev] * 8
        df = pd.DataFrame({'netIncomePerShare': eps, 'price': [10.0] * len(eps)})
        return cm.peg_local(df, rpy=4, crossing_growth=crossing_growth)[0].iloc[0]

    assert np.isfinite(peg(1.0, 0.8)), 'both trailing years positive -> an ordinary growth rate'
    assert not np.isfinite(peg(-1.0, 0.8)), 'no positive earnings -> no P/E -> refuse'
    assert not np.isfinite(peg(-1.0, -0.8)), 'still loss-making -> refuse'
    assert not np.isfinite(peg(1.0, 0.0)), 'a zero prior base is division by zero'

    #  THE TURNAROUND: deferred at build, answerable once the pool supplies a median.
    assert not np.isfinite(peg(1.0, -0.8)), (
        'the crossing row must NOT be answered at build time -- under |E_prev| growth it would '
        'saturate near +100%/yr and, worse, grow with the DEPTH of the prior loss, which is the '
        'over-reward the CEO ruled against')
    assert np.isfinite(peg(1.0, -0.8, crossing_growth=25.0)), (
        'the crossing row must become answerable once the POOL median is supplied -- otherwise '
        'this is the pre-fix refusal, and the 5,089 recovering rows stay failed')
    #  and the answer no longer depends on how bad the prior year was
    assert peg(1.0, -0.8, crossing_growth=25.0) == pytest.approx(
        peg(1.0, -8.0, crossing_growth=25.0))

    #  the OLDEST rows have no prior trailing year -> inadmissible, never silently admitted
    df = pd.DataFrame({'netIncomePerShare': [1.0] * 5, 'price': [10.0] * 5})
    assert not np.isfinite(cm.peg_local(df, rpy=4)[0].iloc[-1])


def test_PEG_criterion_end_to_end_through_the_real_scorer():
    """End to end through the real Stage-1 construction and the real scorer."""
    #  REFUSED: no positive trailing EPS -> no P/E -> the criterion cannot be scored.
    for series, label in (([-0.5] * _PEG_ROWS, 'loss-making throughout'),
                          ([(-0.5 if i % 2 else -0.6) for i in range(_PEG_ROWS)],
                           'loss-making, varying')):
        bm = _stage1(_frame(_PEG_ROWS, netIncomePerShare=series))
        assert np.isnan(bm['PEG']).all(), 'PEG admitted a non-positive trailing EPS: %s' % label
        assert _score(bm, 'PEG', 'special', 'PEG') == 0.0, label

    #  FLAT earnings -> zero growth -> PEG is division by zero -> NaN -> fails.  A company with
    #  no growth failing a growth-adjusted cheapness test is the right answer.
    flat = _stage1(_frame(_PEG_ROWS, netIncomePerShare=0.5))
    assert np.isnan(flat['PEG']).all()
    assert _score(flat, 'PEG', 'special', 'PEG') == 0.0

    #  CHEAP GROWER: 0.95^-4 = +21.6%/yr, PE = 10/1.855 = 5.39 -> PEG 0.25 -> PASSES on every
    #  row of the window, so the criterion earns its full weight.
    growing = [0.5 * (0.95 ** i) for i in range(_PEG_ROWS)]
    ok = _stage1(_frame(_PEG_ROWS, netIncomePerShare=growing))
    assert _score(ok, 'PEG', 'special', 'PEG') == _weight('special', 'PEG')

    #  EXPENSIVE GROWER: same growth, 10x the price -> PEG 2.5 -> fails on the (0,1) threshold,
    #  which this change does not touch.
    rich = _stage1(_frame(_PEG_ROWS, netIncomePerShare=growing, price=100.0))
    assert _score(rich, 'PEG', 'special', 'PEG') == 0.0

    #  SHRINKING earnings -> negative growth -> PEG < 0 -> fails.
    shrinking = [0.5 * (1.05 ** i) for i in range(_PEG_ROWS)]
    down = _stage1(_frame(_PEG_ROWS, netIncomePerShare=shrinking))
    assert _score(down, 'PEG', 'special', 'PEG') == 0.0


def test_PEG_turnaround_row_is_MEASURED_after_the_pool_substitution():
    """End to end across BOTH stages: the real Stage-1 construction, then the real
    cross-sectional substitution.

    Only ONE row of a single sign-crossing can be the turnaround (the trailing-year legs straddle
    the crossing for the rows around it), so this is asserted per ROW rather than through the
    8-row window mean.  The pool needs GROWERS as well as the crossing name, or there is no
    in-domain row for a median and the substitution correctly refuses -- which is the honest
    behaviour and is asserted separately below.
    """
    eps_turn = [0.5] * 4 + [-0.5] * (_PEG_ROWS - 4)     # newest trailing year +, prior year -
    eps_grow = [0.5 * (0.95 ** i) for i in range(_PEG_ROWS)]

    turn_fund = _frame(_PEG_ROWS, netIncomePerShare=eps_turn)
    turn_fund['source'] = 'TURN'
    grow_fund = _frame(_PEG_ROWS, netIncomePerShare=eps_grow)
    grow_fund['source'] = 'GROW'

    bm = pd.concat([_stage1(turn_fund).assign(source='TURN'),
                    _stage1(grow_fund).assign(source='GROW')], ignore_index=True)
    cdx = pd.concat([turn_fund, grow_fund], ignore_index=True)
    cdx[rp.FREQ_COLUMN] = rp.QUARTERLY

    raw = pd.to_numeric(bm.loc[bm['source'] == 'TURN', 'PEG'], errors='coerce').iloc[0]
    assert np.isnan(raw), 'the crossing row must be NaN out of the build (see peg_local)'

    out, stats = cm.substitute_peg_crossing(bm, cdx, verbose=False)
    assert np.isfinite(stats['median_growth']) and stats['n_filled'] >= 1
    v = pd.to_numeric(out.loc[out['source'] == 'TURN', 'PEG'], errors='coerce').iloc[0]
    assert np.isfinite(v), (
        'the turnaround row is STILL not computable after the substitution -- the point of the '
        'whole two-stage design is that these rows become MEASURABLE rather than auto-failed')

    #  NO POOL, NO SUBSTITUTION: with only the crossing name there is no in-domain row, so the
    #  median does not exist and the row stays refused rather than being filled with an invention.
    only_turn = cdx[cdx['source'] == 'TURN'].reset_index(drop=True)
    out2, stats2 = cm.substitute_peg_crossing(
        bm[bm['source'] == 'TURN'].reset_index(drop=True), only_turn, verbose=False)
    assert not np.isfinite(stats2['median_growth'])
    assert np.isnan(pd.to_numeric(out2['PEG'], errors='coerce').iloc[0])


def test_PEG_local_matches_its_own_arithmetic():
    """One hand-computed value, so the formula is pinned and not merely self-consistent."""
    eps = [1.0] * 4 + [0.5] * 8                 # TTM now 4.0, prior year 2.0 -> +100%/yr
    df = pd.DataFrame({'netIncomePerShare': eps, 'price': [40.0] * len(eps)})
    peg, now, prev = cm.peg_local(df, rpy=4)
    assert now.iloc[0] == pytest.approx(4.0)
    assert prev.iloc[0] == pytest.approx(2.0)
    #  PE = 40/4 = 10 ; g = 100*(4-2)/|2| / 1yr = 100 ; PEG = 0.10
    assert peg.iloc[0] == pytest.approx(0.10)


def test_eps_fields_are_captured_at_ingest():
    """`eps` / `epsdiluted` ride free in the income-statement response and are wanted for exact
    parity with the vendor PEG.  Pinned so a prereq tidy-up cannot silently drop them."""
    prereq = cdic.getPreReqDict()
    for f in ('eps', 'epsdiluted'):
        assert f in prereq['inc'], (
            '%s dropped from the income-statement prereq list. It costs no extra API call and '
            'is the basis the vendor PEG is built from.' % f)
    #  and the guard must still be running on the DECLARED proxy, not silently switched
    assert cm._PEG_EPS_FIELD == 'netIncomePerShare', (
        'the PEG guard basis changed. That is allowed, but it moves the guard boundary -- '
        're-measure the four sign cells in the same edit rather than letting it drift.')


def test_returnOnEquity_special_spec():
    special = _dict_by_kind('special')['returnOnEquity']
    assert special['Tier'] == 'C'
    assert special['Sign'] == +1, (
        'returnOnEquity is GUARDED, not inverted -- higher return on equity is still better, '
        'so the sign must not flip')
    assert special['Guard'] == 'equity_positive'


def test_retired_criteria_are_gone_from_the_schema():
    """The pre-inversion columns must not survive anywhere.

    A leftover `mPfcfRatio` would be scored at Tier S ALONGSIDE its replacement -- the same
    double-count hazard DUPLICATE_DIFF_CRITERIA documents, but with opposite signs.
    """
    cols = set(utils.initBoMetric_fromDict()['BoMetric_df'].columns)
    for retired in ('mPfcfRatio', 'mPbRatio', 'dPbRatio'):
        assert retired not in cols, (
            '%s is the PRE-FIX column and is still in the schema; it would be scored beside '
            'its inverted replacement' % retired)
    for live in ('mFreeCashFlowToMarketCap', 'mBookToPrice', 'dBookToPrice'):
        assert live in cols, '%s missing from the BoMetric schema' % live


def test_salesToInventory_stays_weight_zero():
    """dSalesToInventory is the ONE recorded member of the family left unfixed, and the ONLY
    thing making that safe is w = 0.

    inventory -> 0 gives +infinity, which lands on this criterion's GOOD side, so an
    inventory-free business would PASS an inventory-TURN test.  47.1% of sources and 26 of the
    100 deployed pool names have at least one such row.  Activating the criterion must therefore
    be a VISIBLE EVENT: this test fails the moment it leaves Tier N, which routes whoever did it
    to the reasoning in createDicts rather than letting a re-weighting turn +inf into a pass in
    silence.  IT IS NOT A CLAIM THAT TIER N IS CORRECT FOREVER -- it is a tripwire.
    """
    spec = _dict_by_kind('diff')['salesToInventory']
    assert spec['Tier'] == 'N', (
        "dSalesToInventory has left Tier N (now %r), so revenue/inventory now carries weight "
        "and inventory == 0 -> +inf -> PASS is live. Read the entry in createDicts: either "
        "guard the zero-inventory rows or invert the ratio BEFORE giving this weight."
        % spec['Tier'])
    from calcScore import calcByTier
    assert calcByTier('diff', spec['Tier'], spec['Sign'],
                      pd.Series([np.inf] * 8), 0.0, 'salesToInventory', 8) == 0.0, \
        'Tier N must score exactly 0 whatever the value'


# --------------------------------------------------------------------------- #
#  3. BEHAVIOUR AT THE CONSUMER -- the layer that catches a wrong sign        #
# --------------------------------------------------------------------------- #
#  Each case: a synthetic single-source statement frame is pushed through the REAL Stage-1
#  construction (`getData_fmp.build_bometric_rows`, i.e. calc_simpleRatio + the flow factor +
#  the guard + calc_diff) and the REAL scorer (`calcScore.calcByTier`), and the resulting
#  pass/fail is asserted.  Nothing here re-implements a formula.
_ROWS = 12          # > 8 + rpy so the head(8) window is full after the history trim
#  PEG needs MORE history than every other criterion, because it is the only one with a
#  TRAILING-YEAR leg AND a one-year growth lag on top of it: row 7 of the scoring window
#  reads a trailing year ending `rpy` rows further back, i.e. 8 + rpy (trim) + 2*rpy - 1
#  rows in total.  A separate constant rather than a bigger _ROWS, so the frames every
#  other test hand-builds at length 12 are untouched.
_PEG_ROWS = 16

_BASE_ROW = {
    # every preReq field the Stage-1 dicts read, at a bland, healthy, in-domain value
    'totalAssets': 1000.0, 'longTermDebt': 100.0, 'inventory': 50.0,
    'totalStockholdersEquity': 500.0, 'totalLiabilities': 500.0,
    'totalCurrentAssets': 300.0, 'totalCurrentLiabilities': 150.0,
    'propertyPlantEquipmentNet': 200.0, 'otherCurrentAssets': 10.0,
    'netIncome': 50.0, 'grossProfit': 200.0, 'revenue': 800.0,
    'weightedAverageShsOut': 100.0, 'weightedAverageShsOutDil': 100.0,
    'depreciationAndAmortization': 30.0,
    'sellingGeneralAndAdministrativeExpenses': 80.0, 'operatingIncome': 90.0,
    'interestExpense': 5.0,
    'freeCashFlow': 60.0, 'netCashProvidedByOperatingActivities': 90.0,
    'netCashUsedProvidedByFinancingActivities': -10.0, 'dividendsPaid': -5.0,
    'netIncomePerShare': 0.5, 'pbRatio': 2.0, 'earningsYield': 0.05, 'pfcfRatio': 15.0,
    'grahamNumber': 8.0, 'grahamNetNet': 1.0, 'marketCap': 1000.0,
    'returnOnTangibleAssets': 0.05, 'incomeQuality': 1.5, 'bookValuePerShare': 5.0,
    'netDebtToEBITDA': 0.5, 'daysSalesOutstanding': 40.0, 'capexPerShare': 0.3,
    'tangibleBookValuePerShare': 4.0, 'dividendYield': 0.02, 'payoutRatio': 0.2,
    'returnOnEquity': 0.10, 'debtEquityRatio': 0.4, 'currentRatio': 2.0,
    'grossProfitMargin': 0.25, 'netProfitMargin': 0.0625, 'effectiveTaxRate': 0.25,
    'returnOnCapitalEmployed': 0.08, 'returnOnAssets': 0.05,
    'priceEarningsToGrowthRatio': 0.8, 'daysOfInventoryOutstanding': 30.0,
    'capitalExpenditureCoverageRatio': -3.0, 'price': 10.0,
}


def _frame(_rows=None, **overrides):
    """One synthetic source, NEWEST-FIRST, `_rows or _ROWS` identical rows with `overrides`.

    Identical rows are deliberate: it makes every DIFF exactly zero, so a diff criterion's
    pass/fail is decided purely by the guard (a zero diff is not > 0, i.e. a fail), and it keeps
    the LEVEL criteria unambiguous.  Where a diff's sign is what is under test the caller
    overrides a column with a per-row sequence.

    `_rows` exists for PEG alone (`_PEG_ROWS`), whose trailing-year leg plus one-year growth lag
    needs a longer frame than any other criterion.  Defaulting to `_ROWS` keeps every existing
    caller -- and every hand-built 12-element override list -- bit-identical.
    """
    rows = []
    for i in range(_rows or _ROWS):
        r = dict(_BASE_ROW)
        for k, v in overrides.items():
            r[k] = v[i] if isinstance(v, (list, tuple)) else v
        r['date'] = pd.Timestamp('2026-03-31') - pd.DateOffset(months=3 * i)
        rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


def _stage1(tempfund, rpy=4):
    """`tempfund` through the REAL Stage-1 construction -> the BoMetric frame."""
    dicts = cdic.getDicts()
    packed = (dicts[2], dicts[3], dicts[5], dicts[4], dicts[6])   # base, mean, unity, diff, spec
    tmp = pd.DataFrame(columns=list(utils.initBoMetric_fromDict()['BoMetric_df'].columns))
    tmp['date'] = tempfund['date'].values
    tmp['source'] = 'TEST'
    return gdf.build_bometric_rows(tempfund.copy(), tmp, rpy, n=1, dicts=packed)


def _score(bm, column, kind, key, avec=0.0, n=8):
    """The criterion's share of its tier weight, via the REAL scorer."""
    base, mean, diff, unity, special = cdic.getBaseMeanDiffUnitySpecialDicts()
    spec = {'base': base, 'mean': mean, 'diff': diff, 'unity': unity,
            'special': special}[kind][key]
    return cs.calcByTier(kind, spec['Tier'], spec['Sign'], bm[column], avec, key, n)


def _weight(kind, key):
    spec = _dict_by_kind(kind)[key]
    return {'S': 1.0, 'A': 0.75, 'B': 0.5, 'C': 0.3, 'D': 0.1}.get(spec['Tier'], 0.0)


# ---- the headline: negative FCF must not read as cheap ---------------------
def test_negative_fcf_fails_the_tier_S_cheapness_criterion():
    """THE defect this whole change exists for.

    Pre-fix, `mPfcfRatio` = price/FCF went NEGATIVE on a cash-burning company, and a mean test
    with Sign -1 reads "below the pool median" as CHEAP -- so 75.9% of the highest-weighted
    cheapness criterion's passes were cash burners.  The yield form must fail them, and must
    still pass a genuinely cheap cash generator.
    """
    med = 0.04      # a plausible positive panel median FCF yield
    burn = _stage1(_frame(freeCashFlow=-60.0))
    assert _score(burn, 'mFreeCashFlowToMarketCap', 'mean', 'freeCashFlowToMarketCap',
                  avec=med) == 0.0, \
        'negative free cash flow still earns the Tier-S cheapness criterion'

    cheap = _stage1(_frame(freeCashFlow=120.0))       # 12% yield, well above the median
    assert _score(cheap, 'mFreeCashFlowToMarketCap', 'mean', 'freeCashFlowToMarketCap',
                  avec=med) == _weight('mean', 'freeCashFlowToMarketCap'), \
        'a genuinely high FCF yield must still pass -- the fix must not just fail everything'

    # AND THE DIRECTION IS NOW MONOTONE IN THE RIGHT WAY: more cash flow scores more.
    lo = _stage1(_frame(freeCashFlow=10.0))
    assert _score(lo, 'mFreeCashFlowToMarketCap', 'mean', 'freeCashFlowToMarketCap',
                  avec=med) == 0.0


def test_negative_equity_fails_book_to_price_both_forms():
    """price/book went negative on negative equity and read as maximally cheap, in the mean
    form AND in the diff form (an equity base collapsing through zero looked like a big fall
    in P/B)."""
    med = 0.5
    neg = _stage1(_frame(totalStockholdersEquity=-500.0))
    assert _score(neg, 'mBookToPrice', 'mean', 'bookToPrice', avec=med) == 0.0
    pos = _stage1(_frame(totalStockholdersEquity=900.0))
    assert _score(pos, 'mBookToPrice', 'mean', 'bookToPrice', avec=med) == \
        _weight('mean', 'bookToPrice')

    # DIFF form: equity falling THROUGH zero must not score as an improving book yield.
    # Newest-first, so index 0 is the latest: 500 -> -100 is a DETERIORATION.
    collapsing = _stage1(_frame(totalStockholdersEquity=[-100.0] * 6 + [500.0] * 6))
    assert _score(collapsing, 'dBookToPrice', 'diff', 'bookToPrice') == 0.0, \
        'an equity base collapsing through zero scored as a rising book-to-price'
    # ... while a genuinely rising book yield still passes.
    rising = _stage1(_frame(totalStockholdersEquity=[900.0] * 6 + [400.0] * 6))
    assert _score(rising, 'dBookToPrice', 'diff', 'bookToPrice') > 0.0


def test_dBookToPrice_refuses_negative_equity_including_the_MARKET_CAP_leg():
    """REVIEW BLOCKER 1: inverting to yield form does NOT fix the diff form on negative equity,
    because BOTH legs of equity/marketCap invert.

    With equity < 0, a RISING market cap drives the negative ratio toward zero, so the diff is
    POSITIVE and the row passes -- a company getting MORE EXPENSIVE, with its equity deficit
    flat or worse, passing a CHEAPNESS criterion.  Measured on the real panel: 444 of the 1,313
    post-inversion negative-equity passes (33.8%) were driven by the market-cap leg alone, 434
    of them with market cap rising against flat-or-worse equity.

    This is the case an earlier check MISSED by validating "did book-to-price rise", which is
    the pass condition itself.  So it is asserted here on the LEG that moved, holding equity
    FIXED so only the market cap can explain a pass.
    """
    #  equity constant and NEGATIVE; market cap RISING (newest-first: 800 now, 2000 before).
    #  Pre-guard this passed: -500/800 = -0.625 vs -500/2000 = -0.25 -> diff -0.375 ... so the
    #  perverse direction is a FALLING market cap; a RISING one gives the positive diff:
    #  newest 2000 -> -0.25, prior 800 -> -0.625, diff = +0.375 > 0 = PASS.
    mcap_rising = _stage1(_frame(totalStockholdersEquity=-500.0,
                                 marketCap=[2000.0] * 6 + [800.0] * 6))
    assert np.isnan(mcap_rising['dBookToPrice']).all(), (
        'dBookToPrice is still defined on negative equity -- a rising market cap can therefore '
        'still manufacture a positive diff')
    assert _score(mcap_rising, 'dBookToPrice', 'diff', 'bookToPrice') == 0.0, \
        'a company getting MORE EXPENSIVE on negative equity passed a cheapness criterion'

    #  and the guard is TWO-SIDED for free: NaN propagates through calc_diff, so one bad period
    #  is enough.  Newest period fine, PRIOR period negative -> no defined change.
    prior_bad = _stage1(_frame(totalStockholdersEquity=[600.0] * 6 + [-500.0] * 6))
    assert np.isnan(prior_bad['dBookToPrice']).iloc[:6].any(), (
        'a diff spanning a negative-equity period must be refused -- there is no defined change '
        'in a quantity that was undefined')


def test_mBookToPrice_is_NOT_guarded_so_its_ruler_keeps_real_observations():
    """The asymmetry between the two forms is deliberate and load-bearing.

    The LEVEL test needs no guard: the inversion already makes negative equity a negative book
    yield, which fails the mean test on its own (measured: 0 passes on equity < 0, from 3,657).
    Guarding it anyway would DROP those rows from the pool median -- and this criterion is scored
    as `value - median(column)`, so removing 3,661 legitimate observations would MOVE THE BAR for
    every name that was always in domain.  A negative book yield is a true measurement, not a
    domain error.

    This test is what stops someone "tidying up" the asymmetry by adding a guard to both.
    """
    neg = _stage1(_frame(totalStockholdersEquity=-500.0))
    assert not np.isnan(neg['mBookToPrice']).all(), (
        'mBookToPrice has been guarded. It does not need to be (it already fails), and guarding '
        'it removes real negative observations from the pool median, moving the bar for every '
        'other name. Read the createDicts entry before changing this.')
    #  ... and it still FAILS, which is the whole reason no guard is needed.
    assert _score(neg, 'mBookToPrice', 'mean', 'bookToPrice', avec=0.5) == 0.0


def test_guards_are_applied_PER_FORM_not_once_per_key():
    """`bookToPrice` lives in BOTH the mean and diff dicts, and they now carry DIFFERENT guards.

    `build_bometric_rows` iterates a MERGED dict where the last entry wins, so reading the guard
    from the merged entry would force one domain on both forms.  This pins the per-form
    behaviour: same key, same source frame, one column NaN and the other not.
    """
    bm = _stage1(_frame(totalStockholdersEquity=-500.0))
    assert np.isnan(bm['dBookToPrice']).all(), 'diff form must be guarded'
    assert not np.isnan(bm['mBookToPrice']).all(), 'mean form must NOT be guarded'


def test_shared_keys_agree_on_basis_or_are_declared_exceptions():
    """A key in two operational dicts is built ONCE, from the MERGED entry (last wins).

    So if two forms of the same key declare different Upper/Lower, one of them is a lie. There is
    exactly one such case today and it is a KNOWN, REPORTED defect awaiting a ruling -- pinned
    here rather than silently fixed, because correcting it changes the basis of a Tier-S w=1.0
    criterion.  A NEW disagreement must fail this test.
    """
    _pre, _calc, base, mean, diff, unity, _spec = cdic.getDicts()
    dicts = {'base': base, 'mean': mean, 'unity': unity, 'diff': diff}
    #  key -> the reason its two forms legitimately disagree on basis.  Do not extend this
    #  without a ruling; the point of the dict is that additions are visible.
    KNOWN = {
        'returnOnAssets': 'base declares netIncome/totalAssets but the merged diff entry '
                          'supplies FMP returnOnAssets; measured near-equivalent (median ratio '
                          '1.0000, 97.3% within 1%). Spec-vs-code defect, not a scoring one.',
    }
    for key in {k for d in dicts.values() for k in d}:
        forms = {n: d[key] for n, d in dicts.items() if key in d}
        if len(forms) < 2:
            continue
        bases = {(s['Upper'], s['Lower']) for s in forms.values()}
        if len(bases) > 1:
            assert key in KNOWN, (
                "%s declares different Upper/Lower in %s -- the ratio is built ONCE from the "
                "merged dict, so only one of them is real. Either align them or declare the "
                "exception with its reason." % (key, sorted(forms)))
    #  and the known exception must still BE one -- if someone fixes it, remove it from KNOWN.
    ra_bases = {(base['returnOnAssets']['Upper'], base['returnOnAssets']['Lower']),
                (diff['returnOnAssets']['Upper'], diff['returnOnAssets']['Lower'])}
    assert len(ra_bases) > 1, (
        'returnOnAssets no longer disagrees on basis -- good, but remove it from KNOWN above so '
        'the exception list stays honest.')


# ---- the guards: adverse row refused, GENUINE row preserved ---------------
def test_negative_equity_refused_by_debt_equity_and_fcf_to_equity():
    neg = _stage1(_frame(totalStockholdersEquity=-500.0, debtEquityRatio=-0.4))
    assert np.isnan(neg['mDebtEquityRatio']).all(), \
        'negative equity must make debt/equity out of domain, not "unlevered"'
    assert _score(neg, 'mDebtEquityRatio', 'mean', 'debtEquityRatio', avec=0.43) == 0.0
    assert np.isnan(neg['dFreeCashFlowToEquity']).all()
    assert _score(neg, 'dFreeCashFlowToEquity', 'diff', 'freeCashFlowToEquity') == 0.0

    # THE DOUBLE NEGATIVE specifically: FCF < 0 AND equity < 0 gave a POSITIVE ratio.
    dn = _stage1(_frame(totalStockholdersEquity=-500.0, freeCashFlow=-60.0))
    assert np.isnan(dn['dFreeCashFlowToEquity']).all()

    # and a debt-free, positive-equity name is NOT refused -- the guard is on equity only.
    free = _stage1(_frame(debtEquityRatio=0.0))
    assert not np.isnan(free['mDebtEquityRatio']).all(), \
        'a DEBT-FREE company must stay in domain; refusing it would re-create the defect ' \
        'assetsToLongTermLiabilities was inverted to fix'
    assert _score(free, 'mDebtEquityRatio', 'mean', 'debtEquityRatio', avec=0.43) == \
        _weight('mean', 'debtEquityRatio'), 'debt-free must PASS a leverage-safety test'


def test_netDebtToEBITDA_three_branch_rule():
    """THE most load-bearing test in this file -- ALL FOUR SIGN CELLS OF THE THREE-BRANCH RULE.

    (CEO, 2026-08-05.)  The predecessor test pinned the two-branch guarded form; the rule now
    has three branches and the FOURTH CELL FLIPPED from refuse to PASS:

      netDebt > 0, EBITDA > 0  ->  test the annualised ratio < 1.0     (unchanged)
      netDebt < 0, EBITDA > 0  ->  PASS   (genuine net cash; 37.8% of passes, must SURVIVE)
      netDebt > 0, EBITDA <= 0 ->  REFUSED (debt and no earnings -- the 2026-08-04 defect)
      netDebt < 0, EBITDA <= 0 ->  PASS   <- THE FIX.  Net cash means no leverage problem.

    sign(netDebt) is recovered as sign(ratio) x sign(EBITDA proxy), so a cell is set up by
    choosing the RATIO's sign and the PROXY's sign together -- which is also what proves the
    net-cash branch is an OPERAND condition: cell 4's ratio is POSITIVE (negative/negative) and
    it must pass WITHOUT its magnitude being consulted.
    """
    w = _weight('special', 'netDebtToEBITDA')

    # --- cell 1: debt, earnings.  Ordinary test on the ANNUALISED ratio ------------------
    # rpy=4 -> flow factor 0.25, so the `< 1.0` bar is on ratio*0.25, i.e. ratio < 4 annualised.
    assert _score(_stage1(_frame(netDebtToEBITDA=0.5)), 'netDebtToEBITDA', 'special',
                  'netDebtToEBITDA') == w, 'low leverage must pass'
    assert _score(_stage1(_frame(netDebtToEBITDA=8.0)), 'netDebtToEBITDA', 'special',
                  'netDebtToEBITDA') == 0.0, 'high leverage must fail'

    # --- cell 2: NET CASH with positive EBITDA -> ratio negative -> PASS -----------------
    genuine = _stage1(_frame(netDebtToEBITDA=-1.5))
    assert _score(genuine, 'netDebtToEBITDA', 'special', 'netDebtToEBITDA') == w, \
        'GENUINE NET CASH (37.8% of this criterion\'s passes) must still pass'

    # --- cell 3: DEBT with non-positive EBITDA -> REFUSED (NaN -> fail) ------------------
    # ratio NEGATIVE with proxy NEGATIVE -> sign product POSITIVE -> not net cash -> branch 3.
    perverse = _stage1(_frame(operatingIncome=-60.0, depreciationAndAmortization=30.0,
                              netDebtToEBITDA=-2.0))
    assert np.isnan(perverse['netDebtToEBITDA']).all(), \
        'debt with no earnings must be REFUSED, not scored'
    assert _score(perverse, 'netDebtToEBITDA', 'special', 'netDebtToEBITDA') == 0.0, \
        'a levered company with no earnings must not score as the safest balance sheet'

    # --- cell 4: NET CASH with non-positive EBITDA -> PASS.  THE FIX. --------------------
    # ratio POSITIVE (negative netDebt / negative EBITDA) with proxy NEGATIVE.
    net_cash_loss = _stage1(_frame(operatingIncome=-60.0, depreciationAndAmortization=30.0,
                                   netDebtToEBITDA=0.5))
    assert _score(net_cash_loss, 'netDebtToEBITDA', 'special', 'netDebtToEBITDA') == w, \
        'net cash with negative EBITDA must PASS -- there is no leverage problem to have'

    # AND IT MUST NOT BE PASSING ON THE RATIO'S APPARENT MERIT.  A net-cash row with a LARGE
    # positive ratio (netDebt -100 / EBITDA -1 = 100) would FAIL a `< 1` test on magnitude, so
    # if it passes, the branch is keying on the OPERAND SIGN and not on the ratio.  This is the
    # single assertion that separates the required fix from the sign-inversion defect class.
    big = _stage1(_frame(operatingIncome=-60.0, depreciationAndAmortization=30.0,
                         netDebtToEBITDA=100.0))
    assert _score(big, 'netDebtToEBITDA', 'special', 'netDebtToEBITDA') == w, \
        'the net-cash branch must pass on the OPERAND sign, never on the ratio magnitude'
    assert (np.asarray(big['netDebtToEBITDA'], dtype=float) == 1.0).all(), \
        'the net-cash branch must emit its admission SENTINEL, not a computed ratio margin'

    # --- the two unrecoverable states are REFUSED, never rewarded ------------------------
    # EBITDA proxy exactly zero: sign(netDebt) cannot be recovered at all.
    zero_proxy = _stage1(_frame(operatingIncome=-30.0, depreciationAndAmortization=30.0,
                                netDebtToEBITDA=-2.0))
    assert np.isnan(zero_proxy['netDebtToEBITDA']).all(), \
        'a zero EBITDA proxy leaves sign(netDebt) unrecoverable -- refuse, never reward'
    # a missing vendor ratio with healthy EBITDA: branch 2 has nothing to test.
    no_ratio = _stage1(_frame(netDebtToEBITDA=np.nan))
    assert np.isnan(no_ratio['netDebtToEBITDA']).all()


def test_netDebtToEBITDA_carries_no_guard_key():
    """The old `ebitda_positive` guard IS branch 2's condition and must not be re-declared.

    Restoring a `Guard` here would refuse the net-cash cell before the rule ever saw it -- i.e.
    it would silently undo the fix while looking like a tightening.
    """
    spec = _dict_by_kind('special')['netDebtToEBITDA']
    assert spec['Tier'] == 'A', 'the tier is unchanged by the form change'
    assert spec['Sign'] == +1, (
        'the column now holds a verdict-bearing margin (positive = passes), not the leverage '
        'ratio, so the sign is +1')
    assert 'Guard' not in spec, (
        'the three-branch rule states its own domain in calcMetrics.net_debt_three_branch. A '
        '`Guard: ebitda_positive` here would refuse the net-cash-with-negative-EBITDA cell '
        'before the rule ran -- undoing the 2026-08-05 fix while looking like a tightening.')


def test_interestCoverage_bar_and_debt_free_guard():
    """`operatingIncome / interestExpense > 1`, and a DEBT-FREE name is refused not failed."""
    w = _weight('unity', 'interestCoverage')
    # covered 18x (90 / 5) -> passes
    assert _score(_stage1(_frame()), 'uInterestCoverage', 'unity', 'interestCoverage') == w
    # covered 0.5x -> fails the unity bar (it is IN domain: the interest bill is real)
    thin = _stage1(_frame(operatingIncome=5.0, interestExpense=10.0))
    assert not np.isnan(thin['uInterestCoverage']).all(), 'a real interest bill is in domain'
    assert _score(thin, 'uInterestCoverage', 'unity', 'interestCoverage') == 0.0
    # exactly 1.0x is NOT > 1 -> fails.  The bar is stated as `> 1`, so pin the boundary.
    exact = _stage1(_frame(operatingIncome=10.0, interestExpense=10.0))
    assert _score(exact, 'uInterestCoverage', 'unity', 'interestCoverage') == 0.0
    # DEBT-FREE (FMP reports 0) -> REFUSED, not "failed for having no debt".
    free = _stage1(_frame(interestExpense=0.0))
    assert np.isnan(free['uInterestCoverage']).all(), (
        'interestExpense == 0 is a DEBT-FREE name; it has no coverage ratio and must be '
        'refused. The leverage question for it is carried by netDebtToEBITDA.')
    # a negative reported interest expense inverts the ratio's sign -> also refused.
    assert np.isnan(_stage1(_frame(interestExpense=-5.0))['uInterestCoverage']).all()


def test_capex_coverage_bar_is_self_funding_not_two_times():
    """`CFO > |capex|` (the definition of self-funding capex), not `CFO > 2 x |capex|`."""
    # the panel column is FMP's CFO/capex with capex NEGATIVE, so -ratio is CFO/|capex|.
    # 1.5x covered: FAILED under the old 2x bar, PASSES under the derived one.
    mid = _stage1(_frame(capitalExpenditureCoverageRatio=-1.5))
    assert _score(mid, 'capitalExpenditureCoverageRatio', 'special',
                  'capitalExpenditureCoverageRatio') == \
        _weight('special', 'capitalExpenditureCoverageRatio'), \
        'CFO at 1.5x capex IS self-funding and must pass the derived bar'
    # 0.5x covered: not self-funding -> still fails.
    under = _stage1(_frame(capitalExpenditureCoverageRatio=-0.5))
    assert _score(under, 'capitalExpenditureCoverageRatio', 'special',
                  'capitalExpenditureCoverageRatio') == 0.0
    # exactly 1.0x is NOT > 1 -> fails.  Pin the boundary so the bar cannot drift to `>=`.
    exact = _stage1(_frame(capitalExpenditureCoverageRatio=-1.0))
    assert _score(exact, 'capitalExpenditureCoverageRatio', 'special',
                  'capitalExpenditureCoverageRatio') == 0.0


def test_returnOnAssets_base_computes_what_it_declares():
    """Issue I-5: the BASE column is `netIncome/totalAssets`, not FMP's `returnOnAssets` field.

    The frames here set the two to DIFFERENT values on purpose -- that is the only way to tell
    which one the column is built from, and the merged-dict collapse used to make it the vendor
    field silently.
    """
    bm = _stage1(_frame(netIncome=50.0, totalAssets=1000.0, returnOnAssets=-0.99))
    assert np.allclose(np.asarray(bm['returnOnAssets'], dtype=float), 0.05), \
        'the base column must be netIncome/totalAssets as DECLARED, not the vendor field'
    # the DIFF form is UNCHANGED and still reads the vendor field, by declaration.
    dv = _stage1(_frame(returnOnAssets=[0.10] * 6 + [0.02] * 6, netIncome=50.0))
    assert _score(dv, 'dReturnOnAssets', 'diff', 'returnOnAssets') > 0.0, \
        'the diff form still reads the vendor field -- a rising vendor RoA must still score'


def test_effectiveTaxRate_and_grahamNumberToPrice_are_off_the_gate():
    """Both demoted to Tier 'N' (w = 0) by CEO decision 2026-08-05 -- they score EXACTLY 0."""
    base, mean, diff, unity, special = cdic.getBaseMeanDiffUnitySpecialDicts()
    assert diff['effectiveTaxRate']['Tier'] == 'N'
    assert unity['grahamNumberToPrice']['Tier'] == 'N'
    # a criterion at Tier N contributes 0 whatever the data says -- the columns still exist and
    # are still computed, which is what keeps the demotion a one-character edit to reverse.
    falling = _stage1(_frame(effectiveTaxRate=[0.15] * 6 + [0.30] * 6))
    assert _score(falling, 'dEffectiveTaxRate', 'diff', 'effectiveTaxRate') == 0.0
    assert _score(_stage1(_frame()), 'uGrahamNumberToPrice', 'unity',
                  'grahamNumberToPrice') == 0.0
    # the Boundary imputation on grahamNumberToPrice is UNTOUCHED by the demotion.
    assert 'Boundary' in unity['grahamNumberToPrice']


def test_stage1_total_weight_is_17_85():
    """Sigma-w over the Stage-1 registry: 18.65 -> 17.85 (CEO changes of 2026-08-05).

    -1.00 uGrahamNumberToPrice (S->N), -0.30 dEffectiveTaxRate (C->N), +0.50 uInterestCoverage.
    A DERIVED total, summed from the registry rather than transcribed, so it moves with the next
    tier edit instead of becoming a stale literal (which is what 18.65 became).
    """
    _TIER_W = {'S': 1.0, 'A': 0.75, 'B': 0.5, 'C': 0.3, 'D': 0.1}
    total = 0.0
    for d in cdic.getBaseMeanDiffUnitySpecialDicts():
        for spec in d.values():
            total += _TIER_W.get(spec['Tier'], 0.0)
    assert abs(total - 17.85) < 1e-9, 'Stage-1 Sigma-w is %.4f, expected 17.85' % total


def test_negative_effective_tax_rate_refused():
    neg = _stage1(_frame(effectiveTaxRate=-0.5))
    assert np.isnan(neg['dEffectiveTaxRate']).all(), \
        'a negative effective tax rate must not read as improving tax efficiency'
    # A rate FALLING within the admissible domain is still a PASS on the column -- but the
    # criterion is Tier 'N' (w = 0) since 2026-08-05, so its SCORE is 0 by weight and not by
    # domain.  Both are asserted separately, because collapsing them would let a future
    # re-promotion pass this test while the guard was broken.
    falling = _stage1(_frame(effectiveTaxRate=[0.15] * 6 + [0.30] * 6))
    _col = pd.to_numeric(pd.Series(list(falling['dEffectiveTaxRate'])), errors='coerce').head(8)
    assert (_col < 0).any(), (
        'a FALLING tax rate must still produce a negative diff (Sign -1 -> a pass) -- the '
        'demotion to Tier N removes the WEIGHT, not the measurement')
    assert _score(falling, 'dEffectiveTaxRate', 'diff', 'effectiveTaxRate') == 0.0, \
        'Tier N means w = 0, so the criterion scores 0 however well the company does'
    # a ZERO rate is a real answer and stays in domain
    zero = _stage1(_frame(effectiveTaxRate=0.0))
    assert not np.isnan(zero['dEffectiveTaxRate']).all()


def test_returnOnEquity_double_negative_refused_stage1():
    """netIncome < 0 AND equity < 0 gave a POSITIVE ROE that cleared the 12% hurdle."""
    dn = _stage1(_frame(netIncome=-50.0, totalStockholdersEquity=-500.0,
                        returnOnEquity=0.10))     # FMP reports +10% from two negatives
    assert np.isnan(dn['returnOnEquity']).all()
    assert _score(dn, 'returnOnEquity', 'special', 'returnOnEquity') == 0.0, \
        'a loss-making, book-insolvent company cleared a 12% ROE hurdle'
    ok = _stage1(_frame(returnOnEquity=0.20))
    assert _score(ok, 'returnOnEquity', 'special', 'returnOnEquity') == \
        _weight('special', 'returnOnEquity')
    # BELOW the hurdle still fails -- the hurdle itself is untouched by this fix.
    assert _score(_stage1(_frame(returnOnEquity=0.01)), 'returnOnEquity', 'special',
                  'returnOnEquity') == 0.0


def test_returnOnEquity_guarded_in_stage2_too():
    """Stage-1 and Stage-2 both score returnOnEquity; one fix, both halves.

    Stage-2 RANKS the value, so an unguarded negative-equity name did not merely pass a test --
    it sat ABOVE the pool median on the column.
    """
    n = 8
    dn = pd.DataFrame({'returnOnEquity': [0.25] * n,
                       'totalStockholdersEquity': [-500.0] * n})
    assert np.isnan(sm.postbm_metric('returnOnEquity', 'returnOnEquity', dn, n)), \
        'Stage-2 still ranks a return on NEGATIVE equity as a high return'

    ok = pd.DataFrame({'returnOnEquity': [0.25] * n,
                       'totalStockholdersEquity': [500.0] * n})
    assert sm.postbm_metric('returnOnEquity', 'returnOnEquity', ok, n) == pytest.approx(0.25)

    # PARTIAL: the admissible rows carry the metric (the income_quality_accruals convention).
    mixed = pd.DataFrame({'returnOnEquity': [0.25] * 4 + [9.99] * 4,
                          'totalStockholdersEquity': [500.0] * 4 + [-500.0] * 4})
    assert sm.postbm_metric('returnOnEquity', 'returnOnEquity', mixed, n) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
#  4. THE INVERSION IS EQUIVALENT TO THE OLD METRIC WHERE THE OLD ONE WAS OK  #
# --------------------------------------------------------------------------- #
def test_inversion_preserves_ordering_on_the_admissible_domain():
    """An inversion is only a legitimate fix if it AGREES with the old metric wherever the old
    metric was well-defined -- otherwise it is a different criterion wearing a bug fix's
    clothes.  On positive FCF / positive equity, price/X and X/price are strictly
    order-REVERSING, so `cheaper on price/X` and `higher yield on X/price` must select the same
    rows.  Asserted on a grid rather than argued.
    """
    fcf = np.array([5.0, 10.0, 20.0, 40.0, 80.0])
    mcap = 1000.0
    pfcf = mcap / fcf                      # the PRE-FIX quantity
    yld = fcf / mcap                       # the POST-FIX quantity
    # Sign -1 on (pfcf - median) and Sign +1 on (yield - median) must agree row for row.
    old_pass = (-1 * (pfcf - np.median(pfcf))) > 0
    new_pass = (+1 * (yld - np.median(yld))) > 0
    assert (old_pass == new_pass).all(), (
        'the yield form disagrees with price/FCF on the domain where price/FCF was VALID -- '
        'the inversion has changed the criterion, not just fixed its sign')

    eq = np.array([100.0, 250.0, 500.0, 1000.0, 2000.0])
    pb = mcap / eq
    b2p = eq / mcap
    assert (((-1 * (pb - np.median(pb))) > 0) == ((+1 * (b2p - np.median(b2p))) > 0)).all()
