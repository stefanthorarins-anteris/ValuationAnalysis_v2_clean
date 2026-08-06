"""THE ENFORCEMENT for issue E-2 -- the block-derived weight vectors (2026-08-04).

`scoringWeights` no longer stores weights; it stores a BLOCK MODEL and computes them.  That
buys the properties the design cares about (defect demotion by rule, cohorts that follow the
general vector, no transcribed repeating decimals) and it costs one thing: the numbers are no
longer visible in the source, so nothing stops the machinery from computing a DIFFERENT
vector than the one the CEO decided.  This file is that check.  It pins the DESIGN's own
published values -- projects/investment-filter/design/weighting-strategy.md sections 14/15,
at the CEO's equal-durability budgets -- against what the code derives.

WHAT IS PINNED HERE, and why each one:
  * every non-zero weight in all six vectors, to the design's display precision.  A golden
    copy of numbers is normally the defect class this repo removes; here it is the OPPOSITE,
    because the numbers live in a document and the code is the derivation of them.
  * Sigma|w| = 1 EXACTLY, in binary floating point.  The designer's explicit warning was that
    hand-entered 4-dp values give 0.9999 and halt every import.
  * THE THESIS MARGIN (`earnYield` the largest single |w|).  It has already broken once under
    a budget change, so it is asserted rather than trusted -- including a FORWARD check
    against the D3 fix, which is the change most likely to break it next.
  * the two extracted Piotroski components' MISSING-DATA behaviour, which is the project's
    standing premise ("missing data must never reward by default") and the exact defect the
    parent composite has.

No network, no pickles, no API key.  Run it the repo way: `pytest . --ignore=baseline_tools`.
"""
import numpy as np
import pandas as pd
import pytest

import scoringWeights as sw
import stage2_metrics as s2m


# --------------------------------------------------------------------------- #
#  A.  the general vector                                                     #
# --------------------------------------------------------------------------- #
#  Section 15.3's table re-derived at the CEO's DECIDED budgets, AS RE-BUDGETED FOR THE S
#  BLOCK ON 2026-08-06: W_S 0.0860 -> 0.1200 to pay for `interestCoverage` (S's first Tier-1
#  member), with the 0.034 taken PROPORTIONALLY from the other six blocks (x 0.880/0.914).
#  P 0.250328  R 0.157322  N 0.124201  D 0.250328  S 0.120000  M 0.049680  C 0.048140.
#  **19** non-zero weights now, not 18.
#
#  WHAT THE PROPORTIONAL TAKE BUYS, and it is why this table could be re-derived by
#  multiplication rather than re-argued: every ratio among NON-S metrics is preserved exactly.
#  So P stays equal to D (Rule PROP's 1:1 residual split is untouched) and the thesis margin
#  is numerically unchanged at 1.2928.  Only the three S weights are a new argument.
DESIGN_GENERAL = {
    'earnYield':                   0.122035,
    'incomeQuality':               0.094393,
    'freeCashFlowPerShareGrowth':  0.083443,
    'CycleHeat':                  -0.074521,
    'bVpRatio':                    0.065711,
    'freeCashFlowYield':           0.062929,
    'RoA':                         0.062582,
    'returnOnCapitalEmployed':     0.062582,
    #  THE THREE S WEIGHTS -- the only ones that are a new decision rather than a rescale.
    #  W_S = 0.120 over tiers {1,2,3} at tau 3:2:1 gives 0.060 / 0.040 / 0.020, which
    #  DELIBERATELY costs the two incumbents ~30%: holding them harmless would have needed
    #  W_S = 0.1720, and the CEO capped at 0.120 instead.
    'interestCoverage':            0.060000,
    'currentRatio':                0.040000,
    'Altman-Z':                    0.020000,
    'EPStoEPSmean':                0.049681,
    'marketCapRevQuants':          0.049681,
    'grahamNumberToPrice':         0.040678,
    'Piotroski':                   0.032093,
    'tbVpRatio':                   0.021904,
    'grossProfitMargin':           0.020861,
    'returnOnEquity':              0.020861,
    'revenueGrowth':               0.016047,
}


def test_the_general_vector_IS_the_decided_one():
    """Every one of the 18, to the design's published precision."""
    for metric, want in DESIGN_GENERAL.items():
        assert sw.DEPLOYED[metric] == pytest.approx(want, abs=5e-7), metric
    non_zero = {k: v for k, v in sw.DEPLOYED.items() if v != 0.0}
    assert set(non_zero) == set(DESIGN_GENERAL), sorted(
        set(non_zero) ^ set(DESIGN_GENERAL))
    assert len(non_zero) == 19


def test_the_six_zeros_are_the_ones_they_should_be():
    """Three by decision (DcfToPrice / BoScore / priceGrowth) and THREE by DOMAIN -- the
    FIN-1-only columns carry no general-pool weight.  `navPerShareGrowth` joined them on
    2026-08-06: book-value-per-share growth is FIN-1's R-block thesis test, and for an
    operating company book value is a residual rather than the thesis (scoringWeights D.4)."""
    zeros = {k for k, v in sw.DEPLOYED.items() if v == 0.0}
    assert zeros == set(sw.DELIBERATELY_ZEROED) | {'shareCountChange',
                                                   'longTermDebtChange',
                                                   'navPerShareGrowth'}


def test_sigma_abs_w_is_EXACTLY_one_in_float():
    """Not a tolerance.  This is the tripwire the designer warned about: the repeating
    decimals (0.26/3, 0.26/12, 0.086*2/3, 0.086/3, 0.05*2/3, 0.05/3) are why the vector must
    be COMPUTED -- transcribing the display column gives 0.9999 and `_validate()` halts every
    import."""
    assert sw.sum_abs(sw.DEPLOYED) == 1.0


def test_every_block_spends_EXACTLY_its_budget():
    """The structural property the whole scheme rests on (and what makes Sigma|w| = 1 a
    consequence rather than a coincidence): sum of |w| over a block's members = W_B."""
    per_block = {}
    for metric, (block, _sub, _tier, _sign) in sw.GENERAL_ASSIGNMENT.items():
        per_block[block] = per_block.get(block, 0.0) + abs(sw.DEPLOYED[metric])
    assert set(per_block) == set(sw.GENERAL_BUDGETS)
    for block, spent in per_block.items():
        assert spent == pytest.approx(sw.GENERAL_BUDGETS[block], abs=1e-12), block
    assert sum(sw.GENERAL_BUDGETS.values()) == pytest.approx(1.0, abs=1e-12)


def test_CycleHeat_is_still_NEGATIVE():
    """The one sign in the vector, and it has survived every re-weighting: a hot late-cycle
    EPS is a PENALTY.  Cheap to assert, catastrophic to get wrong -- an inverted CycleHeat
    ranks the hottest names first."""
    assert sw.DEPLOYED['CycleHeat'] < 0
    assert sw.GENERAL_ASSIGNMENT['CycleHeat'][3] == -1


# --------------------------------------------------------------------------- #
#  B.  the thesis margin -- asserted, because it has broken before            #
# --------------------------------------------------------------------------- #
def _largest_other(vector, metric):
    return max(abs(w) for k, w in vector.items() if k != metric)


def test_earnYield_is_the_LARGEST_single_weight_in_the_general_vector():
    """"P/E is essentially the final boss" -- the legibility property behind the whole lens.
    It reads 1.29x against `incomeQuality`, and the design records that it FAILED under the
    previous budget pair, which is precisely why it is a test and not a comment.

    THE RATIO IS UNCHANGED BY THE 2026-08-06 S RE-BUDGET, and that is the property the
    proportional take was chosen for: both metrics are non-S, so both scale by the same
    0.880/0.914 and the margin is invariant.  It is asserted anyway -- a flat or hand-picked
    take of the same 0.034 would have moved it, so the invariance is a consequence of the
    METHOD, not a fact about the numbers."""
    ey = abs(sw.DEPLOYED[sw.THESIS_METRIC])
    other = _largest_other(sw.DEPLOYED, sw.THESIS_METRIC)
    assert ey > other, (ey, other)
    assert ey / other == pytest.approx(1.2928, abs=5e-4)


def test_the_margin_SURVIVES_the_D3_fix_and_this_is_the_forward_trap():
    """THE ONE CHANGE MOST LIKELY TO BREAK IT NEXT, checked before it lands.

    Fixing audit D3 promotes `grahamNumberToPrice` from P-E Tier 3 to Tier 2, so P-E splits
    3:2 instead of 3:1 and `earnYield` drops from 0.4875*W_P to 0.39*W_P.  It still clears
    `incomeQuality` -- but only by 0.0034, and ONLY because durability was held equal to
    cheapness: at the proportional alternative (W_R = 0.1748) `incomeQuality` would be
    0.10488 and the D3 fix would break the margin outright.

    THE HEADROOM SHRANK WITH THE 2026-08-06 S RE-BUDGET -- 0.00336 -> 0.003235 -- because both
    sides scale by 0.880/0.914 while their RATIO is unchanged.  So the trap is no more likely
    to spring than it was, but it is 3.7% closer in absolute terms, and any further take out of
    the non-S blocks shrinks it again.

    So this test is the instruction as much as the check: when D3 is fixed, re-run it, and do
    NOT fix D3 and a block budget in the same change."""
    post_d3 = dict(sw.GENERAL_ASSIGNMENT)
    block, sub, _tier, sign = post_d3['grahamNumberToPrice']
    post_d3['grahamNumberToPrice'] = (block, sub, 2, sign)
    v = sw._block_vector(sw.GENERAL_BUDGETS, post_d3)
    ey, iq = abs(v['earnYield']), abs(v['incomeQuality'])
    assert ey == pytest.approx(0.097628, abs=5e-6)
    assert iq == pytest.approx(0.094393, abs=5e-6)
    assert ey > iq, ('the D3 fix has broken the thesis margin -- see scoringWeights B.6; '
                     'the recorded remedy is W_P >= 0.27', ey, iq)
    assert ey - iq == pytest.approx(0.003235, abs=5e-6), 'and the headroom is this thin'


@pytest.mark.parametrize('label', ['FinManager'])
def test_the_margin_also_holds_in_the_non_exempt_cohorts(label):
    """Of the five cohorts, only FIN-2 is expected to carry the property: REIT and FIN-1 are
    exempt BY CONSTRUCTION (`earnYield` is Tier 3 in one and out of domain in the other),
    FIN-3 by declaration (ROE is the industry's own yardstick), and Mining by declaration
    (the cycle question is genuinely co-dominant -- see the next test)."""
    assert label not in sw.THESIS_MARGIN_EXEMPT_COHORTS
    v = sw.COHORT_WEIGHTS[label]
    ey = abs(v[sw.THESIS_METRIC])
    assert ey > _largest_other(v, sw.THESIS_METRIC), label
    assert set(sw.THESIS_MARGIN_EXEMPT_COHORTS) == set(sw.COHORT_LABELS) - {'FinManager'}


def test_the_Mining_exemption_is_recorded_AND_earned():
    """Mining is DECLARED exempt, and the reason is not "it fails" -- it currently PASSES at
    1.17x.  The exemption is that post-D3 the failure is NOT RECOVERABLE at any admissible
    budget: it would need W_P > 1.538 * W_N = 0.280 here, or W_N cut below its cohort
    deviation, and that deviation is the entire purpose of the cohort.

    Both halves are asserted, because an exemption recorded without its mechanism is just a
    disabled test."""
    assert 'Mining' in sw.THESIS_MARGIN_EXEMPT_COHORTS
    v = sw.COHORT_WEIGHTS['Mining']
    ey = abs(v[sw.THESIS_METRIC])
    assert ey > _largest_other(v, sw.THESIS_METRIC), 'today it passes'
    #  UNCHANGED by the 2026-08-06 S re-budget: every Mining block rescales by the same
    #  factor, and neither `earnYield` nor `CycleHeat` is an S metric.
    assert ey / _largest_other(v, sw.THESIS_METRIC) == pytest.approx(1.1707, abs=5e-4)

    post_d3 = dict(sw._cohort_assignment('Mining'))
    block, sub, _tier, sign = post_d3['grahamNumberToPrice']
    post_d3['grahamNumberToPrice'] = (block, sub, 2, sign)
    v2 = sw.normalise(sw._block_vector(sw._cohort_budgets('Mining'), post_d3))
    assert abs(v2['earnYield']) < _largest_other(v2, 'earnYield'), (
        'post-D3 Mining now passes the margin -- the recorded exemption may no longer be '
        'needed; re-read scoringWeights B.6 before deleting it')


# --------------------------------------------------------------------------- #
#  C.  the cohort budgets and vectors                                         #
# --------------------------------------------------------------------------- #
#  Section 15.7's table, re-derived by Rule PROP at the decided general anchor with the P:D
#  residual split 1:1.  The Mining and REIT rows are the two that show the rules doing work:
#  Mining's D is pinned BELOW its residual share by the contamination override (P takes the
#  freed weight), and REIT's P and D come out exactly equal to the general 0.26.
#
#  RE-DERIVED AT THE 2026-08-06 S BUDGETS.  Three of the five are a PURE RESCALE of the row
#  they replace (x 0.880/0.914 on every block), because their S RATIO equals the anchor's 0.10
#  and so their residual scales with everything else: Mining, REIT and FinManager.
#
#  **BalanceSheetFin IS NOT**, and it is the one row a reader must not skim.  Its S ratio is
#  0.14, i.e. ABOVE the 0.10 anchor, so Rule PROP hands it a LARGER absolute S increase
#  (0.1204 -> 0.1680) -- and FIN-3 spends none of it (capital adequacy is still invisible to
#  the pipeline, and `interestCoverage` is declared OOD there: for a bank, interest expense is
#  a cost of goods, not a financing charge).  That extra budget comes OUT OF THE P/D RESIDUAL,
#  so FIN-3's cheapness and durability fall by more than the proportional 3.7% (0.3194 ->
#  0.3001, i.e. -6.0%) and its reported unpriced risk rises 12.04% -> 16.80%.  That is Rule UNM
#  working as designed -- the solvency question got more expensive everywhere, and the one
#  cohort that cannot answer it honestly reports more of its budget unspent -- but it IS a real
#  re-ranking of the FIN-3 side-list, not a rescale.
DESIGN_COHORT_BUDGETS = {
    'Mining':            {'P': 0.2625, 'R': 0.1656, 'N': 0.1822, 'D': 0.1802,
                          'S': 0.1200, 'M': 0.0414, 'C': 0.0481},
    'REIT':              {'P': 0.2503, 'R': 0.1904, 'N': 0.0994, 'D': 0.2503,
                          'S': 0.1200, 'M': 0.0414, 'C': 0.0481},
    'InvestmentVehicle': {'P': 0.5500, 'R': 0.1500, 'N': 0.0000, 'D': 0.0000,
                          'S': 0.1500, 'M': 0.1500, 'C': 0.0000},
    'FinManager':        {'P': 0.2586, 'R': 0.1490, 'N': 0.1159, 'D': 0.2586,
                          'S': 0.1200, 'M': 0.0497, 'C': 0.0481},
    'BalanceSheetFin':   {'P': 0.3001, 'R': 0.0662, 'N': 0.1325, 'D': 0.3001,
                          'S': 0.1680, 'M': 0.0331, 'C': 0.0000},
}


@pytest.mark.parametrize('label', sorted(DESIGN_COHORT_BUDGETS))
def test_the_cohort_block_budgets_ARE_the_designed_ones(label):
    got = sw._cohort_budgets(label)
    assert got == pytest.approx(DESIGN_COHORT_BUDGETS[label], abs=5e-5), label
    assert sum(got.values()) == pytest.approx(1.0, abs=1e-12), label


@pytest.mark.parametrize('label', sorted(DESIGN_COHORT_BUDGETS))
def test_no_cohort_shaves_the_thesis_below_the_P_floor(label):
    """A SAFETY RAIL THAT DOES NOT CURRENTLY BIND -- said plainly, because the earlier version
    of this docstring claimed the floor "does real work in Mining", and that is false under the
    budgets we ship.

    Mining's contamination override gives P the residual, so it lands at 0.2625, ABOVE the
    0.250328 floor; every other cohort clears it as well.  The floor was load-bearing at the old
    proportional anchor (Rule PROP alone put Mining at 0.2256) and is kept against a future
    ratio edit -- but a test whose docstring asserts work it is not doing trains the reader to
    disbelieve the docstrings.  The assertion is still worth having; the claim was not."""
    assert sw._cohort_budgets(label)['P'] >= sw.GENERAL_BUDGETS['P'] - 1e-12


def test_the_P_floor_is_SLACK_everywhere_and_that_is_recorded_not_assumed():
    """The other half of the correction: assert that it does not bind, so the day it starts to
    bind is a visible event rather than a silent change of regime.  If this fails, some cohort
    ratio now pushes a thesis budget down onto the floor -- read Mining's note in D.1 and decide
    deliberately, rather than letting the clamp absorb it."""
    for label in sw.COHORT_LABELS:
        p = sw._cohort_budgets(label)['P']
        assert p > sw.GENERAL_BUDGETS['P'] + 1e-9 or label == 'REIT', (label, p)
    #  REIT lands EXACTLY on the general 0.26 -- by arithmetic coincidence of its own ratios,
    #  not by the clamp.  Pinned separately so "equal to the floor" is not read as "clamped".
    assert sw._cohort_budgets('REIT')['P'] == pytest.approx(sw.GENERAL_BUDGETS['P'], abs=1e-12)
    assert sw._cohort_budgets('Mining')['P'] == pytest.approx(0.2625, abs=5e-5)


def test_Rule_PROP_puts_the_residual_on_P_and_D_in_the_GENERAL_P_to_D_ratio():
    """The rule that replaced the P=D equality.  The equality is no longer a general
    property to propagate (it happens to hold today at 1:1), so what propagates is the
    RATIO -- which means the next time the CEO moves P or D the cohorts follow instead of
    drifting.  Checked on the three cohorts where both blocks take the residual."""
    ratio = sw.GENERAL_BUDGETS['P'] / sw.GENERAL_BUDGETS['D']
    for label in ('REIT', 'FinManager', 'BalanceSheetFin'):
        b = sw._cohort_budgets(label)
        assert b['P'] / b['D'] == pytest.approx(ratio, abs=1e-9), label


def test_REIT_sub_block_scaling_FIRES_and_hands_P_A_the_majority():
    """Rule 5, the only place it does work.  A property company's GAAP E is
    depreciation-dominated, so BOTH P-E members are Tier 3, P-E's share of the cheapness
    budget is scaled by tau_3/tau_1 = 1/3, and renormalising against P-A gives P-E 0.3824 /
    P-A 0.6176.  That reproduces what the pre-E-2 vector achieved by hand-zeroing
    `earnYield`, WITHOUT the collateral damage -- a REIT's earnings multiple is a poor
    instrument, not a meaningless one, and it still scores."""
    v = sw.COHORT_WEIGHTS_RAW['REIT']
    p_e = abs(v['earnYield']) + abs(v['grahamNumberToPrice'])
    p_a = abs(v['bVpRatio']) + abs(v['tbVpRatio'])
    assert p_e / (p_e + p_a) == pytest.approx(0.382353, abs=5e-6)
    assert v['earnYield'] == pytest.approx(v['grahamNumberToPrice'], abs=1e-12), \
        'both Tier 3 -> equal within tier'
    assert v['earnYield'] > 0, 'poor instrument, not zeroed'


def test_REIT_UNPARKS_its_survival_budget_and_FIN3_still_holds_ITS_unspent():
    """RULE UNM, AND THE EVENT IT WAS BUILT FOR (CEO, 2026-08-06).

    REIT's S budget was held unspent under a note reading "until net-debt/EBITDA OR INTEREST
    COVERAGE exists".  `interestCoverage` now exists as a Stage-2 metric, so the condition is
    DISCHARGED and REIT's unpriced risk goes 8.60% -> 0%.  `Altman-Z` and `currentRatio` stay
    OOD -- nothing about THEM improved -- so the block is a lone Tier-1 member holding the
    whole 0.120.

    FIN-3 IS THE CONTROL, and the reason this is one test rather than two: it did NOT unpark.
    Its S question is capital adequacy, which the pipeline still cannot see, and
    `interestCoverage` is OOD there because a bank's interest expense is a cost of goods rather
    than a financing charge.  So its held budget RISES with the general S budget (12.04% ->
    16.80%) instead of being spent -- which is Rule UNM's point: a question that got more
    expensive and that the cohort still cannot answer means MORE unpriced risk reported.

    THE LIMIT ON THE REIT HALF, recorded here because no assertion can carry it: interest
    coverage INSTRUMENTS the refinancing question, it does not ANSWER it.  A REIT's real
    solvency risk is the maturity wall and the LTV covenant, and a name that covers today's
    interest bill can still fail to roll its debt.  0% unpriced means "the block now has an
    instrument", not "REIT leverage risk is priced"."""
    reit = sw.COHORT_WEIGHTS_RAW['REIT']
    assert reit['currentRatio'] == 0.0 and reit['Altman-Z'] == 0.0
    assert reit['interestCoverage'] == pytest.approx(0.120, abs=1e-12), \
        'the lone Tier-1 member takes the whole S budget'
    assert sw.COHORT_UNPRICED_RISK['REIT'] == pytest.approx(0.0, abs=1e-12)

    fin3 = sw.COHORT_WEIGHTS_RAW['BalanceSheetFin']
    assert fin3['interestCoverage'] == 0.0, 'OOD: a bank pays interest as a cost of goods'
    assert fin3['currentRatio'] == 0.0 and fin3['Altman-Z'] == 0.0
    assert sw.COHORT_UNPRICED_RISK['BalanceSheetFin'] == pytest.approx(0.1680, abs=5e-5)

    for label in ('Mining', 'FinManager', 'InvestmentVehicle'):
        assert sw.COHORT_UNPRICED_RISK[label] == pytest.approx(0.0, abs=1e-12), label


def test_interestCoverage_carries_S_TIER_1_wherever_the_question_APPLIES():
    """The other half of the same decision, asserted rather than left as a side effect: it is a
    GENERAL metric, so Mining and FIN-2 inherit it and their S blocks restructure from tiers
    {2,3} to {1,2,3}.  `currentRatio` goes from two-thirds of S to one-third of it in both --
    a real shift in those two side-lists, beyond the proportional rescale."""
    for label, want in (('Mining', 0.060), ('FinManager', 0.060), ('REIT', 0.120)):
        assert sw.COHORT_WEIGHTS_RAW[label]['interestCoverage'] == pytest.approx(
            want, abs=1e-12), label
    for label in ('Mining', 'FinManager'):
        v = sw.COHORT_WEIGHTS_RAW[label]
        assert v['interestCoverage'] > v['currentRatio'] > v['Altman-Z'], label
    #  ... and it is OOD in exactly the two cohorts where the ratio is not a solvency reading
    for label in ('InvestmentVehicle', 'BalanceSheetFin'):
        assert sw.COHORT_WEIGHTS[label]['interestCoverage'] == 0.0, label


def test_FIN3_declares_ROE_above_earnYield_on_purpose():
    """The cohort's stated exemption: for a bank, capital employed is not a meaningful
    denominator and equity IS the regulated scarce resource, so ROE is the industry's own
    yardstick.  Pinned so that "the thesis is not the largest weight here" stays a decision
    rather than becoming a surprise."""
    v = sw.COHORT_WEIGHTS_RAW['BalanceSheetFin']
    assert v['returnOnEquity'] > v['earnYield']
    assert v['returnOnEquity'] == pytest.approx(0.180047, abs=5e-6)
    assert v['RoA'] == pytest.approx(0.120032, abs=5e-6)
    assert v['returnOnCapitalEmployed'] == 0.0, 'OOD: no meaningful capital-employed base'
    assert v['grossProfitMargin'] == 0.0, 'OOD: no gross margin exists'


# --------------------------------------------------------------------------- #
#  D.  FIN-1 -- the tier fix, and what the two new metrics change             #
# --------------------------------------------------------------------------- #
def test_the_FIN1_tier_fix_SURVIVES_the_rescale():
    """THE INVARIANT THE DESIGN ASKS TO BE CHECKED.  Promoting `tbVpRatio` to co-primary with
    `bVpRatio` took FIN-1's largest single column from 0.597 to 0.393.  FIN-1 is constructed
    directly rather than propagated from the general anchor, so the durability decision must
    not move it -- verified here by re-deriving the cohort WITHOUT the two new metrics, i.e.
    exactly the vector the design published."""
    assignment = {m: spec for m, spec in sw._cohort_assignment('InvestmentVehicle').items()
                  if m not in ('shareCountChange', 'longTermDebtChange',
                               'navPerShareGrowth')}
    v = sw.normalise(sw._block_vector(sw._cohort_budgets('InvestmentVehicle'), assignment))
    assert v['bVpRatio'] == pytest.approx(0.392857, abs=5e-6)
    assert v['tbVpRatio'] == pytest.approx(0.392857, abs=5e-6)
    assert v['marketCapRevQuants'] == pytest.approx(0.214286, abs=5e-6)


def test_the_two_new_metrics_UNPARK_FIN1s_R_and_S_blocks():
    """What the extraction actually buys, stated as numbers.  Before it, FIN-1 spent 0.70 of
    its budget and reported 30% unpriced -- R ("is the NAV real?") and S ("can it survive?")
    both held with no instrument.  Each new metric is its block's SOLE member, so it takes
    that block's whole budget, and the largest single column falls further, from 0.393 to
    0.275.

    THE HONEST CAVEAT, which no test can assert and which is recorded in scoringWeights D.3:
    two binary-derived columns INSTRUMENT those blocks; they do not fully ANSWER them.
    Reading the 0% unpriced figure as "FIN-1's NAV risk is now priced" would overclaim."""
    v = sw.COHORT_WEIGHTS['InvestmentVehicle']
    #  R NOW SPLITS 3:2 (2026-08-06): `navPerShareGrowth` took R's Tier 1, so
    #  `shareCountChange` -- which guards ONE way the NAV fails -- drops from the whole 0.15 to
    #  0.06.  S is untouched: `interestCoverage` is OOD in FIN-1 (no operating income), so
    #  `longTermDebtChange` keeps the whole 0.15.
    assert v['navPerShareGrowth'] == pytest.approx(0.09, abs=1e-12)
    assert v['shareCountChange'] == pytest.approx(-0.06, abs=1e-12)
    assert v['longTermDebtChange'] == pytest.approx(-0.15, abs=1e-12)
    assert v['interestCoverage'] == 0.0, 'OOD: a BDC has no operating income'
    assert v['bVpRatio'] == pytest.approx(0.275, abs=1e-12)
    assert v['tbVpRatio'] == pytest.approx(0.275, abs=1e-12)
    assert v['marketCapRevQuants'] == pytest.approx(0.15, abs=1e-12)
    assert max(abs(w) for w in v.values()) == pytest.approx(0.275, abs=1e-12)
    assert sw.COHORT_UNPRICED_RISK['InvestmentVehicle'] == pytest.approx(0.0, abs=1e-12)
    #  and the signs: more dilution / more leverage must be WORSE
    assert v['shareCountChange'] < 0 and v['longTermDebtChange'] < 0
    #  ... and MORE NAV compounding is BETTER, so this one is positive
    assert v['navPerShareGrowth'] > 0


def test_the_three_FIN1_only_metrics_score_in_NO_other_pool():
    """They are FIN-1-only instruments by design.  Asserted because a cohort-only metric that
    leaks a general-pool weight would re-weight the main deliverable silently."""
    for metric in ('shareCountChange', 'longTermDebtChange', 'navPerShareGrowth'):
        assert sw.DEPLOYED[metric] == 0.0, metric
        for label in sw.COHORT_LABELS:
            if label == 'InvestmentVehicle':
                continue
            assert sw.COHORT_WEIGHTS[label][metric] == 0.0, (label, metric)


def test_Piotroski_carries_NO_weight_in_FIN1_or_FIN3():
    """The `C` block is out of domain in both, on COMPONENT-level grounds -- 7 of 9 undefined
    or degenerate in FIN-1, 4-5 of 9 in FIN-3.  See the next test for why undefined is worse
    than useless here."""
    for label in ('InvestmentVehicle', 'BalanceSheetFin'):
        assert sw.COHORT_WEIGHTS[label]['Piotroski'] == 0.0, label
        assert sw.COHORT_WEIGHTS[label]['revenueGrowth'] == 0.0, label
    for label in ('Mining', 'REIT', 'FinManager'):
        assert sw.COHORT_WEIGHTS[label]['Piotroski'] > 0, label


def test_both_new_metrics_reach_the_REVIEW_PAGE_and_the_two_copies_agree():
    """A weighted metric absent from the review artifacts cannot be explained to the CEO.

    `PLAYBOOK_METRICS` is a hand-maintained allow-list that existed in TWO copies with nothing
    tying either to the weight vector -- so E-2's metrics could carry 0.30 of the FIN-1 vector
    and never appear on a FIN-1 page.  Measured before they were registered, FIN-1 explained
    **55.0%** of its own score (45% missing, not 30% -- `marketCapRevQuants` was already gone).
    """
    import generate_presentation as gp
    import reviewReference as rr
    for metric in ('shareCountChange', 'longTermDebtChange',
                   'interestCoverage', 'navPerShareGrowth'):
        assert metric in rr.PLAYBOOK_METRICS, metric
        assert metric in gp.PLAYBOOK_METRICS, metric
        assert metric in rr.METRIC_BASIS, '%s needs a declared basis/units' % metric
    #  the duplication is the hazard, so the two copies are pinned EQUAL, order included
    assert rr.PLAYBOOK_METRICS == gp.PLAYBOOK_METRICS, (
        'the two PLAYBOOK_METRICS copies have drifted: only-in-reviewReference=%s '
        'only-in-generate_presentation=%s'
        % (sorted(set(rr.PLAYBOOK_METRICS) - set(gp.PLAYBOOK_METRICS)),
           sorted(set(gp.PLAYBOOK_METRICS) - set(rr.PLAYBOOK_METRICS))))


def test_the_allow_list_is_TIED_to_the_weight_vector_now():
    """The guard itself -- nothing previously connected the allow-list to the weights, which is
    why the gap was silent.  Also pins that FIN-1 recovered to ~85%, so a regression in either
    direction is visible."""
    import reviewReference as rr
    covered = rr.assert_allow_list_covers_the_weighted_metrics()
    assert covered['InvestmentVehicle'] == pytest.approx(0.85, abs=0.005)
    #  general 0.855 -> 0.8968 on 2026-08-05 (register D-10): `grahamNumberToPrice` was
    #  resolved to be a UNITLESS ratio and moved off `_ALLOW_LIST_EXEMPT` onto the
    #  allow-list, so the general review page now explains 4.2pp more of its own score.
    #  The number moved because the page got better, not because the guard was loosened.
    #  general 0.897 -> 0.9006 on 2026-08-06: `interestCoverage` is on the allow-list and
    #  carries 0.060, so the page explains a further 0.4pp of its own score.
    assert covered['general'] == pytest.approx(0.9006, abs=0.005)
    #  and the residual is NAMED rather than merely tolerated
    for pool, (_cov, missing) in rr.allow_list_coverage().items():
        for m in missing:
            assert m in rr._ALLOW_LIST_EXEMPT, (pool, m)
            assert rr._ALLOW_LIST_EXEMPT[m].strip(), 'an exemption needs a REASON: %s' % m


def test_the_coverage_guard_actually_FIRES():
    """A guard is only worth having if it fails.  Drop a registered metric and the guard must
    name it -- otherwise the previous two tests are decoration."""
    import reviewReference as rr
    original = list(rr.PLAYBOOK_METRICS)
    try:
        rr.PLAYBOOK_METRICS.remove('shareCountChange')
        with pytest.raises(AssertionError) as ei:
            rr.assert_allow_list_covers_the_weighted_metrics()
        assert 'shareCountChange' in str(ei.value)
        assert 'generate_presentation' in str(ei.value), \
            'the message must name the SECOND copy, or half the fix gets made'
    finally:
        rr.PLAYBOOK_METRICS[:] = original
    assert rr.assert_allow_list_covers_the_weighted_metrics()


# --------------------------------------------------------------------------- #
#  E.  the two metrics themselves -- and the missing-data premise             #
# --------------------------------------------------------------------------- #
def _frame(rows):
    """A minimal NEWEST-FIRST cdx slice: rows[0] is the current period."""
    return pd.DataFrame(rows)


def _row(shares=100.0, ltd=50.0, ta=1000.0):
    return dict(weightedAverageShsOut=shares, longTermDebt=ltd, totalAssets=ta,
                netIncome=10.0, netCashProvidedByOperatingActivities=12.0,
                currentRatio=1.5, grossProfitMargin=0.4, revenue=500.0)


def _five(now, then):
    """rpy=4 needs five rows: row 0 = now, rows 1-3 filler, row 4 = one year ago."""
    return _frame([now, _row(), _row(), _row(), then])


def test_shareCountChange_measures_dilution_positive_and_buybacks_negative():
    """The sign convention: the column holds the change AS MEASURED and the weight (negative
    in FIN-1) decides that more of it is worse -- the same arrangement `CycleHeat` uses."""
    diluted = s2m.share_count_change(_five(_row(shares=110.0), _row(shares=100.0)))
    assert diluted == pytest.approx(0.10)
    bought_back = s2m.share_count_change(_five(_row(shares=90.0), _row(shares=100.0)))
    assert bought_back == pytest.approx(-0.10)
    flat = s2m.share_count_change(_five(_row(shares=100.0), _row(shares=100.0)))
    assert flat == 0.0


def test_longTermDebtChange_is_the_change_in_the_RATIO_not_the_raw_delta():
    """Faithful to Piotroski p5, which compares `longTermDebt / totalAssets` across the two
    periods -- and scale-free on both sides, which matters in a pool mixing GBp / SEK / USD /
    CAD reporters.  A company whose debt and assets BOTH double has not levered up."""
    levered_up = s2m.long_term_debt_change(
        _five(_row(ltd=200.0, ta=1000.0), _row(ltd=100.0, ta=1000.0)))
    assert levered_up == pytest.approx(0.10)
    deleveraged = s2m.long_term_debt_change(
        _five(_row(ltd=100.0, ta=1000.0), _row(ltd=200.0, ta=1000.0)))
    assert deleveraged == pytest.approx(-0.10)
    grew_proportionally = s2m.long_term_debt_change(
        _five(_row(ltd=200.0, ta=2000.0), _row(ltd=100.0, ta=1000.0)))
    assert grew_proportionally == pytest.approx(0.0)


@pytest.mark.parametrize('fn,field', [
    (s2m.share_count_change, 'weightedAverageShsOut'),
    (s2m.long_term_debt_change, 'longTermDebt'),
    (s2m.long_term_debt_change, 'totalAssets'),
])
@pytest.mark.parametrize('which', ['now', 'then'])
def test_a_MISSING_input_yields_NaN_never_a_pass_and_never_a_fail(fn, field, which):
    """THE PROJECT'S STANDING PREMISE, and the precise defect the parent composite has:
    `piotroski` scores an undefined component 0, which is indistinguishable from failing it.
    These return NaN, which `normalizeAndDropNA` median-centres and imputes to the column
    MEDIAN -- neither credit nor penalty.

    Note it must be NaN and not 0.0: on a NEGATIVE-weight column a 0.0 would read as "did not
    dilute" / "did not lever up", i.e. a PASS awarded for missing data."""
    now, then = _row(), _row()
    (now if which == 'now' else then)[field] = np.nan
    assert np.isnan(fn(_five(now, then)))


def test_they_are_NOT_a_fraction_of_computable_tests_passed():
    """The trap the brief names explicitly.  A "share of the applicable tests passed" form
    would REWARD a company for having fewer tests apply to it.  These are two independent
    columns: one being uncomputable must leave the other's value untouched and must not
    concentrate weight onto it."""
    now, then = _row(shares=np.nan, ltd=200.0), _row(shares=100.0, ltd=100.0)
    assert np.isnan(s2m.share_count_change(_five(now, then)))
    assert s2m.long_term_debt_change(_five(now, then)) == pytest.approx(0.10), \
        'the computable column is unaffected by its neighbour being absent'


def test_a_REPORTED_zero_long_term_debt_is_an_OBSERVATION_not_a_gap():
    """An unlevered vehicle genuinely has a leverage ratio of 0, and differencing two zeros
    genuinely gives 0.0.  NaN-ing that case would throw away a real reading.

    THE RECORDED LIMIT this exposes: because the metric is a CHANGE, the unlevered company
    reads 0.0 while a levered one that deleverages reads BETTER than it.  That is the
    extracted component's own semantics -- a level term would be a new metric, not an
    extraction (scoringWeights D.3)."""
    unlevered = s2m.long_term_debt_change(_five(_row(ltd=0.0), _row(ltd=0.0)))
    assert unlevered == 0.0
    deleveraged = s2m.long_term_debt_change(_five(_row(ltd=10.0), _row(ltd=200.0)))
    assert deleveraged < unlevered, 'the documented ordering, asserted so it is not a surprise'


def test_a_short_history_yields_NaN_rather_than_a_wrong_lag():
    """Fewer than rpy+1 rows means there is no same-period-one-year-ago row to compare
    against.  Reaching for the oldest available row instead would silently make the
    comparison span a different number of periods per name."""
    two_rows = _frame([_row(shares=110.0), _row(shares=100.0)])
    assert np.isnan(s2m.share_count_change(two_rows, rpy=4))
    assert np.isnan(s2m.long_term_debt_change(two_rows, rpy=4))
    #  ... and a SEMI-ANNUAL filer's year is 2 rows, so the same frame IS computable there
    assert s2m.share_count_change(_frame([_row(shares=110.0), _row(), _row(shares=100.0)]),
                                  rpy=2) == pytest.approx(0.10)


#  A PERFECT-9 pair, so that every point is EARNED and removing one is visible.  Against the
#  flat `_row()` pair only p1/p2/p4/p7 pass, which makes "the point was lost" indistinguish-
#  able from "the point was never won" -- the exact ambiguity these two tests exist to
#  resolve.
def _improved_now():
    return dict(weightedAverageShsOut=90.0,       # p7: bought back
                longTermDebt=100.0, totalAssets=1000.0,   # p5: leverage ratio 0.20 -> 0.10
                netIncome=10.0,                   # p1 / p3
                netCashProvidedByOperatingActivities=12.0,   # p2 / p4
                currentRatio=1.5,                 # p6
                grossProfitMargin=0.4,            # p8
                revenue=500.0)                    # p9


def _weaker_then():
    return dict(weightedAverageShsOut=100.0,
                longTermDebt=200.0, totalAssets=1000.0,
                netIncome=5.0,
                netCashProvidedByOperatingActivities=6.0,
                currentRatio=1.0,
                grossProfitMargin=0.3,
                revenue=400.0)


def _perfect_nine(**override):
    now = dict(_improved_now(), **override)
    return _frame([now, _row(), _row(), _row(), _weaker_then()])


def test_the_perfect_nine_fixture_really_scores_nine():
    """Guards the two tests below: if the fixture stops scoring 9, they would silently start
    measuring "a point that was never won" instead of "a point lost"."""
    assert s2m.piotroski(_perfect_nine()) == 9


def test_the_extraction_agrees_with_the_Piotroski_point_it_came_from():
    """Monotone consistency, asserted rather than claimed: sign(shareCountChange) <= 0 IS p7
    and sign(longTermDebtChange) < 0 IS p5, so nothing about making them continuous reverses
    Piotroski's own judgement.  Probed through `piotroski` itself by moving ONE input at a
    time off a perfect 9 and reading the composite's response."""
    diluted = _perfect_nine(weightedAverageShsOut=110.0)
    assert s2m.share_count_change(diluted) > 0
    assert s2m.piotroski(diluted) == 8, 'p7 lost exactly as the continuous metric turns +ve'

    levered = _perfect_nine(longTermDebt=300.0)
    assert s2m.long_term_debt_change(levered) > 0
    assert s2m.piotroski(levered) == 8, 'p5 lost exactly as the continuous metric turns +ve'

    #  and the other direction: the two metrics stay negative on the frame that scores 9
    assert s2m.share_count_change(_perfect_nine()) < 0
    assert s2m.long_term_debt_change(_perfect_nine()) < 0


def test_the_MECHANISM_the_C_block_OOD_ruling_rests_ON():
    """FLIPPED 2026-08-05, AND THE RULING HAS BEEN RE-READ RATHER THAN ASSUMED.

    This test used to pin the OPPOSITE code fact: that an undefined Piotroski component did NOT
    propagate NaN, so a NaN input made every comparison False, the point scored 0, and the
    composite was systematically PUNITIVE against a cohort with 7 of 9 components undefined.  It
    carried its own instruction for this moment -- "if this ever changes to NaN-propagation the
    OOD ruling still holds but its URGENCY drops from 'actively mis-ranks financials today' to
    'carries no information', so the ruling should be re-read rather than assumed."

    nan-policy.md section 4a made that change (stage2_metrics.piotroski now returns NaN when any
    component input is absent), so the re-read, recorded here:

      * THE C-BLOCK OOD RULING STANDS.  `Piotroski` is still carried at w = 0.0000 in FIN-1 and
        `BalanceSheetFin`.  A NaN carrier is no carrier: after this change the composite is NaN
        for those members, `normalizeAndDropNA` imputes it at the column MEDIAN, and a column
        that is near-constant after the fill spends its weight on nothing.  Zero weight is still
        the right answer -- for the second reason now, not the first.
      * ITS URGENCY DROPS, exactly as predicted.  The framing "actively mis-ranks financials
        today" is retired; the honest framing is "carries no information there".
      * WHAT THE CHANGE IS WORTH, measured [panel = baseline_tools/resdic_2026-07-17_CORRECTED]:
        117 sources (1.51%) [universe], 33 of the 4,287 general-carved names, 0 of the 100
        deployed pool names -- all of them via `netCashProvidedByOperatingActivities` (p2, p4).
        FIN-1's own exposure is only 4 of 79 members, because the "7 of 9 undefined" there is
        CONCEPTUAL inapplicability (a BDC has no gross margin) and FMP reports ZEROS for it, not
        NaNs.  A structural zero is not a NaN and no NaN rule reaches it.
      * SO DO NOT RECORD D-9 AS CLOSED BY THIS.  The structural-zero channel is ~6.5x larger
        (`longTermDebt == 0` in both periods fails p5 on 476 sources, forever) and needs a
        provider-level presence flag at ingest.
    """
    nanned = s2m.piotroski(_perfect_nine(grossProfitMargin=np.nan, currentRatio=np.nan))
    assert np.isnan(nanned), (
        'an undefined Piotroski component must make the COMPOSITE NaN. Scoring the point 0 is '
        'indistinguishable from FAILING the test, i.e. a mark-down for a provider gap -- which '
        'is what nan-policy.md section 4a removed. If this reverts, the C-block OOD ruling '
        'reverts to its "actively mis-ranks" framing too.')
    #  NOT a "fraction of the computable tests passed": that form would REWARD a company for
    #  having fewer tests apply to it.  The whole composite goes unavailable instead.
    assert np.isnan(s2m.piotroski(_perfect_nine(netIncome=np.nan)))
    assert np.isnan(s2m.piotroski(_perfect_nine(revenue=np.nan)))
    #  A REPORTED ZERO IS NOT ABSENT, and this is the boundary of what the fix reaches.  A
    #  PERMANENTLY unlevered company reports longTermDebt == 0 in BOTH periods, still scores,
    #  and still FAILS p5 -- `0 < 0` is False -- so it drops to 8 for a reason no NaN rule can
    #  see.  That is the structural-zero channel (476 sources, 6.17%), ~6.5x this fix and
    #  untouched by it by construction.
    _unlevered = _frame([dict(_improved_now(), longTermDebt=0.0), _row(), _row(), _row(),
                         dict(_weaker_then(), longTermDebt=0.0)])
    assert s2m.piotroski(_unlevered) == 8
    #  the EXTRACTED metrics already answered NaN for the same class of gap; the composite has
    #  now been brought into line with them rather than the other way round
    assert np.isnan(s2m.share_count_change(_perfect_nine(weightedAverageShsOut=np.nan)))
    assert np.isnan(s2m.long_term_debt_change(_perfect_nine(longTermDebt=np.nan)))


# --------------------------------------------------------------------------- #
#  F.  the two 2026-08-06 metrics themselves                                  #
# --------------------------------------------------------------------------- #
def _panel(n=16, **cols):
    """A minimal NEWEST-FIRST quarterly cdx slice carrying both new metrics' inputs."""
    base = dict(bookValuePerShare=5.0, operatingIncome=120.0, interestExpense=10.0)
    base.update({k: v for k, v in cols.items() if not isinstance(v, (list, tuple))})
    d = {k: [v] * n for k, v in base.items()}
    for k, v in cols.items():
        if isinstance(v, (list, tuple)):
            d[k] = list(v)
    d['date'] = pd.date_range('2026-01-01', periods=n, freq='-3ME')
    d['source'] = ['T'] * n
    d['reportingFrequency'] = ['quarterly'] * n
    return pd.DataFrame(d)


def _compounding(rate, n=16, start=5.0):
    """BVPS NEWEST-FIRST compounding at `rate`/yr: row 0 is the newest and largest."""
    return [start * (1.0 + rate) ** (-i / 4.0) for i in range(n)]


def test_navPerShareGrowth_recovers_the_ANNUAL_rate_it_was_given():
    """The arithmetic, on a series constructed to compound at a known rate.  A 16-row
    quarterly window spans (16-1)/4 = 3.75 years, so the endpoint ratio must be un-compounded
    by that and not by the row count."""
    assert s2m.nav_per_share_growth(_panel(bookValuePerShare=_compounding(0.10)),
                                    16, rpy=4) == pytest.approx(0.10, abs=1e-9)
    assert s2m.nav_per_share_growth(_panel(bookValuePerShare=_compounding(-0.05)),
                                    16, rpy=4) == pytest.approx(-0.05, abs=1e-9)
    #  a NAV that does not compound at all -- the signature the metric exists to catch
    assert s2m.nav_per_share_growth(_panel(), 16, rpy=4) == pytest.approx(0.0, abs=1e-12)


def test_navPerShareGrowth_REFUSES_a_non_positive_or_absent_endpoint():
    """NaN, never 0.0.  On a POSITIVE-weight column a 0.0 would assert "did not compound",
    which is a judgement made from missing data -- and a fractional power of a negative base
    is undefined anyway.  NaN imputes to the column median downstream."""
    for endpoint in (0, 15):          # newest row and the oldest row inside the window
        bv = [5.0] * 16
        bv[endpoint] = -1.0
        assert np.isnan(s2m.nav_per_share_growth(_panel(bookValuePerShare=bv), 16, rpy=4))
        bv[endpoint] = 0.0
        assert np.isnan(s2m.nav_per_share_growth(_panel(bookValuePerShare=bv), 16, rpy=4))
        bv[endpoint] = np.nan
        assert np.isnan(s2m.nav_per_share_growth(_panel(bookValuePerShare=bv), 16, rpy=4))


def test_navPerShareGrowth_scales_its_window_to_a_SEMI_ANNUAL_filer():
    """8 semi-annual rows are the same 3.5 CALENDAR years as 8 quarterly rows are 1.75 --
    so the same nominal window must return the same ANNUAL rate for either frequency, which
    is the whole reason `years` is derived from rpy rather than hard-coded."""
    q = s2m.nav_per_share_growth(_panel(n=16, bookValuePerShare=_compounding(0.08, 16)),
                                 16, rpy=4)
    sa = [5.0 * 1.08 ** (-i / 2.0) for i in range(8)]
    h = s2m.nav_per_share_growth(_panel(n=8, bookValuePerShare=sa), 16, rpy=2)
    assert q == pytest.approx(0.08, abs=1e-9)
    assert h == pytest.approx(0.08, abs=1e-9)


def test_interestCoverage_is_x_covered_and_REFUSES_a_debt_free_name():
    """The guard is the substantive half and it matches Stage-1's (createDicts /
    calcMetrics): FMP reports `interestExpense == 0` for a debt-free company, so dividing
    would mark it +/-inf -- a solvency verdict handed out for HAVING NO DEBT.  Refusing the
    row hands the leverage question to the rest of the block instead."""
    assert s2m.interest_coverage(_panel(), 16, rpy=4) == pytest.approx(12.0)
    assert np.isnan(s2m.interest_coverage(_panel(interestExpense=0.0), 16, rpy=4))
    assert np.isnan(s2m.interest_coverage(_panel(interestExpense=-4.0), 16, rpy=4))


def test_interestCoverage_scores_an_OPERATING_LOSS_rather_than_refusing_it():
    """The asymmetry with the guard above, and it is deliberate: a loss that cannot service
    the debt is a REAL and adverse reading -- the exact reading the S block exists to catch --
    whereas a missing denominator is not a reading at all."""
    loss = s2m.interest_coverage(_panel(operatingIncome=-30.0), 16, rpy=4)
    assert loss == pytest.approx(-3.0)
    assert loss < s2m.interest_coverage(_panel(), 16, rpy=4)


def test_the_two_new_metrics_are_REGISTERED_or_the_run_refuses():
    """`postBoScoreRanking` refuses a pool containing an unregistered metric, so a weight
    without a registry row is a dead pipeline rather than a silently-defaulted window."""
    assert s2m.unregistered_metrics(list(sw.METRIC_KEYS)) == []
    for key in ('interestCoverage', 'navPerShareGrowth'):
        assert s2m.flow_factor(key, 2) == 1.0, '%s is scale-free: no per-quarter factor' % key
        assert s2m.window_quarters(key, 16) == 16
        assert key in s2m.windowed_metric_keys()


def test_navPerShareGrowth_is_LABELLED_A_PROXY_where_the_CEO_reads_it():
    """It is book equity per share, not a fund-published NAV -- exact only under ASC 946.
    The caveat has to travel with the column into the review artifact, not just live in a
    docstring, because the artifact is what gets read."""
    import reviewReference as rr
    basis = rr.METRIC_BASIS['navPerShareGrowth']
    assert 'PROXY' in basis
    assert 'BOOK VALUE PER SHARE' in basis.upper()
    assert 'ASC-946' in basis or 'ASC 946' in basis
