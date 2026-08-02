"""THE ENFORCEMENT for the scoring-weight single source of truth (2026-08-02).

`scoringWeights.py` is now the only place a scoring weight is written down.  This suite
is what makes that claim checkable: it pins the SHAPE of `getPostDict()` (which
postBoRank and postBo consume), proves every other weight vector in the repo is a
DERIVATION of the canonical one rather than a copy, and pins the handful of deliberate
differences so nobody "tidies" one of them away.

WHAT IS AND IS NOT PINNED HERE, on purpose:
  * PINNED   -- structure (keys, order, eqMet, types), Sigma|w| = 1, derivations agreeing
                with the canon, and the specific DELIBERATE oddities the CEO owns
                (legacy DcfToPrice = 0.35, cohort BoScore = 0.1, three zeroed metrics).
  * NOT pinned -- the 21 weight VALUES.  A re-weighting is exactly what this refactor
                exists to make safe; a golden copy of the numbers here would be a 7th
                copy to hand-edit and would make the CEO's imminent re-weight harder,
                not safer.  Sigma|w| = 1 plus the derivation tests are what keep a
                re-weight honest.  Bit-identity of the CURRENT vector against the saved
                panel is proved separately by
                `baseline_tools/verify_weight_single_source.py`.

No network, no pickle reads, no API key.  Pure dict arithmetic + two cheap probes.
Run it the repo way: `pytest . --ignore=baseline_tools` (never with an explicit path --
that bypasses conftest.py's collect_ignore guard).
"""
import pytest

import calcScore as cs
import carveOut as co
import createDicts as cdic
import scoringWeights as sw


# --------------------------------------------------------------------------- #
#  getPostDict()'s return shape is the contract                               #
# --------------------------------------------------------------------------- #
POSTBM_ORDER = ('RoA', 'earnYield', 'grahamNumberToPrice', 'bVpRatio', 'revenueGrowth',
                'incomeQuality', 'returnOnEquity', 'returnOnCapitalEmployed',
                'currentRatio', 'grossProfitMargin')
POSTNEW_ORDER = ('freeCashFlowYield', 'freeCashFlowPerShareGrowth', 'DcfToPrice',
                 'marketCapRevQuants', 'Altman-Z', 'Piotroski', 'tbVpRatio', 'BoScore',
                 'EPStoEPSmean', 'priceGrowth', 'CycleHeat')
EQMET = {'RoA': 'returnOnAssets', 'earnYield': 'earningsYield',
         'grahamNumberToPrice': 'grahamNumberToPrice', 'bVpRatio': 'pbRatio',
         'revenueGrowth': 'revenue', 'incomeQuality': 'incomeQuality',
         'returnOnEquity': 'returnOnEquity',
         'returnOnCapitalEmployed': 'returnOnCapitalEmployed',
         'currentRatio': 'currentRatio', 'grossProfitMargin': 'grossProfitMargin'}


@pytest.mark.parametrize('getter', ['getPostDict', 'getPostDict_legacy'])
def test_getPostDict_shape_is_FROZEN(getter):
    """postBoRank builds `postScoreMetric_df`'s columns straight from `.keys()`, so key
    ORDER is as load-bearing as the key set.  Both vectors must present the identical
    shape -- that is what makes swapping legacy in an A/B rather than a different run."""
    postBm, postNew = getattr(cdic, getter)()
    assert tuple(postBm) == POSTBM_ORDER
    assert tuple(postNew) == POSTNEW_ORDER
    for k, v in postBm.items():
        assert tuple(v) == ('eqMet', 'w'), (k, tuple(v))
        assert v['eqMet'] == EQMET[k], (k, v['eqMet'])
    for k, v in postNew.items():
        assert tuple(v) == ('w',), (k, tuple(v))


@pytest.mark.parametrize('getter', ['getPostDict', 'getPostDict_legacy'])
def test_the_getters_hand_back_a_FRESH_dict_every_call(getter):
    """They used to build fresh literals per call, so a caller could scribble on the
    result harmlessly.  Now they assemble from module-level tables -- if they leaked a
    reference, one consumer's `w = 0` experiment would silently re-weight the pipeline."""
    a_bm, a_new = getattr(cdic, getter)()
    b_bm, b_new = getattr(cdic, getter)()
    assert a_bm is not b_bm and a_new is not b_new
    assert a_bm['RoA'] is not b_bm['RoA']          # inner dicts too
    a_bm['RoA']['w'] = 999.0
    a_new['CycleHeat']['w'] = 999.0
    c_bm, c_new = getattr(cdic, getter)()
    assert c_bm['RoA']['w'] != 999.0
    assert c_new['CycleHeat']['w'] != 999.0
    assert sw.DEPLOYED['RoA'] != 999.0 and sw.LEGACY['RoA'] != 999.0


def test_the_deployed_w_TYPES_are_unchanged():
    """Every deployed weight is a plain float (the legacy vector keeps its int literals);
    a numpy scalar or a Decimal sneaking in here would change dtype downstream."""
    postBm, postNew = cdic.getPostDict()
    for k, v in list(postBm.items()) + list(postNew.items()):
        assert type(v['w']) is float, (k, type(v['w']))
    lb, ln = cdic.getPostDict_legacy()
    for k, v in list(lb.items()) + list(ln.items()):
        assert type(v['w']) in (int, float), (k, type(v['w']))


# --------------------------------------------------------------------------- #
#  the invariant the pipeline's published numbers rest on                     #
# --------------------------------------------------------------------------- #
def test_the_deployed_vector_normalises_to_EXACTLY_one():
    """Sigma|w| = 1 is assumed by every published AggScore range, by the presentation
    chip, and by the cohort normalisation that was added to match this scale."""
    postBm, postNew = cdic.getPostDict()
    W = {**{k: v['w'] for k, v in postBm.items()},
         **{k: v['w'] for k, v in postNew.items()}}
    assert len(W) == 21
    assert sw.sum_abs(W) == 1.0, sw.sum_abs(W)


# --------------------------------------------------------------------------- #
#  no two copies can silently disagree -- the point of the refactor           #
# --------------------------------------------------------------------------- #
def test_every_named_vector_covers_EXACTLY_the_canonical_key_set():
    """THE core anti-drift test.  A metric added to DEPLOYED but forgotten in LEGACY or
    in a cohort does not raise downstream -- postBoRank.py and tune_run's `_finish` both
    do `weights.get(col, 1)`, so the forgotten metric scores at weight 1.0 against a
    vector whose other weights are ~0.05.  scoringWeights._validate() refuses at import;
    this asserts the refusal covers every vector."""
    canon = set(sw.METRIC_KEYS)
    assert len(sw.METRIC_KEYS) == 21
    assert set(sw.DEPLOYED) == canon
    assert set(sw.LEGACY) == canon
    for label, vec in sw.COHORT_WEIGHTS_RAW.items():
        assert set(vec) == canon, (label, sorted(canon ^ set(vec)))
    for label, vec in sw.COHORT_WEIGHTS.items():
        assert set(vec) == canon, (label, sorted(canon ^ set(vec)))


def test_validate_REFUSES_a_vector_that_drifts_off_the_canonical_keys(monkeypatch):
    """The guard is only worth having if it actually fires."""
    monkeypatch.setitem(sw.COHORT_WEIGHTS_RAW, 'Mining',
                        {k: 1.0 for k in list(sw.METRIC_KEYS)[:-1]})
    with pytest.raises(RuntimeError) as ei:
        sw._validate()
    assert 'canonical metric key set' in str(ei.value)
    assert 'weight 1.0' in str(ei.value), 'the message must name the failure mode'


def test_validate_REFUSES_a_deployed_vector_that_stops_normalising(monkeypatch):
    monkeypatch.setitem(sw.DEPLOYED, 'RoA', 0.5)
    with pytest.raises(RuntimeError) as ei:
        sw._validate()
    assert 'no longer normalises' in str(ei.value)


def test_tune_run_MU_GENERAL_IS_DERIVED_from_the_deployed_vector():
    """MU_GENERAL was a second set of 21 literals that had to track a re-weighting by
    hand.  It must now equal the deployed vector element for element, while remaining a
    separate object the tuner is free to copy and perturb."""
    tr = _import_baseline_tool('tune_run')
    postBm, postNew = cdic.getPostDict()
    W = {**{k: v['w'] for k, v in postBm.items()},
         **{k: v['w'] for k, v in postNew.items()}}
    assert tr.MU_GENERAL == W
    assert tr.MU_GENERAL is not sw.DEPLOYED, 'must be a copy, not the canon itself'
    # the one documented variant stays a documented variant
    assert tr.MU_GP_MODERATED['grossProfitMargin'] == 0.070
    assert {k: v for k, v in tr.MU_GP_MODERATED.items() if k != 'grossProfitMargin'} == \
           {k: v for k, v in W.items() if k != 'grossProfitMargin'}


def test_new_scorer_bench_W_THEORY_IS_derived_from_the_deployed_vector():
    """W_THEORY is |deployed| over 18 of the 21 keys with marketCapRevQuants renamed to
    the continuous _logmcap channel.  Assert the derivation, not the numbers."""
    nsb = _import_baseline_tool('new_scorer_bench')
    dep = sw.deployed_weights()
    expected = {nsb._BENCH_RENAME.get(k, k): abs(float(w))
                for k, w in dep.items() if k not in nsb._BENCH_EXCLUDED}
    assert nsb.W_THEORY == expected
    assert set(nsb.W_THEORY) == set(nsb.METRICS18)
    # key ORDER is behaviour here -- `_weighted` row-sums a DataFrame in this order, and
    # float addition is not associative.  Pinned, not inherited from METRIC_KEYS.
    assert tuple(nsb.W_THEORY) == nsb._BENCH_KEY_ORDER
    assert nsb._BENCH_KEY_ORDER[:3] == ('grossProfitMargin', 'Piotroski', 'incomeQuality')
    assert abs(sum(nsb.W_THEORY.values()) - 1.0) < 1e-9
    # the sign lives in CFG, so the weight here must be the MAGNITUDE
    assert dep['CycleHeat'] < 0 and nsb.W_THEORY['CycleHeat'] > 0
    assert nsb.CFG['CycleHeat'][1] == -1
    # excluding a metric is only score-neutral while it is zero
    for k in nsb._BENCH_EXCLUDED:
        assert float(dep[k]) == 0.0, k


# --------------------------------------------------------------------------- #
#  the DELIBERATE differences -- pinned so nobody tidies them away            #
# --------------------------------------------------------------------------- #
def test_the_three_zeroed_metrics_are_still_zero_in_the_deployed_vector():
    """DcfToPrice / BoScore / priceGrowth are zeroed by decision, and three separate
    things rest on it: postBoRank's offline-DCF score-neutrality refusal, the PIT
    reproduction's DROP_METRICS, and new_scorer_bench's 18-channel exclusion."""
    assert sw.DELIBERATELY_ZEROED == ('DcfToPrice', 'BoScore', 'priceGrowth')
    for k in sw.DELIBERATELY_ZEROED:
        assert sw.DEPLOYED[k] == 0.000, (k, sw.DEPLOYED[k])


def test_the_legacy_vector_KEEPS_its_0_35_DcfToPrice():
    """A REAL difference, not a bug: legacy weighted the (now broken) DCF metric at 0.35
    where the deployed vector drops it.  It must survive any weight housekeeping -- it is
    what makes `test_the_deployed_DcfToPrice_weight_really_IS_zero` non-trivial."""
    assert sw.LEGACY['DcfToPrice'] == 0.35
    assert sw.DEPLOYED['DcfToPrice'] == 0.000
    _lb, ln = cdic.getPostDict_legacy()
    assert ln['DcfToPrice']['w'] == 0.35


def test_the_legacy_vector_is_NOT_the_deployed_one():
    """Belt-and-braces on the derivation: a bug that pointed both getters at DEPLOYED
    would leave every shape test green while silently killing the A/B arm."""
    assert sw.LEGACY != sw.DEPLOYED
    differ = [k for k in sw.METRIC_KEYS
              if float(sw.LEGACY[k]) != float(sw.DEPLOYED[k])]
    assert len(differ) == 21, 'all 21 differ; got %d' % len(differ)


def test_cohort_BoScore_stays_NON_ZERO_in_all_five_vectors():
    """KNOWN OPEN ISSUE the CEO has not ruled on: BoScore is 0.000 in the general vector
    (dropped) but 0.1 raw in every cohort, so cohort side-lists still weight it.  Pinned
    to keep it from being silently 'fixed' as a side effect of some other change --
    changing it needs a CEO decision, not a tidy-up."""
    assert len(sw.COHORT_WEIGHTS_RAW) == 5
    for label, vec in sw.COHORT_WEIGHTS_RAW.items():
        assert vec['BoScore'] == 0.1, (label, vec['BoScore'])
    for label, vec in sw.COHORT_WEIGHTS.items():
        assert vec['BoScore'] > 0, (label, vec['BoScore'])


def test_cohort_priceGrowth_and_DcfToPrice_stay_zero_in_all_five():
    """Domain review S5: these were non-zero on all five cohort paths and were zeroed."""
    for label, vec in sw.COHORT_WEIGHTS_RAW.items():
        assert vec['priceGrowth'] == 0, (label, vec['priceGrowth'])
        assert vec['DcfToPrice'] == 0, (label, vec['DcfToPrice'])


def test_the_five_cohort_vectors_are_UNCHANGED_and_normalise_to_one():
    """Domain review S7.  The raw sums are a cheap checksum over 105 numbers: any single
    cohort weight edit moves its cohort's sum, and the normalisation must still land on
    exactly 1 so cohort AggScores stay on the general scale."""
    expected_raw_sums = {
        'Mining': 13.85, 'REIT': 7.10, 'InvestmentVehicle': 3.35,
        'FinManager': 14.10, 'BalanceSheetFin': 10.35,
    }
    assert set(sw.COHORT_WEIGHTS_RAW) == set(expected_raw_sums)
    for label, want in expected_raw_sums.items():
        got = sw.sum_abs(sw.COHORT_WEIGHTS_RAW[label])
        assert abs(got - want) < 1e-9, (label, got, want)
        assert abs(sw.sum_abs(sw.COHORT_WEIGHTS[label]) - 1.0) < 1e-12, label


def test_carveOut_still_re_exports_both_cohort_names():
    """Every consumer reads these off carveOut (postBo, tune_run, reviewReference, the
    hardening suite), so the re-export is part of the contract, not a convenience."""
    assert co.COHORT_WEIGHTS is sw.COHORT_WEIGHTS
    assert co.COHORT_WEIGHTS_RAW is sw.COHORT_WEIGHTS_RAW
    assert set(co.COHORT_WEIGHTS) == {'REIT', 'Mining', co.FIN1_VEHICLE,
                                      co.FIN2_MANAGER, co.FIN3_BALSHEET}


# --------------------------------------------------------------------------- #
#  the duplicate Stage-1 criterion (dEPS / dNetIncomePerShare)                #
# --------------------------------------------------------------------------- #
def _tier_weight(tier):
    """The tier's weight, probed from calcByTier ITSELF -- deliberately not a mirrored
    table, since a mirrored table is the defect class this whole change removes.  One
    all-passing observation returns exactly the tier weight."""
    import pandas as pd
    return float(cs.calcByTier('diff', tier, 1, pd.Series([1.0]), 0.0, 'probe', 1))


def test_the_duplicate_criterion_pair_is_still_byte_identical_modulo_Tier():
    """If someone edits one half of the pair, the two stop being the same criterion and
    the DUPLICATE_DIFF_CRITERIA declaration (and this guard) no longer describe reality."""
    diff = cdic.getDicts()[4]
    for carrier, twin in cdic.DUPLICATE_DIFF_CRITERIA:
        assert carrier in diff and twin in diff, (carrier, twin)
        strip = lambda spec: {k: v for k, v in spec.items() if k != 'Tier'}
        assert strip(diff[carrier]) == strip(diff[twin]), (carrier, twin)


def test_at_most_ONE_half_of_the_duplicate_pair_can_CARRY_WEIGHT():
    """THE GUARD.  `EPS` (Tier B, w=0.5) and `netIncomePerShare` (Tier N, w=0) are the
    same criterion on the same quantity; the panel even carries two byte-identical
    columns for them.  Tier N maps to w=0 in calcByTier, so the duplicate contributes
    EXACTLY nothing today (verified end-to-end: dropping the entry leaves Stage-1
    bitwise identical on the 2026-07-17 panel).  The registry entry is kept rather than
    deleted -- deleting it changes BoMetric_df's schema and the Stage-1 NaN-accounting
    readout -- so this test is what stops a future TIER change from silently
    double-counting one quantity at 0.5 + w."""
    diff = cdic.getDicts()[4]
    assert _tier_weight('N') == 0.0, 'Tier N must be weightless for the pair to be inert'
    for carrier, twin in cdic.DUPLICATE_DIFF_CRITERIA:
        weighted = [k for k in (carrier, twin) if _tier_weight(diff[k]['Tier']) > 0]
        assert len(weighted) <= 1, (
            'BOTH halves of the duplicate criterion %r/%r now carry weight (%s) -- the '
            'same quantity would be scored twice. Decide: keep one, or make them '
            'genuinely different criteria.' % (carrier, twin, weighted))
    # and specifically, today: EPS carries it, netIncomePerShare does not
    assert diff['EPS']['Tier'] == 'B' and _tier_weight('B') == 0.5
    assert diff['netIncomePerShare']['Tier'] == 'N'


def test_the_tier_weight_probe_matches_the_documented_ladder():
    """Sanity on the probe itself, so a calcByTier refactor cannot quietly turn the guard
    above into a tautology."""
    assert [_tier_weight(t) for t in ('S', 'A', 'B', 'C', 'D')] == [1, 0.75, 0.5, 0.3, 0.1]
    assert _tier_weight('N') == 0.0
    assert _tier_weight('not-a-tier') == 0.0


# --------------------------------------------------------------------------- #
def _import_baseline_tool(name):
    """Import a baseline_tools module by adding the folder to sys.path, the way the tools
    import each other.  Kept out of the module body so merely COLLECTING this file has no
    side effects, and APPENDED rather than prepended so a repo-root module can never be
    shadowed by a same-named tool (there are none today; this keeps it that way).  Note
    the tools themselves prepend their own folder on import."""
    import importlib
    import os
    import sys
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_tools')
    if here not in sys.path:
        sys.path.append(here)
    os.environ.setdefault('VA_OFFLINE_NO_DCF', '1')   # before postBoRank is reached
    return importlib.import_module(name)
