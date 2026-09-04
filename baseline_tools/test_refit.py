"""Tests on the RE-FIT HARNESS ITSELF -- not on any result it produces.

The four things that must be structurally true, because each is a way the harness could
produce a confident number that means nothing:
  1. the box constraints BIND (the optimiser cannot leave [0.75, 1.3333]);
  2. SIGNS CANNOT FLIP;
  3. the HOLDOUT is genuinely excluded from every CV fold;
  4. the acceptance threshold is READ FROM THE PRE-REGISTRATION, not computed post hoc.
Plus the refuse-to-fit list and the sum-constraint invariant.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refit
import scoringWeights as sw


# --------------------------------------------------------------------------- #
#  1. Boxes bind                                                              #
# --------------------------------------------------------------------------- #
def test_grid_never_leaves_the_prereg_box():
    lo, hi = refit.PREREG["box"]
    assert min(refit.GRID) >= lo - 1e-12
    assert max(refit.GRID) <= hi + 1e-12
    assert 1.0 in refit.GRID, "the prior itself must be a candidate"


def test_box_binds_on_REALISED_multipliers_not_just_the_nominal_grid():
    """Defect M1.  The sum-constraint renormalisation (k = budget/scaled) can push the
    multiplier that ACTUALLY reaches the score outside the box even when every nominal grid
    value is inside it -- measured range was 0.6116x-1.5559x against [0.75, 1.3333].  So the
    box has to be checked on realised values, and `in_box` is what the grid filters on."""
    prior = refit.prior_weights()
    lo, hi = refit.PREREG["box"]
    import itertools
    n_nominal_ok_but_realised_out = 0
    for combo in itertools.product(refit.GRID, repeat=len(refit.CLUSTERS)):
        mult = dict(zip(refit.CLUSTERS, combo))
        r = refit.realised_multipliers(prior, mult)
        outside = any(v < lo - 1e-9 or v > hi + 1e-9 for v in r.values())
        assert refit.in_box(prior, mult) == (not outside)
        if outside:
            n_nominal_ok_but_realised_out += 1
    assert n_nominal_ok_but_realised_out > 0, \
        "if no grid point is nominally-in but realised-out, this guard is untested"


def test_every_in_box_point_has_realised_multipliers_inside_the_box():
    prior = refit.prior_weights()
    lo, hi = refit.PREREG["box"]
    import itertools
    for combo in itertools.product(refit.GRID, repeat=len(refit.CLUSTERS)):
        mult = dict(zip(refit.CLUSTERS, combo))
        if not refit.in_box(prior, mult):
            continue
        for m, v in refit.realised_multipliers(prior, mult).items():
            assert lo - 1e-9 <= v <= hi + 1e-9, "%s realised %.4f outside box" % (m, v)


def test_box_is_tighter_than_the_old_half_to_double():
    """The box should express the noise floor, not a shrug.  Guards against someone
    widening it back to [0.5, 2] without a decision."""
    lo, hi = refit.PREREG["box"]
    assert lo >= 0.70 and hi <= 1.40, "box widened beyond the registered [0.75, 1.33]"


# --------------------------------------------------------------------------- #
#  2. Signs cannot flip                                                       #
# --------------------------------------------------------------------------- #
def test_signs_are_preserved_across_the_whole_grid():
    prior = refit.prior_weights()
    import itertools
    for combo in itertools.product(refit.GRID, repeat=len(refit.CLUSTERS)):
        mult = dict(zip(refit.CLUSTERS, combo))
        w = refit.apply_multipliers(prior, mult)
        for m, v in w.items():
            if prior[m] == 0:
                assert v == 0, "%s had w=0 and must stay 0" % m
            else:
                assert np.sign(v) == np.sign(prior[m]), \
                    "sign flip on %s at %s" % (m, mult)


def test_cycleheat_is_FULLY_frozen_sign_and_magnitude():
    """The refuse-to-fit key claims sign AND magnitude.  It used to sit inside a fitted
    cluster, so the multiplier scaled its magnitude while the label said otherwise.  It must
    now be outside every cluster and byte-identical on every grid point."""
    prior = refit.prior_weights()
    assert prior["CycleHeat"] < 0
    assert not any("CycleHeat" in members for members in refit.CLUSTERS.values()), \
        "CycleHeat is inside a fitted cluster -- its magnitude would be scaled"
    import itertools
    for combo in itertools.product(refit.GRID, repeat=len(refit.CLUSTERS)):
        w = refit.apply_multipliers(prior, dict(zip(refit.CLUSTERS, combo)))
        assert w["CycleHeat"] == prior["CycleHeat"], "CycleHeat magnitude moved"
        assert w["CycleHeat"] < 0


def test_refuse_to_fit_keys_name_metrics_that_exist_and_are_not_in_clusters():
    """Guards the label-vs-code class directly: every refused key must be a real metric that
    no fitted cluster touches.  A key like `CycleHeat_sign_and_magnitude` that does not
    resolve to a metric is exactly how the earlier mismatch hid."""
    prior = refit.prior_weights()
    for key in refit.REFUSE_TO_FIT:
        assert key in prior, ("refuse-to-fit key %r is not a metric name -- it cannot be "
                              "verified against the code" % key)
        assert not any(key in members for members in refit.CLUSTERS.values()), \
            "%s is refused but sits in a fitted cluster" % key


def test_apply_multipliers_asserts_on_a_negative_multiplier():
    """A negative multiplier is the one input that WOULD flip signs -- it must raise, not
    silently produce an inverted vector."""
    prior = refit.prior_weights()
    with pytest.raises(AssertionError):
        refit.apply_multipliers(prior, {**{c: 1.0 for c in refit.CLUSTERS},
                                        "cheapness": -1.0})


# --------------------------------------------------------------------------- #
#  Sum constraint + refuse-to-fit                                             #
# --------------------------------------------------------------------------- #
def test_fitted_budget_and_total_are_preserved():
    prior = refit.prior_weights()
    fitted = [m for c in refit.CLUSTERS.values() for m in c]
    b0 = sum(abs(prior[m]) for m in fitted)
    t0 = sum(abs(v) for v in prior.values())
    for combo in ((0.75, 1.3333, 0.75, 1.3333), (1.3333, 0.75, 1.15, 0.875)):
        w = refit.apply_multipliers(prior, dict(zip(refit.CLUSTERS, combo)))
        assert abs(sum(abs(w[m]) for m in fitted) - b0) < 1e-12, "fitted budget moved"
        assert abs(sum(abs(v) for v in w.values()) - t0) < 1e-12, "total |w| moved"


def test_refused_metrics_are_never_touched():
    prior = refit.prior_weights()
    frozen = ["marketCapRevQuants", "Altman-Z"]
    for f in frozen:
        assert not any(f in members for members in refit.CLUSTERS.values()), \
            "%s is on the refuse-to-fit list but sits inside a fitted cluster" % f
    import itertools
    for combo in itertools.product([0.75, 1.3333], repeat=len(refit.CLUSTERS)):
        w = refit.apply_multipliers(prior, dict(zip(refit.CLUSTERS, combo)))
        for f in frozen:
            assert w[f] == prior[f], "%s moved" % f


def test_refuse_to_fit_list_carries_a_reason_for_each_entry():
    for k, v in refit.REFUSE_TO_FIT.items():
        assert isinstance(v, str) and len(v) > 30, "%s has no substantive reason" % k


# --------------------------------------------------------------------------- #
#  3. The holdout is excluded from every fold                                 #
# --------------------------------------------------------------------------- #
class _FakeAnchor(dict):
    """Minimal anchor stub: `score_anchor` only needs normed / group_of / excess."""
    def __init__(self, tag, n=40, seed=0):
        rng = np.random.default_rng(seed)
        cols = [m for c in refit.CLUSTERS.values() for m in c] + \
               ["marketCapRevQuants", "Altman-Z", "Piotroski", "incomeQuality",
                "currentRatio", "revenueGrowth", "freeCashFlowYield",
                "freeCashFlowPerShareGrowth", "DcfToPrice", "BoScore", "priceGrowth"]
        src = ["%s_%03d" % (tag, i) for i in range(n)]
        d = {"source": src}
        for c in cols:
            d[c] = rng.normal(size=n)
        super().__init__(buy=tag, eval="x", normed=pd.DataFrame(d),
                         group_of={s: s for s in src},
                         excess={s: float(v) for s, v in zip(src, rng.normal(size=n))},
                         bench=0.0, n_pool=n, n_priced=n)


def test_holdout_is_in_no_fold(monkeypatch):
    """The fold builder must never see the holdout.  Enforced by recording every anchor
    `score_anchor` is called with during `fit` AND `lowo`, and asserting the holdout is
    absent from both."""
    train = [_FakeAnchor("TR%d" % i, seed=i) for i in range(4)]
    _holdout = _FakeAnchor("HOLD", seed=99)
    seen = []
    real = refit.score_anchor

    def spy(anchor, w, topn=20, threshold=0.10):
        seen.append(anchor["buy"])
        return real(anchor, w, topn, threshold)

    monkeypatch.setattr(refit, "score_anchor", spy)
    refit.fit(train, refit.prior_weights(), verbose=False)
    refit.lowo(train, refit.prior_weights(), verbose=False)
    assert seen, "fit did not evaluate anything"
    assert "HOLD" not in seen, "THE HOLDOUT LEAKED INTO A CV FOLD"
    assert set(seen) <= {"TR0", "TR1", "TR2", "TR3"}


def test_lowo_scores_the_HELD_OUT_anchor_not_the_training_anchors(monkeypatch):
    """H-BLOCKER 2.  The old implementation scored each candidate on the fold's TRAINING
    anchors, so every fold value equalled the in-sample mean to 5.6e-17.  A genuine LOWO must
    score the anchor the fold never saw -- so the recorded out-of-sample value has to be able
    to DIFFER from the in-sample mean."""
    # anchors with deliberately different outcome structure per anchor, so in-sample and
    # out-of-sample cannot coincide by construction
    train = [_FakeAnchor("TR%d" % i, n=60, seed=7 * i + 1) for i in range(4)]
    folds = refit.lowo(train, refit.prior_weights(), verbose=False)
    assert len(folds) == 4
    diffs = [abs(f["OUT_of_sample_fit"] - f["IN_sample_mean_on_other_folds"])
             for f in folds]
    assert max(diffs) > 1e-6, ("every fold's out-of-sample value equals its in-sample mean "
                               "-- this is the in-sample mean wearing a CV label")
    for f in folds:
        assert f["fold_held_out"] in {"TR0", "TR1", "TR2", "TR3"}
        assert "OUT_of_sample_prior" in f and "n_priced_held_out" in f


def test_lowo_refits_per_fold(monkeypatch):
    """Each fold must re-run the SELECTION, not reuse one vector fitted on everything."""
    train = [_FakeAnchor("TR%d" % i, n=60, seed=13 * i + 5) for i in range(4)]
    selects = []
    real = refit._select

    def spy(anchors, prior, lam, topn=20, verbose=False, tag="", noise_floor=None):
        selects.append(tuple(sorted(a["buy"] for a in anchors)))
        return real(anchors, prior, lam, topn, verbose, tag, noise_floor)

    monkeypatch.setattr(refit, "_select", spy)
    refit.lowo(train, refit.prior_weights(), verbose=False, n_perm=3)
    assert len(selects) == 4, "expected one selection per fold"
    for s in selects:
        assert len(s) == 3, "each fold must fit on exactly n-1 anchors"
    assert len(set(selects)) == 4, "the four folds must use four different anchor subsets"


def test_one_se_is_sd_over_sqrt_n_not_the_sd_of_jackknife_means():
    """H-BLOCKER 1.  sd(jackknife means) = sd(anchors)/(n-1), i.e. 3x too narrow at n=4, and
    that silently disables the parsimony safeguard.  `_cv_score` must return sd/sqrt(n)."""
    class Stub(dict):
        def __init__(self, val):
            super().__init__(_val=val)

    vals = [0.10, 0.30, 0.35, 0.45]
    stubs = [Stub(v) for v in vals]
    import refit as R
    real = R.score_anchor
    R.score_anchor = lambda a, w, topn=20, threshold=0.10: {"beat_rate": a["_val"]}
    try:
        mean, se, arr = R._cv_score(stubs, R.prior_weights())
    finally:
        R.score_anchor = real
    exp_sd = float(np.std(vals, ddof=1))
    assert mean == pytest.approx(np.mean(vals))
    assert se == pytest.approx(exp_sd / np.sqrt(4)), "SE is not sd/sqrt(n)"
    # and it is NOT the jackknife-mean sd, which would be 3x smaller
    jack = [np.mean([v for j, v in enumerate(vals) if j != i]) for i in range(4)]
    assert se != pytest.approx(float(np.std(jack, ddof=1))), "SE is the jackknife-mean sd"
    assert se == pytest.approx(3.0 * float(np.std(jack, ddof=1)) / np.sqrt(4) * 1.0,
                               rel=1e-6), "the 3x relationship should hold at n=4"


def test_selection_noise_floor_is_positive_on_pure_noise():
    """The floor must MEASURE something: on anchors with no learnable signal, the best of ~380
    candidates still beats the prior, and that gap is what the floor has to capture."""
    prior = refit.prior_weights()
    anchors = [_FakeAnchor("N%d" % i, n=60, seed=i) for i in range(4)]
    floor, samples = refit.selection_noise_floor(anchors, prior, lam=0.02, n_perm=6)
    assert len(samples) == 6
    assert floor > 0, "a grid search over 380 candidates cannot find zero improvement on noise"
    assert all(s >= 0 for s in samples)


def test_prior_is_retained_when_improvement_does_not_clear_the_gate():
    """The mechanism, tested deterministically rather than by Monte Carlo: with an enormous
    noise floor no candidate can clear the gate, so the pick must be EXACTLY the prior."""
    prior = refit.prior_weights()
    anchors = [_FakeAnchor("N%d" % i, n=60, seed=i) for i in range(4)]
    mult, _g, _se, _n = refit._select(anchors, prior, lam=0.02, verbose=False,
                                     noise_floor=1e6)
    assert all(v == 1.0 for v in mult.values()), mult


def _opposing_clusters():
    """The two clusters to plant a signal ACROSS, chosen BY BUDGET from the live prior.

    WHY THIS IS DERIVED AND NOT NAMED (fix, 2026-08-04 -- issue E-2).  It used to hard-code
    "cheapness up, gross_margin down", with the documented premise that the improvement must be
    REACHABLE given the clusters' budgets and the +-33% multiplier cap.  That premise is a fact
    about the DEPLOYED VECTOR, not about the gate: E-2 moved cheapness 0.1595 -> 0.2600 and
    gross_margin 0.1000 -> 0.0217, and with only 0.0217 left to take away the multipliers could
    no longer buy enough improvement to clear the gate.

    THE ORIGINAL TEST WAS NOT FLAKY -- correcting my own earlier claim here, which said it was.
    Pre-E-2 it cleared at 1.23-1.29x on EVERY floor draw, deterministically, because the binding
    gate there was the median standard error and not the wobbling noise floor.  Post-E-2 with the
    old pair it fails at 0.51-0.68x -- below the ENTIRE floor range.  Neither state is a coin
    flip: it was a clean pass, then a clean fail.  (The knife-edge I measured at ~0.97x is a
    property of THIS new pair at the OLD n=100, i.e. of my own intermediate fix -- not of the
    test as it shipped.  See PLANTED_POOL_N.)

    Silently raising `strength` until it passed would have been tuning a test to green -- and,
    measured, it could not even have worked (the objective is saturated; see PLANTED_POOL_N).
    The honest fix is to remove the coupling: the property under test is "a FAIL means no signal,
    not that the gate can never fire", which is a property of the gate MACHINERY and must hold
    against whatever vector ships.

    WHY THE TWO LARGEST BUDGETS, stated with its actual limit.  A cluster's reachable swing is
    `budget x (multiplier range)`, so ordering by budget is a good PROXY for ordering by
    reachable swing -- it is NOT a maximum "by construction", which is what I first wrote and is
    only true under uniform within-cluster weights.  `cheapness` is 49% `earnYield` alone, so a
    plant spread evenly across its four members does not exploit its budget evenly.  The proxy is
    adequate here (it is measured to clear comfortably, below) and it removes the hard-coding;
    it is not an optimality claim.

    Returns (up_cluster, down_cluster).  Today: cheapness (0.2600) vs profitability (0.1517).
    """
    prior = refit.prior_weights()
    by_budget = sorted(refit.CLUSTERS,
                       key=lambda c: -sum(abs(prior[m]) for m in refit.CLUSTERS[c]))
    return by_budget[0], by_budget[1]


#  POOL SIZE PER PLANTED ANCHOR.  Raised 100 -> 250 (E-2, 2026-08-04).
#
#  WHY NOT `strength`.  The achievable improvement is SATURATED: sweeping `strength` from 2.0 to
#  6.0 moved it by 0.0014 (0.0591 -> 0.0605), against a gate of 0.059-0.066.  The ceiling is not
#  how much signal is planted but how much a +-33% cluster multiplier can buy on a top-20
#  beat-rate, so no amount of strength produces a ROBUST pass -- it can only produce a lucky one.
#  That is measured, not asserted, and it is why the knob was rejected rather than turned.
#
#  n ATTACKS THE OTHER SIDE.  A deeper pool gives the top-20 objective more room, while the
#  selection noise floor is ~80 Bernoulli draws and is barely a function of n -- so the RATIO
#  improves.  Measured at n=250 / 4 anchors across three independent seed bases: improvement
#  0.1095 / 0.1095 / 0.1350 against gates 0.0629 / 0.0692 / 0.0701 = 1.74x / 1.58x / 1.93x,
#  clearing on every draw with the direction correct every time (up 1.15, down 0.875).
#
#  THIS IS TUNE-TO-ROBUST, NOT A MEASURED OPTIMUM -- correcting my own earlier claim.  n=150 is
#  also 6/6 (1.25-1.64x) and is just as defensible; 250 is not special, it is simply comfortably
#  inside the robust region.  What IS a real finding is that n=400 is WORSE (1.22x on one seed),
#  and the mechanism is not noise: as the pool deepens the PRIOR's own beat-rate approaches its
#  ceiling, which compresses the headroom any multiplier can add.  So "bigger is better" is false
#  here, and that is the part worth remembering if this ever needs moving again.
#
#  CORRECTION (2026-09-04).  "Clearing on every draw" above was a THREE-SEED SAMPLE reported as
#  a property, and it is not one.  Re-measured on the SAME E-2 weight vector and the SAME 23-key
#  column set that comment was written against, seed base 300 gives 0.72x WITH THE PRIOR NOT
#  DISPLACED AT ALL -- so the configuration this file called deterministic was already producing
#  clean failures on the day it was recorded.  Stated as the evidence and not as a rate, because
#  over-reading a handful of draws is the mistake being corrected: 1 of the 4 bases measured
#  fails, and 4 draws does not pin a frequency.  n is not the knob that fixes it, and neither is
#  `strength`; see PLANTED_ANCHORS.
PLANTED_POOL_N = 250

#  ANCHORS PER PLANTED FIXTURE.  Raised 4 -> 8 (2026-09-04).  This is the fix for the knife
#  edge, so the mechanism matters more than the number.
#
#  AT 4 ANCHORS THE GATE ITSELF IS THE NOISY QUANTITY, in BOTH halves.  `floor` is the 99th
#  percentile of the shuffled selection-max gain -- the multiplicity guard, and the half this
#  test exists to exercise.  `se` is sd(per-anchor beat-rate)/sqrt(n_anchors), i.e. a THREE-
#  DEGREE-OF-FREEDOM sd estimate at 4 anchors.  Measured across seed bases at k=4: `se` ran
#  0.0144-0.1297 and `floor` ran 0.049-0.096, and WHICH ONE BINDS swaps from draw to draw
#  (`floor` bound on the seed this test actually uses, `se` bound on others).  So the denominator
#  of the margin moved by a factor of ~2.5 for reasons that have nothing to do with the guard
#  being tested -- which is why 1.19x was never a clean reading of anything.  (It was not pure
#  noise either: the 2026-08 budget changes really did cut the reachable improvement ~11%.  The
#  defect is that the single number could not tell the two apart, and now it can -- see below.)
#
#  WHY ANCHOR COUNT IS THE RIGHT LEVER AND `strength` / `n` ARE NOT.  The planted effect is a
#  PER-ANCHOR quantity, so its size does not depend on how many anchors carry it; both halves of
#  the gate DO shrink with anchor count (`se` as 1/sqrt(k), and `floor` because the selection-max
#  gain is averaged over k anchors before the percentile is taken).  So k raises the ratio
#  without touching the plant, the gate or the bar -- which is exactly what `strength` (saturated)
#  and `n` (non-monotonic) cannot do.  Measured at k=8: `se` 0.0243-0.0582 and `floor`
#  0.0485-0.0649, with `floor` the binding half on 11 of 12 seed bases -- i.e. the guard this
#  test is about is the one that decides the answer, nearly always.
#
#  IT IS A FIXTURE PARAMETER, NOT A CLAIM ABOUT THE DEPLOYMENT.  PREREG still ships 4 train
#  anchors and that is untouched; nothing in `_select` / `selection_noise_floor` special-cases
#  the count, and the n-dependence of the SE itself has its own test above
#  (test_one_se_is_sd_over_sqrt_n_not_the_sd_of_jackknife_means).  The overfit-diagnostic test
#  below deliberately stays at 4 -- it needs differing folds, not margin.
PLANTED_ANCHORS = 8

#  THE MARGIN THE TEST DEMANDS, so it cannot silently drift back to the knife edge.  Nothing
#  previously pinned HOW FAR above the gate the improvement sat, which is precisely how E-2's
#  budget change was able to walk it from a clean pass to a clean fail with no intermediate
#  warning.
#
#  1.3x IS UNCHANGED FROM WHEN IT WAS FIRST SET, AND THAT IS THE POINT.  The 2026-09-04 fix
#  moved the FIXTURE up to the bar, never the bar down to the fixture: a bar re-fitted to
#  whatever its own fixture happens to measure is the pooled median with a longer time constant,
#  which this repo has already rejected once (see meanBars).  What 1.3x was missing was not a
#  different number but a statement of WHAT LIES ON EITHER SIDE OF IT, measured through the
#  IDENTICAL procedure at PLANTED_ANCHORS / PLANTED_POOL_N:
#
#    A REAL SIGNAL   12 seed bases (0, 100, ... 1100): 1.87 1.87 1.99 2.04 2.11 2.14 2.17 2.17
#                    2.56 2.70 2.72 2.93  (min 1.87, median 2.16).  Prior displaced 12/12,
#                    direction correct 12/12.
#    A NULL          6 seed bases with the outcome drawn INDEPENDENTLY of every column:
#                    0.14 0.55 0.66 0.67 0.69 0.96  (max 0.96).  Prior retained 6/6, so a null
#                    never even reaches this assertion -- it fails the `imp > gate` one above.
#
#  1.3x therefore sits in the EMPTY BAND between a null's 0.96x ceiling and a real signal's
#  1.87x floor, with ~30% headroom on each side.  That is what makes a future erosion fail while
#  the gate is still firing, and it is what makes the failure diagnosable rather than mysterious:
#  the message below prints `se` and `floor` separately, and a ratio near 1.0 with `se > floor`
#  means the anchor scatter blew up, while a ratio near 1.0 with `floor` binding and the prior
#  retained means the plant stopped being REACHABLE (a shrinking fitted-cluster budget does
#  this -- it is what the 2026-08-06 / 08-10 / thesis-transfer budget changes did, moving
#  cheapness 0.2600 -> 0.2393 and profitability 0.1517 -> 0.1431).
PLANTED_MARGIN_X = 1.3


def _planted_anchor(tag, n, seed, strength=2.0, noise=0.3):
    """An anchor with a REAL, exploitable relationship between score and outcome.

    The outcome loads POSITIVELY on one cluster and NEGATIVELY on another, so shifting weight
    between those two -- which is exactly what the multipliers can do -- genuinely improves the
    top-20 beat-rate.  A signal planted on ONE cluster is not enough: no single cluster holds
    enough of a unit-total weight vector for a 1.33x cap to move the objective past the gate
    (measured: 0/4 displacements at every strength up to 1.6).  Opposing clusters make the
    effect reachable, and `_opposing_clusters` picks WHICH two from the live prior.

    TWO THINGS THE FIXTURE STOPPED DOING TO ITSELF (2026-09-04).  Both are about making the
    plant REACHABLE by the multipliers; neither touches the gate, the bar, or `strength`.

    1. THE UNFITTED COLUMNS ARE NO LONGER INDEPENDENT NOISE.  It used to fill all 25 canonical
       metric columns with iid N(0,1).  Only 9 of those sit in a fitted cluster; the other 16
       (10 of them carrying non-zero weight) contribute `Sigma w^2` = 0.0364 against the fitted
       clusters' 0.0307 -- i.e. 54% of the score's variance was noise NO multiplier can touch,
       sitting between the plant and the objective.  That is the fixture handicapping its own
       signal for no reason: the deployed pool's metric columns are strongly cross-correlated
       (they are valuation and quality measures of the same names), so sixteen independent noise
       dimensions is not the realistic choice, it is simply the one that suppresses the
       reachable improvement most.
       Measured effect of holding them at their normalised centre instead: improvement rises
       from 0.072-0.104 to 0.116-0.154 at k=8, a ~40% recovery.
    2. THE PLANT LOADS PROPORTIONALLY TO THE PRIOR WEIGHT WITHIN EACH CLUSTER, not equally.  The
       reachable direction is `Delta w` (proportional to `w_m` inside a cluster, since a cluster
       multiplier scales every member by the same factor), so by Cauchy-Schwarz the contrast that
       the multipliers can exploit best is the one aligned with `w`.  Equal loadings gave
       `Delta cov / sd(sig)` = 0.0196 against 0.0222 for weight-proportional -- a free ~14%.
       (Note for anyone tempted by the opposite move: CONCENTRATING the plant on each cluster's
       largest member is WORSE, 0.0175, because it raises `sd(sig)` faster than `Delta cov`.
       That corrects the `_opposing_clusters` note above, which worried about the wrong thing.)
    """
    rng = np.random.default_rng(seed)
    prior = refit.prior_weights()
    up_name, down_name = _opposing_clusters()
    up = refit.CLUSTERS[up_name]
    down = refit.CLUSTERS[down_name]
    fitted = {m for members in refit.CLUSTERS.values() for m in members}
    #  EVERY canonical metric key, taken from the single source of truth rather than listed --
    #  a key added to the vector and forgotten here is the drift class the weights refactor
    #  exists to remove (E-2 added `shareCountChange` / `longTermDebtChange`).
    cols = list(sw.METRIC_KEYS)
    src = ["%s_%03d" % (tag, i) for i in range(n)]
    d = {"source": src}
    for c in cols:
        #  DRAWN FOR EVERY KEY -- so the column set still has to be complete, and a key added to
        #  the vector still moves this fixture -- but HELD AT ZERO outside the fitted clusters.
        #  See point 1 in the docstring.
        d[c] = rng.normal(size=n) * (1.0 if c in fitted else 0.0)
    df = pd.DataFrame(d)
    #  Loadings proportional to |w| inside each cluster (docstring point 2), each side divided by
    #  its own cluster budget so the two sides stay a unit-scale contrast rather than the larger
    #  cluster simply dominating.
    Wu = sum(abs(prior[m]) for m in up)
    Wd = sum(abs(prior[m]) for m in down)
    sig = (sum(df[m].to_numpy() * abs(prior[m]) / Wu for m in up)
           - sum(df[m].to_numpy() * abs(prior[m]) / Wd for m in down))
    ex = strength * sig + rng.normal(scale=noise, size=n)
    return dict(buy=tag, eval="x", normed=df, group_of={s: s for s in src},
                excess={s: float(v) for s, v in zip(src, ex)}, bench=0.0,
                n_pool=n, n_priced=n)


def test_gate_FIRES_on_a_planted_signal_and_finds_the_RIGHT_DIRECTION():
    """THE test that proves a FAIL means "no signal" and not "the gate can never fire".

    The previous version of this test wrapped its only assertion in an `if` and passed
    without asserting anything -- the same defect class as a bare `return` in a test.  This
    one plants a signal the multipliers can exploit and requires FOUR things:
      1. the prior IS displaced;
      2. the improvement genuinely clears the gate (so it fired for the right reason);
      3. it clears it by a STATED MARGIN, not by a hair -- see PLANTED_MARGIN_X;
      4. the direction is CORRECT, matching the plant's own up/down clusters.
    (4) is what distinguishes "the optimiser found the signal" from "the optimiser moved".
    (3) is new (2026-08-04): without it the test could sit at 1.0x and nobody would know, which
    is exactly how E-2's re-weighting walked it from a clean 1.23-1.29x pass to a clean
    0.51-0.68x fail with no intermediate warning.
    Crossover sits at a sustained ~+2.5 to +4pp per-anchor advantage, i.e. sensibly BELOW the
    10.95pp holdout adoption bar -- the gate is meant to pass real-but-small signal through to
    the holdout test, not to pre-empt it.

    THE 2026-09-04 FIX, in one line: (3) was asserting a DETERMINISTIC bound on a quantity whose
    4-anchor draw-to-draw spread runs 0.56x-2.2x, calibrated from a THREE-draw sample.  The bar
    is unchanged; the fixture is now measured over 12 seed bases and clears it 12/12.  See
    PLANTED_ANCHORS for the mechanism and PLANTED_MARGIN_X for the null-vs-signal bands.
    """
    prior = refit.prior_weights()
    anchors = [_planted_anchor("P%d" % i, PLANTED_POOL_N, i)
               for i in range(PLANTED_ANCHORS)]
    floor, _s = refit.selection_noise_floor(anchors, prior, lam=0.02, n_perm=25, seed=11)
    mult, g, se, _n = refit._select(anchors, prior, lam=0.02, verbose=False,
                                    noise_floor=floor)
    best = float(g["objective"].max())
    prior_obj = float(g[np.all([np.isclose(g[c], 1.0) for c in refit.CLUSTERS], axis=0)]
                      ["objective"].iloc[0])
    gate = max(se, floor)
    imp = best - prior_obj
    #  WHICH HALF OF THE GATE BINDS is the diagnosis, not a detail, so both messages report it:
    #  `se > floor` means the per-anchor scatter blew up (a fixture-noise failure, and
    #  PLANTED_ANCHORS is the lever); `floor >= se` means the planted signal stopped being
    #  REACHABLE by the multipliers, which is a real erosion worth reporting.
    _binds = "se" if se > floor else "floor"
    assert imp > gate, \
        ("planted signal did not clear the gate (improvement %.4f vs gate %.4f; se %.4f, "
         "floor %.4f, binding=%s, anchors=%d)"
         % (imp, gate, se, floor, _binds, PLANTED_ANCHORS))
    assert imp > PLANTED_MARGIN_X * gate, (
        "the gate fired, but only by %.2fx against a required %.2fx margin (improvement %.4f, "
        "gate %.4f; se %.4f, floor %.4f, binding=%s). READ PLANTED_MARGIN_X FIRST: it records "
        "the measured null band (max 0.96x) and real-signal band (min 1.87x) that 1.3x sits "
        "between, so the bar is not the thing to move. If binding=se the anchor scatter is the "
        "problem and PLANTED_ANCHORS is the lever; if binding=floor the plant is no longer "
        "REACHABLE, which is a fact about the deployed cluster budgets and belongs in a report, "
        "not in a re-tuned fixture. `strength` is SATURATED and will not fix either, and a "
        "deeper pool is not monotonically better (n=400 is worse than n=250)."
        % (imp / gate, PLANTED_MARGIN_X, imp, gate, se, floor, _binds))
    assert not all(v == 1.0 for v in mult.values()), \
        "improvement cleared the gate but the prior was still returned -- the gate is a " \
        "rubber stamp"
    #  DIRECTION, against the pair the plant actually used -- see `_opposing_clusters`.
    up_name, down_name = _opposing_clusters()
    assert mult[up_name] >= 1.0 and mult[down_name] <= 1.0, \
        "the fit moved but in the WRONG direction: %s (plant was %s UP, %s DOWN)" % (
            mult, up_name, down_name)
    assert mult[up_name] > mult[down_name], mult


def test_planted_signal_also_makes_the_overfit_diagnostic_able_to_fire():
    """Fix 3's premise: per-fold |IN - OUT| must be able to be non-zero.  On planted signal
    the folds select different vectors, so the diagnostic has something to measure -- which is
    exactly why the MEAN version (an identity at 0 when all folds retain the prior) was the
    wrong quantity to gate on."""
    prior = refit.prior_weights()
    #  STAYS AT 4 ANCHORS, deliberately: this test needs the folds to DIFFER, not a margin to
    #  clear a bar, and `lowo` re-runs the whole selection once per fold -- so PLANTED_ANCHORS
    #  here would double its runtime to buy a property it does not assert.
    anchors = [_planted_anchor("Q%d" % i, PLANTED_POOL_N, 50 + i) for i in range(4)]
    folds = refit.lowo(anchors, prior, lam=0.02, verbose=False, n_perm=8)
    gaps = [abs(f["IN_sample_mean_on_other_folds"] - f["OUT_of_sample_fit"]) for f in folds]
    assert max(gaps) > 1e-9, "per-fold IN vs OUT are identical -- nothing to diagnose"


_SLOW = os.environ.get("VA_RUN_SLOW_REFIT_TESTS", "") == "1"
_slow = pytest.mark.skipif(
    not _SLOW, reason="Monte-Carlo at the SHIPPED n_perm=200 takes ~20 min; run with "
                      "VA_RUN_SLOW_REFIT_TESTS=1. The measured numbers are recorded in the "
                      "runbook so a skip does not lose them.")


@_slow
def test_false_adoption_rate_on_pure_noise_at_the_SHIPPED_n_perm():
    """Monte Carlo on the GATE'S OWN design property, at the SHIPPED n_perm -- not a cheaper
    one.

    The earlier version ran at n_perm=8 with a 25% bound, so the "~8% observed" it produced
    was EVIDENCE FOR the under-delivered-percentile defect rather than a check against it: a
    low n_perm biases the floor downward, which raises the false-adoption rate.  Testing a
    percentile guarantee at a different n_perm than ships is testing a different guard.
    """
    prior = refit.prior_weights()
    trials, n_prior = 10, 0
    n_perm = int(refit.PREREG["selection_noise_permutations"])
    for seed in range(trials):
        anchors = [_FakeAnchor("N%d" % i, n=60, seed=1000 * seed + i) for i in range(4)]
        mult, _g, _se, _fl = refit.fit(anchors, prior, lam=0.02, verbose=False, n_perm=n_perm)
        if all(abs(v - 1.0) < 1e-12 for v in mult.values()):
            n_prior += 1
    rate = 1.0 - n_prior / trials
    assert rate <= 0.20, ("false-adoption rate on pure noise is %.0f%% (%d/%d retained) at "
                          "n_perm=%d -- the gate has regressed (it was 75%% with the 3x SE "
                          "bug)" % (100 * rate, n_prior, trials, n_perm))


@_slow
def test_the_floor_DELIVERS_the_registered_percentile():
    """Fix 1, as a standing check.  A percentile estimated from n_perm draws is only the
    registered percentile if n_perm is large enough; at 20 it delivered the ~95th.  Compare
    the shipped-n_perm floor against a high-permutation reference on the same anchors and
    require the delivered percentile to be within a few points of the registered one."""
    prior = refit.prior_weights()
    anchors = [_FakeAnchor("R%d" % i, n=80, seed=300 + i) for i in range(4)]
    q = float(refit.PREREG["selection_noise_gate_percentile"])
    n_perm = int(refit.PREREG["selection_noise_permutations"])
    _f, ref = refit.selection_noise_floor(anchors, prior, 0.02, n_perm=600, seed=99, q=q)
    ref = np.asarray(ref)
    delivered = []
    for r in range(5):
        f, _s = refit.selection_noise_floor(anchors, prior, 0.02, n_perm=n_perm,
                                            seed=4000 + r, q=q)
        delivered.append(100.0 * float((ref < f).mean()))
    med = float(np.median(delivered))
    assert med >= q - 3.0, ("the floor DELIVERS the %.0fth percentile but is registered at "
                            "the %.0fth (delivered %s) -- raise n_perm or re-register"
                            % (med, q, [round(x, 1) for x in delivered]))


def test_main_refuses_when_holdout_is_in_the_training_anchors():
    """The CLI-level guard, exercised without touching a panel."""
    src = open(os.path.join(_HERE, "refit.py"), encoding="utf-8").read()
    assert "if hold in train:" in src and "REFUSING" in src


# --------------------------------------------------------------------------- #
#  4. The threshold comes from the pre-registration                           #
# --------------------------------------------------------------------------- #
def test_threshold_is_read_from_prereg_and_not_derived_from_a_result():
    src = open(os.path.join(_HERE, "refit.py"), encoding="utf-8").read()
    # the verdict line must compare against PREREG, not against anything computed
    assert 'verdict = "PASS" if gain_pp > PREREG["noise_floor_pp"] else "FAIL"' in src
    assert refit.PREREG["noise_floor_pp"] == pytest.approx(10.95, abs=0.01)


def test_prereg_noise_floor_matches_its_stated_derivation():
    """SE of a top-20 beat-rate at p = 0.40 -- the number the bar claims to be."""
    se_pp = 100.0 * np.sqrt(0.40 * 0.60 / 20)
    assert refit.PREREG["noise_floor_pp"] == pytest.approx(se_pp, abs=0.05)


def test_prereg_records_the_expected_outcome_before_the_run():
    assert "FAIL" in refit.PREREG["expected_outcome"].upper()
    assert refit.PREREG["holdout_cell"]["buy"] == "2022-12-30"
    assert tuple(refit.PREREG["holdout_cell"].values()) not in \
        {tuple(t) for t in refit.PREREG["train_anchors"]}


def test_no_l1_knob_exists():
    """L1 was dropped by design; a stray L1 term would re-introduce a zeroing decision.

    Checked on the AST -- identifiers, arguments and attributes -- not on the source text,
    which legitimately DISCUSSES L1 in the docstring explaining why it was dropped.  (The
    first version of this test grepped the text and failed on its own documentation.)
    """
    import ast
    tree = ast.parse(open(os.path.join(_HERE, "refit.py"), encoding="utf-8").read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id.lower())
        elif isinstance(node, ast.arg):
            names.add(node.arg.lower())
        elif isinstance(node, ast.Attribute):
            names.add(node.attr.lower())
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg.lower())
    offenders = [n for n in names if "l1" in n or "lasso" in n]
    assert not offenders, "an L1 knob exists in code: %s" % offenders
    # and the L2 penalty is present and is genuinely squared (not absolute)
    assert "l2_penalty" in names or hasattr(refit, "l2_penalty")
    assert refit.l2_penalty({c: 1.5 for c in refit.CLUSTERS}, 1.0) == pytest.approx(
        sum((1.5 - 1.0) ** 2 for _ in refit.CLUSTERS))


# --------------------------------------------------------------------------- #
#  score_anchor mechanics                                                     #
# --------------------------------------------------------------------------- #
def test_score_anchor_applies_dedup_groups():
    a = _FakeAnchor("T", n=10, seed=3)
    # force every name into ONE issuer group -> at most one survivor in the top-N
    a["group_of"] = {s: "SAME" for s in a["normed"]["source"]}
    out = refit.score_anchor(a, refit.prior_weights(), topn=20)
    assert out["n_eval"] == 1


def test_score_anchor_is_invariant_to_a_positive_rescale_of_all_weights():
    """AggScore is a weighted sum, so scaling every weight by k cannot change the ORDER --
    a property the fit relies on when it renormalises the fitted budget."""
    a = _FakeAnchor("T", n=60, seed=7)
    prior = refit.prior_weights()
    x2 = {k: 2 * v for k, v in prior.items()}
    assert refit.score_anchor(a, prior)["top"] == refit.score_anchor(a, x2)["top"]
