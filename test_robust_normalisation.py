"""THE ENFORCEMENT for the E-1 robust normalisation (2026-08-03).

`normalizeAndDropNA`'s z-path is now: median centre -> Huber Proposal-2 scale on the SAME
observed subset -> algebraic concave squash, with a MAD == 0 degeneracy guard replacing two
hardcoded exemption lists.  See `postBoRank.HUBER_C` / `SQUASH_K` for the design and
projects/investment-filter/design/normalisation-spec.md for the measurements.

THE CHANGE MOVES SCORES BY DESIGN, so bit-identity is not the test.  What is pinned here is the
set of PROPERTIES the design is chosen for, each of which is a defect if it fails:

  1. ZERO TIES -- no two distinct inputs may map to one output.  The +-3 clip this replaced
     collapsed 37 distinct values on the real panel; a strictly monotone squash collapses none.
     This is also the test that catches a swap to `k*tanh(z/k)`, which equals k EXACTLY in
     float64 for |z| >= 18.99k.
  2. NO STEP FUNCTION -- perturbing one name's raw value by an economically trivial amount must
     not materially re-score the others.  The rejected discrete (gap-detection) alternative
     induces up to 3.69 z-units on the other names; this rule induces < 0.01.
  3. THE k PROPERTY -- no single column may carry a median-scoring name into the top 20.
  4. THE ESTIMATOR IS WHAT IT CLAIMS TO BE -- a Huber P2 M-estimate about the median, unbiased
     on a clean normal column, with the degeneracy guard firing exactly on MAD == 0.
  5. MISSINGNESS IS NEUTRAL -- 0 IS the observed median of every column (issue I-3).

No network, no pickle reads, no API key.  Run it the repo way: `pytest . --ignore=baseline_tools`
(never with an explicit path -- that bypasses conftest.py's collect_ignore guard).
"""
import math

import numpy as np
import pandas as pd
import pytest

import postBoRank as pbr
import scoringWeights as sw


# --------------------------------------------------------------------------- #
#  Fixtures -- a pool with the shapes that break things                       #
# --------------------------------------------------------------------------- #
def _pool(n=100, seed=20260803):
    """A pool whose columns span the real panel's shapes: heavy right tail, a bunched
    outlier PAIR (the case the CEO's gap rule was designed for), a bounded discrete code,
    a negative-weight column, a zero-weight diagnostic, and NaNs."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'source': ['S%03d' % i for i in range(n)]})
    df['earnYield'] = rng.normal(0.05, 0.02, n)
    df['incomeQuality'] = rng.lognormal(0.0, 1.0, n)
    df.loc[0, 'incomeQuality'] = 4.0e2                     # one huge honest outlier
    df['currentRatio'] = rng.lognormal(0.7, 0.5, n)
    df.loc[1, 'currentRatio'] = 60.0                       # a BUNCHED PAIR: neither may
    df.loc[2, 'currentRatio'] = 61.0                       # mask the other
    df['Piotroski'] = rng.integers(2, 9, n).astype(float)   # bounded discrete
    df['CycleHeat'] = rng.normal(0.0, 1.0, n)               # w < 0
    df['DcfToPrice'] = rng.normal(1.0, 0.2, n)              # w = 0
    df.loc[5:9, 'earnYield'] = np.nan                       # 5 imputed cells
    return df


def _W():
    return pd.Series({'earnYield': 0.30, 'incomeQuality': 0.25, 'currentRatio': 0.15,
                      'Piotroski': 0.20, 'CycleHeat': -0.10, 'DcfToPrice': 0.0})


def _norm(df, W=None, **kw):
    return pbr.normalizeAndDropNA(df.copy(), weight_series=(W if W is not None else _W()),
                                  **kw)[0]


# --------------------------------------------------------------------------- #
#  1. ZERO TIES                                                               #
# --------------------------------------------------------------------------- #
def test_no_distinct_input_is_collapsed_onto_another(capsys):
    """THE headline property.  Counted the way the defect was counted -- distinct raw inputs
    vs distinct outputs per column -- not by appeal to monotonicity."""
    df = _pool()
    out = _norm(df)
    capsys.readouterr()
    for col in ('earnYield', 'incomeQuality', 'currentRatio', 'Piotroski', 'CycleHeat'):
        a = pd.to_numeric(df[col], errors='coerce')
        b = pd.to_numeric(out[col], errors='coerce')
        both = a.notna() & b.notna()
        assert int(a[both].nunique()) == int(b[both].nunique()), \
            '%s collapsed %d distinction(s)' % (col, a[both].nunique() - b[both].nunique())


def test_the_run_asserts_the_ties_property_itself(capsys):
    """The audit is not just available -- normalizeAndDropNA runs it on every pool."""
    df = _pool()
    out = _norm(df)
    capsys.readouterr()
    num = df.drop(columns=['source']).apply(pd.to_numeric, errors='coerce')
    z = out.drop(columns=['source']).apply(pd.to_numeric, errors='coerce')
    collapsed = pbr.assert_no_collapsed_distinctions(num, z, 'unit')
    assert set(collapsed.values()) == {0}, collapsed


def test_the_ties_audit_CATCHES_a_saturating_squash(capsys):
    """The audit has to be able to fail, and this is the failure it exists for: `k*tanh(z/k)`
    saturates in float64 at |z| >= 18.99k and would tie every name past that bound.  Simulated
    directly, because swapping the production squash to prove it would be the defect."""
    n = 40
    raw = pd.DataFrame({'m': np.linspace(1.0, 100.0, n)})
    k = 3.0
    z = (raw['m'] - raw['m'].median()) / 0.05             # |z| up to ~1000
    tanh_out = pd.DataFrame({'m': k * np.tanh(z / k)})
    #  tanh really does tie them -- state it, don't assume it
    assert int(tanh_out['m'].nunique()) < int(raw['m'].nunique())
    with pytest.raises(AssertionError, match='COLLAPSED DISTINCTIONS'):
        pbr.assert_no_collapsed_distinctions(raw, tanh_out, 'tanh-probe', k)
    #  and the algebraic form does not
    alg = pd.DataFrame({'m': pbr.squash(z, k).to_numpy()})
    assert pbr.assert_no_collapsed_distinctions(raw, alg, 'algebraic-probe', k) == {'m': 0}


def test_tanh_saturates_in_float64_and_the_algebraic_form_does_not():
    """THE MEASURED CONSTANT, re-derived here rather than inherited: `k*tanh(z/k)` equals k
    EXACTLY in float64 for |z| >= 19.0615*k, i.e. 57.185 at k = 3.  (The spec states 18.99*k /
    56.971; that is 0.4% low -- tanh(18.99) is 0.9999999999999999, one ULP short.  The DEFECT is
    unaffected: incomeQuality's honest |z| = 186 on the real panel is past either threshold, so
    tanh would already have tied a cell there.)"""
    k = 3.0
    Z = 19.0615546                  # the float64 threshold, bisected
    assert k * math.tanh((Z * k) / k) == k, 'tanh no longer saturates -- re-check the choice'
    assert k * math.tanh((19.0 * k) / k) < k, 'the threshold moved -- re-derive it'
    #  the algebraic form has eight orders of margin
    assert abs(pbr.squash(pd.Series([1.0e6]), k).iloc[0]) < k
    assert abs(pbr.squash(pd.Series([1.0e6]), k).iloc[0]) > k * (1 - 1e-5)
    assert abs(pbr.squash(pd.Series([1.0e7]), k).iloc[0]) < k


# --------------------------------------------------------------------------- #
#  2. NO STEP FUNCTION                                                        #
# --------------------------------------------------------------------------- #
def test_moving_one_extreme_value_barely_moves_the_others(capsys):
    """The measure that killed the discrete alternative.  Sweep the most extreme value on a
    heavy-tailed column inward and record the worst |dz| induced on the OTHER names.  A
    DISCRETE inclusion cut puts a STEP in sigma-hat, and sigma-hat divides everyone -- measured
    3.69 z-units for the gap rule against < 0.01 here."""
    df = _pool()
    col = 'incomeQuality'
    s = pd.to_numeric(df[col], errors='coerce')
    i = int(s.abs().idxmax())
    start, target = float(s.loc[i]), float(s.quantile(0.90))
    worst, prev = 0.0, None
    for t in range(41):
        s2 = s.copy()
        s2.loc[i] = start + (target - start) * (t / 40.0)
        e = pbr.robust_location_scale(s2)
        z = pbr.squash((s2 - e.mu) / e.sigma).drop(index=i).to_numpy()
        if prev is not None:
            worst = max(worst, float(np.nanmax(np.abs(z - prev))))
        prev = z
    capsys.readouterr()
    assert worst < 0.05, 'a step appeared in the ruler: worst induced |dz| = %.5f' % worst


def test_the_estimation_weight_is_CONTINUOUS_not_an_in_out_cut(capsys):
    """WHY there is no step: nobody is excluded.  min(1, c/|z|) is a smooth weight, so the
    retained fraction sits a little below 1 rather than dropping by 1/n as a name crosses a
    threshold."""
    df = _pool()
    for col in ('earnYield', 'incomeQuality', 'currentRatio'):
        e = pbr.robust_location_scale(pd.to_numeric(df[col], errors='coerce'))
        assert 0.5 < e.weight_retained < 1.0, (col, e.weight_retained)
    capsys.readouterr()


# --------------------------------------------------------------------------- #
#  3. THE k PROPERTY                                                          #
# --------------------------------------------------------------------------- #
def _deployed_shaped_pool(n=100, seed=20260803):
    """A pool with the DEPLOYED vector's SHAPE -- 21 columns, max |w| = 0.100, sum|w| = 1.

    TWO WAYS THIS FIXTURE HAD TO BE MADE REALISTIC, both learned by getting it wrong:
      * WEIGHT DISPERSION.  The property is a joint property of k AND the weight vector, so a
        fixture with 5 columns at |w| ~ 0.25 fails it by construction and tests the fixture.
        (Not hypothetical: on the real carve-out cohorts, whose vectors DO concentrate -- 0.597
        on one column -- the property genuinely does not hold at ANY k.  That is a weighting
        finding for E-2, not a k defect.  See single_column_reach_check.)
      * CROSS-COLUMN CORRELATION.  With 18 INDEPENDENT noise columns the AggScore is a sum of
        18 independent terms and its spread is far narrower than a real pool's, which shrinks
        the median->rank-20 distance and makes the property spuriously hard.  A real pool is
        correlated -- a good company is good on many metrics at once -- so a shared latent
        quality factor is part of being a faithful fixture, not a convenience."""
    rng = np.random.default_rng(seed)
    W = pd.Series(sw.DEPLOYED)
    quality = rng.standard_normal(n)                          # the shared latent factor
    df = pd.DataFrame({'source': ['S%03d' % i for i in range(n)]})
    for j, col in enumerate(W.index):
        common = 0.6 * quality * np.sign(float(W.get(col, 1)) or 1.0)
        idio = rng.standard_normal(n) * 0.8
        if j % 4 == 0:
            df[col] = np.exp(common + idio)                   # heavy right tail
        elif j % 4 == 1:
            df[col] = 0.05 + 0.02 * (common + idio)
        elif j % 4 == 2:
            df[col] = np.clip(np.round(4 + 1.5 * (common + idio)), 0, 9)   # discrete
        else:
            df[col] = common + idio
    df.loc[0, W.index[0]] = 5.0e3                             # one huge honest outlier
    return df, W


def _score(df, W, pool='general'):
    pbr.NORM_DIAGNOSTICS.clear()
    out = pbr.normalizeAndDropNA(df.copy(), weight_series=W, pool_label=pool)[0]
    wt = out.drop(columns=['source'])
    for c in wt.columns:
        wt[c] = out[c].to_numpy() * float(W.get(c, 1) or 0)
    return out, pbr.getAggScore(pd.concat([out[['source']], wt], axis=1))


def test_no_single_column_can_carry_a_median_name_into_the_top_20(capsys):
    """max_c |w_c| * max_i |zeta_ic|  <  AggScore(rank 20) - AggScore(median).

    PANEL-DEPENDENT on the right-hand side, which is exactly why the run checks it per pool
    instead of trusting a constant -- a number with no panel attached cannot be checked, and
    that is how the stale 0.134 survived.  On the real panel
    (resdic_2026-07-17_CORRECTED, general top-100) this reads reach 0.2160 (incomeQuality) vs
    distance 0.2396 = 0.90x, against 0.2666 vs 0.2560 = 1.04x for the ruler it replaced -- i.e.
    the pre-E-1 design VIOLATED it."""
    df, W = _deployed_shaped_pool()
    _out, ranked = _score(df, W)
    res = pbr.single_column_reach_check(ranked, W, 'general')
    capsys.readouterr()
    assert res is not None and res['applicable']
    assert res['ratio'] < 1.0, res


def test_k_IS_the_dial_that_controls_the_reach_ratio(capsys):
    """WHAT IS UNIVERSAL vs WHAT IS PANEL-DEPENDENT, kept apart on purpose.

    Universal, and pinned here: the reach ratio is strictly INCREASING in k, so k is the dial,
    and raising it monotonically erodes the margin.  NOT universal, and therefore NOT asserted
    here: that any particular k violates.  On the real panel k = 3 gives 0.90x and k = 4 gives
    ~1.1x, but on another panel -- or this synthetic one -- the crossing point sits elsewhere.
    Writing "k = 4 violates" into a unit test would re-create precisely the defect this design
    is meant to retire: a threshold with no panel attached.  The per-run
    single_column_reach_check is what tests the crossing, against the panel it is on."""
    df, W = _deployed_shaped_pool()
    ratios = {}
    for k in (1.5, 3.0, 6.0, 12.0):
        pbr.NORM_DIAGNOSTICS.clear()
        out = pbr.normalizeAndDropNA(df.copy(), weight_series=W, squash_k=k,
                                     pool_label='general')[0]
        wt = out.drop(columns=['source'])
        for c in wt.columns:
            wt[c] = out[c].to_numpy() * float(W.get(c, 1) or 0)
        ranked = pbr.getAggScore(pd.concat([out[['source']], wt], axis=1))
        ratios[k] = pbr.single_column_reach_check(ranked, W, 'general')['ratio']
    capsys.readouterr()
    ks = sorted(ratios)
    assert all(ratios[a] < ratios[b] for a, b in zip(ks, ks[1:])), ratios
    assert ratios[pbr.SQUASH_K] < 1.0, ratios


def test_the_reach_check_reports_NOT_APPLICABLE_rather_than_a_negative_distance(capsys):
    """A 25-name cohort has no rank-20 boundary above its own median, so the inequality is
    vacuous there.  The first version of this check printed VIOLATES on all five cohorts for
    that reason; a vacuous test must say so, not fail."""
    df = _pool(n=24)
    W = _W()
    pbr.NORM_DIAGNOSTICS.clear()
    out = _norm(df, W, pool_label='general')
    wt = out.drop(columns=['source'])
    for c in wt.columns:
        wt[c] = out[c].to_numpy() * float(W.get(c, 1) or 0)
    ranked = pbr.getAggScore(pd.concat([out[['source']], wt], axis=1))
    res = pbr.single_column_reach_check(ranked, W, 'general', top_n=20)
    printed = capsys.readouterr().out
    assert res is not None and res['applicable'] is False, res
    assert 'NOT APPLICABLE' in printed
    assert 'VIOLATES' not in printed


def test_a_cohort_gets_the_cohort_shortlist_depth_not_the_general_one():
    """The depth is a JUDGEMENT and is therefore explicit and named, not implicit in a default."""
    assert pbr.REACH_TOP_N_GENERAL == 20
    assert pbr.REACH_TOP_N_COHORT == 5


# --------------------------------------------------------------------------- #
#  4. THE ESTIMATOR IS WHAT IT CLAIMS TO BE                                   #
# --------------------------------------------------------------------------- #
def test_beta_is_the_normal_consistency_constant():
    """beta(c) = E[clip(Z,-c,c)^2] for Z ~ N(0,1).  Checked against a direct Monte-Carlo, not
    against itself: at c = 1.5 the factor is worth 12% of sigma-hat, so a wrong closed form
    would be a 12% bias on every score."""
    for c in (1.0, 1.345, 1.5, 2.0, 3.0):
        rng = np.random.default_rng(7)
        z = rng.standard_normal(4_000_000)
        mc = float(np.mean(np.clip(z, -c, c) ** 2))
        assert pbr.huber_beta(c) == pytest.approx(mc, rel=2e-3), c
    assert pbr.huber_beta(1.5) == pytest.approx(0.778465, abs=1e-6)
    assert pbr.huber_beta(3.0) == pytest.approx(0.995007, abs=1e-6)


def test_the_scale_is_UNBIASED_on_a_clean_normal_column():
    """What beta(c) buys.  Without it the estimator would read ~0.88 sigma on clean data at
    c = 1.5, i.e. it would inflate every clean column's z by ~13%."""
    est = [pbr.robust_location_scale(pd.Series(
        np.random.default_rng(s).standard_normal(200))).sigma for s in range(60)]
    assert float(np.mean(est)) == pytest.approx(1.0, abs=0.02), float(np.mean(est))


def test_it_is_a_FIXED_POINT_of_the_huber_estimating_equation():
    """The defining property, tested as an equation rather than as an output: at the returned
    sigma, mean(psi(u)^2) == beta(c)."""
    x = pd.Series(np.random.default_rng(3).lognormal(0, 1, 300))
    e = pbr.robust_location_scale(x)
    assert e.status == 'ok'
    u = ((x - e.mu) / e.sigma).clip(-pbr.HUBER_C, pbr.HUBER_C)
    assert float((u ** 2).mean()) == pytest.approx(pbr.huber_beta(pbr.HUBER_C), rel=1e-9)


def test_the_scale_RESISTS_contamination_where_the_old_threshold_did_not():
    """The whole point of moving c from 3 to 1.5.  5% of the column at +20 on a unit-sigma
    population: the pre-E-1 parameterisation returned ~1.35 (35% inflation); this must be far
    closer to the truth of 1.0."""
    out = []
    for s in range(40):
        rng = np.random.default_rng(1000 + s)
        x = pd.Series(np.concatenate([rng.standard_normal(95), np.full(5, 20.0)]))
        out.append(pbr.robust_location_scale(x).sigma)
    assert float(np.mean(out)) < 1.15, float(np.mean(out))


def test_a_BUNCHED_PAIR_cannot_mask_itself():
    """The CEO's own case -- 'if there are two bunched up in the upper end, we take them both
    out'.  A smooth weight handles it without a threshold, because the fixed point is set by the
    bulk: at cohort size n=25 the pre-E-1 path returned ~2.19 against a truth of 1.0."""
    out = []
    for s in range(40):
        rng = np.random.default_rng(2000 + s)
        x = pd.Series(np.concatenate([rng.standard_normal(23), [15.0, 15.3]]))
        out.append(pbr.robust_location_scale(x).sigma)
    assert float(np.mean(out)) < 1.30, float(np.mean(out))


def test_location_is_the_MEDIAN_of_the_observed_values_only():
    """'the same subset that is used to calc the mean' -- and an imputation is not in it."""
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan, np.inf, -np.inf])
    e = pbr.robust_location_scale(x)
    assert e.mu == 3.0
    assert e.n_obs == 5


def test_an_extreme_value_ENTERS_the_ruler_partially_and_is_PLACED_on_it():
    """N1, both halves: the extreme is downweighted (never excluded) in the estimation, and it
    still gets its own value out -- 'outliers stay in the ranking, out of the ruler'."""
    base = list(np.random.default_rng(11).normal(0, 1, 60))
    a = pbr.robust_location_scale(pd.Series(base))
    b = pbr.robust_location_scale(pd.Series(base + [500.0]))
    #  the extreme moved sigma-hat by very little ...
    assert abs(b.sigma - a.sigma) / a.sigma < 0.10
    #  ... it was downweighted, not dropped ...
    assert b.weight_retained < 1.0
    #  ... and it comes out as the most extreme name, not tied to anyone
    out = pbr.squash((pd.Series(base + [500.0]) - b.mu) / b.sigma)
    assert out.abs().idxmax() == len(base)
    assert int(out.nunique()) == len(base) + 1


# --------------------------------------------------------------------------- #
#  The degeneracy guard REPLACES the name list                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('name,values,sigma,max_abs_z', [
    #  sigma and max|z| RE-DERIVED here, not copied.  normalisation-spec.md N4 states
    #  (1e5, 10.0), (1.091, 4.58) and (1.068, 3.75); every SIGMA is right and every max|z| is
    #  WRONG -- they are inconsistent with the spec's own sigmas under either ddof convention
    #  (e.g. 990000/1e5 = 9.90, not 10.0).  The pre-E-1 code comment on WINSOR_EXEMPT_BOUNDED
    #  had the Piotroski figure right at |z| = 4.36, so the spec regressed a number the tree
    #  already held.  The values below are the arithmetic.
    ('99x0 + 1x1e6', np.r_[np.zeros(99), 1e6], 1.0e5, 9.9000),
    ('Piotroski 20x7 + 1x2', np.r_[np.full(20, 7.0), 2.0], 1.09109, 4.3644),
    ('CycleHeat 25x(-3) + 2x(+1)', np.r_[np.full(25, -3.0), np.full(2, 1.0)], 1.06752, 3.4694),
])
def test_the_MAD_zero_guard_returns_a_sane_ruler_on_every_pathological_shape(
        name, values, sigma, max_abs_z):
    """The three shapes that defeated the winsorizer.  An UNGUARDED robust estimator implodes
    to sigma-hat ~ 1e-14 here (MAD is exactly 0, so the fixed point is 0) and every minority
    name ties at the bound; the guard falls back to the classical mean/sd and returns a ruler
    in single digits."""
    #  first: the condition really is degenerate, i.e. MAD is exactly 0
    obs = pd.Series(values)
    assert float((obs - obs.median()).abs().median()) == 0.0, name
    e = pbr.robust_location_scale(obs)
    assert e.status == 'degenerate', (name, e)
    assert e.sigma == pytest.approx(sigma, rel=1e-4), (name, e.sigma)
    z = (obs - e.mu) / e.sigma
    assert float(z.abs().max()) == pytest.approx(max_abs_z, abs=0.001), (name, z.abs().max())
    assert float(pbr.squash(z).abs().max()) < pbr.SQUASH_K


def test_the_guard_fires_on_the_CONDITION_and_not_on_a_column_NAME(capsys):
    """The point of replacing `WINSOR_EXEMPT_BOUNDED`: a name list silently mis-handles the
    next discrete metric anyone adds, and it mis-handles the named ones too -- all three named
    columns take a proper robust scale when their shape is ordinary."""
    assert not hasattr(pbr, 'WINSOR_EXEMPT_BOUNDED'), \
        'the exemption list is back -- the z-path must gate on MAD == 0, not on a name'
    rng = np.random.default_rng(5)
    #  a column NOT in any list, with a degenerate shape -> guarded
    odd = pbr.robust_location_scale(pd.Series(np.r_[np.full(30, 2.0), [9.0, 9.5]]))
    assert odd.status == 'degenerate'
    #  a column that IS in the old list, with an ordinary shape -> NOT guarded
    for col in ('Piotroski', 'CycleHeat', 'marketCapRevQuants'):
        assert col in pbr.BOUNDED_DISCRETE_COLUMNS
    ordinary = pbr.robust_location_scale(pd.Series(rng.integers(0, 10, 100).astype(float)))
    assert ordinary.status == 'ok'
    capsys.readouterr()


def test_every_column_takes_the_same_ruler_including_zero_weight_ones(capsys):
    """No exemption survives.  A w = 0 column is SCORE-NEUTRAL (AggScore is sum(z*w)), so
    normalising it the same way costs nothing and keeps postScoreMetric a single-basis frame."""
    df = _pool()
    W = _W()
    pbr.NORM_DIAGNOSTICS.clear()
    out = _norm(df, W, pool_label='unit')
    capsys.readouterr()
    rows = {d['column']: d for d in pbr.NORM_DIAGNOSTICS if d['pool'] == 'unit'}
    for col in ('earnYield', 'incomeQuality', 'Piotroski', 'CycleHeat', 'DcfToPrice'):
        assert rows[col]['status'] in ('ok', 'degenerate'), (col, rows[col])
    assert abs(float(out['DcfToPrice'].median())) < 1e-9      # centred like the rest


# --------------------------------------------------------------------------- #
#  5. MISSINGNESS IS NEUTRAL (issue I-3, closed as a side effect)             #
# --------------------------------------------------------------------------- #
def test_zero_IS_the_observed_median_so_a_missing_metric_is_exactly_neutral(capsys):
    """Under the pre-E-1 mean-centred ruler an unavailable metric was scored ABOVE the typical
    name on 14 of 18 real columns (+0.0739 AggScore for full missingness, 29% of the distance
    to the shortlist).  Median centring makes it exact, and the squash preserves it because
    zeta(0) = 0."""
    df = _pool()
    out = _norm(df)
    capsys.readouterr()
    for col in ('earnYield', 'incomeQuality', 'currentRatio', 'CycleHeat'):
        observed = pd.to_numeric(df[col], errors='coerce').notna()
        med = float(pd.to_numeric(out[col])[observed.to_numpy()].median())
        assert abs(med) < 1e-12, (col, med)


def test_the_fill_happens_AFTER_the_scale_is_estimated(capsys):
    """An imputation must never set the scale it is then measured against (N6).  Stated as the
    property that makes it checkable: a column with k NaN cells must produce EXACTLY the mu and
    sigma of the same column with those k rows DELETED -- if the fill leaked into the estimate,
    the two would differ.  And the filled cells must land at exactly 0, which under median
    centring is the observed median and under the squash is exactly zeta(0)."""
    x = pd.Series(np.random.default_rng(13).lognormal(0, 1, 80))
    holed = x.copy()
    holed.iloc[10:25] = np.nan
    deleted = x.drop(index=range(10, 25))
    a, b = pbr.robust_location_scale(holed), pbr.robust_location_scale(deleted)
    assert (a.mu, a.sigma, a.n_obs) == (b.mu, b.sigma, b.n_obs)
    df = _pool()
    out = _norm(df)
    capsys.readouterr()
    missing = pd.to_numeric(df['earnYield'], errors='coerce').isna().to_numpy()
    assert missing.sum() == 5
    assert (out['earnYield'].to_numpy()[missing] == 0.0).all()


# --------------------------------------------------------------------------- #
#  Shape / plumbing invariants that must survive the change                   #
# --------------------------------------------------------------------------- #
def test_no_name_is_dropped_for_being_extreme(capsys):
    df = _pool()
    out, dropped = pbr.normalizeAndDropNA(df.copy(), weight_series=_W())
    capsys.readouterr()
    assert dropped == []
    assert list(out['source']) == list(df['source'])


def test_an_all_nan_row_is_still_dropped(capsys):
    df = _pool(n=30)
    cols = [c for c in df.columns if c != 'source']
    df.loc[7, cols] = np.nan
    out, dropped = pbr.normalizeAndDropNA(df.copy(), weight_series=_W())
    capsys.readouterr()
    assert dropped == ['S007']
    assert 'S007' not in set(out['source'])


def test_the_squash_is_odd_monotone_and_bounded():
    z = pd.Series(np.linspace(-500, 500, 4001))
    y = pbr.squash(z)
    assert (np.diff(y.to_numpy()) > 0).all(), 'not strictly increasing'
    assert y.abs().max() < pbr.SQUASH_K
    assert pbr.squash(pd.Series([0.0])).iloc[0] == 0.0
    #  odd
    assert np.allclose(pbr.squash(pd.Series([-1.0, -2.0, -7.0])).to_numpy(),
                       -pbr.squash(pd.Series([1.0, 2.0, 7.0])).to_numpy())
    #  identity to first order at the origin
    assert pbr.squash(pd.Series([1e-6])).iloc[0] == pytest.approx(1e-6, rel=1e-9)
    #  NaN in, NaN out -- it must not manufacture a value for a missing cell
    assert bool(pd.isna(pbr.squash(pd.Series([np.nan])).iloc[0]))


def test_the_diagnostic_accumulates_per_pool_and_never_touches_the_score(capsys):
    """N7.  postBoScoreRanking runs once per pool, so a per-call dict would leave only the last
    cohort -- the same single-writer clobber the frequency-conflict CSV hit."""
    df, W = _pool(), _W()
    pbr.NORM_DIAGNOSTICS.clear()
    a = _norm(df, W, pool_label='general')
    b = _norm(df, W, pool_label='REIT')
    capsys.readouterr()
    pools = {d['pool'] for d in pbr.NORM_DIAGNOSTICS}
    assert pools == {'general', 'REIT'}
    assert np.allclose(a.drop(columns=['source']).to_numpy(),
                       b.drop(columns=['source']).to_numpy(), equal_nan=True), \
        'the diagnostic changed the scores'
    for d in pbr.NORM_DIAGNOSTICS:
        for key in ('mu', 'sigma', 'status', 'n_passes', 'weight_retained',
                    'max_abs_z_presquash', 'max_abs_zeta', 'reach_w_x_max_zeta',
                    'span_p50_p80'):
            assert key in d, key


def test_the_pass_bound_is_not_reached_by_a_realistic_cohort_sized_column():
    """The bound is 500 because the cohorts needed up to 133, not because 100 looked tidy."""
    assert pbr.HUBER_MAX_PASSES >= 200
    rng = np.random.default_rng(21)
    for s in range(30):
        x = pd.Series(rng.lognormal(0, 1.2, 25))
        assert pbr.robust_location_scale(x).status == 'ok'


def test_the_constants_are_the_agreed_ones():
    """A tier-1 pin: c and k are the two dials of this design and both were argued from
    measurements.  Moving either is a CEO decision, not a tuning step."""
    assert pbr.HUBER_C == 1.5
    assert pbr.SQUASH_K == 3.0
    assert pbr.MAD_TO_SIGMA == 1.4826


def test_the_deployed_weights_still_normalise_over_the_canonical_key_set():
    """E-1 changed the RULER, not the vector -- and this test used to say so by pinning the
    vector's length at 21 and asserting nothing here nudged a weight.

    E-2 (2026-08-04) then changed the VECTOR, deliberately, and added two keys, so the
    original assertion is no longer the right guard: it would now fail for the intended
    reason, which makes it noise.  What survives from it is the part that is genuinely
    E-1's business -- the ruler must keep operating on a normalised vector over the
    canonical key set -- so that is what is asserted, against `METRIC_KEYS` rather than
    against a literal count that a future re-weighting will move again."""
    assert sw.sum_abs(sw.DEPLOYED) == pytest.approx(1.0, abs=1e-12)
    assert set(sw.DEPLOYED) == set(sw.METRIC_KEYS)
