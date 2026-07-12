"""
NaN-not-0 for degenerate inputs  (design s2B / MED#7, build-blocker).

THE BUG (postBoRank.normalizeAndDropNA:604-605): after cross-sectional z-scoring,
missing metrics are `fillna(0)` -- silently set to the cross-sectional MEAN (z=0)
and summed into AggScore.  A name with a degenerate DCF (FCF0<=0, financial, short
history) therefore gets a FORCED-neutral DcfToPrice instead of being treated as
MISSING.  The design wants NaN to PROPAGATE as missing so reliability-shrinkage
handles it (a name we know less about should score toward the middle by SHRINKAGE,
not by silently imputing the mean on one channel while scoring the rest at face).

SCOPE / invariant.  The live OLD scorer is frozen as the A/B baseline; changing its
fillna(0) would change live AggScore and break the as_of=None bit-for-bit invariant.
So this fix is applied on the PIT / reproduction scoring path (as_of is not None),
NOT by editing the live path.  `reliability_shrink` is the corrected aggregation;
`legacy_neutralize` reproduces the old live aggregation for the A/B baseline -- it
mirrors the live sequence z-score -> fillna(0) -> PER-METRIC WEIGHTS -> sum, so it
matches the frozen live AggScore ONLY when passed the same `weight_series` the live
scorer uses (postBoRank.py:528-535).  Called with weights=None it is an UNWEIGHTED
illustration of the fillna(0) bug, NOT the live AggScore.  The NEW scorer already
handles missing via its Phi term (new_scorer_bench.new_composites) and needs no
change here.

reliability_shrink mirrors the new scorer's Phi:
    Phi_i = 1 - (1 - phi_min) * d_i**theta ,   d_i = fraction of metrics missing
    S_i   = Phi_i * sum_{m present} w_m * z_{i,m}
Missing z stays NaN (min_count=1) -- never imputed to 0.
"""
import numpy as np
import pandas as pd

PHI_MIN = 0.5
THETA = 2.0


def reliability_shrink(z_df, weights=None, phi_min=PHI_MIN, theta=THETA):
    """Aggregate a z-score DataFrame (index=name, columns=metrics) into a composite,
    propagating NaN as MISSING and shrinking under-observed names.

    weights : optional dict metric->weight (default 1 each).  Returns a Series
    indexed by name."""
    metrics = [c for c in z_df.columns if c != "source"]
    Z = z_df[metrics].apply(pd.to_numeric, errors="coerce")
    if weights:
        W = pd.Series({m: weights.get(m, 1.0) for m in metrics})
        contrib = Z.mul(W, axis=1)
    else:
        contrib = Z
    present = Z.notna().sum(axis=1)
    d = 1.0 - present / float(len(metrics))
    phi = 1.0 - (1.0 - phi_min) * d ** theta
    # min_count=1 -> a name with ALL metrics missing stays NaN (dropped downstream),
    # never a spurious 0.
    raw = contrib.sum(axis=1, min_count=1)
    return phi * raw


def legacy_neutralize(z_df, weights=None):
    """Reproduction of the OLD live aggregation for the A/B baseline: fillna(0) ->
    per-metric weights -> sum, mirroring the live path (postBoRank.py: `fillna(0)`
    at :605, then `col * weight_series.get(col, 1)` at :528-535, then the
    `getAggScore` sum at :623).

    weights : dict metric->weight.  Pass the SAME weight_series the live scorer uses
        to faithfully reproduce the frozen live AggScore.  With weights=None this is
        an UNWEIGHTED illustration of the fillna(0) bug only -- it does NOT match the
        live AggScore whenever any metric weight != 1 (e.g. DcfToPrice=0.35).  Do NOT
        use on the PIT path."""
    metrics = [c for c in z_df.columns if c != "source"]
    Z = z_df[metrics].apply(pd.to_numeric, errors="coerce").fillna(0)
    if weights:
        W = pd.Series({m: weights.get(m, 1.0) for m in metrics})
        Z = Z.mul(W, axis=1)
    return Z.sum(axis=1)
