"""Provenance of PUBLISHED metric columns -- the `postRank` weighted-column class.

THE DEFECT THESE PIN (found 2026-07-29).  `postBoRank` multiplies every metric column by its
weight BEFORE assembling `postRank`, so `postRank['<metric>']` is `z x w` for all 20 metric
columns -- none are raw.  `AggScoreTop100-*.csv` copied `CycleHeat` straight out of it under a
comment asserting it was the metric, so the published column was `z x (-0.080)`: correlation
to the true metric EXACTLY -1.0000 on the 07-17 panel, i.e. the CSV's minimum was the pool's
HOTTEST name and the cyclicality read was inverted for every row.

CycleHeat inverts EXACTLY (rather than approximately) because it is winsor-exempt, so its
z-score is an exact affine function of the raw value -- which is also why no downstream
consistency check could have caught it: the column looked perfectly well-behaved.
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

import createDicts as cdic


def _weights():
    a, b = cdic.getPostDict()
    return {**{k: float(a[k]["w"]) for k in a}, **{k: float(b[k]["w"]) for k in b}}


def test_every_metric_column_in_postRank_is_weighted_not_raw():
    """The premise of the whole class, asserted against PRODUCTION -- not a replica.

    An earlier version of this test hand-copied postBoRank's normalise->weight->aggregate
    sequence into the test body, which meant it verified the replica and would have passed
    even if production stopped weighting.  It now drives the real
    `postBoRank.postBoScoreRanking` metric-assembly path through its own helper so the frame
    under test IS the frame production builds.
    """
    import postBoRank as pbr
    import inspect
    src = inspect.getsource(pbr.postBoScoreRanking)
    # the production sequence: normalise, multiply by weight_series, THEN getAggScore
    assert "normalizeAndDropNA(postScoreMetric_df" in src
    assert "temp_normpsmdf_weighted[col] = postScoreMetric_df[col].values * w" in src
    i_w = src.index("temp_normpsmdf_weighted[col] = postScoreMetric_df[col].values * w")
    i_a = src.index("postRank = getAggScore(psmdf_normalized)")
    assert i_w < i_a, "weighting no longer precedes aggregation -- re-derive this test file"

    # and demonstrate numerically, on the real panel's saved artifacts when available
    cache = os.path.join(_HERE, "resdic_2026-07-17_CORRECTED.pickle")
    if not os.path.exists(cache):
        pytest.skip("no saved resdic on this machine; the source-order assertions above hold")
    r = pd.read_pickle(cache)
    pr, raw, W = r["postRank"], r["postScoreMetric_raw"], _weights()
    n_checked = 0
    for c in [c for c in W if W[c] != 0 and c in pr.columns and c in raw.columns]:
        a = pd.to_numeric(pr.set_index("source")[c], errors="coerce")
        b = pd.to_numeric(raw.set_index("source")[c], errors="coerce")
        j = pd.concat([a, b], axis=1).dropna()
        if len(j) < 5:
            continue
        assert not np.allclose(j.iloc[:, 0], j.iloc[:, 1]), \
            "%s in postRank equals the raw metric -- premise stale" % c
        n_checked += 1
    assert n_checked >= 15, "expected to check most metric columns, checked %d" % n_checked


#  EXHAUSTIVE INVENTORY of modules that read a METRIC-NAMED column off a postRank-derived
#  frame.  The 2026-07-29 sweep found THREE sites, not the two an inspection turned up -- which
#  is precisely why this is a frozen inventory rather than a spot check.  Adding a new module
#  to this list is a REVIEW EVENT: postRank's metric columns are `z x w`, so a new reader is
#  wrong by default unless it un-weights or renames.
POSTRANK_METRIC_READERS = {
    "postBo.py": "CycleHeat now published from raw_df; moatScore raw by merge order",
    "backtest_outputs.py": "save_stock_picks renames to *_weighted_contrib and drops the "
                           "w=0 BoScore; compute_ols_weighted_ranking un-weights via the "
                           "shared helper so it matches the fit basis",
    "backtest_unified.py": "un-weights via postBoRank.unweight_postrank_metrics before fitting",
    "backtest_ols_analysis.py": "same shared helper (detected by the DERIVED route -- it "
                                "builds its column list from dtypes, which the literal-only "
                                "scan missed)",
    "carveOut.py": "reads source/AggScore only (partition + dedup)",
    "generate_presentation.py": "sources metrics from cdx_df / a raw recompute, not postRank",
    "reviewReference.py": "uses postScoreMetric_raw by design",
    "postBoRank.py": "the producer, and the home of the un-weighting helper",
    # --- verified 2026-07-29 when the frozen inventory first ran; all safe -------------
    "detectManipulation.py": "reads grossProfitMargin off tempcdx_df (cdx_df), not postRank; "
                             "its only postRank use is ['source']",
    "validate_gate.py": "READS the weighted columns deliberately and says so in its own "
                        "docstring ('postRank holds the weighted metric columns'); it "
                        "SUBTRACTS pr[metric] from AggScore to measure shortlist sensitivity "
                        "and never publishes a metric value. The one genuinely correct use.",
    "stage2_pit.py": "PRODUCES the raw metric frame; only postRank['source'] is read back",
    "tune_run.py": "same -- producer; only postRank['source'] is read back",
    "emit_deck_inputs.py": "reads CycleHeat from postScoreMetric_raw, never postRank",
    # --- verified 2026-07-30, surfaced only once the DERIVED route was added ------------
    "portfolio.py": "postRank['source'] only (portfolio.py:305-306); no metric column read",
    "run_target_test.py": "postRank['AggScore'] only, guarded (line 150); AggScore is a score, "
                          "not a metric",
    # NOTE: `pick_log.py` was on this list and matched NOTHING -- it reads AggScore only and
    # takes its entry valuations from cdx_df.  Removed rather than left as a stale claim.
    # --- reviewed 2026-07-31 ------------------------------------------------------------
    "loo_weight_influence.py":
        "CLEARED by an independent cold arithmetic review (2026-07-31): all 18 arms "
        "re-derived in closed form, max abs diff 0.000e+00 on every statistic, and there is "
        "NO DIVISION BY ANY WEIGHT anywhere in the file. It sources raw z from "
        "postScoreMetric_raw and re-runs normalizeAndDropNA exactly as production does. The "
        "guard fires on this file as a LITERAL-NAME FALSE POSITIVE: the six metric names it "
        "matches sit in a CANDIDATES list, not in a weighted-column read. "
        "BUT READ THIS BEFORE COPYING THE PATTERN -- the exemption is narrower than it looks: "
        "resdic['psmdf_normalized'] IS LITERALLY THE SAME OBJECT as resdic['postRank'] "
        "(getAggScore mutates its argument in place and returns it), so a module reading "
        "'psmdf_normalized' IS reading a postRank frame whose metric columns are z x w. This "
        "script is safe only because its `c in W` filter excludes AggScore and "
        "rankOfRanks_diag; change that filter and the exemption is void. Aliasing verified, "
        "not assumed.",
    "verify_part5_defects.py":
        "CLEARED by audit 2026-08-01 (commissioned by the MD for the Part 5 defect "
        "verification; read-only, no network). It DOES read postRank's metric columns -- the "
        "guard is a TRUE POSITIVE on the trigger, not a misfire -- and all five reads are "
        "weight-aware: FOUR un-weight explicitly by dividing by the metric's own weight "
        "(grahamNumberToPrice twice, marketCapRevQuants, and a six-column block via "
        "`zz[c] = zz[c] / W[c]`), and the FIFTH reads z x w DELIBERATELY, as the finding "
        "itself -- it is the psbrfilter analysis, which exists to show that the -1.5 cutoff is "
        "being applied to a WEIGHTED z, and it prints min(z*w) and min(z) side by side. Every "
        "other `pr[...]` access is pr['source'] or pr['AggScore'], neither of which is a "
        "metric column. The file's own output states the z*w basis in prose. Exempt because it "
        "is CORRECT about the basis, not because it is out of scope.",
}


def _postrank_metric_readers():
    """Modules that reference postRank AND reach metric columns, by EITHER route.

    THE GAP THIS CLOSES (2026-07-30).  The first version matched QUOTED LITERALS only, so it
    would NOT have caught `backtest_ols_analysis.py` -- the very file that carried the
    coefficient-sign defect -- because that file derives its column list from DTYPES
    (`[c for c in postRank.columns if postRank[c].dtype in (...)]`).  A guard that misses the
    known instance of the bug it guards against is worse than none: it reads as coverage.
    So the scan now flags BOTH routes:
      * literal   -- a metric name appears as a quoted string;
      * derived   -- the file iterates or subsets postRank's columns programmatically.
    """
    import glob
    import re
    metric_names = [m for m, w in _weights().items() if w != 0]
    # programmatic column derivation off a postRank-ish frame
    derived_pat = re.compile(
        r"(postRank|postrank_df|postrank|fb_df|fbdf_tocsv|pr)\s*\.\s*columns"
        r"|for\s+\w+\s+in\s+(postRank|postrank_df|fb_df)\b"
        r"|\.select_dtypes\(", re.I)
    found = {}
    for path in glob.glob(os.path.join(_REPO, "*.py")) + \
            glob.glob(os.path.join(_REPO, "baseline_tools", "*.py")):
        base = os.path.basename(path)
        if base.startswith("test_"):
            continue
        txt = open(path, encoding="utf-8", errors="replace").read()
        if "postRank" not in txt and "postrank" not in txt:
            continue
        literal = [m for m in metric_names if ("'%s'" % m) in txt or ('"%s"' % m) in txt]
        derived = bool(derived_pat.search(txt))
        if literal or derived:
            found[base] = {"literal": literal, "derived": derived}
    return found


def test_no_unreviewed_module_reads_a_metric_column_off_postRank():
    """Freeze the inventory.  Any file that reaches postRank's metric columns -- by literal name
    OR by programmatic/dtype derivation -- must be on the reviewed list."""
    found = _postrank_metric_readers()
    offenders = {k: v for k, v in found.items() if k not in POSTRANK_METRIC_READERS}
    assert not offenders, (
        "UNREVIEWED module(s) reach postRank's metric columns: %s\n"
        "postRank's metric columns are z x weight. Either un-weight (see "
        "postBoRank.unweight_postrank_metrics), or rename to declare the basis, then add the "
        "module to POSTRANK_METRIC_READERS with the reason." % offenders)


def test_the_dtype_derivation_route_is_actually_detected():
    """Guard the guard: the scan must flag the file that derives columns from dtypes, because
    that is the route the literal-only version missed."""
    found = _postrank_metric_readers()
    assert "backtest_ols_analysis.py" in found, \
        "the dtype-derivation route is not detected -- the guard has the same blind spot again"
    assert found["backtest_ols_analysis.py"]["derived"] is True


def test_every_allowlist_entry_matches_something():
    """An allow-list entry that matches nothing is a STALE CLAIM -- it makes the list look more
    thorough than it is, and it is how an entry added from reasoning rather than from a hit
    survives.  One did: `pick_log.py`, which reads AggScore only.  Asserted against the scan
    itself so there is a single list, not two that can drift apart.
    """
    found = set(_postrank_metric_readers())
    stale = sorted(k for k in POSTRANK_METRIC_READERS if k not in found)
    assert not stale, ("allow-list entries that match nothing -- remove them, or fix the scan: "
                       "%s" % stale)


def test_save_stock_picks_declares_the_weighted_basis():
    import inspect
    import backtest_outputs as bo
    src = inspect.getsource(bo.save_stock_picks)
    assert "_weighted_contrib" in src, "save_stock_picks still publishes bare metric names"
    # and the raw-by-merge-order column keeps its plain name
    assert "_RAW_COLS = ['moatScore']" in src.replace('"', "'")


@pytest.mark.parametrize("mod", ["backtest_unified", "backtest_ols_analysis"])
def test_regression_consumers_unweight_before_regressing(mod):
    src = open(os.path.join(_REPO, mod + ".py"), encoding="utf-8").read()
    assert "unweight_postrank_metrics" in src, \
        "%s regresses on z x w, so CycleHeat's coefficient sign is inverted" % mod


def test_unweight_helper_recovers_the_metric_z_and_drops_zero_weights():
    import postBoRank as pbr
    W = _weights()
    rng = np.random.default_rng(4)
    n = 50
    z = pd.DataFrame({"source": ["S%02d" % i for i in range(n)]})
    for c in W:
        z[c] = rng.normal(size=n)
    weighted = z.copy()
    for c in W:
        weighted[c] = z[c] * W[c]
    out, kept, dropped = pbr.unweight_postrank_metrics(weighted)
    assert set(dropped) == {c for c, w in W.items() if w == 0}
    for c in kept:
        assert np.allclose(out[c].to_numpy(), z[c].to_numpy(), atol=1e-12), c


def test_OLS_rerank_score_has_the_INTENDED_SIGN_for_a_negative_weight_metric():
    """THE assertion that would have caught both the original defect and the regression my own
    fix introduced.

    `standardize(z x w) == sign(w) * standardize(z)`, so if the coefficients are fitted on the
    un-weighted z but applied to the weighted column (or vice versa), every negative-weight
    metric enters the score with the WRONG SIGN.  Measured when only the fit side was
    un-weighted: corr(OLS_Score, true CycleHeat) went +1.0000 -> -1.0000.  Checking the two
    call sites for a string is not enough -- the failure is in their AGREEMENT, so the test has
    to run the score end-to-end and look at the sign.
    """
    import backtest_outputs as bo
    W = _weights()
    neg = [c for c, w in W.items() if w < 0]
    assert neg, "no negative-weight metric left; this guard needs re-deriving"
    metric = neg[0]                                     # CycleHeat
    rng = np.random.default_rng(11)
    n = 60
    z_true = rng.normal(size=n)
    postrank = pd.DataFrame({"source": ["S%02d" % i for i in range(n)],
                             metric: z_true * W[metric]})      # postRank holds z x w
    # a POSITIVE coefficient fitted on the metric z means "more of this metric -> higher score"
    coefs = pd.DataFrame({"metric": [metric], "coefficient": [1.0]})
    out = bo.compute_ols_weighted_ranking(postrank, coefs)
    assert out is not None
    r = float(np.corrcoef(out["OLS_Score"].to_numpy(),
                          out["source"].map(dict(zip(postrank["source"], z_true)))
                          .to_numpy())[0, 1])
    assert r > 0.99, ("OLS_Score is anti-correlated with the metric its coefficient was "
                      "fitted on (corr=%.4f) -- the fit and re-rank bases disagree" % r)


def test_save_stock_picks_does_not_publish_the_zeroed_BoScore():
    """`postRank['BoScore']` is identically -0.0 (w = 0.000 multiplied it away).  Publishing a
    constant zero under the Stage-1 score's name is the same misrepresentation as the weighted
    columns."""
    import inspect
    import backtest_outputs as bo
    src = inspect.getsource(bo.save_stock_picks)
    i = src.index("_SCORE_COLS")
    line = src[i:src.index("\n", i)]
    assert "BoScore" not in line, "BoScore is still published from postRank as a constant zero"


def test_cycleheat_in_postRank_is_sign_inverted_against_the_metric():
    """The specific mechanism: CycleHeat is winsor-exempt, so z is affine in raw, and w<0
    makes the weighted column an EXACT negative image."""
    import postBoRank as pbr
    assert "CycleHeat" in pbr.WINSOR_EXEMPT_BOUNDED
    W = _weights()
    assert W["CycleHeat"] < 0
    rng = np.random.default_rng(1)
    n = 80
    psm = pd.DataFrame({"source": ["S%02d" % i for i in range(n)],
                        "CycleHeat": rng.normal(size=n)})
    for c in ("RoA", "earnYield"):
        psm[c] = rng.normal(size=n)
    normed, _ = pbr.normalizeAndDropNA(psm.copy(), weight_series=W)
    published_if_buggy = normed["CycleHeat"].to_numpy() * W["CycleHeat"]
    truth = psm["CycleHeat"].to_numpy()
    r = float(np.corrcoef(published_if_buggy, truth)[0, 1])
    assert r == pytest.approx(-1.0, abs=1e-9), \
        "expected an exact sign inversion, got corr=%.6f" % r


def test_writeBoAggToCSV_takes_a_raw_frame_and_refuses_to_fall_back():
    """The fix's contract, checked without invoking the function (it makes ~4 live API calls
    per name).  `raw_df` must exist, and the fallback branch must publish NOTHING rather than
    re-publish the weighted column."""
    import inspect
    import postBo as pb
    sig = inspect.signature(pb.writeBoAggToCSV).parameters
    assert "raw_df" in sig, "writeBoAggToCSV has no raw_df parameter"
    src = inspect.getsource(pb.writeBoAggToCSV)
    # the CycleHeat value must come from raw_df
    assert "raw_df.set_index('source')['CycleHeat']" in src.replace('"', "'")
    # and the no-raw_df branch must NOT assign CycleHeat from fbdf_tocsv
    tail = src.split("if raw_df is not None", 1)[1]
    bad = "BoComp_tocsv['CycleHeat'] = fbdf_tocsv"
    assert bad not in tail.replace('"', "'"), \
        "the fallback still publishes the WEIGHTED CycleHeat"


def test_writeResWrapper_passes_the_raw_frame():
    import inspect
    import postBo as pb
    src = inspect.getsource(pb.writeResWrapper)
    assert "raw_df=resdic.get('postScoreMetric_raw')" in src.replace('"', "'")


def test_moatScore_is_raw_because_it_is_merged_after_scoring():
    """The other column copied out of postRank.  It is SAFE, and the reason is an ordering
    fact worth pinning: the merge happens in Sbocker AFTER postBoScoreRanking has already run
    getAggScore, so moatScore is never weighted and never summed into AggScore.  If that merge
    ever moves earlier, moatScore would silently become a 21st scored column."""
    src = open(os.path.join(_REPO, "Sbocker.py"), encoding="utf-8").read()
    assert "resdic['postRank'].merge(moat_merge, on='source', how='left')" in src
    i_score = src.index("postBoWrapper")
    i_merge = src.index("moat_merge")
    assert i_score < i_merge, "moatScore is merged BEFORE scoring -- it would be summed"
    a, b = cdic.getPostDict()
    assert "moatScore" not in a and "moatScore" not in b, \
        "moatScore is now a weight_series key -- it would be weighted"
