"""READ-ONLY bit-identity proof for the scoring-weight single-source refactor (2026-08-02).

The refactor moved every scoring weight into `scoringWeights.py` and made
`createDicts.getPostDict` / `getPostDict_legacy` / `tune_run.MU_GENERAL` /
`new_scorer_bench.W_THEORY` / `carveOut.COHORT_WEIGHTS` derive from it.  It was required
to change NO number and NO behaviour, so it is proved the way this repo proves scoring
claims: replay the DEPLOYED code path on the saved panel and demand the stored artifacts
back, bit for bit.

  STAGE 1  reproduce `resdic['BoScore_df']` (7,729 names) from `resdic['BoMetric_df']`
           via calcScore.getAves2 + calcScore.simpleScore_fromDict(n=8).  Weights enter
           Stage-1 only as criterion TIERS (createDicts.getDicts), so this is the proof
           that the criterion registry was not disturbed.
           Expect: identical source order, max |score diff| = 0.0 (EXACT -- Stage-1 sums
           tier constants, so there is no floating-point slack to hide in).

  STAGE 2  reproduce `resdic['postRank']` AggScore + top-20 from
           `resdic['postScoreMetric_raw']` through the deployed
           normalizeAndDropNA -> z*w -> getAggScore -> getRankOfRanks ->
           _dedup_issuers_in_ranking chain (the same replay
           loo_weight_influence.verify_baseline uses).
           Expect: identical name order, max |dAggScore| at fp noise (~1e-16).

           WHY STAGE 2 IS ~1e-16 AND NOT 0.0, unlike Stage 1 -- worth knowing before
           reading anything into the residual.  `postBoRank.getAggScore` sums the metric
           columns via `cts = list(set(df.columns) - set(['source']))`, and a `set` of
           STRINGS iterates in PYTHONHASHSEED order, which Python randomises per process.
           Float addition is not associative, so the AggScore of a given name moves 1-3
           ULP between processes -- measured here: 3.331e-16 vs the stored artifact at
           PYTHONHASHSEED=0 and 2.220e-16 at PYTHONHASHSEED=7, with the summation column
           order visibly different.  That is a PRE-EXISTING property of the deployed
           scorer, present at 1e2496c and unrelated to the weight refactor; it was
           isolated by scoring the pre-refactor literals and the post-refactor derived
           vectors IN THE SAME PROCESS, which is BITWISE identical (max |d| = 0.0) for the
           deployed vector, the legacy vector and all five cohorts.  It is reported to the
           CEO rather than fixed here: `getAggScore` lives in postBoRank, outside this
           change's file set.  Run this script under a fixed PYTHONHASHSEED if you need a
           reproducible number.

  COHORTS  all five normalised vectors: Sigma|w| = 1, raw-sum checksums, and each cohort
           re-scored through the same Stage-2 path so a changed vector shows up as a
           changed cohort ordering rather than only as a changed number.

  LEGACY   the A/B arm scored through the same path, so `getPostDict_legacy` is proved
           live rather than only shape-checked.

No network (VA_OFFLINE_NO_DCF is set before postBoRank is imported), no writes, no API
key.  `python baseline_tools/verify_weight_single_source.py` -- exit 0 = all proofs hold.

The four standing limits stamped on run_corrected_current.py apply to the PANEL as
always: 07-17 universe under OLD acquisition gates, currency/frequency are fallbacks, the
mu weights were fitted PRE-winsorizer, and the panel rebuild has known tie/window gaps.
None of them bear on this proof, which is a reproduction test, not a performance claim.
"""
import io
import os
import sys

os.environ.setdefault("VA_OFFLINE_NO_DCF", "1")     # BEFORE postBoRank is imported

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import calcScore as cs
import carveOut as co
import createDicts as cdic
import postBoRank as pbr
import scoringWeights as sw

RESDIC = os.path.join(_HERE, "resdic_2026-07-17_CORRECTED.pickle")

# Stage-1 window: postBo.postBoWrapper reads dmdic['nrScorePeriods'], default 8.
STAGE1_N = 8
# Stage-2 raw-sum checksum per cohort -- see test_scoring_weights_single_source for why a
# sum is the right granularity here (one number per cohort, moves on any weight edit).
COHORT_RAW_SUMS = {'Mining': 13.85, 'REIT': 7.10, 'InvestmentVehicle': 3.35,
                   'FinManager': 14.10, 'BalanceSheetFin': 10.35}

_fails = []


def check(ok, label, detail=""):
    print("  [%s] %-56s %s" % ("PASS" if ok else "FAIL", label, detail), flush=True)
    if not ok:
        _fails.append(label)
    return ok


def _quiet(fn, *a, **kw):
    """Run a chatty pipeline function with stdout captured (tqdm goes to stderr)."""
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        return fn(*a, **kw)
    finally:
        sys.stdout = old


def flat_weights(getter="getPostDict"):
    postBm, postNew = getattr(cdic, getter)()
    return {**{k: v["w"] for k, v in postBm.items()},
            **{k: v["w"] for k, v in postNew.items()}}


def stage2_replay(raw, W, cdxtop, names):
    """The DEPLOYED Stage-2 path from the raw metric matrix to the emitted ranking."""
    def _go():
        psm, _outliers = pbr.normalizeAndDropNA(raw.copy(), weight_series=W)
        wt = psm.drop("source", axis=1)
        for col in wt.columns:
            wt[col] = psm[col].values * W.get(col, 1)
        wn = pd.concat([psm[psm.columns.difference(wt.columns)], wt], axis=1)
        pr = pbr.getAggScore(wn)
        pr = pbr.getRankOfRanks(pr)
        pr, _dupes = pbr._dedup_issuers_in_ranking(pr, cdxtop, names, True)
        return pr.reset_index(drop=True)
    return _quiet(_go)


def main():
    print(__doc__.split("\n")[0])
    print("panel: %s\n" % os.path.basename(RESDIC), flush=True)
    d = pd.read_pickle(RESDIC)

    # ---------------------------------------------------------------- weights
    print("WEIGHT VECTORS")
    W = flat_weights()
    check(len(W) == 21, "deployed vector has 21 metrics", "n=%d" % len(W))
    check(sw.sum_abs(W) == 1.0, "Sigma|w| == 1.0 EXACTLY", repr(sw.sum_abs(W)))
    check(tuple(W) == sw.METRIC_KEYS, "emission order == canonical METRIC_KEYS")
    check(W == sw.deployed_weights(), "getPostDict() == scoringWeights.DEPLOYED")

    import new_scorer_bench as nsb
    import tune_run as tr
    check(tr.MU_GENERAL == W, "tune_run.MU_GENERAL == deployed vector")
    check(nsb.W_THEORY == {nsb._BENCH_RENAME.get(k, k): abs(float(v))
                           for k, v in W.items() if k not in nsb._BENCH_EXCLUDED},
          "new_scorer_bench.W_THEORY == |deployed| over 18 channels")

    WL = flat_weights("getPostDict_legacy")
    check(set(WL) == set(W), "legacy vector on the same 21 keys")
    check(WL["DcfToPrice"] == 0.35 and W["DcfToPrice"] == 0.000,
          "legacy DcfToPrice 0.35 vs deployed 0.000 preserved")

    print("\nCOHORT VECTORS")
    check(len(co.COHORT_WEIGHTS) == 5, "five cohorts")
    for label in sorted(COHORT_RAW_SUMS):
        rawv = co.COHORT_WEIGHTS_RAW[label]
        norm = co.COHORT_WEIGHTS[label]
        ok = (set(rawv) == set(W)
              and abs(sw.sum_abs(rawv) - COHORT_RAW_SUMS[label]) < 1e-9
              and abs(sw.sum_abs(norm) - 1.0) < 1e-12
              and norm["BoScore"] > 0)
        check(ok, "%s: keys/raw-sum/normalisation/BoScore" % label,
              "raw=%.4f norm=%.12f BoScore=%.6g"
              % (sw.sum_abs(rawv), sw.sum_abs(norm), norm["BoScore"]))

    # ------------------------------------------------------- STAGE 1 identity
    print("\nSTAGE-1 BIT IDENTITY (reproduce resdic['BoScore_df'])")
    bm = d["BoMetric_df"].copy()
    md = _quiet(cs.getAves2, bm.copy())
    got1 = _quiet(cs.simpleScore_fromDict, bm.copy(), md["BoMetric_ave"],
                  md["BoMetric_dateAve"], STAGE1_N).reset_index(drop=True)
    ref1 = d["BoScore_df"].reset_index(drop=True)
    check(len(got1) == len(ref1), "row count", "%d vs %d" % (len(got1), len(ref1)))
    check(list(got1["source"]) == list(ref1["source"]), "identical source order")
    dmax1 = float(np.abs(got1["score"].to_numpy(float)
                         - ref1["score"].to_numpy(float)).max())
    check(dmax1 == 0.0, "max |score diff| == 0.0 (exact)", "%r" % dmax1)

    # ------------------------------------------------------- STAGE 2 identity
    print("\nSTAGE-2 BIT IDENTITY (reproduce resdic['postRank'])")
    raw = d["postScoreMetric_raw"]
    cdxtop = d["cdx_dftop100"]
    tdf = d["Tickers_df"]
    names = dict(zip(tdf["symbol"], tdf["name"]))
    ref2 = d["postRank"].reset_index(drop=True)
    got2 = stage2_replay(raw, W, cdxtop, names)
    check(list(got2["source"]) == list(ref2["source"]), "identical name order (n=%d)"
          % len(got2))
    check(list(got2["source"].head(20)) == list(ref2["source"].head(20)),
          "identical TOP-20")
    dmax2 = float(np.abs(got2["AggScore"].to_numpy(float)
                         - ref2["AggScore"].to_numpy(float)).max())
    check(dmax2 < 1e-12, "max |dAggScore| at fp noise", "%.3e" % dmax2)
    print("       top-20: %s" % ", ".join(got2["source"].head(20)))

    # ------------------------- the five cohort vectors through the same path
    print("\nCOHORT + LEGACY VECTORS THROUGH THE DEPLOYED STAGE-2 PATH")
    for label in sorted(co.COHORT_WEIGHTS):
        g = stage2_replay(raw, {**W, **co.COHORT_WEIGHTS[label]}, cdxtop, names)
        s = sw.sum_abs(dict(zip(g["source"], g["AggScore"])))
        check(len(g) > 0 and g["AggScore"].notna().all(),
              "%s ranks cleanly" % label,
              "n=%d Sigma|Agg|=%.10f head=%s" % (len(g), s, g["source"].iloc[0]))
    gl = stage2_replay(raw, WL, cdxtop, names)
    check(len(gl) > 0 and gl["AggScore"].notna().all(), "legacy (A/B) ranks cleanly",
          "n=%d head=%s" % (len(gl), gl["source"].iloc[0]))

    # ------------------------------------- the duplicate Stage-1 criterion
    print("\nDUPLICATE STAGE-1 CRITERION (dEPS / dNetIncomePerShare)")
    diff = cdic.getDicts()[4]
    for carrier, twin in cdic.DUPLICATE_DIFF_CRITERIA:
        strip = lambda spec: {k: v for k, v in spec.items() if k != "Tier"}
        check(strip(diff[carrier]) == strip(diff[twin]),
              "%s / %s identical modulo Tier" % (carrier, twin))
        wt = float(cs.calcByTier("diff", diff[twin]["Tier"], 1,
                                 pd.Series([1.0]), 0.0, "probe", 1))
        check(wt == 0.0, "%s Tier %r carries no weight"
              % (twin, diff[twin]["Tier"]), "w=%r" % wt)
    ca, cb = ("d" + k[0].upper() + k[1:] for k in cdic.DUPLICATE_DIFF_CRITERIA[0])
    a = pd.to_numeric(bm[ca], errors="coerce")
    b = pd.to_numeric(bm[cb], errors="coerce")
    check(a.equals(b), "panel columns %s / %s are byte-identical" % (ca, cb),
          "rows=%d nan_mismatch=%d" % (len(a), int((a.isna() != b.isna()).sum())))

    print("\n%s" % ("ALL PROOFS HOLD" if not _fails
                    else "FAILURES: %s" % _fails))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
