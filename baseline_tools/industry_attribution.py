"""WHICH FIX CREATED THE SHIPPING CONCENTRATION?  Per-correction ablation of the deployed top-20.

THE OBSERVATION (07-17 universe).  `Marine Shipping` is 58 of 7,674 industry-classified names
= **0.756%** of the scored universe.  The SHIPPED pre-fix top-20 held **2** of them (10%, 13x
enrichment).  The corrected top-20 holds **8** (40%, **53x**) -- and a further 4 are Oil & Gas
E&P, so 12 of 20 sit in one asset-heavy cyclical cluster.  A 10% -> 40% jump in one industry
across a correctness wave is either (a) the corrections removing an artifact that was
SUPPRESSING genuinely cheap shipping names, or (b) one or more corrections systematically
favouring the shipping accounting profile -- asset-heavy, high D&A, volatile earnings, low book
multiples.  (a) is good news; (b) means one bias was swapped for another.  This tells them apart.

METHOD.  Start from the fully corrected panel and REVERT ONE CORRECTION AT A TIME, then re-run
the DEPLOYED path (Stage-1 -> cohort carve + $25M floor -> general top-100 -> Stage-2
re-normalised over that pool -> issuer-dedup -> top-20) and count the industry.  A revert that
collapses shipping representation is the correction responsible.

Reverting rather than adding is deliberate: the corrections interact (the corrected `price`
feeds every yield metric AND the Graham ratio), so building up from the pre-fix panel would
attribute interaction effects to whichever fix happened to be applied first.  One-at-a-time
reverts off the full model measure each fix's MARGINAL contribution to the shipped outcome,
which is the quantity the question is about.

ABLATIONS
  price_old        `price` restored to the pre-fix quarterly-PE derivation (~1/4 of the real
                   share price), taken VERBATIM from the original saved panel -- not
                   recomputed -- so the revert is exact.  Prime mechanical suspect: it feeds
                   every yield/value metric, and shipping is a high-yield cohort.
  graham_old       `grahamNumber` restored to FMP's quarterly field from the original panel.
  no_cfoLessEarn   the `CFOlessEarnings` Tier-S criterion removed from Stage-1 scoring.
                   THE SPECIALIST'S PRIME SUSPECT: CFO - NI > 0 is structurally satisfied by
                   any high-depreciation industry, and shipping is among the most D&A-heavy.
  no_ndte_annual   the Stage-1 flow-scale factor disabled for `netDebtToEBITDA` only, i.e. the
                   leverage gate back on a per-period EBITDA.  Shipping is leveraged, so
                   annualising a leverage ratio can flip its gate wholesale.
  no_winsorizer    Stage-2 normalisation reverted to the pre-arc un-winsorized z + |z|>4
                   ejection, metrics untouched.


ALL FIGURES ON THE 07-17 UNIVERSE (old gates: ~523 pricefails ~72% non-US, plus the lenfail
16->8 cohort never fetched).  Industry labels come from `industrydic_fmp_*.pickle` (18,333
symbols, 156 FMP industries), which is already the pipeline's FIN-2/FIN-3 classifier.
"""

import argparse
import os
import sys
from collections import Counter

os.environ.setdefault("VA_OFFLINE_NO_DCF", "1")

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
import data_quality as dq
import panel_upgrade as pu
import postBo as pb
import reporting_period as rp

#  The concentration/counter code THIS script originated now lives at the repo root, because
#  the pipeline emits it every run (postBo.writeResWrapper) and a pipeline stage must not
#  import an analysis script.  Re-exported here so `ia.concentration_line(...)` /
#  `ia.cycle_of(...)` keep working for the existing caller (run_corrected_current.py) and so
#  there is exactly one implementation of the count in the repo.  See
#  industry_concentration.py's docstring for the three rendering differences in the lift.
from industry_concentration import (  # noqa: F401  (re-export)
    CYCLE_CLUSTERS,
    concentration_line,
    counter_block,
    cycle_of,
    industry_counts,
    report_lines,
)

ORIG_PANEL = (r"C:\Users\stefanthorarinsson\Documents\HomeGDrive"
              r"\Boresults_dic-fmp_stock_NA1_EU1_all_2026-07-17_len7752_manelim3692"
              r"_fails2075.pickle")
FOCUS = "Marine Shipping"
ABLATIONS = ["none", "price_old", "graham_old", "no_cfoLessEarn", "no_ndte_annual",
             "no_winsorizer", "shipped_prefix"]
SHIPPED_PREFIX_CSV = os.path.join(
    os.path.expanduser("~"), "Documents", "HomeGDrive",
    "AggScoreTop100-2026-07-17_fmp_stock_NA1_EU1.csv")


def industry_map():
    return co._load_industry_map()


def base_rate(sources, ind):
    known = [s for s in sources if ind.get(s)]
    n_focus = sum(1 for s in known if ind[s] == FOCUS)
    return n_focus, len(known), (n_focus / len(known) if known else float("nan"))


# --------------------------------------------------------------------------- #
#  Ablated panel construction                                                 #
# --------------------------------------------------------------------------- #
def _revert_column(cdx_new, cdx_old, col):
    """Restore `col` from the ORIGINAL panel, matched on (source, date) so a row-order
    difference cannot silently misalign it."""
    key = ["source", "date"]
    a = cdx_new.copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    b = cdx_old.copy()
    b["date"] = pd.to_datetime(b["date"], errors="coerce")
    old = (b.drop_duplicates(subset=key).set_index(key)[col])
    idx = pd.MultiIndex.from_arrays([a["source"], a["date"]])
    a[col] = old.reindex(idx).to_numpy()
    n_na = int(pd.isna(a[col]).sum()) - int(pd.isna(cdx_new[col]).sum())
    print("  revert %-14s: %d rows now NaN that were not before (unmatched (source,date))"
          % (col, max(0, n_na)))
    return a


def build_panel(ablation, verbose=True):
    """Return (cdx, BoMetric) for the requested ablation."""
    up = pd.read_pickle(os.path.join(_HERE, "cache_panel_upgraded_2026-07-17.pickle"))
    orig = pd.read_pickle(ORIG_PANEL)
    cdx = up["cdx_df"].copy()
    if ablation == "price_old":
        cdx = _revert_column(cdx, orig["cdx_df"], "price")
    elif ablation == "graham_old":
        cdx = _revert_column(cdx, orig["cdx_df"], "grahamNumber")

    if ablation in ("none", "no_winsorizer"):
        return cdx, up["BoMetric_df"].copy()

    # every other ablation changes Stage-1 construction, so BoMetric must be rebuilt
    patches = []
    if ablation == "no_ndte_annual":
        real = rp.stage1_flow_factor

        def patched(key, rpy):
            return 1.0 if key == "netDebtToEBITDA" else real(key, rpy)
        rp.stage1_flow_factor = patched
        patches.append(lambda: setattr(rp, "stage1_flow_factor", real))
    try:
        bm = pu.rebuild_bometric(cdx, verbose=verbose)
    finally:
        for undo in patches:
            undo()
    return cdx, bm


def run_deployed(cdx, bm, ablation, verbose=False):
    """The DEPLOYED path via production postBoWrapper.  Returns the ranked source list."""
    orig = pd.read_pickle(ORIG_PANEL)
    dmdic = dict(orig)
    dmdic["cdx_df"] = cdx
    dmdic["BoMetric_df"] = bm
    dmdic["api_key"] = ""
    dmdic["baseurl"] = "OFFLINE"
    for k in ("BoMetric_ave", "BoMetric_dateAve"):
        dmdic.pop(k, None)
    dmdic = dq.apply_data_quality_filter(dmdic, verbose=False, save_log=False)

    undo = []
    if ablation == "no_cfoLessEarn":
        # Drop the Tier-S CFOlessEarnings criterion from Stage-1 scoring by removing it from
        # the criterion dict the scorer reads.  Removing the CRITERION is the right ablation;
        # NaN-ing the column would instead make every name FAIL it, which is a different and
        # much larger intervention.
        real = cdic.getDicts

        def patched():
            t = list(real())
            special = dict(t[6])
            special.pop("CFOlessEarnings", None)
            t[6] = special
            return tuple(t)
        cdic.getDicts = patched
        undo.append(lambda: setattr(cdic, "getDicts", real))
    if ablation == "no_winsorizer":
        import postBoRank as pbr
        real_n = pbr.normalizeAndDropNA

        #  **_kw so this historical stand-in keeps accepting whatever the LIVE
        #  normalizeAndDropNA's signature grows (it took `winsor_sigma` in 2026-07, takes
        #  `huber_c` / `squash_k` / `pool_label` since the 2026-08-03 E-1 change).  The
        #  stand-in reproduces the PRE-ARC path deliberately and must not track those
        #  parameters -- but a TypeError on an added keyword would silently kill the ablation.
        def prearc(df, weight_series=None, method=None, rank_bounded=True, **_kw):
            d = df.copy().reset_index(drop=True)
            mc = [c for c in d.columns if c != "source"]
            for c in mc:
                d[c] = pd.to_numeric(d[c], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan)
            keep = d[mc].isna().sum(axis=1) < len(mc)
            dropped = list(d.loc[~keep, "source"])
            dn = d[keep].copy()
            num = dn.drop("source", axis=1)
            z = ((num - num.mean()) / num.std().replace(0, np.nan).fillna(1)).fillna(0)
            dn[z.columns] = z
            ok = (z.abs() <= 4).all(axis=1)
            return dn[ok].copy(), sorted(set(dropped) | set(dn.loc[~ok, "source"]))
        pbr.normalizeAndDropNA = prearc
        undo.append(lambda: setattr(pbr, "normalizeAndDropNA", real_n))

    try:
        dmdic = {**dmdic, **cs.getAves2(dmdic["BoMetric_df"])}
        dmdic["nrScorePeriods"] = orig.get("nrScorePeriods", 8)
        resdic = pb.postBoWrapper(dmdic, as_of=None)
    finally:
        for u in undo:
            u()
    return list(resdic["postRank"]["source"]), dmdic["cdx_df"]


def report(name, ranked, ind, br, topn=20):
    top = ranked[:topn]
    c = Counter(ind.get(s, "UNKNOWN") for s in top)
    k = c.get(FOCUS, 0)
    k100 = sum(1 for s in ranked[:100] if ind.get(s) == FOCUS)
    enr = (k / len(top)) / br if br and k else 0.0
    print("\n%-16s top-%d %s = %d/%d (%.0f%%)  enrichment %.0fx   | top-100 = %d"
          % (name, topn, FOCUS, k, len(top), 100 * k / len(top), enr, k100))
    print("   industries: %s" % c.most_common(5))
    return {"ablation": name, "focus_top20": k, "focus_top100": k100,
            "pct_top20": 100.0 * k / len(top), "enrichment": enr,
            "top20": ",".join(top),
            "industries_top20": "; ".join("%s=%d" % kv for kv in c.most_common(8))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations", default=",".join(ABLATIONS))
    ap.add_argument("--topn", default=20, type=int)
    ap.add_argument("--out", default=os.path.join(_HERE, "industry_attribution.csv"))
    args = ap.parse_args()

    ind = industry_map()
    bar = "=" * 96
    print(bar)
    print("INDUSTRY ATTRIBUTION of the deployed top-%d -- focus: %s" % (args.topn, FOCUS))
    print("  ALL FIGURES ON THE 07-17 UNIVERSE")
    print(bar, flush=True)

    rows = []
    for ab in [a.strip() for a in args.ablations.split(",") if a.strip()]:
        print("\n" + "-" * 96 + "\nABLATION: %s" % ab + "\n" + "-" * 96, flush=True)
        # PER-ABLATION ISOLATION: one arm failing must not cost the other six.  It already did
        # once -- the old `prearc_all` arm hit calcScore's schema gate and took the summary and
        # the CSV down with it after ~25 minutes of completed work.
        try:
            if ab == "shipped_prefix":
                # The ACTUAL shipped pre-fix list, read from disk.  Re-SCORING the pre-fix
                # panel with current code is impossible by design: calcScore's schema gate
                # refuses a BoMetric without `CFOlessEarnings` rather than mix metric bases
                # silently.  That refusal is the guard working, and the shipped CSV is the
                # right pre-fix reference regardless -- it is what actually went to the CEO.
                ranked = list(pd.read_csv(SHIPPED_PREFIX_CSV)["source"])
                cdx_used = pd.read_pickle(ORIG_PANEL)["cdx_df"]
            else:
                cdx, bm = build_panel(ab)
                ranked, cdx_used = run_deployed(cdx, bm, ab)
            nf, nk, br = base_rate(sorted(set(cdx_used["source"])), ind)
            print("  universe base rate: %d/%d = %.3f%%" % (nf, nk, 100 * br))
            rows.append({**report(ab, ranked, ind, br, args.topn),
                         "universe_focus": nf, "universe_known": nk,
                         "universe_pct": 100 * br})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("  !! ABLATION %s FAILED (%s: %s) -- recorded as failed; others continue."
                  % (ab, type(e).__name__, e), flush=True)
            rows.append({"ablation": "%s [FAILED: %s]" % (ab, type(e).__name__),
                         "focus_top20": -1, "focus_top100": -1,
                         "pct_top20": float("nan"), "enrichment": float("nan"),
                         "top20": "", "industries_top20": "",
                         "universe_focus": -1, "universe_known": -1,
                         "universe_pct": float("nan")})
    df = pd.DataFrame(rows)
    print("\n" + bar)
    print("SUMMARY -- marginal effect of reverting each correction")
    print(bar)
    print(df[["ablation", "focus_top20", "pct_top20", "enrichment", "focus_top100",
              "universe_pct"]].to_string(index=False, float_format=lambda v: "%.2f" % v))
    if "none" in set(df["ablation"]):
        full = int(df.loc[df["ablation"] == "none", "focus_top20"].iloc[0])
        print("\n  DELTA vs the fully-corrected list (%d in top-%d):" % (full, args.topn))
        for _, r in df.iterrows():
            if r["ablation"] == "none":
                continue
            print("    revert %-16s -> %d  (%+d)" % (r["ablation"], r["focus_top20"],
                                                     int(r["focus_top20"]) - full))
    df.to_csv(args.out, index=False)
    print("\nwrote %s" % args.out)
    print(bar, flush=True)


if __name__ == "__main__":
    main()
