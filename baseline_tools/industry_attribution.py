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


#  CYCLE CLUSTERS -- the grouping `top20-real-value-verification.md` S6.3 step 3 actually asks
#  for.  Grouping by industry LABEL under-reports the exposure that matters: on the 07-17
#  corrected list it reads "Marine Shipping 40%" when the real commodity-cycle exposure is
#  8 shipping + 4 oil & gas E&P = 60%.  Two names in different FMP industries can share one
#  cycle; the doctrine's warning is about the CYCLE, not the tag.  Both views are reported --
#  the label view stays because it is what the CSVs and the deck show.
CYCLE_CLUSTERS = {
    "Commodity/Freight cycle": [
        "Marine Shipping", "Oil & Gas Exploration & Production", "Oil & Gas Midstream",
        "Oil & Gas Refining & Marketing", "Oil & Gas Equipment & Services",
        "Oil & Gas Integrated", "Oil & Gas Drilling", "Thermal Coal", "Coking Coal",
        "Railroads", "Trucking", "Integrated Freight & Logistics",
        "Steel", "Aluminum", "Copper", "Other Industrial Metals & Mining",
        "Gold", "Silver", "Other Precious Metals & Mining", "Uranium",
        "Agricultural Inputs", "Chemicals", "Specialty Chemicals", "Lumber & Wood Production",
        "Paper & Paper Products", "Building Materials", "Airlines",
    ],
    "Rate/Credit cycle": [
        "Banks - Regional", "Banks - Diversified", "Credit Services", "Mortgage Finance",
        "Insurance - Property & Casualty", "Insurance - Life", "Insurance - Diversified",
        "Insurance - Reinsurance", "Insurance - Specialty", "Insurance Brokers",
        "Capital Markets", "Asset Management", "Financial - Data & Stock Exchanges",
        "REIT - Diversified", "REIT - Office", "REIT - Retail", "REIT - Residential",
        "REIT - Industrial", "REIT - Hotel & Motel", "REIT - Mortgage",
        "REIT - Healthcare Facilities", "REIT - Specialty",
    ],
    "Consumer cycle": [
        "Auto Manufacturers", "Auto Parts", "Auto & Truck Dealerships",
        "Residential Construction", "Homebuilding", "Furnishings, Fixtures & Appliances",
        "Apparel Manufacturing", "Apparel Retail", "Footwear & Accessories",
        "Restaurants", "Lodging", "Resorts & Casinos", "Travel Services",
        "Leisure", "Department Stores", "Specialty Retail",
    ],
}
_CYCLE_OF = {ind: cl for cl, inds in CYCLE_CLUSTERS.items() for ind in inds}


def cycle_of(industry):
    """Cycle cluster for an FMP industry label, or None if it is not cycle-classified.
    None is NOT a cluster -- an unclassified name is counted separately, never pooled."""
    return _CYCLE_OF.get(industry)


def _fmt_enr(x):
    """Enrichment factor.  `%.0fx` collapses every sub-1.5x value to '0x' or '1x', which
    reads as "no signal" for a 1.4x and as "one times" for a 0.6x -- both wrong.  Use a
    decimal below 10x."""
    if x is None or not np.isfinite(x):
        return "n/a"
    return ("%.1fx" % x) if x < 10 else ("%.0fx" % x)


def concentration_line(top_sources, universe_sources, ind=None, top_k=6, warn_share=0.25):
    """The deck pre-flight's INDUSTRY CONCENTRATION line, as one string.

    Automates `top20-real-value-verification.md` S6.3 step 3, which currently asks the CEO to
    tally industries by hand.  The COUNT alone is not the signal -- 3 of 20 in an industry that
    is 15% of the universe is nothing, while 3 of 20 in one that is 0.8% is 19x.  So the line
    carries the universe base rate and the enrichment factor, which is what turns a tally into
    something interpretable.

    Uses `industrydic_fmp_*.pickle` -- 18,333 symbols over 156 FMP industries, already loaded
    by the deck (`get_industry`) and already the pipeline's primary FIN-2/FIN-3 classifier -- so
    this introduces no new taxonomy and nothing to maintain.
    """
    if ind is None:
        ind = industry_map()
    top = list(top_sources)
    if not top:
        return "INDUSTRY CONCENTRATION: empty top list -- nothing to report."
    known_uni = [s for s in universe_sources if ind.get(s)]
    uni_c = Counter(ind[s] for s in known_uni)
    n_uni = len(known_uni)

    # ---- view 1: by INDUSTRY LABEL (what the CSVs and the deck show) ------------------
    c = Counter(ind.get(s, "UNKNOWN") for s in top)
    parts = []
    for name, k in c.most_common(top_k):
        if name == "UNKNOWN":
            parts.append("Unclassified %d" % k)
            continue
        br = uni_c.get(name, 0) / n_uni if n_uni else float("nan")
        enr = ((k / len(top)) / br) if br else float("inf")
        parts.append("%s %d (base %.2f%%, %s)" % (name, k, 100 * br, _fmt_enr(enr)))
    out = ["INDUSTRY CONCENTRATION (top-%d): %s" % (len(top), " * ".join(parts))]

    # UNKNOWN is NOT an industry and must never win the "one industry" line or trip the
    # warning: 6 unclassified names are a DATA GAP, not a concentration.
    named = Counter({k: v for k, v in c.items() if k != "UNKNOWN"})
    if named:
        top_name, top_n = named.most_common(1)[0]
        br = uni_c.get(top_name, 0) / n_uni if n_uni else float("nan")
        enr = ((top_n / len(top)) / br) if br else float("inf")
        out.append("  -> %d of %d in ONE industry (%s): universe base rate %.2f%%, "
                   "enrichment %s" % (top_n, len(top), top_name, 100 * br, _fmt_enr(enr)))
    if c.get("UNKNOWN"):
        out.append("  -> %d of %d UNCLASSIFIED (a data gap, not a concentration)"
                   % (c["UNKNOWN"], len(top)))

    # ---- view 2: by CYCLE CLUSTER (what S6.3 step 3 asks for) -------------------------
    cyc = Counter()
    for s in top:
        cl = cycle_of(ind.get(s))
        if cl:
            cyc[cl] += 1
    uni_cyc = Counter()
    for s in known_uni:
        cl = cycle_of(ind[s])
        if cl:
            uni_cyc[cl] += 1
    if cyc:
        cparts = []
        for cl, k in cyc.most_common():
            br = uni_cyc.get(cl, 0) / n_uni if n_uni else float("nan")
            enr = ((k / len(top)) / br) if br else float("inf")
            cparts.append("%s %d/%d = %.0f%% (base %.1f%%, %s)"
                          % (cl, k, len(top), 100 * k / len(top), 100 * br, _fmt_enr(enr)))
        out.append("  CYCLE CLUSTER (the grouping S6.3 step 3 asks for -- a shared cycle can "
                   "span several industry tags):")
        for p in cparts:
            out.append("    " + p)

    # THE TWO WARNINGS ARE INDEPENDENT (fix, 2026-07-30).  The first version gated the
    # label-level warning on the cycle view existing (`if cyc: ... elif ...`), so a list that
    # was 50% Software plus ONE shipping name produced a cycle block and NO warning at all --
    # the label concentration was SUPPRESSED by the presence of an unrelated cycle name, and it
    # would have fired before the cycle view was added.  A second view must never silence the
    # first: a 50%-one-industry list is a concentration whether or not that industry maps to a
    # cycle, so both conditions are now evaluated separately and both can fire.
    if named:
        worst_ind, worst_ind_n = named.most_common(1)[0]
        if worst_ind_n / len(top) >= warn_share:
            _mapped = cycle_of(worst_ind)
            out.append("  !! %.0f%% of the list is ONE INDUSTRY (%s)%s -- a concentration on "
                       "its own terms."
                       % (100 * worst_ind_n / len(top), worst_ind,
                          "" if _mapped else ", with no cycle mapping"))
    if cyc:
        worst_cl, worst_n = cyc.most_common(1)[0]
        if worst_n / len(top) >= warn_share:
            out.append("  !! %.0f%% of the list sits in ONE CYCLE (%s). Where both warnings "
                       "fire, the doctrine's cyclicality warning applies to THIS number -- it "
                       "is the larger exposure -- but read both, before position sizing."
                       % (100 * worst_n / len(top), worst_cl))
    return "\n".join(out)


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

        def prearc(df, weight_series=None, winsor_sigma=None, method=None,
                   rank_bounded=True):
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
