"""CORRECTED CURRENT RANKED OUTPUT from the saved 2026-07-17 panel -- the deliverable.

WHAT THIS IS
------------
The measurement arc scored HISTORICAL buy anchors (2020/2021/2022 year-ends) to test whether
the score orders subsequent returns.  It never produced a CURRENT ranked list.  This does:
it takes the `panel_upgrade`d 07-17 panel and runs it as-of its OWN date through the FULL
DEPLOYED path -- Stage-1 -> cohort carve + $25M floor -> general top-100 -> Stage-2
re-normalised OVER THAT POOL -> issuer-dedup -> market-cap bands -> sector side-lists --
using the CURRENT SHIPPED WEIGHTS and the CURRENT WINSORIZED-Z normaliser.

NOTHING IS FITTED HERE.  `getPostDict()` is read as-is.

It is the production `postBo.postBoWrapper`, not a reproduction: the only concession to
running offline is `VA_OFFLINE_NO_DCF=1`, which skips Stage-2's per-ticker DCF call.  That is
provably score-neutral (DcfToPrice w = 0.000) and postBoRank REFUSES if that weight ever
becomes non-zero.

HOW TO READ THE OUTPUT -- FOUR LIMITS, none of them cosmetic
-----------------------------------------------------------
1. **07-17 UNIVERSE, OLD GATES.**  The panel is what the OLD acquisition gates admitted.
   Roughly 500+ names the corrected gates would admit were NEVER FETCHED -- about 523
   price-fails (~72% non-US) plus the lenfail 16->8 cohort -- so no offline recompute can
   put them in.  THIS IS NOT THE LIST A FRESH FETCH PRODUCES.
2. **CURRENCY AND FREQUENCY ARE FALLBACKS, not the ingest verdict.**  `reportedCurrency` is
   absent from this panel, so USD market caps (and therefore the BANDS) come from the coarse
   exchange-suffix table; `period` is absent, so reporting frequency comes from date
   cadence on already-quarter-SNAPPED dates.  Band membership near a cutoff and the
   quarterly/semi-annual split are both softer than a stamped run's.
3. **THE WEIGHTS ARE MISMATCHED TO THE NORMALISER.**  The mu vector was fitted 2026-07-14
   (`38621fd`) through the then-current normaliser, which was un-winsorized z + |z|>4
   ejection; the winsorizer landed 2026-07-25 (`69c3671`).  Measured effect of that
   normaliser change on universe-wide rank-IC: about +0.18 (paired bootstrap, 3 anchors).
   Since winsorization acts as a per-column EFFECTIVE RE-WEIGHTING, the weights this list is
   ordered by are not the weights that were fitted for this normaliser.
   => TREAT MEMBERSHIP AS THE SIGNAL AND ORDERING WITHIN THE LIST AS PROVISIONAL.
4. **REBUILD FIDELITY.**  `panel_upgrade` gaps C and D: tie order is inverted for the 282
   duplicate-snapped-quarter sources (1,639 of 5,501 rows, 0.92% of the panel) and the
   rebuild's windows straddle the 3,522 rows / 551 sources the data-quality filter had
   already removed before the ingest computed Stage-1.

Emits the deliverable CSVs, a diff against the SHIPPED pre-fix 07-17 list with a
per-name attribution of the largest movers, and a `resdic` pickle the presentation
generator can consume.
"""

import argparse
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
import data_quality as dq
import postBo as pb
import postBoRank as pbr

RUN_DATE = "2026-07-17"


def code_provenance():
    """`<sha>[-dirty]` of the tree that produced these files -- see
    emit_deck_inputs.code_provenance for why a dated artifact must name its own code."""
    import subprocess
    try:
        sha = subprocess.run(["git", "-C", _REPO, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=15).stdout.strip()
        if not sha:
            return "unknown"
        dirty = bool(subprocess.run(["git", "-C", _REPO, "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    timeout=15).stdout.strip())
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


#  RECORDED IN THE ARTIFACT because it is the kind of number that gets quoted without its
#  population and then means the opposite of what it says.
DOUBLE_PENALTY_CAVEAT = (
    "CycleHeat/EPStoEPSmean overlap is POOL-DEPENDENT AND THE SIGN FLIPS: Spearman is "
    "-0.6515 (86.7% opposite-sign) on the DEPLOYED top-100 pool, but +0.27 across the WHOLE "
    "07-17 UNIVERSE. The double-penalty finding is real WHERE THE WEIGHTS OPERATE (the pool) "
    "and REVERSES universe-wide. Never quote either figure without naming its population. "
    "This is a CEO weight decision; no weight has been changed.")

BANNER_LIMITS = [
    "07-17 UNIVERSE / OLD GATES -- ~500+ names the corrected gates would admit were never "
    "fetched (~523 pricefails, ~72% non-US, plus the lenfail 16->8 cohort). NOT the list a "
    "fresh fetch produces.",
    "CURRENCY + FREQUENCY ARE FALLBACKS -- reportedCurrency absent (bands use the coarse "
    "exchange-suffix FX table); `period` absent (frequency from date cadence on SNAPPED "
    "dates).",
    "WEIGHTS MISMATCHED TO THE NORMALISER -- mu fitted 2026-07-14 (38621fd) pre-winsorizer; "
    "winsorizer landed 2026-07-25 (69c3671); measured ~+0.18 rank-IC effect. Membership is "
    "the signal; ORDERING inside the list is PROVISIONAL.",
    "REBUILD FIDELITY -- tie order inverted on 282 duplicate-quarter sources (0.92% of rows); "
    "windows straddle the 3,522 data-quality-pruned rows (551 sources).",
]


def build_dmdic(panel_path, orig_panel_path, run_data_quality=True):
    """dmdic for postBoWrapper, from the UPGRADED panel + the original run's config keys."""
    up = pd.read_pickle(panel_path)
    orig = pd.read_pickle(orig_panel_path)
    dmdic = dict(orig)                       # config keys (baseurl, api_key, sectors, ...)
    dmdic["cdx_df"] = up["cdx_df"].copy()
    dmdic["BoMetric_df"] = up["BoMetric_df"].copy()
    if up.get("Tickers_df") is not None:
        dmdic["Tickers_df"] = up["Tickers_df"]
    for k in ("BoMetric_ave", "BoMetric_dateAve"):
        dmdic.pop(k, None)
    # api_key is NEVER read on this path (VA_OFFLINE_NO_DCF=1 skips the only consumer);
    # blank it so the secret cannot reach a log line, and so a regression that re-enables
    # the fetch fails loudly instead of silently spending calls.
    dmdic["api_key"] = ""
    dmdic["baseurl"] = "OFFLINE"

    if run_data_quality:
        # Sbocker runs this before scoring; the saved cdx was already filtered once, so a
        # second pass should be a no-op. Report it either way rather than assume.
        n0 = len(dmdic["BoMetric_df"]), len(dmdic["cdx_df"])
        dmdic = dq.apply_data_quality_filter(dmdic, verbose=False, save_log=False)
        n1 = len(dmdic["BoMetric_df"]), len(dmdic["cdx_df"])
        print("data-quality pass on the rebuilt panel: BoMetric %d->%d, cdx %d->%d"
              % (n0[0], n1[0], n0[1], n1[1]), flush=True)

    # Stage-1 baselines on the frame ACTUALLY scored (audit H-1: never carry stale medians)
    dmdic = {**dmdic, **cs.getAves2(dmdic["BoMetric_df"])}
    dmdic["nrScorePeriods"] = orig.get("nrScorePeriods", 8)
    return dmdic


# --------------------------------------------------------------------------- #
#  Deliverables                                                               #
# --------------------------------------------------------------------------- #
def emit_lists(resdic, dmdic, outdir, tag):
    postRank = resdic["postRank"].reset_index(drop=True)
    general = postRank[["source", "AggScore"]].copy()
    general.insert(0, "rank", np.arange(1, len(general) + 1))

    _names = (dict(zip(dmdic["Tickers_df"]["symbol"], dmdic["Tickers_df"]["name"]))
              if dmdic.get("Tickers_df") is not None else None)
    bres = co.partition_by_marketcap(postRank, dmdic["cdx_df"], names=_names)

    # Stamp provenance + the pool-dependence caveat on the artifact itself, so neither can be
    # separated from the numbers they qualify.
    _prov = ("Generated from code commit %s on %s | %s"
             % (code_provenance(), pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                DOUBLE_PENALTY_CAVEAT))
    general = general.copy()
    general["_PROVENANCE"] = _prov
    general.to_csv(os.path.join(outdir, "CORRECTED_general_top100-%s.csv" % tag), index=False)

    band_tables = {}
    for label, *_ in co.MCAP_BANDS:
        b = bres["bands"].get(label)
        if b is None or len(b) == 0:
            band_tables[label] = pd.DataFrame(columns=["rank", "source", "AggScore"])
            continue
        bb = pd.DataFrame(b)[["source", "AggScore"]].copy()
        bb.insert(0, "rank", np.arange(1, len(bb) + 1))
        band_tables[label] = bb
    pd.concat([t.assign(band=lab, note=bres["band_note"][lab],
                        full_member_count=bres["band_counts"][lab],
                        selective=bres["band_selective"][lab])
               for lab, t in band_tables.items()], ignore_index=True) \
        .to_csv(os.path.join(outdir, "CORRECTED_bands-%s.csv" % tag), index=False)

    # INDICATIVE banding ONLY -- see the printout.  partition_by_marketcap deliberately
    # calls marketcap_usd_by_source WITHOUT the exchange-suffix fallback, so on a panel with
    # no `reportedCurrency` EVERY name is unknown-mcap and routes to General: the pipeline
    # REFUSES to band this data, by design (CEO 2026-07-18: "nothing wrong ships on current
    # data; bands degrade gracefully / marked pending-currency").  Overriding that inside the
    # production call would be inventing a verdict the pipeline declines to give, so instead
    # the suffix-FX view is computed SEPARATELY and labelled indicative-only.
    ind = None
    try:
        mcu = co.marketcap_usd_by_source(dmdic["cdx_df"], allow_suffix_fallback=True)
        rows = []
        for _, r in general.iterrows():
            v = mcu.get(r["source"])
            rows.append({"rank": r["rank"], "source": r["source"],
                         "AggScore": r["AggScore"], "mcap_usd_suffixFX": v,
                         "band_indicative": co.band_for_marketcap_usd(v) or "UNKNOWN"})
        ind = pd.DataFrame(rows)
        ind.to_csv(os.path.join(outdir, "CORRECTED_bands_INDICATIVE-%s.csv" % tag),
                   index=False)
    except Exception as e:
        print("  indicative banding unavailable: %s: %s" % (type(e).__name__, e))

    side = {}
    for lab, rd in (resdic.get("carveout_sidelists") or {}).items():
        if rd is None or "postRank" not in rd:
            side[lab] = pd.DataFrame(columns=["rank", "source", "AggScore"])
            continue
        s = rd["postRank"][["source", "AggScore"]].head(5).copy()
        s.insert(0, "rank", np.arange(1, len(s) + 1))
        side[lab] = s
    if side:
        pd.concat([t.assign(cohort=lab) for lab, t in side.items()], ignore_index=True) \
            .to_csv(os.path.join(outdir, "CORRECTED_sidelists-%s.csv" % tag), index=False)
    return general, band_tables, side, bres, ind


# --------------------------------------------------------------------------- #
#  Diff vs the SHIPPED pre-fix list, with per-name attribution                #
# --------------------------------------------------------------------------- #
def attribute_moves(resdic, old_sources, new_general, top_n=20):
    """For each big mover, the ONE weighted metric contributing most to its AggScore.

    `psmdf_normalized` holds the WEIGHTED, normalised columns that AggScore is the row-sum
    of, so a name's largest-magnitude column IS the largest single driver of where it sits.
    This attributes the LEVEL, not the change (the pre-fix per-name contributions are not
    recoverable from the shipped CSV, which carries no normalised columns) -- so it answers
    "why is this name here now", which is the question, and it does NOT claim to decompose
    the delta.
    """
    w = resdic.get("psmdf_normalized")
    if w is None:
        return pd.DataFrame()
    w = w.copy()
    drop = [c for c in ("AggScore", "rankOfRanks", "rankOfRanks_diag") if c in w.columns]
    cols = [c for c in w.columns if c not in drop + ["source"]]
    rows = []
    newrank = dict(zip(new_general["source"], new_general["rank"]))
    oldrank = {s: i + 1 for i, s in enumerate(old_sources)}
    for _, r in w.iterrows():
        s = r["source"]
        v = pd.to_numeric(pd.Series({c: r[c] for c in cols}), errors="coerce").dropna()
        if v.empty:
            continue
        top = v.abs().sort_values(ascending=False)
        nr, orr = newrank.get(s), oldrank.get(s)
        rows.append({
            "source": s, "new_rank": nr, "old_rank": orr,
            "status": ("NEW" if orr is None else
                       ("moved %+d" % (orr - nr) if nr is not None else "left")),
            "AggScore": float(v.sum()),
            "top_driver": top.index[0], "top_driver_contrib": float(v[top.index[0]]),
            "2nd_driver": top.index[1] if len(top) > 1 else None,
            "2nd_driver_contrib": float(v[top.index[1]]) if len(top) > 1 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("new_rank")


def diff_report(new_general, old_csv, top_n=20):
    old = pd.read_csv(old_csv)
    old_sources = list(old["source"])
    new_sources = list(new_general["source"])
    new_top, old_top = new_sources[:top_n], old_sources[:top_n]
    rows = []
    for s in dict.fromkeys(new_top + old_top):
        nr = new_sources.index(s) + 1 if s in new_sources else None
        orr = old_sources.index(s) + 1 if s in old_sources else None
        in_new_top, in_old_top = s in new_top, s in old_top
        if in_new_top and not in_old_top:
            verdict = "ENTERS top-%d" % top_n
        elif in_old_top and not in_new_top:
            verdict = "LEAVES top-%d" % top_n
        else:
            verdict = "stays"
        rows.append({"source": s, "old_rank": orr, "new_rank": nr,
                     "delta": (orr - nr) if (orr and nr) else None, "verdict": verdict})
    d = pd.DataFrame(rows)
    d["_k"] = d["new_rank"].fillna(9999)
    return d.sort_values("_k").drop(columns="_k"), old_sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=os.path.join(_HERE,
                    "cache_panel_upgraded_2026-07-17.pickle"))
    ap.add_argument("--orig-panel", default=r"C:\Users\stefanthorarinsson\Documents"
                    r"\HomeGDrive\Boresults_dic-fmp_stock_NA1_EU1_all_2026-07-17"
                    r"_len7752_manelim3692_fails2075.pickle")
    ap.add_argument("--old-list", default=r"C:\Users\stefanthorarinsson\Documents"
                    r"\HomeGDrive\AggScoreTop100-2026-07-17_fmp_stock_NA1_EU1.csv")
    ap.add_argument("--outdir", default=_HERE)
    ap.add_argument("--tag", default="2026-07-17-CORRECTED")
    ap.add_argument("--save-resdic", default=None)
    args = ap.parse_args()

    bar = "=" * 100
    print(bar)
    print("CORRECTED CURRENT RANKED OUTPUT -- deployed path on the panel_upgrade'd %s panel"
          % RUN_DATE)
    for i, lim in enumerate(BANNER_LIMITS, 1):
        print("  LIMIT %d: %s" % (i, lim))
    print(bar, flush=True)

    dmdic = build_dmdic(args.panel, args.orig_panel)
    resdic = pb.postBoWrapper(dmdic, as_of=None)
    resdic = {**resdic, **{k: dmdic[k] for k in ("cdx_df", "BoMetric_df", "Tickers_df")
                           if k in dmdic}}

    general, band_tables, side, bres, ind = emit_lists(resdic, dmdic, args.outdir, args.tag)

    print("\n" + bar)
    print("GENERAL TOP-20  (top-5 called out)   [%s]" % args.tag)
    print(bar)
    t20 = general.head(20)
    for _, r in t20.iterrows():
        star = "  <== TOP-5" if r["rank"] <= 5 else ""
        print("  %2d. %-14s AggScore %+.4f%s" % (r["rank"], r["source"], r["AggScore"], star))

    # DECK PRE-FLIGHT LINE (automates top20-real-value-verification.md S6.3 step 3, which
    # currently asks the CEO to tally industries by hand).  Printed with the list rather than
    # after it, because a 40%-one-industry shortlist changes how every subsequent line is read.
    try:
        import industry_attribution as ia
        print("\n" + bar)
        print(ia.concentration_line(list(t20["source"]),
                                    sorted(set(dmdic["cdx_df"]["source"]))))
        print(bar, flush=True)
    except Exception as e:
        print("\n  industry-concentration pre-flight unavailable: %s: %s"
              % (type(e).__name__, e))

    print("\n" + bar)
    print("MARKET-CAP BANDS -- as the PIPELINE returns them")
    print(bar)
    print("  currency_pending = %s | unknown_mcap (routed to General) = %d"
          % (bres["currency_pending"], bres["unknown_mcap"]))
    for lab, t in band_tables.items():
        n_show = 20 if lab == "General" else 5
        print("  %-16s full_count=%-5d selective=%-5s  %s"
              % (lab, bres["band_counts"][lab], bres["band_selective"][lab],
                 bres["band_note"][lab]))
        if len(t):
            print("      %s" % ", ".join("%s(%+.3f)" % (r["source"], r["AggScore"])
                                         for _, r in t.head(n_show).iterrows()))
    if bres["currency_pending"]:
        print("\n  *** THE BAND DELIVERABLE IS NOT AVAILABLE ON THIS PANEL. ***")
        print("  `reportedCurrency` is absent, so marketcap_usd_by_source returns nothing and")
        print("  EVERY name is unknown-mcap -> routed to General. This is the pipeline")
        print("  behaving as specified (CEO 2026-07-18: bands degrade gracefully / marked")
        print("  pending-currency rather than misband a non-USD name). Bands become real on")
        print("  the first fetch that carries reportedCurrency -- not before.")
    if ind is not None:
        print("\n  INDICATIVE ONLY (exchange-suffix FX, NOT the pipeline's verdict) --")
        print("  distribution over the general top-100: %s"
              % ind["band_indicative"].value_counts().to_dict())
        for lab, *_ in co.MCAP_BANDS:
            sub = ind[ind["band_indicative"] == lab].head(20 if lab == "General" else 5)
            if len(sub):
                print("    %-16s %s" % (lab, ", ".join(
                    "%s(#%d)" % (r["source"], r["rank"]) for _, r in sub.iterrows())))

    print("\n" + bar)
    print("SECTOR COHORT SIDE-LISTS (top-5 each)")
    print(bar)
    for lab, t in side.items():
        print("  %-20s %s" % (lab, ", ".join(t["source"].tolist()) or "(empty)"))

    d, old_sources = diff_report(general, args.old_list)
    print("\n" + bar)
    print("DIFF vs the SHIPPED PRE-FIX 07-17 LIST (top-20 both sides)")
    print(bar)
    print(d.to_string(index=False))
    n_enter = int((d["verdict"].str.startswith("ENTERS")).sum())
    print("  %d of 20 slots turn over; overlap = %d/20" % (n_enter, 20 - n_enter))

    attr = attribute_moves(resdic, old_sources, general)
    if not attr.empty:
        print("\n" + bar)
        print("WHY: dominant weighted-metric contribution per new top-20 name")
        print("  (attributes the LEVEL, not the delta -- the shipped CSV carries no "
              "normalised columns, so a true decomposition of the change is not "
              "recoverable from it)")
        print(bar)
        print(attr[attr["new_rank"] <= 20].to_string(index=False,
              float_format=lambda v: "%+.4f" % v))
        attr.to_csv(os.path.join(args.outdir, "CORRECTED_attribution-%s.csv" % args.tag),
                    index=False)
    d.to_csv(os.path.join(args.outdir, "CORRECTED_diff_vs_shipped-%s.csv" % args.tag),
             index=False)

    if args.save_resdic:
        pd.to_pickle(resdic, args.save_resdic)
        print("\nwrote resdic -> %s" % args.save_resdic)
    print("\n" + bar)
    for i, lim in enumerate(BANNER_LIMITS, 1):
        print("  LIMIT %d: %s" % (i, lim))
    print(bar, flush=True)


if __name__ == "__main__":
    main()
