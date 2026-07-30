"""Emit the four run artifacts `generate_presentation` requires -- FULLY OFFLINE.

WHY THIS EXISTS (and a correction to the record).  I previously reported the corrected deck as
BLOCKED, on the grounds that its `AggScoreTop100` input can only be produced by
`postBo.writeBoAggToCSV`, which makes ~4 live FMP calls per name (~400 for a top-100).  That
reasoning was WRONG in the part that mattered: the deck does not need that CSV's API-sourced
columns.  Every one of its ~20 `aggscore_df` reads is individually guarded
(`if aggscore_df is not None`, then a per-ticker row lookup), so a REDUCED-SCHEMA CSV carrying
only the offline-computable columns degrades gracefully to gap tags -- which is the deck's
designed behaviour for missing data.  The only genuine network leg in the deck is the OPTIONAL
yfinance augmentation behind a persistent store, disabled with `--no-augment`.

So the deck is regenerable offline, and the CEO could have had it.  This module makes that
one command.

WHAT IT WRITES into `--run-dir` (the four files `resolve_run_artifacts` requires, all with the
SAME run-date so the deck's cross-run-mixing guard is satisfied):
    postRank_<date>_<ds>_<tf>.pickle       postRank (+ moatScore merged) + cdx_df + moatdf
    Boresults_dic-<ds>_<tf>_all_<date>_*.pickle   Tickers_df + carveout_* + the panel frames
    AggScoreTop100-<date>_<ds>_<tf>.csv    REDUCED SCHEMA -- see below
    ForensicFlagsTop100-<date>_<ds>_<tf>.csv     full, computed offline

REDUCED-SCHEMA HONESTY.  The emitted AggScore CSV carries everything derivable with no network:
`source`, `CycleHeat` (from `postScoreMetric_raw` -- NOT from postRank, which holds `z x w`; see
postBo.py's note), `moatScore`, **`M-Score` and `C-Score`** (from `SLmeanMscore` /
`SLmeanCscore`), `sloanAccruals` and the forensic flags.  Only the API-only columns (`price`,
`PE-ratio`, `beta`, `sector`, `rating_fmp`, `DCF-to-Price`) are ABSENT, and they are absent
rather than blank-filled so the deck shows gaps instead of plausible-looking zeros.

TWO LESSONS BAKED IN, both from the first version of this module:
  * "OFFLINE" IS NOT A LICENCE TO DROP WHAT IS COMPUTABLE.  M-Score / C-Score were left out on
    the assumption they were part of the API-sourced block.  They are not -- they come from
    `detectManipulation`, which is pure-offline -- so the deck gapped the forensic layer on all
    35 pages, in exactly the layer the review lens leans on, and the schema note did not even
    list them as missing.  Anything offline-computable must be computed.
  * SCOPE THE MOAT FRAME TO THE DECK'S WHOLE MEMBERSHIP.  Passing only the general top-100 left
    every moat column NaN for cohort members, and 130 peer bars rendered "pool too small (n=0)"
    -- a message about the pool that was really about an empty column.
A `_SCHEMA_NOTE` column states the remaining absences on every row, and
`generate_presentation.schema_note_banner` now renders it as a page-level banner so the READER
of the deck sees it, not just the reader of the CSV.
"""

import argparse
import os
import sys

os.environ.setdefault("VA_OFFLINE_NO_DCF", "1")

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import detectManipulation as dm
import forensicFlags as ff
import postBo as pb
import reporting_period as rp

#  The note must list EXACTLY what is absent.  Its first version omitted M-Score / C-Score from
#  the list while they were in fact missing from the file -- a schema note that under-reports
#  the gaps is worse than none, because the reader trusts it.  M/C are now COMPUTED, so they
#  are no longer absent; the list below is the real remainder.
SCHEMA_NOTE = ("OFFLINE REDUCED SCHEMA -- this file was NOT written by the pipeline. ABSENT "
               "(not zero, not blank-because-zero): price, PE-ratio, beta, sector, rating_fmp, "
               "DCF-to-Price -- all of which require live API calls. PRESENT and offline-"
               "computed: CycleHeat (RAW metric from postScoreMetric_raw, not the weighted "
               "column), moatScore, M-Score, C-Score, sloanAccruals and the forensic flags. "
               "Any field the deck shows as a gap is a gap in THIS FILE, not in the company.")


def emit(resdic_path, run_dir, run_date, datasource="fmp", tickerfilter="stock_NA1_EU1",
         topn=100, verbose=True):
    os.makedirs(run_dir, exist_ok=True)
    resdic = dict(pd.read_pickle(resdic_path))

    # --- moatScore over the DECK'S WHOLE MEMBERSHIP, not just the general top-100 -------
    # BUG FOUND 2026-07-30: this first passed only postRank['source'] (100 names), but the
    # deck builds its peer pools over the general pool PLUS all five cohort side-lists.  Every
    # moat column for a name outside the 100 then joined as NaN, so 130 peer bars rendered
    # "pool too small (n=0)" -- the pool was not small, the column was empty.  Live Sbocker
    # passes resdic['BoScore_df']['source'] (the full scored universe); the union of the deck's
    # actual membership is the same thing for the deck's purposes and is far cheaper.
    src = list(dict.fromkeys(
        list(resdic["postRank"]["source"])
        + [s for sd in (resdic.get("carveout_sidelists") or {}).values()
           if sd and "postRank" in sd for s in list(sd["postRank"]["source"])]
        + [s for lst in (resdic.get("carve_full_membership") or {}).values()
           for s in list(lst)]))
    print("  moatIdentifier over %d name(s) (general pool + all cohort side-lists)" % len(src))
    moatdf = pb.moatIdentifier(pd.Series(src), resdic["cdx_df"])
    if "moatScore" not in resdic["postRank"].columns and not moatdf.empty:
        resdic["postRank"] = resdic["postRank"].merge(
            moatdf[["source", "moatScore"]], on="source", how="left")
    resdic["moatdf"] = moatdf

    # --- forensic models (offline) ------------------------------------------------------
    resdic = {**resdic, **dm.detectManipulationWrapper(resdic)}
    flag_df = ff.buildForensicFlagTable(resdic, topn)

    # --- the four files ---------------------------------------------------------------
    tag = "%s_%s" % (datasource, tickerfilter)
    pr_path = os.path.join(run_dir, "postRank_%s_%s.pickle" % (run_date, tag))
    pd.to_pickle({"postRank": resdic["postRank"], "cdx_df": resdic["cdx_df"],
                  "moatdf": moatdf}, pr_path)

    bo_path = os.path.join(
        run_dir, "Boresults_dic-%s_all_%s_OFFLINE.pickle" % (tag, run_date))
    pd.to_pickle({"Tickers_df": resdic.get("Tickers_df"),
                  "carveout_sidelists": resdic.get("carveout_sidelists"),
                  "carveout_labels": resdic.get("carveout_labels"),
                  "carveout_diagnostics": resdic.get("carveout_diagnostics"),
                  "cdx_df": resdic["cdx_df"], "BoMetric_df": resdic.get("BoMetric_df"),
                  "postRank": resdic["postRank"]}, bo_path)

    # AggScore CSV -- reduced schema.  CycleHeat comes from the RAW frame; taking it from
    # postRank would republish the sign-inverted `z x w` (the 2026-07-29 defect).
    head = resdic["postRank"].head(topn)
    agg = pd.DataFrame({"source": list(head["source"])})
    raw = resdic.get("postScoreMetric_raw")
    if raw is not None and "CycleHeat" in raw.columns:
        _c = raw.set_index("source")["CycleHeat"]
        agg["CycleHeat"] = [_c.get(s, np.nan) for s in agg["source"]]
    if "moatScore" in head.columns:
        agg["moatScore"] = head["moatScore"].values
    # M-Score / C-Score are FULLY OFFLINE (detectManipulationWrapper) and the deck reads them
    # from this CSV.  Omitting them left a forensic gap on all 35 pages -- in exactly the layer
    # the CEO's review lens leans on -- while the schema note did not even list them as absent.
    # Same per-name fields the live writer uses (postBo.py:593,603).
    # NB the per-name means live in SLmeanMscore / SLmeanCscore, NOT in mscore_df / cscore_df
    # (which hold the per-PERIOD component rows).  postBo.py:593,603 reads the same frames.
    for _key, _df, _col in (("M-Score", resdic.get("SLmeanMscore"), "M_Score_mean"),
                            ("C-Score", resdic.get("SLmeanCscore"), "C_Score_mean")):
        if _df is not None and not _df.empty and _col in _df.columns:
            _m = _df.drop_duplicates("source").set_index("source")[_col]
            agg[_key] = [_m.get(s, np.nan) for s in agg["source"]]
        else:
            print("  !! %s unavailable (%s missing) -- it will gap in the deck"
                  % (_key, _col))
    if flag_df is not None and not flag_df.empty:
        keep = [c for c in ("source", "isFinancial", "financialKind", "forensicValid",
                            "M_flag_gt_-1.78", "M_drivers", "C_flag_ge_4", "C_flags_fired",
                            "sloanAccruals", "sloan_worstQuintile_inShortlist",
                            "forensicTag") if c in flag_df.columns]
        agg = agg.merge(flag_df[keep], on="source", how="left")
    agg["_SCHEMA_NOTE"] = SCHEMA_NOTE
    agg_path = os.path.join(run_dir, "AggScoreTop%d-%s_%s.csv" % (topn, run_date, tag))
    agg.to_csv(agg_path, index=False)

    fo_path = os.path.join(run_dir, "ForensicFlagsTop%d-%s_%s.csv" % (topn, run_date, tag))
    ff.writeForensicFlagsCSV(flag_df, fo_path)

    if verbose:
        for p in (pr_path, bo_path, agg_path, fo_path):
            print("  wrote %-72s %8.1f KB" % (os.path.basename(p),
                                              os.path.getsize(p) / 1024))
        print("  AggScore CSV columns: %s" % list(agg.columns))
    return {"postRank": pr_path, "boresults": bo_path, "aggscore": agg_path,
            "forensic": fo_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resdic", default=os.path.join(
        _HERE, "resdic_2026-07-17_CORRECTED.pickle"))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-date", default="2026-07-17")
    ap.add_argument("--topn", default=100, type=int)
    args = ap.parse_args()
    print("emit_deck_inputs -> %s (run-date %s)" % (args.run_dir, args.run_date))
    emit(args.resdic, args.run_dir, args.run_date, topn=args.topn)
    print("done -- now:\n  python generate_presentation.py --run-dir %s --no-augment "
          "--run-date %s --out presentations/presentation_%s_CORRECTED.html"
          % (args.run_dir, args.run_date, args.run_date))


if __name__ == "__main__":
    main()
