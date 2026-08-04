"""Dump the PRE-NORMALISATION Stage-2 metric frame (`psm`) + Stage-1 BoScore for one panel.

WHY A DUMP.  The attribution question -- did the pre-fix AggScore invert because of the
METRICS, or because of two MISSING-DATA REWARDS the arc happened to remove? -- is a
2x2-plus design over (metric frame) x (normalisation / fill treatment).  Running it as
whole-pipeline configurations conflates the two factors and forces every treatment to be
implemented twice, once per code tree.  Dumping the metric frame instead makes the factor
boundary explicit: the frame is the ONLY thing the tree determines, and every treatment is
then applied in ONE place (baseline_tools/attribution_arms.py) to both frames identically.

Runs unmodified in the PRE-ARC tree and the current tree.  Records the tree's identity in
the dump so no arm can end up on disk without a provenance stamp -- the defect the
devils-advocate gate found in the first pass, where the pre-arc arm existed only as two
summary rows with no basis stamp and no artifact.
"""

import argparse
import inspect
import os
import subprocess
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage2_pit as s2


def _tree_stamp():
    """Best-effort identity of the code tree this is running in."""
    out = {"tree_path": _REPO}
    try:
        out["git_sha"] = subprocess.run(
            ["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
            text=True, timeout=20).stdout.strip() or None
        out["git_dirty"] = bool(subprocess.run(
            ["git", "-C", _REPO, "status", "--porcelain"], capture_output=True,
            text=True, timeout=20).stdout.strip())
    except Exception as e:
        out["git_sha"], out["git_dirty"] = None, None
        out["git_error"] = "%s: %s" % (type(e).__name__, e)
    # Structural markers that identify the tree even with no git metadata (a `git archive`
    # export has none): these are the exact features the correctness arc added.
    out["has_reporting_period"] = os.path.exists(os.path.join(_REPO, "reporting_period.py"))
    try:
        import postBoRank as pbr
        sig = inspect.signature(pbr.normalizeAndDropNA).parameters
        out["normalize_params"] = list(sig)
        #  Three eras of the z-path, distinguishable without git metadata:
        #    winsorizer only          -> has_winsorizer True,  has_robust_scale False
        #    winsorizer + rank switch -> + has_rank_method True
        #    robust ruler (2026-08-03 E-1) -> has_winsorizer False, has_robust_scale True
        out["has_winsorizer"] = hasattr(pbr, "_winsorize_raw")
        out["has_robust_scale"] = hasattr(pbr, "robust_location_scale")
        out["has_rank_method"] = "method" in sig
    except Exception as e:
        out["normalize_params"] = "ERR %s" % e
    try:
        import stage2_metrics as sm
        out["has_MCAP_QUANT_MISSING"] = hasattr(sm, "MCAP_QUANT_MISSING")
    except Exception:
        out["has_MCAP_QUANT_MISSING"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--buy", default="2022-12-30")
    ap.add_argument("--nq-stage1", default=8, type=int)
    ap.add_argument("--nq-stage2", default=16, type=int)
    ap.add_argument("--bs-from", default=None,
                    help="reuse the Stage-1 BoScore from an existing dump instead of "
                         "computing it.  This is what makes a CONFOUND-FREE code contrast "
                         "possible: run THIS tree's Stage-2 metric loop on ANOTHER tree's "
                         "panel and Stage-1 output, so the two arms differ only in the "
                         "Stage-2 metric definitions.  Required when the panel's BoMetric "
                         "schema and this tree's Stage-1 criteria disagree (a pre-arc tree "
                         "cannot score a rebuilt BoMetric: it looks for the retired "
                         "`uIncomeQuality` and would fail that criterion for everyone).")
    args = ap.parse_args()

    p = pd.read_pickle(args.panel)
    dmdic = {"cdx_df": p["cdx_df"], "BoMetric_df": p["BoMetric_df"],
             "Tickers_df": p.get("Tickers_df")}

    bm_pit, cdx_pit = s2.prepare_pit(dmdic, args.buy, na1_only=False)
    if args.bs_from:
        _src = pd.read_pickle(args.bs_from)
        bs = _src["bs"].copy()
        bs = bs[bs["source"].isin(set(cdx_pit["source"]))].reset_index(drop=True)
        stage1_from = {"stage1_borrowed_from": args.bs_from,
                       "stage1_source_label": _src["info"].get("label"),
                       "stage1_n_borrowed": len(_src["bs"]), "stage1_n_kept": len(bs)}
    else:
        if "cdx_pit" in inspect.signature(s2.stage1_boscore).parameters:
            bs = s2.stage1_boscore(bm_pit, nq_stage1=args.nq_stage1, cdx_pit=cdx_pit)
        else:
            bs = s2.stage1_boscore(bm_pit, nq_stage1=args.nq_stage1)
        stage1_from = {"stage1_borrowed_from": None}
    bs = bs.sort_values("score", ascending=False).reset_index(drop=True)

    cdxtop = cdx_pit[cdx_pit["source"].isin(bs["source"])].reset_index(drop=True)
    psm = s2._stage2_metric_loop_offline(bs, cdxtop, nq=args.nq_stage2)

    # rows-per-source as of the anchor: the scorability floor the cell definition uses,
    # carried in the dump so downstream never has to re-open the panel
    d = pd.to_datetime(dmdic["cdx_df"]["date"], errors="coerce")
    rows_asof = (dmdic["cdx_df"].loc[d <= pd.Timestamp(args.buy)]
                 .groupby("source").size().rename("cdx_rows_asof"))

    info = {"label": args.label, "panel": args.panel, "buy": args.buy,
            "nq_stage1": args.nq_stage1, "nq_stage2": args.nq_stage2,
            "n_stage1": len(bs), "n_psm_rows": len(psm),
            "psm_columns": list(psm.columns),
            "dropped_metrics": list(getattr(s2, "DROP_METRICS", [])),
            **stage1_from, **_tree_stamp()}
    pd.to_pickle({"psm": psm, "bs": bs, "rows_asof": rows_asof, "info": info}, args.out)

    print("=" * 84)
    print("dump_psm: %s" % args.label)
    for k, v in info.items():
        if k == "psm_columns":
            print("  %-24s %d cols: %s" % (k, len(v), v))
        else:
            print("  %-24s %s" % (k, v))
    print("  wrote %s (%.1f MB)" % (args.out, os.path.getsize(args.out) / 1e6))
    print("=" * 84, flush=True)


if __name__ == "__main__":
    main()
