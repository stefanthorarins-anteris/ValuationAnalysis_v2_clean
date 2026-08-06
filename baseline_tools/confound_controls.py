"""IS THE FLIP INFORMATION, OR A SIZE/COVERAGE FACTOR BET?  Plus the three tests the
round-2 gate required.  (Round-3 answer to the devils-advocate gate, 2026-07-27.)

THE QUESTION THAT BLOCKS EVERYTHING ELSE
----------------------------------------
Replacing the pre-arc un-winsorized z-score with EITHER the winsorized z OR the rank map
swings the universe-wide rank-IC by ~+0.18 on frozen metrics and frozen nominal weights.
The first reading was "tail control restored the ordering", i.e. a METHOD property.  The
competing reading is that `n_missing` (how many weighted metrics a name is missing) is a
VENDOR-COVERAGE / SIZE / LIQUIDITY proxy, and that the arms which flip are the arms whose
score became strongly ANTI-correlated with it -- so the score acquired a small-cap/
well-covered FACTOR BET that happened to pay over 2021-25, not information.

Everything observed is consistent with the factor reading:
  * rho(score, n_missing) is ~0 in the arms that do NOT flip (+0.004..+0.016) and
    -0.29..-0.57 in every arm that DOES;
  * IC(n_missing, excess) is negative at all three anchors (-0.184 / -0.202 / -0.137), so
    incompleteness itself predicts the outcome;
  * 42-66% of the flipped arms' IC disappears once n_missing is partialled out;
  * every decile's median excess is about -50pp against a developed-large-cap benchmark on a
    ~72%-non-US micro-cap panel -- the signature of a size/geography tilt, not of stock
    selection.
So: PARTIAL THE SCORE'S IC ON MARKET-CAP DECILE AS WELL AS n_missing.  If the residual
collapses, the honest headline is "the normalizer connected the score to a size/coverage
factor and that factor paid over this window" -- a regime bet whose sign is not guaranteed
to persist -- NOT "the normalizer made the score informative".

Market cap comes from `stage2_metrics._mcap_for_quants`, which is the SAME field the score's
own `marketCapRevQuants` is cut over.  Using the score's own size variable is deliberate: it
makes the control as tight as it can be, and it cannot be accused of controlling for a
different notion of size than the one the score reacts to.

THAT FIELD IS NOW USD-OR-NOTHING (register D-5, CEO 2026-08-06).  It used to read "USD where
derivable, coarse exchange-suffix fallback otherwise"; the fallback is gone, because absolute
bands cannot absorb a currency guess.  On any panel saved BEFORE the fetch that captures
`reportedCurrency` it therefore returns ALL-NaN, and this tool is **BLOCKED** rather than
degraded: `mcap_by_source` RAISES with instructions instead of running a size control that
controls for nothing.  This is a WAIT, not an abandonment -- one post-fetch panel unblocks it
and the guard is then deleted.  See `_MCAP_BLOCKED`.

ALSO HERE
---------
  2. PAIRED bootstrap of the IC DIFFERENCE.  The marginal CIs reported so far cannot settle
     "inside noise": B0-vs-A4 are two orderings of the SAME names correlated at ~0.99, so the
     SE of their DIFFERENCE is far smaller than either marginal SE.  Resample names ONCE,
     recompute both ICs on that resample, take the difference.  Required because the house
     doctrine is that a small persistent edge is never "nothing" -- so "inside noise" has to
     be measured, not asserted.
  3. DEPLOYED TOP-N read.  Arm A4 borrows the current Stage-1 output by design, which
     isolates the Stage-2 definition effect but therefore EXCLUDES the metric fixes' main
     channel: they moved three of the heaviest Stage-1 criteria by 16-22pp, i.e. WHICH 100
     NAMES REACH STAGE-2.  This module reproduces the deployed shape -- Stage-1 top-100,
     Stage-2 re-normalised OVER THAT POOL (not the universe), top-20 by AggScore -- at every
     anchor.  Issuer-dedup is NOT applied (flagged, not silent): it only collapses
     same-issuer lines, and reproducing it needs the panel this module does not load.

REPORTING RULE (adopted after two rounds in which the decisive evidence was in a column the
script computed and the summary omitted): every column computed here is printed, and the
printer ASSERTS that the printed set equals the computed set.  Omission must be a code
change, not an oversight.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import createDicts as cdic
import postBoRank as pbr
import stage2_metrics as sm
import decile_test as dt
import attribution_arms as aa


# --------------------------------------------------------------------------- #
#  Controls                                                                   #
# --------------------------------------------------------------------------- #
#  The message the BLOCKED state prints.  Module-level so the wording is stated once and a
#  post-fetch reader gets the same text wherever they hit it.
_MCAP_BLOCKED = """
{bar}
!!! confound_controls IS BLOCKED -- WAITING FOR DATA, NOT ABANDONED !!!
{bar}
This tool's whole purpose is to PARTIAL THE SCORE'S IC ON MARKET-CAP DECILE, so it needs a
real market cap per name.  It cannot get one from this panel:

    panel : {panel}
    carries `reportedCurrency` : {has_rc}
    carries `marketCap_usd`    : {has_usd}
    usable USD market cap      : NO

WHY, and why this is a WAIT.  `marketCap` alone is NOT a market cap on a comparable scale --
it is stored in each company's REPORTING currency, mixed across the universe, so a SEK
reporter reads ~10x an equally sized USD reporter and a KRW reporter far more.  Deciling
that column would decile companies partly by which currency they report in, and the
resulting "size control" would absorb currency, not size -- which would make this tool's
headline (how much of the IC is size) WRONG IN AN UNKNOWABLE DIRECTION rather than merely
noisy.  Register D-5 (CEO 2026-08-06) removed the coarse exchange-suffix FX guess that used
to paper over this, from `stage2_metrics._mcap_for_quants` and from the $25M universe floor
alike, because absolute edges do not cancel a systematic currency error.  So the field this
tool reads is now honestly EMPTY instead of dishonestly full.

>>> THE FIX IS AVAILABLE AS SOON AS A POST-FETCH PANEL EXISTS.  The next full fetch captures
>>> `reportedCurrency` (the ingest already materializes `marketCap_usd` alongside it -- the
>>> 2026-08-04 TEST1 panel carries both).  Point --panel-current / --panel-prearc at a panel
>>> from that run and DELETE THIS GUARD: `_mcap_for_quants` will return real USD and the
>>> decile control becomes correct, with no other change needed to this tool.
>>>
>>> DO NOT "fix" this by re-enabling a suffix-FX guess or by deciling the raw column.  That
>>> is the exact defect D-5 removed, and here it would silently corrupt the finding.
{bar}
""".strip()


def _usd_mcap_available(cdx):
    """Does this panel carry a market cap on a COMPARABLE (USD) scale?  Delegates to the
    pipeline's own predicate so the tool and the pipeline cannot disagree about it."""
    try:
        import carveOut as _co
        return bool(_co.currency_data_present(cdx))
    except Exception:
        #  carveOut unavailable: fall back to the same two-column test it applies, rather
        #  than assuming either answer.
        cols = getattr(cdx, "columns", [])
        if "marketCap_usd" in cols:
            return bool(pd.to_numeric(cdx["marketCap_usd"], errors="coerce").notna().any())
        return False


def mcap_by_source(panel_path, buy):
    """source -> USD market cap as of `buy`, via the SAME field mcapQuants is cut over.

    RAISES (loudly, with instructions) when the panel carries no USD market cap -- see
    `_MCAP_BLOCKED`.  A silent all-NaN return is the failure mode this guard exists to
    prevent: every name would land in the `missing-cap = own level` decile dummy, the
    "market-cap control" would control for NOTHING, and `IC_partial_mcap` would come back
    equal to the unpartialled IC -- a result that LOOKS like "size explains none of the IC"
    and is in fact "no size variable was present".  That is the strongest possible wrong
    conclusion this tool can emit, so it must fail instead of returning it.
    """
    cdx = pd.read_pickle(panel_path)["cdx_df"].copy()
    cdx["date"] = pd.to_datetime(cdx["date"], errors="coerce")
    cdx = cdx[cdx["date"] <= pd.Timestamp(buy)].copy()
    if not _usd_mcap_available(cdx):
        cols = getattr(cdx, "columns", [])
        msg = _MCAP_BLOCKED.format(
            bar="!" * 78, panel=panel_path,
            has_rc="yes" if "reportedCurrency" in cols else "NO",
            has_usd="yes" if "marketCap_usd" in cols else "NO")
        print(msg, file=sys.stderr, flush=True)
        print(msg, flush=True)
        raise RuntimeError(
            "confound_controls: BLOCKED pending a post-fetch panel carrying "
            "reportedCurrency -- no USD market cap on %s (register D-5). See the banner "
            "above; do NOT substitute the raw mixed-currency marketCap column." % panel_path)
    cdx["_mc"] = pd.to_numeric(pd.Series(sm._mcap_for_quants(cdx)).values, errors="coerce")
    # newest non-NaN row per source (cdx is ascending, so .last())
    s = cdx.dropna(subset=["_mc"]).groupby("source")["_mc"].last()
    if s.empty:
        raise RuntimeError(
            "confound_controls: panel %s reports a usable currency but yielded no finite "
            "USD market cap as of %s -- refusing to run an empty size control."
            % (panel_path, buy))
    return s


def _resid_on(y_rank, X):
    """OLS residual of a ranked vector on a design matrix (intercept added)."""
    A = np.column_stack([np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(A, y_rank, rcond=None)
    return y_rank - A @ beta


def _design(cell, use_nmissing, use_mcap_decile, n_dec=10):
    """Control design matrix.  n_missing enters as its RANK; market cap enters as DECILE
    DUMMIES (not a rank), so a non-monotone size effect is absorbed too -- a linear-in-rank
    size control would leave a hump-shaped size effect in the residual and understate how
    much of the IC is size."""
    cols = []
    names = []
    if use_nmissing:
        cols.append(pd.Series(cell["n_missing"].to_numpy(dtype=float)).rank().to_numpy())
        names.append("rank(n_missing)")
    if use_mcap_decile:
        mc = pd.to_numeric(cell["mcap"], errors="coerce")
        # a name with NO market cap is its own category, never silently pooled with a real one
        d = pd.Series(np.where(mc.isna(), -1,
                               pd.qcut(mc.rank(method="first"), n_dec,
                                       labels=False, duplicates="drop")), index=cell.index)
        for lvl in sorted(d.unique())[1:]:            # drop one level as the reference
            cols.append((d == lvl).to_numpy(dtype=float))
        names.append("mcap decile dummies (%d levels, missing-cap = own level)"
                     % d.nunique())
    if not cols:
        return np.zeros((len(cell), 0)), names
    return np.column_stack(cols), names


def partial_ic(cell, score_col="AggScore", use_nmissing=False, use_mcap_decile=False):
    c = cell.dropna(subset=[score_col, "excess"]).copy()
    s = pd.Series(c[score_col].to_numpy(dtype=float)).rank().to_numpy()
    e = pd.Series(c["excess"].to_numpy(dtype=float)).rank().to_numpy()
    X, _ = _design(c, use_nmissing, use_mcap_decile)
    if X.shape[1] == 0:
        return float(np.corrcoef(s, e)[0, 1]), len(c)
    rs, re_ = _resid_on(s, X), _resid_on(e, X)
    return float(np.corrcoef(rs, re_)[0, 1]), len(c)


def partial_ic_boot(cell, score_col="AggScore", use_nmissing=False, use_mcap_decile=False,
                    n_boot=1000, seed=0):
    c = cell.dropna(subset=[score_col, "excess"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(c), len(c))
        out[i] = partial_ic(c.iloc[idx].reset_index(drop=True), score_col,
                            use_nmissing, use_mcap_decile)[0]
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


# --------------------------------------------------------------------------- #
#  Paired bootstrap of an IC DIFFERENCE                                       #
# --------------------------------------------------------------------------- #
def paired_ic_diff(cell_a, cell_b, label_a, label_b, n_boot=4000, seed=0):
    """CI for IC(b) - IC(a) on the SAME names, resampling names ONCE per draw.

    This is the test "inside noise" actually requires.  Two scores of the same universe are
    ~0.99 correlated, so their ICs move together under resampling and the difference is far
    better determined than either level.
    """
    a = cell_a[["source", "AggScore", "excess"]].rename(columns={"AggScore": "sa"})
    b = cell_b[["source", "AggScore"]].rename(columns={"AggScore": "sb"})
    j = a.merge(b, on="source").dropna()
    sa = j["sa"].to_numpy(float); sb = j["sb"].to_numpy(float); y = j["excess"].to_numpy(float)
    def ic(x, yy):
        return float(pd.Series(x).rank().corr(pd.Series(yy).rank()))
    point = ic(sb, y) - ic(sa, y)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, len(j), len(j))
        d[i] = ic(sb[k], y[k]) - ic(sa[k], y[k])
    lo, hi = float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))
    return {"contrast": "%s MINUS %s" % (label_b, label_a), "n_paired": len(j),
            "spearman_between_scores": float(pd.Series(sa).rank().corr(pd.Series(sb).rank())),
            "d_IC": point, "d_lo95": lo, "d_hi95": hi,
            "excludes_zero": bool(lo > 0 or hi < 0),
            "boot_frac_le_zero": float(np.mean(d <= 0))}


# --------------------------------------------------------------------------- #
#  The DEPLOYED two-stage read                                                #
# --------------------------------------------------------------------------- #
def deployed_topn(psm, bs, normalizer, price_source, buy, eval_, n_pool=100,
                  tops=(5, 10, 20)):
    """Stage-1 top-`n_pool` -> Stage-2 re-normalised OVER THAT POOL -> top-N by AggScore.

    This is the shape the pipeline actually ships, and it is NOT what a universe-wide decile
    test measures: the cross-sectional NORMALISATION runs over ~100 names, so a name's z --
    and therefore its AggScore -- depends on the pool it is scored in.  That reason stands and
    is why this function exists.

    ONE reason that USED to be given here has EXPIRED: `marketCapRevQuants` was a within-pool
    quantile, i.e. a different variable at pool scale than at universe scale.  Register D-5
    (CEO 2026-08-06) made it an ABSOLUTE USD band, which depends on the row's own market cap
    and on nothing else, so that metric is now pool-INVARIANT.  Corrected rather than deleted
    so a reader of a pre-D-5 artifact knows which claim moved.  Issuer-dedup is deliberately
    NOT applied here (see module docstring).
    """
    pool = list(bs.sort_values("score", ascending=False)["source"])[:n_pool]
    sub = psm[psm["source"].isin(set(pool))].reset_index(drop=True)
    normed, _dropped, _ej = normalizer(sub)
    postBm, postNew = cdic.getPostDict()
    ws = {**{k: postBm[k]["w"] for k in postBm}, **{k: postNew[k]["w"] for k in postNew}}
    w = normed.drop("source", axis=1)
    for c in w.columns:
        w[c] = normed[c].values * ws[c]
    frame = pd.concat([normed[normed.columns.difference(w.columns)], w], axis=1)
    ranked = pbr.getAggScore(frame)[["source", "AggScore"]]
    cell, bench = dt.measurement_cell(ranked, price_source, buy, eval_, verbose=False)
    cell = cell.sort_values("AggScore", ascending=False)
    rows = []
    for n in tops:
        g = cell.head(n)
        k = int((g["excess"] >= 0.10).sum())
        p, lo, hi = dt._wilson(k, len(g))
        rows.append({"top_n": n, "n_priced": len(g), "n_beat_10pp": k, "beat10_rate": p,
                     "beat10_lo95": lo, "beat10_hi95": hi,
                     "median_excess": float(g["excess"].median()),
                     "mean_excess": float(g["excess"].mean()),
                     "share_beat": float((g["excess"] > 0).mean())})
    rows.append({"top_n": n_pool, "n_priced": len(cell.head(n_pool)),
                 "n_beat_10pp": int((cell.head(n_pool)["excess"] >= 0.10).sum()),
                 "beat10_rate": float((cell.head(n_pool)["excess"] >= 0.10).mean()),
                 "beat10_lo95": np.nan, "beat10_hi95": np.nan,
                 "median_excess": float(cell.head(n_pool)["excess"].median()),
                 "mean_excess": float(cell.head(n_pool)["excess"].mean()),
                 "share_beat": float((cell.head(n_pool)["excess"] > 0).mean())})
    return pd.DataFrame(rows), len(cell)


# --------------------------------------------------------------------------- #
#  Printing that cannot silently omit a column                                #
# --------------------------------------------------------------------------- #
def print_all(df, title, float_fmt="%+.4f"):
    print("\n--- %s ---" % title)
    printed = list(df.columns)
    assert set(printed) == set(df.columns), "a computed column is not being printed"
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df.to_string(index=False, float_format=lambda v: float_fmt % v))
    print("    [columns printed: %d of %d computed]" % (len(printed), len(df.columns)))


ANCHORS = [("2020-12-31", "2023-12-29"), ("2021-12-31", "2024-12-31"),
           ("2022-12-30", "2025-12-31")]

ARM_NORMALIZERS = {
    "A7 pre-arc frame, un-winsorized z (NO eject)": aa._prearc_normalize_noeject,
    "A6 pre-arc frame, WINSORIZED z": aa._current_zscore,
    "A2 pre-arc frame, RANK": aa._rank_normalize,
    "B0 current frame, WINSORIZED z": aa._current_zscore,
    "C0 current frame, RANK": aa._rank_normalize,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-prearc", required=True)
    ap.add_argument("--panel-current", required=True)
    ap.add_argument("--min-cdx-rows", default=8, type=int)
    ap.add_argument("--out-prefix", default=os.path.join(_HERE, "confound"))
    args = ap.parse_args()

    ps = dt.build_price_source()
    bar = "=" * 150
    print(bar)
    print("CONFOUND CONTROLS -- is the normalizer flip INFORMATION or a SIZE/COVERAGE "
          "FACTOR BET?   [ALL FIGURES ON THE 07-17 UNIVERSE]")
    print(bar, flush=True)

    ctrl_rows, topn_rows, paired_rows = [], [], []
    for buy, eval_ in ANCHORS:
        pa = os.path.join(_HERE, "psm_PREARC_%s.pickle" % buy)
        pc = os.path.join(_HERE, "psm_CURRENT_%s.pickle" % buy)
        if not (os.path.exists(pa) and os.path.exists(pc)):
            print("  SKIP %s (dump missing)" % buy)
            continue
        A, B = pd.read_pickle(pa), pd.read_pickle(pc)
        mc_a = mcap_by_source(args.panel_prearc, buy)
        mc_b = mcap_by_source(args.panel_current, buy)

        cells = {}
        for arm, norm in ARM_NORMALIZERS.items():
            D = A if arm.startswith(("A2", "A6", "A7")) else B
            mc = mc_a if arm.startswith(("A2", "A6", "A7")) else mc_b
            scores, info = aa.score(D["psm"], norm)
            cell, bench = dt.measurement_cell(scores, ps, buy, eval_, verbose=False)
            cell = cell.merge(aa.n_missing_per_name(D["psm"]), on="source", how="left")
            cell["mcap"] = cell["source"].map(mc)
            ra = D["rows_asof"]
            cell = cell[cell["source"].isin(set(ra[ra >= args.min_cdx_rows].index))].copy()
            cells[arm] = cell

            raw, n = partial_ic(cell)
            p_nm, _ = partial_ic(cell, use_nmissing=True)
            p_mc, _ = partial_ic(cell, use_mcap_decile=True)
            p_both, _ = partial_ic(cell, use_nmissing=True, use_mcap_decile=True)
            lo, hi = partial_ic_boot(cell, use_nmissing=True, use_mcap_decile=True)
            dtab = dt.decile_table(cell, "AggScore")
            ctrl_rows.append({
                "buy": buy, "arm": arm, "n": n,
                "IC_raw": raw,
                "IC_partial_nmissing": p_nm,
                "IC_partial_mcap": p_mc,
                "IC_partial_BOTH": p_both,
                "BOTH_lo95": lo, "BOTH_hi95": hi,
                "frac_IC_surviving_BOTH": (p_both / raw) if raw else np.nan,
                "rho_score_nmissing": aa._ic(cell["AggScore"], cell["n_missing"]),
                "rho_score_mcap": aa._ic(cell["AggScore"], cell["mcap"].fillna(-1)),
                "IC_nmissing_excess": aa._ic(cell["n_missing"], cell["excess"]),
                "IC_mcap_excess": aa._ic(cell["mcap"].fillna(-1), cell["excess"]),
                "n_mcap_missing": int(cell["mcap"].isna().sum()),
                "d1_minus_d10_median": dtab["median_excess"].iloc[0] - dtab["median_excess"].iloc[-1],
                "beat_spread": dtab["share_beat"].iloc[0] - dtab["share_beat"].iloc[-1],
                "beat10_spread": dtab["share_beat_10pp"].iloc[0] - dtab["share_beat_10pp"].iloc[-1],
            })

            tn, n_cell = deployed_topn(D["psm"], D["bs"], norm, ps, buy, eval_)
            for _, r in tn.iterrows():
                topn_rows.append({"buy": buy, "arm": arm, "pool_cell_n": n_cell, **r.to_dict()})

        # paired contrasts, same names, resampled once per draw
        for a_arm, b_arm in (("A7 pre-arc frame, un-winsorized z (NO eject)",
                              "A6 pre-arc frame, WINSORIZED z"),
                             ("A6 pre-arc frame, WINSORIZED z", "B0 current frame, WINSORIZED z"),
                             ("A2 pre-arc frame, RANK", "C0 current frame, RANK"),
                             ("B0 current frame, WINSORIZED z", "C0 current frame, RANK")):
            paired_rows.append({"buy": buy,
                                **paired_ic_diff(cells[a_arm], cells[b_arm], a_arm, b_arm)})

    ctrl = pd.DataFrame(ctrl_rows)
    print_all(ctrl, "HEADLINE: partial rank-IC under n_missing and MARKET-CAP-DECILE controls")
    print("\n  READ: `IC_partial_BOTH` is what survives once completeness AND size are held "
          "fixed.\n  `frac_IC_surviving_BOTH` < ~0.5 means most of the arm's apparent skill "
          "is the size/coverage factor.")

    pr = pd.DataFrame(paired_rows)
    print_all(pr, "PAIRED bootstrap of IC DIFFERENCES (same names, resampled once per draw)",
              float_fmt="%+.5f")

    tn = pd.DataFrame(topn_rows)
    print_all(tn, "DEPLOYED shape: Stage-1 top-100 -> Stage-2 re-normalised over that pool "
                  "-> top-N (no issuer-dedup)")

    ctrl.to_csv(args.out_prefix + "_partial_ic.csv", index=False)
    pr.to_csv(args.out_prefix + "_paired_diff.csv", index=False)
    tn.to_csv(args.out_prefix + "_deployed_topn.csv", index=False)
    print("\nwrote %s_{partial_ic,paired_diff,deployed_topn}.csv" % args.out_prefix)
    print(bar, flush=True)


if __name__ == "__main__":
    main()
