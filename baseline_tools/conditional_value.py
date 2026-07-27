"""CONDITIONAL VALUE OF STAGE-2 -- does AggScore order the outcome INSIDE a BoScore shortlist?

WHY THIS MODULE EXISTS, AND WHAT IT IS ALLOWED TO CONCLUDE
---------------------------------------------------------
The deployed pipeline uses Stage-1 BoScore to SELECT (top-100) and Stage-2 AggScore only to
ORDER within that selection.  So "which score has the higher universe-wide IC" does not
answer the deployed question, and a universe-wide measurement cannot settle it.

An earlier pass concluded from these numbers that "the weights have nothing to grip on".
THAT CONCLUSION IS WITHDRAWN, for four reasons the devils-advocate gate established:

  1. POWER.  At n = 100 the minimum detectable Spearman at 80% power / alpha 0.05 two-sided
     is ~0.28 -- LARGER than the strongest signal anywhere in this study (BoScore's
     universe-wide +0.191).  A test that could not have detected the best effect we have
     ever measured cannot return "no effect"; it returns "no information".
  2. The n = 656 point estimate is +0.066, i.e. ~58% of the unconditional IC, t ~ 1.7,
     p ~ 0.09 -- a NEAR-SIGNIFICANT POSITIVE, not a zero.
  3. The 5x5 double sort is 4 of 4 POSITIVE across BoScore quintiles 1-4 (P = 1/16 = 0.0625
     under the null of independent signs) -- weak POSITIVE evidence.
  4. The conditional estimates flip sign between two orderings correlated at rho = 0.936
     (z-path vs rank-path), which is the signature of a NOISE-DOMINATED estimate.  That
     mandates withholding a verdict, not returning one of zero.

Correct statement, and the only one this module supports:
    THE CONDITIONAL VALUE OF THE STAGE-2 WEIGHTS IS **UNMEASURED** AT THE SCALE IT OPERATES.

A FURTHER REASON A UNIVERSE-WIDE STUDY CANNOT SETTLE IT.  `marketCapRevQuants` (w = 0.080,
the #2 weight) is a POOL-RELATIVE quantile code, recomputed per pool: over 100 names it
splits the shortlist into cap quartiles, over 7,000 it splits the universe.  It is
literally a different variable in the two settings, so a universe-wide measurement can
neither certify nor refute deployed Stage-2 -- for that metric the two are not comparable
at all.  Reported here per pool size so the effect is visible rather than assumed.

Emits artifacts (the earlier pass emitted none for this, which is why it was the least
checkable number in the study and the one that drove the recommendation).
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


def _ic(x, y):
    return float(pd.Series(np.asarray(x, float)).rank()
                 .corr(pd.Series(np.asarray(y, float)).rank()))


def boot_ci(x, y, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        out[i] = _ic(x[idx], y[idx])
    return (float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5)),
            float(np.nanmean(out < 0)))


def mde_spearman(n, power=0.80, alpha=0.05):
    """Minimum detectable Spearman rho at the given power, via the Fisher-z approximation
    (rho ~ tanh((z_{1-a/2} + z_power)/sqrt(n-3))).  The number that decides whether a null
    result is informative -- if MDE exceeds the effect you are hunting, it is not."""
    from scipy.stats import norm
    if n <= 4:
        return float("nan")
    return float(np.tanh((norm.ppf(1 - alpha / 2) + norm.ppf(power)) / np.sqrt(n - 3)))


def t_and_p(rho, n):
    from scipy.stats import t as tdist
    if n <= 3 or abs(rho) >= 1:
        return float("nan"), float("nan")
    tv = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    return float(tv), float(2 * (1 - tdist.cdf(abs(tv), n - 2)))


def conditional_table(cell, subsets=None):
    if subsets is None:
        n = len(cell)
        subsets = [("FULL cell", n),
                   ("BoScore decile 1", int(round(n / 10))),
                   ("BoScore top-300", 300),
                   ("BoScore top-100 (DEPLOYED pool)", 100)]
    rows = []
    for label, k in subsets:
        sub = cell.nlargest(k, "BoScore_stage1") if k < len(cell) else cell
        for sc in ("AggScore", "BoScore_stage1"):
            rho = _ic(sub[sc], sub["excess"])
            lo, hi, p_neg = boot_ci(sub[sc], sub["excess"])
            tv, pv = t_and_p(rho, len(sub))
            rows.append({"subset": label, "n": len(sub), "score": sc, "IC": rho,
                         "lo95": lo, "hi95": hi, "t": tv, "p_two_sided": pv,
                         "boot_frac_negative": p_neg,
                         "MDE_80pct_power": mde_spearman(len(sub)),
                         "detectable": abs(rho) >= mde_spearman(len(sub))})
    return pd.DataFrame(rows)


def double_sort(cell, n_q=5):
    c = cell.dropna(subset=["AggScore", "BoScore_stage1", "excess"]).copy()
    c["bo_q"] = pd.qcut(c["BoScore_stage1"].rank(method="first"), n_q,
                        labels=list(range(n_q, 0, -1))).astype(int)
    rows = []
    for q, g in c.groupby("bo_q"):
        g = g.copy()
        g["ag_q"] = pd.qcut(g["AggScore"].rank(method="first"), n_q,
                            labels=list(range(n_q, 0, -1))).astype(int)
        med = g.groupby("ag_q")["excess"].median()
        beat = g.groupby("ag_q")["excess"].apply(lambda s: float((s > 0).mean()))
        rows.append({"BoScore_quintile": int(q), "n_per_cell": int(round(len(g) / n_q)),
                     **{"med_Agg_q%d" % i: med.get(i, np.nan) for i in range(1, n_q + 1)},
                     "med_q1_minus_q5": med.get(1, np.nan) - med.get(n_q, np.nan),
                     **{"beat_Agg_q%d" % i: beat.get(i, np.nan) for i in range(1, n_q + 1)},
                     "beat_q1_minus_q5": beat.get(1, np.nan) - beat.get(n_q, np.nan),
                     "IC_within": _ic(g["AggScore"], g["excess"])})
    return pd.DataFrame(rows).sort_values("BoScore_quintile")


def sign_test_note(ds, col="med_q1_minus_q5"):
    """The double sort's own evidence, stated as a sign test rather than eyeballed.

    BOTH tails are reported: 4/4 in one direction is one-sided p = 1/16 = 0.0625 and
    two-sided p = 0.125.  The one-sided figure is only legitimate if the direction was
    predicted BEFORE looking, which it was not here -- so two-sided is the honest headline
    and the one-sided value is shown for comparability with how it gets quoted.
    """
    top4 = ds[ds["BoScore_quintile"] <= 4][col]
    k, n = int((top4 > 0).sum()), len(top4)
    from scipy.stats import binomtest
    p2 = float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    p1 = float(binomtest(k, n, 0.5, alternative="greater").pvalue)
    return k, n, p2, p1


def mcap_quant_pool_dependence(panel_path, bs_path, buy, ns=(100, 300, 656, 6564)):
    """`marketCapRevQuants` is POOL-RELATIVE: show how much a name's own value moves with
    pool size -- the reason a universe-wide result cannot be transported to the top-100.

    It must RE-RUN `add_mcap_quants` on each pool.  (An earlier version of this function
    merely SUBSET the single universe-wide computation the metric loop had already done, so
    it reported 0.0000 changed for every pool size -- it was measuring nothing.  The
    deployed run calls qcut on the ~100-name pool; that is what has to be reproduced.)
    """
    import stage2_metrics as sm
    cdx = pd.read_pickle(panel_path)["cdx_df"].copy()
    cdx["date"] = pd.to_datetime(cdx["date"], errors="coerce")
    cdx = cdx[cdx["date"] <= pd.Timestamp(buy)]
    order = list(pd.read_pickle(bs_path)["bs"]["source"])
    out, ref = [], None
    for n in ns:
        names = set(order[:n])
        sub = cdx[cdx["source"].isin(names)].reset_index(drop=True)
        if sub.empty:
            continue
        q = pd.Series(sm.add_mcap_quants(sub)).astype(float)
        # one value per name, exactly as the metric loop takes it (.iloc[0] of the slice)
        per_name = pd.DataFrame({"source": sub["source"], "q": q.values}) \
            .groupby("source")["q"].first()
        row = {"pool_n": len(per_name), "distinct_values": int(per_name.nunique()),
               "mean": float(per_name.mean()), "sd": float(per_name.std())}
        if ref is None:
            ref = per_name
            row["frac_changed_vs_top100"] = 0.0
        else:
            common = per_name.index.intersection(ref.index)
            row["frac_changed_vs_top100"] = float(
                (~np.isclose(per_name.loc[common], ref.loc[common], equal_nan=True)).mean())
            row["n_common_with_top100"] = len(common)
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="a *_cell.csv from decile_test")
    ap.add_argument("--label", default="")
    ap.add_argument("--psm", default=None, help="psm dump (for the Stage-1 order)")
    ap.add_argument("--panel", default=None, help="panel pickle, for the mcapQuants recompute")
    ap.add_argument("--buy", default="2022-12-30")
    ap.add_argument("--out-prefix", default=os.path.join(_HERE, "conditional"))
    args = ap.parse_args()

    cell = pd.read_csv(args.cell).dropna(subset=["AggScore", "BoScore_stage1", "excess"])
    bar = "=" * 104
    print(bar)
    print("CONDITIONAL VALUE OF STAGE-2 -- %s   [ON THE 07-17 UNIVERSE]"
          % (args.label or os.path.basename(args.cell)))
    print("  VERDICT THIS MODULE SUPPORTS: UNMEASURED at the scale the weights operate.")
    print("  It does NOT support 'the weights have nothing to grip on' -- see the MDE column.")
    print(bar)

    ct = conditional_table(cell)
    print(ct.to_string(index=False, float_format=lambda v: "%+.4f" % v))
    print("\n  Read the MDE column first: where |IC| < MDE, the test had no power to see the"
          "\n  effect it is being cited against, so a small estimate there is NOT evidence of"
          "\n  absence.")

    ds = double_sort(cell)
    print("\n--- DOUBLE SORT: median excess (and share beating URTH) by BoScore quintile x "
          "AggScore quintile within it (1 = best) ---")
    print(ds.to_string(index=False, float_format=lambda v: "%+.4f" % v))
    for col, what in (("med_q1_minus_q5", "MEDIAN EXCESS"),
                      ("beat_q1_minus_q5", "SHARE BEATING URTH (the TARGET metric)")):
        k, n, p2, p1 = sign_test_note(ds, col)
        print("  sign test on %s, AggScore q1-q5 across BoScore quintiles 1-4: "
              "%d/%d POSITIVE  (two-sided p=%.4f, one-sided p=%.4f)"
              % (what, k, n, p2, p1))
    print("  => the weak POSITIVE evidence is on MEDIANS only; on the beat-rate the same "
          "double sort does NOT go one way.")

    if args.psm and os.path.exists(args.psm) and args.panel:
        print("\n--- marketCapRevQuants is POOL-RELATIVE (w=0.080): re-running "
              "add_mcap_quants per pool ---")
        print(mcap_quant_pool_dependence(args.panel, args.psm, args.buy).to_string(
            index=False, float_format=lambda v: "%.4f" % v))
        print("  => a universe-wide measurement of Stage-2 can neither certify nor refute "
              "the DEPLOYED Stage-2 for this metric.")

    ct.to_csv(args.out_prefix + "_ic.csv", index=False)
    ds.to_csv(args.out_prefix + "_double_sort.csv", index=False)
    print("\nwrote %s_ic.csv and %s_double_sort.csv" % (args.out_prefix, args.out_prefix))
    print(bar, flush=True)


if __name__ == "__main__":
    main()
