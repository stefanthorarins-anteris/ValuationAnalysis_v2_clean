"""
Re-run the model-vs-metric IC on REAL adjusted-close forward returns (offline).

We test REAL 24-month forward-return IC across even-year buy anchors (2018/2020/2022),
plus a 48-month check (2018->2022, 2020->2024), and compare to the RECONSTRUCTED-price IC
over the identical buy dates + horizon.

ANCHORS ARE RESOLVED FROM THE PRICE FILE, NOT HARDCODED (2026-08-22). The four dates one
2026-07 fetch happened to land on used to be literals here; the grid moved and the stage died
with KeyError: '2020-12-28' while its IC half printed a verdict computed entirely from NaN.
See ANCHOR_TOLERANCE_DAYS and verdict_line.

AND THE MATRIX IS KEYED ON `date_requested` (the ANCHOR), never on `date_actual` (the venue's
settlement day). Resolving against the wrong axis is a second, quieter way to manufacture a
verdict -- it yields a finite IC on a venue-biased slice, which no finiteness guard can catch.
See load_real, which is the one place that decides this.

Purpose: does "composite < best single metric" (the model-problem smoking gun)
HOLD on real returns, and how much do IC magnitudes move vs the proxy?

Caveats kept: 24/36mo horizons differ (fetch throttling forced even-year dates);
metric 'date' are quarter-starts (the pipeline lookahead), applied identically to
real & reconstructed so the COMPARISON is apples-to-apples.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_vs_metric as mvm

REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_data", "real_prices.csv")

#  ============================ ANCHORS ARE RESOLVED, NEVER HARDCODED ======================
#  WHAT BROKE (08-20 and 08-22 runs, both universes).  This module used to hardcode the price
#  grid's column labels -- "2020-12-28", "2022-12-27", "2024-12-28" -- which were the trading
#  days ONE PARTICULAR 2026-07 fetch happened to land on.  The grid it now reads is labelled
#  ['2018-12-31','2019-12-30','2020-12-30','2021-12-30','2022-12-29','2023-12-28','2024-12-30'],
#  so "2020-12-28" IS NOT A COLUMN AT ALL and `real.at[s, D]` raised KeyError: '2020-12-28'.
#
#  AND THE WORSE HALF, which is why this is not a cosmetic constant update.  `ic_table` guarded
#  the same lookups with `if buy in real.columns else None`, so the IC half of the stage did NOT
#  raise -- it computed an ALL-NaN table and then PRINTED A VERDICT off NaN comparisons:
#      "COMPOSITE IC_real=+nan vs best single (RoA) IC_real=+nan -> smoking gun DOES NOT hold"
#  A NaN < NaN comparison is False, so the stage RELIABLY rendered "DOES NOT hold" while
#  measuring nothing whatsoever. Every IC_real figure from either run is void. A stage that
#  cannot compute its statistic must say so, not render a conclusion -- see `verdict_line`.
#
#  RESOLVING IS ONLY HALF OF IT -- THE AXIS HAS TO BE RIGHT FIRST (2026-08-22, reviewer C1).
#  The first version of this fix resolved anchors correctly against the WRONG COLUMN SPACE:
#  `load_real` pivoted on `date_actual`, so one fetch anchor appears as SEVERAL columns (one
#  per venue calendar) and the resolver picked the exact-dated MINORITY fragment.  A resolver
#  cannot detect that -- both fragments are real columns with plausible dates -- and the
#  resulting IC is finite, so `verdict_line` cannot refuse it either.  The axis is now
#  `date_requested`, matching `returns_core.PriceSource`; see `load_real`.
#
#  WHY A TOLERANCE WINDOW AND NOT "NEAREST ON-OR-BEFORE".  On-or-before is the intuitive rule
#  and it is WRONG HERE: the nearest column on-or-before 2020-12-28 in the grid above is
#  2019-12-30 -- a different year, 364 days away -- so the resolver would silently substitute
#  the wrong anchor and every IC would be computed over a 12-month-shifted window. The anchors
#  are CALENDAR YEAR-ENDS, and a year-end moves only by weekends and holidays, so the true
#  displacement is at most a handful of days while the distance to the adjacent year's column is
#  ~358+. Those are two orders of magnitude apart, so any cut between them separates them
#  cleanly; 7 days sits there with headroom and cannot reach into another year.
ANCHOR_TOLERANCE_DAYS = 7

#  The anchors this diagnostic is ABOUT (even-year year-ends), stated as intent. What the fetch
#  actually landed on is a property of the FILE and is resolved from it at run time.
INTENDED_IC_ANCHORS_24M = ["2018-12-31", "2020-12-31", "2022-12-31", "2024-12-31"]
INTENDED_PROFIT_TIMING_ANCHORS = ["2020-12-31", "2022-12-31", "2024-12-31"]


def resolve_anchor(columns, intended, tolerance_days=ANCHOR_TOLERANCE_DAYS):
    """The price-grid column that REPRESENTS `intended`, or None if the grid has none.

    Nearest by absolute day distance within `tolerance_days`; an exact tie resolves to the
    EARLIER date, so the choice is deterministic and never depends on column order. Returns
    the column LABEL as it appears in the frame (callers index with it).
    """
    want = pd.Timestamp(intended)
    best, best_gap = None, None
    for col in columns:
        try:
            when = pd.Timestamp(col)
        except (ValueError, TypeError):
            continue          # a non-date column is not an anchor candidate
        if pd.isna(when):
            continue
        gap = abs((when - want).days)
        if gap > tolerance_days:
            continue
        if best_gap is None or gap < best_gap or (gap == best_gap and when < pd.Timestamp(best)):
            best, best_gap = col, gap
    return best


def resolve_anchors(columns, intended, tolerance_days=ANCHOR_TOLERANCE_DAYS, what="anchor"):
    """{intended -> actual column} for every `intended`, RAISING on any that cannot resolve.

    Raising is the point. The previous code's silent `else None` is what let an all-NaN table
    reach a printed verdict, so an unresolvable anchor has to stop the stage and name itself.
    """
    got, missing = {}, []
    for d in intended:
        col = resolve_anchor(columns, d, tolerance_days=tolerance_days)
        if col is None:
            missing.append(d)
        else:
            got[d] = col
    if missing:
        raise RuntimeError(
            "real_ic: cannot resolve %d %s date(s) %s against the price grid within %d day(s). "
            "The grid carries %s. NOTHING is reported rather than reporting a statistic "
            "computed from missing columns -- the price grid does not cover the window this "
            "diagnostic is defined on. FIX: refresh the price grid, or state anchors the grid "
            "actually covers."
            % (len(missing), what, missing, tolerance_days, list(columns)))
    return got


def verdict_line(comp_ic, best_metric, best_ic, horizon_label="24m"):
    """The smoking-gun verdict, or a REFUSAL if either side is not a finite number.

    THIS IS THE GUARD THAT WAS MISSING. `nan < nan` is False, so the old formatting produced a
    confident "DOES NOT hold" out of two NaNs on both the 08-20 and 08-22 runs. A comparison
    between non-numbers is not a finding in either direction.
    """
    if not np.isfinite(comp_ic) or not np.isfinite(best_ic):
        raise RuntimeError(
            "real_ic: REFUSING to state a %s smoking-gun verdict -- the IC table is not "
            "numeric (COMPOSITE IC_real=%r, best single %s IC_real=%r). A comparison "
            "between non-numbers is False, so printing it would render a definite negative "
            "verdict out of a measurement that never happened. Usual causes: the price grid "
            "does not overlap "
            "the scored universe, or fewer than 50 names are jointly priced and scored at "
            "every anchor pair." % (horizon_label, comp_ic, best_metric, best_ic))
    return ("  COMPOSITE IC_real=%+.3f vs best single (%s) IC_real=%+.3f  -> smoking gun %s"
            % (comp_ic, best_metric, best_ic,
               "HOLDS" if comp_ic < best_ic else "DOES NOT hold"))


def load_real(path=None, anchors=None):
    """symbol x ANCHOR matrix of adjClose, keyed on `date_requested`.

    ==========================================================================================
    THE AXIS IS `date_requested`, NOT `date_actual`.  THIS IS THE WHOLE FUNCTION.
    ==========================================================================================
    `date_requested` is the ANCHOR the fetch asked for; `date_actual` is the trading day each
    venue happened to settle on.  One anchor therefore fragments into SEVERAL `date_actual`
    columns -- one per venue calendar -- and pivoting on `date_actual` turns a 7-anchor grid
    into a 10-column matrix of partial cross-sections.

    MEASURED on baseline_tools/price_data/real_prices.csv (443,893 rows):
        date_requested 2020-12-31 -> date_actual 2020-12-28 (58,838 rows) + 2020-12-31 (9,901)
        date_requested 2022-12-30 -> date_actual 2022-12-27 (64,490 rows) + 2022-12-30 (15,441)
        date_requested 2024-12-31 -> date_actual 2024-12-28  (3,589 rows) + 2024-12-31 (62,189)

    WHY THIS WAS WORSE THAN THE BUG IT REPLACED, and it is worth stating plainly.  The
    2026-08-22 fix resolved anchors against `real.columns` -- but against the `date_actual`
    axis, so the resolver saw BOTH fragments of an anchor, preferred the exact intended
    calendar date, and thereby selected the MINORITY fragment (9,901 rows over 58,838).
    Joint-priced symbols per 24m pair collapsed 77-82% on the first two pairs.  And the
    selection is SYSTEMATIC, not random: `URTH` is absent from both newly-chosen fragments and
    present in both majority ones.  Because a 15% venue-biased subsample still yields a FINITE
    IC, `verdict_line` could not refuse it -- so the stage printed a confident smoking-gun
    verdict off it, under a reassuring "anchor pairs (intended -> resolved)" line.  Exactly the
    manufactured-conclusion class the fix was for, one level down and harder to see.

    IT NOW AGREES WITH THE GRADER BY CONSTRUCTION.  `returns_core.PriceSource` reads
    `usecols=["date_requested","symbol","adjClose"]` and never looks at `date_actual`
    (returns_core.py:70-76), keying its LUT on (symbol, date_requested) and keeping the FIRST
    occurrence.  Both rules are reproduced here -- the anchor filter and keep-first -- so a
    symbol the depth-grid can price is a symbol this matrix can price.  Anything else means
    the IC diagnostic and the return grid are describing different populations.

    `anchors` defaults to returns_core.DEFAULT_ANCHORS (imported, never restated) so the
    column space is the grader's anchor space and cannot drift from it.
    """
    import returns_core as rc
    df = pd.read_csv(path or REAL, usecols=["date_requested", "symbol", "adjClose"])
    df["adjClose"] = pd.to_numeric(df["adjClose"], errors="coerce")
    df["date_requested"] = df["date_requested"].astype(str)
    keep = list(rc.DEFAULT_ANCHORS if anchors is None else anchors)
    df = df[df["date_requested"].isin(keep) & df["symbol"].notna()]
    #  keep-FIRST, matching PriceSource; `pivot` (not `pivot_table`) then cannot silently
    #  aggregate two rows into one, so a duplicate key is a hard error rather than a mean.
    df = df.drop_duplicates(subset=["symbol", "date_requested"], keep="first")
    piv = df.pivot(index="symbol", columns="date_requested", values="adjClose")
    return piv


def build_pairs_24m(columns):
    """The three consecutive-even-year (buy, eval) pairs, as ACTUAL grid columns.

    One place builds them, so `run_in_pipeline` and `main` cannot drift apart -- they carried
    two independent copies of the same stale literals, and both were wrong.
    """
    r = resolve_anchors(columns, INTENDED_IC_ANCHORS_24M, what="24m IC anchor")
    a = [r[d] for d in INTENDED_IC_ANCHORS_24M]
    return [(a[0], a[1]), (a[1], a[2]), (a[2], a[3])]


def build_pairs_48m(columns):
    """The two 48-month (buy, eval) pairs: 2018->2022 and 2020->2024, as ACTUAL columns."""
    r = resolve_anchors(columns, INTENDED_IC_ANCHORS_24M, what="48m IC anchor")
    a = [r[d] for d in INTENDED_IC_ANCHORS_24M]
    return [(a[0], a[2]), (a[1], a[3])]


def metric_xs(panel, asof):
    return panel[panel["date"] <= pd.Timestamp(asof)].groupby("source").tail(1).set_index("source")


def ic_table(panel, real, buy_eval_pairs, horizon_label):
    """Mean Spearman IC per metric (+COMPOSITE) over the given (buy,eval) pairs,
    computed for REAL and RECONSTRUCTED returns side by side."""
    out = {m: {"real": [], "recon": []} for m in mvm.METRICS + ["COMPOSITE"]}
    for buy, ev in buy_eval_pairs:
        xs = metric_xs(panel, buy)
        # reconstructed fwd return from panel _price (nearest <= dates)
        p_buy = metric_xs(panel, buy)["_price"]
        p_ev = metric_xs(panel, ev)["_price"]
        recon_ret = (p_ev / p_buy - 1).replace([np.inf, -np.inf], np.nan)
        # real fwd return.  STRICT: the caller resolves anchors against the grid BEFORE
        # calling, so a missing column here is a defect, not a data condition.  The old
        # `if buy in real.columns else None` swallowed exactly that and produced the all-NaN
        # table that got a verdict printed off it.
        for _d in (buy, ev):
            if _d not in real.columns:
                raise KeyError(
                    "real_ic.ic_table: %r is not a column of the real price grid (it carries "
                    "%s). Resolve anchors with resolve_anchors() before calling; a silently "
                    "skipped anchor yields an all-NaN IC table that reads like a result."
                    % (_d, list(real.columns)))
        rb = real[buy]
        re = real[ev]
        real_ret = (re / rb - 1).replace([np.inf, -np.inf], np.nan)

        # composite = cross-sectional weighted z-sum
        z = pd.DataFrame(index=xs.index)
        for mm in mvm.METRICS:
            col = pd.to_numeric(xs[mm], errors="coerce")
            z[mm] = (col - col.mean()) / col.std() * mvm.WEIGHTS[mm]
        comp = z.sum(axis=1, min_count=1)

        for m in mvm.METRICS + ["COMPOSITE"]:
            val = comp if m == "COMPOSITE" else pd.to_numeric(xs[m], errors="coerce")
            for tag, ret in (("real", real_ret), ("recon", recon_ret)):
                if ret is None:
                    continue
                j = pd.concat([val.rename("v"), ret.rename("r")], axis=1).dropna()
                if len(j) >= 50:
                    ic, _ = spearmanr(j["v"], j["r"])
                    out[m][tag].append(ic)
    rows = []
    for m in mvm.METRICS + ["COMPOSITE"]:
        rows.append({
            "metric": m,
            "IC_real": np.nanmean(out[m]["real"]) if out[m]["real"] else np.nan,
            "IC_recon": np.nanmean(out[m]["recon"]) if out[m]["recon"] else np.nan,
            "n": len(out[m]["real"]),
        })
    return pd.DataFrame(rows), horizon_label


#  Fewest priceable names in a LEG before its median is printed as a median.
#  WHY 5, and why this constant exists at all.  The 2026-08-31 and 2026-08-29 runs both
#  printed `EXITERS (n=6)` against `STAYERS (n=1)` and then a comparison of the two
#  medians -- "after[+2y,+4y]" -15.8% for the stayers -- which is one company's three-year
#  price move typeset as a finding about churn.  A median over one name is that name; over
#  two it is their mean; the block's entire reading is a COMPARISON of two such numbers, so
#  the weaker leg sets the strength of the claim.  Five is the point at which a median is at
#  least the middle of a distribution rather than a re-labelled observation -- deliberately a
#  low bar, because the aim is to stop the block asserting a direction it cannot support, not
#  to suppress the data.  THE NUMBERS ARE STILL PRINTED either way; what is withheld below
#  the bar is the DIRECTIONAL COMPARISON, and the reason is stated on the line.
PROFIT_TIMING_MIN_LEG_N = 5


def profit_timing_lines(ex, st, exdf, stdf, min_leg_n=PROFIT_TIMING_MIN_LEG_N):
    """The profit-timing block as LINES, with its own n governing what it may claim.

    ONE DEFINITION, TWO CALLERS.  `run_in_pipeline` and `main` each carried their own copy of
    this print loop and the copies were byte-identical, which is how they would have drifted:
    a qualification added to the pipeline's copy would have left the hand-run one asserting
    the old thing.  The n-guard is exactly the kind of change that gets applied to one of two
    copies.

    WHAT IT STILL CANNOT SEE: nothing here knows whether a name is missing from `real` because
    it delisted or because the grid never priced its venue, so a small leg is not evidence
    about churn either.  The line says the count and refuses the direction; it does not
    explain the count."""
    L = ["top-20 at D=2020: exit by +2y=%d, stay=%d" % (len(ex), len(st))]
    legs = [("EXITERS", exdf), ("STAYERS", stdf)]
    thin = [lbl for lbl, d in legs if d is None or len(d) < min_leg_n]
    for lbl, d in legs:
        if d is None or d.empty:
            L.append("  %s: no priceable" % lbl)
            continue
        L.append("  %s (n=%d): during[D,+2y]=%+.1f%%  after[+2y,+4y]=%+.1f%%  full=%+.1f%%"
                 % (lbl, len(d), d["during"].median() * 100, d["after"].median() * 100,
                    d["full"].median() * 100))
        if len(d) < min_leg_n:
            #  The individual observations, so nothing is hidden -- the point is to stop the
            #  word "median" doing work it cannot do, not to withhold the data.
            L.append("     ^ n=%d < %d: this is %s, not a median.  Per name: %s"
                     % (len(d), min_leg_n,
                        "one observation" if len(d) == 1 else "%d observations" % len(d),
                        ", ".join("%+.1f%%" % (v * 100) for v in d["full"])))
    if thin:
        L.append("  NO DIRECTIONAL CONCLUSION IS DRAWN: %s below n=%d, so the EXITERS-vs-"
                 "STAYERS comparison" % (" and ".join(thin), min_leg_n))
        L.append("  is NOT reported this run.  Read the rows as observations, not as a "
                 "result about churn.")
    return L


def profit_timing_real(dmdic, panel, real):
    import io, contextlib
    import stage2_pit as s2
    #  RESOLVED from the grid in hand, not hardcoded.  These three literals were
    #  "2020-12-28", "2022-12-27", "2024-12-28" and none of them is a column of the current
    #  grid, which is the KeyError: '2020-12-28' that killed this stage on both runs.
    _res = resolve_anchors(real.columns, INTENDED_PROFIT_TIMING_ANCHORS,
                           what="profit-timing anchor")
    D, D1, D3 = (_res[INTENDED_PROFIT_TIMING_ANCHORS[0]],
                 _res[INTENDED_PROFIT_TIMING_ANCHORS[1]],
                 _res[INTENDED_PROFIT_TIMING_ANCHORS[2]])
    print("[real_ic] profit-timing anchors resolved: %s"
          % ", ".join("%s->%s" % (k, _res[k]) for k in INTENDED_PROFIT_TIMING_ANCHORS),
          flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        bm0, cdx0 = s2.prepare_pit(dmdic, D, na1_only=False)
        top0 = s2.stage2_top(s2.stage1_boscore(bm0, cdx_pit=cdx0), cdx0)
        bm1, cdx1 = s2.prepare_pit(dmdic, D1, na1_only=False)
        top1 = s2.stage2_top(s2.stage1_boscore(bm1, cdx_pit=cdx1), cdx1)
    ex = [s for s in top0 if s not in set(top1)]
    st = [s for s in top0 if s in set(top1)]

    def _t(names):
        r = []
        for s in names:
            if s not in real.index:
                continue
            p0, p1, p3 = real.at[s, D], real.at[s, D1], real.at[s, D3]
            if any(pd.isna(x) or x <= 0 for x in (p0, p1, p3)):
                continue
            r.append({"during": p1 / p0 - 1, "after": p3 / p1 - 1, "full": p3 / p0 - 1})
        return pd.DataFrame(r)
    return ex, st, _t(ex), _t(st)


def run_in_pipeline(dmdic, price_source=None, real_prices_csv=None, log=None):
    """IN-MEMORY entry point for the automatic pipeline (post-pick analysis suite).

    Re-runs the IC (real vs reconstructed price) + profit-timing decomposition on
    TONIGHT's fresh scoring dict `dmdic`, NOT the hardcoded stale PICKLE that main()
    loads.  Real adjusted-close prices come from real_prices.csv (the same file the
    guarded price-fetch step ensures is present); dmdic supplies the metric panel.

    `real_prices_csv` overrides the default REAL path (machine-independence); if the
    real-price file is missing the caller's guard banners -- this raises rather than
    silently producing an empty table.  price_source is accepted for parity (unused;
    real_ic reads the year-anchor CSV directly).  Never prints any api_key.
    """
    log = log or (lambda *a: None)
    cdx = dmdic.get("cdx_df")
    if cdx is None:
        raise ValueError("dmdic has no 'cdx_df' -- cannot build the metric panel")
    log("[real_ic] building panel + loading real prices (in-memory) ...")
    panel = mvm.build_panel(cdx)
    real = load_real(real_prices_csv)
    ov = len(set(panel['source'].unique()) & set(real.index))
    print("\n" + "#" * 72)
    print("# REAL-IC diagnostic  (tonight's model, real adjusted-close returns)")
    print("#" * 72)
    print(f"real price matrix: {real.shape[0]} symbols x {list(real.columns)}", flush=True)
    print(f"symbols overlapping pipeline universe: {ov}", flush=True)

    print("\n" + "=" * 72)
    print("IC on REAL vs RECONSTRUCTED returns  (24-month horizon; buy 2018/2020/2022)")
    print("=" * 72)
    pairs24 = build_pairs_24m(real.columns)
    print("  anchor pairs (intended -> resolved): %s" % (pairs24,), flush=True)
    tbl, _ = ic_table(panel, real, pairs24, "24m")
    tbl = tbl.sort_values("IC_real", key=lambda s: s.abs(), ascending=False)
    print(tbl.to_string(index=False, formatters={"IC_real": "{:+.3f}".format,
                                                  "IC_recon": "{:+.3f}".format}))
    singles = tbl[tbl["metric"] != "COMPOSITE"]
    comp = tbl[tbl["metric"] == "COMPOSITE"].iloc[0]
    best_real = singles.iloc[0]
    print("\n" + verdict_line(comp["IC_real"], best_real["metric"], best_real["IC_real"],
                              "24m"), flush=True)

    print("\n" + "=" * 72)
    print("PROFIT-TIMING vs CHURN on REAL prices (D=2020-12 -> +2y -> +4y)")
    print("=" * 72)
    ex, st, exdf, stdf = profit_timing_real(dmdic, panel, real)
    for _line in profit_timing_lines(ex, st, exdf, stdf):
        print(_line)
    print("\n[real_ic] DONE.", flush=True)
    return {"ic_24m": tbl, "exiters": exdf, "stayers": stdf}


def main():
    print("Loading pickle + building panel (offline) ...", flush=True)
    dmdic = pd.read_pickle(mvm.PICKLE)
    panel = mvm.build_panel(dmdic["cdx_df"])
    real = load_real()
    print(f"real price matrix: {real.shape[0]} symbols x {list(real.columns)}", flush=True)
    ov = len(set(panel['source'].unique()) & set(real.index))
    print(f"symbols overlapping pipeline universe: {ov}", flush=True)

    print("\n" + "=" * 72)
    print("IC on REAL vs RECONSTRUCTED returns  (24-month horizon; buy 2018/2020/2022)")
    print("=" * 72)
    pairs24 = build_pairs_24m(real.columns)
    print("  anchor pairs (intended -> resolved): %s" % (pairs24,), flush=True)
    tbl, _ = ic_table(panel, real, pairs24, "24m")
    tbl = tbl.sort_values("IC_real", key=lambda s: s.abs(), ascending=False)
    print(tbl.to_string(index=False, formatters={"IC_real": "{:+.3f}".format,
                                                  "IC_recon": "{:+.3f}".format}))
    singles = tbl[tbl["metric"] != "COMPOSITE"]
    comp = tbl[tbl["metric"] == "COMPOSITE"].iloc[0]
    best_real = singles.iloc[0]
    print("\n" + verdict_line(comp["IC_real"], best_real["metric"], best_real["IC_real"],
                              "24m"), flush=True)

    print("\n  (48-month check: buy 2018->2022, 2020->2024)")
    pairs48 = build_pairs_48m(real.columns)
    tbl48, _ = ic_table(panel, real, pairs48, "48m")
    tbl48 = tbl48.sort_values("IC_real", key=lambda s: s.abs(), ascending=False)
    c48 = tbl48[tbl48.metric == "COMPOSITE"].iloc[0]["IC_real"]
    b48 = tbl48[tbl48.metric != "COMPOSITE"].iloc[0]
    print("  48m: " + verdict_line(c48, b48["metric"], b48["IC_real"], "48m"), flush=True)

    print("\n" + "=" * 72)
    print("PROFIT-TIMING vs CHURN on REAL prices (D=2020-12 -> +2y -> +4y)")
    print("=" * 72)
    ex, st, exdf, stdf = profit_timing_real(dmdic, panel, real)
    for _line in profit_timing_lines(ex, st, exdf, stdf):
        print(_line)


if __name__ == "__main__":
    main()
