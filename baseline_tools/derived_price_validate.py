"""
DERIVED-LEG VALIDATION HARNESS  --  the side-by-side audit of the derived total-return
price source against the real-price route (offline, no network).

This is the artifact that makes the derived leg's fidelity AUDITABLE rather than asserted.
It answers four questions, each against the real `adjClose` leg on the overlap where both
routes have a price:

  1. FIDELITY      -- Spearman(derived 36m TR, real 36m TR) per anchor pair, price-only
                      vs total-return, so the dividend leg's contribution is visible.
  2. LEVEL BIAS    -- median log gap.  Price-only runs -0.035..-0.044 (the missing yield);
                      the TR leg must close it to ~0.
  3. ANCHOR RULE   -- Dec-31 cutoff vs the naive price-file grid-date cutoff, which pushes
                      December filers back a full quarter at the 12-30 / 12-29 anchors.
  4. COVERAGE      -- what the derived route buys: names per anchor, derived vs real.

Plus `ic_residual_table`, the convention bake-off (additive vs multiplicative reinvestment,
clip vs reject) that selected what `derived_prices` ships.  Its headline is in that module's
docstring; it lives here because it needs the real leg as ground truth.

Run:  python baseline_tools/derived_price_validate.py            (from the repo root)
      python baseline_tools/derived_price_validate.py --csv out.csv

`test_derived_prices.py` imports `find_panel` / `fidelity_table` / `lag_comparison` for its
acceptance tests, and SKIPS when the panel or the price CSVs are absent.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import derived_prices as dpx                                    # noqa: E402
import returns_core as rc                                       # noqa: E402

#  One definition of the panel location, in derived_prices, so the harness and the
#  shipped source can never point at different panels.
PANEL_GLOB = dpx.DEFAULT_PANEL_GLOB
PRICES = os.path.join(_HERE, "price_data", "real_prices.csv")
PRICES_2025 = os.path.join(_HERE, "price_data", "real_prices_2025.csv")

#  36-month windows on the standard anchor grid.  Note the eval labels are the REAL
#  trading dates the price file uses (12-30 / 12-29); the derived leg maps each to its
#  calendar year-end internally, which is the whole point of `_anchor_cutoff`.
PAIRS = [("2018-12-31", "2021-12-31"),
         ("2019-12-31", "2022-12-30"),
         ("2020-12-31", "2023-12-29"),
         ("2021-12-31", "2024-12-31"),
         ("2022-12-30", "2025-12-31")]

#  Proxy value/quality signals for the IC bake-off, computed from RAW panel inputs rather
#  than consumed as vendor ratio fields.  Split deliberately:
#    MC_* have marketCap in the denominator -- they SHARE it with the derived price
#    NM_* contain no marketCap at all       -- they are the clean control
#  That split is what localises the residual (see derived_prices' docstring).
MC_SIGNALS = {"MC_ey": ("netIncome", "marketCap"),
              "MC_bp": ("totalStockholdersEquity", "marketCap"),
              "MC_fcfy": ("freeCashFlow", "marketCap")}
NM_SIGNALS = {"NM_roe": ("netIncome", "totalStockholdersEquity"),
              "NM_npm": ("netIncome", "revenue"),
              "NM_gm": ("grossProfit", "revenue"),
              "NM_lev": ("totalDebt", "totalAssets")}

_PANEL_CACHE = {}


def find_panel(pattern=PANEL_GLOB):
    """Newest date-stamped deep panel, or None when it is not on this machine."""
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def load_panel(path):
    """Cached cdx_df.  The deep panel is ~145MB, and the acceptance tests want it twice."""
    if path not in _PANEL_CACHE:
        _PANEL_CACHE[path] = dpx._load_cdx(path)
    return _PANEL_CACHE[path]


def _spearman(a, b):
    """Rank correlation without a scipy dependency in the hot path."""
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def real_source(anchors=None, prices_csv=None, supp_csv=None):
    """The real-price leg -- the ground truth every comparison here is against.

    `prices_csv` overrides the repo-local dev grid.  It exists because the two grids on this
    project answer DIFFERENT questions and both are needed: the dev grid has ~98.7% coverage
    and so gives the big OVERLAP the bias table needs, while the RUN MACHINE's grid is the
    one the pipeline actually reads and is where the seven wholly-unpriced venues -- the
    population the gap-fill route is for -- actually exist.  A bias table measured on one and
    a clear/refuse count measured on the other is not an inconsistency; it is the only way to
    get both numbers.
    """
    main = prices_csv or PRICES
    if supp_csv is None:
        cand = (os.path.join(os.path.dirname(main), "real_prices_2025.csv")
                if prices_csv else PRICES_2025)
        supp_csv = cand if os.path.exists(cand) else None
    return rc.PriceSource(main, anchors=anchors, supp_csv=supp_csv)


# --------------------------------------------------------------------------- #
#  1 + 2.  FIDELITY AND LEVEL BIAS                                            #
# --------------------------------------------------------------------------- #
def fidelity_table(panel, **derived_kw):
    """Per anchor pair: Spearman and median log gap, price-only vs total-return.

    Restricted to the OVERLAP where both routes price both legs -- that is the only set on
    which the two are comparable at all.  The coverage the derived leg adds OUTSIDE that
    overlap is `coverage_table`'s job, and is the reason the leg exists.
    """
    cdx = load_panel(panel)
    real = real_source()
    tr = dpx.DerivedPriceSource(cdx, benchmark_source=real, **derived_kw)
    #  price-only twin: same construction, dividend term disabled by an unreachable
    #  ceiling of 0 -> every period yield is rejected -> div_factor == 1.
    #  a ceiling of -1 rejects every period yield (y >= 0 always), so div_factor == 1 and
    #  the twin is a pure PRICE leg.  Pop any caller-supplied ceiling so it cannot collide.
    po_kw = {k: v for k, v in derived_kw.items() if k != "max_period_yield"}
    po = dpx.DerivedPriceSource(cdx, benchmark_source=real, max_period_yield=-1.0, **po_kw)
    rows = []
    for buy, ev in PAIRS:
        names = [t for (t, a) in tr._lut if a == buy]
        rec = []
        for t in names:
            d_b, d_e = tr.price(t, buy), tr.price(t, ev)
            p_b, p_e = po.price(t, buy), po.price(t, ev)
            r_b, r_e = real.price(t, buy), real.price(t, ev)
            if None in (d_b, d_e, p_b, p_e, r_b, r_e):
                continue
            if min(d_b, d_e, p_b, p_e, r_b, r_e) <= 0:
                continue
            rec.append((d_e / d_b, p_e / p_b, r_e / r_b))
        if len(rec) < 3:
            rows.append({"pair": f"{buy[:4]}->{ev[:4]}", "n": len(rec)})
            continue
        g_tr, g_po, g_re = (np.array([x[i] for x in rec]) for i in range(3))
        lg = np.log(g_tr) - np.log(g_re)
        rows.append({
            "pair": f"{buy[:4]}->{ev[:4]}", "n": len(rec),
            "spearman_price_only": round(_spearman(g_po, g_re), 4),
            "spearman_tr": round(_spearman(g_tr, g_re), 4),
            "median_log_gap_price_only": round(float(np.median(np.log(g_po) - np.log(g_re))), 4),
            "median_log_gap_tr": round(float(np.median(lg)), 4),
            #  THE MEAN IS THE ONE THAT MATTERS.  A median over a universe that is ~78%
            #  currency-matched reads 0.0000 by construction and cannot see a -0.12 mean on
            #  the mismatched fifth -- that blindness is what let the currency defect ship.
            "mean_log_gap_tr": round(float(np.mean(lg)), 4),
            "p05_log_gap_tr": round(float(np.percentile(lg, 5)), 4),
            "p95_log_gap_tr": round(float(np.percentile(lg, 95)), 4),
            "median_div_factor": round(float(np.median(g_tr / g_po)), 4),
        })
    return pd.DataFrame(rows)


def currency_split_table(panel, pairs=None, **derived_kw):
    """Per anchor pair, the derived-vs-real gap SPLIT BY reporting/listing currency match.

    This is the table the acceptance gate should have been built on.  With the listing-currency
    guard ON (the default) the mismatched population is empty by construction -- that is the
    fix working.  Pass `require_listing_currency_match=False` to see the defect it removes:
    measured n=456 of 2,057 (22.2%) at 2021->2024 with a mean log gap of -0.1234 against
    +0.0067 on the matched names.
    """
    cdx = load_panel(panel)
    real = real_source()
    tr = dpx.DerivedPriceSource(cdx, benchmark_source=real, **derived_kw)
    cur = (dpx.DerivedPriceSource._clean(cdx)
           .groupby("source", sort=False)["reportedCurrency"].last())
    rows = []
    for buy, ev in (pairs or PAIRS):
        acc = {True: [], False: []}
        for t in {t for (t, a) in tr._lut if a == buy}:
            d_b, d_e = tr.price(t, buy), tr.price(t, ev)
            r_b, r_e = real.price(t, buy), real.price(t, ev)
            if None in (d_b, d_e, r_b, r_e) or min(d_b, d_e, r_b, r_e) <= 0:
                continue
            matched = dpx._listing_currency(t) == cur.get(t)
            acc[matched].append(np.log((d_e / d_b) / (r_e / r_b)))
        out = {"pair": f"{buy[:4]}->{ev[:4]}"}
        for m, label in ((True, "matched"), (False, "mismatched")):
            v = np.asarray(acc[m], dtype=float)
            out[f"n_{label}"] = int(v.size)
            out[f"mean_gap_{label}"] = round(float(v.mean()), 4) if v.size else float("nan")
        tot = out["n_matched"] + out["n_mismatched"]
        out["pct_mismatched"] = round(100.0 * out["n_mismatched"] / tot, 1) if tot else 0.0
        rows.append(out)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  2d.  PER-VENUE x PER-REPORTING-CURRENCY BIAS  --  and what it cannot reach  #
# --------------------------------------------------------------------------- #
def _gap_stats(v):
    """mean / p05 / p95 FIRST.  The median is last and is reference only.

    A MEDIAN OVER A MOSTLY-MATCHED POPULATION IS 0.0000 BY CONSTRUCTION, and that exact
    vacuity is what let the -0.1234 currency defect ship on this module: the acceptance gate
    asserted a median, the median read 0.0000 throughout, and the mean was -0.0671.  So the
    order of these columns is deliberate and the median is not allowed to lead.
    """
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return {}
    return {"n": int(v.size),
            "mean": round(float(v.mean()), 4),
            "p05": round(float(np.percentile(v, 5)), 4),
            "p95": round(float(np.percentile(v, 95)), 4),
            "sd": round(float(v.std(ddof=1)), 4) if v.size > 1 else float("nan"),
            "median_ref_only": round(float(np.median(v)), 4)}


def venue_currency_bias_table(panel, pairs=None, prices_csv=None, min_n=3, **derived_kw):
    """Derived-vs-real log return gap per (VENUE, reportedCurrency), POOLED over the anchor
    pairs, on the OVERLAP where both legs price BOTH legs of a window.

    THE LIMIT IS STRUCTURAL AND IS THE POINT OF THE TABLE, NOT A FOOTNOTE.  A cell can only
    appear here if the REAL leg prices the name -- so the seven venues the real grid cannot
    price at any anchor (`.PA .KS .OL .KQ .BR .AS .LS`) are ABSENT BY CONSTRUCTION, and those
    are precisely the venues the gap-fill route uses the derived leg on.  Bias there is
    UNMEASURABLE, full stop.  `unmeasurable_venue_table` enumerates them and says whether any
    measurable venue at least shares their currency, which is the strongest thing that can be
    said about an extrapolation into them.

    With the listing-currency guard ON (the default, and what ships) every surviving cell has
    reportedCurrency == listing currency, so the table reads as a per-venue check on the
    RESIDUAL bias after the guard.  Pass require_listing_currency_match=False to see the
    population the guard removes.
    """
    cdx = load_panel(panel)
    real = real_source(prices_csv=prices_csv)
    tr = dpx.DerivedPriceSource(cdx, benchmark_source=real, **derived_kw)
    cur = (dpx.DerivedPriceSource._clean(cdx)
           .groupby("source", sort=False)["reportedCurrency"].last())
    acc = {}
    for buy, ev in (pairs or PAIRS):
        for t in {t for (t, a) in tr._lut if a == buy}:
            d_b, d_e = tr.price(t, buy), tr.price(t, ev)
            r_b, r_e = real.price(t, buy), real.price(t, ev)
            if None in (d_b, d_e, r_b, r_e) or min(d_b, d_e, r_b, r_e) <= 0:
                continue
            key = (dpx._venue_of(t), cur.get(t))
            acc.setdefault(key, []).append(np.log((d_e / d_b) / (r_e / r_b)))
    rows = []
    for (venue, ccy), vals in acc.items():
        st = _gap_stats(vals)
        if st.get("n", 0) < min_n:
            continue
        rows.append({"venue": venue, "reported_ccy": ccy,
                     "listing_ccy": dpx.venue_listing_currency(venue),
                     **st})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("n", ascending=False).reset_index(drop=True)


def unmeasurable_venue_table(panel, prices_csv=None, **derived_kw):
    """The venues where the derived leg WILL be used and its bias CANNOT be measured.

    For every venue the real grid cannot price at ANY anchor: how many panel names sit there,
    how many the derived leg can price, the listing currency, and -- the only part that makes
    an extrapolation arguable -- whether any MEASURABLE venue reports in that same currency.

    A same-currency measurable sibling is NOT proof the bias carries over.  It is the
    difference between "EUR-on-EUR behaves like this on `.DE`, so `.PA` plausibly too" and
    "nothing in KRW or NOK is measurable anywhere, so any number carried into Korea or Oslo
    is invented".  Both of those get said out loud rather than averaged into one figure.
    """
    cdx = load_panel(panel)
    real = real_source(prices_csv=prices_csv)
    tr = dpx.DerivedPriceSource(cdx, benchmark_source=real, **derived_kw)
    universe = sorted(set(cdx["source"].dropna().unique()))
    real_priced = {t for t in universe if any(real.price(t, a) is not None
                                              for a in real.anchors)}
    derived_priced = {t for (t, _a) in tr._lut}

    by_venue = {}
    for t in universe:
        by_venue.setdefault(dpx._venue_of(t), []).append(t)

    #  Currencies for which SOME measurable (= real-priced and derived-priced) name exists.
    measurable_ccys = {}
    cur = (dpx.DerivedPriceSource._clean(cdx)
           .groupby("source", sort=False)["reportedCurrency"].last())
    for t in real_priced & derived_priced:
        c = cur.get(t)
        if isinstance(c, str):
            measurable_ccys.setdefault(c, set()).add(dpx._venue_of(t))

    rows = []
    for venue, members in sorted(by_venue.items()):
        n_real = len(set(members) & real_priced)
        if n_real:
            continue                      # measurable somewhere -> not this table's business
        lc = dpx.venue_listing_currency(venue)
        sib = sorted(measurable_ccys.get(lc, set()))
        rows.append({"venue": venue, "n_panel": len(members),
                     "n_real_priced": 0,
                     "n_derived_priceable": len(set(members) & derived_priced),
                     "listing_ccy": lc,
                     "measurable_same_ccy_venues": ",".join(sib) if sib else "-- NONE --",
                     "bias": "UNMEASURABLE (no real price to compare against)"})
    return pd.DataFrame(rows).sort_values("n_panel", ascending=False).reset_index(drop=True)


def gapfill_assignment_tables(panel, prices_csv=None, **derived_kw):
    """(per_venue, refusals, diagnostics) for the 'real+derived' gap-fill route.

    This is the clear/refuse count, MEASURED, rather than an expectation about which venues
    ought to clear.
    """
    cdx = load_panel(panel)
    real = real_source(prices_csv=prices_csv)
    #  Built directly rather than through `build_price_source` so the already-loaded panel
    #  frame and the already-built real leg are reused; the wiring is identical (see
    #  build_price_source's 'real+derived' branch) and test_derived_gapfill pins that.
    src = dpx.GapFillPriceSource(
        real, dpx.DerivedPriceSource(cdx, benchmark_source=real, **derived_kw))
    universe = sorted(set(cdx["source"].dropna().unique()))
    return src.per_venue_counts(universe), src.refusal_reasons(universe), src.diagnostics()


#  The four configurations whose progression shows the revaluation channel closing.  Order
#  matters: each row ADDS one guard to the row above it.
GUARD_PROGRESSION = [
    ("1 no guards (as first shipped)",
     dict(require_listing_currency_match=False, reject_repeated_price=False,
          max_period_yield=0.25)),
    ("2 + currency-matched routing",
     dict(require_listing_currency_match=True, reject_repeated_price=False,
          max_period_yield=0.25)),
    ("3 + start-cap yield, ceiling 1.0",
     dict(require_listing_currency_match=True, reject_repeated_price=False,
          max_period_yield=1.0)),
    ("4 + backfill rejection (SHIPPED)",
     dict(require_listing_currency_match=True, reject_repeated_price=True,
          max_period_yield=1.0)),
]


def guard_progression_table(panel):
    """The revaluation channel, guard by guard: mean log gap on the names BOTH legs price.

    The mean is the statistic under test -- the old gate used a median and was blind to this
    entire defect.  `n` falls as guards restrict the derived leg; that is the cost side, and
    the composite hands every dropped name to the real leg rather than losing it.
    """
    rows = []
    for label, kw in GUARD_PROGRESSION:
        # start-cap is not switchable: pre-fix END-cap behaviour is emulated by the ceiling
        # only, so row 1/2 differ from the original ship only in the ceiling, which is the
        # part that was actually doing the damage.
        f = fidelity_table(panel, **kw)
        rows.append({"config": label, "n_total": int(f["n"].sum()),
                     "mean_log_gap_tr": round(float((f["mean_log_gap_tr"] * f["n"]).sum()
                                                    / f["n"].sum()), 4),
                     "median_log_gap_tr": round(float(f["median_log_gap_tr"].median()), 4),
                     "mean_spearman_tr": round(float(f["spearman_tr"].mean()), 4)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  3.  THE ANCHOR RULE                                                        #
# --------------------------------------------------------------------------- #
def lag_comparison(panel):
    """Dec-31 cutoff vs the naive grid-date cutoff, per anchor.

    The two rules AGREE at every anchor whose label is already a Dec-31, and diverge by a
    full quarter at 2022-12-30 / 2023-12-29 -- which is the entire defect.
    """
    cdx = load_panel(panel)
    df = dpx.DerivedPriceSource._clean(cdx)
    rows = []
    for a in rc.DEFAULT_ANCHORS:
        out = {"anchor": a}
        for label, cut in (("dec31", dpx.DerivedPriceSource._anchor_cutoff(a)),
                           ("griddate", pd.Timestamp(a))):
            pk = df[df["periodEndDate"] <= cut].groupby("source", sort=False).tail(1)
            lag = (cut - pk["periodEndDate"]).dt.days
            out[f"n_{label}"] = int(len(pk))
            out[f"median_lag_{label}"] = float(lag.median())
            out[f"mean_lag_{label}"] = round(float(lag.mean()), 1)
            out[f"p90_lag_{label}"] = float(lag.quantile(0.90))
            out[f"max_lag_{label}"] = float(lag.max())
            out[f"pct_over_45d_{label}"] = round(float(100.0 * (lag > 45).mean()), 2)
        rows.append(out)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  4.  COVERAGE -- what the derived route actually buys                       #
# --------------------------------------------------------------------------- #
def coverage_table(panel, **derived_kw):
    """Per anchor: names priced by each route, and the net gain from the derived leg.

    `n_derived_only` is the point of the whole exercise; `n_real_only` is the honest other
    direction (names the price file has and the panel does not).

    READ THE DENOMINATOR BEFORE QUOTING THE GAIN.  `n_universe` here is the PANEL's own
    source list, which is SURVIVORS-ONLY.  It is not the backtest universe: the grid scores
    the panel plus the delisted registry, and only 4 of that registry's 9,277 symbols appear
    in cdx_df while 3,948 are priced by real_prices.csv.  So the gain below is real *within
    the live universe* and is NOT a net gain for a universe-wide backtest -- on the actual
    grid the pure derived route LOST 1,298 included names and inflated the 36-month top-20
    return from +0.34 to +0.75 at buy2018 by dropping the dead half.  Use the
    'derived+real' composite route for anything universe-wide.
    """
    cdx = load_panel(panel)
    real = real_source()
    tr = dpx.DerivedPriceSource(cdx, benchmark_source=real, **derived_kw)
    universe = set(cdx["source"].dropna().unique())
    rows = []
    for a in rc.DEFAULT_ANCHORS:
        d = {t for (t, an) in tr._lut if an == a}
        r = {t for t in universe if real.price(t, a) is not None}
        rows.append({"anchor": a, "n_universe": len(universe),
                     "n_derived": len(d), "n_real": len(r),
                     "n_both": len(d & r), "n_derived_only": len(d - r),
                     "n_real_only": len(r - d), "net_gain": len(d) - len(r)})
    return pd.DataFrame(rows)


def deep_history_table(panel, anchors=("2008-12-31", "2012-12-31"), horizon_years=3,
                       **derived_kw):
    """Reach-back: sources with a FULL 36-month pair at pre-price-file anchors.

    The real-price file starts at 2018-12-31, so every one of these windows is
    unreachable by the real route at any fidelity -- this is coverage that does not
    otherwise exist.
    """
    cdx = load_panel(panel)
    rows = []
    for a in anchors:
        ev = f"{int(a[:4]) + horizon_years}-12-31"
        src = dpx.DerivedPriceSource(cdx, anchors=[a, ev], **derived_kw)
        b = {t for (t, an) in src._lut if an == a}
        e = {t for (t, an) in src._lut if an == ev}
        rows.append({"buy": a, "eval": ev, "n_buy": len(b), "n_eval": len(e),
                     "n_full_pair": len(b & e)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  5.  THE CONVENTION BAKE-OFF                                                #
# --------------------------------------------------------------------------- #
def _cum(df, y, multiplicative):
    g = df.assign(_y=y).groupby("source", sort=False)["_y"]
    return g.transform(lambda s: (1.0 + s).cumprod()) if multiplicative else g.transform("cumsum")


def ic_residual_table(panel, ceiling=dpx.MAX_PERIOD_YIELD):
    """Rank-IC residual (derived IC - real IC) per anchor-cell for the four conventions.

    This is the measurement that chose multiplicative+reject, and that localises the
    surviving residual to marketCap-denominated signals.  Returns the long table; the
    CLI prints the two summaries that matter.
    """
    cdx = load_panel(panel)
    sig_cols = sorted({c for pair in list(MC_SIGNALS.values()) + list(NM_SIGNALS.values())
                       for c in pair})
    df = dpx.DerivedPriceSource._clean(cdx, extra_cols=sig_cols)
    df, _ = dpx.DerivedPriceSource._guard_currency(df, True)
    for name, (num, den) in {**MC_SIGNALS, **NM_SIGNALS}.items():
        d = pd.to_numeric(df[den], errors="coerce")
        df[name] = pd.to_numeric(df[num], errors="coerce") / d.where(d > 0)

    dp = df["dividendsPaid"]
    y_raw = pd.Series(np.where(dp < 0, -dp.to_numpy(), 0.0) / df["marketCap"].to_numpy(),
                      index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    variants = {
        "price_only":     None,
        "add_clip":       _cum(df, y_raw.clip(upper=ceiling), False),
        "add_reject":     _cum(df, y_raw.where(y_raw <= ceiling, 0.0), False),
        "mult_clip":      _cum(df, y_raw.clip(upper=ceiling), True),
        "mult_reject":    _cum(df, y_raw.where(y_raw <= ceiling, 0.0), True),
    }
    for k, v in variants.items():
        if v is not None:
            df[f"F_{k}"] = v.to_numpy()

    real = real_source()
    picks = {}
    for a in {x for p in PAIRS for x in p}:
        cut = dpx.DerivedPriceSource._anchor_cutoff(a)
        picks[a] = (df[df["periodEndDate"] <= cut]
                    .groupby("source", sort=False).tail(1).set_index("source"))
    rows = []
    for buy, ev in PAIRS:
        pb, pe = picks[buy], picks[ev]
        idx = pb.index.intersection(pe.index)
        rb = np.array([real.price(t, buy) or np.nan for t in idx], dtype=float)
        re_ = np.array([real.price(t, ev) or np.nan for t in idx], dtype=float)
        Pb, Pe = pb.loc[idx, "price"].to_numpy(), pe.loc[idx, "price"].to_numpy()
        r_real = re_ / rb - 1.0
        rets = {}
        for k in variants:
            if k == "price_only":
                rets[k] = Pe / Pb - 1.0
            elif k.startswith("mult"):
                rets[k] = (Pe / Pb) * (pe.loc[idx, f"F_{k}"].to_numpy()
                                       / pb.loc[idx, f"F_{k}"].to_numpy()) - 1.0
            else:
                rets[k] = (Pe / Pb) + (pe.loc[idx, f"F_{k}"].to_numpy()
                                       - pb.loc[idx, f"F_{k}"].to_numpy()) - 1.0
        for sig in list(MC_SIGNALS) + list(NM_SIGNALS):
            sv = pb.loc[idx, sig].to_numpy(dtype=float)
            base = np.isfinite(sv) & np.isfinite(r_real)
            if base.sum() < 100:
                continue
            ic_real = _spearman(sv[base], r_real[base])
            for k in variants:
                m = base & np.isfinite(rets[k])
                rows.append({"pair": f"{buy[:4]}->{ev[:4]}", "signal": sig,
                             "kind": "MC" if sig in MC_SIGNALS else "NM",
                             "convention": k, "n": int(m.sum()),
                             "ic_real": round(ic_real, 4),
                             "residual": round(_spearman(sv[m], rets[k][m]) - ic_real, 4)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--panel", default=None, help="panel pickle (default: newest found)")
    ap.add_argument("--csv", default=None, help="write the fidelity table to this CSV")
    ap.add_argument("--skip-ic", action="store_true", help="skip the convention bake-off")
    ap.add_argument("--prices", default=None,
                    help="real-price grid to measure against (default: the repo-local dev "
                         "grid).  Point this at the RUN MACHINE's grid to get the "
                         "clear/refuse counts that the pipeline would actually see.")
    args = ap.parse_args(argv)

    panel = args.panel or find_panel()
    if panel is None or not os.path.exists(panel):
        print(f"FATAL: no deep panel found at {PANEL_GLOB}", file=sys.stderr)
        return 2
    if not os.path.exists(PRICES):
        print(f"FATAL: missing {PRICES}", file=sys.stderr)
        return 2
    prices = args.prices or PRICES
    if not os.path.exists(prices):
        print(f"FATAL: missing {prices}", file=sys.stderr)
        return 2
    print(f"panel : {panel}")
    print(f"prices: {prices}\n")

    src = dpx.DerivedPriceSource(load_panel(panel), benchmark_source=real_source())
    diag = src.diagnostics()
    print("=" * 78)
    print("DERIVED SOURCE DIAGNOSTICS")
    print("=" * 78)
    for k, v in diag.items():
        if k != "ic_caveat":
            print(f"  {k:32s} {v}")
    print(f"\n  !! {diag['ic_caveat']}\n")

    print("=" * 78)
    print("1+2. FIDELITY AND LEVEL BIAS vs the real adjClose leg (both-priced overlap)")
    print("=" * 78)
    fid = fidelity_table(panel)
    print(fid.to_string(index=False))
    print("\n  spearman_tr must beat spearman_price_only, and median_log_gap_tr ~ 0:")
    print("  the price-only gap IS the missing dividend yield over 36 months.\n")

    print("=" * 78)
    print("2b. CURRENCY MATCH -- the defect that drove 55% of the composite's error")
    print("=" * 78)
    print("  guard ON (shipped): the mismatched population is empty by construction")
    print(currency_split_table(panel).to_string(index=False))
    print("\n  guard OFF: what the derived leg looked like before the fix")
    print(currency_split_table(panel, require_listing_currency_match=False).to_string(index=False))
    print()

    print("=" * 78)
    print("2d. PER-VENUE x PER-REPORTING-CURRENCY BIAS on the OVERLAP (both legs price)")
    print("=" * 78)
    print("  mean / p05 / p95 lead; the median is reference-only, because a median over a")
    print("  mostly currency-matched population is 0.0000 BY CONSTRUCTION and that exact")
    print("  vacuity already let one defect ship on this module.")
    vb = venue_currency_bias_table(panel, prices_csv=prices)
    print(vb.to_string(index=False) if len(vb) else "  (no cell reached the minimum n)")
    if len(vb):
        allv = np.concatenate([np.full(int(r.n), r.mean) for r in vb.itertuples()])
        print(f"\n  n-weighted mean of the per-cell means: {allv.mean():+.4f}"
              f"   cells: {len(vb)}   names: {int(vb['n'].sum())}")
        worst = vb.reindex(vb["mean"].abs().sort_values(ascending=False).index).head(5)
        print("  largest |mean| cells:")
        print(worst.to_string(index=False))
    print()
    print("  !! THE LIMIT, STATED PLAINLY: a cell can only appear above if the REAL leg")
    print("     prices the name.  The venues the gap-fill route actually uses the derived")
    print("     leg on are the ones the real grid CANNOT price -- so they are absent here")
    print("     BY CONSTRUCTION and their bias is UNMEASURABLE.  Anything carried into them")
    print("     is an EXTRAPOLATION, and none is applied.")
    print()
    um = unmeasurable_venue_table(panel, prices_csv=prices)
    print("  venues where the derived leg WILL be used and bias CANNOT be measured:")
    print(um.to_string(index=False) if len(um) else "  (none on this grid)")
    print()

    print("=" * 78)
    print("2e. GAP-FILL ROUTE 'real+derived' -- MEASURED clear/refuse per venue")
    print("=" * 78)
    pv, refus, gdiag = gapfill_assignment_tables(panel, prices_csv=prices)
    print("  real = the real leg prices it (route inert);  derived_fill = real EMPTY and the")
    print("  derived leg cleared;  REFUSED = real EMPTY and the derived leg refuses.")
    print(pv.to_string(index=False))
    print("\n  refusals by (venue, reason) -- PUBLISHED, never substituted:")
    print(refus.to_string(index=False) if len(refus) else "  (none)")
    print(f"\n  gapfilled tickers: {gdiag['n_tickers_gapfilled']}   "
          f"real-priceable: {gdiag['n_tickers_real_priceable']}   "
          f"derived-priceable: {gdiag['n_tickers_derived_priceable']}")
    print(f"  {gdiag['bias_measurability']}")
    print()

    print("=" * 78)
    print("2c. GUARD PROGRESSION -- the revaluation channel closing, guard by guard")
    print("=" * 78)
    print(guard_progression_table(panel).to_string(index=False))
    print("\n  mean_log_gap_tr is the statistic; the old acceptance gate asserted a MEDIAN")
    print("  and was blind to all of it.\n")

    print("=" * 78)
    print("3. ANCHOR RULE -- Dec-31 cutoff vs the naive price-file grid-date cutoff")
    print("=" * 78)
    lc = lag_comparison(panel)
    print(lc[["anchor", "n_dec31", "median_lag_dec31", "mean_lag_dec31", "p90_lag_dec31",
              "max_lag_dec31", "pct_over_45d_dec31", "median_lag_griddate",
              "pct_over_45d_griddate"]].to_string(index=False))
    print("\n  the 12-30 / 12-29 rows are the defect: a full quarter of silent lag.\n")

    print("=" * 78)
    print("4. COVERAGE -- what the derived route buys")
    print("=" * 78)
    print(coverage_table(panel).to_string(index=False))
    print("\n  !! n_universe is the PANEL's source list and is SURVIVORS-ONLY -- NOT the")
    print("     backtest universe.  The gain above is real within the live universe only;")
    print("     the pure 'derived' route drops the delisted half of a real grid. Use the")
    print("     'derived+real' composite route for anything universe-wide.")
    print("\n  reach-back BEFORE the price file exists at all:")
    print(deep_history_table(panel).to_string(index=False))
    print()

    if not args.skip_ic:
        print("=" * 78)
        print("5. CONVENTION BAKE-OFF -- rank-IC residual (derived IC - real IC)")
        print("=" * 78)
        ic = ic_residual_table(panel)
        summ = ic.groupby("convention")["residual"].agg(
            mean="mean", median="median", mean_abs=lambda s: s.abs().mean(),
            max_abs=lambda s: s.abs().max(), n_cells="size")
        print(summ.round(4).to_string())
        print("\n  residual by signal kind (MC = shares marketCap with the price):")
        print(ic.pivot_table(index="convention", columns="kind", values="residual",
                             aggfunc="mean").round(4).to_string())
        print("\n  the MC/NM split is the diagnosis: the surviving residual is a")
        print("  shared-denominator artifact, not a dividend defect.\n")

    if args.csv:
        fid.to_csv(args.csv, index=False)
        print(f"fidelity table written to: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
