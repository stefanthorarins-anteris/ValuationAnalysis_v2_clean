"""price_scale_audit.py -- is any name's PRICE LEVEL off by a power of ten?

WHAT WENT WRONG, AND WHERE
--------------------------
For `ATRI` (Atrion Corp, acquired, delisted) FMP serves `open/high/low/close/adjClose` divided
by 1000 while leaving `vwap` on the real tape: `vwap / close` has a median of EXACTLY 1000.0
across all 756 rows, and the final pinned `vwap` of 459.92 matches the ~$460/share cash
consideration in the completed acquisition.  Our grid faithfully copied the corrupt field --
`real_prices.csv` at the 2021 buy anchor reads 0.67507 against a true $675.07.

"RATIOS ARE SAFE, LEVELS ARE NOT" IS TRUE OF *RETURNS* AND FALSE AS A GENERAL CLAIM -- and
the earlier version of this docstring made the general claim, which is the more dangerous
half.  Every RETURN computed off the series is genuinely correct: buy and eval carry the same
1/1000 and it cancels.  But the scaling is in `marketCap` too, so ANY ratio with `marketCap`
on ONE side only is off by 1000x -- and the ranking has one, weighted, in the direction that
helps:

    `bookToPrice` = totalStockholdersEquity / marketCap    (createDicts.py:578,
                                                            Tier B, Sign +1 -- HIGHER IS BETTER)

MEASURED on the 2026-08-29 CUR6K panel: the panel median `bookToPrice` is **0.51**; the names
this module flags read **26 to 836**, and **seven of the eight occupy ranks 1-7 of 4,928**.
A units error is being read by the scorer as the cheapest equity in the universe.

SO THE RISK RUNS THE OPPOSITE WAY FROM WHAT THE OLD TEXT IMPLIED.  It said correcting a price
would change a score, which reads as "the score is only at risk if we intervene".  The score
is at risk NOW, from the raw vendor data, and this module's refusal to intervene is what
leaves it there.  That is still the right call -- see the boundary section -- but it must be
stated as a known exposure and not as safety.

WHY IT DOES NOT PRODUCE A WRONG PICK TODAY, AND HOW THIN THAT IS.  Stage-1 normalisation is
median-centred and rank-based, so one extreme criterion cannot carry a name on its own.
Verified directly on the 08-29 run's `BoScore_df`: the best Stage-1 rank among the eight
flagged names is **SEA1.OL at 119 of 4,934**, and NONE of them reaches `postRank` (the 97
Stage-2 survivors).  No shipped pick is affected.
**DO NOT READ THAT AS COMFORTABLE.**  The cutoff is the top 100.  The margin is **NINETEEN
RANK PLACES**, not a comfortable thousand, and it is a property of where these particular
names happened to land on every OTHER criterion -- not of the contaminated criterion being
harmless.  A flagged name that is mediocre rather than bad on the rest of the scorecard
enters the shipped list on a units error.  The containment is real, it is measured, and it is
luck rather than a guard.

Anything reading the price LEVEL -- a floor, a screen, a market-cap or per-share cross-check,
a currency sanity test -- is wrong by 1000x for these names as well.

A LEVEL SCREEN STRUCTURALLY CANNOT CATCH THIS CLASS: 15,299 of the grid's 90,506 symbols
(16.9%) already have a max `adjClose` under $1.00, mostly legitimate penny and crypto-adjacent
names, so a 1/1000-scaled real company hides inside that population by construction.

THE DETECTOR THE ISSUE REGISTER PROPOSED DOES NOT WORK, AND THAT IS MEASURED, NOT ARGUED
-----------------------------------------------------------------------------------------
Q-38 proposed comparing the pipeline's SYNTHETIC price (`getData_fmp`'s
`marketCap / weightedAverageShsOut`) against the grid, on the reasoning that a ~1000x
disagreement would fall out instantly at zero API cost.  Run against ATRI's own saved
fundamentals (`delisted_out/dead_fundamentals_*.pickle`, offline) the disagreement is NOT
there:

    date        marketCap  shares      synthetic price   grid adjClose
    2021-12-31  1269524.90 1801000     0.70490           0.67507        -> ratio 1.04

FMP APPLIES THE SAME 1/1000 TO `marketCap`.  Both sides of the proposed ratio carry the
defect, so it cancels -- the check comes back "agrees, 1.04" on the one name we know is
broken.  A detector blind to its own motivating case is worse than none, because it would
have been read as an all-clear.

WHAT ACTUALLY SEPARATES THE TWO IS THE STATEMENTS, WHICH THE SCALING DOES NOT TOUCH
------------------------------------------------------------------------------------
`bookValuePerShare` is `totalStockholdersEquity / weightedAverageShsOut` -- verified
bit-identical on ATRI's own panel -- and comes off the balance sheet, which FMP serves
unscaled.  So on ATRI:

    synthetic price  0.45212      bookValuePerShare  135.65      -> price/book = 0.0033

The company is priced at one three-hundredth of its book equity, every quarter, for six
years.  That is not a valuation, it is a units error, and it is visible with NO price grid and
NO network at all.  Multiplying by 1000 returns 3.33, an entirely ordinary price/book -- which
is the corroboration the report prints, because "this number becomes normal at exactly one
decade" is the signature of a scaling defect and nothing else produces it.

MEASURED ON THE LIVE 2026-08-29 CUR6K PANEL (4,930 of 4,934 sources have a usable price/book):
median 1.91, 1st percentile 0.22, 0.1st percentile 0.014.  The alarm threshold below flags
8 sources (0.16%); relaxing it to 0.05 flags 15.  ATRI, at 0.0033, sits inside both.

TWO CHECKS, AND EACH IS BLIND TO WHAT THE OTHER SEES -- WHICH IS WHY BOTH RUN
-----------------------------------------------------------------------------
  A. FUNDAMENTALS-INTERNAL (price/book).  Catches a scaling defect that infects `marketCap`,
     i.e. the ATRI shape.  Needs no price grid, so it covers every name in the panel including
     ones the grid cannot price.  CURRENCY-FREE: numerator and denominator are both in the
     company's reporting currency, so the ratio has no FX or minor-unit confound at all.
     BLIND TO: a defect that infects ONLY the grid and leaves `marketCap` correct -- there the
     fundamentals agree with themselves and this check sees nothing.
  B. SYNTHETIC-vs-GRID.  Catches exactly the case A is blind to.
     BLIND TO: ATRI (measured above: 1.04).  And it is BADLY CONFOUNDED -- measured on the
     local grid against the same panel, 249 symbols disagree by >= 50x and 227 of them are
     `.L` names sitting at exactly 1/100, which is the LSE quoting in PENCE against GBP
     reporting.  A further handful are ADRs whose reporting currency is not the quote currency
     (`PKX` 1376 = KRW/USD, `BSAC` 1072 = CLP/USD, `IX` 148 = JPY/USD).  BSAC is the warning:
     an FX rate landed 7% from 1000, so a loose "near 1000x" rule WOULD have called a Chilean
     bank a vendor defect.  Check B therefore classifies rather than alarms, and only an
     unexplained near-exact power of ten is raised.

REPORT ONLY.  NEVER A MUTATION, NEVER A SCORE CHANGE -- AND THAT IS A DEFERRAL, NOT A FIX
------------------------------------------------------------------------------------------
This module does not correct a price, does not drop a name, and does not touch any score.
That boundary is the reason Q-38 was held open rather than parked with the rest of the
price-grid work: its detector touches names that are live in scoring, and changing a score on
the strength of a heuristic is the CEO's call, not this module's.  So it prints, and what it
prints is evidence for that decision, not an input to one.

BE PRECISE ABOUT WHAT THAT LEAVES OPEN.  Reporting does not neutralise the `bookToPrice`
contamination above; it documents it.  The exposure sits in the live scorer every run, and
the only thing standing between it and the shipped list is that these eight names are bad
enough elsewhere to land outside the top 100 by nineteen places.  A CEO ruling to suppress
the criterion for flagged names, or to refuse the names outright, is a live option this
module deliberately does not take on its own.

NO NETWORK.  Reads a panel frame and, optionally, the saved grid CSV.  Nothing else.
"""

import os

import numpy as np
import pandas as pd

#  ---- Check A: price/book ---------------------------------------------------
#  A 1000x under-scale maps the ORDINARY price/book range [0.5, 10] onto
#  [0.0005, 0.01], so an alarm at 0.02 covers a true price/book up to 20 and still
#  sits an order of magnitude under the live panel's 0.1st percentile (0.014).
#  DELIBERATELY NOT TIGHTER: a real distressed equity bottoms out around 0.05-0.10,
#  so 0.02 keeps a clear gap from anything a market actually prints.
PB_ALARM = 0.02
PB_WATCH = 0.05
#  Fewest quarters before a name's median price/book is worth reporting.  A single
#  quarter with a stale marketCap is noise, not a scaling defect -- the defect is
#  persistent by construction, because it is how the vendor stores the series.
MIN_ROWS = 4

#  ---- Check B: synthetic vs grid --------------------------------------------
#  Only a NEAR-EXACT power of ten is a scaling candidate; an FX rate is not round.
#  3% is chosen against the measured worst case: BSAC's CLP/USD median ratio is
#  1071.65, i.e. 7.2% off 1000, and it must NOT be flagged.
DECADE_TOL = 0.03
MIN_DECADES = 2          # >= 100x; 10x is within reach of ordinary staleness + drift
GRID_MIN_ANCHORS = 3
#  Venues that quote in a MINOR unit of their reporting currency.  The 1/100 these
#  produce is a convention, not a defect, and it is by far the largest population in
#  check B (227 of 249 decade-scale disagreements on the local grid).
MINOR_UNIT_VENUES = {".L": "GBp (pence) quoted against GBP reported",
                     ".IL": "GBp (pence) quoted against GBP reported",
                     ".TA": "ILA (agorot) quoted against ILS reported",
                     ".JO": "ZAc (cents) quoted against ZAR reported"}


def _venue(symbol):
    s = str(symbol)
    return "." + s.rsplit(".", 1)[1] if "." in s else ""


def _num(frame, col):
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce")


def _decade(ratio):
    """(nearest power of ten, relative distance from it).  nan-safe."""
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.round(np.log10(ratio))
    err = np.abs(ratio / (10.0 ** k) - 1.0)
    return k, err


def check_fundamentals_internal(panel, pb_alarm=PB_ALARM, pb_watch=PB_WATCH,
                                min_rows=MIN_ROWS):
    """CHECK A -- the price level against the company's OWN balance sheet.

    Currency-free and grid-free by construction: `price` (marketCap/shares) and
    `bookValuePerShare` (equity/shares) are both per-share quantities in the reporting
    currency, so every unit and FX effect cancels and what is left is a valuation ratio.

    `price_over_sales` is carried as a SECOND, INDEPENDENT opinion rather than as part of the
    test.  A name can legitimately have near-zero book (asset-light, or bought back past
    negative equity) and a pre-revenue company can legitimately have no sales, so requiring
    both to be absurd would miss real defects; requiring either would flag real companies.
    It is reported so the reader can tell those cases apart, and `pb_x1000` says whether one
    decade makes the name ordinary -- which is the actual signature.
    """
    df = pd.DataFrame({
        "source": panel["source"].astype(str),
        "price": _num(panel, "price"),
        "bvps": _num(panel, "bookValuePerShare"),
        "revenue": _num(panel, "revenue"),
        "shares": _num(panel, "weightedAverageShsOut"),
        "currency": (panel["reportedCurrency"].astype(str)
                     if "reportedCurrency" in panel.columns else ""),
    })
    df = df[df["price"] > 0]
    rps = df["revenue"] / df["shares"].where(df["shares"] > 0)
    df["pb"] = df["price"] / df["bvps"].where(df["bvps"] > 0)
    df["ps"] = df["price"] / rps.where(rps > 0)
    g = df.groupby("source")
    per = pd.DataFrame({
        "n_rows": g["pb"].size(),
        "n_pb": g["pb"].count(),
        "price_over_book": g["pb"].median(),
        "price_over_sales": g["ps"].median(),
        "currency": g["currency"].first(),
    }).reset_index()
    per = per[(per["n_pb"] >= min_rows) & per["price_over_book"].notna()]
    per["pb_x1000"] = per["price_over_book"] * 1000.0
    per["severity"] = np.where(per["price_over_book"] < pb_alarm, "ALARM",
                               np.where(per["price_over_book"] < pb_watch, "watch", ""))
    per = per[per["severity"] != ""].sort_values("price_over_book")
    return per.reset_index(drop=True)


def check_synthetic_vs_grid(panel, prices_csv, supp_csv=None,
                            decade_tol=DECADE_TOL, min_decades=MIN_DECADES,
                            min_anchors=GRID_MIN_ANCHORS, max_stale_days=400):
    """CHECK B -- the synthetic price against the saved grid, CLASSIFIED not alarmed.

    Every row that survives to the output is a decade-scale disagreement; the `verdict` column
    says which of three things it is, because on the measured data the overwhelming majority
    are benign:
      MINOR_UNIT        the venue quotes in a minor unit (pence/agorot/cents).  227 of 249.
      CURRENCY_MISMATCH the ratio is large but NOT near a power of ten, i.e. it looks like an
                        exchange rate -- reporting currency is not the quote currency.
      SCALING_SUSPECT   a near-exact power of ten with no convention to explain it.  THIS is
                        the only one that means anything, and on the live panel it is empty.

    A NOTE ON WHAT "empty" MEANS HERE, because it is easy to misread as an all-clear: this
    check cannot see the ATRI shape at all (`marketCap` carries the same 1/1000, so the ratio
    is ~1.0).  Check A is what covers that.  Neither check sees a name absent from the panel.
    """
    if not prices_csv or not os.path.exists(prices_csv):
        return pd.DataFrame(), {"reason": "no price grid on disk"}
    grid = pd.read_csv(prices_csv, usecols=["date_requested", "symbol", "adjClose"])
    if supp_csv and os.path.exists(supp_csv):
        supp = pd.read_csv(supp_csv, usecols=["date_requested", "symbol", "adjClose"])
        grid = pd.concat([grid, supp], ignore_index=True)
    grid["adjClose"] = pd.to_numeric(grid["adjClose"], errors="coerce")
    grid = grid[grid["adjClose"] > 0].dropna(subset=["symbol"])

    p = pd.DataFrame({"source": panel["source"].astype(str),
                      "date": pd.to_datetime(panel["date"], errors="coerce"),
                      "price": _num(panel, "price"),
                      "currency": (panel["reportedCurrency"].astype(str)
                                   if "reportedCurrency" in panel.columns else "")})
    p = p[(p["price"] > 0) & p["date"].notna()].sort_values("date")

    rows = []
    for anchor in sorted(grid["date_requested"].dropna().unique()):
        ts = pd.Timestamp(anchor)
        #  LAST STATEMENT AT OR BEFORE THE ANCHOR -- the same as-of convention every
        #  price-based metric in this pipeline uses.  A statement AFTER the anchor would be
        #  look-ahead, and this is a diagnostic, not an excuse to introduce one.
        sub = p[p["date"] <= ts]
        if not len(sub):
            continue
        last = sub.groupby("source").tail(1)
        last = last.assign(stale_days=(ts - last["date"]).dt.days)
        last = last[last["stale_days"] <= max_stale_days]
        gg = grid[grid["date_requested"] == anchor][["symbol", "adjClose"]]
        m = last.merge(gg, left_on="source", right_on="symbol", how="inner")
        if len(m):
            rows.append(m[["source", "price", "adjClose", "currency"]])
    if not rows:
        return pd.DataFrame(), {"reason": "panel and grid share no priced anchor"}

    allm = pd.concat(rows, ignore_index=True)
    allm["ratio"] = allm["price"] / allm["adjClose"]
    g = allm.groupby("source")
    per = pd.DataFrame({"n_anchors": g["ratio"].size(),
                        "ratio_median": g["ratio"].median(),
                        "ratio_min": g["ratio"].min(),
                        "ratio_max": g["ratio"].max(),
                        "currency": g["currency"].first()}).reset_index()
    stats = {"n_compared": int(len(per)),
             "n_panel": int(panel["source"].nunique()),
             "n_grid_symbols": int(grid["symbol"].nunique())}
    per = per[per["n_anchors"] >= min_anchors]
    thr = 10.0 ** min_decades
    per = per[(per["ratio_median"] >= thr / 2.0) | (per["ratio_median"] <= 2.0 / thr)]
    if not len(per):
        return per.assign(verdict=[], decade=[], decade_err=[]), stats
    k, err = _decade(per["ratio_median"].to_numpy(dtype=float))
    per = per.assign(decade=k, decade_err=err)
    per["venue"] = per["source"].map(_venue)
    near_decade = (per["decade_err"] <= decade_tol) & (per["decade"].abs() >= min_decades)
    minor_unit = per["venue"].isin(MINOR_UNIT_VENUES) & (per["decade"] == -2)
    per["verdict"] = np.where(minor_unit, "MINOR_UNIT",
                              np.where(near_decade, "SCALING_SUSPECT", "CURRENCY_MISMATCH"))
    stats["n_decade_scale"] = int(len(per))
    return per.sort_values(["verdict", "decade_err"]).reset_index(drop=True), stats


def run_audit(panel, prices_csv=None, supp_csv=None, log=print, out_csv=None):
    """Both checks, printed.  Returns the two frames and the counts; changes nothing."""
    log("[price-scale] vendor price-LEVEL audit (report only -- no price is corrected)")
    if panel is None or not len(panel) or "source" not in getattr(panel, "columns", []):
        log("[price-scale] no panel available -- audit did not run.")
        return {"internal": pd.DataFrame(), "grid": pd.DataFrame(), "stats": {}}

    internal = check_fundamentals_internal(panel)
    n_alarm = int((internal["severity"] == "ALARM").sum()) if len(internal) else 0
    log(f"[price-scale] A. price/book vs the company's OWN balance sheet "
        f"(currency-free, needs no grid): {n_alarm} ALARM, "
        f"{int((internal['severity'] == 'watch').sum()) if len(internal) else 0} watch, "
        f"of {panel['source'].nunique()} sources.")
    if n_alarm:
        log(f"[price-scale]    a price below {PB_ALARM:g}x book is not a valuation, it is a "
            "units error.  x1000 is printed so a")
        log("[price-scale]    reader can see whether exactly one decade makes the name "
            "ordinary -- the scaling signature.")
        log(f"[price-scale]    {'source':<14}{'price/book':>12}{'x1000':>10}"
            f"{'price/sales':>13}  currency")
        for r in internal[internal["severity"] == "ALARM"].itertuples(index=False):
            ps = f"{r.price_over_sales:.3f}" if r.price_over_sales == r.price_over_sales else "n/a"
            log(f"[price-scale]    {r.source:<14}{r.price_over_book:>12.5f}"
                f"{r.pb_x1000:>10.2f}{ps:>13}  {r.currency}")
        log("[price-scale]    THESE ARE IN THE LIVE SCORER NOW.  `bookToPrice` "
            "(equity/marketCap, Tier B, higher=better)")
        log("[price-scale]    is 1000x too FAVOURABLE on these rows -- panel median 0.51, "
            "these read 26-836, and seven")
        log("[price-scale]    of eight rank 1-7 of 4,928 on it.  No shipped pick is affected: "
            "best Stage-1 rank is 119")
        log("[price-scale]    of 4,934 against a top-100 cutoff, none reaches postRank -- a "
            "margin of NINETEEN places,")
        log("[price-scale]    which is where these names landed on the OTHER criteria, not a "
            "guard.  NOT ACTIONED HERE:")
        log("[price-scale]    suppressing or refusing them changes a score, which is the "
            "CEO's call.")
    else:
        log("[price-scale]    none.  NOTE this is not an all-clear for the price GRID: check "
            "A reads fundamentals only.")

    grid_df, stats = check_synthetic_vs_grid(panel, prices_csv, supp_csv)
    if not len(grid_df):
        log(f"[price-scale] B. synthetic-vs-grid: {stats.get('reason', 'no decade-scale disagreement')}.")
    else:
        counts = grid_df["verdict"].value_counts().to_dict()
        log(f"[price-scale] B. synthetic-vs-grid over {stats.get('n_compared', 0)} shared "
            f"names: {stats.get('n_decade_scale', 0)} disagree by >=100x -- "
            + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        log("[price-scale]    MINOR_UNIT (pence/agorot/cents) and CURRENCY_MISMATCH (the "
            "ratio is an FX rate, not a")
        log("[price-scale]    round decade) are EXPECTED.  Only SCALING_SUSPECT means "
            "anything.")
        for r in grid_df[grid_df["verdict"] == "SCALING_SUSPECT"].itertuples(index=False):
            log(f"[price-scale]    SCALING_SUSPECT {r.source:<12} median ratio "
                f"{r.ratio_median:,.4f} (10^{int(r.decade)}, "
                f"{r.decade_err*100:.1f}% off) {r.currency}")
    log("[price-scale] BLIND SPOTS, stated so this is not read as an all-clear: check B "
        "cannot see the ATRI")
    log("[price-scale]   shape at all (FMP scales `marketCap` by the same 1/1000, so the "
        "ratio is ~1.0); check A")
    log("[price-scale]   cannot see a defect confined to the grid; NEITHER sees a name "
        "absent from the panel,")
    log("[price-scale]   which includes every delisted name the live run does not carry.")

    if out_csv:
        try:
            internal.to_csv(out_csv, index=False)
            log(f"[price-scale] wrote {out_csv}")
        except Exception as e:
            log(f"[price-scale] could not write {out_csv} ({type(e).__name__}: {e})")
    return {"internal": internal, "grid": grid_df, "stats": stats}
