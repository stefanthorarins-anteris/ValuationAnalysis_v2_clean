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

THIS MODULE IS STILL REPORT-ONLY.  THE PIPELINE IS NOT.  (Q-48 ACTIONED, 2026-09-01)
------------------------------------------------------------------------------------
The paragraph that stood here said the deferral was the right call and that a ruling to
suppress the criterion was "a live option this module deliberately does not take on its own".
THE CEO REOPENED IT AND RULED THE OTHER WAY: reading a corrupted vendor number is a BUG, not a
scoring preference, and a correctness defect gets fixed rather than reported.

WHAT CHANGED, AND IT IS NOT IN THIS FILE.  `nan_policy.price_scale_hits` is a third
input-sanity rule, hooked into `refuse_impossible_cells` at `the refusal hook in getData_fmp.getFundamentalsData` -- the one
place `tempfund` is both the frame every Stage-1 metric is computed from AND the frame that
becomes `cdx_df`.  A row whose `price / bookValuePerShare` is under `PB_ALARM` has `price`,
`marketCap`, `bookValuePerShare` and `earningsYield` set to NaN, and every metric in either
stage that reads one of them is then absent rather than wrong.  ABSENCE, NOT CORRECTION: no
price is multiplied by anything, because the observed defects are not one decade (measured:
QBY0.DE/0CHZ.L are ~100x on the SHARE COUNT, ATRI is 1/1000 on the PRICE, CCM has negative
equity and no decade at all), so there is no factor to assert.

WHAT THIS MODULE STILL DOES, AND WHY IT MATTERS MORE NOW.  It reports.  But the refusal blanks
the two legs check A reads, so on a post-2026-09-01 panel the rows this module was built to
name are invisible to it -- `0 ALARM` over a defect that WAS found.  `refused_upstream` reads
the `nan_policy.SANITY_REFUSED_COLUMN` stamp and `run_audit` prints the refused sources FIRST,
so "found nothing" and "already refused" can never print as the same line.  Read the A0 block
before the ALARM count: on a refusing panel the ALARM count is what SURVIVED the refusal.

WHAT IS STILL OPEN, stated plainly.  The rule UNDER-REACHES: the share-count corruption
signature is just as common in the 0.02-0.10 price/book band, which is left alone (the density
table is in `nan_policy`, beside the constant).  And check B is unchanged -- a defect confined
to the price GRID still only classifies, never refuses.

NO NETWORK.  Reads a panel frame and, optionally, the saved grid CSV.  Nothing else.
"""

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nan_policy as npol

#  ---- Check A: price/book ---------------------------------------------------
#  A 1000x under-scale maps the ORDINARY price/book range [0.5, 10] onto
#  [0.0005, 0.01], so an alarm at 0.02 covers a true price/book up to 20 and still
#  sits an order of magnitude under the live panel's 0.1st percentile (0.014).
#  DELIBERATELY NOT TIGHTER: a real distressed equity bottoms out around 0.05-0.10,
#  so 0.02 keeps a clear gap from anything a market actually prints.
#
#  THE NUMBER NOW LIVES IN `nan_policy`, AND THIS MODULE IMPORTS IT (Q-48, 2026-09-01).
#  Since that date the pipeline does not merely REPORT this defect, it REFUSES the
#  contradicted cells at ingest (`nan_policy.price_scale_hits`, hooked into
#  `refuse_impossible_cells`).  A reporting threshold and a refusing threshold that
#  could drift apart would be two definitions of "contaminated" -- the exact failure
#  this repo has been bitten by repeatedly -- so there is one constant and the
#  REFUSING side owns it.  The derivation above is the record of how it was set and
#  is unchanged; only its home moved.
PB_ALARM = npol.PRICE_SCALE_PB_ALARM
#  THE REFUSING RULE HAS TWO LEVELS SINCE Q-75 AND THIS SIDE MUST CARRY BOTH, for exactly the
#  reason the paragraph above gives.  `PB_ALARM` alone is no longer "the number the refusal
#  uses": a row between `PB_ALARM` and `PB_WIDE` is ALSO refused when the share-count witness
#  fires, and an audit that named only the floor would under-report the refusal it exists to
#  make legible -- the same drift the single-constant rule was written to prevent, one level
#  along.  Both are IMPORTED, never re-declared, and
#  `test_there_is_exactly_one_definition_of_contaminated` pins both.
PB_WIDE = npol.PRICE_SCALE_PB_WIDE
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


def bookToPrice_ranks(panel, sources):
    """Where `sources` sit on `bookToPrice` within the WHOLE panel, computed now.

    `bookToPrice` is `totalStockholdersEquity / marketCap` (createDicts.py:578, Tier B,
    Sign +1 -- HIGHER IS BETTER), which is the column a 1000x-under-scaled `marketCap`
    inflates by exactly one decade.  Ranked DESCENDING, so rank 1 is the most favourable
    reading in the universe -- which is where a units error lands.

    Returns (ranks:{source: rank}, n_ranked, panel_median, route).  `ranks` is empty when the
    panel cannot produce the column at all, and the caller must then say the exposure was NOT
    QUANTIFIED rather than print a number it did not compute.

    FALLBACK, and it is exact rather than approximate: with `totalStockholdersEquity` or
    `marketCap` absent, `bookValuePerShare / price` is the same ratio with `shares`
    cancelled top and bottom (the module docstring records that identity as verified on
    ATRI's own panel), so the ranking is unchanged.  Which route was taken is returned so
    the log can say it.
    """
    cols = getattr(panel, "columns", [])
    if "totalStockholdersEquity" in cols and "marketCap" in cols:
        num, den, route = _num(panel, "totalStockholdersEquity"), _num(panel, "marketCap"), "equity/marketCap"
    elif "bookValuePerShare" in cols and "price" in cols:
        num, den, route = _num(panel, "bookValuePerShare"), _num(panel, "price"), "bookValuePerShare/price"
    else:
        return {}, 0, float("nan"), "unavailable"
    df = pd.DataFrame({"source": panel["source"].astype(str),
                       "b2p": num / den.where(den > 0)})
    per = df.groupby("source")["b2p"].median().dropna()
    if not len(per):
        return {}, 0, float("nan"), route
    #  method='min' so tied readings share the BEST rank -- reporting a contaminated name
    #  one place better than it is would be the conservative direction, and reporting it
    #  worse than it is would understate the exposure.
    ranked = per.rank(ascending=False, method="min").astype(int)
    want = [str(x) for x in sources]
    return ({s: int(ranked[s]) for s in want if s in ranked.index},
            int(len(per)), float(per.median()), route)


def refused_upstream(panel, report=None):
    """{source: n_rows} for sources whose price-scale cells `nan_policy` already refused.

    WHY THIS EXISTS AND WHY IT IS NOT OPTIONAL.  Since 2026-09-01 the ingest REFUSES the
    contradicted cells rather than only reporting them (`nan_policy.price_scale_hits`).
    Check A reads `price` and `bookValuePerShare`; the refusal sets BOTH to NaN.  So on a
    panel built after that change, the very rows this module exists to name are invisible to
    it, and `0 ALARM` would print over a defect that was found and handled -- indistinguishable
    from a defect that was never there.  This module's own docstring says the rule: "a detector
    blind to its own motivating case is worse than none, because it would have been read as an
    all-clear."

    Reads the `nan_policy.SANITY_REFUSED_COLUMN` stamp, which rides the panel through
    `pd.concat`, the saved pickle and the `-loadbometric` reload.  A panel without that column
    (every pre-2026-09-01 pickle, every test frame) refused nothing and returns {}.

    Returns {source: {"rows": int, "worst_ratio": float}}.  `worst_ratio` is NaN unless the
    run's refusal `report` is passed -- the ratio is NOT recoverable from a refused panel.
    """
    col = npol.SANITY_REFUSED_COLUMN
    if panel is None or col not in getattr(panel, "columns", []) or "source" not in panel.columns:
        return {}
    #  `price` is the leg check A cannot do without, so it is the one to key on: any row this
    #  rule refused has it.  THAT IT IS UNIQUE TO THIS RULE IS AN INVARIANT, NOT A FACT OF
    #  NATURE (review S4-2): a future `nan_policy` rule naming `price` would make this
    #  over-report and attribute another rule's refusals to this one.
    #  `test_price_scale_refusal.py` pins it.
    mask = npol.refused_fields_mask(panel, "price").fillna(False)
    if not bool(mask.any()):
        return {}
    hit = panel.loc[mask.to_numpy(dtype=bool), "source"].astype(str)
    out = {src: {"rows": int(n), "worst_ratio": float("nan"), "witness": float("nan")}
           for src, n in hit.value_counts().to_dict().items()}
    #  THE WITNESS IS RECOVERABLE FROM THE PANEL AND THE RATIO IS NOT, and that asymmetry is
    #  the point of reporting it here.  `marketCap` is one of the cells this rule blanks, so
    #  the price/book multiple can only come from the refusal report -- but
    #  `weightedAverageShsOut` is refused by NO producer, so the share-count witness can always
    #  be recomputed from the panel in hand.  Without it a reader of A0 cannot tell WHICH LEVEL
    #  refused a source: under `PB_ALARM` unconditionally, or in the witnessed band above it.
    #  That distinction is the whole content of the Q-75 widening.
    try:
        wr = npol.share_count_witness_ratio(panel)
        wsrc = panel["source"].astype(str)
        for src_name, g in wr[mask.to_numpy(dtype=bool)].groupby(
                wsrc[mask.to_numpy(dtype=bool)]):
            if str(src_name) in out and g.notna().any():
                out[str(src_name)]["witness"] = float(g.max())
    except Exception:
        #  A panel without the share-count column reports the rows and the ratio and says
        #  nothing about the witness -- never a wrong number in place of an absent one.
        pass
    #  THE RATIO, WHEN THE RUN'S REFUSAL REPORT IS TO HAND.  `price_scale_hits` records it per
    #  cell precisely so the cut can be argued with, and A0 that names a source without saying
    #  HOW FAR under the cut it sat carries strictly less than the check A line it replaces.
    #  It is NOT recoverable from the panel -- `marketCap` is one of the refused cells.
    if report is not None and len(report):
        rcols = getattr(report, "columns", [])
        if all(c in rcols for c in ("source", "relation", "ratio")):
            r = report[report["relation"] == npol.PRICE_SCALE_RELATION]
            if len(r):
                for src, v in r.groupby("source")["ratio"].min().items():
                    if str(src) in out:
                        out[str(src)]["worst_ratio"] = float(v)
    return out


def _stage1_rank_map(stage1_scores):
    """{source: rank} from a Stage-1 score frame or an already-ranked sequence.

    Accepts `resdic['BoScore_df']` (a frame with `source` + `score`) and re-sorts it here
    rather than trusting the order it arrives in -- the rank is the claim, so the sort that
    produces it belongs beside the claim.  A plain sequence of sources is taken as already
    ranked.  Anything else -> {} -> the caller says NOT CHECKED."""
    if stage1_scores is None:
        return {}
    cols = getattr(stage1_scores, "columns", None)
    if cols is not None and "source" in cols:
        df = stage1_scores
        if "score" in cols:
            df = df.sort_values("score", ascending=False)
        order = [str(x) for x in df["source"].tolist()]
    else:
        try:
            order = [str(x) for x in stage1_scores]
        except TypeError:
            return {}
    return {s: i + 1 for i, s in enumerate(order)}


def _containment_lines(internal, panel, stage1_scores, shipped_sources, topn_stage1,
                       label, extra_flagged=()):
    """The containment paragraph, COMPUTED -- or an explicit refusal to assert one.

    WHAT THIS REPLACES, because the shape matters more than the numbers.  This paragraph was
    a FROZEN 2026-08-29 measurement printed unconditionally whenever `n_alarm > 0`: "seven of
    eight rank 1-7 of 4,928 ... best Stage-1 rank is 119 of 4,934".  The 08-31 run printed it
    verbatim against a panel of 4,941, and it said "eight" only because that night's ALARM
    count happened to be eight as well.  The conclusion was true that night and the code did
    not check it -- so the first run whose ALARM set moves would have asserted a false
    all-clear about the shipped list, in the CEO's own words, in the log he reads.

    THE OTHER HALF OF THE FIX IS THE REFUSAL.  With neither a selected list nor a Stage-1
    ranking to hand, this says CONTAINMENT NOT CHECKED and asserts nothing.  An absent input
    must never come out as an all-clear; that is the same failure one level down.
    """
    #  THE REFUSED SOURCES ARE PART OF THIS POPULATION (review S3-2, 2026-09-01).  A0 fixed the
    #  ALARM COUNT and left the sentence underneath it: this paragraph derives its population
    #  from `internal`, and `check_fundamentals_internal` drops refused rows (`price > 0`) and
    #  then drops thin sources (`n_pb >= MIN_ROWS`).  So a partially-refused source silently
    #  DOWNGRADES out of ALARM, and a fully-refused one leaves `internal` altogether -- and if
    #  that emptied `names`, this function returned [] and printed NOTHING AT ALL.  Not "not
    #  checked", not an all-clear: silence, over a defect that HAD been found.  That is the
    #  exact failure the note above forbids, one level up.
    alarm = [str(x) for x in internal.loc[internal["severity"] == "ALARM", "source"]] if len(internal) else []
    refused = sorted(str(x) for x in (extra_flagged or ()))
    names = sorted(set(alarm) | set(refused))
    L = []
    if not names:
        return L
    if refused:
        L.append("    POPULATION: %d ALARM on this panel + %d REFUSED UPSTREAM (invisible to "
                 "check A) = %d name(s)." % (len(alarm), len(refused), len(names)))
    ranks, n_ranked, med, route = bookToPrice_ranks(panel, names)
    #  THE DENOMINATOR MUST NOT COUNT NAMES THAT CANNOT BE RANKED (review 3, S3-2).
    #  `bookToPrice_ranks` needs a computable equity/marketCap median, and a FULLY REFUSED
    #  source has none -- so it leaves `ranks` while staying in `len(names)`, and the head
    #  count reads "1 of 2 rank in the top 2" about a population whose MOST contaminated
    #  member silently dropped out of the numerator.  The most-refused names are exactly the
    #  ones that vanish, so the sentence errs toward an all-clear -- the failure this whole
    #  block exists to prevent, one level in.
    n_unrankable = len([x for x in names if x not in ranks])
    if ranks:
        top = sorted(ranks.values())
        n_in_head = sum(1 for r in top if r <= len(ranks))
        #  NAMES THE PANEL, because this function now runs over two of them and "the live
        #  scorer" is true of only one.  A containment sentence that does not say which
        #  population it is about is the frozen-paragraph defect in a different costume.
        L.append("    THESE FEED A RANKING (%s).  `bookToPrice` (%s, Tier B, higher=better)"
                 % (label or "live scorer", route))
        L.append("    is 1000x too FAVOURABLE on a scaled row.  Panel median %.2f; %d of %d "
                 "RANKABLE flagged names rank" % (med, n_in_head, len(ranks)))
        if n_unrankable:
            L.append("    (%d of the %d flagged name(s) are NOT RANKABLE here -- refused on "
                     "every row, so they have no" % (n_unrankable, len(names)))
            L.append("    computable bookToPrice at all.  They are the MOST contaminated of "
                     "the set, not the least.)")
        #  `len(ranks)`, NOT `len(names)` -- THIS LINE IS THE CONSUMER OF THE THRESHOLD SET
        #  FOUR LINES UP.  The S3-2 fix moved `n_in_head`'s bound and the printed DENOMINATOR
        #  to the rankable count and left this, the printed THRESHOLD, on the flagged count.
        #  The sentence then read "1 of 2 ... in the top 3", which is simply false: both
        #  ranks were <= 3.  Because `ranks` is a subset of `names` the error is one-signed --
        #  it UNDER-reports, and it fires exactly when a source is fully refused, which is the
        #  all-clear bias S3-2 existed to remove.  Fixing lines and not their readers is the
        #  repeat defect this change has now produced seven times.
        L.append("    in the top %d of %d on it (ranks %s)."
                 % (len(ranks), n_ranked,
                    ", ".join(str(r) for r in top[:12])
                    + (", ..." if len(top) > 12 else "")))
    else:
        #  WORDING (review 3, S4-3): the PANEL may be perfectly computable -- what is not
        #  computable is bookToPrice FOR THESE NAMES, typically because every one of their
        #  rows was refused.  "not computable on this panel" overstates it and reads as a
        #  data-availability problem rather than as the refusal doing its job.
        L.append("    bookToPrice NOT COMPUTABLE FOR ANY OF THE %d FLAGGED NAME(S) (the panel "
                 "itself may be fine; a fully" % len(names))
        L.append("    refused source has no computable bookToPrice).  The exposure is NOT "
                 "quantified this run.")

    #  THE TWO CONTAINMENT QUESTIONS ARE INDEPENDENT and are answered independently.
    #  "does a flagged name REACH the shipped list" is the one that matters and needs only
    #  the list; "how far from the cutoff is the nearest one" is the margin and needs the
    #  Stage-1 ranking.  Coupling them (the first version did) meant the LIVE pass could
    #  answer both while the PIT pass -- which has a shipped list per anchor but no Stage-1
    #  frame -- answered neither, and printed NOT CHECKED over a question it could answer.
    s1 = _stage1_rank_map(stage1_scores)
    #  NOT `shipped_sources or []`: `or` invokes `__bool__`, which RAISES
    #  `ValueError: The truth value of a Series is ambiguous` for any pandas Series or
    #  multi-element ndarray.  Production passes a list so it was latent, but a diagnostic
    #  that dies on a Series is a diagnostic nobody gets (review S4-3).
    shipped = set(str(x) for x in ([] if shipped_sources is None else shipped_sources))
    flagged = set(names)
    answered = False

    if shipped:
        answered = True
        in_shipped = sorted(flagged & shipped)
        if in_shipped:
            L.append("    !!! A FLAGGED NAME IS IN THE SELECTED LIST: %s.  NOT CONTAINED."
                     % ", ".join(in_shipped))
        else:
            L.append("    None of the %d flagged names is among the %d selected -- "
                     "containment holds this run," % (len(flagged), len(shipped)))
            L.append("    MEASURED on this run's own list, not inherited from a previous "
                     "one.")
    if s1:
        hit = sorted((r, n) for n, r in s1.items() if n in flagged)
        if hit:
            answered = True
            best_r, best_n = hit[0]
            #  A NEGATIVE MARGIN IS A DIFFERENT SENTENCE, not a smaller number.  A flagged
            #  name INSIDE the Stage-1 cutoff has already reached the Stage-2 pool on a
            #  contaminated criterion, whether or not it survives to the shipped list --
            #  printing that as "a margin of -12 places" would bury the one case this
            #  paragraph exists to surface.
            if best_r <= topn_stage1:
                L.append("    !!! A FLAGGED NAME IS INSIDE THE STAGE-1 CUTOFF: %s at rank %d "
                         "of %d, top-%d." % (best_n, best_r, len(s1), topn_stage1))
                L.append("    It reached the Stage-2 pool carrying a contaminated "
                         "`bookToPrice`.  Whether it survives")
                L.append("    to the selected list is the line above; that it got that far "
                         "is this one.")
            else:
                L.append("    Best Stage-1 rank among them: %s at %d of %d against a top-%d "
                         "cutoff -- a margin" % (best_n, best_r, len(s1), topn_stage1))
                L.append("    of %d places.  That is where these names happened to land on "
                         "the OTHER criteria; it is" % (best_r - topn_stage1))
                L.append("    measured containment, NOT a guard, and a flagged name that is "
                         "merely mediocre elsewhere")
                L.append("    enters the list on a units error.")
        else:
            L.append("    None of the flagged names appears in the Stage-1 ranking passed "
                     "here (%d ranked), so" % len(s1))
            L.append("    no margin is computable for them.")
    if not answered:
        L.append("    CONTAINMENT NOT CHECKED: neither a selected list nor a Stage-1 "
                 "ranking was passed to this")
        L.append("    audit, so whether a flagged name reaches a shipped or graded list is "
                 "UNKNOWN this run.")
        L.append("    Read it as unknown.  It is NOT an all-clear.")
    else:
        #  WAS "NOT ACTIONED: suppressing or refusing a name changes a score, which is the
        #  CEO's call."  That sentence was true until the CEO reopened Q-48 and ruled that
        #  reading a corrupted vendor number is a BUG.  It is actioned now, upstream, per ROW
        #  -- so a name still reaching this paragraph is one whose SURVIVING rows read this
        #  way, which is a different and smaller statement than the old one.
        L.append("    ACTIONED UPSTREAM per row (`nan_policy.price_scale_hits`): a row "
                 "under price/book %g -- or under %g" % (PB_ALARM, PB_WIDE))
        L.append("    with a share count %gx off its source's own median -- has its"
                 % npol.PRICE_SCALE_WITNESS_FACTOR)
        L.append("    price/marketCap/book-per-share/earningsYield refused at ingest.  A name "
                 "listed here still has")
        L.append("    enough SURVIVING rows to read this way -- see the A0 block for what was "
                 "refused on this panel.")
    if label:
        L.append("    (panel: %s)" % label)
    return L


def run_audit(panel, prices_csv=None, supp_csv=None, log=print, out_csv=None,
              stage1_scores=None, shipped_sources=None, topn_stage1=100,
              panel_label=None, run_grid_check=True, refusal_report=None):
    """Both checks, printed.  Returns the two frames and the counts; changes nothing.

    stage1_scores   : the run's Stage-1 score frame (`resdic['BoScore_df']`) or an already-
                      ranked sequence of sources.  Absent -> the containment claim is NOT
                      MADE (see `_containment_lines`); it is never assumed.
    shipped_sources : the Stage-2 survivors actually shipped (`resdic['postRank']['source']`).
                      Absent -> likewise not asserted.
    run_grid_check  : check B (synthetic-vs-grid).  False runs check A alone, which is the
                      shape the PIT dead-merged pass needs: that panel's names are precisely
                      the ones the price grid does not carry, so check B has nothing to
                      compare and its "no disagreement" would read as an all-clear.
    """
    log("[price-scale] vendor price-LEVEL audit (report only -- no price is corrected)%s"
        % (" -- panel: %s" % panel_label if panel_label else ""))
    if panel is None or not len(panel) or "source" not in getattr(panel, "columns", []):
        log("[price-scale] no panel available -- audit did not run.")
        return {"internal": pd.DataFrame(), "grid": pd.DataFrame(), "stats": {}}

    internal = check_fundamentals_internal(panel)
    n_alarm = int((internal["severity"] == "ALARM").sum()) if len(internal) else 0
    #  BEFORE the ALARM count, deliberately: on a post-2026-09-01 panel the refusal is the
    #  reason the count is low, and printing the count first invites reading it as an
    #  all-clear.  See `refused_upstream`.
    _refused = refused_upstream(panel, report=refusal_report)
    if _refused:
        log("[price-scale] A0. ALREADY REFUSED UPSTREAM: %d source(s), %d row(s) had "
            "`price`/`marketCap`/`bookValuePerShare`/"
            % (len(_refused), sum(v["rows"] for v in _refused.values())))
        log("[price-scale]     `earningsYield` set to NaN at ingest by "
            "`nan_policy.price_scale_hits`.  TWO LEVELS (Q-75):")
        log("[price-scale]       (i)  marketCap/equity < %g            -- unconditional floor"
            % PB_ALARM)
        log("[price-scale]       (ii) marketCap/equity < %g AND the share count >= %gx off "
            "the source's" % (PB_WIDE, npol.PRICE_SCALE_WITNESS_FACTOR))
        log("[price-scale]            OWN median -- the corroborated band.  `witness` below "
            "is that multiple, so a")
        log("[price-scale]            source with witness < %g was refused by (i) and one "
            "at or above it may be" % npol.PRICE_SCALE_WITNESS_FACTOR)
        log("[price-scale]            either.  Measured false-refusal rate of band (ii): "
            "11.4%-31.4% -- see `nan_policy`.")
        log("[price-scale]     THOSE ROWS ARE INVISIBLE TO CHECK A BELOW -- its two legs are "
            "the cells that were refused.  The")
        log("[price-scale]     ALARM count is therefore a count of what SURVIVED the refusal, "
            "not of the run's exposure.")
        for _s, _d in sorted(_refused.items(),
                             key=lambda kv: (-kv[1]["rows"], kv[0]))[:25]:
            _wr = _d.get("worst_ratio", float("nan"))
            _wt = _d.get("witness", float("nan"))
            log("[price-scale]     refused %-14s %3d row(s)   worst marketCap/equity %s"
                "   share-count witness %s"
                % (_s, _d["rows"],
                   ("%.5f" % _wr) if _wr == _wr else "n/a (no refusal report passed)",
                   ("%.2fx" % _wt) if _wt == _wt else "n/a"))
        if len(_refused) > 25:
            log("[price-scale]     ... and %d more (full list in the run's input-sanity CSV)."
                % (len(_refused) - 25))
    else:
        log("[price-scale] A0. no price-scale cells were refused upstream on this panel "
            "(either none fired, or the panel")
        log("[price-scale]     predates the 2026-09-01 refusal rule and carries no "
            "`%s` stamp)." % npol.SANITY_REFUSED_COLUMN)
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
        #  COMPUTED, every number of it, from THIS run's panel and THIS run's ranking --
        #  and explicitly NOT CHECKED when the inputs for it were not passed.  See
        #  `_containment_lines` for what the frozen version of this paragraph did.
        for _line in _containment_lines(internal, panel, stage1_scores, shipped_sources,
                                        topn_stage1, panel_label,
                                        extra_flagged=sorted(_refused)):
            log("[price-scale]" + _line)
    else:
        log("[price-scale]    none.  NOTE this is not an all-clear for the price GRID: check "
            "A reads fundamentals only.")
        #  ZERO ALARM DOES NOT MEAN ZERO POPULATION.  When every contaminated source was
        #  refused upstream, `internal` is empty and the containment paragraph used to be
        #  skipped entirely -- silence over a defect that was found.  It runs on the refused
        #  set alone.
        if _refused:
            for _line in _containment_lines(internal, panel, stage1_scores, shipped_sources,
                                            topn_stage1, panel_label,
                                            extra_flagged=sorted(_refused)):
                log("[price-scale]" + _line)

    grid_df, stats = ((pd.DataFrame(), {"reason": "NOT RUN on this panel"})
                      if not run_grid_check
                      else check_synthetic_vs_grid(panel, prices_csv, supp_csv))
    if not run_grid_check:
        #  DECLINED, and it says so.  "check B found nothing" and "check B was not run" are
        #  different statements about the data and must never print as the same line -- the
        #  same distinction `stage1_veto` draws between DECLINED and found-nothing.
        log("[price-scale] B. synthetic-vs-grid: NOT RUN on this panel.  Its names are "
            "largely the ones the")
        log("[price-scale]    price grid does not carry, so 'no disagreement' here would be "
            "an artifact of")
        log("[price-scale]    absence rather than a finding.")
    elif not len(grid_df):
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
        "absent from THIS panel.")
    log("[price-scale]   THE PANEL IS THE SCOPE, and it is the one that moved: run against "
        "the LIVE cdx_df this")
    log("[price-scale]   misses every delisted name, including ATRI -- 0 rows in cdx_df on "
        "both 08-29 and 08-31 --")
    log("[price-scale]   which is the name the stage was built for.  The suite therefore "
        "runs check A a SECOND")
    log("[price-scale]   time over the PIT dead-merged panel, where those names live and "
        "where `bookToPrice`")
    log("[price-scale]   feeds every backtest ranking.  If you are reading only one of the "
        "two passes, you are")
    log("[price-scale]   reading only one population.")

    if out_csv:
        try:
            internal.to_csv(out_csv, index=False)
            log(f"[price-scale] wrote {out_csv}")
        except Exception as e:
            log(f"[price-scale] could not write {out_csv} ({type(e).__name__}: {e})")
    return {"internal": internal, "grid": grid_df, "stats": stats}
