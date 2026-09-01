"""
RETURNS CORE  --  the ticker-agnostic total-return PRIMITIVE (offline, no network).

RETURNS ARE THE PRIMITIVE.  Everything the project measures about performance --
average return, benchmark excess, beat-rate, hit-rate -- is a THIN DERIVED VIEW over
per-ticker total returns produced here.  This module knows nothing about ranking,
"top-20", or the screen; it takes ANY set of tickers and two dates and returns their
total returns.  Callers (the depth-grid, a beat-rate view, a pick-log evaluator, ...)
layer their own aggregation on top.

RETURN DEFINITION (dividend-inclusive):
    total_return = adjClose_eval / adjClose_buy - 1
adjClose is FMP's split+DIVIDEND-adjusted close, so this IS a total return -- no
separate dividend term.  This is the SAME per-name math as beat_rate.compute_returns_table
(`stock_tr = eval/buy - 1`); the two share the definition.  (beat_rate.py still consumes
the bundle two-leg price interface rather than this primitive -- unifying them onto one
source of truth is a low-risk follow-up, deferred so the certified beat-rate is untouched.)

TERMINAL-VALUE POLICY for a ticker MISSING its eval-leg price (delisted-before-eval OR a
price-data coverage gap -- NOT distinguished by the price source):
    PRIMARY : terminal = last available adjClose strictly BEFORE the eval date
              (approximate exit value).  total_return uses this.
    FLOOR   : terminal = 0  ->  total_return_floor = -100% (total loss).
    Both are returned per ticker (total_return / total_return_floor) with terminal_flag
    set, so the sensitivity is a column, not a policy baked into the number.
    MISSING BUY price -> status='no_buy', returns NaN; derived views EXCLUDE it.

ISSUER CONTINUITY IS CHECKED BEFORE THE TERMINAL POLICY (register Q-42, 2026-08-31)
-----------------------------------------------------------------------------------
The terminal policy above answers "the eval leg is missing, what is this worth?".  It is the
right question only when we do not know WHY the leg is missing.  For a LISTING-LINE
DISCONTINUITY -- re-domicile, ticker change, exchange move, share-class reorganisation,
preferred-series redemption -- we sometimes DO know, and then both readings are wrong in the
same direction: the FLOOR books a total capital loss for a company that is demonstrably
still trading, and PRIMARY marks the position at a price one or two anchor-years stale.

So `compute_returns` now consults `issuer_continuity` FIRST, and only for tickers whose eval
leg is missing:

    status='ok'            the successor line prices BOTH chartered anchors, so the position
                           is followed onto it and the return is that line's OWN two-leg
                           ratio -- never a splice of the old buy leg against the new eval
                           leg, which would book an unquantified FX return.  This IS an
                           observation of the window; `target_clauses.measured` counts it.
    status='indeterminate' the line is a KNOWN discontinuity but no successor return is
                           measurable.  total_return AND total_return_floor are NaN: the
                           -100% is refuted by the evidence in the map row, and the stale
                           substituted price was never an observation.  UNMEASURED, and
                           counted as its own bucket so it is never silently dropped.

`terminal_flag` is True for 'indeterminate' as well as 'terminal', so every existing caller
that branches on `terminal_flag` to apply a missing-value policy keeps treating it as
missing rather than reading a number off it.  The `continuity` column carries the audit
string (which line, which successor, which event) for the rows where the map fired, and is
"" everywhere else.  Pass `continuity={}` to reproduce the pre-Q-42 behaviour exactly.

PRICE SOURCE is injected (dependency-inversion) so the primitive is reusable against any
adjusted-close provider.  PriceSource here reads the year-end anchor grid in
real_prices.csv; buy/eval dates must be anchors on that grid (exact match).

THE HOLIDAY UNION IS PER-VENUE AT EVERY ANCHOR, NOT AT ONE (generalised 2026-08-22)
----------------------------------------------------------------------------------
`_merge_supplementary` used to do prefer-anchor / fill-from-neighbour for the single
anchor 2025-12-31, filling from 2025-12-30 because ".DE/.T/.KS ... only report on
2025-12-30".  That calendar collision is not a property of 2025.  Measured on the
repo-local dev grid, `.DE`/`.ST`/`.IC` are unpriceable at 2018-12-31, 2019-12-31,
2021-12-31 AND 2024-12-31, and `.KS`/`.KQ` additionally at 2023-12-29 -- the same
December-31-is-a-venue-holiday problem, unfixed at every earlier anchor.

So the union is now a general FILL LAYER (`_fill_from_neighbour_dates`) applied at every
anchor: a symbol with no price AT the anchor takes the newest `date_actual` at or before
the anchor within `fill_window_days`.  Two properties are load-bearing:

  * IT IS PER SYMBOL, therefore per venue.  It is NOT a global "did this date return
    anything" test.  `fetch_prices.run_bulk`'s step-back is exactly that global test and
    it is why six venues silently vanished: ~34,000 US rows make a body non-empty, the
    step-back never fires, and Paris/Korea/Oslo are absent with nothing complaining.
    Filling per symbol cannot repeat that shape -- a venue that is missing is missing
    name by name, and each name is looked up on its own.
  * IT ONLY EVER ADDS.  A price already present at the anchor is never overwritten, and
    a `date_actual` AFTER the anchor is never eligible (that would be look-ahead).  So
    coverage is monotonically >= the previous rule and no existing leg moves.

WHAT IT CANNOT DO, AND THIS IS THE HONEST HEADLINE.  A fill needs a neighbouring body to
fill FROM, and neither grid on disk has one at the damaged anchors: the run machine's
grid resolved each anchor to exactly ONE `date_actual`, and the dev grid's 2018/2019/2021
anchors likewise.  So this change is measurably INERT on both grids today -- verified,
not assumed.  It becomes a coverage gain only once a fetch supplies the adjacent trading
day, and `run_bulk` will not supply it while its step-back is global.  The mechanism is
now correct; the data is still missing.
"""

import os

import numpy as np
import pandas as pd

import benchmark_loader as bl
import issuer_continuity as icont

# Year-end anchors present in the standard real_prices.csv, chronological.
DEFAULT_ANCHORS = ["2018-12-31", "2019-12-31", "2020-12-31",
                   "2021-12-31", "2022-12-30", "2023-12-29", "2024-12-31",
                   "2025-12-31"]

BENCHMARK_SYMBOL = "URTH"
BENCHMARK_VARIANT = "URTH (iShares MSCI World ETF) adjClose, TR-proxy for MSCI World Net TR USD"

RETURNS_COLS = ["ticker", "buy_adjClose", "eval_adjClose", "terminal_adjClose",
                "total_return", "total_return_floor", "terminal_flag", "status",
                "continuity"]

#  The status vocabulary, named once so callers stop spelling the strings.  'indeterminate'
#  is the Q-42 addition and it is NOT a sub-kind of 'terminal': every place that decides what
#  is MEASURED must exclude both, and every place that partitions coverage must count them
#  separately, because "we do not know why the leg is missing" and "we know the line changed
#  and cannot price the successor" are different facts about the grid.
STATUS_OK = "ok"
STATUS_TERMINAL = "terminal"
STATUS_INDETERMINATE = icont.STATUS_INDETERMINATE
STATUS_NO_BUY = "no_buy"

#  Statuses that are NOT an observation of the window.  One definition so `target_clauses`
#  and the grid cannot drift apart on it.
UNMEASURED_STATUSES = (STATUS_TERMINAL, STATUS_INDETERMINATE)

#  How far back the per-venue holiday fill may reach for a symbol unpriced AT an anchor.
#
#  4 days is `fetch_prices.py --max-lookback`'s default, deliberately: the fill can only
#  ever consume what the fetch was allowed to pull, so a wider window here would describe
#  dates that cannot be in the file.  It also covers every observed collision -- the
#  2025-12-31 -> 2025-12-30 case is 1 day, and a Dec-31 Thursday holiday reaches back to
#  the preceding Friday in 3.
#
#  The window is bounded well below the ~365-day anchor spacing, so a fill can never reach
#  into a neighbouring ANCHOR's body and quietly price a name a year early.
DEFAULT_FILL_WINDOW_DAYS = 4


class PriceSource:
    """Adjusted-close access over an anchor grid (default: real_prices.csv year-ends).

    adjClose is split+dividend adjusted => returns built on it are dividend-inclusive.
    Exposes exactly what the returns primitive needs:
        price(ticker, anchor)         -> float | None
        last_before(ticker, anchor)   -> (anchor, price) | None   (strictly earlier, latest)
    """

    #  The on-disk schema.  `date_actual` is the day the venue actually settled and is what
    #  makes the fill layer possible; the previous version dropped it at read time, which is
    #  why the holiday union had to be spelled out as two hardcoded 2025 dates.
    PRICE_COLS = ["date_requested", "date_actual", "symbol", "adjClose"]

    def __init__(self, prices_csv, anchors=None, supp_csv=None,
                 fill_window_days=DEFAULT_FILL_WINDOW_DAYS):
        """prices_csv : the canonical year-end anchor grid (real_prices.csv), read-only.
        supp_csv    : OPTIONAL supplementary price CSV (or an iterable of them) merged
                      in-memory; the canonical file is never mutated.  This is how the
                      2025-12-31 anchor arrives from real_prices_2025.csv.
        fill_window_days : how far back the per-venue holiday fill may reach (see
                      DEFAULT_FILL_WINDOW_DAYS).  0 disables the fill entirely, which is
                      the switch to reproduce the pre-generalisation behaviour minus the
                      2025 special case.

        TWO LAYERS, IN THIS ORDER, and the order is the whole rule:
          1. ANCHOR LAYER -- rows whose `date_requested` is an anchor, keep-first.  This is
             the certified selection (the `date_requested` axis, C1 2026-08-22) and it is
             UNCHANGED.  The canonical file is read before any supplementary, so on a tie
             the canonical wins.
          2. FILL LAYER -- for a symbol with nothing from layer 1 at an anchor, the newest
             `date_actual` at or before that anchor within `fill_window_days`.  Per symbol,
             every anchor, add-only.  See the module docstring.
        """
        self.anchors = list(anchors) if anchors is not None else list(DEFAULT_ANCHORS)
        self._idx = {a: i for i, a in enumerate(self.anchors)}
        self.fill_window_days = int(fill_window_days or 0)

        rows = pd.concat([self._read_price_csv(pth)
                          for pth in self._supp_paths(prices_csv, supp_csv)],
                         ignore_index=True)

        lut = {}
        anchored = rows[rows["date_requested"].isin(self.anchors)]
        for r in anchored.itertuples(index=False):
            key = (r.symbol, r.date_requested)
            if key not in lut:  # keep first occurrence
                lut[key] = float(r.adjClose)
        self._n_anchor_layer = len(lut)
        self._fill_lag = self._fill_from_neighbour_dates(lut, rows)
        self._lut = lut

    @staticmethod
    def _supp_paths(prices_csv, supp_csv):
        """[canonical] + supplementary path(s).  `supp_csv` may be None, a path, or an
        iterable of paths -- an iterable because "the files the fill may draw from" is
        naturally plural once the union is not hardcoded to one supplement."""
        paths = [prices_csv]
        if supp_csv is None:
            return paths
        if isinstance(supp_csv, (str, bytes, os.PathLike)):
            return paths + [supp_csv]
        return paths + [p for p in supp_csv if p]

    @classmethod
    def _read_price_csv(cls, path):
        """One price CSV, coerced to the schema.  `date_actual` is tolerated as absent so a
        hand-written or legacy file without it still loads -- such rows simply cannot serve
        as fill sources, which is the correct degradation rather than a crash."""
        df = pd.read_csv(path)
        missing = [c for c in ("date_requested", "symbol", "adjClose")
                   if c not in df.columns]
        if missing:
            raise KeyError("price csv %r missing required column(s): %s"
                           % (path, missing))
        if "date_actual" not in df.columns:
            df = df.assign(date_actual=pd.NA)
        return df.loc[:, cls.PRICE_COLS]

    def _fill_from_neighbour_dates(self, lut, rows):
        """THE GENERALISED HOLIDAY UNION.  Fill, per symbol, per anchor, add-only.

        For each anchor A and each symbol with no price at A from the anchor layer, take the
        row with the LARGEST `date_actual` in [A - fill_window_days, A] -- inclusive at both
        ends, see the comment on `lo` below.  Returns
        {(symbol, anchor): lag_days} for the fills that were applied, so the staleness the
        fill introduces is reportable rather than invisible -- a filled leg is a price from a
        few days before the anchor being used AS the anchor price, and that timing error is
        real even though it is small.

        ONE DELIBERATE DETAIL: a NaN `adjClose` on the closest day does NOT block the fill --
        the next-closest day in the window is used instead.  (A first draft reduced to one row
        per symbol before the NaN check, so a NaN on the newest day left the name unpriced
        even though an older valid price sat in the window.)  Neither price file on disk
        carries a NaN adjClose, so this changes nothing today; filling from the older valid
        price is simply the better rule.

        WHAT THIS CANNOT DETECT.  It cannot tell a genuine venue holiday from a truncated
        body: if the fetch dropped `.PA` at every date in the window, `.PA` stays unpriced
        and nothing here says whether the exchange was shut or the payload was short.  It
        also cannot notice that a filled price is materially different from what the anchor
        close would have been.  Both of those are fetch-side questions.
        """
        fill_lag = {}
        if self.fill_window_days <= 0 or rows.empty:
            return fill_lag
        #  GROUPED BY SETTLEMENT DAY ONCE rather than a boolean mask per anchor: a price file
        #  carries a HANDFUL of distinct `date_actual` values (10 on the dev grid, 7 on the
        #  run machine's) against hundreds of thousands of rows, so eight full-frame masks
        #  were paying repeatedly for information that fits in a dict.
        #
        #  BE HONEST ABOUT WHAT THIS BOUGHT: almost nothing.  Measured on the dev grid,
        #  PriceSource construction is 0.77s against HEAD's 0.41s and the grouping did not
        #  move it -- the masks were 0.007s and the cost is the ~578,000-row Python loop
        #  below, because every `date_actual` on these grids happens to fall within 4 days of
        #  some anchor, so the window excludes nothing.  The grouping is kept because it is
        #  the clearer expression, NOT because it is a measured win.
        #
        #  THE +0.36s IS NOT WORTH OPTIMISING and is deliberately left: the constructor runs
        #  once per pipeline run and a few dozen times across the test suite, so the total is
        #  seconds against a ~27-minute suite.  Recorded so nobody has to re-measure it.
        actual = pd.to_datetime(rows["date_actual"], errors="coerce")
        by_day = {day: idx for day, idx in rows.groupby(actual, sort=True).groups.items()}
        for a in self.anchors:
            A = pd.Timestamp(a)
            #  INCLUSIVE at both ends, matching `fetch_prices --max-lookback`: that loop is
            #  `range(max_lookback + 1)`, so back=N is a date the fetch may have pulled and
            #  a window that excluded it would be describing a different lookback.
            lo = A - pd.Timedelta(days=self.fill_window_days)
            #  NEWEST SETTLEMENT DAY FIRST, so a symbol that traded on several days in the
            #  window is filled from the day CLOSEST to the anchor; `key in lut` then makes
            #  the first insertion win.  Within one day the order is file order (canonical
            #  before supplementary), matching the anchor layer's keep-first.
            for day in sorted((d for d in by_day if lo <= d <= A), reverse=True):
                lag = int((A - day).days)
                for r in rows.loc[by_day[day]].itertuples(index=False):
                    key = (r.symbol, a)
                    if key in lut:
                        continue
                    px = float(r.adjClose)
                    if px != px:        # NaN in the file is not a price
                        continue
                    lut[key] = px
                    fill_lag[key] = lag
        return fill_lag

    def fill_report(self):
        """Per anchor: how many legs came from the fill layer and how stale they are.

        Reported, never asserted.  `n_filled == 0` at every anchor is the CORRECT reading on
        both grids currently on disk -- there is no neighbouring body to fill from -- and it
        is precisely why this generalisation is a mechanism fix rather than a coverage gain.
        """
        rows = []
        for a in self.anchors:
            lags = [v for (_s, an), v in self._fill_lag.items() if an == a]
            rows.append({"anchor": a,
                         "n_priced": sum(1 for (_s, an) in self._lut if an == a),
                         "n_filled": len(lags),
                         "max_fill_lag_days": max(lags) if lags else 0})
        return pd.DataFrame(rows)

    def price(self, ticker, anchor):
        return self._lut.get((ticker, anchor))

    def last_before(self, ticker, anchor):
        """Latest (anchor, price) strictly before `anchor` where the ticker has a price."""
        j = self._idx.get(anchor)
        if j is None:
            return None
        for k in range(j - 1, -1, -1):
            a = self.anchors[k]
            p = self._lut.get((ticker, a))
            if p is not None:
                return a, p
        return None

    def benchmark_series(self, symbol=BENCHMARK_SYMBOL):
        """A date->level pd.Series for `symbol` over the anchor grid, for
        benchmark_loader.window_return."""
        rows = [(a, self._lut[(symbol, a)]) for a in self.anchors
                if (symbol, a) in self._lut]
        if not rows:
            raise RuntimeError(f"benchmark symbol {symbol!r} absent from price source")
        s = pd.Series({pd.Timestamp(a): lvl for a, lvl in rows}).sort_index()
        return s


# --------------------------------------------------------------------------- #
#  THE PRIMITIVE                                                              #
# --------------------------------------------------------------------------- #
def compute_returns(tickers, buy_date, eval_date, price_source, continuity=None):
    """Per-ticker total return over [buy_date, eval_date].  TICKER-AGNOSTIC.

    Parameters
    ----------
    tickers      : iterable of ticker/source strings (input ORDER is preserved in the
                   output, so a caller can slice `.head(N)` for a top-N view).
    buy_date     : anchor at which the position is opened (adjClose_buy).
    eval_date    : anchor at which the position is valued (adjClose_eval).
    price_source : a PriceSource (or any object with .price / .last_before).
    continuity   : the issuer-continuity map (see the module docstring and
                   `issuer_continuity`).  None -> the module's own validated table, which
                   is the DEFAULT and the fixed behaviour.  `{}` -> the map is off and the
                   pre-Q-42 terminal policy is reproduced bit-for-bit; that switch exists so
                   a test can pin the defect rather than describe it.

    Returns
    -------
    DataFrame[RETURNS_COLS], one row per input ticker (order preserved):
      total_return       : PRIMARY policy -- real eval leg, or (eval missing, line followed
                           through a discontinuity) the successor line's own two-leg return,
                           or (eval missing, no continuity) the last available adjClose
                           before eval as terminal.  NaN if no buy leg, and NaN for an
                           'indeterminate' pick.
      total_return_floor : FLOOR policy   -- eval missing => -100%.  Equals total_return
                           when the eval leg is present or the line was followed.  NaN if no
                           buy leg, and NaN for an 'indeterminate' pick -- a -100% there
                           would contradict the evidence that put the pick in that bucket.
      terminal_flag      : True iff no eval leg was observed for THIS pick and a policy had
                           to be applied -- 'terminal' and 'indeterminate' alike, so a caller
                           branching on it keeps treating both as missing.
      status             : 'ok' | 'terminal' | 'indeterminate' | 'no_buy'.
      continuity         : audit string when the continuity map fired; "" otherwise.

    ORDER OF THE THREE TESTS IS LOAD-BEARING.  Buy leg, then REAL eval leg, then continuity.
    A line that prices both chartered anchors is measured on its own legs and the map is
    never consulted -- so a map row can never override a real observation, and a future price
    refetch that fills VMD.TO's 2024 anchor silently takes precedence over the map rather
    than fighting it.
    """
    cmap = icont.load() if continuity is None else dict(continuity)
    rows = []
    for t in tickers:
        p_buy = price_source.price(t, buy_date)
        if p_buy is None or p_buy == 0 or (isinstance(p_buy, float) and np.isnan(p_buy)):
            rows.append((t, np.nan, np.nan, np.nan, np.nan, np.nan, False,
                         STATUS_NO_BUY, ""))
            continue
        p_eval = price_source.price(t, eval_date)
        if p_eval is not None and not (isinstance(p_eval, float) and np.isnan(p_eval)):
            r = p_eval / p_buy - 1.0
            rows.append((t, p_buy, p_eval, p_eval, r, r, False, STATUS_OK, ""))
            continue
        # eval leg missing -> is this a KNOWN listing-line discontinuity?
        cont = icont.resolve(t, buy_date, eval_date, price_source, cmap) if cmap else None
        if cont is not None and cont["kind"] == "continued":
            #  BOTH legs are the SUCCESSOR's, so `eval/buy - 1 == total_return` still holds
            #  row-wise and anything recomputing the ratio from the columns gets the same
            #  number.  The pick's own buy price is recoverable from the grid and from the
            #  map row; what must not happen is the two currencies meeting in one ratio.
            r = cont["total_return"]
            rows.append((t, cont["buy_px"], cont["eval_px"], cont["eval_px"], r, r, False,
                         STATUS_OK, cont["note"]))
            continue
        if cont is not None:
            rows.append((t, p_buy, np.nan, np.nan, np.nan, np.nan, True,
                         STATUS_INDETERMINATE, cont["note"]))
            continue
        # unexplained missing eval leg -> terminal policy, UNCHANGED
        lb = price_source.last_before(t, eval_date)
        terminal = lb[1] if lb is not None else p_buy  # buy leg guaranteed present
        r_primary = terminal / p_buy - 1.0
        rows.append((t, p_buy, np.nan, terminal, r_primary, -1.0, True,
                     STATUS_TERMINAL, ""))
    return pd.DataFrame(rows, columns=RETURNS_COLS)


# --------------------------------------------------------------------------- #
#  DERIVED VIEWS (all computed FROM the per-ticker returns above)             #
# --------------------------------------------------------------------------- #
def _ret_col(floor):
    return "total_return_floor" if floor else "total_return"


def included(returns_df):
    """Rows with a valid buy leg (the denominator for every derived view)."""
    return returns_df[returns_df["status"] != "no_buy"]


def average_return(returns_df, floor=False):
    """Equal-weight average total return over tickers that had a buy leg AND a return.

    THE SECOND HALF OF THAT SENTENCE IS NEW AND IT IS A DENOMINATOR CHANGE, so it is spelled
    out rather than left to pandas.  Before Q-42 every row with a buy leg carried a number
    under both policies, so "had a buy leg" and "has a return" were the same set and the
    distinction could not bite.  An 'indeterminate' pick has a buy leg and NO return under
    either policy, and `Series.mean()` would have skipped it silently -- an average whose
    denominator disagrees with the `n_included` printed beside it.  `n_averaged` below is
    what a caller must print next to this number; `counts()` gives the buckets.
    """
    r = averaged_returns(returns_df, floor=floor)
    return float(r.mean()) if len(r) else float("nan")


def averaged_returns(returns_df, floor=False):
    """The exact rows `average_return` averages -- buy leg present AND a non-NaN return."""
    inc = included(returns_df)
    if not len(inc):
        return pd.Series(dtype=float)
    return pd.to_numeric(inc[_ret_col(floor)], errors="coerce").dropna()


def excess_return(returns_df, benchmark_ret, floor=False):
    """average_return - benchmark_ret."""
    avg = average_return(returns_df, floor=floor)
    return (avg - benchmark_ret) if avg == avg else float("nan")


def counts(returns_df):
    """(n_included, n_terminal, n_indeterminate, n_no_buy) for a ticker set.

    `n_terminal` stays EXACTLY `status == 'terminal'` -- it does not absorb the Q-42
    'indeterminate' bucket.  Folding them together would have been the smaller diff and it
    would have hidden the whole finding: an unexplained missing leg and an identified
    line-discontinuity are different facts, and a reader who cannot see the split cannot
    tell whether the continuity map is doing anything.
    """
    return {
        "n_included": int((returns_df["status"] != STATUS_NO_BUY).sum()),
        "n_terminal": int((returns_df["status"] == STATUS_TERMINAL).sum()),
        "n_indeterminate": int((returns_df["status"] == STATUS_INDETERMINATE).sum()),
        "n_no_buy": int((returns_df["status"] == STATUS_NO_BUY).sum()),
    }


def continuity_counts(returns_df):
    """What the issuer-continuity map DID to this returns table.

    n_reattached    picks whose eval leg was missing and whose position was followed onto a
                    successor line -- these are MEASURED, on the successor's own two legs.
    n_indeterminate picks with an identified discontinuity and no measurable successor
                    return -- unmeasured, and explicitly NOT booked at -100%.
    n_terminal      picks with a missing eval leg and NO map row -- the unchanged terminal
                    policy, still carrying a stale PRIMARY and a -100% FLOOR.

    The third number is the one to read first: it is the population the map does NOT cover,
    i.e. how much of the defect is still live.
    """
    st = returns_df["status"].astype(str)
    cont = returns_df["continuity"].astype(str) if "continuity" in returns_df else None
    reattached = int(((st == STATUS_OK) & (cont != "")).sum()) if cont is not None else 0
    return {
        "n_reattached": reattached,
        "n_indeterminate": int((st == STATUS_INDETERMINATE).sum()),
        "n_terminal": int((st == STATUS_TERMINAL).sum()),
    }


def continuity_report_line(returns_df, where=""):
    """ONE line saying what the map did, for a run log.  Printed even when it did nothing --
    "the map found nothing here" and "the map never ran" must not look alike."""
    c = continuity_counts(returns_df)
    tag = (" %s" % where) if where else ""
    return ("[issuer-continuity]%s re-attached=%d  indeterminate=%d  "
            "booked terminal (no map row, still -100%% under FLOOR)=%d"
            % (tag, c["n_reattached"], c["n_indeterminate"], c["n_terminal"]))


def benchmark_return(price_source, buy_date, eval_date, symbol=BENCHMARK_SYMBOL,
                     require_exact=False):
    """Benchmark total return over the window, via benchmark_loader.window_return.

    require_exact (LOW-2): when True, both anchors must be exact index entries in
    the benchmark series -- window_return raises rather than forward-filling a
    stale level. Default False keeps the certified path bit-for-bit. The rebalance
    engine / tuner pass True so a missing benchmark anchor fails loudly."""
    return bl.window_return(price_source.benchmark_series(symbol), buy_date,
                            eval_date, require_exact=require_exact)


def benchmark_return_or_none(price_source, buy_date, eval_date, window_id="",
                             where="", symbol=BENCHMARK_SYMBOL, log=None):
    """`benchmark_return(require_exact=True)`, but a missing anchor costs ONE WINDOW.

    THE COMPOSITION DEFECT THIS EXISTS FOR.  `require_exact=True` RAISES on purpose, so a
    missing benchmark anchor can never be papered over with a forward-filled stale level.
    Every caller then put that bare call inside a PER-ANCHOR LOOP, and every one of those
    loops runs under a stage guard that swallows exceptions (`pipeline_analysis._run_stage`
    catches everything and returns None so a stage "can never crash the run").  The two
    correct-in-isolation decisions compose into the wrong thing: ONE unpriceable anchor did
    not cost its own window, it destroyed the WHOLE stage -- the two-clause target, the
    beat-rate table, the per-band split, the gate attribution -- and took the perfectly
    measurable windows down with it.  A coverage defect erasing a filter measurement, which
    is precisely what `target_clauses` spends three docstring paragraphs preventing one layer
    up, where nothing was preventing it.

    THE STRICTNESS IS KEPT.  `require_exact=True` is still passed; it was never the problem,
    the blast radius was.  A window with no exact benchmark leg is SAID and skipped, and the
    remaining windows still report.

    Returns the benchmark return, or None.  `None` must be treated as "skip this window" --
    NEVER coerced to 0.0, which would silently score every pick in the window against a flat
    benchmark and read as a result.
    """
    log = print if log is None else log
    try:
        return benchmark_return(price_source, buy_date, eval_date, symbol=symbol,
                                require_exact=True)
    except Exception as e:
        tag = ("%s " % window_id) if window_id else ""
        loc = (" in %s" % where) if where else ""
        log("!!! %s(%s->%s) SKIPPED%s: no exact benchmark leg (%s: %s)."
            % (tag, buy_date, eval_date, loc, type(e).__name__, e))
        log("!!! This window contributes NOTHING here -- it is not a zero and not a failure "
            "of the filter.")
        return None


def beat_rate(returns_df, benchmark_ret, threshold=0.10, missing="fail", floor=False):
    """Beat-rate DERIVED from per-ticker returns: share of names whose excess over the
    benchmark is >= threshold.  Mirrors beat_rate.py's missing-eval policy so it is a
    faithful derived view (no_buy excluded; 'fail' => missing-eval counts as not beating;
    'drop' => excluded; 'zero' => treat stock return as 0).

    AN 'indeterminate' PICK TAKES THE MISSING BRANCH, not a NaN comparison.  It carries
    `terminal_flag=True` precisely so it lands here and is governed by the caller's stated
    missing-policy; without that flag `(nan - bench) >= threshold` would evaluate False and
    the pick would be scored a silent miss under EVERY policy, including 'drop'.
    """
    inc = included(returns_df)
    if benchmark_ret != benchmark_ret:
        return float("nan"), 0
    flags = []
    for _, row in inc.iterrows():
        if row["terminal_flag"]:
            if missing == "drop":
                continue
            if missing == "zero":
                flags.append((0.0 - benchmark_ret) >= threshold)
                continue
            if missing == "fail":
                flags.append(False)
                continue
        r = row[_ret_col(floor)]
        flags.append((r - benchmark_ret) >= threshold)
    if not flags:
        return float("nan"), 0
    return float(np.mean(flags)), len(flags)
