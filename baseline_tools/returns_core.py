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

PRICE SOURCE is injected (dependency-inversion) so the primitive is reusable against any
adjusted-close provider.  PriceSource here reads the year-end anchor grid in
real_prices.csv; buy/eval dates must be anchors on that grid (exact match).
"""

import os

import numpy as np
import pandas as pd

import benchmark_loader as bl

# Year-end anchors present in the standard real_prices.csv, chronological.
DEFAULT_ANCHORS = ["2018-12-31", "2019-12-31", "2020-12-31",
                   "2021-12-31", "2022-12-30", "2023-12-29", "2024-12-31",
                   "2025-12-31"]

BENCHMARK_SYMBOL = "URTH"
BENCHMARK_VARIANT = "URTH (iShares MSCI World ETF) adjClose, TR-proxy for MSCI World Net TR USD"

RETURNS_COLS = ["ticker", "buy_adjClose", "eval_adjClose", "terminal_adjClose",
                "total_return", "total_return_floor", "terminal_flag", "status"]


class PriceSource:
    """Adjusted-close access over an anchor grid (default: real_prices.csv year-ends).

    adjClose is split+dividend adjusted => returns built on it are dividend-inclusive.
    Exposes exactly what the returns primitive needs:
        price(ticker, anchor)         -> float | None
        last_before(ticker, anchor)   -> (anchor, price) | None   (strictly earlier, latest)
    """

    def __init__(self, prices_csv, anchors=None, supp_csv=None):
        """prices_csv : the canonical year-end anchor grid (real_prices.csv), read-only.
        supp_csv    : OPTIONAL supplementary price CSV merged in-memory (canonical file
                      is never mutated).  Used to add the 2025-12-31 anchor from
                      real_prices_2025.csv without touching the canonical grid -- see
                      _merge_supplementary for the union rule.
        """
        self.anchors = list(anchors) if anchors is not None else list(DEFAULT_ANCHORS)
        self._idx = {a: i for i, a in enumerate(self.anchors)}
        rp = pd.read_csv(prices_csv, usecols=["date_requested", "symbol", "adjClose"])
        rp = rp[rp["date_requested"].isin(self.anchors)]
        lut = {}
        for r in rp.itertuples(index=False):
            key = (r.symbol, r.date_requested)
            if key not in lut:  # keep first occurrence
                lut[key] = float(r.adjClose)
        if supp_csv is not None:
            self._merge_supplementary(lut, supp_csv)
        self._lut = lut

    @staticmethod
    def _merge_supplementary(lut, supp_csv, anchor="2025-12-31", fill_from="2025-12-30"):
        """Merge a supplementary price CSV into `lut` under the single anchor
        date_requested == `anchor`, WITHOUT modifying any file on disk.

        UNION per symbol: PREFER the `anchor` (2025-12-31) adjClose; FILL from
        `fill_from` (2025-12-30) for symbols with no `anchor` row -- needed because
        exchanges closed on 2025-12-31 (.DE/.T/.KS ...) only report on 2025-12-30 and
        the universe includes such names.  The merged rows carry date_requested ==
        `anchor` so they survive PriceSource's anchor filter.
        """
        sp = pd.read_csv(supp_csv, usecols=["date_requested", "symbol", "adjClose"])
        preferred, fill = {}, {}
        for r in sp.itertuples(index=False):
            if r.date_requested == anchor:
                if r.symbol not in preferred:  # keep first occurrence
                    preferred[r.symbol] = float(r.adjClose)
            elif r.date_requested == fill_from:
                if r.symbol not in fill:
                    fill[r.symbol] = float(r.adjClose)
        merged = dict(fill)
        merged.update(preferred)  # 2025-12-31 overrides the 2025-12-30 fill
        for sym, px in merged.items():
            lut[(sym, anchor)] = px

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
def compute_returns(tickers, buy_date, eval_date, price_source):
    """Per-ticker total return over [buy_date, eval_date].  TICKER-AGNOSTIC.

    Parameters
    ----------
    tickers      : iterable of ticker/source strings (input ORDER is preserved in the
                   output, so a caller can slice `.head(N)` for a top-N view).
    buy_date     : anchor at which the position is opened (adjClose_buy).
    eval_date    : anchor at which the position is valued (adjClose_eval).
    price_source : a PriceSource (or any object with .price / .last_before).

    Returns
    -------
    DataFrame[RETURNS_COLS], one row per input ticker (order preserved):
      total_return       : PRIMARY policy -- real eval leg, or (eval missing) the last
                           available adjClose before eval as terminal.  NaN if no buy leg.
      total_return_floor : FLOOR policy   -- eval missing => -100%.  Equals total_return
                           when the eval leg is present.  NaN if no buy leg.
      terminal_flag      : True iff the eval leg was missing and a terminal was applied.
      status             : 'ok' | 'terminal' | 'no_buy'.
    """
    rows = []
    for t in tickers:
        p_buy = price_source.price(t, buy_date)
        if p_buy is None or p_buy == 0 or (isinstance(p_buy, float) and np.isnan(p_buy)):
            rows.append((t, np.nan, np.nan, np.nan, np.nan, np.nan, False, "no_buy"))
            continue
        p_eval = price_source.price(t, eval_date)
        if p_eval is not None and not (isinstance(p_eval, float) and np.isnan(p_eval)):
            r = p_eval / p_buy - 1.0
            rows.append((t, p_buy, p_eval, p_eval, r, r, False, "ok"))
            continue
        # eval leg missing -> terminal policy
        lb = price_source.last_before(t, eval_date)
        terminal = lb[1] if lb is not None else p_buy  # buy leg guaranteed present
        r_primary = terminal / p_buy - 1.0
        rows.append((t, p_buy, np.nan, terminal, r_primary, -1.0, True, "terminal"))
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
    """Equal-weight average total return over tickers that had a buy leg."""
    inc = included(returns_df)
    return float(inc[_ret_col(floor)].mean()) if len(inc) else float("nan")


def excess_return(returns_df, benchmark_ret, floor=False):
    """average_return - benchmark_ret."""
    avg = average_return(returns_df, floor=floor)
    return (avg - benchmark_ret) if avg == avg else float("nan")


def counts(returns_df):
    """(n_included, n_terminal, n_no_buy) for a ticker set."""
    return {
        "n_included": int((returns_df["status"] != "no_buy").sum()),
        "n_terminal": int((returns_df["status"] == "terminal").sum()),
        "n_no_buy": int((returns_df["status"] == "no_buy").sum()),
    }


def benchmark_return(price_source, buy_date, eval_date, symbol=BENCHMARK_SYMBOL,
                     require_exact=False):
    """Benchmark total return over the window, via benchmark_loader.window_return.

    require_exact (LOW-2): when True, both anchors must be exact index entries in
    the benchmark series -- window_return raises rather than forward-filling a
    stale level. Default False keeps the certified path bit-for-bit. The rebalance
    engine / tuner pass True so a missing benchmark anchor fails loudly."""
    return bl.window_return(price_source.benchmark_series(symbol), buy_date,
                            eval_date, require_exact=require_exact)


def beat_rate(returns_df, benchmark_ret, threshold=0.10, missing="fail", floor=False):
    """Beat-rate DERIVED from per-ticker returns: share of names whose excess over the
    benchmark is >= threshold.  Mirrors beat_rate.py's missing-eval policy so it is a
    faithful derived view (no_buy excluded; 'fail' => missing-eval counts as not beating;
    'drop' => excluded; 'zero' => treat stock return as 0).
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
