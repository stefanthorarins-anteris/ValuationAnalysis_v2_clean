"""
STRATEGY-B REBALANCE ENGINE  --  rebalance-every-k on top of returns_core (offline).

The CEO's two strategies unified into ONE swept parameter k (2026-07-14):
  * Strategy A (buy-and-hold) = the DEGENERATE case k == horizon: rank ONCE at the
    buy anchor, hold to eval.  Reproduces the certified 36mo beat-rate harness when
    tx_cost_bps == 0 (proven in test_rebalance_engine.py).
  * Strategy B (reoptimize every k) = rank at each rebalance date, take top-N, hold k,
    then re-rank and rebalance; CHAIN the sub-period returns across the horizon.
The "OPTIMAL REOPTIMIZATION WINDOW" is argmax over k of the target beat-rate NET of
the turnover/transaction cost k incurs.

ARCHITECTURE (returns-first, dependency-inverted, mirrors depth_horizon_grid):
  * RETURNS ARE THE PRIMITIVE.  All per-name return math is returns_core.compute_returns.
  * BEAT-RATE IS NOT RE-IMPLEMENTED.  Every beat-rate flows through the certified
    rc.beat_rate primitive (this folds in the METHOD hardening item -- unify onto the
    primitive).  For a chained (rebalanced) horizon we build a synthetic per-name
    returns table with the RETURNS_COLS schema whose total_return column carries the
    chained sleeve return, then hand it to rc.beat_rate exactly as A does.
  * RANKING IS INJECTED.  The engine never imports the scorer; the caller passes
    rank_fn(as_of_date) -> ordered list of `source` strings (top of the PIT re-rank).
    stage2_pit + the current weight set live in the caller (tuner / driver), so the
    engine is reusable and cheap to test with a stub ranker.

ANCHOR GRID (load-bearing data constraint, surfaced -- NOT a code limit):
  The engine schedules rebalances on an ANCHOR GRID (returns_core.DEFAULT_ANCHORS =
  YEAR-ENDS only).  k_months and horizon_m are converted to a whole number of anchor
  steps via `anchor_step_months` (12 on the annual grid).  A rebalance point that does
  not land on an existing anchor RAISES (no fake prices).  Consequence on the current
  data: over a 36mo horizon only k in {12, 36} months divide onto the annual grid and
  are evaluable; k in {3, 6} months REQUIRE a quarterly price grid that real_prices.csv
  does not contain -- they raise a clear UnevaluableK error.  Drop in a quarterly grid
  (anchor_step_months=3) and 3mo/6mo resolve with no code change.

TRANSACTION COST MODEL (parameterised; default 20bps one-way; set 0 for frictionless):
  Per-period portfolio cost fraction c_p applied UNIFORMLY across the equal-weight book
  (a name's weight is 1/N):
    * formation (p == 0): c_0 = bps/1e4               (establish the full book -- buy N)
    * rebalance (p >= 1): c_p = (|entered|+|exited|)/N * bps/1e4   (round-trip the churn)
  Each sub-period gross return is netted: (1 + r_period) * (1 - c_p).  At bps == 0 every
  c_p == 0, so the NET chain equals the frictionless chain EXACTLY -> Strategy A with
  tx_cost_bps == 0 reproduces the certified number regardless of the cost policy.
  Uniform-across-sleeves is a deliberate modelling choice for an equal-weight book
  (documented, not hidden): the set-based turnover is economically correct; spreading
  its drag equally over the 1/N-weighted names is the equal-weight approximation.

No network I/O; never prints any api_key.
"""

import numpy as np
import pandas as pd

import returns_core as rc


class UnevaluableK(Exception):
    """Raised when a requested reoptimization window k (or horizon) cannot be placed
    on the available price anchor grid (e.g. 3mo/6mo on an annual-only grid)."""


# --------------------------------------------------------------------------- #
#  Anchor-grid scheduling                                                     #
# --------------------------------------------------------------------------- #
def rebalance_schedule(buy_anchor, horizon_m, k_months, anchors,
                       anchor_step_months=12):
    """Ordered list of anchor dates [buy, buy+k, buy+2k, ..., eval] over the horizon.

    Returns the schedule (rebalance dates, INCLUDING the terminal eval date as the
    last element).  Sub-periods are consecutive pairs.  RAISES UnevaluableK if k or
    the horizon is not a whole number of anchor steps, or if any scheduled date is
    absent from `anchors`.
    """
    if k_months % anchor_step_months or horizon_m % anchor_step_months:
        raise UnevaluableK(
            f"k={k_months}mo / horizon={horizon_m}mo not a multiple of the anchor "
            f"step ({anchor_step_months}mo); needs a finer price grid")
    k_steps = k_months // anchor_step_months
    h_steps = horizon_m // anchor_step_months
    if k_steps < 1:
        raise UnevaluableK(f"k={k_months}mo < one anchor step")
    if buy_anchor not in anchors:
        raise UnevaluableK(f"buy anchor {buy_anchor!r} absent from price grid")
    b_idx = anchors.index(buy_anchor)
    if b_idx + h_steps >= len(anchors):
        raise UnevaluableK(
            f"horizon {horizon_m}mo from {buy_anchor} runs past the anchor grid")

    # rebalance points at 0, k, 2k, ... < horizon, then the terminal eval point.
    step_offsets = list(range(0, h_steps, k_steps))
    dates = []
    for off in step_offsets:
        idx = b_idx + off
        dates.append(anchors[idx])
    dates.append(anchors[b_idx + h_steps])  # terminal eval anchor
    return dates


# --------------------------------------------------------------------------- #
#  One (window, k) evaluation                                                 #
# --------------------------------------------------------------------------- #
def evaluate_strategy(buy_anchor, horizon_m, k_months, rank_fn, price_source,
                      N=20, tx_cost_bps=20.0, threshold=0.10, missing="fail",
                      anchors=None, anchor_step_months=12,
                      charge_initial_entry=True):
    """Evaluate the rebalance-every-k strategy for ONE (buy window, k).

    Parameters
    ----------
    rank_fn      : callable(as_of_date_str) -> ordered list of `source` strings.
                   The engine slices the first N as the basket at that rebalance.
    price_source : returns_core.PriceSource (or any .price/.last_before provider).
    tx_cost_bps  : one-way transaction cost in basis points per unit turnover.
                   0.0 -> frictionless.
    charge_initial_entry : if True, the formation of the book at buy costs bps
                   (buying N names).  Irrelevant at tx_cost_bps == 0.

    Returns a dict with, per sleeve (constant rank-position over the horizon):
      sleeve_returns_frictionless / sleeve_returns_net : list[N] chained returns
      sleeve_terminal_flag                             : list[N] (any sub-period
                                                          hit the terminal policy)
      basket_return_frictionless / basket_return_net   : chain of the equal-weight
                                                          portfolio sub-period returns
      benchmark_return                                 : URTH buy->eval (exact)
      excess_frictionless / excess_net                 : basket - benchmark
      turnover_oneway_total / turnover_by_period       : set-based book turnover
      schedule / n_periods
    Beat-rate is NOT computed here (it pools across windows) -- the driver builds the
    synthetic returns table from sleeve_returns_* and calls rc.beat_rate.
    """
    anchors = list(anchors) if anchors is not None else list(rc.DEFAULT_ANCHORS)
    sched = rebalance_schedule(buy_anchor, horizon_m, k_months, anchors,
                               anchor_step_months)
    eval_anchor = sched[-1]
    n_periods = len(sched) - 1

    # Benchmark: single URTH series, buy->eval directly (no chaining needed -- it's
    # the same index level at both ends).  Exact-anchor guard (LOW-2).
    bench = rc.benchmark_return(price_source, buy_anchor, eval_anchor,
                                require_exact=True)

    # Per period: rank at the period-start anchor, take top-N basket, per-name returns
    # over [period_start, period_end].  Track baskets for turnover.
    baskets = []                      # list[list[source]] the top-N at each period
    period_name_returns = []          # list[DataFrame] per-name returns tables
    for p in range(n_periods):
        d0, d1 = sched[p], sched[p + 1]
        ranked = list(rank_fn(d0))
        basket = ranked[:N]
        baskets.append(basket)
        rdf = rc.compute_returns(basket, d0, d1, price_source)
        period_name_returns.append(rdf)

    # ---- turnover (set-based, correct book economics) ----
    turnover_by_period = []           # one-way traded fraction at the START of period p
    for p in range(n_periods):
        if p == 0:
            frac = 1.0 if charge_initial_entry else 0.0   # buy the whole book
        else:
            prev, cur = set(baskets[p - 1]), set(baskets[p])
            entered = cur - prev
            exited = prev - cur
            denom = max(len(cur), len(prev), 1)
            frac = (len(entered) + len(exited)) / denom   # round-trip fraction
        turnover_by_period.append(frac)
    cost_frac = [t * tx_cost_bps / 1e4 for t in turnover_by_period]

    # ---- sleeves: constant rank-position chained across periods ----
    # Sleeve r accrues the return of the name occupying rank r in each period.
    sleeve_fr = [1.0] * N
    sleeve_net = [1.0] * N
    sleeve_valid = [True] * N         # a sleeve is valid only if every period had a
                                      # name (short rankings can starve deep sleeves)
    sleeve_terminal = [False] * N
    for p in range(n_periods):
        rdf = period_name_returns[p]
        c = cost_frac[p]
        # rdf preserves basket order -> row i is rank i.
        for r in range(N):
            if r >= len(rdf):
                sleeve_valid[r] = False
                continue
            row = rdf.iloc[r]
            if row["status"] == "no_buy":
                # no buy leg this period -> the sleeve cannot be evaluated cleanly.
                sleeve_valid[r] = False
                continue
            gross = row["total_return"]      # primary terminal policy already applied
            if row["terminal_flag"]:
                sleeve_terminal[r] = True
            sleeve_fr[r] *= (1.0 + gross)
            sleeve_net[r] *= (1.0 + gross) * (1.0 - c)

    sleeve_returns_frictionless = [
        (sleeve_fr[r] - 1.0) if sleeve_valid[r] else np.nan for r in range(N)]
    sleeve_returns_net = [
        (sleeve_net[r] - 1.0) if sleeve_valid[r] else np.nan for r in range(N)]

    # ---- equal-weight portfolio (basket) chained return ----
    basket_fr, basket_net = 1.0, 1.0
    for p in range(n_periods):
        rdf = period_name_returns[p]
        c = cost_frac[p]
        avg = rc.average_return(rdf, floor=False)   # equal-weight over included names
        if avg != avg:                              # NaN -> empty period, abort chain
            basket_fr = basket_net = float("nan")
            break
        basket_fr *= (1.0 + avg)
        basket_net *= (1.0 + avg) * (1.0 - c)
    basket_return_frictionless = (basket_fr - 1.0) if basket_fr == basket_fr else float("nan")
    basket_return_net = (basket_net - 1.0) if basket_net == basket_net else float("nan")

    return {
        "buy_anchor": buy_anchor, "eval_anchor": eval_anchor,
        "horizon_m": horizon_m, "k_months": k_months, "N": N,
        "tx_cost_bps": tx_cost_bps, "n_periods": n_periods, "schedule": sched,
        "benchmark_return": bench,
        "sleeve_returns_frictionless": sleeve_returns_frictionless,
        "sleeve_returns_net": sleeve_returns_net,
        "sleeve_terminal_flag": sleeve_terminal,
        "sleeve_valid": sleeve_valid,
        "basket_return_frictionless": basket_return_frictionless,
        "basket_return_net": basket_return_net,
        "excess_frictionless": (basket_return_frictionless - bench)
                               if basket_return_frictionless == basket_return_frictionless else float("nan"),
        "excess_net": (basket_return_net - bench)
                      if basket_return_net == basket_return_net else float("nan"),
        "turnover_by_period": turnover_by_period,
        "turnover_oneway_total": float(np.sum(turnover_by_period)),
        "baskets": baskets,
    }


# --------------------------------------------------------------------------- #
#  Beat-rate over sleeves -- ROUTED THROUGH rc.beat_rate                      #
# --------------------------------------------------------------------------- #
def _sleeves_to_returns_df(result, use_net):
    """Build a synthetic RETURNS_COLS table from a result's sleeves so rc.beat_rate
    consumes it exactly as it consumes a compute_returns table.

    Each valid sleeve -> one row: total_return = chained sleeve return, status 'ok'
    (or 'terminal' if any sub-period was terminal), terminal_flag set accordingly.
    Invalid sleeves (starved / no-buy) -> status 'no_buy' so rc.beat_rate EXCLUDES
    them from the denominator, matching the certified missing-buy handling.

    THE TRAILING "" IS THE `continuity` COLUMN (RETURNS_COLS, Q-42) and it is empty on
    purpose: a SLEEVE is a chain of sub-period returns, not a listing line, so no continuity
    string is meaningful at this level.  The continuity handling has already happened inside
    each `rc.compute_returns` call the chain was built from -- a sub-period that followed an
    issuer onto a successor line contributed its real return to `rets[r]` and is NOT flagged
    terminal, which is the whole benefit arriving here without this function knowing about it.
    """
    key = "sleeve_returns_net" if use_net else "sleeve_returns_frictionless"
    rets = result[key]
    term = result["sleeve_terminal_flag"]
    valid = result["sleeve_valid"]
    rows = []
    for r in range(result["N"]):
        tag = f"sleeve{r}"
        if not valid[r] or rets[r] != rets[r]:
            rows.append((tag, np.nan, np.nan, np.nan, np.nan, np.nan, False, "no_buy", ""))
            continue
        rr = rets[r]
        if term[r]:
            rows.append((tag, 1.0, np.nan, np.nan, rr, rr, True, "terminal", ""))
        else:
            rows.append((tag, 1.0, 1.0 + rr, 1.0 + rr, rr, rr, False, "ok", ""))
    return pd.DataFrame(rows, columns=rc.RETURNS_COLS)


def window_beat_rate(result, use_net, threshold=0.10, missing="fail"):
    """Per-sleeve beat-rate for ONE (window, k) via the certified rc.beat_rate.
    Returns (rate, n_evaluated)."""
    rdf = _sleeves_to_returns_df(result, use_net)
    return rc.beat_rate(rdf, result["benchmark_return"], threshold=threshold,
                        missing=missing, floor=False)


def pooled_beat_rate(results, use_net, threshold=0.10, missing="fail"):
    """Pool the per-sleeve beat-rate across windows THROUGH rc.beat_rate (count-
    weighted aggregation of per-window primitive calls -- NOT a re-implementation).

    Equivalent to pooling the per-name beat/not booleans: pooled_rate =
    sum_w(rate_w * n_w) / sum_w(n_w).  At k == horizon with tx == 0 this reproduces
    the certified pooled-clean per-name beat-rate exactly.
    Returns (pooled_rate, total_n).
    """
    num, den = 0.0, 0
    for res in results:
        rate, n = window_beat_rate(res, use_net, threshold=threshold, missing=missing)
        if rate == rate and n > 0:
            num += rate * n
            den += n
    return (num / den if den else float("nan")), den
