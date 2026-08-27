"""The TWO-CLAUSE target, computed.  Pure functions over a `returns_core` returns table.

WHY THIS MODULE EXISTS.  The charter's target has had two clauses since 2026-08-20 and only
ONE was ever wired.  `pipeline_analysis.beat_rate_vs_urth` measured the UPSIDE clause and the
run printed it as though it were the target; the DOWNSIDE clause was hand-computed twice and
built zero times.  A run that prints half a target reads as a pass when the half nobody
measured is the half that fails, so both clauses now come out of the same stage or neither
does.

THE TARGET (charter, `TWO-CLAUSE TARGET`, verbatim in substance):

  UPSIDE    >= 60% of the top-20 beat MSCI World by >= 10pp over 36 months   [per pick]
  DOWNSIDE  the EQUAL-WEIGHT top-20 total return over 36 months exceeds a flat 3%/yr
            bond (>= 9.27% compounded over the window)                       [per portfolio]

  BOTH MUST PASS for a period to count as success, and they are NOT tradable against each
  other -- that is the stated point of the pair: a filter can always buy a higher hit rate
  with tail risk, and the single-clause target could not see it.

THE BOND BAR IS FLAT 3% NOMINAL, DELIBERATELY NOT THE BUY-DATE TREASURY.  The CEO chose flat
for comparability across anchors.  The cost is known and is carried in the printed output
rather than left to be rediscovered: it FLATTERS the filter at the 2022 anchor (real 3y
Treasury ~4.3%) and PENALISES it at 2020 (~0.2%), so the clause partly measures the rate
cycle.  Accepted for a stable bar -- do not "improve" it into a floating rate here.

THE DOWNSIDE CLAUSE IS THE SOFTEST OF THE FOUR CANDIDATES CONSIDERED, on purpose.  A single
strong name can carry a 20-name equal-weight portfolio over a 9.27% bar, so the clause can
pass while most of the list is under water.  That weakness is why the charter also names
three DIAGNOSTICS -- p25 of the picks, the worst pick, the count below zero -- which are
COMPUTED AND PRINTED BUT NEVER GATE.  Do not quietly promote a diagnostic into the clause;
if the bar should be harder, that is a CEO decision recorded in the charter.

TWO POLICIES, BOTH REPORTED, because a missing eval price is not a loss:
  * PRIMARY -- an unpriceable eval leg is marked at the last observed price before eval
    (`returns_core` terminal policy).  The headline.
  * FLOOR   -- an unpriceable eval leg is -100%.
The run currently flags interior price holes (a name priced BEFORE and AFTER a missing
anchor) which FLOOR marks as total losses although a later price proves the name alive.  So
FLOOR is a lower bound on the portfolio, not a reading of it, and the two are printed side by
side precisely so "the filter picked losers" can never be read off a grid that could not
price the winners.

COVERAGE IS PART OF THE ANSWER, NOT A FOOTNOTE.  An equal-weight return over the 9 of 20
picks that happened to be priceable is a DIFFERENT QUANTITY from the chartered clause.  This
module therefore never returns a bare number: it returns the point estimate on the priced
names, the strict lower bound (every unpriced pick at -100%), and `flip_return` -- the average
return the unpriced picks would need for the portfolio to clear the bar.  `flip_return` is
what makes an incomplete verdict actionable: +6% is a real coin-flip, +400% is a FAIL in all
but name.

A PICK PRICED AT THE BUY ANCHOR ONLY IS NOT A MEASUREMENT.  `returns_core.compute_returns`
falls back to `terminal = p_buy` when a name has no eval leg AND no earlier price to fall back
to, which yields `total_return` of EXACTLY 0.0 with `terminal_flag` set.  That 0.0 is
fabricated -- it says "we know nothing", not "the position was flat".  Left in, it counts
toward coverage, so a window could report `coverage = 1.0` and a DEFINITIVE verdict while
resting on names nothing priced; flip the sign of the bar and one such name takes the
chartered clause from FAIL to PASS "at full coverage".  This module therefore classes those
picks as UNMEASURED, alongside `no_buy`, and reports them as `n_buy_only`.

  The interaction is what made it dangerous, and it is worth naming: the wider grid (the
  allow-list removal) moves names that USED to be honestly `no_buy` into this silently
  marked-at-break-even state.  The two halves of this change set created it together.

  The test is `terminal_flag and terminal_adjClose == buy_adjClose`, which has one
  false-positive mode: a genuine last-before price bit-identical to the buy price.  That
  direction is SAFE and the safety is asserted, not assumed -- reclassifying a pick as
  unmeasured can only ever turn a definitive verdict INDETERMINATE, never manufacture a PASS.
  (Downside: it lowers the strict lower bound, which is the only route to a partial-coverage
  PASS.  Upside: it lowers `lo` and leaves `hi` unchanged, so neither a PASS nor a FAIL can
  be conjured.)

COVERAGE DISCIPLINE APPLIES TO BOTH CLAUSES, and the asymmetry between them is the point.  A
PORTFOLIO RETURN is unbounded above, so a missing pick can never be ruled out and partial
coverage can only ever prove PASS (via the -100% lower bound) or stay INDETERMINATE.  A
BEAT-RATE is bounded on BOTH sides -- every missing pick either beats or does not -- so
partial coverage yields a genuine interval [lo, hi] and can prove PASS *or* FAIL.  An earlier
version of this module applied the discipline only to the downside clause and let the upside
clause read `FAIL` off the priced subset; because `period_verdict` lets FAIL dominate, the run
printed `PERIOD: FAIL` on the same line as `DOWNSIDE: INDETERMINATE`, contradicting itself and
reporting a price-grid defect as a filter defect after all.

NO I/O, NO NETWORK, NO PIPELINE IMPORTS.  Everything here is a function of a returns table.
"""

import math

import numpy as np
import pandas as pd

import returns_core as rc

#  Flat nominal annual bond rate.  CEO decision 2026-08-20; see the module docstring for the
#  accepted cost of flat-vs-buy-date.  ONE definition, so the printed bar and any test agree.
BOND_RATE_ANNUAL = 0.03

#  The charter states the DOWNSIDE clause on the top-20.  It is not stated on the top-10, and
#  computing it there would invent a clause the CEO did not set -- the top-10 rung (52.5%) is
#  an UPSIDE rung only.
CHARTERED_DEPTH = 20
CHARTERED_HORIZON_M = 36


def bond_bar(horizon_m=CHARTERED_HORIZON_M, annual=BOND_RATE_ANNUAL):
    """Compounded total return of the flat bond over `horizon_m` months.

    36 months at 3%/yr -> 1.03**3 - 1 = 0.092727..., i.e. the charter's ">= 9.27%".  Written
    as a function of the horizon so a 12- or 24-month read cannot silently keep the 36-month
    bar; the charter's number is recovered exactly at horizon_m=36.
    """
    return (1.0 + annual) ** (horizon_m / 12.0) - 1.0


def buy_only_mask(inc):
    """Picks priced at the BUY anchor only -- no eval leg and nothing earlier to fall back to.

    `returns_core.compute_returns` marks these `terminal` with `terminal = p_buy`, so the
    return is exactly 0.0 by construction rather than by measurement.  Detected structurally
    (`terminal_flag` AND terminal price identical to the buy price) because `compute_returns`
    does not record which branch of its fallback it took, and widening its schema would ripple
    through every stage that reads RETURNS_COLS.
    """
    tf = inc["terminal_flag"].astype(bool).fillna(False)
    same = (pd.to_numeric(inc["terminal_adjClose"], errors="coerce")
            == pd.to_numeric(inc["buy_adjClose"], errors="coerce"))
    return tf & same


def measured(returns_df):
    """Picks whose return is an OBSERVATION.  The denominator for every figure here.

    Two exclusions, for the same reason: `no_buy` (never opened) and `buy_only` (opened, but
    nothing ever priced the other leg).  Both are UNKNOWN, and an unknown is coverage, not a
    data point.
    """
    inc = rc.included(returns_df)
    if not len(inc):
        return inc
    return inc[~buy_only_mask(inc)]


def coverage_counts(returns_df, depth_n):
    """One place that decides what is measured and what is merely absent."""
    inc = rc.included(returns_df)
    n_buy_only = int(buy_only_mask(inc).sum()) if len(inc) else 0
    cnt = rc.counts(returns_df)
    n_measured = int(len(measured(returns_df)))
    n_selected = int(depth_n)
    return {"n_selected": n_selected, "n_measured": n_measured,
            "n_missing": max(0, n_selected - n_measured),
            "n_terminal": int(cnt["n_terminal"]), "n_buy_only": n_buy_only,
            "n_no_buy": int(cnt["n_no_buy"]),
            "coverage": (float(n_measured) / n_selected) if n_selected else float("nan")}


def _returns(returns_df, floor):
    """Per-pick MEASURED returns under one policy."""
    col = "total_return_floor" if floor else "total_return"
    m = measured(returns_df)
    if not len(m):
        return pd.Series(dtype=float)
    return pd.to_numeric(m[col], errors="coerce").dropna()


def diagnostics(returns_df, bar, floor=False):
    """The three charter DIAGNOSTICS.  Reported, NEVER gating.

    p25 IS TAKEN AS AN ORDER STATISTIC, not an interpolated percentile, because the charter
    glosses it itself: "p25 of the 20 picks -- the 5th-worst must clear it".  ceil(0.25*n)
    gives the 5th-smallest at n=20, which is that sentence exactly; an interpolating
    `np.percentile` would land between the 5th and 6th and quietly disagree with the charter's
    own words.  `p25_rank` is returned so the reading is checkable at any n.

    EVERY FIGURE CARRIES ITS n.  "3 picks ended below zero" out of 8 priced is not the same
    statement as out of 20, and the run has been thin enough that the difference decides the
    reading.

    COMPUTED OVER MEASURED PICKS ONLY.  A pick priced at the buy anchor alone contributes a
    fabricated 0.0, which would sit above the worst pick, below the median, and count as "not
    below zero" -- polluting all three diagnostics at once with a number nobody observed.
    """
    r = _returns(returns_df, floor)
    n = int(len(r))
    if n == 0:
        return {"n": 0, "p25": float("nan"), "p25_rank": 0, "p25_clears_bar": None,
                "worst": float("nan"), "worst_clears_bar": None,
                "n_below_zero": 0, "share_below_zero": float("nan")}
    ordered = np.sort(r.to_numpy(dtype=float))
    rank = max(1, int(math.ceil(0.25 * n)))
    p25 = float(ordered[rank - 1])
    worst = float(ordered[0])
    n_below = int((ordered < 0.0).sum())
    return {"n": n,
            "p25": p25, "p25_rank": rank, "p25_clears_bar": bool(p25 >= bar),
            "worst": worst, "worst_clears_bar": bool(worst >= bar),
            "n_below_zero": n_below, "share_below_zero": float(n_below) / n}


def downside_clause(returns_df, depth_n, horizon_m=CHARTERED_HORIZON_M, floor=False,
                    annual=BOND_RATE_ANNUAL):
    """The chartered DOWNSIDE clause on ONE anchor: equal-weight portfolio vs the bond.

    `depth_n` is the number of picks the portfolio was MEANT to hold (20), not the number
    that turned out to be priceable -- the difference between those two IS the finding.

    VERDICTS, and why there are three:
      PASS           -- the clause clears the bar on evidence that the unpriced picks cannot
                        overturn: either every pick is priced, or the strict lower bound
                        (unpriced at -100%) already clears.
      FAIL           -- every pick is priced and the portfolio is under the bar.  Only ever
                        returned at full coverage, because that is the only case where the
                        number IS the chartered clause.
      INDETERMINATE  -- picks are missing.  A missing pick is unbounded ABOVE, so no amount of
                        shortfall in the priced names can close the question; saying FAIL here
                        would be reporting a coverage defect as a filter defect.  Read
                        `flip_return` before treating it as anything.

    Returned keys:
      bar                 the bond hurdle for this horizon
      n_selected/n_priced/n_terminal/n_no_buy/coverage
      portfolio_return    equal-weight mean over the PRICED picks -- the point estimate, and
                          NOT the chartered clause when coverage < 1
      lower_bound         equal-weight mean over all `depth_n` with every unpriced pick at -100%
      flip_return         average return the unpriced picks would need for the FULL portfolio
                          to clear the bar (nan at full coverage -- nothing to flip)
      verdict             PASS | FAIL | INDETERMINATE
    """
    bar = bond_bar(horizon_m, annual)
    r = _returns(returns_df, floor)
    out = {"bar": bar, "horizon_m": horizon_m, "annual_bond": annual, "floor": bool(floor)}
    out.update(coverage_counts(returns_df, depth_n))
    n_selected = out["n_selected"]
    n_priced = int(len(r))
    out["n_priced"] = n_priced

    if n_priced == 0:
        out.update({"portfolio_return": float("nan"), "lower_bound": float("nan"),
                    "flip_return": float("nan"), "verdict": "INDETERMINATE",
                    "verdict_reason": "no pick in this window has a buy price"})
        return out

    total = float(r.sum())
    out["portfolio_return"] = total / n_priced
    n_missing = max(0, n_selected - n_priced)
    out["lower_bound"] = (total + n_missing * (-1.0)) / n_selected if n_selected else float("nan")

    if n_missing == 0:
        out["flip_return"] = float("nan")
        clears = out["portfolio_return"] >= bar
        out["verdict"] = "PASS" if clears else "FAIL"
        #  "Full coverage" here means every pick is MEASURED, not merely present in the frame.
        #  A pick priced at the buy anchor only is excluded upstream, so this sentence cannot
        #  be said over a fabricated break-even return the way it once could.
        out["verdict_reason"] = "full coverage: the portfolio return IS the chartered clause"
        return out

    #  What the missing picks would have to average for the FULL n_selected portfolio to
    #  clear the bar.  Solves (total + n_missing*x)/n_selected >= bar for x.
    out["flip_return"] = (bar * n_selected - total) / n_missing
    if out["lower_bound"] >= bar:
        out["verdict"] = "PASS"
        out["verdict_reason"] = ("clears even with every unpriced pick at -100%")
    else:
        out["verdict"] = "INDETERMINATE"
        out["verdict_reason"] = (
            "%d of %d picks unpriced; they are unbounded above, so the priced shortfall "
            "cannot decide the clause" % (n_missing, n_selected))
    return out


def upside_clause(returns_df, benchmark_ret, depth_n, threshold=0.10,
                  hit_rate_bar=0.60, floor=False):
    """The chartered UPSIDE clause on ONE anchor, WITH the same coverage discipline as the
    downside clause -- and with the asymmetry that makes it stronger.

    A beat-rate is a proportion, so it is bounded on BOTH sides: every unmeasured pick either
    beats the benchmark or does not.  That gives a genuine interval over the picks the anchor
    actually shipped --

        lo = n_beat / n_selected                    (every unmeasured pick fails)
        hi = (n_beat + n_missing) / n_selected      (every unmeasured pick beats)

    -- and unlike the portfolio return, a partial-coverage FAIL is therefore PROVABLE: if even
    the most generous assignment cannot reach 60%, the anchor has failed the clause whatever
    the missing names did.  This is exactly what an earlier version got wrong by reading FAIL
    off the priced subset.  On the live 2026-08-27 shape (~2 of 9 beating, 11 unmeasured of
    20) the interval is [10%, 65%] -- straddling the 60% bar, so genuinely undecidable, and a
    bare `FAIL` was an overclaim.

    Returns the same key shape as `downside_clause` so the pair prints symmetrically.
    `rate_measured` is the point estimate over measured picks only and is NOT the clause.
    """
    out = coverage_counts(returns_df, depth_n)
    out.update({"bar": hit_rate_bar, "threshold": threshold, "floor": bool(floor)})
    n_selected = out["n_selected"]
    m = measured(returns_df)
    if benchmark_ret != benchmark_ret or not len(m) or not n_selected:
        out.update({"n_beat": 0, "rate_measured": float("nan"),
                    "lo": float("nan"), "hi": float("nan"), "verdict": "INDETERMINATE",
                    "verdict_reason": ("no measured pick to compute a beat-rate on"
                                       if n_selected else "no pick shipped at this anchor")})
        return out
    col = "total_return_floor" if floor else "total_return"
    r = pd.to_numeric(m[col], errors="coerce").dropna()
    n_beat = int(((r - benchmark_ret) >= threshold).sum())
    n_measured = int(len(r))
    n_missing = max(0, n_selected - n_measured)
    lo = float(n_beat) / n_selected
    hi = float(n_beat + n_missing) / n_selected
    out.update({"n_beat": n_beat, "rate_measured": float(n_beat) / n_measured,
                "lo": lo, "hi": hi})
    if lo >= hit_rate_bar:
        out["verdict"] = "PASS"
        out["verdict_reason"] = "clears even if every unmeasured pick failed"
    elif hi < hit_rate_bar:
        out["verdict"] = "FAIL"
        out["verdict_reason"] = "cannot reach the bar even if every unmeasured pick beat"
    else:
        out["verdict"] = "INDETERMINATE"
        out["verdict_reason"] = (
            "%d of %d picks unmeasured; the bar sits inside [%.1f%%, %.1f%%]"
            % (n_missing, n_selected, lo * 100, hi * 100))
    return out


def period_verdict(upside, downside):
    """BOTH clauses must pass for a period to count as success (charter).

    An INDETERMINATE on either side makes the PERIOD indeterminate -- it is NOT a pass with a
    caveat and it is NOT a fail.  This is the whole reason the pair was introduced: a period
    that scores well on one clause and is unmeasurable on the other has not been shown to
    succeed, and the previous single-clause target would have called it a win.
    """
    vs = (upside.get("verdict"), downside.get("verdict"))
    if "FAIL" in vs:
        return "FAIL"
    if "INDETERMINATE" in vs:
        return "INDETERMINATE"
    return "PASS"


#  The caveats that must travel WITH the numbers.  Kept here, next to the computation, so a
#  second printer of these figures cannot ship them bare.
BOND_BAR_CAVEAT = (
    "flat 3%/yr nominal, NOT the buy-date Treasury (CEO 2026-08-20, chosen for comparability "
    "across anchors). Known cost: it FLATTERS the 2022 anchor (real 3y Treasury ~4.3%) and "
    "PENALISES 2020 (~0.2%), so the clause partly measures the rate cycle.")

SOFTNESS_CAVEAT = (
    "the DOWNSIDE clause is the softest of the four candidates considered (CEO's choice): a "
    "single strong name can carry a 20-name equal-weight portfolio over the bar. The three "
    "diagnostics below measure how close to failure it is; they do NOT gate.")

POLICY_CAVEAT = (
    "PRIMARY marks an unpriceable eval leg at the last observed price; FLOOR marks it -100%. "
    "Interior price holes (priced before AND after a missing anchor) are floored at -100% "
    "though a later price proves the name alive -- so FLOOR is a lower bound, not a reading. "
    "Where the two verdicts differ, the disagreement is a PRICE-GRID finding, not a filter one.")

COVERAGE_CAVEAT = (
    "n_selected is what the anchor SHIPPED; n_measured is what could be priced. buy_only picks "
    "(priced at the buy anchor only) are UNMEASURED, not flat -- their 0.0 return is a fallback, "
    "not an observation. Both clauses are graded over n_selected with the unmeasured picks "
    "treated as unknown: the portfolio is unbounded above so it can only prove PASS or stay "
    "INDETERMINATE, while the beat-rate is bounded both ways so [lo, hi] can prove PASS or FAIL.")
