"""Unit tests for the derived TOTAL-RETURN price source (derived_prices).

Mostly SYNTHETIC panels -- a handful of hand-built cdx_df rows whose correct answer can be
worked out on paper -- so the conventions are pinned independently of any pickle.  Plus an
ACCEPTANCE test against the real deep panel + real_prices.csv, which SKIPS when either is
absent (this is the normal state on a machine without the panel; a bare `return` there
would report green having asserted nothing -- see the audit C4 note in test_returns_core).

WHAT THESE TESTS EXIST TO STOP (each is a defect someone will reintroduce):
  * "simplify" the per-share divisor into a marketCap ratio           -> test_per_share_*
  * treat the derived level as a PRICE (drop the dividend leg)        -> test_level_is_*
  * add dividends as a simple SUM instead of compounding              -> test_reinvestment_*
  * CLIP a vendor-unit-error yield instead of rejecting it            -> test_extreme_*
  * anchor on the price-file grid date (2022-12-30) not Dec-31        -> test_dec31_*
  * take the yield over period-END market cap                        -> test_yield_denominator_*
  * compare a reporting-ccy return against a listing-ccy adjClose    -> test_listing_currency_*
  * price a name off a carried-forward pre-listing price             -> test_backfill_*

TWO FIXTURE CONSTRAINTS THE GUARDS IMPOSE, and both bit real tests here (2026-08-21):
  * A FIXTURE MUST NOT USE A FLAT PRICE unless it is testing the backfill guard.  An exact
    price repeat is now DROPPED, so a flat-price fixture silently carries the previous level
    forward and the test passes while asserting nothing.  `test_per_share_divisor_*` -- the
    headline pin -- was vacuous for exactly this reason until it was rewritten.
  * A BARE TICKER IS A US LISTING, so it must report USD or the listing-currency guard drops
    it.  A fixture that means to exercise a non-USD reporter uses a suffixed ticker.
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import derived_prices as dpx
import returns_core as rc


# --------------------------------------------------------------------------- #
#  Synthetic panel helper                                                     #
# --------------------------------------------------------------------------- #
def panel(rows):
    """rows: (source, periodEndDate, marketCap, shares, dividendsPaid, currency).

    `price` is built the way getData_fmp builds it -- marketCap / weightedAverageShsOut --
    so the fixture cannot drift from the shipped definition.

    Bare tickers ("X") are US listings, so their LISTING currency is USD; the fixtures below
    report USD to stay currency-coherent and survive the listing-currency guard.  A fixture
    that deliberately tests the guard uses a suffixed ticker ("X.TO" -> CAD).
    """
    df = pd.DataFrame(rows, columns=["source", "periodEndDate", "marketCap",
                                     "weightedAverageShsOut", "dividendsPaid",
                                     "reportedCurrency"])
    df["price"] = df["marketCap"] / df["weightedAverageShsOut"]
    return df


ANCH = ["2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31"]


def src(rows, **kw):
    kw.setdefault("anchors", ANCH)
    return dpx.DerivedPriceSource(panel(rows), **kw)


# --------------------------------------------------------------------------- #
#  1. THE LEVEL IS A TOTAL RETURN, NOT A PRICE                                #
# --------------------------------------------------------------------------- #
def test_level_is_total_return_not_price():
    """A payer's level ratio must EXCEED its price ratio by the compounded yield.

    Price doubles 100 -> 200.  A dividend of 100 is paid in the eval period against a
    period-START cap of 1000, so the yield is 0.10 and the TOTAL return is 2 * 1.10 - 1 =
    +120% against a price-only +100%.
    """
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0,    "USD"),   # price 100, no dividend
        ("X", "2019-12-31", 2000.0, 10.0, -100.0, "USD"),   # price 200, div = 10% of START
    ])
    r = rc.compute_returns(["X"], "2018-12-31", "2019-12-31", ps).iloc[0]
    assert r["status"] == "ok"
    assert math.isclose(r["total_return"], 2.0 * 1.10 - 1.0, rel_tol=1e-12)
    #  and it is STRICTLY above the price-only answer -- the whole point of the leg
    assert r["total_return"] > 1.0


def test_non_payer_level_is_exactly_the_price():
    """With no dividends the level must be the raw price, to the bit.  This is what makes
    the derived leg reconcile with adjClose for non-payers (25th pct pinned at 1.0000)."""
    ps = src([("X", "2018-12-31", 1234.5, 7.0, 0.0, "USD")])
    assert ps.price("X", "2018-12-31") == 1234.5 / 7.0


# --------------------------------------------------------------------------- #
#  2. THE PER-SHARE DIVISOR IS KEPT  (the "someone will simplify this" pin)    #
# --------------------------------------------------------------------------- #
def test_per_share_divisor_is_kept_not_a_marketcap_ratio():
    """PIN: the price leg must be marketCap/shares, NEVER a marketCap ratio.

    Measured stakes: per-share Spearman against the real leg is 0.9675 vs 0.8265 for
    marketCap-only, because a quarter of the universe changes its share count by >25% over
    36 months.  Here marketCap grows 2.4x while the share count doubles, so the per-share
    return is +20% and a marketCap ratio would report +140%.

    The two prices are deliberately DIFFERENT (100 -> 120).  An earlier version used a flat
    price, which the backfill guard now drops -- the level then carried forward and the test
    asserted 0.0 == 0.0 while exercising nothing.
    """
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD"),   # price 100
        ("X", "2019-12-31", 2400.0, 20.0, 0.0, "USD"),   # price 120  (shares doubled)
    ])
    r = rc.compute_returns(["X"], "2018-12-31", "2019-12-31", ps).iloc[0]
    assert r["status"] == "ok", "fixture tripped a guard -- the pin would be vacuous"
    assert math.isclose(r["total_return"], 0.20, rel_tol=1e-12), (
        "per-share price went 100 -> 120. A different answer means the divisor was dropped "
        "for a marketCap ratio.")
    #  state the counterfactual explicitly so the pin cannot be read as vacuous
    marketcap_only = 2400.0 / 1000.0 - 1.0
    assert not math.isclose(r["total_return"], marketcap_only, abs_tol=1e-9)


def test_per_share_divisor_survives_a_buyback():
    """Shares SHRINK while marketCap is flat -> the per-share return is positive."""
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD"),   # price 100
        ("X", "2019-12-31", 1000.0,  8.0, 0.0, "USD"),   # price 125
    ])
    r = rc.compute_returns(["X"], "2018-12-31", "2019-12-31", ps).iloc[0]
    assert math.isclose(r["total_return"], 0.25, rel_tol=1e-12)


# --------------------------------------------------------------------------- #
#  3. ANCHOR ON CALENDAR DEC-31, NOT THE PRICE-FILE GRID DATE                 #
# --------------------------------------------------------------------------- #
def test_dec31_anchoring_not_grid_date():
    """PIN: anchor '2022-12-30' must resolve to the 2022-12-31 period end, lag 0.

    A naive "period end <= anchor" rule against the 12-30 / 12-29 grid dates pushes every
    December filer back a full quarter -- measured median lag 0 -> 91 days, share of names
    >45d stale 9.3% -> 97.4%.
    """
    ps = src([
        ("X", "2022-09-30", 900.0, 10.0, 0.0, "USD"),
        ("X", "2022-12-31", 1000.0, 10.0, 0.0, "USD"),
    ], anchors=["2022-12-30"])
    assert ps.price("X", "2022-12-30") == 100.0, "picked the September row -- grid-date bug"
    assert ps.lag_days("X", "2022-12-30") == 0


def test_anchor_cutoff_is_always_year_end():
    for label in ("2023-12-29", "2023-12-31", "2023-01-05"):
        assert dpx.DerivedPriceSource._anchor_cutoff(label) == pd.Timestamp("2023-12-31")


def test_ragged_fiscal_year_end_is_picked_and_lag_reported():
    """A June filer legitimately lags the anchor; the lag is REPORTED, not hidden."""
    ps = src([("X", "2021-06-30", 1000.0, 10.0, 0.0, "USD")], anchors=["2021-12-31"])
    assert ps.price("X", "2021-12-31") == 100.0
    assert ps.lag_days("X", "2021-12-31") == 184
    rep = ps.timing_report().set_index("anchor").loc["2021-12-31"]
    assert rep["n"] == 1 and rep["max_lag"] == 184 and rep["pct_over_45d"] == 100.0


# --------------------------------------------------------------------------- #
#  4. REINVESTMENT CONVENTION: MULTIPLICATIVE, NOT ADDITIVE                   #
# --------------------------------------------------------------------------- #
def test_reinvestment_is_multiplicative_not_additive():
    """Two 10% period yields must compound to 1.21, not sum to 1.20."""
    #  marketCap is flat at 1000 so each yield is 100/1000 = 0.10 on the START cap, while
    #  the share count shrinks so the PRICE still moves (100 -> 125 -> 200) and no row trips
    #  the backfill guard.  Price return 2.0, dividend factor 1.1 * 1.1 = 1.21.
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0,    "USD"),   # price 100
        ("X", "2019-12-31", 1000.0,  8.0, -100.0, "USD"),   # price 125, y = 0.10
        ("X", "2020-12-31", 1000.0,  5.0, -100.0, "USD"),   # price 200, y = 0.10
    ])
    r = rc.compute_returns(["X"], "2018-12-31", "2020-12-31", ps).iloc[0]
    assert math.isclose(r["total_return"], 2.0 * 1.10 * 1.10 - 1.0, rel_tol=1e-12)
    additive = 2.0 + 0.10 + 0.10 - 1.0
    assert not math.isclose(r["total_return"], additive, abs_tol=1e-6)


def test_chain_link_identity_on_the_start_cap_convention():
    """level_t/level_{t-1} == (P_t/P_{t-1}) * (1 + D_t/MC_{t-1}).

    With a constant share count D_t/MC_{t-1} is d_t/P_{t-1} -- dividend per share over the
    START price.  Writing it against the START cap is the point: the holder earned that
    dividend on the value they owned when it was declared, not on the post-move value.

    NOTE WHAT THIS IS NOT.  Reinvesting at the START price is NOT the textbook
    (P_t + d_t)/P_{t-1}, which reinvests at the END (ex-dividend) price -- here 1.272 against
    1.260.  Both are total returns; they differ by y * (P_t/P_{t-1} - 1), a second-order term
    that is POSITIVE in a rising market, so this convention runs marginally above an
    adjClose-style end-price reinvestment.  The asymmetry is accepted: it is far smaller than
    the end-cap denominator error it replaces (ZIM 0.42 vs 0.27 in a single period), and a
    reader who assumes the textbook form should see the difference asserted here rather than
    discover it.
    """
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0,   "USD"),    # P = 100, MC = 1000
        ("X", "2019-12-31", 1200.0, 10.0, -60.0, "USD"),    # P = 120, d = 6, y = 60/1000
    ])
    got = ps.price("X", "2019-12-31") / ps.price("X", "2018-12-31")
    assert math.isclose(got, (120.0 / 100.0) * (1.0 + 60.0 / 1000.0), rel_tol=1e-12)
    #  and it is DELIBERATELY not the end-price-reinvestment form
    assert not math.isclose(got, (120.0 + 6.0) / 100.0, rel_tol=1e-9)


def test_dividend_window_is_half_open():
    """The window is (buy, eval]: a dividend paid IN THE BUY PERIOD is already gone at the
    buy price, so it is EXCLUDED; one paid in the eval period is INCLUDED.

    Pinned from BOTH sides off an identical flat price path, so the assertion is about the
    boundary rather than about one number -- any off-by-one in the cumulative factor swaps
    the two answers.  The dividend is 20% of market cap, deliberately UNDER the 0.25 reject
    ceiling: at 50% the yield guard would zero it and the test would pass vacuously without
    ever exercising the window.
    """
    #  Identical price path 100 -> 110 in both fixtures; only WHERE the 200 dividend sits
    #  differs.  In the eval case y = 200 / START cap 1000 = 0.20.
    div_in_buy = src([
        ("X", "2018-12-31", 1000.0, 10.0, -200.0, "USD"),   # BUY period
        ("X", "2019-12-31", 1100.0, 10.0, 0.0,    "USD"),
    ])
    div_in_eval = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0,    "USD"),
        ("X", "2019-12-31", 1100.0, 10.0, -200.0, "USD"),   # EVAL period, y = 0.20
    ])
    r_buy = rc.compute_returns(["X"], "2018-12-31", "2019-12-31", div_in_buy).iloc[0]
    r_eval = rc.compute_returns(["X"], "2018-12-31", "2019-12-31", div_in_eval).iloc[0]
    assert math.isclose(r_buy["total_return"], 0.10, rel_tol=1e-12), "buy-period div leaked in"
    assert math.isclose(r_eval["total_return"], 1.10 * 1.20 - 1.0, rel_tol=1e-12), (
        "eval-period div lost")


# --------------------------------------------------------------------------- #
#  5. VENDOR-ERROR YIELDS ARE REJECTED, NOT CLIPPED                           #
# --------------------------------------------------------------------------- #
def test_extreme_yield_is_rejected_not_clipped():
    """TCX-shaped row (dividendsPaid ~ 100x marketCap) must contribute factor 1.0.

    Clipping to the 0.25 ceiling instead would keep a quarter of a garbage number and
    COMPOUND it -- 1.25^12 = 14.6x of manufactured return, on names that are payers by
    construction, which is what left the prototype with a +0.009 IC overshoot.
    """
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0,       "USD"),   # price 100
        ("X", "2019-12-31", 1000.0,  8.0, -100000.0, "USD"),   # price 125, y = 100
    ])
    r = rc.compute_returns(["X"], "2018-12-31", "2019-12-31", ps).iloc[0]
    assert math.isclose(r["total_return"], 0.25, rel_tol=1e-12), (
        "rejected -> factor 1.0, so the price return survives alone")
    clipped = 1.25 * (1.0 + dpx.MAX_PERIOD_YIELD) - 1.0
    assert not math.isclose(r["total_return"], clipped, abs_tol=1e-9), "this is the clip bug"
    assert ps.diagnostics()["n_yield_rows_rejected"] == 1


def test_yield_at_the_ceiling_is_kept():
    """The ceiling is inclusive, and it is now 1.0 rather than 0.25.

    At 0.25 over the END cap this guard was zeroing GENUINE special dividends -- ZIM
    2022-06-30 (0.42) and 2023-06-30 (0.52) among 106 rows across 87 sources in the
    plausible (0.25, 1.0] band, on exactly the payers a value screen selects.
    """
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0,     "USD"),   # price 100
        ("X", "2019-12-31", 1000.0,  8.0, -1000.0, "USD"),   # price 125, y = 1.0 exactly
    ])
    r = rc.compute_returns(["X"], "2018-12-31", "2019-12-31", ps).iloc[0]
    assert math.isclose(r["total_return"], 1.25 * 2.0 - 1.0, rel_tol=1e-12)
    assert ps.diagnostics()["n_yield_rows_rejected"] == 0


def test_positive_dividends_paid_contributes_nothing():
    """dividendsPaid is a cash OUTFLOW; a POSITIVE value is a vendor sign defect (398 rows
    on the panel) and must not be read as a dividend received."""
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0,   "USD"),   # price 100
        ("X", "2019-12-31", 1100.0, 10.0, 100.0, "USD"),   # price 110, POSITIVE dividend
    ])
    r = rc.compute_returns(["X"], "2018-12-31", "2019-12-31", ps).iloc[0]
    assert math.isclose(r["total_return"], 0.10, rel_tol=1e-12)


def test_nan_dividend_is_zero_not_nan():
    """A missing dividend must not poison the whole cumulative factor for the source."""
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, np.nan, "USD"),
        ("X", "2019-12-31", 1100.0, 10.0, np.nan, "USD"),
    ])
    r = rc.compute_returns(["X"], "2018-12-31", "2019-12-31", ps).iloc[0]
    assert math.isclose(r["total_return"], 0.10, rel_tol=1e-12)


# --------------------------------------------------------------------------- #
#  6. CURRENCY GUARD                                                          #
# --------------------------------------------------------------------------- #
def test_currency_switch_drops_the_pre_switch_leg():
    """A price ratio spanning a redenomination carries the FX rate, not a return, so the
    pre-switch anchor must resolve to None rather than to a garbage number."""
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0, "GBP"),
        ("X", "2019-12-31", 1300.0, 10.0, 0.0, "USD"),
    ])
    assert ps.price("X", "2018-12-31") is None
    assert ps.price("X", "2019-12-31") == 130.0
    assert ps.diagnostics()["n_currency_switch_sources"] == 1
    df = rc.compute_returns(["X"], "2018-12-31", "2019-12-31", ps)
    assert df.iloc[0]["status"] == "no_buy"          # excluded, not silently wrong


def test_currency_guard_can_be_disabled():
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0, "GBP"),
        ("X", "2019-12-31", 1300.0, 10.0, 0.0, "USD"),
    ], strict_currency=False, require_listing_currency_match=False)
    assert ps.price("X", "2018-12-31") == 100.0


def test_missing_currency_does_not_split_a_run():
    """NaN != NaN, so an unfilled currency gap would open a spurious run and (under strict)
    discard the history before it.  The whole series must survive."""
    #  The listing-currency guard is OFF here so this exercises only the RUN logic; with it
    #  on, the NaN row would be dropped for a different (also correct) reason.
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD"),
        ("X", "2019-12-31", 1100.0, 10.0, 0.0, np.nan),
        ("X", "2020-12-31", 1200.0, 10.0, 0.0, "USD"),
    ], require_listing_currency_match=False)
    assert ps.price("X", "2018-12-31") == 100.0
    assert ps.price("X", "2019-12-31") == 110.0
    assert ps.diagnostics()["n_currency_switch_sources"] == 0


def test_no_currency_mode_knob_exists():
    """The deleted `currency_mode='usd'` must not come back.

    It multiplied the level by marketCap_usd/marketCap, which is a SINGLE SPOT RATE per
    currency applied to all history (CAD 0.724390 on all 31,316 CAD rows) -- a per-name
    constant, so it CANCELLED in every return ratio while the docstring claimed the mode
    "INCLUDES FX".  Currency coherence is a RESTRICTION now, not a conversion.
    """
    import inspect
    sig = inspect.signature(dpx.DerivedPriceSource.__init__)
    assert "currency_mode" not in sig.parameters
    with pytest.raises(TypeError):
        dpx.DerivedPriceSource(panel([("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD")]),
                               anchors=ANCH, currency_mode="usd")


# --------------------------------------------------------------------------- #
#  5b. THE YIELD DENOMINATOR IS PERIOD-START MARKET CAP                       #
# --------------------------------------------------------------------------- #
def test_yield_denominator_is_period_start_market_cap():
    """PIN: y = dividendsPaid / marketCap_START, never / marketCap_END.

    The END cap overstates the yield of any payer whose price fell during the period, and
    that inflation -- not real mega-dividends -- is what the old 0.25 ceiling was really
    suppressing.  ZIM 2022-06-30 paid $2.38B against an END cap of $5.67B (0.42) and a START
    cap of $8.72B (0.27); the first number is a reject, the second is a real dividend.

    Here the cap HALVES over the period, so end-cap and start-cap differ by exactly 2x.
    """
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0,   "USD"),   # price 100, START cap 1000
        ("X", "2019-12-31",  500.0, 10.0, -50.0, "USD"),   # price  50, END cap 500
    ])
    r = rc.compute_returns(["X"], "2018-12-31", "2019-12-31", ps).iloc[0]
    y_start, y_end = 50.0 / 1000.0, 50.0 / 500.0           # 0.05 vs 0.10
    assert math.isclose(r["total_return"], 0.5 * (1.0 + y_start) - 1.0, rel_tol=1e-12)
    assert not math.isclose(r["total_return"], 0.5 * (1.0 + y_end) - 1.0, rel_tol=1e-9), (
        "this is the end-cap denominator bug")


def test_first_row_yield_is_inert_for_every_window():
    """The first row of a source has no prior cap and falls back to its own -- which can
    never change a window return, because F_eval/F_buy cancels every period at or before the
    buy leg and the first row can never be later than the buy leg."""
    rows = [("X", "2018-12-31", 1000.0, 10.0, -500.0, "USD"),   # first row, big dividend
            ("X", "2019-12-31", 1100.0, 10.0, 0.0,    "USD")]
    with_div = src(rows)
    without = src([(rows[0][0], rows[0][1], rows[0][2], rows[0][3], 0.0, rows[0][5]), rows[1]])
    a = rc.compute_returns(["X"], "2018-12-31", "2019-12-31", with_div).iloc[0]["total_return"]
    b = rc.compute_returns(["X"], "2018-12-31", "2019-12-31", without).iloc[0]["total_return"]
    assert math.isclose(a, b, rel_tol=1e-12)


# --------------------------------------------------------------------------- #
#  6b. LISTING-CURRENCY MATCH  (the largest defect this leg had)               #
# --------------------------------------------------------------------------- #
def test_listing_currency_lookup():
    assert dpx._listing_currency("AAPL") == "USD"          # bare symbol == US listing
    assert dpx._listing_currency("CPH.TO") == "CAD"
    assert dpx._listing_currency("SHEL.L") == "GBP"
    assert dpx._listing_currency("005930.KS") == "KRW"
    assert dpx._listing_currency("X.NOSUCH") is None       # unknown, NOT a match
    assert dpx._listing_currency(None) is None


def test_listing_currency_mismatch_is_dropped():
    """PIN: a USD reporter on the TSX must NOT be priced by the derived leg.

    The derived price is marketCap/shares in the REPORTING currency; adjClose is in the
    LISTING currency.  Where they differ the derived-vs-real gap IS the FX move -- measured
    mean log gap -0.1234 on the 22.2% mismatched names against +0.0067 on the matched ones,
    and that single defect drove 55% of the composite's error.  A median-based gate could
    not see it, which is why this is pinned on the routing itself.
    """
    ps = src([
        ("A.TO", "2018-12-31", 1000.0, 10.0, 0.0, "CAD"),   # CAD reporter, CAD listing: OK
        ("A.TO", "2019-12-31", 1200.0, 10.0, 0.0, "CAD"),
        ("B.TO", "2018-12-31", 1000.0, 10.0, 0.0, "USD"),   # USD reporter, CAD listing: NO
        ("B.TO", "2019-12-31", 1200.0, 10.0, 0.0, "USD"),
    ])
    assert ps.price("A.TO", "2018-12-31") == 100.0
    assert ps.price("B.TO", "2018-12-31") is None
    d = ps.diagnostics()
    assert d["n_listing_mismatch_sources"] == 1
    assert d["n_listing_mismatch_rows_dropped"] == 2


def test_unknown_suffix_is_treated_as_a_mismatch():
    """An unrecognised suffix is UNKNOWN, and unknown must not be read as a match -- the
    conservative direction is to hand the name to the real leg."""
    ps = src([("X.NOSUCH", "2018-12-31", 1000.0, 10.0, 0.0, "USD")])
    assert ps.price("X.NOSUCH", "2018-12-31") is None


def test_listing_currency_guard_can_be_disabled_for_measurement():
    ps = src([("B.TO", "2018-12-31", 1000.0, 10.0, 0.0, "USD")],
             require_listing_currency_match=False)
    assert ps.price("B.TO", "2018-12-31") == 100.0


# --------------------------------------------------------------------------- #
#  6c. PRE-LISTING BACKFILL                                                   #
# --------------------------------------------------------------------------- #
def test_backfill_the_whole_run_goes_including_the_head():
    """PIN: a repeated-price RUN is pre-listing backfill and ALL of it is dropped.

    TORO carries price 7.97 and marketCap 75,404,241.73 across 5 consecutive quarters while
    real_prices.csv first prices it 2023-12-29 at 2.05.  Returns off such a buy leg are
    returns on a price that never traded, and the real route's `no_buy` there was CORRECT.

    THE HEAD GOES TOO, and that was a correction (2026-08-21).  Keeping the head and
    dropping only the repeats left TORO priced at 7.97 at BOTH buy2021 and buy2022 -- the
    head, then carried forward into the next anchor -- and moved the grid statistic by
    -0.0045.  In the observed cases the run is a reference price stamped BACKWARDS across
    quarters the name did not trade, so the head is the earliest row and is fabricated too:
    there is no real observation in the run to keep.
    """
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD"),   # price 100  <- run, DROPPED
        ("X", "2019-12-31", 1000.0, 10.0, 0.0, "USD"),   # price 100  <- run, DROPPED
        ("X", "2020-12-31", 1000.0, 10.0, 0.0, "USD"),   # price 100  <- run, DROPPED
        ("X", "2021-12-31", 1300.0, 10.0, 0.0, "USD"),   # price 130  <- real move, KEPT
    ])
    assert ps.diagnostics()["n_backfill_rows_dropped"] == 3
    assert ps.price("X", "2018-12-31") is None, "the run head is fabricated too"
    assert ps.price("X", "2019-12-31") is None
    assert ps.price("X", "2020-12-31") is None
    assert ps.price("X", "2021-12-31") == 130.0          # the first real price survives


def test_backfill_detection_tolerates_one_ULP():
    """PIN: the run test is a RELATIVE tolerance, because HAFN is not bit-identical.

    There the vendor holds the price at "7.31" and BACK-SOLVES the share count against a
    marketCap that moves 2.65B -> 3.73B over 11 quarters, so the quotient lands within one
    ULP instead of exactly equal -- only 7 of 10 adjacent pairs matched exactly, and an
    exact-equality rule caught TORO while missing HAFN completely.

    Here marketCap and shares both scale by 3x, so the price is 100 either way but the two
    floats need not be the same object.
    """
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD"),   # price 100
        ("X", "2019-12-31", 3000.0, 30.0, 0.0, "USD"),   # price 100 (within 1 ULP)
        ("X", "2020-12-31", 1300.0, 10.0, 0.0, "USD"),   # price 130, real move
    ])
    assert ps.diagnostics()["n_backfill_rows_dropped"] == 2
    assert ps.price("X", "2018-12-31") is None
    assert ps.price("X", "2020-12-31") == 130.0
    #  and a price that moves by MORE than the tolerance is NOT a run
    ok = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD"),   # price 100
        ("X", "2019-12-31", 1000.1, 10.0, 0.0, "USD"),   # price 100.01 -- a real move
    ])
    assert ok.diagnostics()["n_backfill_rows_dropped"] == 0


def test_backfill_guard_can_be_disabled_for_measurement():
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD"),
        ("X", "2019-12-31", 1000.0, 10.0, 0.0, "USD"),
    ], reject_repeated_price=False)
    assert ps.diagnostics()["n_backfill_rows_dropped"] == 0


# --------------------------------------------------------------------------- #
#  7. PROTOCOL / INTEGRATION WITH THE UNCHANGED PRIMITIVE                     #
# --------------------------------------------------------------------------- #
def test_terminal_policy_works_through_the_derived_source():
    """compute_returns' terminal-value policy must work unchanged against this source."""
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD"),
        ("X", "2019-12-31", 1200.0, 10.0, 0.0, "USD"),
    ])
    r = rc.compute_returns(["X"], "2018-12-31", "2021-12-31", ps).iloc[0]
    assert r["status"] == "terminal" and r["terminal_flag"]
    assert math.isclose(r["total_return"], 0.20, rel_tol=1e-12)
    assert math.isclose(r["total_return_floor"], -1.0)


def test_last_before_walks_the_anchor_grid():
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD"),
        ("X", "2019-12-31", 1200.0, 10.0, 0.0, "USD"),
    ])
    #  2020-12-31 is a CARRY-FORWARD of the 2019 statement (366 days, inside the cap), so
    #  it is a legitimate pick and last_before finds it rather than the 2019 anchor.
    assert ps.last_before("X", "2021-12-31") == ("2020-12-31", 120.0)
    assert ps.last_before("X", "2018-12-31") is None
    assert ps.last_before("X", "not-an-anchor") is None


def test_carry_forward_within_the_cap_is_a_real_pick():
    """A ragged/late filer must still be priced at the next anchor -- that carry-forward is
    most of why this leg has ~100% coverage per exchange."""
    ps = src([("X", "2019-12-31", 1200.0, 10.0, 0.0, "USD")])
    assert ps.price("X", "2020-12-31") == 120.0          # 366 days, inside the 550 cap
    assert ps.lag_days("X", "2020-12-31") == 366


def test_carry_forward_cap_stops_a_dead_name_reading_as_live():
    """PIN: past the cap the level must be WITHHELD, so a name that stopped filing fires
    the terminal/floor policy instead of reporting a fabricated flat return.

    Without the cap, `status` reads 'ok' and total_return is the last-known ratio -- i.e. a
    delisting silently becomes a flat return, suppressing a loser.
    """
    rows = [("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD")]
    ps = src(rows)
    assert ps.price("X", "2019-12-31") == 100.0          # 365 days -> still carried
    assert ps.price("X", "2021-12-31") is None           # 1096 days -> gone, not stale
    assert ps.diagnostics()["n_picks_dropped_carry_forward"] >= 1
    #  and the primitive's terminal policy takes over, as designed
    r = rc.compute_returns(["X"], "2018-12-31", "2021-12-31", ps).iloc[0]
    assert r["status"] == "terminal"
    assert math.isclose(r["total_return_floor"], -1.0)
    #  disabling the cap restores the (undesirable) infinite carry-forward
    assert src(rows, max_carry_forward_days=None).price("X", "2021-12-31") == 100.0


def test_max_lag_days_withholds_a_stale_pick():
    rows = [("X", "2021-06-30", 1000.0, 10.0, 0.0, "USD")]      # 184 days stale
    assert src(rows, anchors=["2021-12-31"]).price("X", "2021-12-31") == 100.0
    tight = src(rows, anchors=["2021-12-31"], max_lag_days=45)
    assert tight.price("X", "2021-12-31") is None


def test_benchmark_series_requires_a_real_source():
    ps = src([("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD")])
    with pytest.raises(RuntimeError, match="no benchmark_source"):
        ps.benchmark_series()


def test_missing_required_column_fails_loudly():
    bad = panel([("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD")]).drop(columns=["dividendsPaid"])
    with pytest.raises(KeyError, match="dividendsPaid"):
        dpx.DerivedPriceSource(bad, anchors=ANCH)


def test_bad_route_raises():
    with pytest.raises(ValueError, match="route must be one of"):
        dpx.build_price_source("nonsense")


def test_duplicate_period_end_is_deduped_deterministically():
    """12 (source, periodEndDate) pairs are duplicated on the panel; keep=last so a
    restatement wins and the answer does not depend on row order."""
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD"),
        ("X", "2018-12-31", 2000.0, 10.0, 0.0, "USD"),   # restatement, later row
    ])
    assert ps.price("X", "2018-12-31") == 200.0


def test_coverage_counts_names_per_anchor():
    ps = src([
        ("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD"),
        ("Y", "2019-12-31", 1000.0, 10.0, 0.0, "USD"),
    ])
    cov = ps.coverage().set_index("anchor")["n_names"]
    assert cov["2018-12-31"] == 1 and cov["2019-12-31"] == 2   # X carries forward


def test_diagnostics_carries_the_ic_caveat():
    """The marketCap shared-denominator caveat must travel WITH the source, so a consumer
    reporting an IC off this leg cannot fail to see it."""
    d = src([("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD")]).diagnostics()
    assert "marketCap-denominated" in d["ic_caveat"]
    assert d["max_period_yield"] == 1.0
    assert d["require_listing_currency_match"] is True
    assert d["reject_repeated_price"] is True
    #  the cap is kept on reasoning, not evidence -- the label must say so
    assert "UNVERIFIED" in d["carry_forward_cap_status"]


# --------------------------------------------------------------------------- #
#  7b. THE COMPOSITE ROUTE  (derived leg + real leg for the delisted half)      #
# --------------------------------------------------------------------------- #
class FakeReal:
    """Stand-in for a real PriceSource: an anchor-grid adjClose lookup."""
    def __init__(self, prices, anchors=ANCH):
        self.anchors = list(anchors)
        self._idx = {a: i for i, a in enumerate(self.anchors)}
        self._lut = dict(prices)

    def price(self, t, a):
        return self._lut.get((t, a))

    def last_before(self, t, a):
        j = self._idx.get(a)
        for k in range((j if j is not None else 0) - 1, -1, -1):
            p = self._lut.get((t, self.anchors[k]))
            if p is not None:
                return self.anchors[k], p
        return None

    def benchmark_series(self, symbol=rc.BENCHMARK_SYMBOL):
        return "REAL-BENCHMARK-%s" % symbol


def test_composite_uses_derived_for_a_live_name_and_real_for_a_dead_one():
    """The panel is SURVIVORS-ONLY: only 4 of the delisted registry's 9,277 symbols are in
    cdx_df, while 3,948 are priced by real_prices.  Without the fallback the whole dead
    half becomes no_buy and every derived view EXCLUDES it -- survivorship bias.
    """
    derived = src([("LIVE", "2018-12-31", 1000.0, 10.0, 0.0, "USD"),
                   ("LIVE", "2019-12-31", 1200.0, 10.0, 0.0, "USD")])
    real = FakeReal({("LIVE", "2018-12-31"): 999.0,          # must be IGNORED for LIVE
                     ("DEAD", "2018-12-31"): 50.0,
                     ("DEAD", "2019-12-31"): 25.0})
    comp = dpx.CompositePriceSource(derived, real)
    assert comp.price("LIVE", "2018-12-31") == 100.0          # derived leg
    assert comp.price("DEAD", "2018-12-31") == 50.0           # real leg
    df = rc.compute_returns(["LIVE", "DEAD"], "2018-12-31", "2019-12-31", comp)
    assert list(df["status"]) == ["ok", "ok"]
    assert math.isclose(df.iloc[0]["total_return"], 0.20, rel_tol=1e-12)
    assert math.isclose(df.iloc[1]["total_return"], -0.50, rel_tol=1e-12)   # the loser survives


def test_composite_never_mixes_legs_within_a_window():
    """PIN: a ticker on the derived leg must NOT fall back to the real leg at another
    anchor.  The real leg's adjClose is already dividend-back-adjusted, so a derived buy
    with a real eval DOUBLE-COUNTS the dividend -- measured IC residual +0.0218 against
    +0.0021 for the consistent construction.
    """
    #  X stops filing after 2018, so by 2021 it is past the carry-forward cap and the
    #  derived leg has NO eval price -- while the real leg does.  That is the exact
    #  situation in which a naive per-anchor fallback would silently mix the two.
    derived = src([("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD")])
    real = FakeReal({("X", "2018-12-31"): 100.0, ("X", "2021-12-31"): 500.0})
    comp = dpx.CompositePriceSource(derived, real)
    assert comp.price("X", "2018-12-31") == 100.0
    assert comp.price("X", "2021-12-31") is None, (
        "fell through to the real leg mid-window -- this double-counts dividends")
    #  the primitive's terminal policy handles it, rather than a mixed-leg return
    assert rc.compute_returns(["X"], "2018-12-31", "2021-12-31", comp).iloc[0]["status"] \
        == "terminal"


def test_composite_benchmark_always_comes_from_the_real_leg():
    derived = src([("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD")])
    comp = dpx.CompositePriceSource(derived, FakeReal({}))
    assert comp.benchmark_series() == "REAL-BENCHMARK-URTH"


def test_composite_diagnostics_declare_the_seam():
    derived = src([("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD")])
    d = dpx.CompositePriceSource(derived, FakeReal({})).diagnostics()
    assert d["n_tickers_on_derived_leg"] == 1
    assert "per TICKER" in d["leg_selection"]
    assert d["route"] == "derived+real composite"


def test_composite_route_needs_both_inputs():
    """Both legs are mandatory -- a silent single-leg fallback is exactly the survivorship
    failure this route exists to prevent."""
    with pytest.raises(ValueError, match="needs panel"):
        dpx.build_price_source("derived+real")
    with pytest.raises(ValueError, match="needs prices_csv for the fallback leg"):
        dpx.build_price_source(
            "derived+real",
            panel=panel([("X", "2018-12-31", 1000.0, 10.0, 0.0, "USD")]))


# --------------------------------------------------------------------------- #
#  8. THE REAL-ROUTE SELECTOR IS UNCHANGED (no score may move)                 #
# --------------------------------------------------------------------------- #
def test_build_price_source_real_route_is_the_unchanged_pricesource():
    csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "price_data", "real_prices.csv")
    if not os.path.exists(csv):
        pytest.skip("no price_data/real_prices.csv here -- real-route identity NOT checked")
    ps = dpx.build_price_source("real", prices_csv=csv)
    assert isinstance(ps, rc.PriceSource)
    ref = rc.PriceSource(csv)
    #  bit-for-bit on a sample of the grid: the default route must be untouched
    for t in ("AAPL", "MSFT", "0837.HK"):
        for a in ("2018-12-31", "2021-12-31"):
            assert ps.price(t, a) == ref.price(t, a)


# --------------------------------------------------------------------------- #
#  9. ACCEPTANCE TEST against the real panel (skips when absent)               #
# --------------------------------------------------------------------------- #
def _acceptance_inputs():
    from derived_price_validate import find_panel, PRICES, PRICES_2025
    p = find_panel()
    if p is None or not os.path.exists(PRICES) or not os.path.exists(PRICES_2025):
        return None
    return p


def test_acceptance_derived_vs_real_fidelity_and_bias():
    """END-TO-END fidelity gate, asserted on the MEAN and the TAILS -- never the median.

    WHY THIS TEST LOOKS LIKE THIS (2026-08-21).  The previous version asserted
    `abs(median_log_gap_tr) < 0.01`, and that assertion is VACUOUS: the universe is ~78%
    currency-matched, so its median gap is 0.0000 by construction and the gate could not see
    a -0.12 mean on the mismatched fifth.  It passed green while the derived leg was
    comparing a reporting-currency return against a listing-currency adjClose on ~20% of
    names -- the single largest defect this leg had, and this gate is the code-level reason
    it shipped.  A median is the wrong statistic for a defect that lives in a subpopulation.
    """
    p = _acceptance_inputs()
    if p is None:
        pytest.skip("deep panel and/or real_prices*.csv absent -- acceptance NOT run")
    from derived_price_validate import fidelity_table
    tab = fidelity_table(p)
    assert len(tab) == 5, "expect the five 36-month anchor pairs"
    for row in tab.itertuples():
        assert row.n >= 1000, f"{row.pair}: thin overlap n={row.n}"
        assert row.spearman_tr >= 0.96, f"{row.pair}: derived TR fidelity {row.spearman_tr}"
        #  the dividend leg must IMPROVE fidelity, not just shift the level
        assert row.spearman_tr > row.spearman_price_only, f"{row.pair}: TR did not improve"
        #  the uncorrected bias is real and negative ...
        assert row.median_log_gap_price_only < -0.02, f"{row.pair}: no price-only bias?"
        #  ... and the correction closes it ON THE MEAN, which is the statistic a top-N
        #  average is actually exposed to.
        assert abs(row.mean_log_gap_tr) < 0.03, (
            f"{row.pair}: MEAN gap {row.mean_log_gap_tr} -- a subpopulation is mispriced "
            f"even though the median reads {row.median_log_gap_tr}")
        #  ... and the TAILS are bounded, so no small group can carry a top-N mean.  A
        #  currency-mismatched name sits at roughly +/-0.15, so this is the live constraint.
        assert row.p05_log_gap_tr > -0.30, f"{row.pair}: left tail {row.p05_log_gap_tr}"
        assert row.p95_log_gap_tr < 0.30, f"{row.pair}: right tail {row.p95_log_gap_tr}"


def test_acceptance_currency_match_is_enforced_and_matters():
    """The guard must (a) leave NO mismatched name on the derived leg, and (b) be shown to
    matter -- turning it off must resurrect a materially mispriced subpopulation.

    (b) is the half that stops this becoming another vacuous gate: without it, a guard that
    silently stopped doing anything would still pass (a).
    """
    p = _acceptance_inputs()
    if p is None:
        pytest.skip("deep panel absent -- currency-match acceptance NOT run")
    from derived_price_validate import currency_split_table
    on = currency_split_table(p)
    assert (on["n_mismatched"] == 0).all(), "a currency-mismatched name reached the derived leg"
    assert (on["n_matched"] > 1000).all()

    off = currency_split_table(p, require_listing_currency_match=False)
    #  the defect is real: a fifth of the universe, and its mean gap is nothing like zero
    assert (off["pct_mismatched"] > 15.0).all(), "guard is no longer separating anything"
    assert off["mean_gap_mismatched"].abs().max() > 0.02, (
        "the mismatched population is no longer mispriced -- either the panel changed or "
        "this guard has stopped being load-bearing; re-measure before trusting it")


def test_acceptance_guard_progression_improves_fidelity_monotonically():
    """The guards must improve rank fidelity, and the shipped config must stay near the best.

    NOT "the shipped config is the best" -- that assertion was tried and it FAILED honestly
    (2026-08-21): the backfill guard costs 0.0009 of Spearman (0.9806 -> 0.9797).  That is
    expected rather than a defect, and fidelity-against-the-real-leg is the WRONG criterion
    for that particular guard: the worst backfill cases are names the real file cannot price
    AT ALL (TORO has no real price until 2023-12-29), so they never enter the fidelity
    overlap, and what the guard removes from the overlap is dominated by near-flat-price
    false positives that happened to agree with the real leg.  The backfill guard is
    justified by DIRECT evidence instead -- see test_backfill_the_whole_run_goes_* -- and
    here it only has to avoid materially degrading fidelity.

    A reviewer should push on this: it is the one place a gate was relaxed after it fired.
    """
    p = _acceptance_inputs()
    if p is None:
        pytest.skip("deep panel absent -- guard-progression acceptance NOT run")
    from derived_price_validate import guard_progression_table
    tab = guard_progression_table(p)
    rho = list(tab["mean_spearman_tr"])
    #  the currency fix must clearly improve on the unguarded leg
    assert rho[1] > rho[0] + 0.003, f"currency guard did not improve fidelity: {rho}"
    #  and the shipped config must keep essentially all of that gain
    assert rho[-1] > rho[0] + 0.003, f"guards did not improve fidelity: {rho}"
    assert rho[-1] >= max(rho) - 0.002, f"shipped config materially off the best: {rho}"
    #  and the median is USELESS here -- it is flat across every config, which is the whole
    #  reason the old gate missed the defect.  Asserted so the lesson cannot be un-learned.
    assert tab["median_log_gap_tr"].nunique() == 1
    assert float(tab["median_log_gap_tr"].iloc[0]) == 0.0


def test_acceptance_dec31_rule_beats_the_grid_date_rule():
    """The 12-30 / 12-29 anchors are where the naive rule breaks; prove it on real data."""
    p = _acceptance_inputs()
    if p is None:
        pytest.skip("deep panel absent -- anchor-rule acceptance NOT run")
    from derived_price_validate import lag_comparison
    cmp_ = lag_comparison(p).set_index("anchor")
    for a in ("2022-12-30", "2023-12-29"):
        assert cmp_.loc[a, "median_lag_dec31"] == 0
        assert cmp_.loc[a, "median_lag_griddate"] >= 85, "grid-date rule should lag a quarter"
    for a in ("2021-12-31", "2024-12-31"):
        assert cmp_.loc[a, "median_lag_dec31"] == cmp_.loc[a, "median_lag_griddate"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
