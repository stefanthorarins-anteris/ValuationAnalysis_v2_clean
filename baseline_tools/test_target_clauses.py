"""The TWO-CLAUSE target: the DOWNSIDE clause, its three diagnostics, and the wiring.

WHY THESE EXIST.  The charter set a two-clause target on 2026-08-20 and only the UPSIDE clause
was ever built.  The run printed a beat-rate and it read as the target for six days.  The
tests below are written so that shipping half the target again FAILS rather than looks fine:
the last one asserts the pipeline stage returns and prints BOTH clauses, and it fails against
HEAD ca71e05 where no bond bar, no p25-of-picks and no below-zero count existed anywhere in
the tree.

OFFLINE.  No network, no price CSV, no pipeline run -- every test is a function of a
hand-built returns table or a fake price source.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import returns_core as rc
import target_clauses as tc


# --------------------------------------------------------------------------- #
#  helpers -- build a returns table with the exact status mix we want to test  #
# --------------------------------------------------------------------------- #
def _rdf(ok=(), terminal=(), no_buy=0, buy_only=0):
    """A returns_core-shaped table.

    ok       : iterable of realised total returns ('ok' rows -- both legs priced)
    terminal : iterable of PRIMARY returns for names whose EVAL leg is missing.  Under FLOOR
               these become -100%, which is exactly the interior-price-hole case the run
               flags (priced before AND after the missing anchor).
    no_buy   : how many picks had no BUY price at all -- they were never opened, so their
               return is UNKNOWN.  They are the coverage gap, not a loss.
    buy_only : how many picks were priced at the BUY anchor ONLY -- no eval leg and nothing
               earlier to fall back to.  `compute_returns` emits these with
               `terminal = p_buy`, so `total_return` is EXACTLY 0.0 by construction.  That
               0.0 is a fallback, not an observation, and must not count as coverage.
    """
    rows = []
    for i, r in enumerate(ok):
        rows.append((f"OK{i}", 100.0, 100.0 * (1 + r), 100.0 * (1 + r), r, r, False, "ok"))
    for i, r in enumerate(terminal):
        rows.append((f"TM{i}", 100.0, np.nan, 100.0 * (1 + r), r, -1.0, True, "terminal"))
    for i in range(no_buy):
        rows.append((f"NB{i}", np.nan, np.nan, np.nan, np.nan, np.nan, False, "no_buy"))
    for i in range(buy_only):
        #  EXACTLY what returns_core.compute_returns emits on the `lb is None` branch:
        #  terminal falls back to the buy price, so the return is a fabricated 0.0.
        rows.append((f"BO{i}", 100.0, np.nan, 100.0, 0.0, -1.0, True, "terminal"))
    return pd.DataFrame(rows, columns=rc.RETURNS_COLS)


# --------------------------------------------------------------------------- #
#  1. THE BAR ITSELF                                                          #
# --------------------------------------------------------------------------- #
def test_the_bond_bar_is_the_charters_9_27_percent():
    """The charter writes the clause two ways -- 'flat 3%/yr' and '>= 9.27% compounded over
    the window'.  They must be the same number, or the printed bar and the written target
    disagree and nobody can tell which one a verdict used."""
    assert tc.bond_bar(36) == pytest.approx(1.03 ** 3 - 1.0)
    assert round(tc.bond_bar(36) * 100, 2) == 9.27


def test_the_bar_follows_the_horizon_instead_of_being_a_frozen_constant():
    """9.27% hardcoded would silently mis-grade any non-36-month read.  A shorter horizon must
    face a smaller bar."""
    assert tc.bond_bar(12) == pytest.approx(0.03)
    assert tc.bond_bar(24) < tc.bond_bar(36)


def test_the_bar_is_flat_not_the_buy_date_treasury():
    """CEO decision, and it is load-bearing: the bar must NOT vary by anchor.  A future
    'improvement' to a floating rate changes what every historical verdict means, so it is
    pinned here and the cost is carried in the printed caveat instead."""
    assert tc.BOND_RATE_ANNUAL == 0.03
    assert "not the buy-date treasury" in tc.BOND_BAR_CAVEAT.lower()
    #  and the accepted cost travels with it rather than being rediscovered
    assert "2022" in tc.BOND_BAR_CAVEAT and "2020" in tc.BOND_BAR_CAVEAT


# --------------------------------------------------------------------------- #
#  2. THE CLAUSE, AT FULL COVERAGE (the only case where a number IS the clause) #
# --------------------------------------------------------------------------- #
def test_full_coverage_gives_a_definitive_PASS():
    d = tc.downside_clause(_rdf(ok=[0.20] * 20), depth_n=20)
    assert d["coverage"] == 1.0
    assert d["portfolio_return"] == pytest.approx(0.20)
    assert d["verdict"] == "PASS"


def test_full_coverage_gives_a_definitive_FAIL():
    d = tc.downside_clause(_rdf(ok=[0.05] * 20), depth_n=20)
    assert d["verdict"] == "FAIL"      # 5% < the 9.27% bond bar


def test_the_clause_is_the_EQUAL_WEIGHT_PORTFOLIO_not_a_count_of_winners():
    """The two clauses answer different questions and this is the one that separates them: a
    list where 19 names are flat and one triples CLEARS the portfolio bar while its beat-rate
    is 5%.  If this ever fails, the downside clause has been quietly turned into a second
    hit-rate and the pair has stopped doing its job."""
    d = tc.downside_clause(_rdf(ok=[0.0] * 19 + [3.0]), depth_n=20)
    assert d["portfolio_return"] == pytest.approx(3.0 / 20)
    assert d["verdict"] == "PASS"


def test_the_softness_is_PRESERVED_not_quietly_fixed():
    """The CEO chose the SOFTEST of four candidate clauses knowing a single strong name can
    carry it.  A diagnostic must never gate: here the worst pick is -90% and the 5th-worst is
    under water, and the clause still PASSES.  Anyone tightening the clause has to change the
    charter first, and this test is what makes that visible."""
    d = tc.downside_clause(_rdf(ok=[-0.90] * 5 + [0.0] * 10 + [1.50] * 5), depth_n=20)
    dg = tc.diagnostics(_rdf(ok=[-0.90] * 5 + [0.0] * 10 + [1.50] * 5), tc.bond_bar(36))
    assert d["verdict"] == "PASS"
    assert dg["worst_clears_bar"] is False
    assert dg["p25_clears_bar"] is False
    assert dg["n_below_zero"] == 5


# --------------------------------------------------------------------------- #
#  3. COVERAGE -- the part that actually bites this project                    #
# --------------------------------------------------------------------------- #
def test_missing_picks_make_the_clause_INDETERMINATE_not_a_FAIL():
    """THE FAILURE THIS PREVENTS.  Reporting 'the portfolio returned 4%, below the bond' when
    9 of the 20 picks could not be priced is reporting a PRICE-GRID defect as a FILTER defect.
    A missing pick is unbounded above, so the priced shortfall cannot decide the clause."""
    d = tc.downside_clause(_rdf(ok=[0.04] * 11, no_buy=9), depth_n=20)
    assert d["n_priced"] == 11 and d["n_no_buy"] == 9
    assert d["coverage"] == pytest.approx(11 / 20)
    assert d["verdict"] == "INDETERMINATE"
    assert "unpriced" in d["verdict_reason"]


def test_a_verdict_that_survives_the_worst_case_is_still_a_PASS():
    """INDETERMINATE must not become a way to never conclude anything.  When the portfolio
    clears the bar even with every unpriced pick marked at -100%, the missing names cannot
    overturn it and the answer is PASS."""
    d = tc.downside_clause(_rdf(ok=[1.00] * 18, no_buy=2), depth_n=20)
    assert d["lower_bound"] == pytest.approx((18 * 1.00 - 2) / 20)
    assert d["lower_bound"] >= d["bar"]
    assert d["verdict"] == "PASS"


def test_flip_return_is_what_makes_an_INDETERMINATE_actionable():
    """`flip_return` is the average return the UNPRICED picks would need for the full
    portfolio to clear the bar.  Checked by substituting it back: a portfolio where the
    missing names return exactly that lands exactly on the bar.  +6% means the verdict is a
    real coin-flip; +400% means it is a FAIL in all but name."""
    d = tc.downside_clause(_rdf(ok=[0.02] * 15, no_buy=5), depth_n=20)
    assert d["verdict"] == "INDETERMINATE"
    x = d["flip_return"]
    reconstructed = (15 * 0.02 + 5 * x) / 20
    assert reconstructed == pytest.approx(d["bar"])


def test_a_no_buy_pick_is_NOT_counted_as_a_zero_or_a_total_loss_in_the_point_estimate():
    """A pick with no buy price was never opened. Folding it in as 0% (or as -100%) invents a
    position that was never taken; it belongs in the coverage count instead."""
    with_gap = tc.downside_clause(_rdf(ok=[0.30] * 10, no_buy=10), depth_n=20)
    without = tc.downside_clause(_rdf(ok=[0.30] * 10), depth_n=10)
    assert with_gap["portfolio_return"] == pytest.approx(without["portfolio_return"])
    assert with_gap["n_no_buy"] == 10 and without["n_no_buy"] == 0


def test_nothing_priced_is_INDETERMINATE_rather_than_a_zero():
    d = tc.downside_clause(_rdf(no_buy=20), depth_n=20)
    assert d["verdict"] == "INDETERMINATE"
    assert d["portfolio_return"] != d["portfolio_return"]     # NaN, not 0.0


# --------------------------------------------------------------------------- #
#  4. PRIMARY vs FLOOR -- real losses separated from unpriceable ones          #
# --------------------------------------------------------------------------- #
def test_floor_and_primary_differ_ONLY_via_the_terminal_names():
    """An interior price hole -- a name priced before AND after a missing anchor -- is floored
    at -100% although a later price proves it alive.  So FLOOR is a LOWER BOUND on the
    portfolio, never a reading of it, and both must be available side by side."""
    r = _rdf(ok=[0.10] * 18, terminal=[0.10, 0.10])
    prim = tc.downside_clause(r, depth_n=20, floor=False)
    flr = tc.downside_clause(r, depth_n=20, floor=True)
    assert prim["portfolio_return"] == pytest.approx(0.10)
    assert flr["portfolio_return"] == pytest.approx((18 * 0.10 + 2 * -1.0) / 20)
    assert flr["portfolio_return"] < prim["portfolio_return"]
    assert prim["n_terminal"] == flr["n_terminal"] == 2


def test_the_two_policies_can_disagree_on_the_verdict_and_both_are_reported():
    """The case that must never be collapsed into one number: PASS on the prices we have,
    FAIL if every unpriceable name is assumed dead.  Reporting either one alone is a claim the
    data does not support."""
    r = _rdf(ok=[0.15] * 16, terminal=[0.15] * 4)
    assert tc.downside_clause(r, 20, floor=False)["verdict"] == "PASS"
    assert tc.downside_clause(r, 20, floor=True)["verdict"] == "FAIL"


def test_below_zero_count_reads_differently_under_the_two_policies():
    """'4 picks ended below zero' is a different sentence when the 4 are unpriceable rather
    than down. The diagnostic carries its n and its policy for exactly this reason."""
    r = _rdf(ok=[0.15] * 16, terminal=[0.15] * 4)
    assert tc.diagnostics(r, tc.bond_bar(36), floor=False)["n_below_zero"] == 0
    assert tc.diagnostics(r, tc.bond_bar(36), floor=True)["n_below_zero"] == 4


# --------------------------------------------------------------------------- #
#  5. THE DIAGNOSTICS                                                          #
# --------------------------------------------------------------------------- #
def test_p25_is_the_5th_worst_of_20_as_the_charter_glosses_it():
    """The charter says "p25 of the 20 picks -- the 5th-worst must clear it", so p25 is an
    ORDER STATISTIC.  Built so an interpolating np.percentile would disagree: the 5th and 6th
    smallest are far apart, and linear interpolation lands between them."""
    vals = [-0.50, -0.40, -0.30, -0.20, -0.10] + [0.90] * 15
    d = tc.diagnostics(_rdf(ok=vals), tc.bond_bar(36))
    assert d["n"] == 20 and d["p25_rank"] == 5
    assert d["p25"] == pytest.approx(-0.10)                       # the 5th-smallest
    assert np.percentile(sorted(vals), 25) != pytest.approx(-0.10)  # interpolation differs


def test_p25_rank_is_reported_so_the_reading_is_checkable_at_any_n():
    """The run is thin: a p25 over 8 priced picks is the 2nd-smallest, not the 5th-worst.
    Printing the rank is what stops that being read as the chartered diagnostic."""
    assert tc.diagnostics(_rdf(ok=[0.1] * 8), tc.bond_bar(36))["p25_rank"] == 2
    assert tc.diagnostics(_rdf(ok=[0.1] * 20), tc.bond_bar(36))["p25_rank"] == 5


def test_every_diagnostic_carries_its_n():
    """'3 picks below zero' out of 8 priced is not the same statement as out of 20, and the
    thinness IS the binding problem on this run."""
    d = tc.diagnostics(_rdf(ok=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), tc.bond_bar(36))
    assert d["n"] == 8 and d["n_below_zero"] == 3
    assert d["share_below_zero"] == pytest.approx(3 / 8)


def test_the_worst_pick_is_tracked_for_magnitude_and_is_expected_to_fail():
    d = tc.diagnostics(_rdf(ok=[-0.85] + [1.0] * 19), tc.bond_bar(36))
    assert d["worst"] == pytest.approx(-0.85)
    assert d["worst_clears_bar"] is False


def test_diagnostics_on_an_empty_set_refuse_rather_than_return_zero():
    d = tc.diagnostics(_rdf(no_buy=20), tc.bond_bar(36))
    assert d["n"] == 0 and d["p25"] != d["p25"]
    assert d["p25_clears_bar"] is None and d["worst_clears_bar"] is None


# --------------------------------------------------------------------------- #
#  6. THE PAIR -- both clauses must pass for a period to count                 #
# --------------------------------------------------------------------------- #
def test_both_clauses_must_pass_for_the_period_to_pass():
    #  13 of 20 beat by >=10pp against a flat benchmark -> 65% >= 60%, at FULL coverage.
    up_pass = tc.upside_clause(_rdf(ok=[0.50] * 13 + [0.0] * 7), 0.0, 20)
    up_fail = tc.upside_clause(_rdf(ok=[0.50] * 6 + [0.0] * 14), 0.0, 20)
    dn_pass = tc.downside_clause(_rdf(ok=[0.20] * 20), 20)
    dn_fail = tc.downside_clause(_rdf(ok=[0.01] * 20), 20)
    assert up_pass["verdict"] == "PASS" and up_fail["verdict"] == "FAIL"
    assert tc.period_verdict(up_pass, dn_pass) == "PASS"
    assert tc.period_verdict(up_pass, dn_fail) == "FAIL"
    assert tc.period_verdict(up_fail, dn_pass) == "FAIL"


def test_the_clauses_cannot_be_traded_against_each_other():
    """The stated reason the pair exists: a filter can buy a hit rate with tail risk.  A list
    that clears 60% on the upside while the portfolio is under the bond must NOT count as a
    success, however good the beat-rate looks."""
    picks = _rdf(ok=[0.60] * 16 + [-0.75] * 4)
    up = tc.upside_clause(picks, 0.0, 20)
    dn = tc.downside_clause(_rdf(ok=[0.60] * 12 + [-0.75] * 8), 20)
    assert up["verdict"] == "PASS"          # 16/20 = 80% beat the +10pp bar
    assert dn["verdict"] == "FAIL"
    assert tc.period_verdict(up, dn) == "FAIL"


def test_an_unmeasURABLE_clause_makes_the_PERIOD_indeterminate_not_a_pass():
    """A period scoring well on the half we can measure has not been shown to succeed.  The
    single-clause target would have called this a win, which is the whole reason for the pair."""
    picks = _rdf(ok=[0.50] * 7 + [0.0] * 4, no_buy=9)
    up = tc.upside_clause(picks, 0.0, 20)
    dn = tc.downside_clause(_rdf(ok=[0.04] * 11, no_buy=9), 20)
    assert up["verdict"] == "INDETERMINATE"     # lo=35%, hi=80% straddles the 60% bar
    assert dn["verdict"] == "INDETERMINATE"
    assert tc.period_verdict(up, dn) == "INDETERMINATE"


def test_an_empty_upside_is_INDETERMINATE_rather_than_a_silent_fail():
    assert tc.upside_clause(_rdf(no_buy=20), 0.0, 20)["verdict"] == "INDETERMINATE"
    #  and a NaN benchmark cannot be read as a fail either
    assert tc.upside_clause(_rdf(ok=[0.5] * 20), float("nan"), 20)["verdict"] == "INDETERMINATE"


# --------------------------------------------------------------------------- #
#  6b. THE UPSIDE CLAUSE GETS THE SAME COVERAGE DISCIPLINE AS THE DOWNSIDE ONE #
#      -- and the asymmetry between them is the reason it is a separate block. #
# --------------------------------------------------------------------------- #
def test_the_upside_clause_reports_an_INTERVAL_under_partial_coverage():
    """A beat-rate is a proportion, so every unmeasured pick either beats or does not: the
    honest reading is bounds over the picks the anchor SHIPPED, not a rate over the subset that
    happened to be priceable."""
    up = tc.upside_clause(_rdf(ok=[0.50] * 2 + [0.0] * 7, no_buy=11), 0.0, 20)
    assert up["n_measured"] == 9 and up["n_beat"] == 2
    assert up["lo"] == pytest.approx(2 / 20)          # every unmeasured pick fails
    assert up["hi"] == pytest.approx((2 + 11) / 20)   # every unmeasured pick beats


def test_THE_LIVE_SHAPE_is_undecidable_and_must_not_read_as_FAIL():
    """THE DEFECT THIS CLOSES, at the numbers it actually occurred at.  On the 2026-08-27 run
    roughly 2 of 9 measured picks beat the bar with 11 of 20 unmeasured.  The bounds are
    [10%, 65%], which STRADDLE the 60% bar -- so the anchor is genuinely undecidable, and the
    shipped code called it FAIL and let that sink the whole period."""
    up = tc.upside_clause(_rdf(ok=[0.50] * 2 + [0.0] * 7, no_buy=11), 0.0, 20)
    assert (round(up["lo"] * 100, 1), round(up["hi"] * 100, 1)) == (10.0, 65.0)
    assert up["lo"] < 0.60 < up["hi"]
    assert up["verdict"] == "INDETERMINATE"


def test_a_partial_coverage_upside_FAIL_is_still_PROVABLE():
    """The asymmetry that makes the upside clause stronger than the downside one: if even the
    most generous assignment of the unmeasured picks cannot reach the bar, the anchor has
    failed whatever those picks did.  INDETERMINATE must not become a way to never conclude."""
    up = tc.upside_clause(_rdf(ok=[0.0] * 9, no_buy=11), 0.0, 20)
    assert up["n_measured"] == 9 and up["n_beat"] == 0 and up["n_missing"] == 11
    #  n_missing is n_selected - n_measured, NOT the no_buy count -- an earlier draft of this
    #  test confused the two and asserted 2/20.  Zero measured beats plus ALL 11 unmeasured
    #  assumed to beat is 55%, still short of the 60% bar, so the FAIL is provable.
    assert up["hi"] == pytest.approx(11 / 20)
    assert up["hi"] < up["bar"]
    assert up["verdict"] == "FAIL"
    assert "even if every unmeasured pick beat" in up["verdict_reason"]


def test_a_partial_coverage_upside_PASS_is_provable_too():
    up = tc.upside_clause(_rdf(ok=[0.50] * 13, no_buy=7), 0.0, 20)
    assert up["lo"] == pytest.approx(13 / 20)
    assert up["verdict"] == "PASS"


def test_the_period_can_no_longer_read_FAIL_while_a_clause_is_INDETERMINATE():
    """THE CONTRADICTION, pinned.  The shipped output printed `DOWNSIDE: INDETERMINATE` and
    `PERIOD: FAIL` on one line, because the upside clause was handed a rate over the priced
    subset and `period_verdict` lets FAIL dominate.  With both clauses under the same
    discipline, an undecidable anchor is undecidable end to end."""
    picks = _rdf(ok=[0.50] * 2 + [0.0] * 7, no_buy=11)
    up = tc.upside_clause(picks, 0.0, 20)
    dn = tc.downside_clause(picks, 20)
    assert dn["verdict"] == "INDETERMINATE"
    assert up["verdict"] == "INDETERMINATE"
    assert tc.period_verdict(up, dn) == "INDETERMINATE"


# --------------------------------------------------------------------------- #
#  6c. A PICK PRICED AT THE BUY ANCHOR ONLY IS NOT A MEASUREMENT               #
# --------------------------------------------------------------------------- #
def test_a_buy_only_pick_does_NOT_count_as_coverage():
    """`compute_returns` falls back to `terminal = p_buy` when there is no eval leg and nothing
    earlier, producing a return of exactly 0.0.  That is "we know nothing", not "flat"."""
    d = tc.downside_clause(_rdf(ok=[0.20] * 15, buy_only=5), 20)
    assert d["n_buy_only"] == 5
    assert d["n_measured"] == 15
    assert d["coverage"] == pytest.approx(15 / 20)
    assert d["verdict"] != "FAIL"          # cannot be definitive at partial coverage


def test_a_buy_only_pick_can_no_longer_fake_FULL_COVERAGE():
    """THE DEFECT: every pick present in the frame, `coverage == 1.0`, and a verdict whose
    stated reason is "the portfolio return IS the chartered clause" -- resting on names nothing
    priced."""
    d = tc.downside_clause(_rdf(ok=[0.20] * 15, buy_only=5), 20)
    assert d["coverage"] != 1.0
    assert "full coverage" not in d["verdict_reason"]


def test_a_buy_only_pick_cannot_MANUFACTURE_a_PASS():
    """The safety property behind the detection heuristic, asserted rather than assumed.

    The test is `terminal_flag and terminal_adjClose == buy_adjClose`, whose one false-positive
    mode is a genuine last-before price bit-identical to the buy price.  Reclassifying a pick
    as unmeasured must only ever weaken a verdict: it lowers the strict lower bound (the only
    route to a partial-coverage downside PASS) and lowers `lo` while leaving `hi` unchanged on
    the upside.  Here the fabricated zeros would have DRAGGED the portfolio under the bar, so
    excluding them RAISES the point estimate -- and the verdict still must not become PASS."""
    r = _rdf(ok=[0.30] * 10, buy_only=10)
    d = tc.downside_clause(r, 20)
    assert d["portfolio_return"] == pytest.approx(0.30)       # excluded from the mean
    assert d["portfolio_return"] > d["bar"]                   # and it clears, on its face
    assert d["verdict"] == "INDETERMINATE"                    # but coverage forbids PASS
    assert d["lower_bound"] < d["bar"]


def test_a_buy_only_pick_is_excluded_from_all_three_diagnostics():
    """A fabricated 0.0 sits above the worst pick, below the median and counts as "not below
    zero" -- polluting every diagnostic at once with a number nobody observed."""
    with_bo = tc.diagnostics(_rdf(ok=[-0.50] * 4 + [0.80] * 4, buy_only=12), tc.bond_bar(36))
    without = tc.diagnostics(_rdf(ok=[-0.50] * 4 + [0.80] * 4), tc.bond_bar(36))
    assert with_bo["n"] == without["n"] == 8
    assert with_bo["p25"] == pytest.approx(without["p25"])
    assert with_bo["n_below_zero"] == without["n_below_zero"] == 4


def test_a_GENUINE_terminal_is_still_measured():
    """The heuristic must not swallow real terminals.  A name with a real last-before price
    different from its buy price is an observation and stays in the denominator."""
    d = tc.downside_clause(_rdf(ok=[0.10] * 18, terminal=[-0.40, -0.40]), 20)
    assert d["n_buy_only"] == 0
    assert d["n_measured"] == 20 and d["coverage"] == 1.0
    assert d["n_terminal"] == 2


# --------------------------------------------------------------------------- #
#  7. THE WIRING -- the run must never again print half a target               #
#     FAILS AGAINST HEAD ca71e05: no downside clause existed in the tree.      #
# --------------------------------------------------------------------------- #
class _FakePriceSource:
    """Minimal PriceSource surface: `.price`, `.last_before`, `.benchmark_series`."""

    def __init__(self, lut, anchors):
        self._lut, self.anchors = lut, list(anchors)

    def price(self, ticker, anchor):
        return self._lut.get((ticker, anchor))

    def last_before(self, ticker, anchor):
        j = self.anchors.index(anchor)
        for k in range(j - 1, -1, -1):
            p = self._lut.get((ticker, self.anchors[k]))
            if p is not None:
                return self.anchors[k], p
        return None

    def benchmark_series(self, symbol=rc.BENCHMARK_SYMBOL):
        rows = [(a, self._lut[(symbol, a)]) for a in self.anchors
                if (symbol, a) in self._lut]
        return pd.Series({pd.Timestamp(a): v for a, v in rows}).sort_index()


def _stage_fixture(pick_return=0.50, n_picks=20, n_unpriced=0):
    """per_anchor + price_source for the two CLEAN 36-month windows."""
    import depth_horizon_grid as dhg
    anchors = list(rc.DEFAULT_ANCHORS)
    lut = {(rc.BENCHMARK_SYMBOL, a): 100.0 * (1.05 ** i) for i, a in enumerate(anchors)}
    per_anchor = {}
    for wid, buy in dhg.BUY_ANCHORS:
        if wid not in ("buy2021", "buy2022"):
            continue
        ev = anchors[dhg.ANCHOR_IDX[buy] + 3]
        names = [f"{wid}_N{i}" for i in range(n_picks)]
        for i, nm in enumerate(names):
            if i >= n_picks - n_unpriced:
                continue                       # deliberately absent from the grid
            lut[(nm, buy)] = 100.0
            lut[(nm, ev)] = 100.0 * (1 + pick_return)
        per_anchor[wid] = {"top20_deduped": names, "ranking": names}
    return per_anchor, _FakePriceSource(lut, anchors)


def test_the_stage_RETURNS_the_downside_clause_beside_the_beat_rate():
    """Against HEAD the stage returned only per_window/pooled/bands -- the target's second
    clause had no home in the return value, so nothing downstream could even ask for it."""
    import pipeline_analysis as pa
    per_anchor, ps = _stage_fixture()
    out = pa.beat_rate_vs_urth(per_anchor, ps, log=lambda *a: None)
    assert "two_clause" in out, "the stage still reports only the UPSIDE clause"
    tcl = out["two_clause"]
    assert tcl["bar"] == pytest.approx(tc.bond_bar(36))
    assert tcl["n_windows"] == 2
    #  both policies, per anchor, with an explicit period verdict on each
    per = tcl["per_anchor"]
    assert {r["policy"] for r in per} == {"primary", "floor"}
    for r in per:
        assert r["upside"]["verdict"] in ("PASS", "FAIL", "INDETERMINATE")
        assert r["downside"]["verdict"] in ("PASS", "FAIL", "INDETERMINATE")
        assert r["period"] in ("PASS", "FAIL", "INDETERMINATE")
        assert "diagnostics" in r and "p25_rank" in r["diagnostics"]


def test_the_stage_PRINTS_both_clauses_and_the_three_diagnostics(capsys):
    """A reader of the run log must see both halves of the target and be told the diagnostics
    do not gate.  Printed, not just returned -- the run log is what the CEO actually reads."""
    import pipeline_analysis as pa
    per_anchor, ps = _stage_fixture()
    pa.beat_rate_vs_urth(per_anchor, ps, log=lambda *a: None)
    text = capsys.readouterr().out
    for needle in ("TWO-CLAUSE TARGET", "BOTH must pass", "DOWNSIDE", "UPSIDE",
                   "bond", "p25", "worst", "below0", "NEVER gating"):
        assert needle in text, f"the run log does not show {needle!r}"
    assert "9.27%" in text, "the charter's bond bar is not on the readout"


def test_the_stage_reports_INDETERMINATE_when_the_grid_cannot_price_the_picks(capsys):
    """The 2026-08-27 shape, in miniature: half the picks unpriceable.  The stage must say so
    rather than print a portfolio return over the priceable half as though it were the clause.
    This is the tie back to the price-grid fix -- the clause becomes decidable when the grid
    covers the picks."""
    import pipeline_analysis as pa
    per_anchor, ps = _stage_fixture(pick_return=0.04, n_unpriced=10)
    out = pa.beat_rate_vs_urth(per_anchor, ps, log=lambda *a: None)
    prim = [r for r in out["two_clause"]["per_anchor"] if r["policy"] == "primary"]
    assert prim and all(r["downside"]["verdict"] == "INDETERMINATE" for r in prim)
    assert all(r["downside"]["n_priced"] == 10 for r in prim)
    assert "INDETERMINATE" in capsys.readouterr().out


def test_the_bond_bar_has_ONE_definition_not_a_hardcoded_9_27_in_the_stage():
    """Two copies of a bar kept in lockstep by comment is drift this repo has already paid for
    (dead_merge.NA1_EXCHANGES).  The stage must consult `target_clauses`, not restate 0.0927."""
    import ast
    import inspect
    import pipeline_analysis as pa
    src = inspect.getsource(pa._two_clause_report)
    assert "bond_bar" in {n.attr for n in ast.walk(ast.parse(src))
                          if isinstance(n, ast.Attribute)}
    for literal in ("0.0927", "9.27", "0.03"):
        assert literal not in src, f"{literal} is restated in the stage instead of imported"


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
