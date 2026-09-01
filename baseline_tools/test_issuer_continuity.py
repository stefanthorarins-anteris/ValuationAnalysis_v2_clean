"""ISSUER CONTINUITY (register Q-42) -- follow the ISSUER, not the ticker.

WHAT EVERY TEST HERE IS AND IS NOT
----------------------------------
The house has a repeat defect -- fifteen recorded instances -- of a test that PINS the very
behaviour it is supposed to guard, and of a guard structurally blind to its own target.  So
each test below states, in its own docstring, WHAT IT CANNOT DETECT.  Read those lines
before trusting any green run.

THE SHARED CONSTRUCTION THAT MAKES THESE FAIL ON THE UNFIXED CODE.  `compute_returns` takes
`continuity={}` to switch the map OFF, which reproduces the pre-Q-42 terminal policy exactly.
Every behavioural test asserts the OLD number under `continuity={}` and the NEW number under
the default in the SAME test.  That is not decoration -- it is what makes the test fail on
HEAD-before-the-fix rather than merely describe the fix: on the unfixed code the two branches
are the same code path, so the pair cannot both hold.  A test that only asserted the new
number would pass trivially the moment someone deleted the map and reverted the primitive
only if the numbers happened to match; asserting the DELTA cannot.

THE ONE THING NONE OF THESE TESTS CAN DO, stated once at the top because it is the honest
headline: THEY CANNOT TELL YOU THE MAP IS RIGHT.  Every test takes `CONTINUITY_TABLE`'s
claims as given and checks only that the machinery does what the row says.  If VMD.TO's
successor were wrong -- a recycled ticker, the wrong share class, a company that merely
shares a name -- every test in this file still passes.  That claim rests entirely on the
2026-08-27 settlement work cited in each row's `source` field, and nothing offline can
re-verify it.
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import issuer_continuity as ic
import returns_core as rc
import target_clauses as tc


class FakePS:
    """Anchor grid A0<A1<A2<A3; prices given as {(ticker, anchor): price}.

    Same shape as `test_returns_core.FakePS` on purpose -- the primitive's contract is
    `.price` / `.last_before`, and a test that needed a real PriceSource would be testing
    the CSV loader instead of the policy.
    """
    def __init__(self, prices, anchors=("A0", "A1", "A2", "A3")):
        self.anchors = list(anchors)
        self._idx = {a: i for i, a in enumerate(self.anchors)}
        self._lut = dict(prices)

    def price(self, t, a):
        return self._lut.get((t, a))

    def last_before(self, t, a):
        j = self._idx.get(a)
        if j is None:
            return None
        for k in range(j - 1, -1, -1):
            p = self._lut.get((t, self.anchors[k]))
            if p is not None:
                return self.anchors[k], p
        return None


#  A two-row stand-in for the shipped table: one share-for-share continuation, one identified
#  discontinuity with no successor.  Used instead of the real table wherever the point is the
#  MECHANISM, so a future edit to the shipped rows cannot quietly turn these green or red.
FAKE_TABLE = [
    {"line": "OLD", "successor": "NEW", "event": "exchange move",
     "reason": "same issuer, share-for-share", "evidence": "fixture",
     "source": "fixture", "currency": "CAD -> USD"},
    {"line": "PFD", "successor": None, "event": "preferred call at par",
     "reason": "issuer alive, exit price unknown", "evidence": "fixture",
     "source": "fixture", "currency": ""},
]
FAKE_MAP = ic.load(FAKE_TABLE)


# --------------------------------------------------------------------------- #
#  1. The defect itself: a line discontinuity booked as a total loss           #
# --------------------------------------------------------------------------- #
def test_a_line_discontinuity_is_MEASURED_on_the_successor_and_was_a_LOSS_before():
    """THE CORE CASE.  OLD stops after A1; NEW prices A0 and A2.

    Unfixed (`continuity={}`): status 'terminal', FLOOR -100%, PRIMARY a stale A1 price.
    Fixed: status 'ok' with NEW's own A0->A2 return, and NOTHING at -100%.

    CANNOT DETECT: that OLD and NEW are the same issuer.  The fixture asserts it, so this
    test measures plumbing, not truth.  It also cannot detect a wrong ORDER of the three
    tests inside `compute_returns` -- see the dedicated test below, which exists because
    this one would pass even if the map overrode a real eval leg.
    """
    ps = FakePS({("OLD", "A0"): 10.0, ("OLD", "A1"): 8.0,
                 ("NEW", "A0"): 20.0, ("NEW", "A2"): 30.0})

    before = rc.compute_returns(["OLD"], "A0", "A2", ps, continuity={}).iloc[0]
    assert before["status"] == "terminal"
    assert math.isclose(before["total_return_floor"], -1.0)      # the defect, pinned
    assert math.isclose(before["total_return"], -0.2)            # stale A1 price, 8/10-1

    after = rc.compute_returns(["OLD"], "A0", "A2", ps, continuity=FAKE_MAP).iloc[0]
    assert after["status"] == "ok"
    assert not after["terminal_flag"]
    assert math.isclose(after["total_return"], 0.5)              # NEW: 30/20 - 1
    assert math.isclose(after["total_return_floor"], 0.5)        # no floor left to apply
    assert "OLD->NEW" in after["continuity"]


def test_the_successors_OWN_TWO_LEGS_are_used_and_the_legs_are_NEVER_SPLICED():
    """THE FX DEFECT, GUARDED DIRECTLY.  A splice would compute NEW_eval / OLD_buy.

    The fixture is chosen so the two readings cannot coincide: the splice gives 30/10-1 =
    +200%, the correct successor ratio gives 30/20-1 = +50%.  The assertion on the leg
    COLUMNS is the part that matters -- `buy_adjClose` must be the successor's leg, so that
    anything recomputing `eval/buy - 1` from the frame reproduces `total_return` instead of
    silently disagreeing with it.

    CANNOT DETECT: an actual currency error.  Nothing in this repo knows OLD is CAD and NEW
    is USD; the residual (the return is in the SUCCESSOR's currency) is a stated limitation,
    not a guarded one, and this test would pass identically if that residual were a disaster.
    """
    ps = FakePS({("OLD", "A0"): 10.0, ("NEW", "A0"): 20.0, ("NEW", "A2"): 30.0})
    r = rc.compute_returns(["OLD"], "A0", "A2", ps, continuity=FAKE_MAP).iloc[0]
    assert not math.isclose(r["total_return"], 2.0), "spliced NEW_eval / OLD_buy"
    assert math.isclose(r["total_return"], 0.5)
    assert math.isclose(r["buy_adjClose"], 20.0)
    assert math.isclose(r["eval_adjClose"], 30.0)
    assert math.isclose(r["eval_adjClose"] / r["buy_adjClose"] - 1.0, r["total_return"])


# --------------------------------------------------------------------------- #
#  2. The unmeasurable half: INDETERMINATE, never -100%, never dropped         #
# --------------------------------------------------------------------------- #
def test_an_identified_discontinuity_with_no_successor_is_INDETERMINATE_not_MINUS_100():
    """PFD's line ends; there is no successor line to follow.

    Unfixed: 'terminal' with FLOOR -100%.  Fixed: 'indeterminate' with NaN under BOTH
    policies -- absence outranks a fabricated verdict.  `terminal_flag` stays True so every
    caller that branches on it keeps applying its missing-value policy.

    CANNOT DETECT: whether the exit was actually benign.  The row asserts the issuer lived;
    if that were wrong, INDETERMINATE would be hiding a real -100% and this test would still
    be green.  It also cannot detect the pick being dropped from a report that never prints
    the bucket -- for that, see the partition test below.
    """
    ps = FakePS({("PFD", "A0"): 20.0, ("PFD", "A1"): 24.0})

    before = rc.compute_returns(["PFD"], "A0", "A2", ps, continuity={}).iloc[0]
    assert before["status"] == "terminal"
    assert math.isclose(before["total_return_floor"], -1.0)      # the defect, pinned

    after = rc.compute_returns(["PFD"], "A0", "A2", ps, continuity=FAKE_MAP).iloc[0]
    assert after["status"] == rc.STATUS_INDETERMINATE
    assert after["terminal_flag"] is True or after["terminal_flag"] == True  # noqa: E712
    assert np.isnan(after["total_return"])
    assert np.isnan(after["total_return_floor"])
    assert "INDETERMINATE" in after["continuity"]


def test_a_MAPPED_line_whose_successor_cannot_price_BOTH_anchors_is_INDETERMINATE():
    """The half-covered case: NEW exists at A2 but not at A0, so no ratio is computable.

    It must NOT fall back to the old terminal reading.  For a line the map says did not die,
    the stale PRIMARY and the -100% FLOOR are both refusals of the same kind, so the honest
    output is no number at all.

    CANNOT DETECT: a map row that is silently INERT because the successor symbol is
    misspelled -- that produces exactly this outcome and looks identical from here.
    `issuer_continuity.verify_against_grid` is the report that would show it, and it is a
    report, not an assertion.
    """
    ps = FakePS({("OLD", "A0"): 10.0, ("OLD", "A1"): 8.0, ("NEW", "A2"): 30.0})
    r = rc.compute_returns(["OLD"], "A0", "A2", ps, continuity=FAKE_MAP).iloc[0]
    assert r["status"] == rc.STATUS_INDETERMINATE
    assert np.isnan(r["total_return_floor"])
    assert "unpriced at A0" in r["continuity"]


# --------------------------------------------------------------------------- #
#  3. Order of operations: the map must never override a real observation      #
# --------------------------------------------------------------------------- #
def test_a_line_that_prices_BOTH_anchors_IGNORES_the_map_entirely():
    """OLD prices A0 AND A2, so it is measured on its own legs and the map is never asked.

    This is the guard against the fix's own worst failure mode: a map row quietly replacing a
    real measurement with a successor's.  It also encodes the intended degradation -- when a
    future price refetch fills the missing anchor, the map stops firing for that window with
    no code change and no argument.

    CANNOT DETECT: the reverse mistake at the BUY leg.  A pick with no buy price is 'no_buy'
    before the map is consulted, so a discontinuity that also lost its buy leg is invisible
    to the whole mechanism -- by design, but untested here because there is nothing to test.
    """
    ps = FakePS({("OLD", "A0"): 10.0, ("OLD", "A2"): 11.0,
                 ("NEW", "A0"): 20.0, ("NEW", "A2"): 60.0})
    r = rc.compute_returns(["OLD"], "A0", "A2", ps, continuity=FAKE_MAP).iloc[0]
    assert r["status"] == "ok"
    assert r["continuity"] == ""
    assert math.isclose(r["total_return"], 0.1)          # OLD's own legs, not NEW's +200%


def test_an_UNMAPPED_line_keeps_the_terminal_policy_BIT_FOR_BIT():
    """The blast radius is exactly the mapped lines.  An unmapped discontinuity is unchanged.

    Asserted as an EQUALITY between the map-on and map-off frames rather than as a repeat of
    the expected numbers, so it cannot drift out of agreement with the terminal policy if that
    policy is ever legitimately changed.

    CANNOT DETECT: that the population of unmapped discontinuities is large -- which it is.
    This test is the reason the fix is safe; it is not evidence the fix is sufficient.
    """
    ps = FakePS({("ZZZ", "A0"): 10.0, ("ZZZ", "A1"): 8.0})
    off = rc.compute_returns(["ZZZ"], "A0", "A2", ps, continuity={})
    on = rc.compute_returns(["ZZZ"], "A0", "A2", ps, continuity=FAKE_MAP)
    pd.testing.assert_frame_equal(off, on)
    assert off.iloc[0]["status"] == "terminal"
    assert math.isclose(off.iloc[0]["total_return_floor"], -1.0)


# --------------------------------------------------------------------------- #
#  4. Coverage accounting: the new bucket must not break the partition         #
# --------------------------------------------------------------------------- #
def test_the_coverage_buckets_still_PARTITION_the_shipped_picks():
    """measured + stale + buy_only + indeterminate + no_buy == shipped.

    This is the test that catches "not silently dropped".  The failure it exists for is the
    quiet one: a fourth status appearing in the data and in no bucket, so the printed columns
    stop summing to the shipped count while every individual number still looks plausible.

    CANNOT DETECT: a bucket that is counted here and never PRINTED.  The sum closing in
    `coverage_counts` says nothing about `pipeline_analysis`'s table having a column for it.
    """
    ps = FakePS({
        ("OLD", "A0"): 10.0, ("NEW", "A0"): 20.0, ("NEW", "A2"): 30.0,   # -> continued/ok
        ("PFD", "A0"): 20.0, ("PFD", "A1"): 24.0,                        # -> indeterminate
        ("STALE", "A0"): 10.0, ("STALE", "A1"): 9.0,                     # -> terminal stale
        ("BUYONLY", "A0"): 10.0,                                         # -> buy_only
        ("PLAIN", "A0"): 10.0, ("PLAIN", "A2"): 12.0,                    # -> ok
        ("NOBUY", "A2"): 5.0,                                            # -> no_buy
    })
    names = ["OLD", "PFD", "STALE", "BUYONLY", "PLAIN", "NOBUY"]
    rdf = rc.compute_returns(names, "A0", "A2", ps, continuity=FAKE_MAP)
    c = tc.coverage_counts(rdf, depth_n=len(names))

    assert c["n_measured"] == 2                    # OLD (via NEW) and PLAIN
    assert c["n_continued"] == 1                   # OLD only; NOT part of the sum
    assert c["n_indeterminate"] == 1
    assert c["n_terminal_stale"] == 1
    assert c["n_buy_only"] == 1
    assert c["n_no_buy"] == 1
    assert (c["n_measured"] + c["n_terminal_stale"] + c["n_buy_only"]
            + c["n_indeterminate"] + c["n_no_buy"]) == c["n_selected"]


def test_an_INDETERMINATE_pick_is_NOT_counted_as_MEASURED():
    """`measured` excludes it BY NAME, not via the old `status != 'terminal'` test.

    The specific defect guarded: a new status slipping through a not-equal test into the
    measured set, carrying NaN returns that `_returns` then drops -- so the pick inflates
    `n_measured` and coverage while contributing to no figure.  Coverage would read high
    exactly where the data is worst.

    CANNOT DETECT: the same class of bug for a SIXTH status added later.  The exclusion list
    is `rc.UNMEASURED_STATUSES`, and nothing forces a new status to be classified; this test
    would pass while a new one leaked.  The guard below is the partial answer to that.
    """
    ps = FakePS({("PFD", "A0"): 20.0, ("PFD", "A1"): 24.0,
                 ("PLAIN", "A0"): 10.0, ("PLAIN", "A2"): 12.0})
    rdf = rc.compute_returns(["PFD", "PLAIN"], "A0", "A2", ps, continuity=FAKE_MAP)
    m = tc.measured(rdf)
    assert list(m["ticker"]) == ["PLAIN"]
    assert len(tc._returns(rdf, floor=False)) == 1


def test_EVERY_status_the_primitive_can_emit_is_CLASSIFIED_measured_or_not():
    """GUARD ON THE GUARD.  Enumerate the statuses `compute_returns` writes and require each
    to be either STATUS_OK or in UNMEASURED_STATUSES or STATUS_NO_BUY.

    Reads the statuses off the SOURCE (the literals the function can write) rather than off a
    sample frame, because a sample can only contain the statuses the fixture happened to
    provoke -- which is precisely how a new status escapes classification.

    CANNOT DETECT: a status written through a variable rather than a module constant, or
    assembled at runtime.  The scan is a text scan over the constants; it is a tripwire, not
    a type system.
    """
    import inspect
    src = inspect.getsource(rc.compute_returns)
    used = {n for n in ("STATUS_OK", "STATUS_TERMINAL", "STATUS_INDETERMINATE",
                        "STATUS_NO_BUY") if n in src}
    assert used, "compute_returns no longer writes status via the module constants; " \
                 "this guard has gone blind and must be rewritten"
    known = {rc.STATUS_OK, rc.STATUS_NO_BUY} | set(rc.UNMEASURED_STATUSES)
    for name in used:
        assert getattr(rc, name) in known, (
            "%s is written by compute_returns but is neither 'ok', 'no_buy', nor in "
            "UNMEASURED_STATUSES -- every clause that grades coverage would misclassify it"
            % name)


def test_the_AVERAGE_denominator_is_REPORTED_and_is_not_n_included():
    """An indeterminate pick has a buy leg and no return, so `n_included` overstates the mean's
    denominator.  `averaged_returns` is the set actually averaged and must be the smaller one.

    The failure this guards is a silent one: `Series.mean()` skips NaN, so before this the
    average would simply have used a different denominator than the count printed beside it,
    with nothing anywhere disagreeing.

    CANNOT DETECT: a CALLER that prints `n_included` next to the average anyway.  This pins
    the primitive's honesty, not every report built on it.
    """
    ps = FakePS({("PFD", "A0"): 20.0, ("PFD", "A1"): 24.0,
                 ("PLAIN", "A0"): 10.0, ("PLAIN", "A2"): 12.0})
    rdf = rc.compute_returns(["PFD", "PLAIN"], "A0", "A2", ps, continuity=FAKE_MAP)
    assert rc.counts(rdf)["n_included"] == 2
    assert len(rc.averaged_returns(rdf)) == 1
    assert math.isclose(rc.average_return(rdf), 0.2)      # PLAIN alone, not 0.2/2


def test_beat_rate_treats_an_INDETERMINATE_pick_as_MISSING_under_every_policy():
    """It must take the missing branch, not a NaN comparison that silently reads False.

    `missing='drop'` is the discriminating case: under a NaN comparison the pick would be
    counted and scored a miss, so `n` would come back 2 instead of 1.

    CANNOT DETECT: whether 'fail' is the right POLICY for such a pick.  Scoring an unknown as
    'did not beat' is a stance the caller chooses; this only checks it is applied via the
    stated policy rather than by accident.
    """
    ps = FakePS({("PFD", "A0"): 20.0, ("PFD", "A1"): 24.0,
                 ("WIN", "A0"): 10.0, ("WIN", "A2"): 20.0})
    rdf = rc.compute_returns(["PFD", "WIN"], "A0", "A2", ps, continuity=FAKE_MAP)
    rate, n = rc.beat_rate(rdf, benchmark_ret=0.05, threshold=0.10, missing="drop")
    assert n == 1 and math.isclose(rate, 1.0)
    rate, n = rc.beat_rate(rdf, benchmark_ret=0.05, threshold=0.10, missing="fail")
    assert n == 2 and math.isclose(rate, 0.5)


def test_the_POOLED_grid_averages_the_BENCHMARK_over_the_SAME_rows_as_the_RETURN():
    """A defect this change INTRODUCED and then removed, kept as a guard.

    `bench` is constant within a window, so a pooled benchmark mean is really a
    window-weighted average weighted by how many picks each window contributes.  Averaging it
    over `n_included` while the return is averaged over `n_averaged` gives the two legs of
    `excess` different window weights the moment any pick is unmeasured.  In the fixture below
    that is a 6.7pp error in `excess_primary` -- small, silent, and entirely an artefact of
    the NaN rows this change added.

    CANNOT DETECT: the same misalignment anywhere else.  This pins `depth_horizon_grid._pool`
    only; `beat_rate`, `skill_baseline` and the two-clause pooling each average their own way
    and none of them is checked here.
    """
    import depth_horizon_grid as dhg
    rows = []
    for wid, st, rp, rf, b, pos in [
            ("W1", "ok", 0.20, 0.20, 0.10, 1), ("W1", "ok", 0.30, 0.30, 0.10, 2),
            ("W2", "ok", 0.40, 0.40, 0.50, 1),
            ("W2", rc.STATUS_INDETERMINATE, float("nan"), float("nan"), 0.50, 2)]:
        rows.append({"wid": wid, "clean": True, "horizon_m": 36, "rank_pos": pos,
                     "status": st, "ret_primary": rp, "ret_floor": rf, "bench": b,
                     "continuity": ""})
    cell = [r for r in dhg._pool(rows, clean_only=False) if r["depth_N"] == 2][0]
    assert cell["n_included"] == 4 and cell["n_averaged"] == 3
    assert math.isclose(cell["avg_ret_primary"], 0.30)
    assert math.isclose(cell["bench_ret"], (0.10 + 0.10 + 0.50) / 3)   # NOT /4
    assert math.isclose(cell["excess_primary"], 0.30 - (0.70 / 3))


# --------------------------------------------------------------------------- #
#  5. The table itself: shape, and consistency with the grid                   #
# --------------------------------------------------------------------------- #
def test_the_shipped_table_VALIDATES_and_every_row_carries_its_justification():
    """Shape only: fields present, no duplicate line, no self-reference, no chain, and
    non-empty reason/evidence/source on every row.

    CANNOT DETECT: a reason that is text-shaped nonsense, or evidence citing a file that says
    something else.  "Non-empty" is the whole of what code can check here; the substance is a
    review question and that is why a new row is a code change.
    """
    ic.validate()
    for row in ic.CONTINUITY_TABLE:
        for field in ("reason", "evidence", "source"):
            assert len(row[field].strip()) > 20, (row["line"], field)


@pytest.mark.parametrize("bad,fragment", [
    ([{"line": "A", "successor": "A", "event": "e", "reason": "r" * 30,
       "evidence": "e" * 30, "source": "s", "currency": ""}], "points at itself"),
    ([{"line": "A", "successor": "B", "event": "e", "reason": "r" * 30,
       "evidence": "e" * 30, "source": "s", "currency": ""},
      {"line": "B", "successor": "C", "event": "e", "reason": "r" * 30,
       "evidence": "e" * 30, "source": "s", "currency": ""}], "CHAIN refused"),
    ([{"line": "A", "successor": "B", "event": "e", "reason": "r" * 30,
       "evidence": "e" * 30, "source": "s", "currency": ""},
      {"line": "A", "successor": "C", "event": "e", "reason": "r" * 30,
       "evidence": "e" * 30, "source": "s", "currency": ""}], "TWO rows"),
    ([{"line": "A", "successor": "B", "event": "e", "reason": "  ",
       "evidence": "e" * 30, "source": "s", "currency": ""}], "empty reason"),
])
def test_validate_REFUSES_the_shapes_it_says_it_refuses(bad, fragment):
    """Each malformed table must raise, with the stated reason in the message.

    A chain is REFUSED rather than resolved: two rows written independently do not add up to
    a verified two-hop corporate history, and resolving them transitively would invent one.

    CANNOT DETECT: a chain expressed as a single row with the wrong intermediate successor --
    that is well-formed and wrong, and no shape check reaches it.
    """
    with pytest.raises(ValueError, match=fragment):
        ic.validate(bad)


# --------------------------------------------------------------------------- #
#  6. Against the REAL price grid -- offline, read-only, no network            #
# --------------------------------------------------------------------------- #
_PRICES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "price_data", "real_prices.csv")
_PRICES_2025 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "price_data", "real_prices_2025.csv")


@pytest.fixture(scope="module")
def real_ps():
    if not os.path.exists(_PRICES):
        pytest.skip("real_prices.csv absent (never committed) -- grid tests skipped")
    supp = _PRICES_2025 if os.path.exists(_PRICES_2025) else None
    return rc.PriceSource(_PRICES, supp_csv=supp)


def test_VMD_TO_on_the_real_grid_was_a_MINUS_100_FLOOR_and_is_now_PLUS_53_6_PERCENT(real_ps):
    """THE NAMED CASE, on the shipped grid: buy 2021-12-31, eval 2024-12-31.

    Before: 'terminal', FLOOR -100%, PRIMARY +55.6% off a 2022 CAD price two years stale --
    and UNMEASURED, so it contributed to no clause.
    After: 'ok', +53.64% from VMD's own 5.22 -> 8.02, and MEASURED.

    Note what this shows about the framing "currently booked as a loss": under the FLOOR
    policy, `lower_bound` and `beat_rate(missing='fail')` it was; under PRIMARY it read
    +55.6%.  The real gain is not that a -100% became a +53.6%, it is that an UNMEASURED
    pick became a measured one with the right number.

    CANNOT DETECT: a change to the price grid.  The expected values are literals from
    real_prices.csv as of 2026-08-31; a refetch that revises VMD's 2021 or 2024 close makes
    this test fail for a reason that is not a regression.  That is the intended trade -- a
    pinned number that fails loudly beats a recomputed one that can never disagree.
    """
    off = rc.compute_returns(["VMD.TO"], "2021-12-31", "2024-12-31", real_ps,
                             continuity={}).iloc[0]
    assert off["status"] == "terminal"
    assert math.isclose(off["total_return_floor"], -1.0)
    assert math.isclose(off["total_return"], 10.27 / 6.60 - 1.0, rel_tol=1e-9)

    on = rc.compute_returns(["VMD.TO"], "2021-12-31", "2024-12-31", real_ps).iloc[0]
    assert on["status"] == "ok"
    assert math.isclose(on["total_return"], 8.02 / 5.22 - 1.0, rel_tol=1e-9)
    assert 0.536 < on["total_return"] < 0.537
    assert "VMD.TO->VMD" in on["continuity"]
    assert len(tc.measured(pd.DataFrame([on]))) == 1


def test_CMRE_PE_on_the_real_grid_stops_being_a_MINUS_100_and_becomes_INDETERMINATE(real_ps):
    """buy 2022-12-30, eval 2025-12-31.  The common (CMRE) lives; the preferred's exit does not
    exist on disk, and the common's return is a DIFFERENT security's return.

    Before: FLOOR -100%.  After: NaN under both policies, status 'indeterminate'.

    CANNOT DETECT: that the true exit was near $25 par.  Refusing to guess is the point, and
    a guess that happened to be right would pass this test too if someone encoded it.
    """
    off = rc.compute_returns(["CMRE-PE"], "2022-12-30", "2025-12-31", real_ps,
                             continuity={}).iloc[0]
    assert off["status"] == "terminal"
    assert math.isclose(off["total_return_floor"], -1.0)

    on = rc.compute_returns(["CMRE-PE"], "2022-12-30", "2025-12-31", real_ps).iloc[0]
    assert on["status"] == rc.STATUS_INDETERMINATE
    assert np.isnan(on["total_return"]) and np.isnan(on["total_return_floor"])
    assert "CMRE-PE" in on["continuity"]


def test_CMRE_PE_is_NOT_measured_at_the_COMMONS_return(real_ps):
    """THE COUNTEREXAMPLE THAT CHOSE THE DESIGN.  CMRE and CMRE-PE share an issuer and would
    share `universe_dedup`'s fundamental fingerprint, so a fingerprint-derived successor map
    would have measured the preferred at the common's return.  It must not.

    CANNOT DETECT: someone later adding a `CMRE-PE -> CMRE` row by hand.  The table is
    hand-written precisely so that would be a review event; nothing here prevents it.
    """
    common = rc.compute_returns(["CMRE"], "2022-12-30", "2025-12-31", real_ps).iloc[0]
    pref = rc.compute_returns(["CMRE-PE"], "2022-12-30", "2025-12-31", real_ps).iloc[0]
    assert common["status"] == "ok"
    assert pref["status"] == rc.STATUS_INDETERMINATE
    assert not (pref["total_return"] == common["total_return"])


def test_the_grid_is_CONSISTENT_with_every_row_in_the_shipped_table(real_ps):
    """`verify_against_grid` as an assertion, for the shipped rows only: the line must stop
    pricing before the end of the grid, and a successor row's successor must price at least
    one anchor after the line's last.

    A row failing this is either stale (the line came back) or inert (the successor is
    misspelled or absent) -- both silent otherwise.

    CANNOT DETECT: the successor being the WRONG company.  Consistency with the grid is a
    necessary condition and nowhere near a sufficient one.
    """
    rows = {r["line"]: r for r in ic.verify_against_grid(real_ps)}
    assert set(rows) == {r["line"] for r in ic.CONTINUITY_TABLE}
    for line, r in rows.items():
        assert r["line_priced_anchors"] > 0, "%s prices nothing; the row cannot apply" % line
        assert r["line_last_priced"] < real_ps.anchors[-1], (
            "%s still prices at the last anchor -- the row is stale" % line)
        if r["successor"]:
            assert r["covers_the_gap"], (
                "%s -> %s: the successor does not price past the line's last anchor, so the "
                "row can only ever produce INDETERMINATE" % (line, r["successor"]))
