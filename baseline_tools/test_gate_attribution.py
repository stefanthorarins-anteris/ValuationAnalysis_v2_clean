"""GATE ATTRIBUTION: the arithmetic that decides whether the readout can mislead.

WHAT THESE PIN.  The readout exists to answer "did the Stage-1 solvency gate help", and the
three ways it could lie are all counting errors, not modelling ones:

  1. crediting the gate for removing a name NOBODY CAN PRICE (the coverage trap, which bites
     harder here than anywhere else because ejected names are enriched in companies whose
     prices stop -- that is most of what a solvency gate is for);
  2. blaming the gate for a pick that left the top-20 because the Stage-2 NORMALISATION POOL
     moved, which the veto causes but no flag is answerable for;
  3. adding up per-flag ejection counts that are not a partition (`EJECT_MIN_FLAGS = 1`, so
     most ejected names fail several flags and the per-flag counts sum past the total).

OFFLINE.  Fake price source, hand-built rankings.  No pickle, no network, no pipeline run.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import returns_core as rc
import gate_attribution as ga

BUY, EVAL = "2022-12-30", "2025-12-31"


class _PS:
    """Minimal PriceSource surface. A name absent from `lut` at EVAL is UNPRICEABLE."""

    def __init__(self, lut):
        self._lut = lut
        self.anchors = list(rc.DEFAULT_ANCHORS)

    def price(self, t, a):
        return self._lut.get((t, a))

    def last_before(self, t, a):
        j = self.anchors.index(a)
        for k in range(j - 1, -1, -1):
            p = self._lut.get((t, self.anchors[k]))
            if p is not None:
                return self.anchors[k], p
        return None

    def benchmark_series(self, symbol=rc.BENCHMARK_SYMBOL):
        import pandas as pd
        rows = [(a, self._lut[(symbol, a)]) for a in self.anchors
                if (symbol, a) in self._lut]
        return pd.Series({pd.Timestamp(a): v for a, v in rows}).sort_index()


def _source(returns, bench=0.0):
    """{ticker: total_return or None-for-unpriceable} -> a price source."""
    lut = {(rc.BENCHMARK_SYMBOL, BUY): 100.0,
           (rc.BENCHMARK_SYMBOL, EVAL): 100.0 * (1 + bench)}
    for t, r in returns.items():
        lut[(t, BUY)] = 100.0
        if r is not None:
            lut[(t, EVAL)] = 100.0 * (1 + r)
    return _PS(lut)


def _anchor(top20, ranking=None, ejected=(), ejected_flags=None, n_in=1000, basis=None):
    ej = list(ejected)
    return {
        "buy": BUY, "top20_deduped": list(top20),
        "ranking": list(ranking if ranking is not None else top20),
        "basis": basis or ("VETOED (stage-1 solvency gate applied, %d ejected)" % len(ej)
                           if ej else "un-vetoed"),
        "stage1_veto": {"enabled": True, "applies": True, "n_in": n_in,
                        "n_out": n_in - len(ej), "n_ejected": len(ej),
                        "ejected": ej, "ejected_flags": dict(ejected_flags or {}),
                        "by_flag": {}},
    }


# --------------------------------------------------------------------------- #
#  1. THE COVERAGE TRAP -- the rule the whole readout rests on                 #
# --------------------------------------------------------------------------- #
def test_an_unpriceable_ejected_name_is_UNKNOWN_not_a_win_for_the_gate():
    """THE TRTN SHAPE, and the single most load-bearing rule here.  A gate that removes a name
    the price grid cannot value has shown nothing: the name may have been acquired at a premium
    or may have gone to zero, and the grid says neither.  It must not appear as a saved loss,
    and it must not appear in any mean."""
    ps = _source({"KEEP": 0.20, "GONE": None, "REPL": 0.10})
    v = _anchor(["KEEP", "REPL"], ejected=["GONE"],
                ejected_flags={"GONE": ["netDebtToEBITDA"]})
    u = _anchor(["KEEP", "GONE"])
    a = ga.attribute_anchor("buy2022", BUY, EVAL, v, u, ps, depth_n=2)
    d = a["cohorts"]["dropped_by_veto"]
    assert a["dropped_by_veto"] == ["GONE"]
    assert d["n"] == 1 and d["n_measured"] == 0 and d["n_unknown"] == 1
    assert d["n_below_zero"] == 0, "an unpriceable ejection is not a loss avoided"
    assert d["n_beat"] == 0, "and it is not a winner lost either"
    assert d["mean_return"] != d["mean_return"]      # NaN: no mean over nothing
    #  the swap cannot be scored, and the report must say so rather than print a number
    assert a["swap"]["effect_measured"] != a["swap"]["effect_measured"]
    text = ga.format_report({"per_anchor": [a], "horizon_m": 36, "depth_n": 2,
                             "threshold": 0.10, "spotlight": []})
    assert "UNKNOWN" in text and "GONE" in text


def test_the_flip_says_what_the_unpriceable_pick_would_have_had_to_RETURN():
    """An unscoreable swap is not the end of the answer.  If the replacement made +10% and the
    ejected name cannot be priced, "it would have had to make +10% for the gate to break even"
    is a sentence the CEO can judge -- and it is the same idiom as the clause's flip_return."""
    ps = _source({"KEEP": 0.0, "GONE": None, "REPL": 0.10})
    v = _anchor(["KEEP", "REPL"], ejected=["GONE"], ejected_flags={"GONE": ["uCurrentRatio"]})
    u = _anchor(["KEEP", "GONE"])
    a = ga.attribute_anchor("buy2022", BUY, EVAL, v, u, ps, depth_n=2)
    assert a["swap"]["flip_return"] == pytest.approx(0.10)


def test_a_priced_ejected_WINNER_is_reported_as_a_cost_of_the_gate():
    """The direction that is uncomfortable and must not be softened: if the gate ejected a name
    that went on to beat the benchmark, that is the finding."""
    ps = _source({"KEEP": 0.0, "WINNER": 1.00, "REPL": 0.05}, bench=0.0)
    v = _anchor(["KEEP", "REPL"], ejected=["WINNER"],
                ejected_flags={"WINNER": ["netDebtToEBITDA"]})
    u = _anchor(["KEEP", "WINNER"])
    a = ga.attribute_anchor("buy2022", BUY, EVAL, v, u, ps, depth_n=2)
    d = a["cohorts"]["dropped_by_veto"]
    assert d["n_beat"] == 1 and d["mean_return"] == pytest.approx(1.00)
    #  (1/2) * (0.05 - 1.00) -- the gate cost the portfolio, and the sign says so
    assert a["swap"]["effect_measured"] == pytest.approx(0.5 * (0.05 - 1.00))
    assert a["swap"]["effect_measured"] < 0


# --------------------------------------------------------------------------- #
#  2. A DROPPED PICK IS NOT AUTOMATICALLY A GATE KILL                          #
# --------------------------------------------------------------------------- #
def test_a_pick_that_left_for_a_POOL_SHIFT_is_not_charged_to_the_gate():
    """The veto ejects BEFORE the Stage-1 head() cut, so the Stage-2 normalisation population
    changes and a name never ejected can still fall out of the top-N.  Charging that to the
    gate -- or to one of its flags -- would overstate the gate in whichever direction the name
    happened to move."""
    ps = _source({"KEEP": 0.0, "EJECTED": -0.50, "SHIFTED": 0.90, "REPL": 0.10,
                  "REPL2": 0.10})
    v = _anchor(["KEEP", "REPL", "REPL2"], ejected=["EJECTED"],
                ejected_flags={"EJECTED": ["returnOnAssets"]})
    u = _anchor(["KEEP", "EJECTED", "SHIFTED"])
    a = ga.attribute_anchor("buy2022", BUY, EVAL, v, u, ps, depth_n=3)
    assert a["dropped_by_veto"] == ["EJECTED"]
    assert a["dropped_other"] == ["SHIFTED"]
    assert a["cohorts"]["dropped_by_veto"]["mean_return"] == pytest.approx(-0.50)
    assert a["cohorts"]["dropped_other"]["mean_return"] == pytest.approx(0.90)
    #  and no flag is answerable for the shifted name
    assert "SHIFTED" not in sum((e["any"] for e in a["by_flag"].values()), [])


# --------------------------------------------------------------------------- #
#  3. PER-FLAG COUNTS ARE NOT A PARTITION, AND THE READOUT SAYS WHICH IS WHICH #
# --------------------------------------------------------------------------- #
def test_n_any_and_n_solely_answer_different_questions():
    """`EJECT_MIN_FLAGS = 1`, so a name goes the moment ANY flag fails and most failed names
    fail several.  `n_any` therefore sums past the ejection total and cannot answer "what if I
    drop this flag"; `n_solely` can, because removing that flag un-ejects exactly those names.
    Reporting only the first is how a flag gets credited for kills it did not decide."""
    ps = _source({"KEEP": 0.0, "A": -0.5, "B": -0.5, "R1": 0.1, "R2": 0.1})
    v = _anchor(["KEEP", "R1", "R2"], ejected=["A", "B"],
                ejected_flags={"A": ["netDebtToEBITDA"],
                               "B": ["netDebtToEBITDA", "uCurrentRatio"]})
    u = _anchor(["KEEP", "A", "B"])
    a = ga.attribute_anchor("buy2022", BUY, EVAL, v, u, ps, depth_n=3)
    nd = a["by_flag"]["netDebtToEBITDA"]
    uc = a["by_flag"]["uCurrentRatio"]
    assert nd["n_any"] == 2 and nd["n_solely"] == 1 and nd["solely"] == ["A"]
    assert uc["n_any"] == 1 and uc["n_solely"] == 0
    #  the two n_any's sum to 3 over 2 actual kills -- not a partition, by construction
    assert sum(e["n_any"] for e in a["by_flag"].values()) > len(a["dropped_by_veto"])


def test_pool_wide_solely_counts_are_the_counterfactual_not_the_headline():
    """`stage1_veto`'s own `by_flag` is the n_any form. The pool-wide `solely` count is what a
    reader needs to answer "drop this flag and how many names come back"."""
    flags = {"A": ["returnOnAssets"], "B": ["returnOnAssets", "uCurrentRatio"],
             "C": ["uCurrentRatio"]}
    assert ga._solely_counts(flags) == {"returnOnAssets": 1, "uCurrentRatio": 1}


# --------------------------------------------------------------------------- #
#  4. THE FUNNEL -- an ejection count is a SIZE, not an EFFECT                 #
# --------------------------------------------------------------------------- #
def test_the_funnel_narrows_from_the_pool_to_the_shipped_list():
    """THE NUMBER THAT MAKES THE HEADLINE LEGIBLE.  "1,014 of 2,148 ejected" invites the reading
    that half the shortlist changed; measured on the 2026-08-28 panel, 6 of those ejections were
    in the 100 names that reached Stage-2 and ONE was in the shipped top-20.  Without the
    narrowing, the gate's size reads as its effect."""
    ps = _source({"K1": 0.0, "K2": 0.0, "EJ_TOP": -0.2, "EJ_DEEP": -0.2, "REPL": 0.1})
    v = _anchor(["K1", "K2", "REPL"], ranking=["K1", "K2", "REPL", "FILLER"],
                ejected=["EJ_TOP", "EJ_DEEP", "EJ_FAR"], n_in=900,
                ejected_flags={"EJ_TOP": ["netDebtToEBITDA"],
                               "EJ_DEEP": ["netDebtToEBITDA"],
                               "EJ_FAR": ["returnOnAssets"]})
    u = _anchor(["K1", "K2", "EJ_TOP"], ranking=["K1", "K2", "EJ_TOP", "EJ_DEEP"])
    a = ga.attribute_anchor("buy2022", BUY, EVAL, v, u, ps, depth_n=3)
    f = a["funnel"]
    assert f["n_scored"] == 900 and f["n_ejected"] == 3
    assert f["n_unvetoed_pool_ejected"] == 2      # EJ_TOP + EJ_DEEP reached Stage-2
    assert f["n_unvetoed_top_ejected"] == 1       # only EJ_TOP was going to be bought
    assert a["flag_funnel"]["netDebtToEBITDA"]["solely_in_top"] == 1
    assert a["flag_funnel"]["returnOnAssets"]["solely_in_top"] == 0
    assert a["flag_funnel"]["returnOnAssets"]["solely_pool_wide"] == 1


# --------------------------------------------------------------------------- #
#  5. THE HONEST NULL RESULT                                                   #
# --------------------------------------------------------------------------- #
def test_a_gate_that_touched_NOTHING_at_the_top_says_so_rather_than_nothing():
    """The likeliest real outcome, and it must not read as an empty section.  A veto can eject
    half the pool and change none of the shipped picks; "no flag has an effect to measure here"
    is a finding about the gate, and a blank table is not."""
    ps = _source({"K1": 0.1, "K2": 0.1})
    v = _anchor(["K1", "K2"], ejected=["DEEP"], ejected_flags={"DEEP": ["returnOnAssets"]})
    u = _anchor(["K1", "K2"])
    a = ga.attribute_anchor("buy2022", BUY, EVAL, v, u, ps, depth_n=2)
    assert a["dropped_all"] == [] and a["added"] == [] and a["by_flag"] == {}
    text = ga.format_report({"per_anchor": [a], "horizon_m": 36, "depth_n": 2,
                             "threshold": 0.10, "spotlight": []})
    assert "NO FLAG has" in text


def test_no_comparable_anchor_reads_as_UNMEASURED_not_as_no_effect():
    """"We could not measure it" and "the gate did nothing" are different statements, and the
    second is the one a silent empty report would imply."""
    text = ga.format_report({"per_anchor": [], "horizon_m": 36, "depth_n": 20,
                             "threshold": 0.10, "spotlight": []})
    assert "UNMEASURED" in text and "not 'the gate did nothing'" in text


# --------------------------------------------------------------------------- #
#  6. THE BASIS TRAVELS WITH THE NUMBERS                                       #
# --------------------------------------------------------------------------- #
def test_both_bases_are_printed_so_neither_can_be_read_as_the_other():
    ps = _source({"K": 0.1, "E": 0.1, "R": 0.1})
    v = _anchor(["K", "R"], ejected=["E"], ejected_flags={"E": ["uInterestCoverage"]})
    u = _anchor(["K", "E"])
    a = ga.attribute_anchor("buy2022", BUY, EVAL, v, u, ps, depth_n=2)
    text = ga.format_report({"per_anchor": [a], "horizon_m": 36, "depth_n": 2,
                             "threshold": 0.10, "spotlight": []})
    assert "BASIS vetoed" in text and "BASIS un-vetoed" in text
    assert a["basis_vetoed"].startswith("VETOED") and a["basis_unvetoed"] == "un-vetoed"


# --------------------------------------------------------------------------- #
#  7. THE SPOTLIGHT -- a named case survives the aggregate                     #
# --------------------------------------------------------------------------- #
def test_the_spotlight_reports_a_named_ticker_even_when_it_is_unpriceable():
    """The CEO's own example.  At buy2022 TRTN sits inside the un-vetoed top-20, is ejected on
    netDebtToEBITDA alone, and has no eval price -- so the aggregate can say nothing about it
    and the name-level row is the whole answer."""
    ps = _source({"K": 0.1, "TRTN": None, "R": 0.1})
    v = _anchor(["K", "R"], ranking=["K", "R", "X"], ejected=["TRTN"],
                ejected_flags={"TRTN": ["netDebtToEBITDA"]})
    u = _anchor(["K", "TRTN"], ranking=["K", "TRTN", "X"])
    rows = ga.spotlight_rows([("buy2022", BUY, EVAL)], {"buy2022": v}, {"buy2022": u}, ps,
                             tickers=("TRTN",), depth_n=2)
    r = rows[0]
    assert r["in_unvetoed_top20"] and not r["in_vetoed_top20"]
    assert r["ejected"] and r["ejected_flags"] == ["netDebtToEBITDA"]
    assert r["unvetoed_rank"] == 2
    assert not r["measured"] and r["total_return"] != r["total_return"]


def test_a_spotlight_ticker_that_is_nowhere_still_gets_a_row():
    """Absence is an answer. A silent omission reads as "not checked"."""
    ps = _source({"K": 0.1})
    v = _anchor(["K"])
    rows = ga.spotlight_rows([("buy2022", BUY, EVAL)], {"buy2022": v}, {"buy2022": v}, ps,
                             tickers=("NOPE",), depth_n=1)
    assert len(rows) == 1 and rows[0]["unvetoed_rank"] is None
    assert not rows[0]["ejected"] and not rows[0]["in_unvetoed_top20"]


# --------------------------------------------------------------------------- #
#  8. THE COHORT PRIMITIVE USES THE CLAUSE'S OWN MEASURED-SET DEFINITION       #
# --------------------------------------------------------------------------- #
def test_a_STALE_terminal_price_is_not_a_gate_outcome_either():
    """One definition of "measured", shared with `target_clauses`.  A name priced at buy and at
    an EARLIER anchor gets a substituted terminal from `compute_returns`; counting that as a
    gate outcome would re-import the exact defect the target clause was just fixed for."""
    lut = {(rc.BENCHMARK_SYMBOL, BUY): 100.0, (rc.BENCHMARK_SYMBOL, EVAL): 100.0,
           ("STALE", "2021-12-31"): 50.0, ("STALE", BUY): 100.0}
    c = ga.cohort(["STALE"], BUY, EVAL, _PS(lut), 0.0)
    assert c["n"] == 1 and c["n_measured"] == 0 and c["n_unknown"] == 1
    assert c["unknown"] == ["STALE"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
