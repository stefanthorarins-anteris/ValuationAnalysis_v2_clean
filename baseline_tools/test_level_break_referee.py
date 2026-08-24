"""
The LEVEL-BREAK REFEREE: the derived leg as a second opinion on the real leg (offline).

WHAT IT IS FOR.  `SIMINN.IC` adjClose steps 0.2256 -> 9.23 across one anchor and reads a
+36,787% 36-month return; the derived leg reads +25.4%.  A referee detects that without
needing an invented cap on the return.

TWO THINGS THESE TESTS PIN THAT THE BRIEF DID NOT ASK FOR, both measured:

  1. IT MUST NOT AUTO-CORRECT.  3 of the 17 flagged cells on the run machine's grid are the
     DERIVED leg's fault -- g_derived == 1.000000 exactly, because the derived anchor rule
     priced both ends off the SAME filing.  `test_a_same_filing_derived_leg_is_labelled_
     UNINFORMATIVE_not_a_real_break` is the guard: an automatic "real is suspect" rule would
     have shipped three known-wrong corrections.
  2. THE CONJUNCTION IS THE INSTRUMENT, not the gap alone.  Of 392 cells whose real step is
     >= 5x, 354 (90.3%) are corroborated by the derived leg -- genuine volatility.
     `test_a_CORROBORATED_extreme_move_is_not_flagged` is that case.

AND THE THRESHOLDS ARE A CHOICE.  The disagreement distribution is CONTINUOUS (p99 0.661,
p99.5 0.863, p99.9 1.969, max 5.511; widest jump in the top 0.5% is 0.78) -- unlike the
payload floor, there is no separating band, so these tests pin BEHAVIOUR at the shipped cut
and deliberately do not claim the cut is derivable.
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

import derived_prices as dpx
import returns_core as rc

ANCH = ["2020-12-31", "2021-12-31", "2022-12-30"]
A, B = ANCH[0], ANCH[1]


def _real(tmp_path, priced):
    rows = [{"date_requested": a, "date_actual": a, "symbol": s, "adjClose": px}
            for s, per in priced.items() for a, px in per.items()]
    p = tmp_path / "real_prices.csv"
    pd.DataFrame(rows, columns=["date_requested", "date_actual", "symbol",
                                "adjClose"]).to_csv(p, index=False)
    return rc.PriceSource(str(p), anchors=ANCH)


def _derived(levels, real=None, shares=1000.0, **kw):
    rows = []
    for sym, per in levels.items():
        for a, lvl in per.items():
            rows.append((sym, f"{a[:4]}-12-31", lvl * shares, shares, 0.0, "USD"))
    df = pd.DataFrame(rows, columns=["source", "periodEndDate", "marketCap",
                                     "weightedAverageShsOut", "dividendsPaid",
                                     "reportedCurrency"])
    df["price"] = df["marketCap"] / df["weightedAverageShsOut"]
    return dpx.DerivedPriceSource(df, anchors=ANCH, benchmark_source=real, **kw)


def _run(tmp_path, real_px, der_lv, **kw):
    real = _real(tmp_path, real_px)
    der = _derived(der_lv, real=real)
    return dpx.level_break_candidates(real, der, **kw), real, der


# --------------------------------------------------------------------------- #
#  THE CONJUNCTION                                                            #
# --------------------------------------------------------------------------- #
def test_an_UNCORROBORATED_extreme_move_is_flagged(tmp_path):
    """The SIMINN shape: the real leg 41x in a period, the derived leg flat."""
    cand, _r, _d = _run(tmp_path,
                        {"BREAK": {A: 0.2256, B: 9.23}},
                        {"BREAK": {A: 1000.0, B: 930.0}})
    assert len(cand) == 1
    row = cand.iloc[0]
    assert row["symbol"] == "BREAK"
    assert row["verdict"] == "legs_disagree"
    assert row["real_step"] > 40
    assert row["log_gap"] > 3.5


def test_a_CORROBORATED_extreme_move_is_not_flagged(tmp_path):
    """90.3% of >=5x real steps are corroborated: a genuine 10-bagger must not be flagged, or
    the referee would fire on exactly the names a value screen is looking for."""
    cand, _r, _d = _run(tmp_path,
                        {"REALWIN": {A: 1.0, B: 10.0}},
                        {"REALWIN": {A: 500.0, B: 5100.0}})
    assert cand.empty


def test_a_BIG_GAP_with_a_SMALL_real_step_is_not_flagged(tmp_path):
    """The other half of the conjunction.  A 2x real move with a wildly disagreeing derived
    leg is a derived-leg quality problem, not a level break in the real prices -- and the
    referee exists to protect the real grid, not to audit the panel."""
    cand, _r, _d = _run(tmp_path,
                        {"NOISY": {A: 10.0, B: 20.0}},
                        {"NOISY": {A: 1000.0, B: 150.0}})
    assert cand.empty


def test_a_move_just_under_each_cut_is_not_flagged(tmp_path):
    """Boundary behaviour at the shipped cut, pinned so a future tweak is visible."""
    cand, _r, _d = _run(tmp_path,
                        {"EDGE": {A: 1.0, B: 4.9}},          # step 4.9 < 5
                        {"EDGE": {A: 1000.0, B: 1000.1}})
    assert cand.empty


# --------------------------------------------------------------------------- #
#  IT DETECTS.  IT DOES NOT CORRECT.                                          #
# --------------------------------------------------------------------------- #
def test_a_same_filing_derived_leg_is_labelled_UNINFORMATIVE_not_a_real_break(tmp_path):
    """THE GUARD AGAINST AUTO-CORRECTION.  With no filing in the later anchor year the derived
    anchor rule carries the earlier one forward, so g_derived is exactly 1.0 and the derived
    leg has NO opinion.  3 of the 17 real flagged cells are this -- SNYR, MAXENT-B.ST, OBD.L.
    Overriding the real leg on those would corrupt a good price with a fabricated flat."""
    cand, _r, der = _run(tmp_path,
                         {"CARRIED": {A: 23.5, B: 1.0}},
                         {"CARRIED": {A: 1000.0}})        # no level at B at all
    assert len(cand) == 1
    row = cand.iloc[0]
    assert row["g_derived"] == pytest.approx(1.0)
    assert bool(row["derived_same_filing"]) is True
    assert row["verdict"] == "derived_uninformative"
    assert der.picked_period_end("CARRIED", A) == der.picked_period_end("CARRIED", B)


def test_running_the_referee_changes_NO_price(tmp_path):
    """It is a detector.  The price sources must be byte-identical before and after."""
    real = _real(tmp_path, {"BREAK": {A: 0.2256, B: 9.23}})
    der = _derived({"BREAK": {A: 1000.0, B: 930.0}}, real=real)
    before_r, before_d = dict(real._lut), dict(der._lut)
    dpx.level_break_candidates(real, der)
    assert real._lut == before_r
    assert der._lut == before_d


def test_the_verdict_never_asserts_which_leg_is_wrong(tmp_path):
    """Among the survivors, GDHG 2023->2024 has BOTH legs collapsing (0.0557 vs 0.0069) and
    disagreeing only about how far.  So 'legs_disagree' is the strongest honest verdict."""
    cand, _r, _d = _run(tmp_path,
                        {"BOTHDOWN": {A: 10.0, B: 0.557}},
                        {"BOTHDOWN": {A: 1000.0, B: 6.9}})
    assert set(cand["verdict"]) <= {"legs_disagree", "derived_uninformative"}
    assert "real_is_wrong" not in set(cand["verdict"])


# --------------------------------------------------------------------------- #
#  BLIND SPOTS, asserted rather than promised                                  #
# --------------------------------------------------------------------------- #
def test_a_name_only_the_REAL_leg_prices_produces_no_cell(tmp_path):
    """A referee needs two opinions.  This is why the seven venues the real grid cannot price
    contribute ZERO cells -- and why a small flagged count must never be read as a clean bill
    of health for them."""
    cand, _r, _d = _run(tmp_path,
                        {"ONLYREAL": {A: 1.0, B: 100.0}},
                        {"OTHER": {A: 1000.0, B: 1100.0}})
    assert cand.empty


def test_a_break_BOTH_legs_share_reads_as_agreement(tmp_path):
    """A vendor error that hits marketCap and adjClose together moves them in lockstep.  The
    referee is structurally blind to it -- stated in the docstring, asserted here."""
    cand, _r, _d = _run(tmp_path,
                        {"SHARED": {A: 1.0, B: 50.0}},
                        {"SHARED": {A: 1000.0, B: 50000.0}})
    assert cand.empty


def test_a_break_SPREAD_over_two_periods_is_diluted_below_the_cut(tmp_path):
    """Adjacent anchors only, so a break taken in two steps hides.  Each step here is ~7x on
    the real leg against a flat derived leg, but split so neither single log gap reaches the
    cut... and the test asserts what actually happens rather than what would be nice."""
    real_px = {"SLOW": {A: 1.0, B: 2.5, ANCH[2]: 6.0}}
    der_lv = {"SLOW": {a: 1000.0 for a in ANCH}}
    cand, _r, _d = _run(tmp_path, real_px, der_lv)
    assert cand.empty, "a 6x break taken in two 2.5x steps is invisible to this instrument"


# --------------------------------------------------------------------------- #
#  WIRING                                                                     #
# --------------------------------------------------------------------------- #
def test_the_thresholds_are_read_at_CALL_time(monkeypatch, tmp_path):
    """So a caller can widen the net without editing the module -- and so the mutation test
    below actually mutates something."""
    #  real step 6.0, derived growth 4.0 -> log gap = ln(1.5) = 0.405, above the 0.3 test cut
    #  and below the shipped 1.0.  (An earlier draft used 1.4 and asserted "~0.45"; the real
    #  gap was ln(6/1.4) = 1.455, which the shipped cut catches -- the arithmetic was mine.)
    real_px = {"MILD": {A: 1.0, B: 6.0}}
    der_lv = {"MILD": {A: 1000.0, B: 4000.0}}
    cand, _r, _d = _run(tmp_path, real_px, der_lv)
    assert cand.empty
    cand2, _r2, _d2 = _run(tmp_path, real_px, der_lv, min_log_gap=0.3)
    assert len(cand2) == 1


def test_MUTATION_dropping_the_step_gate_floods_the_flag_list(tmp_path):
    """Sensitivity control for the conjunction: with the step gate at 1.0 the corroborated
    small-move noise comes back in, which is what the gate exists to keep out."""
    real_px = {"NOISY": {A: 10.0, B: 20.0}}
    der_lv = {"NOISY": {A: 1000.0, B: 150.0}}
    assert _run(tmp_path, real_px, der_lv)[0].empty
    flooded, _r, _d = _run(tmp_path, real_px, der_lv, min_real_step=1.0)
    assert len(flooded) == 1


def test_the_report_states_the_corpus_and_the_blind_spots(tmp_path):
    real = _real(tmp_path, {"BREAK": {A: 0.2256, B: 9.23}})
    der = _derived({"BREAK": {A: 1000.0, B: 930.0}}, real=real)
    rep = dpx.level_break_report(real, der)
    assert rep["n_cells_flagged"] == 1
    assert rep["n_legs_disagree"] == 1
    assert rep["n_derived_uninformative"] == 0
    assert rep["n_symbols_both_legs_price"] == 1
    assert "DETECT AND PUBLISH ONLY" in rep["action"]
    assert "seven venues" in rep["blind_to"]


def test_an_empty_candidate_set_still_returns_the_full_schema(tmp_path):
    """A caller that writes this to CSV must not get a different shape on a clean run."""
    cand, _r, _d = _run(tmp_path, {"CALM": {A: 10.0, B: 11.0}},
                        {"CALM": {A: 1000.0, B: 1100.0}})
    assert cand.empty
    for col in ("symbol", "venue", "log_gap", "real_step", "verdict"):
        assert col in cand.columns


# --------------------------------------------------------------------------- #
#  PIPELINE WIRING -- default ON, because it only prints                       #
# --------------------------------------------------------------------------- #
def test_the_referee_stage_is_ON_by_default_and_silenceable(tmp_path, monkeypatch, capsys):
    """It overrides nothing, so the default is ON.  `level_break_referee=0` silences it."""
    import pipeline_analysis as pa
    monkeypatch.setattr(pa, "_PRICES_CSV", str(tmp_path / "nope.csv"))
    logged = []
    #  absent grid -> the stage must SKIP with a line, never raise
    assert pa.run_level_break_referee_stage({}, {}, logged.append) is None
    assert any("skipped" in x for x in logged)
    logged.clear()
    assert pa.run_level_break_referee_stage({}, {"level_break_referee": 0},
                                            logged.append) is None
    assert any("disabled by config" in x for x in logged)


def test_an_absent_panel_SKIPS_rather_than_failing_the_run(tmp_path, monkeypatch):
    """A referee needs two legs.  With one it has no second opinion, and that is a skip, not
    an error -- the analysis stages below it are unaffected either way."""
    import pipeline_analysis as pa
    rows = [{"date_requested": a, "date_actual": a, "symbol": "X", "adjClose": 1.0}
            for a in rc.DEFAULT_ANCHORS]
    csv = tmp_path / "real_prices.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    monkeypatch.setattr(pa, "_PRICES_CSV", str(csv))
    monkeypatch.setattr(pa, "_PRICES_2025_CSV", str(tmp_path / "absent.csv"))
    monkeypatch.setattr(dpx, "DEFAULT_PANEL_GLOB", str(tmp_path / "no_such_panel_*.pickle"))
    logged = []
    assert pa.run_level_break_referee_stage({}, {}, logged.append) is None
    assert any("panel unavailable" in x for x in logged)


def test_the_suite_entry_point_runs_the_referee_stage():
    """AST guard on the WIRING: a detector nobody calls detects nothing."""
    import ast
    import inspect
    import pipeline_analysis as pa
    tree = ast.parse(inspect.getsource(pa))
    names = {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
             for n in ast.walk(tree) if isinstance(n, ast.Call)}
    args = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and (getattr(n.func, "id", None) == "_run_stage"):
            args |= {getattr(a, "id", None) for a in n.args}
    assert "run_level_break_referee_stage" in args, \
        "the referee stage is defined but never dispatched by the suite"
