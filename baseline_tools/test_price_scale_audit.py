"""THE 1000x VENDOR PRICE-SCALE DETECTOR fires on the name that motivated it.

WHY THAT IS THE FIRST TEST.  The detector the issue register proposed -- synthetic price
(`marketCap/weightedAverageShsOut`) against the grid -- does NOT fire on ATRI, because FMP
scales `marketCap` by the same 1/1000 and the ratio cancels to 1.04.  A guard blind to its own
motivating case is worse than no guard, because it reads as an all-clear.  So the first thing
pinned here is that the shipped check catches the ATRI shape, and the second is that the
rejected check demonstrably does not -- kept as an executable record of WHY the design changed,
so nobody re-proposes it from the register text.

WHAT THESE TESTS CANNOT SEE, stated plainly:
  * They cannot tell a real deep-value company from a scaled one.  The check is a HEURISTIC on
    a valuation ratio; a genuine liquidation at 0.015x book is indistinguishable here and will
    be flagged.  That is why the stage reports and never gates.
  * They cannot detect a scaling defect applied CONSISTENTLY to the balance sheet as well as
    the price -- if equity were scaled by the same 1/1000, price/book returns to normal and
    both checks go quiet.  Nothing on disk would reveal that; only `vwap` (which the grid does
    not carry) or an external price source would.
  * The fixtures are hand-built frames.  A change that stops `cdx_df` carrying
    `bookValuePerShare` would make the audit silently report zero names, and only
    `test_the_audit_says_so_when_it_cannot_run` covers that -- by asserting it complains
    rather than returning a clean empty result.

OFFLINE.  No network, no pickle, no price fetch.
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

import price_scale_audit as psa


def _panel(rows):
    """rows: (source, marketCap, shares, equity, revenue, currency) per quarter."""
    out = []
    for i, (src, mcap, sh, eq, rev, cur) in enumerate(rows):
        out.append({"source": src,
                    "date": pd.Timestamp("2021-03-31") + pd.DateOffset(months=3 * (i % 8)),
                    "price": mcap / sh,
                    "bookValuePerShare": eq / sh,
                    "revenue": rev,
                    "weightedAverageShsOut": sh,
                    "reportedCurrency": cur})
    return pd.DataFrame(out)


def _repeat(src, mcap, sh, eq, rev, cur="USD", n=8):
    return [(src, mcap, sh, eq, rev, cur)] * n


#  ATRI's REAL saved numbers, from `delisted_out/dead_fundamentals_20260713_104350.pickle`
#  (2021-12-31 quarter): marketCap 1,269,524.90 -- itself scaled 1/1000 -- against 1,801,000
#  shares and a balance sheet FMP serves unscaled.  Equity is reconstructed from the
#  `bookValuePerShare` the same file reports (135.65 at the 2024 quarters; ~123 here).
_ATRI = _repeat("ATRI", 1269524.90, 1801000, 1801000 * 123.0, 48773000 * 4)


def test_the_shipped_check_FIRES_on_ATRI_the_name_that_motivated_it():
    out = psa.check_fundamentals_internal(_panel(_ATRI))
    row = out[out["source"] == "ATRI"]
    assert len(row) == 1, "ATRI was not flagged at all"
    assert row.iloc[0]["severity"] == "ALARM"
    assert row.iloc[0]["price_over_book"] < 0.02
    #  and the corroboration that makes it readable: one decade makes it ordinary
    assert 0.5 < row.iloc[0]["pb_x1000"] < 50


def test_the_REJECTED_check_does_NOT_fire_on_ATRI_which_is_why_it_was_rejected(tmp_path):
    """Executable record of the measurement that killed the register's proposal.  FMP applies
    the 1/1000 to `marketCap` too, so synthetic-vs-grid cancels to ~1.0 on the broken name."""
    grid = pd.DataFrame([{"date_requested": d, "symbol": "ATRI", "adjClose": 0.67507}
                         for d in ("2021-12-31", "2022-12-30", "2023-12-29")])
    gpath = os.path.join(str(tmp_path), "real_prices.csv")
    grid.to_csv(gpath, index=False)
    panel = _panel(_ATRI)
    panel["date"] = pd.Timestamp("2021-12-31")
    out, _ = psa.check_synthetic_vs_grid(panel, gpath)
    assert "ATRI" not in set(out["source"] if len(out) else []), (
        "synthetic-vs-grid flagged ATRI -- if this now passes, re-read the docstring: it "
        "would mean marketCap is no longer scaled and the rejected check has become viable")


def test_a_normal_company_is_not_flagged():
    """price/book 2.0 -- the live panel's median is 1.91."""
    out = psa.check_fundamentals_internal(_panel(_repeat("GOOD", 2000.0, 1000, 1000.0, 5000)))
    assert not len(out)


def test_a_single_quarter_of_noise_is_not_a_scaling_defect():
    """The defect is persistent by construction -- it is how the vendor stores the series.
    One stale marketCap must not raise an alarm."""
    rows = _repeat("BLIP", 2000.0, 1000, 1000.0, 5000, n=7) + \
        [("BLIP", 1.0, 1000, 1000.0, 5000, "USD")]
    out = psa.check_fundamentals_internal(_panel(rows))
    assert not len(out), "a one-quarter outlier moved the median"


def test_the_check_is_currency_free_and_the_pence_venues_cannot_confound_it():
    """Both legs are per-share quantities in the REPORTING currency, so a GBp-quoting London
    listing -- 227 of the 249 decade-scale disagreements the grid-based check produces -- is
    structurally invisible to this one.  That is the whole reason it is check A."""
    lon = _repeat("PENCE.L", 2000.0, 1000, 1000.0, 5000, cur="GBP")
    krw = _repeat("ADR", 2000.0, 1000, 1000.0, 5000, cur="KRW")
    out = psa.check_fundamentals_internal(_panel(lon + krw))
    assert not len(out)


# --------------------------------------------------------------------------- #
#  Check B: classify, do not alarm                                            #
# --------------------------------------------------------------------------- #
def _grid(tmp_path, rows):
    p = os.path.join(str(tmp_path), "real_prices.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


_ANCHORS = ("2021-12-31", "2022-12-30", "2023-12-29", "2024-12-31")


def _panel_at_anchors(source, price, cur="USD"):
    return pd.DataFrame([{"source": source, "date": pd.Timestamp(a), "price": price,
                          "bookValuePerShare": price / 2.0, "revenue": 1000.0,
                          "weightedAverageShsOut": 100.0, "reportedCurrency": cur}
                         for a in _ANCHORS])


def test_a_pence_venue_is_classified_MINOR_UNIT_not_alarmed(tmp_path):
    """The LSE quotes in pence against GBP reporting: an exact 1/100 on hundreds of names.
    Alarming on that would bury the one finding that matters under 227 non-findings."""
    g = _grid(tmp_path, [{"date_requested": a, "symbol": "X.L", "adjClose": 100.0}
                         for a in _ANCHORS])
    out, _ = psa.check_synthetic_vs_grid(_panel_at_anchors("X.L", 1.0, "GBP"), g)
    assert list(out["verdict"]) == ["MINOR_UNIT"]


def test_an_FX_rate_is_classified_CURRENCY_MISMATCH_not_a_scaling_defect(tmp_path):
    """BSAC's CLP/USD median ratio is 1071.65 on the real grid -- 7.2% off 1000.  A loose
    'near 1000x' rule would have called a Chilean bank a vendor defect.  The tolerance is set
    against that measurement, so this test is the one that pins it."""
    g = _grid(tmp_path, [{"date_requested": a, "symbol": "BSAC", "adjClose": 1.0}
                         for a in _ANCHORS])
    out, _ = psa.check_synthetic_vs_grid(_panel_at_anchors("BSAC", 1071.65, "CLP"), g)
    assert list(out["verdict"]) == ["CURRENCY_MISMATCH"]


def test_an_unexplained_exact_decade_IS_raised(tmp_path):
    """The case check B exists for: the grid is scaled but `marketCap` is not, on a venue with
    no minor-unit convention."""
    g = _grid(tmp_path, [{"date_requested": a, "symbol": "ZZZ", "adjClose": 0.5}
                         for a in _ANCHORS])
    out, _ = psa.check_synthetic_vs_grid(_panel_at_anchors("ZZZ", 500.0, "USD"), g)
    assert list(out["verdict"]) == ["SCALING_SUSPECT"]


def test_a_10x_disagreement_is_below_the_floor(tmp_path):
    """10x is within reach of ordinary marketCap staleness plus dividend back-adjustment over
    six years; only >= 100x is treated as a units question."""
    g = _grid(tmp_path, [{"date_requested": a, "symbol": "ZZZ", "adjClose": 5.0}
                         for a in _ANCHORS])
    out, _ = psa.check_synthetic_vs_grid(_panel_at_anchors("ZZZ", 50.0, "USD"), g)
    assert not len(out)


# --------------------------------------------------------------------------- #
#  The stage reports and NEVER mutates                                        #
# --------------------------------------------------------------------------- #
def test_the_audit_changes_nothing_it_reads():
    """Q-38 stayed OPEN rather than being parked because its detector reaches names that are
    live in scoring.  The boundary is that it reads and prints; nothing downstream consumes
    it and no price is corrected."""
    panel = _panel(_ATRI + _repeat("GOOD", 2000.0, 1000, 1000.0, 5000))
    before = panel.copy(deep=True)
    psa.run_audit(panel, prices_csv=None, log=lambda *a: None)
    pd.testing.assert_frame_equal(panel, before)


def test_the_stage_is_wired_in_and_does_not_gate():
    import inspect
    import pipeline_analysis as pa
    src = inspect.getsource(pa.run_post_pick_analysis_suite) \
        if hasattr(pa, "run_post_pick_analysis_suite") else inspect.getsource(pa)
    assert "_audit_price_scale_stage" in src
    stage = inspect.getsource(pa._audit_price_scale_stage)
    for forbidden in ("adjClose =", "= psa.correct", "drop(", "raise RuntimeError"):
        assert forbidden not in stage, f"the stage looks like it does more than report: {forbidden}"


def test_the_audit_says_so_when_it_cannot_run():
    """Silence must never be the output of a missing input -- that is how a guard becomes
    decoration.  An empty or column-less panel must produce a complaint, not a clean zero."""
    said = []
    psa.run_audit(pd.DataFrame(), log=said.append)
    assert any("did not run" in s for s in said)


def test_the_report_names_its_own_blind_spots_every_run():
    """A detector that prints '0 flagged' and nothing else reads as an all-clear for a
    question it never asked."""
    said = []
    psa.run_audit(_panel(_repeat("GOOD", 2000.0, 1000, 1000.0, 5000)),
                  prices_csv=None, log=said.append)
    text = "\n".join(said)
    assert "BLIND SPOT" in text.upper()
    assert "marketCap" in text, "the reason check B misses ATRI is not stated"
    assert "absent from THIS panel" in text
    #  The blind spot that mattered most was not the one the paragraph named.  Check A reads
    #  the panel it is given, and the LIVE panel is survivor-only -- ATRI has 0 rows in it on
    #  both 08-29 and 08-31 -- so the check written because check B cancels on ATRI could not
    #  see ATRI either.  The paragraph must now say which POPULATION it read and that a second
    #  pass covers the other one, or a reader takes one pass for both.
    assert "ATRI" in text and "0 rows in cdx_df" in text
    assert "PIT dead-merged" in text, (
        "the report does not say that the dead-merged population is covered elsewhere")


# --------------------------------------------------------------------------- #
#  THE CONTAINMENT PARAGRAPH IS COMPUTED, NOT RECITED                         #
# --------------------------------------------------------------------------- #
def _mixed_panel(n_good=40, bad=("BADA", "BADB", "BADC")):
    """`n_good` ordinary names + `bad` scaled by 1/1000 (price only, ATRI's shape)."""
    rows = []
    for i in range(n_good):
        rows += _repeat("G%03d" % i, 2000.0, 1000, 1000.0, 5000)
    for b in bad:
        rows += _repeat(b, 2.0, 1000, 1000.0, 5000)
    return _panel(rows)


def test_the_containment_numbers_MOVE_WITH_THE_PANEL():
    """THE DEFECT THIS PINS.  `price_scale_audit` printed, unconditionally whenever
    `n_alarm > 0`, a frozen 2026-08-29 measurement -- "seven of eight rank 1-7 of 4,928 ... No
    shipped pick is affected: best Stage-1 rank is 119 of 4,934" -- against a 2026-08-31 panel
    of 4,941.  It read as tonight's number, it happened to be true, and the code did not check
    it: the first run whose ALARM set moved would have asserted a false all-clear about the
    shipped list in the log the CEO reads.

    Two panels differing in SIZE and in ALARM COUNT must therefore produce different numbers.
    A frozen paragraph passes neither half of this."""
    def _run(panel, bo):
        said = []
        psa.run_audit(panel, prices_csv=None, stage1_scores=bo,
                      shipped_sources=["G000"], run_grid_check=False, log=said.append)
        return "\n".join(said)

    small = _mixed_panel(n_good=40, bad=("BADA", "BADB"))
    large = _mixed_panel(n_good=120, bad=("BADA", "BADB", "BADC"))
    bo_small = pd.DataFrame({"source": sorted(small["source"].unique()),
                             "score": np.linspace(1, 0, small["source"].nunique())})
    bo_large = pd.DataFrame({"source": sorted(large["source"].unique()),
                             "score": np.linspace(1, 0, large["source"].nunique())})
    t_small, t_large = _run(small, bo_small), _run(large, bo_large)

    assert "of 42 on it" in t_small, t_small          # 40 good + 2 scaled
    assert "of 123 on it" in t_large, t_large         # 120 good + 3 scaled
    #  "RANKABLE" since 2026-09-01: the head count's denominator is the number of flagged
    #  names that CAN be ranked, not the number flagged.  A fully-refused source has no
    #  computable bookToPrice, so counting it in the denominator dropped the MOST
    #  contaminated names out of the numerator and read as an all-clear (review 3, S3-2).
    #  Both fixtures here are rankable, so the numbers are unchanged -- only the word moved.
    assert "2 of 2 RANKABLE flagged names" in t_small
    assert "3 of 3 RANKABLE flagged names" in t_large
    assert "of 42 of" not in t_large, "the panel size did not move with the panel"


def _stripped(fn):
    """`fn`'s source with every docstring removed.

    SCANNED ON THE BODY, NOT THE PROSE.  The dated 08-29 numbers belong in the docstring --
    that is the record of what went wrong -- and a token scan that read them there would fail
    on its own explanation, which trains people to delete the explanation.  Same lesson as
    `test_price_grid_refetch_decision._strip_docstrings`, and it is a lesson this repo has
    already had to learn once."""
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Module)) and ast.get_docstring(node) is not None:
            node.body = node.body[1:]
    return ast.unparse(tree)


def test_no_frozen_08_29_measurement_survives_in_what_the_audit_PRINTS():
    """The literals themselves, banned from the executable body.  A recomputation that leaves
    the old numbers in a neighbouring printed sentence has fixed nothing a reader can see."""
    src = _stripped(psa.run_audit) + _stripped(psa._containment_lines)
    for frozen in ("4,928", "4,934", "seven of eight", "NINETEEN", "119"):
        assert frozen not in src, (
            "a hardcoded 2026-08-29 measurement is still printed as if computed: %r" % frozen)


def test_an_absent_input_produces_NOT_CHECKED_and_never_an_all_clear():
    """The other half of the fix.  With no selected list and no Stage-1 ranking, the audit
    must SAY it did not check -- an absent input coming out as 'no shipped pick is affected'
    is the same defect one level down."""
    said = []
    psa.run_audit(_mixed_panel(), prices_csv=None, run_grid_check=False, log=said.append)
    text = "\n".join(said)
    assert "CONTAINMENT NOT CHECKED" in text
    assert "NOT an all-clear" in text
    for reassurance in ("containment holds", "None of the", "Best Stage-1 rank"):
        assert reassurance not in text, (
            "the audit reassured about a list it was never given: %r" % reassurance)


def test_a_flagged_name_INSIDE_the_selected_list_is_reported_as_NOT_CONTAINED():
    """The case the frozen paragraph could never report, because it asserted the opposite by
    construction.  This is the whole reason the sentence had to become a computation."""
    panel = _mixed_panel()
    said = []
    psa.run_audit(panel, prices_csv=None, shipped_sources=["G000", "BADB"],
                  run_grid_check=False, log=said.append)
    text = "\n".join(said)
    assert "NOT CONTAINED" in text and "BADB" in text
    assert "containment holds" not in text


def test_a_flagged_name_INSIDE_the_stage1_cutoff_gets_its_own_sentence():
    """A negative margin is a different statement, not a smaller number.

    A flagged name inside the top-100 has already reached the Stage-2 pool carrying a
    contaminated `bookToPrice` -- whether it survives to the shipped list is a separate
    question.  Formatting that as "a margin of -12 places" would bury the one case the
    containment paragraph exists to surface, which is how the frozen version read: it could
    only ever describe comfort."""
    panel = _mixed_panel(n_good=30, bad=("BADA",))
    #  BADA ranked 2nd of 31 -> well inside a top-10 cutoff
    order = ["G000", "BADA"] + ["G%03d" % i for i in range(1, 30)]
    bo = pd.DataFrame({"source": order, "score": np.linspace(1, 0, len(order))})
    said = []
    psa.run_audit(panel, prices_csv=None, stage1_scores=bo, shipped_sources=["G000"],
                  topn_stage1=10, run_grid_check=False, log=said.append)
    text = chr(10).join(said)
    assert "INSIDE THE STAGE-1 CUTOFF" in text and "BADA" in text
    assert "margin" not in text, "a name inside the cutoff was reported as a margin"


def test_check_B_says_NOT_RUN_rather_than_reporting_no_disagreement():
    """A check that did not run and a check that found nothing are different facts about the
    data.  On the dead-merged panel the grid carries almost none of the names, so a silent
    'no decade-scale disagreement' would be an artifact of absence read as an all-clear."""
    said = []
    psa.run_audit(_mixed_panel(), prices_csv=None, run_grid_check=False, log=said.append)
    text = "\n".join(said)
    assert "NOT RUN on this panel" in text
    assert "no decade-scale disagreement" not in text


def test_bookToPrice_ranking_is_the_same_by_either_route():
    """`equity/marketCap` (the scorer's own definition) and `bookValuePerShare/price` are the
    same ratio with `shares` cancelled, so the ranking must not depend on which columns the
    panel happens to carry.  The fallback exists because `cdx_df` variants differ; a fallback
    that ranked differently would be a second opinion wearing the first one's label."""
    panel = _mixed_panel(n_good=20)
    with_direct = panel.assign(
        marketCap=panel["price"] * panel["weightedAverageShsOut"],
        totalStockholdersEquity=panel["bookValuePerShare"] * panel["weightedAverageShsOut"])
    names = sorted(panel["source"].unique())
    a = psa.bookToPrice_ranks(panel, names)
    b = psa.bookToPrice_ranks(with_direct, names)
    assert a[3] == "bookValuePerShare/price" and b[3] == "equity/marketCap"
    assert a[0] == b[0], "the two routes disagree about the ranking"
    assert a[1] == b[1]


# --------------------------------------------------------------------------- #
#  THE AUDIT MUST REACH THE POPULATION IT WAS BUILT FOR                       #
# --------------------------------------------------------------------------- #
#  ATRI has ZERO rows in the live `cdx_df` on 2026-08-29 and on 2026-08-31: it delisted in
#  2024, and the live panel is survivor-only.  Check B cancels on the ATRI shape by
#  construction (FMP scales `marketCap` by the same 1/1000), and check A -- the check written
#  BECAUSE of that -- was reading a panel ATRI is not in.  Both checks blind to the motivating
#  name, in a module whose first test asserts it "fires on the name that motivated it".
#  Where ATRI IS live is the PIT dead-merged pool, and there `bookToPrice` is a Tier-B,
#  Sign +1, higher-is-better input to every backtest ranking.
def _pa():
    import pipeline_analysis as pa
    return pa


def test_the_suite_runs_check_A_over_the_DEAD_MERGED_panel_too():
    """Wiring, not availability: the second pass has to be a stage the run executes."""
    import inspect
    pa = _pa()
    src = inspect.getsource(pa.run_analysis_suite)
    assert "_audit_price_scale_pit_stage" in src, (
        "the backtest population is still unaudited -- check A reads only the live panel")
    #  and it must come AFTER the grid, because that is where the graded selections exist
    assert src.index("depth x horizon avg-TR grid") < src.index(
        "_audit_price_scale_pit_stage"), (
        "the PIT audit runs before per_anchor exists, so it can check no containment")


def test_the_PIT_pass_does_NOT_run_check_B():
    """Its names are largely the ones the price grid does not carry, so 'no disagreement'
    would be an artifact of absence -- an all-clear manufactured by a missing input."""
    import inspect
    src = inspect.getsource(_pa()._audit_price_scale_pit_stage)
    assert "run_grid_check=False" in src


def test_the_PIT_pass_REFUSES_to_run_when_the_run_is_not_survivorship_clean():
    """`_build_pit_inputs` degrades to `merged is dmdic` when the delisted inputs are absent.
    Auditing that a second time would print an identical report under a heading claiming it
    covered the dead names -- a stage that READS as coverage it does not have is worse than
    an absent stage."""
    said = []
    out = _pa()._audit_price_scale_pit_stage({"cdx_df": _mixed_panel()}, {}, None,
                                             False, said.append)
    assert out is None
    text = chr(10).join(said)
    assert "SKIPPED" in text and "NOT audited" in text


def test_the_PIT_pass_NAMES_the_alarms_the_live_pass_could_not_see():
    """The difference IS the finding.  A contaminated name present in the dead-merged panel
    and absent from the live one is a name whose `bookToPrice` is 1000x too favourable in
    every backtest ranking and invisible to the shipped-list audit -- the ATRI shape exactly.

    DEAD1 stands in for ATRI: scaled, and absent from the live ALARM set."""
    live_panel = _mixed_panel(n_good=30, bad=("BADA",))
    pit_panel = _mixed_panel(n_good=30, bad=("BADA", "DEAD1"))
    live = psa.run_audit(live_panel, prices_csv=None, run_grid_check=False,
                         log=lambda *_a: None)
    said = []
    _pa()._audit_price_scale_pit_stage(
        {"cdx_df": pit_panel},
        {"buy2021": {"top20_deduped": ["G000", "G001"], "ranking": ["G000", "G001"]}},
        live, True, said.append)
    text = chr(10).join(said)
    assert "DEAD1" in text, text
    assert "ONLY IN THE BACKTEST POPULATION" in text
    assert "BADA" not in text.split("ONLY IN THE BACKTEST POPULATION")[1].split(chr(10))[0], (
        "a name the live pass already flagged was reported as new")


def test_the_PIT_pass_says_so_when_it_finds_nothing_new():
    """'The second pass found nothing' and 'the second pass did not run' must not look alike
    -- the distinction this module already draws for check B, applied to itself."""
    panel = _mixed_panel(n_good=30, bad=("BADA",))
    live = psa.run_audit(panel, prices_csv=None, run_grid_check=False, log=lambda *_a: None)
    said = []
    _pa()._audit_price_scale_pit_stage({"cdx_df": panel}, {}, live, True, said.append)
    text = chr(10).join(said)
    assert "no ALARM name is unique to the dead-merged panel" in text


def test_the_live_pass_is_handed_the_runs_OWN_ranking_and_shipped_list():
    """Without them the containment paragraph can only say NOT CHECKED -- which is honest but
    useless, and the frozen paragraph existed precisely because nobody had wired the inputs
    that would let it be computed."""
    import inspect
    src = inspect.getsource(_pa()._audit_price_scale_stage)
    assert "stage1_scores=" in src and "BoScore_df" in src
    assert "shipped_sources=" in src and "postRank" in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
