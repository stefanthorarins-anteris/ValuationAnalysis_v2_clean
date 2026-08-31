"""THE MEASUREMENT-BASIS STAMP reaches the two artifacts a reader actually KEEPS.

WHY THESE EXIST.  `53799d4` said "Basis is stamped everywhere."  It was stamped in
`rank_all_anchors`'s log lines and in the two-clause banner -- both of which live in a run
log nobody archives -- and in NEITHER of the two files that outlive the run: the depth x
horizon grid report, and the `scoring_compare` side-by-side.  So the first vetoed grid report
ever produced was byte-shaped exactly like every un-vetoed report in the archive, and
`scoring_compare` put a documented-un-vetoed ORIGINAL column beside three vetoed columns in
one table with no label on any of them.

WHAT THESE TESTS CANNOT SEE, stated because this project keeps shipping guards that are blind
to the thing beneath them:
  * They check that the stamp is PRESENT and that it AGREES with the `basis` values on the
    per-anchor dicts.  They do not and cannot check that `stage2_pit` wrote a TRUE stamp: if
    the veto silently no-ops while still stamping VETOED, every assertion here still passes
    and the report confidently prints the wrong basis.  Truth of the stamp is
    `test_stage1_veto_pit`'s job, not this file's.
  * `tag()` is deliberately coarse, so these tests cannot detect two DIFFERENT vetoed bases
    (e.g. different flag sets that both stamp "VETOED") being pooled.  `of()` catches that
    via its MIXED path only when the stamp STRINGS differ.
  * Nothing here exercises the real pipeline inputs; the fixtures are hand-built dicts, so a
    change that stops `rank_all_anchors` populating `basis` at all would be caught by the
    key-name assertion below and by nothing else in this file.

OFFLINE.  No network, no pickle, no price CSV.
"""
import os
import sys

import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import basis_stamp as bs
import depth_horizon_grid as dhg
import scoring_compare as sc

_VETOED = "VETOED (stage-1 solvency gate applied, %d ejected)"
_UNVETOED = "un-vetoed"


# --------------------------------------------------------------------------- #
#  1. ONE reader, not two                                                     #
# --------------------------------------------------------------------------- #
def test_the_basis_reader_has_exactly_one_implementation():
    """`pipeline_analysis` cannot be imported by `depth_horizon_grid` (the import runs the
    other way), which is exactly the pressure that produces a second copy of the parse.  The
    ejection-count false alarm was fixed once; a second copy is where it comes back."""
    import pipeline_analysis as pa
    per_anchor = {"a": {"basis": _VETOED % 1125}, "b": {"basis": _VETOED % 955}}
    assert pa._basis_of(per_anchor) == bs.of(per_anchor)
    assert pa._basis_kind(_VETOED % 7) == bs.kind(_VETOED % 7)
    #  SCANNED ON AN AST, NOT ON RAW TEXT.  A substring scan for "ejected)" read the prose
    #  in these modules -- they legitimately DISCUSS the stamp format -- so the guard fired
    #  on comments and would have been weakened rather than heeded.  What actually must not
    #  reappear is a second REGEX over the stamp, which is a structural question.
    import ast
    for mod in (dhg, sc):
        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fname = ast.unparse(node.func)
                assert not fname.startswith("re."), (
                    f"{mod.__name__} calls {fname} -- if that parses the basis stamp it is a "
                    "second implementation; the reader lives in basis_stamp")


def test_an_unrecognised_stamp_is_never_defaulted_to_a_basis():
    """`tag` answers "may I put these two numbers side by side".  Guessing on an unfamiliar
    stamp is the original defect in miniature -- a number silently acquiring a basis."""
    assert bs.tag("VETOED (stage-1 solvency gate applied, 5 ejected)") == "VETOED"
    assert bs.tag("un-vetoed") == "un-vetoed"
    assert bs.tag("un-vetoed (veto DECLINED: panel missing uCurrentRatio)") == "DECLINED"
    assert bs.tag("MIXED -- anchors disagree: x | y") == "MIXED"
    assert bs.tag(None) == "UNSTAMPED"
    assert bs.tag(bs.UNSTAMPED) == "UNSTAMPED"
    assert bs.tag("something nobody has written yet") == "UNKNOWN"


# --------------------------------------------------------------------------- #
#  2. The GRID REPORT -- the file that outlives the run log                    #
# --------------------------------------------------------------------------- #
def _cells(scope, basis_irrelevant=None):
    out = []
    for h in (36,):
        for N in (1, 20):
            out.append({"scope": scope, "clean": scope in dhg.CLEAN_BUY_IDS,
                        "buy": "2021-12-31", "eval": "2024-12-31",
                        "horizon_m": h, "depth_N": N, "n_requested": N,
                        "n_available_rank": 100, "n_included": N,
                        "n_missing_buy": 0, "n_affected_eval": 0,
                        "avg_ret_primary": 0.10, "avg_ret_floor": 0.10,
                        "bench_ret": 0.05, "excess_primary": 0.05, "excess_floor": 0.05})
    return out


def _per_anchor(basis_by_wid):
    return {wid: {"buy": "2021-12-31", "ranking": [], "rank_depth": 100,
                  "top20_deduped": [], "universe_size": 1000, "n_pit_scored": 900,
                  "n_pit_live": 500, "n_pit_dead": 400, "basis": b}
            for wid, b in basis_by_wid.items()}


def test_the_grid_report_names_its_basis_in_the_header():
    """THE DEFECT: the 2026-08-28 grid report -- the first VETOED one ever produced -- was
    shaped identically to every un-vetoed report in the archive."""
    pa = _per_anchor({"buy2021": _VETOED % 1125, "buy2022": _VETOED % 955})
    text = dhg.build_report(pa, _cells("buy2021"), _cells("POOLED-ALL"),
                            _cells("POOLED-CLEAN"))
    assert "MEASUREMENT BASIS" in text
    assert "VETOED" in text
    assert "Do NOT compare a" in text, "the standing comparison warning is missing"
    #  the ejection counts survive as magnitudes, and do NOT fire a false MIXED
    assert "MIXED" not in text.split("UNIVERSE & DEGENERACY")[0]
    assert "1125" in text and "955" in text


def test_the_grid_report_flags_a_POOLED_block_that_averages_across_two_bases():
    """The pooled blocks average across anchors.  Two anchors on different bases make the
    pooled cell a blend of the two, which is a reading of neither."""
    pa = _per_anchor({"buy2021": _VETOED % 1125, "buy2022": _UNVETOED})
    text = dhg.build_report(pa, _cells("buy2021"), _cells("POOLED-ALL"),
                            _cells("POOLED-CLEAN"))
    assert "MIXED" in text
    assert "DIFFERENT" in text.upper()
    #  and the per-anchor column says WHICH rows mixed it
    rows = [l for l in text.splitlines() if l.strip().startswith("buy202")]
    assert any("VETOED" in r for r in rows) and any("un-vetoed" in r for r in rows)


def test_an_unstamped_grid_report_says_UNSTAMPED_rather_than_saying_nothing():
    """An old pickle path that never stamps must not read as 'un-vetoed by assertion'."""
    pa = _per_anchor({"buy2021": None})
    text = dhg.build_report(pa, _cells("buy2021"), _cells("POOLED-ALL"),
                            _cells("POOLED-CLEAN"))
    assert "basis not stamped" in text


def test_rank_all_anchors_still_carries_the_key_the_report_reads():
    """The report reads `per_anchor[wid]['basis']`.  If `rank_all_anchors` ever stops writing
    that key the report silently prints UNSTAMPED forever, which looks like a fact."""
    import inspect
    src = inspect.getsource(dhg.rank_all_anchors)
    assert '"basis"' in src


# --------------------------------------------------------------------------- #
#  3. scoring_compare -- ONE table, TWO bases, no label                        #
# --------------------------------------------------------------------------- #
def _write_cells(workdir, config, basis=None, sentinel=0.10):
    rows = []
    for scope in ("POOLED-CLEAN", "buy2021"):
        for N in (1, 20):
            rows.append({"config": config, "scope": scope, "horizon_m": 36, "depth_N": N,
                         "avg_ret_primary": sentinel, "excess_primary": 0.05,
                         "bench_ret": 0.05})
    df = pd.DataFrame(rows)
    if basis is not None:
        df.insert(1, "basis", basis)
    df.to_csv(os.path.join(workdir, f"cells_{config}.csv"), index=False)


def test_a_side_by_side_mixing_two_bases_says_so_loudly(tmp_path):
    """THE DEFECT: ORIGINAL is documented as running on a REVERTED, pre-veto tree while
    BASELINE/CARVE/EQUAL rank on HEAD with the solvency gate applied -- four columns, two
    bases, one table, no label anywhere.  `scoring_compare` is the tool reached for when
    retuning gates, which is the worst place for that ambiguity to live."""
    w = str(tmp_path)
    _write_cells(w, "ORIGINAL", basis=None)          # reverted tree: no column at all
    _write_cells(w, "BASELINE", basis=_VETOED % 1125)
    out = os.path.join(w, "cmp.out")
    sc.format_comparison(w, out, None, log=lambda *a: None)
    text = open(out, encoding="utf-8").read()
    assert "MEASUREMENT BASIS PER COLUMN" in text
    assert "NOT ALL ON THE SAME BASIS" in text
    assert "confounded" in text
    assert "MIXED: columns not comparable" in text, "the per-table line is missing"


def test_the_basis_line_is_repeated_on_EVERY_table_not_only_the_header(tmp_path):
    """These blocks get pasted out of the .out one at a time.  A stamp 200 lines up does not
    travel with the table someone copies into a report -- which is how a labelled file still
    produces an unlabelled number."""
    w = str(tmp_path)
    _write_cells(w, "BASELINE", basis=_VETOED % 1125)
    _write_cells(w, "EQUAL", basis=_VETOED % 1125)
    out = os.path.join(w, "cmp.out")
    sc.format_comparison(w, out, None, log=lambda *a: None)
    text = open(out, encoding="utf-8").read()
    blocks = [t for t, _ in sc.SCOPE_ORDER if f"ANCHOR {t}" in text or "POOLED" in t]
    assert text.count("  BASIS: ") >= 2, "not every scope block carries the basis line"
    #  same basis on both columns -> no false alarm
    assert "MIXED: columns not comparable" not in text
    assert "NOT ALL ON THE SAME BASIS" not in text


def test_the_stamp_travels_in_the_cells_file_not_only_in_a_sibling(tmp_path):
    """`format` is a separate invocation over whatever cells_*.csv are in the workdir.  A
    basis living only in `diag_<CFG>.csv` can be stale relative to the numbers, or absent."""
    import inspect
    src = inspect.getsource(sc.run_configs)
    assert '"basis"' in src and "df.insert" in src


def test_diag_is_the_documented_FALLBACK_for_files_written_before_the_column(tmp_path):
    """Cells files already on disk predate the column; they must still get labelled rather
    than all reading UNSTAMPED and drowning the real unstamped case in noise."""
    w = str(tmp_path)
    _write_cells(w, "BASELINE", basis=None)
    pd.DataFrame([{"config": "BASELINE", "anchor": "buy2021", "basis": _VETOED % 1125,
                   "n_pit_scored": 900}]).to_csv(os.path.join(w, "diag_BASELINE.csv"),
                                                 index=False)
    assert bs.tag(sc._basis_for_config(
        pd.read_csv(os.path.join(w, "cells_BASELINE.csv")), w, "BASELINE")) == "VETOED"


def test_an_unstamped_ORIGINAL_is_read_as_expected_but_an_unstamped_BASELINE_is_not(tmp_path):
    """ORIGINAL is un-vetoed BY CONSTRUCTION, so no stamp there is expected.  The same
    absence on BASELINE means something unknown produced the file, and the report must not
    invite the reader to treat the two the same way."""
    w = str(tmp_path)
    _write_cells(w, "ORIGINAL", basis=None)
    _write_cells(w, "BASELINE", basis=None)
    out = os.path.join(w, "cmp.out")
    sc.format_comparison(w, out, None, log=lambda *a: None)
    text = open(out, encoding="utf-8").read()
    assert "NOTE on ORIGINAL" in text
    assert "is not safe" in text


def test_a_MIXED_pool_CONTAINING_an_unstamped_anchor_still_tags_MIXED():
    """THE ORDERING BUG, ONE CASE OVER FROM THE ONE ALREADY FIXED, and the earlier test
    missed it because it used a SYNTHETIC MIXED string with no unstamped anchor in it.

    `of()` builds MIXED by joining the disagreeing kinds, so a pool with one unstamped anchor
    yields a string CONTAINING "basis not stamped".  Checking that substring before the MIXED
    prefix collapsed a DISAGREEMENT into UNSTAMPED -- the one thing this tag must never
    swallow."""
    real_mixed = bs.of({"buy2021": {"basis": _VETOED % 1125}, "buy2022": {"basis": None}})
    assert real_mixed.startswith("MIXED")
    assert "basis not stamped" in real_mixed, "fixture no longer reproduces the case"
    assert bs.tag(real_mixed) == "MIXED"


def test_the_MIXED_suppression_is_gone_end_to_end_in_scoring_compare(tmp_path):
    """REPRODUCED WHERE IT MATTERED, not just on the helper.  With the tags collapsed, the
    per-column basis set had ONE value, so the 'NOT ALL ON THE SAME BASIS ... confounded'
    warning never printed and a VETOED column was labelled [UNSTAMPED] beside its own VETOED
    string -- a table silently declaring itself comparable when it is not."""
    w = str(tmp_path)
    mixed = bs.of({"buy2021": {"basis": _VETOED % 1125}, "buy2022": {"basis": None}})
    _write_cells(w, "BASELINE", basis=mixed)
    _write_cells(w, "EQUAL", basis=_VETOED % 1125)
    out = os.path.join(w, "cmp.out")
    sc.format_comparison(w, out, None, log=lambda *a: None)
    text = open(out, encoding="utf-8").read()
    assert "NOT ALL ON THE SAME BASIS" in text, "the mixed/vetoed disagreement was suppressed"
    assert "confounded" in text
    assert "[MIXED" in text, "the mixed column is not labelled MIXED"
    assert "BASELINE  [UNSTAMPED" not in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))


# --------------------------------------------------------------------------- #
#  WHAT WAS ACTUALLY DEMONSTRATED, versus what was shipped                     #
# --------------------------------------------------------------------------- #
def _suite_closure():
    """Every `baseline_tools` module reachable from `pipeline_analysis`'s imports.

    Function-local imports count -- every suite stage imports its module inside the stage
    closure -- and `ast` rather than `importlib`, so nothing here executes module-level code.
    Same derivation as `test_price_grid_refetch_decision._suite_modules`; duplicated rather
    than imported because a test file importing another test file for a helper is a worse
    coupling than nine lines, and both are pinned to the same source of truth (the import
    graph) rather than to each other."""
    import ast
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    local = {f[:-3] for f in os.listdir(here) if f.endswith(".py")}

    def _imports(mod):
        with open(os.path.join(here, mod + ".py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        out = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                out |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                out.add(node.module.split(".")[0])
        return out & local

    seen, stack = set(), ["pipeline_analysis"]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        stack.extend(_imports(m))
    return seen


def test_scoring_compare_is_still_a_hand_run_tool():
    """HALF THE 2026-08-27 STAMPING BATCH HAS NEVER EXECUTED, and that is a fact about what
    was demonstrated rather than a defect.

    `scoring_compare` has NO production call site -- the nightly suite does not import it, so
    its basis stamps are exercised by these tests and by a person running the script, and by
    nothing else.  That was not visible from the commit, and the 08-31 run review had to
    establish it by grep.  It is written down in the module docstring now; this test is what
    keeps the note true.

    THE ASSERTION RUNS BOTH WAYS.  If somebody wires the tool into the suite, this fails --
    which is correct: the docstring would then be wrong, and 'its stamps have never run
    against real data' would stop being true and would need deleting rather than aging."""
    closure = _suite_closure()
    assert "scoring_compare" not in closure, (
        "scoring_compare is now reachable from pipeline_analysis.  That may well be right -- "
        "but its docstring says it is a hand-run tool whose stamps have never executed "
        "against real data, and that paragraph is now false.  Update it, then update this "
        "test.")
    #  and the note must actually be there to be kept true
    import inspect
    doc = inspect.getdoc(sc) or ""
    assert "NOT A PIPELINE STAGE" in doc
    assert "never touches" in doc or "does not import this module" in doc


def test_the_stamped_reports_the_suite_DOES_produce_are_named_correctly():
    """The other half: `depth_horizon_grid` and `skill_baseline` DO run every night, so their
    stamps are the ones a reader is actually holding.  `skill_baseline` was unstamped until
    2026-08-31 -- it printed an UN-VETOED filter beat-rate beside a VETOED one -- so its stamp
    is pinned here rather than assumed."""
    closure = _suite_closure()
    assert {"depth_horizon_grid", "skill_baseline"} <= closure
    import skill_baseline as sb
    src = inspect_source(sb.format_report)
    assert "banner_lines" in src, "the skill-baseline report is unstamped again"


def inspect_source(fn):
    import inspect
    return inspect.getsource(fn)
