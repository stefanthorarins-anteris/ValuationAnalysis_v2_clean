"""THE REFETCH DECISION CONSULTS FRESHNESS -- and still never spends on its own.

THE DEFECT (Q-7).  `run_price_fetch_stage` decided with `not os.path.exists(_PRICES_CSV)`, a
pure presence check, so a grid written months ago satisfied it for ever.  The freshness signal
existed two stages away in `_audit_price_grid_stage` and had no route to the decision it was
about: the audit could report STALE all night, and the refetch decision had already been taken
without asking.

THE FIX IS A REFUSAL, NOT A FETCH.  Stale + present now produces a loud, named refusal to
REFETCH plus the key that authorises one.  It is deliberately not a refusal to RUN -- the
2026-08-22 ruling stands that withholding five working analysis stages makes the spend decision
for the CEO by starving him of what he would decide from.

WHAT THESE TESTS CANNOT SEE:
  * They pin that the DECISION consults the audit and that no path fetches without the key.
    They cannot check that the audit's STALE verdict is CORRECT -- `price_grid_audit`'s own
    tests own that, and the reviewer's C2 finding is the standing proof it can be wrong (it
    once reported venues as zero-priced at an anchor the pipeline prices fine).  A wrong STALE
    now produces a loud refusal on a non-defect; a wrong OK produces the old silence.
  * They do not exercise a real fetch.  Nothing here can prove the authorised branch actually
    fetches correctly, only that it is the sole branch that sets `need_main`.
"""
import inspect
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_analysis as pa


def _strip_docstrings(fn):
    """Source of `fn` with every docstring removed.

    TOKEN SCANS MUST NOT READ PROSE.  An earlier guard here banned the substring "days"
    anywhere in a function INCLUDING its docstring, so editing a comment could fail a test
    about behaviour.  Anything below that scans for a token scans the stripped body.
    """
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Module)) and ast.get_docstring(node) is not None:
            node.body = node.body[1:]
    return ast.unparse(tree)


def _fetch_src():
    return inspect.getsource(pa.run_price_fetch_stage)


def test_the_decision_now_reads_the_audit_rather_than_only_the_filesystem():
    src = _fetch_src()
    assert "_grid_stale_findings" in src, "the refetch decision still never asks about freshness"
    assert "os.path.exists(_PRICES_CSV)" in src, "presence must still decide the ABSENT case"


def test_the_freshness_read_borrows_the_audits_definition_rather_than_copying_it():
    """Two staleness rules is how the presence check and the audit became two unrelated
    opinions in the first place.

    SCANNED ON THE STRIPPED BODY, not the raw source: the first version banned the substring
    "days" anywhere in the function INCLUDING its docstring, so editing prose failed a test
    about behaviour.  A guard that fires on its own explanation trains people to weaken it."""
    src = _strip_docstrings(pa._grid_stale_findings)
    assert "price_grid_audit" in src and "audit_price_grid" in src
    for reinvention in ("getmtime", "days", "datetime.today"):
        assert reinvention not in src, f"looks like a second staleness rule ({reinvention})"


def test_a_stale_grid_does_NOT_fetch_without_the_explicit_key():
    """The one thing that must never regress: staleness is not authorisation to spend."""
    src = _fetch_src()
    #  `need_main = True` may appear exactly once, and only under the override key.
    assigns = [l for l in src.splitlines() if "need_main = True" in l]
    assert len(assigns) == 1, f"more than one path sets need_main True: {assigns}"
    idx = src.index("need_main = True")
    guard = src[max(0, idx - 400):idx]
    assert "price_grid_refetch_when_stale" in guard, (
        "need_main is set True somewhere not guarded by the override key")


def test_the_guard_fails_towards_spending_NOTHING():
    """An audit that cannot run must never be the thing that authorises a paid call."""
    src = inspect.getsource(pa._grid_stale_findings)
    assert "except Exception" in src
    assert src.rstrip().endswith("return None"), (
        "the failure path must return None (= no findings = no refetch)")

    said = []
    #  a resdic with nothing usable in it -- the read cannot succeed
    out = pa._grid_stale_findings({}, said.append)
    assert out is None or isinstance(out, list)


def test_the_refusal_names_the_key_that_resolves_it():
    """A warning that does not say what to do about it is a warning nobody acts on -- this is
    the difference between the refusal and the report that already existed."""
    src = _fetch_src()
    assert "price_grid_refetch_when_stale" in src
    assert "WILL NOT REFETCH" in src
    assert "SPENDS PAID API CALLS" in src


def test_no_age_threshold_silently_trips_a_fetch():
    """The failure mode this design exists to avoid: a rule that looks conservative and
    quietly spends the CEO's money on a schedule nobody approved."""
    src = (_strip_docstrings(pa.run_price_fetch_stage)
           + _strip_docstrings(pa._grid_stale_findings))
    for automatic in ("timedelta(", "> 30", "max_age", "auto_refetch"):
        assert automatic not in src, f"an automatic refetch trigger appeared: {automatic}"


# --------------------------------------------------------------------------- #
#  A MISSING BENCHMARK ANCHOR MUST COST ONE WINDOW, NOT THE WHOLE STAGE        #
# --------------------------------------------------------------------------- #
def test_one_unpriceable_benchmark_anchor_does_not_destroy_every_other_window():
    """FOUND WHILE PROMOTING buy2020 (Q-28), and latent independently of it.
    `benchmark_return(..., require_exact=True)` raises by design; every caller put that bare
    call in a PER-ANCHOR LOOP; every one of those loops runs under a stage guard that
    swallows exceptions.  So ONE missing URTH anchor returned an EMPTY two-clause target,
    beat-rate table, per-band split and gate attribution -- on runs where the other windows
    were perfectly measurable.

    Exercised on the REAL helper against a price source that cannot serve the benchmark, not
    on a mock of it, so the strictness itself is covered rather than assumed."""
    import returns_core as rc

    class _NoBenchmark:
        def benchmark_series(self, symbol):
            raise KeyError("URTH absent from this grid")

    said = []
    out = rc.benchmark_return_or_none(_NoBenchmark(), "2020-12-31", "2023-12-29",
                                      "buy2020", "a stage", log=said.append)
    assert out is None, "a missing benchmark must never be substituted with a number"
    text = chr(10).join(said)
    assert "SKIPPED" in text and "buy2020" in text
    assert "not a zero" in text, "the skip must say what it is NOT, or it reads as a result"


def test_the_helper_still_requires_an_EXACT_benchmark_anchor():
    """The strictness was never the problem, the blast radius was.  A forward-filled stale
    benchmark level is the thing `require_exact=True` exists to refuse, and wrapping the call
    must not quietly relax it."""
    import returns_core as rc
    seen = {}

    class _Spy:
        def benchmark_series(self, symbol):
            return "series"

    real = rc.bl.window_return

    def _spy(series, buy, ev, require_exact=False):
        seen["require_exact"] = require_exact
        return 0.10

    rc.bl.window_return = _spy
    try:
        assert rc.benchmark_return_or_none(_Spy(), "2020-12-31", "2023-12-29") == 0.10
    finally:
        rc.bl.window_return = real
    assert seen["require_exact"] is True


#  ----------------------------------------------------------------------------------- #
#  THE SUITE, DERIVED -- not a list somebody has to remember to extend                  #
#  ----------------------------------------------------------------------------------- #
#  The first version of the sweep below named THREE modules and its own docstring conceded
#  "it covers the modules named here".  `depth_horizon_grid` was the fourth suite module: it
#  ran the same per-anchor loop under the same swallowing stage guard, it produced the
#  LARGEST report in the suite -- and it kept a bare
#  `rc.benchmark_return(price_source, buy, ev)` with `require_exact` defaulting False, i.e. a
#  missing anchor silently FORWARD-FILLED a stale benchmark level into every `excess` column.
#  A guard that cannot see the defect beneath it is this project's recurring failure, and a
#  hand-maintained module list is how it got built that way.  So the list is DERIVED from
#  `pipeline_analysis`'s own transitive import graph: a stage that runs in the suite must be
#  imported by it (directly, or through another suite module), so a NEW stage is covered the
#  moment it is wired in -- by construction, not by anybody remembering this file exists.


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _suite_modules():
    """Every `baseline_tools` module reachable from `pipeline_analysis`'s imports.

    FUNCTION-LOCAL IMPORTS COUNT, and that is essential rather than incidental: every stage
    in `run_analysis_suite` imports its module INSIDE the stage closure
    (`import depth_horizon_grid as dhg`), so a module-level-only scan would see almost none
    of the suite.  `ast` rather than `importlib`, deliberately -- resolving these for real
    would execute module-level code, and `tuner` / `tune_run` are heavy."""
    import ast
    local = {f[:-3] for f in os.listdir(_HERE) if f.endswith(".py")}

    def _imports(mod):
        tree = ast.parse(_read(os.path.join(_HERE, mod + ".py")))
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
    return sorted(seen)


def _benchmark_return_calls(mod_name):
    """[(lineno, passes_require_exact_True)] for every `benchmark_return(...)` CALL.

    Matches `rc.benchmark_return(...)` and a bare `benchmark_return(...)`, and deliberately
    NOT `benchmark_return_or_none(...)` -- that one is the guarded helper, which is the whole
    point.  AST rather than substring, so a mention in a comment or a docstring cannot fail a
    test about behaviour (the lesson `_strip_docstrings` above already carries)."""
    import ast
    tree = ast.parse(_read(os.path.join(_HERE, mod_name + ".py")))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = (f.attr if isinstance(f, ast.Attribute)
                else f.id if isinstance(f, ast.Name) else None)
        if name != "benchmark_return":
            continue
        exact = any(kw.arg == "require_exact"
                    and isinstance(kw.value, ast.Constant) and kw.value.value is True
                    for kw in node.keywords)
        out.append((node.lineno, exact))
    return out


#  Suite modules that call `benchmark_return` STRICTLY AND FATALLY on purpose, each with the
#  reason.  Checked for EQUALITY, not containment: a new module taking the strict-fatal shape
#  fails until someone writes down why, and an entry that has stopped being true fails too,
#  so the registry cannot rot into a list of things that used to be exceptions.
STRICT_AND_FATAL_BY_DESIGN = {
    "returns_core":
        "owns both `benchmark_return` and the `_or_none` wrapper; the wrapper's own call is "
        "the strict one every other caller borrows.",
    "tune_run":
        "search OBJECTIVES (`surrogate_score`, `true_beat_rate`).  Silently dropping a "
        "window there would change the objective the tuner optimises rather than cost a "
        "report row, so a raise is the correct outcome -- and `tune_run` runs only under "
        "run_estimation, never in the nightly suite's guarded stages.",
    "rebalance_engine":
        "one benchmark leg for a whole rebalance schedule, computed BEFORE the per-period "
        "loop and not inside one.  With no leg there is no run to report, so there is no "
        "window to skip.",
}


def test_no_suite_module_can_FORWARD_FILL_a_missing_benchmark_anchor():
    """The `depth_horizon_grid.py:442` defect, stated as a property of the whole suite.

    `benchmark_return` defaults `require_exact=False`, and that default is the dangerous one:
    a missing anchor is answered with the last known benchmark LEVEL carried forward, so
    `excess_primary` / `excess_floor` measure against a stale index and nothing says so.  No
    suite module may take that default -- there is no reading of a forward-filled benchmark
    that anybody wants."""
    offenders = []
    for mod in _suite_modules():
        for lineno, exact in _benchmark_return_calls(mod):
            if not exact:
                offenders.append("%s.py:%d" % (mod, lineno))
    assert not offenders, (
        "these call benchmark_return with require_exact defaulting FALSE, i.e. a missing "
        "anchor is silently forward-filled: " + ", ".join(offenders)
        + ".  Route them through returns_core.benchmark_return_or_none.")


def test_every_STRICT_benchmark_call_in_the_suite_is_one_we_wrote_down():
    """The blast-radius half, which `require_exact=True` alone does not fix.

    A strict call inside a per-anchor loop RAISES; every such loop runs under
    `pipeline_analysis._run_stage`; that guard swallows everything -- so one unpriceable
    anchor erased a whole stage and took the measurable windows with it.  The remedy is
    `benchmark_return_or_none`; the exceptions are the three modules registered above, where
    a raise is the right outcome.  EQUALITY, so a fourth cannot appear unexplained and a
    stale entry cannot survive."""
    strict = set(mod for mod in _suite_modules()
                 if any(exact for _l, exact in _benchmark_return_calls(mod)))
    registered = set(STRICT_AND_FATAL_BY_DESIGN)
    assert strict == registered, (
        "unregistered strict-and-fatal callers: %s; registered but no longer strict-and-"
        "fatal: %s.  Either route the call through benchmark_return_or_none, or add a "
        "one-line reason to STRICT_AND_FATAL_BY_DESIGN."
        % (sorted(strict - registered), sorted(registered - strict)))


def test_the_sweep_actually_reaches_the_module_it_was_blind_to():
    """A guard that cannot see its own target is the failure this sweep was widened to fix,
    so the widening is checked rather than assumed.  `depth_horizon_grid` is the module that
    sat outside the old three-name list; the other three are the ones that sat inside it."""
    got = set(_suite_modules())
    for must in ("pipeline_analysis", "gate_attribution", "skill_baseline",
                 "depth_horizon_grid", "returns_core", "target_clauses"):
        assert must in got, "the derived suite list lost %s: %s" % (must, sorted(got))
    #  and it must genuinely be READING dhg's calls, not merely listing the module
    assert _benchmark_return_calls("depth_horizon_grid") == [], (
        "depth_horizon_grid still calls benchmark_return directly; it must go through "
        "benchmark_return_or_none")


def test_the_guard_lives_in_ONE_place():
    """Four call sites fixed by copying the fix four times is four things that drift."""
    import inspect
    import pipeline_analysis as pa
    import returns_core as rc
    assert "benchmark_return_or_none" in _strip_docstrings(pa._bench_or_skip)
    assert "require_exact=True" in _strip_docstrings(rc.benchmark_return_or_none)


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
