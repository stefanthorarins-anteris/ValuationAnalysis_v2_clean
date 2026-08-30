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


def test_EVERY_per_anchor_loop_in_the_suite_uses_the_guarded_helper():
    """THE FIX STOPPED ONE CALL SITE SHORT the first time.  `pipeline_analysis` was fixed
    while `gate_attribution` (2 sites) and `skill_baseline` (3) kept the bare call -- both
    run in this suite, and the buy2020 promotion widened both to three anchors.  Latent only
    because URTH happens to be priced at every anchor, which is luck, not a guard.

    WHAT THIS CANNOT SEE: it covers the modules named here.  A NEW module added to the suite
    with a bare call is not caught by anything."""
    import gate_attribution
    import pipeline_analysis
    import skill_baseline
    for mod in (pipeline_analysis, gate_attribution, skill_baseline):
        src = _strip_docstrings(mod)
        assert "benchmark_return(" not in src.replace("benchmark_return_or_none(", ""), (
            f"{mod.__name__} still calls benchmark_return directly inside a loop")


def test_the_guard_lives_in_ONE_place():
    """Four call sites fixed by copying the fix four times is four things that drift."""
    import inspect
    import pipeline_analysis as pa
    import returns_core as rc
    assert "benchmark_return_or_none" in _strip_docstrings(pa._bench_or_skip)
    assert "require_exact=True" in _strip_docstrings(rc.benchmark_return_or_none)


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
