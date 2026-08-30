"""
Price-grid staleness audit self-checks (offline; no network, no fetch, ever).

WHAT WENT WRONG.  `pipeline_analysis.run_price_fetch_stage` decided whether the grading price
grid needed fetching with `need_main = not os.path.exists(_PRICES_CSV)` -- a PURE PRESENCE
CHECK -- so a `real_prices.csv` written 2026-07-17 satisfied it for ever.  Measured against the
08-22 CUR6K panel that grid prices 4,095 of 5,819 names, and SEVEN venues (`.PA`, `.KS`, `.OL`,
`.KQ`, `.BR`, `.AS`, `.LS` = 1,421 names) have ZERO rows at EVERY anchor.  Eight of that run's
top-20 sit on venues it cannot price.  Nothing said so.

THE TEST THAT WOULD HAVE MISSED IT.  A scalar "overlap is above X%" assertion.  70.4% overlap
looks survivable and says nothing at all about a market being wholly absent, which is the
finding that actually invalidates a cross-sectional average.  So the venue tests below are the
load-bearing ones, and the overlap number is deliberately NOT a trigger anywhere.

THE OTHER THING THESE PIN: the audit must never cause a fetch.  `test_the_audit_stage_makes_no
_network_call_and_no_fetch_decision` asserts the fetch decision is untouched and that nothing
in the audit path can reach the HTTP helpers.
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

import price_grid_audit as pga
import returns_core as rc

#  THE FIXTURES USE THE GRADER'S OWN ANCHOR LIST, not a copy of it (2026-08-22, reviewer C2).
#  A local copy is how the audit came to measure over an anchor space the grader does not use:
#  it harvested 9 `date_requested` values from the file while the grader works over 8, because
#  `PriceSource`'s fill layer unions `2025-12-30` into the `2025-12-31` anchor.  If
#  DEFAULT_ANCHORS ever changes, these tests must move with it or they are testing nothing.
ANCHORS = list(rc.DEFAULT_ANCHORS)

#  `date_actual` is the venue settlement day.  Nothing under test reads it any more -- that is
#  the point of C1/C2 -- but the column is part of the on-disk schema, so the fixtures still
#  write it, and DELIBERATELY write a value that differs from the anchor so that any code which
#  starts reading it again fails these tests loudly instead of quietly agreeing.
def _actual_for(anchor):
    return str(pd.Timestamp(anchor) - pd.Timedelta(days=3))[:10]


def _grid(tmp_path, coverage, name='real_prices.csv', price=100.0):
    """`coverage` = {anchor: [symbols priced there]}.  Writes the real schema."""
    rows = []
    for a in ANCHORS:
        for sym in coverage.get(a, []):
            rows.append({'date_requested': a, 'date_actual': _actual_for(a),
                         'symbol': sym, 'adjClose': price})
    p = tmp_path / name
    pd.DataFrame(rows, columns=['date_requested', 'date_actual', 'symbol',
                                'adjClose']).to_csv(p, index=False)
    return str(p)


def _with_benchmark(coverage):
    """Add URTH at every anchor, so a test about VENUES is not also a test about the benchmark.

    The benchmark is a separate finding (it is never in the panel), so a fixture that omits it
    would make every venue test STALE for the wrong reason."""
    return {a: list(coverage.get(a, [])) + [rc.BENCHMARK_SYMBOL] for a in ANCHORS}


def _names(prefix, n, suffix=''):
    return ['%s%03d%s' % (prefix, i, suffix) for i in range(n)]


# --------------------------------------------------------------------------- #
#  the finding a coverage percentage cannot make                              #
# --------------------------------------------------------------------------- #
def test_a_wholly_absent_venue_is_caught_even_at_high_overall_overlap():
    """THE CASE.  A grid can price the large majority of names and still be blind to a market.

    Constructed so the scalar overlap is HIGH (90%) while one venue is completely unpriceable.
    Any audit that keyed off overlap alone would call this healthy -- which is exactly what the
    presence check did, silently, on every run."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        us = _names('US', 900)
        kq = _names('K', 100, '.KQ')
        panel = us + kq
        grid = _grid(tmp, _with_benchmark({a: us for a in ANCHORS}))

        rep = pga.audit_price_grid(grid, panel)

    assert rep['overlap_frac'] == pytest.approx(0.9), rep['overlap_frac']
    assert rep['verdict'] == 'STALE', rep
    assert rep['venues']['.KQ']['absent_everywhere'] is True
    assert rep['venues']['.KQ']['n_priced_any_anchor'] == 0
    assert rep['venues'][pga.NO_SUFFIX]['absent_everywhere'] is False
    assert any('.KQ is WHOLLY ABSENT' in x for x in rep['findings']), rep['findings']
    #  the count is named, so the reader can size the hole
    assert any('100 panel name(s)' in x for x in rep['findings']), rep['findings']


def test_a_venue_missing_at_ONE_anchor_is_reported_as_that_not_as_wholly_absent():
    """The real `.DE` / `.ST` / `.IC` case: present from 2019 on, empty at 2018-12-31.

    Distinguished from wholly-absent because the remedy differs -- one anchor needs a re-pull,
    a missing venue needs the fetch universe widened."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        de = _names('D', 40, '.DE')
        us = _names('US', 100)
        cov = {a: (us + de) for a in ANCHORS}
        cov['2018-12-31'] = us                     # the hole
        rep = pga.audit_price_grid(_grid(tmp, _with_benchmark(cov)), us + de)

    v = rep['venues']['.DE']
    assert v['absent_everywhere'] is False
    assert v['zero_anchors'] == ['2018-12-31'], v['zero_anchors']
    assert v['n_priced_any_anchor'] == 40
    assert rep['verdict'] == 'STALE'
    assert any('.DE has 40 panel name(s) but ZERO priced at anchor(s) 2018-12-31' in x
               for x in rep['findings']), rep['findings']
    assert not any('WHOLLY ABSENT' in x for x in rep['findings']), rep['findings']


def test_a_thinly_represented_venue_does_not_by_itself_declare_the_grid_stale():
    """Zero-of-two is not evidence about a fetch; zero-of-many is.  The floor keeps the verdict
    from firing on noise -- but the venue is still REPORTED, so nothing is hidden."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        us = _names('US', 200)
        tiny = _names('T', 2, '.XX')
        rep = pga.audit_price_grid(_grid(tmp, _with_benchmark({a: us for a in ANCHORS})),
                                   us + tiny)

    assert rep['venues']['.XX']['absent_everywhere'] is True     # reported...
    assert rep['verdict'] == 'OK', rep['findings']               # ...but not a trigger
    assert pga.VENUE_MIN_PANEL_NAMES > 2


def test_a_fully_covering_grid_is_OK_and_the_verdict_is_not_hardwired():
    """The negative control for the whole module: if this ever says STALE, the audit is
    crying wolf and every other assertion here is worthless."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        panel = _names('US', 300) + _names('L', 50, '.L') + _names('K', 20, '.KQ')
        rep = pga.audit_price_grid(_grid(tmp, _with_benchmark({a: panel for a in ANCHORS})),
                                   panel)

    assert rep['verdict'] == 'OK', rep['findings']
    assert rep['findings'] == []
    assert rep['overlap_frac'] == 1.0
    assert all(rep['per_anchor'][a] == len(panel) for a in ANCHORS)


def test_an_anchor_that_prices_nothing_is_a_finding():
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        panel = _names('US', 50)
        cov = {a: panel for a in ANCHORS}
        cov['2020-12-31'] = ['OTHER_UNIVERSE_ONLY']
        rep = pga.audit_price_grid(_grid(tmp, _with_benchmark(cov)), panel)
    assert rep['per_anchor']['2020-12-31'] == 0
    assert any("anchor 2020-12-31 prices NONE" in x for x in rep['findings']), rep['findings']


def test_rows_with_no_usable_price_are_not_counted_as_coverage():
    """A null/zero adjClose is a row, not a price.  Counting it would reproduce the presence
    check one level down: a grid full of blanks reporting full coverage."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        panel = _names('US', 20)
        rows = []
        for a in ANCHORS:
            rows.append({'date_requested': a, 'date_actual': _actual_for(a),
                         'symbol': rc.BENCHMARK_SYMBOL, 'adjClose': 100.0})
            for i, sym in enumerate(panel):
                rows.append({'date_requested': a, 'date_actual': _actual_for(a),
                             'symbol': sym,
                             'adjClose': '' if i < 15 else 100.0})
        p = tmp / 'real_prices.csv'
        pd.DataFrame(rows).to_csv(p, index=False)
        rep = pga.audit_price_grid(str(p), panel)
    assert rep['n_overlap'] == 5, rep['n_overlap']
    assert rep['overlap_frac'] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
#  C2: it measures THROUGH the grader, so it cannot disagree with it           #
# --------------------------------------------------------------------------- #
def test_the_2025_holiday_union_is_NOT_reported_as_a_missing_venue():
    """THE FALSE FLAG.  `PriceSource`'s per-anchor fill layer unions `2025-12-30` into the
    `2025-12-31` anchor precisely for venues that do not trade on the 31st, so those names ARE
    priced at 2025-12-31 as far as every grading stage is concerned.

    The file-based audit saw `2025-12-30` as an anchor of its own and reported
    `venue .DE has N panel name(s) but ZERO priced at anchor(s) 2025-12-31` about names the
    pipeline prices perfectly well.  With `price_grid_refuse_when_stale` enabled that false
    STALE is a PERMANENT hard refusal on a non-defect, which is why it is a test and not a
    footnote."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        us = _names('US', 200)
        de = _names('D', 40, '.DE')
        panel = us + de

        #  main grid: everyone at every anchor EXCEPT the 2025 one (which lives in the supp)
        cov = {a: (panel + [rc.BENCHMARK_SYMBOL]) for a in ANCHORS if a != '2025-12-31'}
        main = _grid(tmp, cov)

        #  supp: US settles on the 31st, .DE only on the 30th -- the real-world shape
        srows = [{'date_requested': '2025-12-31', 'date_actual': '2025-12-31',
                  'symbol': s, 'adjClose': 100.0} for s in us + [rc.BENCHMARK_SYMBOL]]
        srows += [{'date_requested': '2025-12-30', 'date_actual': '2025-12-30',
                   'symbol': s, 'adjClose': 100.0} for s in de]
        supp = tmp / 'real_prices_2025.csv'
        pd.DataFrame(srows).to_csv(supp, index=False)

        rep = pga.audit_price_grid(main, panel, supp_csv=str(supp))

    #  the union happened: .DE is priced at the 2025-12-31 ANCHOR
    assert rep['venues']['.DE']['per_anchor']['2025-12-31'] == 40, \
        rep['venues']['.DE']['per_anchor']
    assert rep['venues']['.DE']['zero_anchors'] == [], rep['venues']['.DE']['zero_anchors']
    #  ...and 2025-12-30 is not an anchor of the audit's own invention
    assert '2025-12-30' not in rep['anchors'], rep['anchors']
    assert rep['verdict'] == 'OK', rep['findings']


def test_the_audit_and_the_GRADER_agree_name_by_name():
    """The property that replaces every hand-rolled coverage rule.

    For each panel name the audit calls priced-at-an-anchor, `returns_core.compute_returns`
    must be able to open a position there (status != 'no_buy'), and vice versa.  Asserted as a
    set equality over a grid built to be awkward: blanks, a zero, and a venue present at only
    some anchors."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        panel = _names('US', 12) + _names('K', 6, '.KQ')
        rows = []
        for i, a in enumerate(ANCHORS):
            for j, sym in enumerate(panel):
                if (i + j) % 4 == 0:
                    continue                       # simply absent
                px = {1: '', 2: 0.0}.get((i + j) % 5, 100.0 + j)
                rows.append({'date_requested': a, 'date_actual': _actual_for(a),
                             'symbol': sym, 'adjClose': px})
            rows.append({'date_requested': a, 'date_actual': _actual_for(a),
                         'symbol': rc.BENCHMARK_SYMBOL, 'adjClose': 100.0})
        gp = tmp / 'real_prices.csv'
        pd.DataFrame(rows).to_csv(gp, index=False)

        rep = pga.audit_price_grid(str(gp), panel)
        ps = rc.PriceSource(str(gp), anchors=list(ANCHORS))

        for a in ANCHORS:
            grader = {t for t in panel
                      if rc.compute_returns([t], a, a, ps).iloc[0]['status'] != 'no_buy'}
            audit = {t for t in panel if pga._is_priced(ps.price(t, a))}
            assert audit == grader, (a, sorted(audit ^ grader))
            assert rep['per_anchor'][a] == len(grader), (a, rep['per_anchor'][a], len(grader))


def test_a_NEGATIVE_price_counts_as_priced_because_the_grader_would_use_it():
    """Deliberate mirroring, not an oversight.  A negative adjClose is nonsense, but
    `compute_returns` only rejects None/0/NaN, so it WOULD build a return off one.  An audit
    that filtered negatives would under-report coverage relative to the printed numbers -- it
    must describe the grader that exists, not the one it wishes for."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        panel = ['NEG1', 'NEG2', 'OK1']
        rows = []
        for a in ANCHORS:
            for sym, px in (('NEG1', -5.0), ('NEG2', -0.01), ('OK1', 10.0),
                            (rc.BENCHMARK_SYMBOL, 100.0)):
                rows.append({'date_requested': a, 'date_actual': _actual_for(a),
                             'symbol': sym, 'adjClose': px})
        gp = tmp / 'real_prices.csv'
        pd.DataFrame(rows).to_csv(gp, index=False)
        rep = pga.audit_price_grid(str(gp), panel)
        ps = rc.PriceSource(str(gp), anchors=list(ANCHORS))
    assert rep['overlap_frac'] == 1.0, rep['per_anchor']
    #  and the grader really does open a position on the negative one
    assert rc.compute_returns(['NEG1'], ANCHORS[0], ANCHORS[1], ps).iloc[0]['status'] == 'ok'


def test_the_audit_never_builds_its_own_anchor_list():
    """AST guard.  A local anchor literal is how the disagreement got in; the module must take
    its axis from returns_core and nowhere else."""
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(pga.__file__).read_text(encoding='utf-8'))
    dates = [c.value for c in ast.walk(tree)
             if isinstance(c, ast.Constant) and isinstance(c.value, str)
             and len(c.value) == 10 and c.value[4] == '-' and c.value[7] == '-'
             and c.value[:4].isdigit()]
    assert not dates, 'price_grid_audit hardcodes anchor date(s): %s' % dates
    #  and it must USE the grader's anchors and price source in CODE, not merely mention them
    #  in prose -- a text scan passes on a docstring, which is how this test was too weak to
    #  kill the file-based version it was written against.
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert 'DEFAULT_ANCHORS' in attrs, attrs
    assert 'PriceSource' in attrs, attrs
    assert 'price' in attrs, 'coverage is not being read through PriceSource.price()'


# --------------------------------------------------------------------------- #
#  P1: the benchmark is its own coverage question                             #
# --------------------------------------------------------------------------- #
def test_a_wholly_absent_benchmark_is_a_FINDING_though_URTH_is_never_in_the_panel():
    """The one way the beat-rate stage can still fail after the carve unblocks it.

    `PriceSource.benchmark_series` RAISES when URTH is absent, and URTH is an ETF, so it is
    never in the scored panel -- the venue table is structurally blind to it.  This is the
    second cause the F3 cascade analysis could not rule out on the run machine."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        panel = _names('US', 100)
        rep = pga.audit_price_grid(_grid(tmp, {a: panel for a in ANCHORS}), panel)

    assert rep['overlap_frac'] == 1.0                      # the panel itself is fine...
    assert rep['benchmark_missing_anchors'] == list(ANCHORS)
    assert rep['verdict'] == 'STALE'                       # ...and it is still STALE
    assert any('benchmark URTH is absent from EVERY anchor' in x for x in rep['findings']), \
        rep['findings']
    assert any('benchmark_series' in x for x in rep['findings']), rep['findings']


def test_a_benchmark_hole_at_ONE_anchor_is_reported_as_that_not_as_total_absence():
    """A partial hole breaks only the windows that span it, so it must not be described as the
    stage being unable to run at all -- the remedies differ."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        panel = _names('US', 100)
        cov = _with_benchmark({a: panel for a in ANCHORS})
        cov['2021-12-31'] = list(panel)                    # benchmark dropped here only
        rep = pga.audit_price_grid(_grid(tmp, cov), panel)

    assert rep['benchmark_missing_anchors'] == ['2021-12-31']
    assert rep['benchmark_per_anchor']['2018-12-31'] is True
    assert any('missing at anchor(s) 2021-12-31' in x for x in rep['findings']), rep['findings']
    assert not any('absent from EVERY anchor' in x for x in rep['findings']), rep['findings']
    #  and the operator-facing text shows the per-anchor row
    txt = pga.format_audit(rep)
    assert '2021-12-31=NO' in txt and '2018-12-31=y' in txt, txt


# --------------------------------------------------------------------------- #
#  vintage and overlap are REPORTED, never triggers                           #
# --------------------------------------------------------------------------- #
def test_age_and_overlap_are_reported_but_never_trigger_on_their_own():
    """No measured healthy reference exists for either, so a threshold would be invented.

    A grid that covers everything is OK no matter how old, and the report still states the
    age so a human can judge it."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        panel = _names('US', 40)
        p = _grid(tmp, _with_benchmark({a: panel for a in ANCHORS}))
        old = 60 * 60 * 24 * 400          # 400 days
        st = os.stat(p)
        os.utime(p, (st.st_atime - old, st.st_mtime - old))
        rep = pga.audit_price_grid(p, panel)

    assert rep['age_days'] >= 399, rep['age_days']
    assert rep['verdict'] == 'OK', rep['findings']
    txt = pga.format_audit(rep)
    assert 'day(s) ago' in txt and 'REPORTED ONLY' in txt
    assert '%%' not in txt, 'a literal %% leaked into the operator-facing text'


# --------------------------------------------------------------------------- #
#  emission: loud when stale, and switchable to a refusal                      #
# --------------------------------------------------------------------------- #
def test_a_stale_grid_banners_on_stdout_and_says_nothing_was_fetched(capsys):
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        us = _names('US', 200)
        kq = _names('K', 60, '.KQ')
        rep = pga.run_audit(_grid(tmp, _with_benchmark({a: us for a in ANCHORS})),
                            us + kq)
    out = capsys.readouterr().out
    assert rep['verdict'] == 'STALE'
    assert 'PRICE GRID IS STALE' in out, out[-1500:]
    assert 'NOTHING WAS FETCHED' in out, out[-1500:]
    assert 'WHOLLY ABSENT' in out, out[-1500:]


def test_refuse_when_stale_is_OPT_IN_and_off_by_default(capsys):
    """Warn-and-proceed is the default because the remedy costs money the house cannot
    authorise; the CEO can flip it.  Both branches asserted, so neither is assumed."""
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        us = _names('US', 200)
        kq = _names('K', 60, '.KQ')
        grid = _grid(tmp, _with_benchmark({a: us for a in ANCHORS}))

        rep = pga.run_audit(grid, us + kq)                       # default: proceeds
        assert rep['verdict'] == 'STALE'

        with pytest.raises(RuntimeError) as e:
            pga.run_audit(grid, us + kq, refuse_when_stale=True)
        assert 'STALE' in str(e.value)
    capsys.readouterr()


# --------------------------------------------------------------------------- #
#  wiring: the audit runs, and it cannot fetch                                #
# --------------------------------------------------------------------------- #
def test_the_audit_is_wired_as_its_own_pipeline_stage():
    """AST guard.  The module is worthless if the suite never calls it, and the audit has to be
    its OWN stage so it still runs on the (normal) run where the fetch stage does nothing."""
    import ast
    import pathlib
    src = (pathlib.Path(_HERE) / 'pipeline_analysis.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    #  the stage is passed BY NAME to _run_stage, so look for that argument shape.
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, 'id', None) == '_run_stage'
             and len(c.args) >= 2
             and getattr(c.args[1], 'id', None) == '_audit_price_grid_stage']
    assert calls, 'the price-grid audit is not dispatched through _run_stage'
    #  and it must be its own dispatch, not folded into the fetch stage's
    fetch = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, 'id', None) == '_run_stage'
             and len(c.args) >= 2
             and getattr(c.args[1], 'id', None) == 'run_price_fetch_stage']
    assert fetch, 'the fetch stage dispatch vanished'
    assert calls[0] is not fetch[0]


def test_staleness_alone_can_NEVER_spend_money():
    """THIS TEST PINNED THE DEFECT IT WAS COVERING, and it is rewritten rather than deleted.

    It used to be called `..._is_still_PRESENCE_only_...` and asserted that `need_main` and
    `need_supp` were BOTH pure `not os.path.exists(...)` expressions.  But presence-only was
    Q-7 -- the refetch decision could not see the freshness signal that existed two stages
    away, so a months-old grid satisfied it for ever.  The test encoded the SHAPE that
    happened to deliver the invariant, so it made the defect unfixable without a test failure,
    which is exactly the trap this repo has now hit eleven times.

    THE INVARIANT ITSELF IS UNCHANGED AND IS WHAT IS ASSERTED HERE: an auto-refetch on
    staleness would bill the CEO on a schedule nobody approved, so nothing may set
    `need_main` except (a) the file being absent, or (b) an EXPLICIT override key.  Staleness
    is an input to a REFUSAL, never to a fetch.

    WHAT THIS CANNOT SEE: it reads the AST of the decision, so it proves no *code path* in
    this function spends without the key.  It cannot prove the key is hard to set by accident,
    and it says nothing about whether the staleness verdict itself is right.
    """
    import ast
    import pathlib
    src = (pathlib.Path(_HERE) / 'pipeline_analysis.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == 'run_price_fetch_stage')

    assigns = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and getattr(n.targets[0], 'id', None) in ('need_main', 'need_supp'):
            assigns.append((n.targets[0].id, ast.unparse(n.value), n.lineno))
    names = {k for k, _v, _l in assigns}
    assert names == {'need_main', 'need_supp'}, assigns

    #  1. The BASELINE decision for each is still the existence check.  Absent -> fetch.
    for key in ('need_main', 'need_supp'):
        first = min((a for a in assigns if a[0] == key), key=lambda a: a[2])
        assert first[1].startswith('not os.path.exists('), first

    #  2. Every OTHER assignment must sit under the explicit override key.  This is the
    #     invariant the old shape-check was standing in for.
    for key, value, lineno in assigns:
        if value.startswith('not os.path.exists('):
            continue
        enclosing = [n for n in ast.walk(fn) if isinstance(n, ast.If)
                     and n.lineno <= lineno <= max(
                         getattr(c, 'lineno', n.lineno) for c in ast.walk(n))]
        guards = ' '.join(ast.unparse(n.test) for n in enclosing)
        assert 'price_grid_refetch_when_stale' in guards, (
            f'{key} is set at line {lineno} to {value!r} outside the override key -- '
            'staleness must never authorise a paid fetch on its own')

    #  3. No time-based trigger may creep in: an age rule looks conservative and quietly
    #     spends on a schedule, which is the exact thing this test exists to forbid.
    body = ast.unparse(fn)
    for automatic in ('getmtime', 'timedelta(', 'max_age', 'auto_refetch'):
        assert automatic not in body, f'a time-based refetch trigger appeared: {automatic}'

    #  4. THE REPORT AND THE SPEND DECISION STAY APART, in the form that still matters.  The
    #     fetch stage may now consult the audit's PURE verdict function, but it must not reach
    #     `run_audit`, which prints the banner, writes to stderr and can RAISE on
    #     `refuse_when_stale` -- a reporting entry point inside the spend decision is how the
    #     two would become one thing again.
    called = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert not any(c.endswith('run_audit') for c in called), (
        'the fetch stage calls the audit REPORTER; it may only ask the pure verdict function')


def test_the_audit_module_contains_no_network_surface():
    """Offline by construction, asserted on the AST rather than by string search.

    Checked on IMPORTS and NAMES, not on raw text: the module's prose names
    `baseline_tools/fetch_prices.py` as the human remedy, and a text scan would read that
    sentence as the very dependency it is telling the operator to invoke by hand."""
    import ast
    import pathlib
    tree = ast.parse((pathlib.Path(pga.__file__)).read_text(encoding='utf-8'))
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split('.')[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split('.')[0])
    for forbidden in ('requests', 'urllib', 'http', 'socket', 'fetch_prices',
                      'delisted_ingest'):
        assert forbidden not in imported, 'price_grid_audit imports %r' % forbidden
    #  no key handling of any kind, in code or in a literal
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not ({'api_key', 'apikey'} & names), names & {'api_key', 'apikey'}
    consts = {c.value for c in ast.walk(tree)
              if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    assert not any('apikey' in c.lower() for c in consts), 'an apikey literal is present'


# --------------------------------------------------------------------------- #
#  INTERIOR HOLES -- the defect a "wholly absent venue" check cannot see       #
# --------------------------------------------------------------------------- #
#  A wholly-unpriceable venue is comparatively harmless to the NUMBERS: those names get
#  status='no_buy' and every derived view in returns_core EXCLUDES them.  A name priced
#  BEFORE and AFTER an anchor it is missing at is the dangerous case -- compute_returns
#  fires the TERMINAL policy, the default beat-rate policy (missing='fail') scores it a
#  miss and total_return_floor puts it at -100%, while a later price proves the company was
#  alive.  Measured on the run machine's grid: 177 name(s) / 291 cell(s), 164 of them .L and
#  6 of them US -- so this is live on the CURRENT NA1-only default, not only on a widened
#  exchange scope.
def _tmp():
    import tempfile
    import pathlib
    d = tempfile.TemporaryDirectory()
    return d, pathlib.Path(d.name)


def test_an_INTERIOR_price_hole_is_found_and_named():
    """Priced at the first and last anchor, missing in the middle -> a hole."""
    d, tmp = _tmp()
    with d:
        cov = {a: ['HOLEY'] for a in ANCHORS}
        cov[ANCHORS[3]] = []                    # punch one anchor out
        grid = _grid(tmp, _with_benchmark(cov))
        rep = pga.audit_price_grid(grid, ['HOLEY'])
    assert rep['n_interior_hole_symbols'] == 1
    assert rep['n_interior_hole_cells'] == 1
    assert rep['interior_holes']['HOLEY'] == [ANCHORS[3]]
    assert rep['verdict'] == 'STALE'
    assert any('INTERIOR price hole' in x for x in rep['findings']), rep['findings']


def test_a_LEADING_gap_is_NOT_a_hole():
    """Before a name's first price it is simply not listed yet, and the terminal policy is
    the right answer -- calling that a defect would fire on every recent IPO."""
    d, tmp = _tmp()
    with d:
        cov = {a: (['LATECO'] if i >= 3 else []) for i, a in enumerate(ANCHORS)}
        grid = _grid(tmp, _with_benchmark(cov))
        rep = pga.audit_price_grid(grid, ['LATECO'])
    assert rep['n_interior_hole_symbols'] == 0


def test_a_TRAILING_gap_is_NOT_a_hole():
    """After a name's last price it may genuinely have stopped trading; the price file cannot
    distinguish that from a gap, and the terminal policy exists for exactly this case."""
    d, tmp = _tmp()
    with d:
        cov = {a: (['GONECO'] if i <= 2 else []) for i, a in enumerate(ANCHORS)}
        grid = _grid(tmp, _with_benchmark(cov))
        rep = pga.audit_price_grid(grid, ['GONECO'])
    assert rep['n_interior_hole_symbols'] == 0


def test_the_hole_count_is_reported_PER_VENUE_so_the_market_is_visible():
    """The 164-of-177-on-.L shape: without the per-venue split it reads as diffuse noise."""
    d, tmp = _tmp()
    with d:
        lse = _names('L', 5, '.L')
        cov = {a: list(lse) for a in ANCHORS}
        cov[ANCHORS[6]] = []
        grid = _grid(tmp, _with_benchmark(cov))
        rep = pga.audit_price_grid(grid, lse)
    assert rep['venues']['.L']['n_interior_hole_symbols'] == 5
    assert rep['venues']['.L']['n_interior_hole_cells'] == 5
    assert '5 hole name(s)' in pga.format_audit(rep)


def test_a_fully_priced_grid_reports_NO_holes():
    """The false-positive side, and the mutation control for the tests above: the same
    fixture with nothing punched out must be clean."""
    d, tmp = _tmp()
    with d:
        grid = _grid(tmp, _with_benchmark({a: ['SOLID'] for a in ANCHORS}))
        rep = pga.audit_price_grid(grid, ['SOLID'])
    assert rep['n_interior_hole_symbols'] == 0
    assert rep['n_interior_hole_cells'] == 0
    assert not any('INTERIOR' in x for x in rep['findings'])
    assert rep['verdict'] == 'OK'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
