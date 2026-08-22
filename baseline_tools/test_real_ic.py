"""
real_ic self-checks (offline, no network).

WHAT WENT WRONG, because these tests exist to stop exactly it coming back.

`real_ic.py` hardcoded the price grid's column labels -- "2020-12-28", "2022-12-27",
"2024-12-28" -- which were the trading days one 2026-07 fetch happened to land on.  The grid
the 08-20 and 08-22 runs read is labelled ['2018-12-31','2019-12-30','2020-12-30','2021-12-30',
'2022-12-29','2023-12-28','2024-12-30'].  Two distinct failures followed from the one cause:

  * `profit_timing_real` indexed `real.at[s, "2020-12-28"]` and died with
    `KeyError: '2020-12-28'`, killing the whole stage; and
  * `ic_table` guarded the SAME lookups with `if buy in real.columns else None`, so it did not
    die -- it produced an ALL-NaN table, and the stage printed
        "COMPOSITE IC_real=+nan vs best single (RoA) IC_real=+nan -> smoking gun DOES NOT hold"
    on both runs.  `nan < nan` is False, so the verdict was not merely unreliable, it was a
    CONSTANT rendered from nothing.

THE TEST-DESIGN CONSTRAINT THAT FOLLOWS.  A test asserting "the IC table has the expected
shape / the expected metric rows" would have passed happily while every cell was NaN -- it is
the defect wearing a test.  So the assertions below are on VALUES (finite, correctly SIGNED,
the right anchor labels) and on REFUSAL BEHAVIOUR (it raises rather than printing), never on
shape alone.
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

import model_vs_metric as mvm
import real_ic as ric

#  SETTLEMENT-DAY labels -- what the run machine's `date_actual` column happened to hold.
#  `load_real` no longer keys on that column (see the C1 block at the bottom of this file), so
#  these are retained ONLY as a column set that is DISPLACED from the calendar year-ends, which
#  is what exercises the resolver's tolerance path.  Do not read them as "what the file yields".
LIVE_GRID = ['2018-12-31', '2019-12-30', '2020-12-30', '2021-12-30',
             '2022-12-29', '2023-12-28', '2024-12-30']
#  The grid the retired literals came from.
LEGACY_GRID = ['2018-12-31', '2020-12-28', '2022-12-27', '2024-12-28']


# --------------------------------------------------------------------------- #
#  anchor resolution                                                          #
# --------------------------------------------------------------------------- #
def test_year_end_anchors_resolve_onto_the_live_grid_columns():
    """The failing case, by value: every intended even-year end must land on ITS OWN year."""
    got = ric.resolve_anchors(LIVE_GRID, ric.INTENDED_IC_ANCHORS_24M)
    assert got == {'2018-12-31': '2018-12-31',
                   '2020-12-31': '2020-12-30',
                   '2022-12-31': '2022-12-29',
                   '2024-12-31': '2024-12-30'}, got


def test_resolution_is_backward_compatible_with_the_grid_the_literals_came_from():
    """The retired constants must be REPRODUCED, not merely replaced.

    If the resolver returned different dates on the legacy grid, the fix would be silently
    re-defining what the diagnostic measures rather than repairing how it finds it."""
    pairs = ric.build_pairs_24m(LEGACY_GRID)
    assert pairs == [('2018-12-31', '2020-12-28'),
                     ('2020-12-28', '2022-12-27'),
                     ('2022-12-27', '2024-12-28')], pairs


def test_the_resolver_refuses_rather_than_stepping_back_into_the_PREVIOUS_year():
    """THE TRAP IN THE OBVIOUS IMPLEMENTATION.

    "Nearest column on-or-before the intended anchor" is the intuitive rule and it is wrong
    here: on-or-before 2020-12-31 in a grid whose only earlier column is 2019-12-30 gives a
    date 367 days away, and every IC would then be computed over a 12-month-shifted window
    while looking perfectly healthy.  A missing anchor must be MISSING."""
    assert ric.resolve_anchor(['2019-12-30'], '2020-12-31') is None
    with pytest.raises(RuntimeError) as e:
        ric.resolve_anchors(['2018-12-31', '2019-12-30'], ['2020-12-31'])
    assert 'cannot resolve' in str(e.value)
    #  and it says what the grid DOES carry, so the operator can act on it
    assert '2019-12-30' in str(e.value)


def test_the_tolerance_is_wide_enough_for_holiday_drift_and_far_from_a_year():
    """The threshold is grounded, not picked: the true displacement of a year-end anchor is a
    handful of days; the distance to the adjacent year's column is ~358+."""
    assert 4 <= ric.ANCHOR_TOLERANCE_DAYS < 180
    #  a 4-day holiday displacement resolves...
    assert ric.resolve_anchor(['2020-12-27'], '2020-12-31') == '2020-12-27'
    #  ...and the neighbouring year never does, from either side.
    assert ric.resolve_anchor(['2021-12-30'], '2020-12-31') is None
    assert ric.resolve_anchor(['2019-12-30'], '2020-12-31') is None


def test_an_exact_distance_tie_resolves_deterministically_to_the_earlier_date():
    """Two candidates equidistant from the anchor must not depend on column order."""
    assert ric.resolve_anchor(['2021-01-03', '2020-12-29'], '2020-12-31') == '2020-12-29'
    assert ric.resolve_anchor(['2020-12-29', '2021-01-03'], '2020-12-31') == '2020-12-29'


def test_non_date_columns_are_ignored_not_crashed_on():
    """The grid is a pivot, so a stray non-date column must not take the whole stage down."""
    assert ric.resolve_anchor(['symbol', 'note', '2020-12-30'], '2020-12-31') == '2020-12-30'


# --------------------------------------------------------------------------- #
#  the verdict -- the half that printed a conclusion from NaN                  #
# --------------------------------------------------------------------------- #
def test_a_verdict_cannot_be_stated_from_NaN():
    """`nan < nan` is False, which is how "DOES NOT hold" got printed twice off no data."""
    for comp, best in ((np.nan, np.nan), (np.nan, 0.05), (0.02, np.nan),
                       (np.inf, 0.05), (0.02, -np.inf)):
        with pytest.raises(RuntimeError) as e:
            ric.verdict_line(comp, 'RoA', best)
        assert 'REFUSING' in str(e.value)
        #  the refusal must not read like the finding it declines to make
        assert 'DOES NOT hold' not in str(e.value)
        assert 'HOLDS' not in str(e.value)


def test_a_verdict_on_real_numbers_reads_the_comparison_the_right_way_round():
    """Asserted in BOTH directions, so the guard cannot have been bought by hardwiring one."""
    assert 'HOLDS' in ric.verdict_line(0.01, 'RoA', 0.05)
    assert 'DOES NOT hold' in ric.verdict_line(0.05, 'RoA', 0.01)
    assert 'DOES NOT hold' in ric.verdict_line(0.05, 'RoA', 0.05)   # not strictly less


# --------------------------------------------------------------------------- #
#  ic_table -- values, not shape                                              #
# --------------------------------------------------------------------------- #
def _synthetic_panel_and_grid(n=200, seed=0):
    """A panel whose RoA genuinely predicts the forward return, and a matching price grid.

    Built so `IC_real` has a KNOWN SIGN.  That is the whole point: an all-NaN table has the
    right shape and the right rows, so only a signed, finite value distinguishes a working
    measurement from the broken one this module shipped.
    """
    rng = np.random.default_rng(seed)
    srcs = ['S%04d' % i for i in range(n)]
    roa = rng.normal(size=n)

    #  Price at each anchor: level 100 at the first, then compounding in RoA order, so the
    #  forward return is monotone increasing in RoA -> IC_real must be strongly POSITIVE.
    grid = {}
    for k, d in enumerate(LIVE_GRID):
        grid[d] = 100.0 * (1.0 + 0.10 * roa) ** k
    real = pd.DataFrame(grid, index=srcs)
    real.index.name = 'symbol'

    #  Every other metric gets its OWN noise draw, and `_price` mirrors the real grid, so the
    #  reconstructed leg is computable too.  Both matter: a panel of constant columns makes
    #  spearmanr undefined and would hand the test an all-NaN table for a REASON OF ITS OWN
    #  MAKING -- i.e. it would stop distinguishing a working module from the broken one.
    noise = {m: rng.normal(size=n) for m in mvm.METRICS}
    rows = []
    for d in LIVE_GRID:
        for i, s in enumerate(srcs):
            r = {'source': s, 'date': pd.Timestamp(d), '_price': float(grid[d][i])}
            for m in mvm.METRICS:
                r[m] = float(noise[m][i])
            r['RoA'] = float(roa[i])
            rows.append(r)
    panel = pd.DataFrame(rows)
    return panel, real


def test_ic_real_is_a_FINITE_correctly_SIGNED_number_not_NaN():
    """The regression test with teeth: pre-fix this table was all NaN on the live grid."""
    panel, real = _synthetic_panel_and_grid()
    pairs = ric.build_pairs_24m(real.columns)
    tbl, label = ric.ic_table(panel, real, pairs, '24m')

    row = tbl[tbl['metric'] == 'RoA'].iloc[0]
    assert np.isfinite(row['IC_real']), 'IC_real is NaN -- the all-NaN defect is back'
    assert row['IC_real'] > 0.9, row['IC_real']       # constructed monotone in RoA
    assert row['n'] == len(pairs), row['n']
    #  and the table as a whole is not silently half-empty
    assert np.isfinite(tbl['IC_real']).all(), tbl
    #  the reconstructed leg is computed off the same prices, so it must agree closely --
    #  a divergence here means one leg silently lost its anchors again.
    assert np.isfinite(row['IC_recon']), row
    assert abs(row['IC_real'] - row['IC_recon']) < 0.05, row


def test_ic_table_RAISES_on_an_unresolved_anchor_instead_of_returning_NaN():
    """The `if buy in real.columns else None` that manufactured the all-NaN table.

    Asked for a column the grid does not have, the table must refuse -- because a skipped
    anchor is indistinguishable, downstream, from an anchor that had nothing to say."""
    panel, real = _synthetic_panel_and_grid()
    with pytest.raises(KeyError) as e:
        ric.ic_table(panel, real, [('2018-12-31', '2020-12-28')], '24m')
    assert '2020-12-28' in str(e.value)


def test_the_pipeline_and_script_paths_build_the_SAME_anchor_pairs():
    """`run_in_pipeline` and `main` carried two independent copies of the stale literals and
    both were wrong.  One builder now serves both; pin that they agree."""
    import ast
    import pathlib
    src = (pathlib.Path(ric.__file__)).read_text(encoding='utf-8')
    tree = ast.parse(src)
    for fname in ('run_in_pipeline', 'main'):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == fname)
        called = {getattr(c.func, 'id', None) for c in ast.walk(fn)
                  if isinstance(c, ast.Call)}
        assert 'build_pairs_24m' in called, (
            '%s builds its own anchor pairs again -- that is how the two copies drifted' % fname)
    #  and the retired dates survive ONLY as prose.  Checked on the AST rather than by
    #  string search, so the comments and docstrings that EXPLAIN the defect do not read as
    #  the defect: what matters is that no executable string constant is one of them again.
    docstrings = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            head = n.body[0] if n.body else None
            if (isinstance(head, ast.Expr) and isinstance(head.value, ast.Constant)
                    and isinstance(head.value.value, str)):
                docstrings.add(id(head.value))
    stale = {'2020-12-28', '2022-12-27', '2024-12-28'}
    live = [c.value for c in ast.walk(tree)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
            and id(c) not in docstrings and c.value in stale]
    assert not live, 'retired anchor literal(s) are back as CODE in real_ic.py: %s' % live


# --------------------------------------------------------------------------- #
#  profit_timing_real -- the line that raised KeyError                        #
# --------------------------------------------------------------------------- #
def test_profit_timing_resolves_its_anchors_and_no_longer_KeyErrors(monkeypatch):
    """Exercises the exact statement that killed the stage.

    `stage2_pit` is stubbed because the pick reproduction is not what is under test here --
    the anchor lookup is.  Pre-fix this call raised KeyError: '2020-12-28' on the live grid.
    """
    import stage2_pit as s2
    panel, real = _synthetic_panel_and_grid(n=60)

    calls = {'n': 0}

    def _prepare_pit(dmdic, D, na1_only=False):
        calls['n'] += 1
        calls.setdefault('dates', []).append(D)
        return None, None

    def _stage2_top(_bs, _cdx):
        #  first anchor holds 4 names, the +2y anchor holds 2 of them
        return ['S0000', 'S0001', 'S0002', 'S0003'] if calls['n'] == 1 else \
               ['S0002', 'S0003']

    monkeypatch.setattr(s2, 'prepare_pit', _prepare_pit)
    monkeypatch.setattr(s2, 'stage1_boscore', lambda *a, **k: None)
    monkeypatch.setattr(s2, 'stage2_top', _stage2_top)

    ex, st, exdf, stdf = ric.profit_timing_real({}, panel, real)

    #  the anchors it used are the RESOLVED live-grid ones, not the retired literals
    assert calls['dates'] == ['2020-12-30', '2022-12-29'], calls['dates']
    #  and it produced real numbers off them
    assert sorted(ex) == ['S0000', 'S0001'] and sorted(st) == ['S0002', 'S0003']
    assert len(exdf) == 2 and len(stdf) == 2
    for d in (exdf, stdf):
        assert np.isfinite(d['during']).all() and np.isfinite(d['full']).all()


def test_profit_timing_REFUSES_when_the_grid_cannot_reach_its_third_anchor():
    """A grid that stops at 2022 cannot answer a D->+2y->+4y question.  It must say so rather
    than silently shortening the horizon or returning empty frames that read as 'no priceable'."""
    panel, real = _synthetic_panel_and_grid(n=60)
    truncated = real[[c for c in real.columns if c < '2023']]
    with pytest.raises(RuntimeError) as e:
        ric.profit_timing_real({}, panel, truncated)
    assert 'profit-timing anchor' in str(e.value)
    assert '2024-12-31' in str(e.value)


# --------------------------------------------------------------------------- #
#  C1: THE PIVOT AXIS.  `date_requested` (the anchor), never `date_actual`.    #
# --------------------------------------------------------------------------- #
#  THE DEFECT THESE PIN, and it was introduced BY the previous fix rather than left by it.
#  `load_real` pivoted on `date_actual` -- the day each venue settled -- so ONE fetch anchor
#  appears as SEVERAL columns.  Measured on baseline_tools/price_data/real_prices.csv:
#  `date_requested` 2020-12-31 splits into `date_actual` 2020-12-28 (58,838 rows) and
#  2020-12-31 (9,901); 2022-12-30 splits into 2022-12-27 (64,490) and 2022-12-30 (15,441).
#  The resolver then preferred the exact intended calendar date -- the MINORITY fragment --
#  and joint-priced symbols per 24m pair collapsed 77-82% on the first two pairs.
#
#  WHY NO EXISTING GUARD COULD CATCH IT.  Both fragments are real columns with plausible
#  dates, so the resolver cannot tell them apart; and a 15%-of-the-universe venue-biased slice
#  still yields a FINITE IC, so `verdict_line` has nothing to refuse.  The stage printed a
#  confident smoking-gun verdict under a reassuring "anchor pairs (intended -> resolved)" line.
#  That is why these tests assert on JOINT-PRICED COUNTS and on AGREEMENT WITH THE GRADER, not
#  on the column names alone -- a column-name test would pass on either axis.
def _fragmented_csv(tmp_path, anchors, n=200):
    """A grid where each anchor settles on TWO different days, majority/minority.

    Mirrors the real file's shape: most symbols settle a few days early, a minority on the
    nominal date, and the benchmark is in the MAJORITY group -- which is exactly the systematic
    (not random) selection bias the wrong axis produced.
    """
    import returns_core as rc
    majority = ['M%03d' % i for i in range(n)] + [rc.BENCHMARK_SYMBOL]
    minority = ['N%03d' % i for i in range(10)]
    rows = []
    for k, a in enumerate(anchors):
        early = str(pd.Timestamp(a) - pd.Timedelta(days=3))[:10]
        for sym in majority:
            rows.append({'date_requested': a, 'date_actual': early,
                         'symbol': sym, 'adjClose': 100.0 * (1.05 ** k)})
        for sym in minority:
            rows.append({'date_requested': a, 'date_actual': a,
                         'symbol': sym, 'adjClose': 50.0 * (1.05 ** k)})
    fp = tmp_path / 'real_prices.csv'
    pd.DataFrame(rows).to_csv(fp, index=False)
    return str(fp), set(majority), set(minority)


def test_load_real_keys_on_the_ANCHOR_not_the_settlement_day(tmp_path):
    """The columns must be the ANCHORS asked for, one per anchor -- not one per venue calendar."""
    import returns_core as rc
    anchors = list(rc.DEFAULT_ANCHORS)[:4]
    fp, maj, mino = _fragmented_csv(tmp_path, anchors)
    real = ric.load_real(fp)
    assert list(real.columns) == anchors, list(real.columns)
    #  the settlement days must NOT appear as columns of their own
    for a in anchors:
        early = str(pd.Timestamp(a) - pd.Timedelta(days=3))[:10]
        assert early not in real.columns, early


def test_the_fragmented_grid_does_not_SHRINK_the_joint_priced_population(tmp_path):
    """THE MEASUREMENT THAT MATTERS.  Both settlement groups must land in one column, so the
    joint-priced count is the whole population -- not the 10-name minority the resolver used to
    select. Asserted as a COUNT, because that is the quantity the IC is computed over."""
    import returns_core as rc
    anchors = list(rc.DEFAULT_ANCHORS)[:4]
    fp, maj, mino = _fragmented_csv(tmp_path, anchors)
    real = ric.load_real(fp)
    b, e = anchors[0], anchors[2]
    joint = pd.concat([real[b].rename('b'), real[e].rename('e')], axis=1).dropna()
    assert len(joint) == len(maj | mino), (len(joint), len(maj | mino))
    assert len(joint) > 200, len(joint)          # not the 10-name minority slice


def test_the_BENCHMARK_survives_the_axis_because_it_is_in_the_majority_group(tmp_path):
    """The tell that the old selection was SYSTEMATIC, not random: URTH was NaN in both
    newly-chosen columns and present in both majority ones.  A benchmark-less column silently
    disables every excess/beat-rate readout built on it."""
    import returns_core as rc
    anchors = list(rc.DEFAULT_ANCHORS)[:4]
    fp, _maj, _mino = _fragmented_csv(tmp_path, anchors)
    real = ric.load_real(fp)
    for a in anchors:
        assert rc.BENCHMARK_SYMBOL in real.index
        assert not pd.isna(real.at[rc.BENCHMARK_SYMBOL, a]), a


def test_load_real_agrees_with_the_GRADER_on_every_cell(tmp_path):
    """The invariant that makes the axis choice checkable rather than argued.

    `returns_core.PriceSource` is what the depth-grid and beat-rate stages price through, and it
    reads `date_requested` only (returns_core.py:70-76). Every non-NaN cell of this matrix must
    be a price the grader also has, at the same anchor, with the same value -- otherwise the IC
    diagnostic and the return grid describe different populations."""
    import returns_core as rc
    anchors = list(rc.DEFAULT_ANCHORS)[:5]
    fp, maj, mino = _fragmented_csv(tmp_path, anchors, n=40)
    real = ric.load_real(fp, anchors=anchors)
    ps = rc.PriceSource(fp, anchors=anchors)
    checked = 0
    for sym in sorted(maj | mino):
        for a in anchors:
            mine = real.at[sym, a] if sym in real.index and a in real.columns else None
            theirs = ps.price(sym, a)
            if theirs is None:
                assert mine is None or pd.isna(mine), (sym, a, mine)
            else:
                assert mine == theirs, (sym, a, mine, theirs)
                checked += 1
    assert checked > 100, checked


def test_a_duplicate_key_keeps_the_FIRST_row_exactly_as_PriceSource_does(tmp_path):
    """Two rows for one (symbol, anchor) -- the normal end state of a top-up fetch. The grader
    keeps the first occurrence; anything else (a mean, or the last) would make the IC read a
    price the return grid never used."""
    import returns_core as rc
    a = rc.DEFAULT_ANCHORS[0]
    rows = [{'date_requested': a, 'date_actual': a, 'symbol': 'DUP', 'adjClose': 11.0},
            {'date_requested': a, 'date_actual': '2018-12-28', 'symbol': 'DUP',
             'adjClose': 99.0}]
    fp = tmp_path / 'real_prices.csv'
    pd.DataFrame(rows).to_csv(fp, index=False)
    real = ric.load_real(str(fp), anchors=[a])
    ps = rc.PriceSource(str(fp), anchors=[a])
    assert real.at['DUP', a] == 11.0, real.at['DUP', a]
    assert ps.price('DUP', a) == 11.0
    assert real.at['DUP', a] == ps.price('DUP', a)


def test_load_real_never_reads_the_settlement_column():
    """AST guard.  The axis is one identifier in one function, and re-introducing it would be a
    one-word edit that every value test above would still pass on a NON-fragmented fixture."""
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(ric.__file__).read_text(encoding='utf-8'))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == 'load_real')
    consts = {c.value for c in ast.walk(fn)
              if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    assert 'date_actual' not in consts, 'load_real reads date_actual again'
    assert 'date_requested' in consts, consts


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
