"""
Ad-hoc saved-run analysis script: the verdict must be a measurement, not a formatting choice.

WHY THIS FILE EXISTS (2026-08-22, reviewer C3).  The pipeline path was fixed to refuse a
smoking-gun verdict computed from a non-finite IC.  The SAME defect was live two files away and
in a worse form:

    return comp, best["metric"], float(best["IC_real"]), comp < float(best["IC_real"])

`run_analysis_on_saved_run._summ` compared two floats with no finiteness guard, and the caller
printed `does NOT hold` off the result.  `nan < nan` is False, so an all-NaN IC table produced a
confident negative verdict -- identical to the
`COMPOSITE IC_real=+nan ... -> smoking gun DOES NOT hold` the 08-20 and 08-22 runs printed.

It was MORE NaN-prone than the pipeline path, not less: its `HORIZONS` pairs are
`date_requested` labels while `real_ic.load_real` pivoted on `date_actual`, so most pairs were
not columns at all -- and the loop then SILENTLY DROPPED them (`if b in real.columns and e in
real.columns`) and handed `ic_table` a shorter, sometimes empty, pair list.

The module imports only stdlib at import time (pandas et al. live inside `main`), so `_summ` is
testable directly with no pickle, no price file and no network.
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

import real_ic as ric
import run_analysis_on_saved_run as ras


def _tbl(comp, singles):
    """An IC table shaped as `ic_table` returns it: COMPOSITE plus named single metrics."""
    rows = [{'metric': 'COMPOSITE', 'IC_real': comp, 'IC_recon': comp, 'n': 3}]
    for m, v in singles:
        rows.append({'metric': m, 'IC_real': v, 'IC_recon': v, 'n': 3})
    return pd.DataFrame(rows)


def test_summ_REFUSES_an_all_nan_table_instead_of_returning_a_verdict():
    """The exact table the 08-20/08-22 grid produced. It must raise, not return `False`."""
    tbl = _tbl(np.nan, [('RoA', np.nan), ('earnYield', np.nan)])
    with pytest.raises(RuntimeError) as e:
        ras._summ(tbl, '24m', ric)
    assert 'REFUSING' in str(e.value)
    #  and the refusal must not itself read like the finding it declines to make
    assert 'DOES NOT hold' not in str(e.value)
    assert 'HOLDS' not in str(e.value)


def test_summ_REFUSES_when_only_ONE_side_is_nan():
    """A half-measured comparison is not half a finding."""
    for comp, best in ((np.nan, 0.04), (0.02, np.nan)):
        with pytest.raises(RuntimeError):
            ras._summ(_tbl(comp, [('RoA', best)]), '24m', ric)


def test_summ_returns_a_correct_verdict_LINE_on_real_numbers_both_ways():
    """Both directions, so the guard cannot have been bought by hardwiring one answer."""
    comp, bm, bic, line = ras._summ(_tbl(0.01, [('RoA', 0.05)]), '24m', ric)
    assert (comp, bm, bic) == (0.01, 'RoA', 0.05)
    assert 'HOLDS' in line and 'DOES NOT hold' not in line

    comp, bm, bic, line = ras._summ(_tbl(0.05, [('RoA', 0.01)]), '24m', ric)
    assert 'DOES NOT hold' in line


def test_summ_picks_the_best_single_by_ABSOLUTE_ic_as_before():
    """Behaviour that must NOT have changed: the strongest single metric is chosen on |IC|, so a
    strongly NEGATIVE single still wins. The guard was added; the selection was not touched."""
    comp, bm, bic, _line = ras._summ(
        _tbl(0.02, [('RoA', 0.03), ('CycleHeat', -0.40)]), '24m', ric)
    assert bm == 'CycleHeat' and bic == -0.40


def test_summ_reuses_real_ic_verdict_line_rather_than_its_own_comparison():
    """ONE guard, in one place. A second implementation is how the two paths came to disagree
    about what counts as a measurement in the first place."""
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(ras.__file__).read_text(encoding='utf-8'))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == '_summ')
    called = {getattr(c.func, 'attr', None) for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert 'verdict_line' in called, 'the finiteness guard is re-implemented locally'
    #  and the RETURNED verdict is not a bare comparison.  Checked on the AST, not by string
    #  search: the docstring quotes the retired line verbatim, and a text scan reads the
    #  explanation of the defect as the defect (which is how this very test first failed).
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    assert isinstance(ret.value, ast.Tuple) and len(ret.value.elts) == 4, ast.unparse(ret)
    assert not any(isinstance(el, ast.Compare) for el in ret.value.elts), (
        '_summ still returns a raw comparison as its verdict: %s' % ast.unparse(ret))
    assert isinstance(ret.value.elts[3], ast.Name), ast.unparse(ret)


def test_the_ic_loop_RESOLVES_its_anchors_and_no_longer_drops_them_silently():
    """The silent filter is the other half of C3: dropping unresolvable windows without a word
    handed `ic_table` a shorter -- possibly empty -- pair list, which comes back all-NaN and
    reads exactly like a measurement."""
    import ast
    import pathlib
    src = pathlib.Path(ras.__file__).read_text(encoding='utf-8')
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = {getattr(c.func, 'attr', None) for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert 'resolve_anchors' in calls, 'the ad-hoc path still trusts raw literals'
    #  AST again, and NOT a `'in real.columns' not in src` text scan: `real.columns` is a
    #  legitimate ARGUMENT to resolve_anchors now, and the comment above the fix quotes the old
    #  filter.  What must be gone is a COMPREHENSION FILTERED on `.columns` -- the silent drop.
    filtered = [c for c in ast.walk(fn) if isinstance(c, ast.comprehension)
                and any(isinstance(a, ast.Attribute) and a.attr == 'columns'
                        for cond in c.ifs for a in ast.walk(cond))]
    assert not filtered, 'a comprehension still silently filters pairs on real.columns'


def test_the_horizon_pairs_are_all_resolvable_against_the_graders_anchor_grid():
    """A live check on the constants themselves, not just on the machinery.

    Every date in `HORIZONS` must resolve against `returns_core.DEFAULT_ANCHORS` -- which is the
    column space `load_real` now produces. If a future edit puts a settlement-day literal back
    in `HORIZONS`, this fails here rather than at 3am inside a saved-run analysis."""
    import ast
    import pathlib
    import returns_core as rc
    src = pathlib.Path(ras.__file__).read_text(encoding='utf-8')
    tree = ast.parse(src)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Assign) and len(n.targets) == 1
                and getattr(n.targets[0], 'id', None) == 'HORIZONS')
    horizons = ast.literal_eval(node.value)
    wanted = sorted({d for prs in horizons.values() for pr in prs for d in pr})
    assert wanted, wanted
    got = ric.resolve_anchors(list(rc.DEFAULT_ANCHORS), wanted, what='HORIZONS anchor')
    #  every one resolves, and on this grid they are exact -- no silent displacement
    assert set(got) == set(wanted)
    assert all(got[d] == d for d in wanted), {d: got[d] for d in wanted if got[d] != d}


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
