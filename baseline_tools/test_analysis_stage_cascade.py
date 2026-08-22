"""
Post-pick analysis suite: STAGE CASCADE self-checks (offline).

WHY THIS FILE EXISTS.  The 08-20 and 08-22 runs logged FOUR dead stages, and one of them was
not its own defect:

    RuntimeError: beat-rate stage skipped: per_anchor/price_source missing

`price_source` was fine on both runs (`<<< stage OK: build-price-source (0.1s)`).  `per_anchor`
was None because `per_anchor = grid_out[1] if grid_out else None` and the GRID stage had raised
inside carveOut's sector-coverage guard.  So the beat-rate failure was a pure CASCADE, and
"fixing" it locally -- by softening the gate, or by rebuilding per_anchor a second way -- would
have hidden the grid failure while producing a beat-rate off an unranked universe.

These tests pin the two things that make that claim checkable:
  1. `per_anchor` has EXACTLY ONE source (the grid stage's return), and the beat gate has
     exactly one other condition (`price_source`, itself its own stage).  So a missing
     `per_anchor` can only ever mean "the grid stage failed" -- there is no second cause to
     chase, and no local patch that could be right.
  2. The carve call that raised inside the grid stage NO LONGER RAISES on a PIT-shaped
     universe.  That is the one link in the chain the fix had to break.

NOT A LIVE RUN.  Neither test executes the pipeline (no prices, no 12h fetch, no network here).
They establish the cascade structurally and the unblocking behaviourally; a live run remains
the only way to see the beat-rate stage actually print.
"""
import ast
import os
import pathlib
import sys

import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import carveOut as co


def _suite_tree():
    src = (pathlib.Path(_HERE) / 'pipeline_analysis.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == 'run_analysis_suite')
    return fn


def test_per_anchor_has_exactly_one_source_the_grid_stage():
    """If a second assignment to `per_anchor` ever appears, the cascade claim stops holding and
    a beat-rate failure could mean two different things."""
    fn = _suite_tree()
    assigns = [n for n in ast.walk(fn)
               if isinstance(n, ast.Assign) and len(n.targets) == 1
               and getattr(n.targets[0], 'id', None) == 'per_anchor']
    assert len(assigns) == 1, [ast.unparse(a) for a in assigns]
    expr = ast.unparse(assigns[0].value)
    assert expr == 'grid_out[1] if grid_out else None', expr

    grid_assigns = [n for n in ast.walk(fn)
                    if isinstance(n, ast.Assign) and len(n.targets) == 1
                    and getattr(n.targets[0], 'id', None) == 'grid_out']
    assert len(grid_assigns) == 1
    assert '_run_stage' in ast.unparse(grid_assigns[0].value)


def test_the_beat_rate_gate_still_refuses_and_was_NOT_softened():
    """The gate is CORRECT -- it must stay.  A beat-rate computed without the PIT rankings would
    be a number off nothing, which is the failure mode of the whole 08-22 run in miniature."""
    fn = _suite_tree()
    beat = next(n for n in ast.walk(fn)
                if isinstance(n, ast.FunctionDef) and n.name == '_beat')
    raises = [n for n in ast.walk(beat) if isinstance(n, ast.Raise)]
    assert len(raises) == 1, 'the beat-rate precondition was removed, not unblocked'
    guard = next(n for n in ast.walk(beat) if isinstance(n, ast.If))
    assert ast.unparse(guard.test) == 'per_anchor is None or price_source is None', \
        ast.unparse(guard.test)


def test_price_source_is_its_own_stage_so_the_other_gate_condition_is_independent():
    """`price_source` was OK on both failing runs; pin that it comes from its own dispatch, so
    the gate's two conditions cannot fail together for one reason."""
    fn = _suite_tree()
    assigns = [n for n in ast.walk(fn)
               if isinstance(n, ast.Assign) and len(n.targets) == 1
               and getattr(n.targets[0], 'id', None) == 'price_source']
    assert len(assigns) == 1
    v = ast.unparse(assigns[0].value)
    assert v.startswith('_run_stage(') and '_build_price_source' in v, v


# --------------------------------------------------------------------------- #
#  the link that had to break: the PIT carve no longer raises                  #
# --------------------------------------------------------------------------- #
def _pit_inputs(n_live=400, n_dead=500, live_covered=None):
    live = ['L%04d' % i for i in range(n_live)]
    dead = ['D%04d' % i for i in range(n_dead)]
    pool = live + dead
    tickers = pd.DataFrame({'symbol': pool, 'name': ['N' + s for s in pool]})
    #  distinct market caps so the issuer de-dup has nothing to collapse
    cdx = pd.DataFrame({'source': pool, 'date': pd.Timestamp('2025-01-01'),
                        'marketCap': [1e8 + 1e5 * i for i in range(len(pool))],
                        'totalStockholdersEquity': 5e7, 'totalAssets': 1e8,
                        'revenue': 5e7, 'weightedAverageShsOut': 1e6, 'netIncome': 1e6})
    cov = live if live_covered is None else live[:live_covered]
    smap = {s: 'Technology' for s in cov}
    return set(pool), cdx, tickers, set(live), smap


def test_the_pit_carve_no_longer_raises_and_returns_a_usable_general_pool(tmp_path,
                                                                         monkeypatch):
    """The exact call inside the grid stage.  Pre-fix this raised RuntimeError and every
    downstream stage that needed `per_anchor` died with it."""
    monkeypatch.chdir(tmp_path)
    import depth_horizon_grid as dh
    uni, cdx, tickers, live, smap = _pit_inputs()
    monkeypatch.setattr(co, '_load_sector_map', lambda *a, **k: smap)
    monkeypatch.setattr(co, '_load_industry_map', lambda *a, **k: {})

    lines = []
    general = dh.carve_general_universe(uni, cdx, tickers, lines.append,
                                        coverage_scope=live)

    assert isinstance(general, set) and general, 'the carve returned nothing to rank'
    #  the sectorless (dead) names are IN it -- the accepted bias -- and it SAID so
    assert any(s.startswith('D') for s in general)
    assert any('SECTORLESS LEAK' in x for x in lines), lines


def test_the_same_call_WOULD_still_raise_without_the_scope(tmp_path, monkeypatch):
    """The control: the guard is intact and the fix is the SCOPE, not its removal.

    If this stops raising, `carve_general_universe` has been made permissive for everyone and
    a genuinely poisoned map would sail through the live path too."""
    monkeypatch.chdir(tmp_path)
    import depth_horizon_grid as dh
    uni, cdx, tickers, live, smap = _pit_inputs()
    monkeypatch.setattr(co, '_load_sector_map', lambda *a, **k: smap)
    monkeypatch.setattr(co, '_load_industry_map', lambda *a, **k: {})
    with pytest.raises(RuntimeError) as e:
        dh.carve_general_universe(uni, cdx, tickers, lambda *a: None)
    assert 'covers only' in str(e.value)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
