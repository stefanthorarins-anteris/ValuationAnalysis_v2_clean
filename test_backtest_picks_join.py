"""The backtest's picks table must describe the universe that was actually graded.

THE DEFECT (2026-08-08).  `backtest_outputs.save_all_outputs` chose its postRank with
`glob.glob('postRank_*.pickle')` + `sort(reverse=True)` -- the alphabetically-last FILENAME
in the working directory.  On the 2026-08-08 run that selected
`postRank_2026-08-04_fmp_stock_TEST1.pickle`, the **126-name curated TEST universe** whose own
provenance note reads "POOL-RELATIVE OUTPUT IS MEANINGLESS HERE ... never to read a pick",
while the scenarios were graded on the **2,613-source CUR3K panel**.  `stock_picks.csv` and
the HTML report showed picks from one universe beside returns from another, silently.

SAME CLASS AS the already-fixed `verify_test_universe.newest_panel()` filename-sort bug -- and
a DATE sort would not have fixed this one either, because 2026-08-04 TEST1 genuinely was the
newest artifact present.  The question is "which postRank belongs to THIS panel", and the
`universe_fingerprint` stamp exists to answer exactly that.

THE RULE UNDER TEST: match on fingerprint, or REFUSE and say so in the run folder.  A missing
picks table is a visible absence; a wrong one is an invisible error.
"""

import os
import sys

import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import backtest_outputs as bo


def _postrank_pickle(path, fingerprint, universe, sources):
    pd.to_pickle({'universe': universe,
                  'universe_fingerprint': fingerprint,
                  'postRank': pd.DataFrame({'source': list(sources),
                                            'AggScore': range(len(sources))})},
                 path)


@pytest.fixture()
def two_postranks(tmp_path, monkeypatch):
    """The real shape of the defect: a TEST-universe artifact that sorts LAST by filename
    and by date, beside the production artifact that actually belongs to the panel."""
    monkeypatch.chdir(tmp_path)
    _postrank_pickle(tmp_path / 'postRank_2026-08-04_fmp_stock_TEST1.pickle',
                     '6f8b8825dc90', 'stock_TEST1', ['TESTA', 'TESTB'])
    panel_dir = tmp_path / 'panels_2026-08-07'
    panel_dir.mkdir()
    _postrank_pickle(panel_dir / 'postRank_2026-08-07_fmp_stock_CUR3K.pickle',
                     '844b3eaf1ea2', 'stock_CUR3K', ['AAPL', 'MSFT', 'SHEL'])
    return tmp_path, panel_dir


def test_it_joins_on_FINGERPRINT_and_finds_the_panel_s_OWN_postRank(two_postranks):
    _root, panel_dir = two_postranks
    df, path, note = bo.select_postrank_for_panel(
        {'universe': 'stock_CUR3K', 'universe_fingerprint': '844b3eaf1ea2',
         'dir': str(panel_dir)}, verbose=False)
    assert df is not None and list(df['source']) == ['AAPL', 'MSFT', 'SHEL']
    assert 'CUR3K' in os.path.basename(path)
    assert '844b3eaf1ea2' in note


def test_the_TEST_universe_artifact_is_NOT_chosen_even_though_it_sorts_LAST(two_postranks):
    """The exact 2026-08-08 miss: TEST1 sorts last by filename AND is the newest by date,
    so neither a filename sort nor a date sort would have avoided it."""
    _root, panel_dir = two_postranks
    df, path, _n = bo.select_postrank_for_panel(
        {'universe': 'stock_CUR3K', 'universe_fingerprint': '844b3eaf1ea2',
         'dir': str(panel_dir)}, verbose=False)
    assert 'TEST1' not in os.path.basename(path)
    assert 'TESTA' not in set(df['source'])


def test_it_REFUSES_when_no_postRank_belongs_to_the_panel(two_postranks, capsys):
    """A picks table from a different universe is worse than no picks table."""
    df, path, note = bo.select_postrank_for_panel(
        {'universe': 'stock_OTHER', 'universe_fingerprint': 'deadbeef0000',
         'dir': ''}, verbose=True)
    out = capsys.readouterr().out
    assert df is None and path is None
    assert 'NO postRank matches the graded panel' in note
    assert 'STOCK-PICKS TABLE NOT WRITTEN' in out
    assert 'UNAFFECTED' in out, 'must say what is NOT invalidated by the refusal'


def test_an_UNSTAMPED_panel_REFUSES_rather_than_guessing(two_postranks):
    """A pre-2026-08-02 panel carries no fingerprint, so no join can be established.  The
    old code would happily have joined it to anything."""
    df, path, note = bo.select_postrank_for_panel(
        {'universe': 'stock_NA1_EU1'}, verbose=False)
    assert df is None and path is None
    assert 'NO universe_fingerprint' in note


def test_an_UNSTAMPED_postRank_can_never_match(tmp_path, monkeypatch):
    """A stamped panel must not be joined to an unstamped artifact on the strength of the
    name alone -- `stock_NA1_EU1` MEANS something different before and after 2026-08-02."""
    monkeypatch.chdir(tmp_path)
    pd.to_pickle({'universe': 'stock_CUR3K',
                  'postRank': pd.DataFrame({'source': ['X'], 'AggScore': [1]})},
                 tmp_path / 'postRank_2026-08-07_fmp_stock_CUR3K.pickle')
    df, path, note = bo.select_postrank_for_panel(
        {'universe': 'stock_CUR3K', 'universe_fingerprint': '844b3eaf1ea2', 'dir': ''},
        verbose=False)
    assert df is None and 'unstamped' in note


def test_the_REFUSAL_is_RECORDED_in_the_run_folder(tmp_path):
    """An absent `stock_picks.csv` and a picks table nobody attempted look identical on
    disk.  The run folder is the only artifact a later reader has."""
    folder = tmp_path / 'backtest_results_x'
    (folder / 'data').mkdir(parents=True)
    path = bo._write_picks_refusal(str(folder), 'because the fingerprints differ',
                                   verbose=False)
    assert os.path.basename(path) == 'stock_picks_NOT_WRITTEN.txt'
    body = open(path, encoding='utf-8').read()
    assert 'deliberately NOT written' in body and 'fingerprints differ' in body


def test_run_all_PASSES_THE_STAMP_so_the_join_is_never_a_filename_sort():
    """The stamp has to actually reach `save_all_outputs`; without it the selector refuses,
    which would silently drop the picks table from every backtest."""
    import inspect
    import backtest_unified as bt
    src = inspect.getsource(bt.run_all)
    assert 'panel_stamp=_panel_stamp' in src
    assert "'universe_fingerprint': dmdic.get('universe_fingerprint')" in src
