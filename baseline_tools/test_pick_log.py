"""
Prospective pick-log self-checks (offline).

The live Sbocker run needs the network + built pickles, so we do NOT run the pipeline here.
We test the WRITER directly on a synthetic resdic that mimics the real shapes
(resdic['postRank'] general pool + resdic['carveout_sidelists'][label]['postRank'] for the
five cohorts), plus an AST wiring guard that Sbocker.main actually CALLS the writer.

The append-only contract is the crux: writing twice must NEVER mutate or drop an earlier
row -- the earlier bytes must be identical and the row count must strictly grow.

Run:  python baseline_tools/test_pick_log.py
"""
import ast
import inspect
import os
import sys
import tempfile
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import pandas as pd
import pick_log as plog


# --- synthetic resdic -------------------------------------------------------
def _rank_frame(sources, base=10.0):
    """A postRank-style frame: 'source' + descending 'AggScore' (+ a decoy column)."""
    return pd.DataFrame({
        'source': sources,
        'AggScore': [base - i for i in range(len(sources))],
        'rankOfRanks': list(range(1, len(sources) + 1)),
    })


def _fake_resdic():
    gen = _rank_frame([f'GEN{i}' for i in range(25)], base=100.0)   # >20 so head(20) bites
    tickers = pd.DataFrame({
        'symbol': [f'GEN{i}' for i in range(25)] + ['REITa', 'MINa', 'F1a', 'F2a', 'F3a'],
        'name':   [f'General Co {i}' for i in range(25)]
                  + ['Reit Co', 'Miner Co', 'Vehicle Co', 'Manager Co', 'Bank Co'],
    })
    sidelists = {
        'REIT': {'postRank': _rank_frame(['REITa', 'REITb'])},
        'Mining': {'postRank': _rank_frame(['MINa', 'MINb', 'MINc'])},
        plog._co.FIN1_VEHICLE: {'postRank': _rank_frame(['F1a'])},
        plog._co.FIN2_MANAGER: {'postRank': _rank_frame(['F2a', 'F2b'])},
        plog._co.FIN3_BALSHEET: {'postRank': _rank_frame(['F3a'])},
    }
    return {
        'postRank': gen,
        'carveout_sidelists': sidelists,
        'Tickers_df': tickers,
        'ntopxlsx': 20,
        'ntopagg': 100,
    }


def _read(path):
    with open(path, 'r', newline='', encoding='utf-8') as f:
        return f.read()


def test_format_columns_lists_ranks():
    """Correct header, all six lists tagged, ranks 1-based within each list, company filled."""
    resdic = _fake_resdic()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'pick_log.csv')
        plog.write_pick_log(resdic, as_of='2026-07-14', path=path,
                            logged_at='T1', filter_commit='abc1234')
        df = pd.read_csv(path)

    assert list(df.columns) == plog.PICK_LOG_COLUMNS, df.columns.tolist()
    assert set(df['list']) == {'GENERAL', 'REIT', 'MINING', 'FIN1', 'FIN2', 'FIN3'}, set(df['list'])
    # GENERAL depth == ntopxlsx (20), even though the frame had 25 names.
    assert (df['list'] == 'GENERAL').sum() == 20, (df['list'] == 'GENERAL').sum()
    # ranks are 1-based and contiguous within each list.
    for lst, grp in df.groupby('list'):
        ranks = list(grp['rank'])
        assert ranks == list(range(1, len(ranks) + 1)), (lst, ranks)
    # company came through the Tickers_df name map.
    gen1 = df[(df['list'] == 'GENERAL') & (df['rank'] == 1)].iloc[0]
    assert gen1['ticker'] == 'GEN0' and gen1['company'] == 'General Co 0', gen1.to_dict()
    # run-level stamps present on every row.
    assert set(df['as_of']) == {'2026-07-14'} and set(df['filter_commit']) == {'abc1234'}
    print(f"  [ok] format: cols + 6 lists + 1-based ranks + company ({len(df)} rows)")


def test_append_only_does_not_mutate():
    """Writing twice must APPEND -- earlier rows byte-identical, header once, count grows."""
    resdic = _fake_resdic()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'pick_log.csv')
        n1 = plog.write_pick_log(resdic, as_of='2026-07-14', path=path,
                                 logged_at='T1', filter_commit='c1')
        first_bytes = _read(path)
        rows1 = first_bytes.count('\n')

        n2 = plog.write_pick_log(resdic, as_of='2026-07-15', path=path,
                                 logged_at='T2', filter_commit='c2')
        second_bytes = _read(path)

        # the entire first write is an exact PREFIX of the file after the second write.
        assert second_bytes.startswith(first_bytes), "earlier rows were mutated/reordered!"
        # strictly more rows.
        rows2 = second_bytes.count('\n')
        assert rows2 == rows1 + n2, (rows1, rows2, n2)
        assert rows2 > rows1, (rows1, rows2)
        # header appears exactly once.
        header = ','.join(plog.PICK_LOG_COLUMNS)
        assert second_bytes.count(header) == 1, second_bytes.count(header)
        assert n1 == n2, (n1, n2)
    print(f"  [ok] append-only: 1st write is exact prefix; header once; rows {rows1}->{rows2}")


def test_rerun_same_as_of_appends_new_block():
    """Re-running the SAME as_of appends a NEW block with a distinct logged_at, old intact."""
    resdic = _fake_resdic()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'pick_log.csv')
        plog.write_pick_log(resdic, as_of='2026-07-14', path=path,
                            logged_at='RUN_A', filter_commit='c1')
        first_bytes = _read(path)
        plog.write_pick_log(resdic, as_of='2026-07-14', path=path,   # SAME as_of
                            logged_at='RUN_B', filter_commit='c1')
        second_bytes = _read(path)
        df = pd.read_csv(path)

    # the first block is untouched (exact prefix) even on a same-as_of re-run.
    assert second_bytes.startswith(first_bytes), "re-run mutated the earlier same-as_of block!"
    # both blocks present for the same as_of, distinguished by logged_at.
    same = df[df['as_of'] == '2026-07-14']
    assert set(same['logged_at']) == {'RUN_A', 'RUN_B'}, set(same['logged_at'])
    # the earlier block was not lost: still a full RUN_A block.
    assert (same['logged_at'] == 'RUN_A').sum() == (same['logged_at'] == 'RUN_B').sum()
    print("  [ok] re-run same as_of -> new block (RUN_A + RUN_B), old block intact")


def test_missing_sidelist_is_skipped_not_crash():
    """A None/absent cohort side-list is skipped (logged), never crashes; GENERAL still logs."""
    resdic = _fake_resdic()
    resdic['carveout_sidelists']['REIT'] = None          # degenerate cohort
    del resdic['carveout_sidelists'][plog._co.FIN3_BALSHEET]  # missing entirely
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'pick_log.csv')
        plog.write_pick_log(resdic, as_of='2026-07-14', path=path,
                            logged_at='T1', filter_commit='c1')
        df = pd.read_csv(path)
    lists = set(df['list'])
    assert 'GENERAL' in lists and 'REIT' not in lists and 'FIN3' not in lists, lists
    assert {'MINING', 'FIN1', 'FIN2'} <= lists, lists
    print("  [ok] missing/None side-list skipped, GENERAL + present cohorts still logged")


def test_non_cp1252_company_name_survives():
    """A non-cp1252 issuer name (non-US universe) must round-trip -- utf-8 encoding on the
    file open. Before the fix, Windows' cp1252 default would raise UnicodeEncodeError here
    and abort the whole stage. The CJK/accented chars below are outside cp1252."""
    resdic = _fake_resdic()
    # override the top general name with non-cp1252 characters
    tdf = resdic['Tickers_df'].copy()
    tdf.loc[tdf['symbol'] == 'GEN0', 'name'] = 'Nippon 株式 Ǆ Co'   # CJK + U+01C4, not cp1252
    resdic['Tickers_df'] = tdf
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'pick_log.csv')
        plog.write_pick_log(resdic, as_of='2026-07-14', path=path,
                            logged_at='T1', filter_commit='c1')
        df = pd.read_csv(path, encoding='utf-8')
    row = df[(df['list'] == 'GENERAL') & (df['rank'] == 1)].iloc[0]
    assert row['company'] == 'Nippon 株式 Ǆ Co', repr(row['company'])
    print("  [ok] non-cp1252 company name round-trips (utf-8)")


def test_empty_general_warns_loudly():
    """A GENERAL frame missing its 'source' column must NOT silently log zero rows -- it
    must emit a loud !!! banner on stdout while side-lists still log."""
    import contextlib
    import io
    resdic = _fake_resdic()
    resdic['postRank'] = pd.DataFrame({'notsource': [1, 2, 3]})   # malformed general frame
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'pick_log.csv')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            plog.write_pick_log(resdic, as_of='2026-07-14', path=path,
                                logged_at='T1', filter_commit='c1')
        out = buf.getvalue()
        df = pd.read_csv(path)
    assert 'GENERAL SHORTLIST LOGGED ZERO ROWS' in out, out[-500:]
    assert 'GENERAL' not in set(df['list']), set(df['list'])
    # side-lists still logged despite the empty general block.
    assert {'REIT', 'MINING', 'FIN1', 'FIN2', 'FIN3'} <= set(df['list']), set(df['list'])
    print("  [ok] empty GENERAL frame -> loud banner, side-lists still logged")


def test_truncated_prior_row_gets_newline_padded():
    """A prior file whose last byte is not a newline (crash-truncated) must be newline-padded
    before the new block, so rows can't merge -- and the prior bytes stay a prefix."""
    resdic = _fake_resdic()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'pick_log.csv')
        # simulate a crash-truncated prior run: header + a partial row, no trailing newline.
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(','.join(plog.PICK_LOG_COLUMNS) + '\n')
            f.write('2026-07-13,T0,c0,GENERAL,1,AAA,Alpha')   # NO trailing newline
        prior = _read(path)
        plog.write_pick_log(resdic, as_of='2026-07-14', path=path,
                            logged_at='T1', filter_commit='c1')
        after = _read(path)
    # prior bytes preserved verbatim as a prefix; a newline now separates the partial row.
    assert after.startswith(prior), "prior (truncated) bytes were altered!"
    assert after[len(prior)] == '\n', repr(after[len(prior):len(prior) + 20])
    # the new block's first data row is on its own line, not merged onto 'Alpha'.
    assert 'Alpha2026-07-14' not in after, "new block merged onto the truncated row!"
    print("  [ok] crash-truncated prior row is newline-padded, no row merge")


def test_git_hash_never_raises():
    h = plog._git_short_hash()
    assert isinstance(h, str) and h, h
    print(f"  [ok] _git_short_hash() -> {h!r} (never raises)")


def test_sbocker_wiring_present():
    """Sbocker.main must actually CALL the pick-log writer as LIVE code (AST-verified, so a
    commented-out / doc-only mention cannot pass)."""
    import Sbocker
    src = textwrap.dedent(inspect.getsource(Sbocker.main))
    tree = ast.parse(src)
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                called.add(f.attr)
            elif isinstance(f, ast.Name):
                called.add(f.id)
    assert 'run_pick_log_stage' in called, \
        "Sbocker.main does not CALL pick_log.run_pick_log_stage (AST)"
    print("  [ok] Sbocker.main CALLS run_pick_log_stage (AST)")


if __name__ == '__main__':
    print("Prospective pick-log self-checks")
    test_format_columns_lists_ranks()
    test_append_only_does_not_mutate()
    test_rerun_same_as_of_appends_new_block()
    test_missing_sidelist_is_skipped_not_crash()
    test_non_cp1252_company_name_survives()
    test_empty_general_warns_loudly()
    test_truncated_prior_row_gets_newline_padded()
    test_git_hash_never_raises()
    test_sbocker_wiring_present()
    print("ALL PICK-LOG SELF-CHECKS PASSED")
