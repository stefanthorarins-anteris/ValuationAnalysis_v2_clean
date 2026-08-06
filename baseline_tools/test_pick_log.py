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
        # Universe stamp as Sbocker puts it on resdic (universe_provenance_for_run); the writer
        # reads THESE keys, never configdic.
        'universe': 'stock_NA1_EU1',
        'universe_fingerprint': 'aaaa1111bbbb',
    }


#  The EXACT header of a pick_log.csv written before 2026-08-04, i.e. before the universe columns
#  existed (copied from the quarantined test-universe log). Hardcoded on purpose: derived from
#  PICK_LOG_COLUMNS it would silently follow a future schema change and stop being the regression
#  fixture it is meant to be.
_PRE_UNIVERSE_HEADER = [
    'as_of', 'logged_at', 'filter_commit', 'list', 'rank', 'ticker', 'company', 'aggscore',
    'reporting_currency', 'entry_periodend_price_reporting_ccy',
    'entry_periodend_trailing_PE', 'entry_periodend_PB_fmp_basis',
    'entry_periodend_grahamNumberToPrice', 'entry_industry_median_periodend_PE',
    'entry_industry_median_n']


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


def test_universe_provenance_on_every_row():
    """H-5: every row carries the universe NAME and FINGERPRINT, POPULATED -- not blank, not
    'unknown'. This is the only artifact that cannot be regenerated, so an unstamped row is a
    permanent provenance hole."""
    resdic = _fake_resdic()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'pick_log.csv')
        plog.write_pick_log(resdic, as_of='2026-08-05', path=path,
                            logged_at='T1', filter_commit='c1')
        df = pd.read_csv(path)

    assert 'universe' in df.columns and 'universe_fingerprint' in df.columns, df.columns.tolist()
    assert set(df['universe']) == {'stock_NA1_EU1'}, set(df['universe'])
    assert set(df['universe_fingerprint']) == {'aaaa1111bbbb'}, set(df['universe_fingerprint'])
    # populated on EVERY row of EVERY list, and not a placeholder.
    assert df['universe_fingerprint'].notna().all() and (df['universe_fingerprint'] != '').all()
    assert not df['universe_fingerprint'].astype(str).str.startswith('unknown').any()
    assert len(set(df['list'])) == 6, set(df['list'])
    print(f"  [ok] universe + fingerprint populated on all {len(df)} rows (6 lists)")


def test_test_universe_row_distinguishable_by_fingerprint():
    """A test-universe pick must be tellable from a production pick BY FINGERPRINT -- including
    the case the fingerprint exists for: the SAME name denoting a different universe (the
    2026-08-02 European restoration moved stock_NA1_EU1 by 1,046 names)."""
    prod = _fake_resdic()                                   # stock_NA1_EU1 / aaaa1111bbbb
    test_u = _fake_resdic()
    test_u['universe'] = 'stock_TEST1'
    test_u['universe_fingerprint'] = '6f8b8825dc90'         # real curated-test fingerprint
    renamed = _fake_resdic()                                # SAME name, different definition
    renamed['universe_fingerprint'] = 'cccc3333dddd'

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'pick_log.csv')
        plog.write_pick_log(prod, as_of='2026-08-05', path=path,
                            logged_at='PROD', filter_commit='c1')
        plog.write_pick_log(test_u, as_of='2026-08-06', path=path,
                            logged_at='TEST', filter_commit='c1')
        plog.write_pick_log(renamed, as_of='2026-08-07', path=path,
                            logged_at='RESTORED', filter_commit='c1')
        df = pd.read_csv(path)

    by_run = df.groupby('logged_at')[['universe', 'universe_fingerprint']].first()
    # test-universe block is separable from production.
    assert by_run.loc['TEST', 'universe_fingerprint'] == '6f8b8825dc90'
    assert by_run.loc['PROD', 'universe_fingerprint'] == 'aaaa1111bbbb'
    # THE case the fingerprint exists for: name is IDENTICAL, universe is NOT.
    assert by_run.loc['PROD', 'universe'] == by_run.loc['RESTORED', 'universe']
    assert (by_run.loc['PROD', 'universe_fingerprint']
            != by_run.loc['RESTORED', 'universe_fingerprint'])
    # so grouping by NAME alone would have merged two different pools; by fingerprint it does not.
    assert df['universe'].nunique() == 2 and df['universe_fingerprint'].nunique() == 3
    print("  [ok] test vs production separable by fingerprint; same-name/different-definition too")


def test_missing_universe_stamp_warns_loudly_and_logs_unknown():
    """A resdic with NO universe stamp must NOT log a blank (which reads as 'not applicable'):
    it logs an explicit unknown and shouts, because the rows are permanent."""
    import contextlib
    import io
    resdic = _fake_resdic()
    del resdic['universe']
    del resdic['universe_fingerprint']
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'pick_log.csv')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            plog.write_pick_log(resdic, as_of='2026-08-05', path=path,
                                logged_at='T1', filter_commit='c1')
        out = buf.getvalue()
        df = pd.read_csv(path)
    assert 'NO UNIVERSE FINGERPRINT' in out, out[-600:]
    assert set(df['universe']) == {plog.UNKNOWN_UNIVERSE}, set(df['universe'])
    assert set(df['universe_fingerprint']) == {plog.UNKNOWN_UNIVERSE_FINGERPRINT}
    # the picks themselves are still recorded -- an honest unknown beats no forward record.
    assert len(df) > 20 and 'GENERAL' in set(df['list'])
    print("  [ok] unstamped resdic -> loud banner + explicit unknown (rows still logged)")


def test_old_schema_log_is_refused_not_migrated():
    """A pre-2026-08-04 (no-universe-columns) pick_log.csv must be REFUSED, not appended to and
    not migrated: the old bytes stay identical, no row is added, and the message says what to do."""
    resdic = _fake_resdic()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'pick_log.csv')
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(','.join(_PRE_UNIVERSE_HEADER) + '\n')
            f.write('2026-08-04,T0,c0,GENERAL,1,OLY.TO,Olympia,0.61,CAD,119.0,16.3,6.2,0.46,,\n')
        before = _read(path)

        try:
            plog.write_pick_log(resdic, as_of='2026-08-05', path=path,
                                logged_at='T1', filter_commit='c1')
            raise AssertionError("old-schema log was APPENDED TO instead of refused!")
        except RuntimeError as e:
            msg = str(e)

        after = _read(path)

    assert after == before, "the old-schema file was MODIFIED -- it must be left untouched!"
    assert 'SCHEMA DRIFT' in msg, msg
    # names the mismatch concretely...
    assert "'universe'" in msg and "'universe_fingerprint'" in msg, msg
    assert '15' in msg and str(len(plog.PICK_LOG_COLUMNS)) in msg, msg
    # ...and tells the operator what to do, without suggesting a hand-edit.
    assert 'FIX:' in msg and 'move the existing log aside' in msg, msg
    assert 'Do NOT hand-edit' in msg, msg
    print("  [ok] old-schema log REFUSED, file byte-identical, message actionable")


def test_old_schema_refusal_surfaces_through_the_guarded_stage():
    """The refusal must actually REACH the operator: the pipeline stage swallows exceptions so a
    pick-log problem can't kill the run, so it has to print the loud banner and return None --
    otherwise the refusal would be silent, which is worse than the corruption it prevents."""
    import contextlib
    import io
    resdic = _fake_resdic()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'pick_log.csv')
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(','.join(_PRE_UNIVERSE_HEADER) + '\n')
        before = _read(path)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rv = plog.run_pick_log_stage(resdic, as_of='2026-08-05', path=path)
        out = buf.getvalue()
        after = _read(path)
    assert rv is None, rv
    assert after == before, "the guarded stage still modified the old-schema file!"
    assert 'PICK-LOG STAGE FAILED' in out, out[-800:]
    assert 'SCHEMA DRIFT' in out, out[-800:]
    print("  [ok] refusal surfaces via the guarded stage (loud banner, returns None, run lives)")


def test_unreadable_header_is_refused():
    """A non-empty file from which no header row can be read must also be refused -- an
    unestablishable on-disk schema is the same permanent mis-alignment risk as a known-wrong one."""
    resdic = _fake_resdic()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'pick_log.csv')
        with open(path, 'wb') as f:
            f.write(b'\n')          # non-empty, but no parseable header row
        before = _read(path)
        try:
            plog.write_pick_log(resdic, as_of='2026-08-05', path=path,
                                logged_at='T1', filter_commit='c1')
            raise AssertionError("a file with no readable header was appended to!")
        except RuntimeError as e:
            msg = str(e)
        after = _read(path)
    assert after == before, "the unreadable file was MODIFIED!"
    assert 'HEADER UNREADABLE' in msg and 'FIX:' in msg, msg
    print("  [ok] unreadable/absent header on a non-empty file is refused")


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


#  --- universe-definition drift (WARN ONLY, CEO 2026-08-06) -------------------------------
#  These exist because the FAILURE MODE OF A DRIFT DETECTOR IS SILENCE. The first draft of
#  `check_fingerprint_drift` referenced `pd` -- pick_log has no module-level pandas import --
#  and the NameError was swallowed by a broad `except Exception: return None`, so it reported
#  "no drift" on every run while never actually working. `test_drift_detected_when_definition_
#  changes` is the guard against that exact regression: it asserts the detector FIRES, not
#  merely that it does not crash.
def _drift_log(tmpdir, rows):
    """Write a minimal pick-log with just the three columns the drift check reads."""
    import csv as _csv
    p = os.path.join(tmpdir, 'pick_log.csv')
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=['as_of', 'universe', 'universe_fingerprint'])
        w.writeheader()
        for r in rows:
            w.writerow(dict(zip(['as_of', 'universe', 'universe_fingerprint'], r)))
    return p


def test_drift_detected_when_definition_changes():
    td = tempfile.mkdtemp()
    p = _drift_log(td, [('2026-07-01', 'stock_CUR3K', 'aaaa1111bbbb'),
                        ('2026-07-15', 'stock_CUR3K', 'aaaa1111bbbb')])
    rec = plog.check_fingerprint_drift('stock_CUR3K', 'ffff9999eeee', p)
    assert rec is not None, (
        "DRIFT NOT DETECTED. A recorded fingerprint of aaaa1111bbbb and a current one of "
        "ffff9999eeee is a definition change, and this is the whole point of the check. A "
        "None here means the detector is silently inert -- the regression this test exists "
        "for. Check for a swallowed exception inside check_fingerprint_drift.")
    assert rec['previous_fingerprint'] == 'aaaa1111bbbb'
    assert rec['current_fingerprint'] == 'ffff9999eeee'
    assert rec['previous_as_of'] == '2026-07-15', "must compare against the LATEST prior stamp"
    assert rec['previous_rows_logged'] == 2
    #  WARN ONLY: it must RECORD and change nothing else.
    assert 'WARN ONLY' in rec['action_taken']
    notes = [f for f in os.listdir(td) if f.startswith('UniverseDefinitionDrift')]
    assert notes, "a detected drift must leave a dated record, not only a console banner"
    print("  drift detected + recorded (%s)" % notes[0])


def test_no_drift_when_fingerprint_unchanged():
    td = tempfile.mkdtemp()
    p = _drift_log(td, [('2026-07-01', 'stock_CUR3K', 'aaaa1111bbbb')])
    assert plog.check_fingerprint_drift('stock_CUR3K', 'aaaa1111bbbb', p) is None
    #  A quiet run must not litter a drift note.
    assert not [f for f in os.listdir(td) if f.startswith('UniverseDefinitionDrift')]
    print("  unchanged fingerprint stays silent")


def test_drift_check_is_silent_on_first_run_and_legacy_logs():
    td = tempfile.mkdtemp()
    #  A name never logged before is not drift.
    p = _drift_log(td, [('2026-07-01', 'stock_NA1_EU1', 'cccc2222dddd')])
    assert plog.check_fingerprint_drift('stock_CUR3K', 'ffff9999eeee', p) is None
    #  No log at all.
    assert plog.check_fingerprint_drift(
        'stock_CUR3K', 'ffff9999eeee', os.path.join(td, 'absent.csv')) is None
    #  A pre-provenance log (no fingerprint column) is expected, not an error.
    legacy = os.path.join(td, 'legacy.csv')
    with open(legacy, 'w', encoding='utf-8') as f:
        f.write('as_of,universe\n2026-01-01,stock_CUR3K\n')
    assert plog.check_fingerprint_drift('stock_CUR3K', 'ffff9999eeee', legacy) is None
    #  An UNSTAMPED current run is already warned about elsewhere; do not double-warn.
    p2 = _drift_log(td, [('2026-07-01', 'stock_CUR3K', 'aaaa1111bbbb')])
    assert plog.check_fingerprint_drift(
        'stock_CUR3K', plog.UNKNOWN_UNIVERSE_FINGERPRINT, p2) is None
    print("  first-run / no-log / legacy-log / unstamped all silent")


def test_drift_check_covers_sample_rates_and_must_include():
    """The RESIDUAL this closes: a definition change that leaves the member COUNT unmoved.

    Not a test of pick_log -- a test that the fingerprint it compares actually keys off the
    three things the CEO asked to be hashed. If `definition_fingerprint` ever stops covering
    sample rates or must-include, the warn path would still "work" while being blind to the
    only cases the wallclock guard and cohort-sum pin do not already catch.
    """
    import universes as un
    base = un.definition_fingerprint('stock_CUR3K')
    assert isinstance(base, str) and base, "stock_CUR3K must fingerprint"
    ent = un._entry('stock_CUR3K')
    assert un.sample_rates('stock_CUR3K'), (
        "stock_CUR3K is the sampled universe; if it has no sample rates the fingerprint "
        "cannot be covering them")
    #  Perturb each leg of the basis in turn and require the hash to move. The entry's
    #  `sample` is a dict and `must_include` a tuple, so both legs are perturbed by REBINDING
    #  the key and restored from a snapshot in `finally` -- the registry is module-global and
    #  a leaked mutation would corrupt every later test in the session.
    _first_code = sorted(ent['sample'])[0]
    for leg, key, newval in (
            ('sample rate', 'sample',
             {**ent['sample'], _first_code: ent['sample'][_first_code] + 1}),
            ('must-include', 'must_include',
             tuple(ent['must_include']) + ('ZZZZ.TEST',))):
        before = un.definition_fingerprint('stock_CUR3K')
        snap = ent[key]
        try:
            ent[key] = newval
            after = un.definition_fingerprint('stock_CUR3K')
            assert after != before, (
                "changing the %s did NOT move the fingerprint -- the drift check is blind "
                "to it, which is exactly the residual it was added to close." % leg)
        finally:
            ent[key] = snap
        assert un.definition_fingerprint('stock_CUR3K') == base, "restore failed"
        print("  fingerprint moves on a %s change" % leg)


if __name__ == '__main__':
    print("Prospective pick-log self-checks")
    test_drift_detected_when_definition_changes()
    test_no_drift_when_fingerprint_unchanged()
    test_drift_check_is_silent_on_first_run_and_legacy_logs()
    test_drift_check_covers_sample_rates_and_must_include()
    test_format_columns_lists_ranks()
    test_append_only_does_not_mutate()
    test_rerun_same_as_of_appends_new_block()
    test_missing_sidelist_is_skipped_not_crash()
    test_non_cp1252_company_name_survives()
    test_empty_general_warns_loudly()
    test_truncated_prior_row_gets_newline_padded()
    test_universe_provenance_on_every_row()
    test_test_universe_row_distinguishable_by_fingerprint()
    test_missing_universe_stamp_warns_loudly_and_logs_unknown()
    test_old_schema_log_is_refused_not_migrated()
    test_old_schema_refusal_surfaces_through_the_guarded_stage()
    test_unreadable_header_is_refused()
    test_git_hash_never_raises()
    test_sbocker_wiring_present()
    print("ALL PICK-LOG SELF-CHECKS PASSED")
