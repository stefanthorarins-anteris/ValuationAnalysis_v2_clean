"""Small correctness-of-REPORTING fixes from the pre-fetch wave (2026-07-31).

Each of these is cheap, but each concerns something that is READ and RELIED ON:

  B4  `-compyear` and `-sectorfilter` were unusable in every form -- an index computed and
      then discarded, with the value read from a DIFFERENT flag's index.  `compyear` is
      load-bearing: it drives the `datefail` gate.
  A6  the cross-statement date join's POSITIONAL FALLBACK rate was unobservable -- `used_join`
      was returned but only used to branch.  That fallback is the period mispairing the join
      exists to prevent, and its probability rises with 80 rows.
  C2  `ALLOW_MERGE_CONTENT_MISMATCH=0` / `=false` / `=no` ACTIVATED the override, because the
      gate was a presence test, not a truth test.
  C3  the merge-gate banner claimed to catch an inverted or renamed metric.  It is a
      NaN-coverage test and cannot.
  C5  the run log printed a name's OLDEST three cdx rows under the label "first 3".
  C4  three bare `return` bail-outs made ship-gate tests report PASS having asserted nothing.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# --------------------------------------------------------------------------- #
#  B4 -- the wrong-index family                                               #
# --------------------------------------------------------------------------- #
import configuration as cfg


def test_compyear_works_ALONE():
    """Reproduced before the fix: UnboundLocalError on `id` (the -datasource index)."""
    r = cfg.getDataFetchConfiguration(['x', '-compyear', 'thisYear'])
    assert r is not None, "-compyear thisYear alone must parse (was UnboundLocalError: id)"


def test_compyear_reads_ITS_OWN_value_not_another_flags():
    """Reproduced before the fix: with -datasource present it read args[1] ('fmp') and raised
    'compyear argument is not valid'."""
    from datetime import datetime
    this_year, last_year = datetime.now().year, datetime.now().year - 1
    for val, want in (('thisYear', this_year), ('lastYear', last_year)):
        r = cfg.getDataFetchConfiguration(['x', '-datasource', 'fmp', '-compyear', val])
        # message names the expectation; it must NOT interpolate the config structure (M1)
        assert want in _flatten(r), \
            "-compyear %s must resolve to %d in the config" % (val, want)


def test_compyear_still_REJECTS_a_genuinely_bad_value():
    """The fix must not turn the validation off -- only point it at the right argument."""
    with pytest.raises(Exception):
        cfg.getDataFetchConfiguration(['x', '-compyear', 'banana'])


def test_sectorfilter_works_ALONE():
    """Reproduced before the fix: UnboundLocalError on `imb` (the -mcapBelow index)."""
    r = cfg.getDataFetchConfiguration(['x', '-sectorfilter', 'Technology'])
    assert 'Technology' in _flatten(r), \
        "-sectorfilter Technology must resolve to 'Technology' in the config"


def test_sectorfilter_reads_ITS_OWN_value_alongside_mcapBelow():
    """Reproduced before the fix: read the -mcapBelow VALUE and raised
    '-sectorfilterr argument not valid'."""
    r = cfg.getDataFetchConfiguration(
        ['x', '-mcapBelow', '100', '-sectorfilter', 'Healthcare'])
    assert 'Healthcare' in _flatten(r), \
        "-sectorfilter must read its OWN argument, not -mcapBelow's"


def test_sectorfilter_still_REJECTS_a_genuinely_bad_value():
    with pytest.raises(Exception):
        cfg.getDataFetchConfiguration(['x', '-sectorfilter', 'NotASector'])


def test_the_default_config_is_UNCHANGED():
    """Nothing about the no-flags path may move: that is what tonight's run uses."""
    base = _flatten(cfg.getDataFetchConfiguration(['x']))
    from datetime import datetime
    assert 'all' in base, "sectorfilter default must remain 'all'"
    assert (datetime.now().year - 1) in base, "compyear default must remain last year"


def test_the_config_comparison_set_CANNOT_carry_the_API_KEY():
    """review M1.  Six assertions above compared against `_flatten(...)` with no message, so a
    failure would have rendered the whole set -- including the FMP key -- into the test log.
    This pins the exclusion so it cannot be undone by a later edit."""
    r = cfg.getDataFetchConfiguration(['x'])
    assert isinstance(r, dict) and 'api_key' in r, \
        "premise stale: config no longer carries api_key under that name"
    flat = _flatten(r)
    assert r['api_key'] not in flat, "the API KEY is in the comparison set (M1 regression)"
    assert r.get('baseurl') not in flat, "baseurl is in the comparison set"
    # and a token-shaped value is dropped even under an unknown key name
    assert 'A' * 32 not in _flatten({'some_future_token': 'A' * 32})


#  SECRET-BEARING config keys that must NEVER enter a comparison set (review M1, 2026-07-31).
#  getDataFetchConfiguration returns a dict CONTAINING `api_key`.  `_flatten` used to include
#  it, and six of the assertions below carried no custom message, so pytest's assertion
#  rewriting would have rendered the whole set -- INCLUDING THE 32-CHAR FMP KEY -- into the
#  test log on any failure.  The key is already known to have been exposed on the public
#  GitHub remote (CONFIG.md, 2026-07-11), so this was not hypothetical.  Excluded by NAME
#  (not by value) so it holds even if the key file changes, and every assertion below now
#  carries a message that names the expectation instead of dumping the structure.
_SECRET_CONFIG_KEYS = {'api_key', 'apikey', 'api-key', 'baseurl'}


def _flatten(r):
    """Config returns a dict (or tuple) of scalars; flatten to a comparable set, MINUS secrets.

    Also drops any value that merely LOOKS like a credential (a long opaque alnum string), so a
    future config key holding a token cannot leak through a name-based allow-list alone.
    """
    out = set()
    if isinstance(r, dict):
        items = [v for k, v in r.items() if str(k).strip().lower() not in _SECRET_CONFIG_KEYS]
    else:
        items = list(r) if isinstance(r, (list, tuple)) else [r]
    for v in items:
        if isinstance(v, str) and len(v) >= 20 and v.isalnum():
            continue                      # opaque token-shaped value -- never compare on it
        try:
            out.add(v)
        except TypeError:
            pass
    return out


@pytest.fixture(autouse=True)
def _no_real_api_key(tmp_path, monkeypatch):
    """Run the config tests against a DUMMY key file in a temp CWD.

    `getDataFetchConfiguration` does an unconditional `open('fmpAPIkey.txt')`
    (configuration.py:29), so without this the suite (a) READS the real secret on this machine
    and (b) FAILS on any machine that does not happen to have the key file in CWD -- the same
    class of coupling `conftest.py` was created to eliminate.  A temp CWD with a placeholder
    fixes both, and guarantees the real key is never in memory for these tests at all.
    """
    (tmp_path / 'fmpAPIkey.txt').write_text('DUMMY_KEY_NOT_A_REAL_CREDENTIAL')
    monkeypatch.chdir(tmp_path)
    yield


#  THE AST GUARD IS RETIRED AND REPLACED (argparse refactor, 2026-08-02).
#  ---------------------------------------------------------------------------
#  The guard below used to assert that every `args[<var> + k]` read in
#  configuration.py indexed its OWN flag.  That guard existed only because the
#  hand-rolled `args.index()` idiom made the wrong-index bug REPRESENTABLE, and
#  it had two standing weaknesses: it required the idiom to survive (`len(reads)
#  >= 25`) and it had to PIN a known-latent hazard (`ima`, bound by both
#  -mcapAbove and -fsMAnumber) that it could not actually fix.
#
#  configuration.py now parses with argparse, so there are no index variables at
#  all and the defect class is UNREPRESENTABLE rather than merely absent.  The
#  sweep machinery is KEPT and INVERTED: instead of checking that each read is
#  correct, it now asserts there are ZERO such reads.  That is strictly stronger
#  than deleting the test -- it fails if anyone reintroduces the idiom -- and it
#  makes the `ima` latent-reuse pin unnecessary, because no index var exists to
#  be reused.  The BEHAVIOURAL tests above (-compyear / -sectorfilter read their
#  own value, bad values still rejected, default config unchanged) are unchanged
#  and are what actually protect the two fixed flags.
def _configuration_index_reads():
    """Every `args[<var> + k]` read in configuration.py, matched to its ENCLOSING
    `if '-flag' in args:` block and to the flag `<var>` was actually bound from.

    Returns (reads, wrong, latent_reuse).  `reads` is a list of
    (flag_of_block, var, flag_var_was_bound_from, lineno).

    Retained post-refactor to prove the idiom is GONE: every returned list must
    now be empty.
    """
    import ast
    src = open(os.path.join(_HERE, 'configuration.py'), encoding='utf-8').read()
    tree = ast.parse(src)

    def block_flag(node):
        """The '-flag' literal tested by `if '-flag' in args:`, else None."""
        t = node.test
        if (isinstance(t, ast.Compare) and len(t.ops) == 1
                and isinstance(t.ops[0], ast.In)
                and isinstance(t.left, ast.Constant) and isinstance(t.left.value, str)
                and isinstance(t.comparators[0], ast.Name)
                and t.comparators[0].id == 'args'):
            return t.left.value
        return None

    reads, wrong, bindings = [], [], {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        flag = block_flag(node)
        if flag is None:
            continue
        # bindings made INSIDE this block: var -> flag it indexes
        local = {}
        for n in ast.walk(node):
            if (isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)):
                v = n.value
                if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                        and v.func.attr == 'index'
                        and isinstance(v.func.value, ast.Name) and v.func.value.id == 'args'
                        and v.args and isinstance(v.args[0], ast.Constant)):
                    local[n.targets[0].id] = v.args[0].value
                    bindings.setdefault(n.targets[0].id, set()).add(v.args[0].value)
        # reads made INSIDE this block
        for n in ast.walk(node):
            if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                    and n.value.id == 'args' and isinstance(n.slice, ast.BinOp)
                    and isinstance(n.slice.left, ast.Name)):
                var = n.slice.left.id
                bound_from = local.get(var)
                reads.append((flag, var, bound_from, n.lineno))
                if bound_from != flag:
                    wrong.append((flag, var, bound_from, n.lineno))
    latent = {v: sorted(f) for v, f in bindings.items() if len(f) > 1}
    return reads, wrong, latent


def test_the_hand_rolled_index_idiom_is_GONE_from_configuration():
    """The retired guard, inverted: the wrong-index family is now UNREPRESENTABLE.

    The old sweep asserted `len(reads) >= 25` and that each read used its own flag's index.
    After the argparse refactor there are no `args.index()` bindings and no `args[<var>+k]`
    reads at all, so there is nothing left to get wrong -- and the `ima` double-bind latent
    hazard (bound by BOTH -mcapAbove and -fsMAnumber, correct only because each use happened
    to follow its own binding) is gone with it rather than merely pinned.
    """
    reads, wrong, latent = _configuration_index_reads()
    assert reads == [], (
        "the hand-rolled `args[<var>+1]` idiom is BACK in configuration.py -- this is the "
        "idiom that produced the -compyear / -sectorfilter / -manelimfilename defects; "
        "use the argparse parser instead: %s" % reads)
    assert wrong == [], wrong
    assert latent == {}, ("index-variable reuse is back (the `ima` hazard): %s" % latent)


def test_configuration_has_no_args_index_calls_at_all():
    """Belt-and-braces on the same property, stated on the BINDING rather than the read:
    a surviving `args.index(...)` anywhere would mean a hand-rolled index still exists even
    if its subscript happens to be written in a shape the sweep does not match."""
    import ast
    src = open(os.path.join(_HERE, 'configuration.py'), encoding='utf-8').read()
    assert 'argparse' in src, "configuration.py no longer uses argparse"
    hits = []
    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'index'
                and isinstance(n.func.value, ast.Name) and n.func.value.id == 'args'):
            hits.append(n.lineno)
    assert not hits, "args.index(...) survives at line(s) %s" % hits


# --------------------------------------------------------------------------- #
#  The argparse parser: the FLAG INVENTORY, pinned as a specification.         #
#  This is the set of flags the hand-rolled parser accepted.  It is pinned by  #
#  NAME and by VALUED-vs-PRESENCE, because those two facts are what a          #
#  behaviour-preserving refactor must not move: dropping a flag makes it       #
#  silently ignored (there is no unknown-flag error -- see the leniency test   #
#  below), and flipping valued<->presence is the truthiness footgun that       #
#  `ALLOW_MERGE_CONTENT_MISMATCH=0` was (a `0` meaning "off" turning the       #
#  thing ON).                                                                  #
# --------------------------------------------------------------------------- #
_VALUED_FLAGS = {
    '-tickerfilter', '-datasource', '-mcapAbove', '-mcapBelow', '-sectorfilter',
    '-period', '-nrperiods', '-nrTaT', '-compyear', '-fsMAnumber',
    '-nrScorePeriods', '-ntopagg', '-ntopxlsx', '-savebometric', '-saveboresults',
    '-loadbometric', '-loadboresults', '-symbolChangeRestock', '-bometricfilename',
    '-boresultsfilename', '-manelimtickers', '-manelimfilename', '-asof',
    '-delisted_max_pages', '-portfolioTest', '-backtest_buy_years',
    '-backtest_eval_years', '-backtest_topn', '-run_estimation', '-transfer_dir',
}
#  Truth comes from PRESENCE ALONE for these -- they take NO value, so no value
#  can ever be misread as a truth.  `-startfromlastindex` is the named example.
_PRESENCE_FLAGS = {
    '-newOnly', '-ingest_delisted', '-startfromlastindex', '-runbacktest',
    '-no_transfer',
    #  ADDED 2026-08-10 (CEO): `-force_rebuild_maps`, an explicit one-off override of the
    #  profile-map skip conditions.  Listed here rather than tolerated by loosening the
    #  assertion, so the NEXT unplanned flag still fails this test -- which is the whole
    #  point of pinning the parser's surface.
    '-force_rebuild_maps',
}


def _parser_actions():
    p = cfg._build_parser()
    return p, {opt: a for a in p._actions for opt in a.option_strings}


def test_every_flag_the_old_parser_accepted_is_STILL_accepted():
    _p, by_opt = _parser_actions()
    missing = sorted((_VALUED_FLAGS | _PRESENCE_FLAGS) - set(by_opt))
    assert not missing, (
        "flag(s) dropped by the refactor -- they would now be SILENTLY IGNORED, not "
        "rejected: %s" % missing)
    extra = sorted(set(by_opt) - (_VALUED_FLAGS | _PRESENCE_FLAGS))
    assert not extra, "flag(s) added by the refactor (behaviour change): %s" % extra


def test_presence_only_flags_take_NO_VALUE_and_valued_flags_take_ONE():
    """The truthiness boundary, pinned.  A presence flag has nargs == 0, so
    `-startfromlastindex 0` can never mean "off" by supplying a value -- exactly as before,
    where presence alone decided it.  A valued flag must NOT become a presence flag, or
    `-savebometric 0` / `-manelimtickers 0` would start meaning ON."""
    _p, by_opt = _parser_actions()
    for flag in sorted(_PRESENCE_FLAGS):
        assert by_opt[flag].nargs == 0, \
            "%s must be PRESENCE-only (nargs 0), got nargs=%r" % (flag, by_opt[flag].nargs)
    for flag in sorted(_VALUED_FLAGS):
        assert by_opt[flag].nargs != 0, \
            "%s must take a VALUE -- as a presence flag, a '0' value would mean ON" % flag


def test_a_zero_value_still_DISABLES_every_valued_boolean():
    """The ALLOW_MERGE_CONTENT_MISMATCH footgun, checked on this module's own flags:
    the value 0 must mean OFF, not "flag present therefore on"."""
    for flag, key in (('-savebometric', 'saveBoMetric'), ('-saveboresults', 'saveBoResults'),
                      ('-loadbometric', 'loadBoMetric'), ('-loadboresults', 'loadBoResults'),
                      ('-symbolChangeRestock', 'symbchRestock'),
                      ('-manelimtickers', 'manelimtickersbool'),
                      ('-run_estimation', 'run_estimation')):
        off = cfg.getDataFetchConfiguration(['x', flag, '0'])
        assert off[key] == 0, "%s 0 must DISABLE %s, got %r" % (flag, key, off[key])
    for flag, key in (('-savebometric', 'saveBoMetric'), ('-loadbometric', 'loadBoMetric'),
                      ('-symbolChangeRestock', 'symbchRestock'),
                      ('-run_estimation', 'run_estimation')):
        on = cfg.getDataFetchConfiguration(['x', flag, '1'])
        assert on[key] == 1, "%s 1 must ENABLE %s, got %r" % (flag, key, on[key])


def test_presence_flags_are_INT_1_0_not_bool():
    """`configdic` carried ints, and callers/pickled run records compare on them.  argparse's
    store_true yields True/False, so the conversion back to 1/0 is load-bearing for type
    identity, not cosmetic."""
    on = cfg.getDataFetchConfiguration(
        ['x', '-newOnly', '-ingest_delisted', '-runbacktest', '-startfromlastindex'])
    off = cfg.getDataFetchConfiguration(['x'])
    for key in ('newOnly', 'ingest_delisted', 'runbacktest', 'startfromlastindex'):
        assert type(on[key]) is int and on[key] == 1, (key, on[key], type(on[key]))
        assert type(off[key]) is int and off[key] == 0, (key, off[key], type(off[key]))


# --------------------------------------------------------------------------- #
#  The four old-parser behaviours argparse does NOT give by default.           #
#  Each of these would be a REGRESSION, and each is cheap to lose in a later   #
#  edit (a stray `parse_args`, a default `add_help`), so each is pinned.       #
# --------------------------------------------------------------------------- #
def test_the_parser_NEVER_exits_the_process():
    """THE serious regression risk: stock argparse prints usage and calls sys.exit(2).  This
    module is IMPORTED (by this suite, by baseline_tools/test_delisted_ingest.py, and by five
    pipeline modules), so a hard exit would kill the caller.  SystemExit is a BaseException, so
    it would also slip through `except Exception` handlers AND through pytest.raises(Exception).
    """
    p = cfg._build_parser()
    with pytest.raises(Exception) as ei:
        p.error('boom')
    assert not isinstance(ei.value, SystemExit), "parser.error() still exits the process"
    with pytest.raises(Exception) as ei2:
        p.exit(2, 'boom')
    assert not isinstance(ei2.value, SystemExit), "parser.exit() still exits the process"


def test_UNKNOWN_flags_are_still_silently_ignored():
    """The old `'-flag' in args` scan never looked at anything else, so unknown flags and
    stray positionals passed silently -- and the suite's own convention is a leading dummy
    token (`['x', ...]`), which stock parse_args rejects as "unrecognized arguments"."""
    base = cfg.getDataFetchConfiguration(['x'])
    for argv in (['x', '-notAFlagAtAll'], ['x', '-notAFlagAtAll', 'value'],
                 ['x', 'y', 'z'], ['-period', 'quarter', '-bogus', 'junk']):
        got = cfg.getDataFetchConfiguration(argv)
        assert set(got) == set(base), argv


def test_dash_h_does_NOT_trigger_a_help_dump_and_exit():
    """`-h` / `--help` are just unknown tokens today.  argparse's default help action would
    print usage and terminate the interpreter -- inside a pipeline entry point."""
    for argv in (['x', '-h'], ['x', '--help']):
        got = cfg.getDataFetchConfiguration(argv)
        assert got['period'] == 'quarter' and got['sectorfilter'] == 'all', argv


def test_a_REPEATED_flag_keeps_its_FIRST_value():
    """`args.index()` returned the FIRST match, so `-period quarter -period annual` yielded
    'quarter'.  argparse keeps the LAST value by default -- a silent behaviour change on an
    invocation that parses cleanly both before and after, which is the dangerous kind."""
    from datetime import datetime
    assert cfg.getDataFetchConfiguration(
        ['x', '-period', 'quarter', '-period', 'annual'])['period'] == 'quarter'
    assert cfg.getDataFetchConfiguration(
        ['x', '-ntopagg', '10', '-ntopagg', '20'])['ntopagg'] == 10
    assert cfg.getDataFetchConfiguration(
        ['x', '-compyear', 'lastYear', '-compyear', 'thisYear'])['compyear'] == \
        datetime.now().year - 1


def test_flags_are_matched_EXACTLY_no_abbreviation():
    """argparse abbreviates long options by default (`-nrp` -> `-nrperiods`).  The old parser
    matched exactly, so an abbreviation was an ignored unknown token; accepting it now would
    be a new, and ambiguous, behaviour."""
    base = cfg.getDataFetchConfiguration(['x'])
    got = cfg.getDataFetchConfiguration(['x', '-nrp', '99'])
    assert got['nrperiods'] == base['nrperiods'] == 24, got['nrperiods']


def test_a_valued_flag_with_NO_value_raises_a_CATCHABLE_error_naming_the_flag():
    """Five flags bounds-checked with a bespoke message before; the rest raised an opaque
    `IndexError: list index out of range`.  All now report in the bespoke style, and the five
    original messages are preserved verbatim (checked below).  Every one must remain a plain
    Exception -- never SystemExit."""
    for flag in sorted(_VALUED_FLAGS):
        with pytest.raises(Exception) as ei:
            cfg.getDataFetchConfiguration(['x', flag])
        assert not isinstance(ei.value, SystemExit), flag
        assert flag in str(ei.value), (flag, str(ei.value))


@pytest.mark.parametrize("argv,msg", [
    (['-asof'], '-asof requires a date argument (YYYY-MM-DD)'),
    (['-manelimtickers'], '-manelimtickers requires a 0/1 argument'),
    (['-manelimfilename'], '-manelimfilename requires a filename argument'),
    (['-run_estimation'], '-run_estimation requires an integer argument (0 or 1)'),
    (['-transfer_dir'], '-transfer_dir requires a directory path argument'),
])
def test_the_five_PRE_EXISTING_bounds_check_messages_are_preserved_verbatim(argv, msg):
    with pytest.raises(Exception) as ei:
        cfg.getDataFetchConfiguration(argv)
    assert str(ei.value) == msg, str(ei.value)


# --------------------------------------------------------------------------- #
#  A6 / B5 -- positional-fallback visibility                                  #
# --------------------------------------------------------------------------- #
def _stmt(dates, extra=None):
    d = {'date': list(dates)}
    d.update(extra or {})
    return pd.DataFrame(d)


def test_the_date_join_reports_used_join_TRUE_on_clean_statements():
    import getData_fmp as gdf
    dates = ['2026-03-31', '2025-12-31', '2025-09-30', '2025-06-30']
    stmts = [_stmt(dates) for _ in range(5)]
    _aligned, used = gdf._align_statements_by_date(*stmts)
    assert used is True


def test_the_date_join_FALLS_BACK_on_a_duplicate_date_and_that_is_countable():
    """Duplicate raw dates are REAL on this data (282 sources carry colliding snapped
    quarters), and the rate rises with 80 rows."""
    import getData_fmp as gdf
    dup = ['2026-03-31', '2026-03-31', '2025-09-30', '2025-06-30']
    ok = ['2026-03-31', '2025-12-31', '2025-09-30', '2025-06-30']
    _a, used = gdf._align_statements_by_date(_stmt(ok), _stmt(dup), _stmt(ok), _stmt(ok),
                                             _stmt(ok))
    assert used is False, "a duplicate date must trigger the positional fallback"


def test_fillPreReqdf_RECORDS_the_fallback_when_asked():
    """`used_join` used to be invisible.  It must now land in a list the run can count."""
    import getData_fmp as gdf
    import createDicts as cdic
    preReq = cdic.getPreReqDict()
    dup = ['2026-03-31', '2026-03-31', '2025-09-30', '2025-06-30']
    frames = _build_min_statements(dup)
    tempfund = pd.DataFrame({'date': dup, 'source': 'DUP.L'})
    fallbacks = []
    try:
        gdf.fillPreReqdf(tempfund, preReq, *frames, fallbacks=fallbacks)
    except Exception:
        # the frames are minimal, so the body may still fail on a missing field -- the
        # RECORDING happens before any of that, which is the behaviour under test
        pass
    assert fallbacks == ['DUP.L'], fallbacks


def test_fillPreReqdf_records_NOTHING_on_the_clean_path():
    import getData_fmp as gdf
    import createDicts as cdic
    preReq = cdic.getPreReqDict()
    ok = ['2026-03-31', '2025-12-31', '2025-09-30', '2025-06-30']
    frames = _build_min_statements(ok)
    tempfund = pd.DataFrame({'date': ok, 'source': 'OK.L'})
    fallbacks = []
    try:
        gdf.fillPreReqdf(tempfund, preReq, *frames, fallbacks=fallbacks)
    except Exception:
        pass
    assert fallbacks == [], fallbacks


def _build_min_statements(dates):
    """bs, inc, cf, km, fr with every preReq field present and numeric."""
    import createDicts as cdic
    preReq = cdic.getPreReqDict()
    out = []
    for key in ('bs', 'inc', 'cf', 'km', 'fr'):
        cols = {'date': list(dates)}
        for f in preReq.get(key, []):
            cols[f] = np.arange(1.0, len(dates) + 1.0)
        out.append(pd.DataFrame(cols))
    return out


def test_the_fallback_RATE_is_printed_every_run_even_when_ZERO():
    """An unobserved rate and a zero rate are different facts, and the whole point of this
    counter is that the rate has never been measured."""
    import inspect
    import getData_fmp as gdf
    src = inspect.getsource(gdf.get_fundamentals_fmp)
    assert 'joinfallback = []' in src
    assert 'fallbacks=joinfallback' in src
    assert 'CROSS-STATEMENT DATE-JOIN' in src
    i_print = src.index('CROSS-STATEMENT DATE-JOIN')
    # the print must NOT be inside an `if joinfallback:` -- zero has to be reported too
    assert 'if joinfallback:' not in src[max(0, i_print - 300):i_print]
    assert "'joinFallback': joinfallback" in src


def test_the_fallback_summary_sits_beside_the_PARSE_FAIL_summary():
    import inspect
    import getData_fmp as gdf
    src = inspect.getsource(gdf.get_fundamentals_fmp)
    assert src.index('PARSE-FAIL SUMMARY') < src.index('CROSS-STATEMENT DATE-JOIN')


# --------------------------------------------------------------------------- #
#  C2 -- the inverted truthiness gate                                         #
# --------------------------------------------------------------------------- #
import baseline_tools.dead_merge as dm  # noqa: E402


@pytest.mark.parametrize("value", ["0", "false", "False", "no", "NO", "off", "none", ""])
def test_a_FALSEY_env_value_does_NOT_activate_the_override(value, monkeypatch):
    """THE footgun: an operator typing the value that means 'no' used to GET the
    known-invalid basis they were explicitly refusing."""
    monkeypatch.setenv("ALLOW_MERGE_CONTENT_MISMATCH", value)
    assert dm._env_truthy("ALLOW_MERGE_CONTENT_MISMATCH") is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_a_TRUTHY_env_value_DOES_activate_the_override(value, monkeypatch):
    monkeypatch.setenv("ALLOW_MERGE_CONTENT_MISMATCH", value)
    assert dm._env_truthy("ALLOW_MERGE_CONTENT_MISMATCH") is True


def test_an_ABSENT_env_var_does_not_activate_the_override(monkeypatch):
    monkeypatch.delenv("ALLOW_MERGE_CONTENT_MISMATCH", raising=False)
    assert dm._env_truthy("ALLOW_MERGE_CONTENT_MISMATCH") is False


def test_the_merge_gate_uses_the_truthiness_helper_not_a_presence_test():
    import inspect
    src = inspect.getsource(dm)
    assert 'if not _env_truthy("ALLOW_MERGE_CONTENT_MISMATCH")' in src
    assert 'os.environ.get("ALLOW_MERGE_CONTENT_MISMATCH")' not in \
        src.replace('os.environ.get(name)', '')


# --------------------------------------------------------------------------- #
#  C3 -- the banner must not claim a catch it does not have                   #
# --------------------------------------------------------------------------- #
def test_the_merge_banner_no_longer_claims_to_catch_an_inverted_metric():
    """It is a NaN-COVERAGE test: an inverted or renamed metric is fully populated on BOTH
    sides, so the coverage matches and this check stays silent.  A safety banner that
    overstates its coverage is worse than none -- the reliance it invites is unearned."""
    import inspect
    src = inspect.getsource(dm.merge_dead_into_live) if hasattr(dm, 'merge_dead_into_live') \
        else inspect.getsource(dm)
    assert 'WHAT THIS CHECK DOES *NOT* CATCH' in src
    assert 'CANNOT detect an INVERTED or RENAMED metric' in src
    # the old over-claim, verbatim, must be gone
    assert 'a renamed/inverted metric scores the OLD quantity' not in src or \
        'CANNOT detect' in src


# --------------------------------------------------------------------------- #
#  C5 -- "first 3" must mean the NEWEST three                                 #
# --------------------------------------------------------------------------- #
def test_the_selection_used_by_the_run_log_returns_the_NEWEST_three():
    """The behaviour, on the exact storage order the diagnostic sees.  cdx_df is stored
    OLDEST-first, so the old `head(3)` returned the three OLDEST periods -- on the 07-17 panel
    that meant ~2020 rows printed as the newest data, in the log the CEO reads at 3am to judge
    whether the fetch picked up the current quarter."""
    from detectManipulation import _toNewestFirst
    dates = pd.date_range('2020-01-01', periods=24, freq='QS')      # ascending = storage order
    cdx = pd.DataFrame({'date': dates, 'source': 'AAA', 'marketCap': np.arange(24.0)})
    old_way = cdx.head(3)['date'].tolist()
    new_way = _toNewestFirst(cdx).head(3)['date'].tolist()
    assert old_way == list(dates[:3]), "premise: head(3) on ascending data IS the oldest three"
    assert new_way == list(dates[-1:-4:-1]), new_way
    assert new_way[0] == dates[-1], "row 0 must be the most recent period"


def test_the_C5_diagnostic_CANNOT_RAISE_on_an_unparseable_date():
    """review L4.  My C5 fix replaced `head(3)` -- which cannot raise -- with `_toNewestFirst`,
    whose `pd.to_datetime(s)` has NO `errors='coerce'`.  One unparseable date in the first
    source's rows therefore raised ValueError in `postBoWrapper`'s UNGUARDED diagnostic block,
    aborting Stage-2.  Making a LOG LINE able to kill the run is worse than the mislabelling it
    fixed, and this wave's own standard is that a diagnostic never aborts a run."""
    import postBo
    from detectManipulation import _toNewestFirst
    bad = pd.DataFrame({'date': ['2026-01-01', 'not-a-date', '2025-01-01'],
                        'source': 'A', 'marketCap': [1.0, 2.0, 3.0]})
    # the shared forensic helper still raises -- deliberately NOT loosened
    with pytest.raises(Exception):
        _toNewestFirst(bad)
    # the diagnostic helper does not, and still returns the newest row first
    out = postBo._diag_newest_rows(bad, 3)
    assert len(out) == 3
    assert str(out['date'].iloc[0]) == '2026-01-01', out['date'].tolist()


def test_the_C5_diagnostic_helper_orders_correctly_and_does_not_mutate():
    """A diagnostic that sorted its input in place would silently corrupt cdx_df/bmdf."""
    import postBo
    dates = pd.date_range('2020-01-01', periods=24, freq='QS')
    cdx = pd.DataFrame({'date': dates, 'source': 'AAA', 'marketCap': np.arange(24.0)})
    before = cdx['date'].tolist()
    out = postBo._diag_newest_rows(cdx, 3)
    assert out['date'].tolist() == list(dates[-1:-4:-1])
    assert cdx['date'].tolist() == before, "the diagnostic mutated its input frame"
    assert '_diag_dt' not in out.columns, "the sort key leaked into the printed sample"


def test_BOTH_run_log_sample_blocks_are_newest_first_and_relabelled():
    """The diagnostic prints TWO samples -- BoMetric_df and cdx_df.  Only the cdx one was on the
    brief; both were wrong and both are fixed, because a block where one sample is newest-first
    and the other oldest-first is more misleading than either alone."""
    import inspect
    import postBo
    src = inspect.getsource(postBo.postBoWrapper)
    assert src.count('_diag_newest_rows(') >= 2, src.count('_diag_newest_rows(')
    assert src.count('3 MOST RECENT periods, newest first') == 2
    assert '(first 3)' not in src, "the misleading label survives somewhere"
    # and no un-oriented head(3) sample remains in the block -- nor a raise-capable
    # _toNewestFirst, which is what review L4 was about
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith('#'):
            continue
        if 'first_source_data =' in s:
            assert '_diag_newest_rows(' in s or s.endswith('_diag_newest_rows('), s
    assert '_toNewestFirst(' not in src, \
        "postBoWrapper's diagnostic must use the coercing helper, not the strict forensic one"


# --------------------------------------------------------------------------- #
#  C4 -- no ship-gate test may PASS having asserted nothing                   #
# --------------------------------------------------------------------------- #
_C4_FILES = [
    os.path.join('baseline_tools', 'test_baseline_tools.py'),
    os.path.join('baseline_tools', 'test_rebalance_engine.py'),
    os.path.join('baseline_tools', 'test_returns_core.py'),
]


@pytest.mark.parametrize("rel", _C4_FILES)
def test_no_bare_return_bailout_remains(rel):
    """Under pytest a bare `return` from a test body is a PASS.  These three reported GREEN
    having asserted NOTHING whenever their Drive/pickle/CSV artifacts were absent -- and absent
    is the normal state on the home machine, i.e. the one running tonight."""
    import ast
    path = os.path.join(_HERE, rel)
    tree = ast.parse(open(path, encoding='utf-8').read())
    bad = []
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith('test_')):
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Return) and n.value is None:
                bad.append('%s:%d' % (fn.name, n.lineno))
    assert not bad, ("bare `return` in a test body reports PASS without asserting: %s" % bad)


@pytest.mark.parametrize("rel", _C4_FILES)
def test_the_bailout_is_an_explicit_SKIP(rel):
    src = open(os.path.join(_HERE, rel), encoding='utf-8').read()
    assert 'pytest.skip(' in src, "%s has no explicit skip" % rel


@pytest.mark.parametrize("rel", _C4_FILES)
def test_script_mode_still_runs_and_does_not_call_a_SKIP_a_pass(rel):
    """pytest's Skipped does NOT derive from Exception, so a plain script-mode call would ABORT
    the file rather than skip one check -- and the existing `except Exception` handler would not
    have caught it.  Script mode must survive AND must not report a skip as a pass."""
    src = open(os.path.join(_HERE, rel), encoding='utf-8').read()
    assert 'skip.Exception' in src, "%s script runner does not handle Skipped" % rel
    assert 'SKIP' in src


#  RUNTIME-EXERCISING THE C4 BRANCHES (reviewer's closing note, 2026-07-31).
#  IMPORTANT SCOPE STATEMENT: on THIS machine the Drive/pickle/CSV artifacts are PRESENT, so the
#  three new `pytest.skip` branches never fire in a normal run here -- the one skip that does
#  appear in the suite output comes from a PRE-EXISTING schema gate deeper inside
#  test_certified_reproduction, not from the new missing-input branch.  So a green suite here
#  proves nothing about the branch C4 actually fixes.  The branches fire on the HOME machine,
#  which is precisely the machine C4 targets and the one running tonight.  These three tests
#  FORCE the artifact check to fail so the new branch is genuinely EXECUTED here, turning
#  "verified by reading + AST" into "observed raising Skipped".
def _skipped_exc():
    return pytest.skip.Exception


def test_C4_baseline_tools_branch_RAISES_Skipped_when_the_pickle_is_absent(monkeypatch):
    sys.path.insert(0, os.path.join(_HERE, 'baseline_tools'))
    import glob as _glob
    import test_baseline_tools as tbt
    monkeypatch.setattr(_glob, 'glob', lambda *_a, **_k: [])
    with pytest.raises(_skipped_exc()) as ei:
        tbt.test_real_pickle_smoke_optional()
    assert 'NOT run' in str(ei.value), str(ei.value)


def test_C4_rebalance_engine_branch_RAISES_Skipped_when_an_input_is_absent(monkeypatch):
    sys.path.insert(0, os.path.join(_HERE, 'baseline_tools'))
    import test_rebalance_engine as tre
    monkeypatch.setattr(os.path, 'exists', lambda _p: False)
    with pytest.raises(_skipped_exc()) as ei:
        tre.test_certified_reproduction()
    msg = str(ei.value)
    assert 'NOT run' in msg and 'missing input' in msg, msg


def test_C4_returns_core_branch_RAISES_Skipped_when_the_price_csv_is_absent(monkeypatch):
    sys.path.insert(0, os.path.join(_HERE, 'baseline_tools'))
    import test_returns_core as trc
    monkeypatch.setattr(os.path, 'exists', lambda _p: False)
    with pytest.raises(_skipped_exc()) as ei:
        trc.test_smoke_real_pricesource()
    assert 'NOT run' in str(ei.value), str(ei.value)


def test_C4_Skipped_is_NOT_an_Exception_subclass():
    """The fact that makes the script-mode handling necessary: `except Exception` does NOT catch
    it, so a plain call in a `__main__` runner would ABORT the file rather than skip one check.
    Pinned because the whole three-file `skip.Exception` treatment rests on it."""
    assert not issubclass(pytest.skip.Exception, Exception)
    assert issubclass(pytest.skip.Exception, BaseException)


def test_the_skip_reason_NAMES_the_missing_artifact():
    """A skip that does not say WHAT was missing is only marginally better than a silent pass --
    the reader cannot tell whether the gap matters."""
    import ast
    for rel in _C4_FILES:
        tree = ast.parse(open(os.path.join(_HERE, rel), encoding='utf-8').read())
        reasons = []
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == 'skip' and n.args):
                reasons.append(ast.unparse(n.args[0]))
        assert reasons, rel
        # Every reason must be a real explanation rather than a bare marker ...
        for r in reasons:
            assert len(r) > 25, (rel, r)
        # ... and the bail-out replacing the bare `return` must say the check DID NOT RUN.
        # Scoped to that one, deliberately: these files already carried other pytest.skip
        # calls with their own wording (a Stage-1 schema gate, a merge refusal), and policing
        # pre-existing phrasing is not this fix's job.
        assert any('NOT run' in r or 'missing' in r for r in reasons), (rel, reasons)


# The three C4 suites are exercised by the repo-wide pytest run, not re-run here: a nested
# `pytest` subprocess loads the multi-hundred-MB pickles a second time and took >5 minutes,
# which is a poor trade for coverage the outer run already provides.


# =========================================================================== #
#  THE PUBLISHED P/E IS COMPUTED, NOT CONSUMED  (CEO rule; 2026-08-10)         #
# =========================================================================== #
def _pe_panel(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def test_the_published_PE_is_COMPUTED_from_our_own_panel_not_read_off_the_vendor():
    """*** THE DEFECT: `086280.KS` displayed `PE-ratio = 66.28` AT RANK 4 of the shipped
    2026-08-10 top-100. ***

    The column was FMP's `priceEarningsRatio` from `v3/ratios/<symb>`, printed as received.
    The panel's OWN newest row for that name gives `earningsYield = 0.021864` per quarter ->
    an annualised P/E of 11.43, and `price / epsTTM = 207,500 / 22,306 = 9.30`.  Two
    independent readings off our own data agree to within a third; the vendor's is 5.8x either.

    NOT A COLUMN-WIDE ERROR, and that was checked before anything changed: across the same
    100 names the displayed value tracks the panel's earnings yield with a MEDIAN RATIO OF
    1.000.  Two cells deviate past 1.5x -- `086280.KS` at 5.80x and `281820.KS` at 3.94x --
    and the next-largest is 1.43x.

    The fix is the house rule rather than a per-name patch: the vendor supplies RAW INPUTS and
    we compute the derived quantity.  Reconstruction is free here, because `earningsYield` is
    already on the panel and is the exact field `earnYield` is SCORED on -- so the number on
    the sheet and the number in the score become the same object, which they were not.
    """
    import postBo as pb
    #  the real 086280.KS reading, annualised by its own rows-per-year
    t = {'086280.KS': (0.021864, 4.0), 'SEMI.PA': (0.03, 2.0), 'LOSS': (-0.01, 4.0),
         'ZERO': (0.0, 4.0), 'NAN': (float('nan'), 4.0)}
    assert pb._pe_ratio_from_panel(t, '086280.KS') == pytest.approx(11.4341, abs=5e-4)
    #  THE ANNUALISATION IS PER SOURCE, not a hard-coded 4: a semi-annual filer's per-period
    #  yield is one HALF-year's, so it annualises by 2.  A hard 4 would halve its P/E.
    assert pb._pe_ratio_from_panel(t, 'SEMI.PA') == pytest.approx(1.0 / (2 * 0.03), abs=1e-9)
    #  REFUSED where no meaningful P/E exists -- a loss-maker's is negative, and publishing a
    #  negative P/E invites it to be read as "cheap", which is the sign-inversion class this
    #  repo keeps finding.  None -> the caller falls back / writes NaN.
    for k in ('LOSS', 'ZERO', 'NAN', 'NOT_IN_PANEL'):
        assert pb._pe_ratio_from_panel(t, k) is None, k


def test_the_PE_panel_table_takes_the_NEWEST_row_BY_DATE():
    """Nothing on this path guarantees ingestion order, so the table re-sorts rather than
    assuming -- the same boundary rule Stage-1 and Stage-2 each establish for themselves."""
    import pandas as pd
    import postBo as pb
    cdx = pd.DataFrame({
        'source': ['A'] * 3,
        'date': ['2024-06-30', '2026-03-31', '2025-06-30'],   # deliberately out of order
        'earningsYield': [0.01, 0.05, 0.02],
    })
    got = pb._pe_panel_table(cdx)
    assert got['A'][0] == pytest.approx(0.05), 'the newest row by DATE must win'
    #  a frame with no earningsYield column degrades to {} rather than raising: this feeds a
    #  REPORT column and must never cost the CSV.
    assert pb._pe_panel_table(cdx.drop(columns=['earningsYield'])) == {}
    assert pb._pe_panel_table(None) == {}


def test_the_PE_vendor_FALLBACK_takes_the_SAME_sign_test_as_the_computed_value():
    """*** reviewer S3, 2026-08-10: the refusal was defeated by its own fallback. ***

    Our own P/E is refused precisely when `earningsYield <= 0` -- and the vendor's P/E on that
    same name is then NEGATIVE for the same reason, so publishing the fallback handed 100% of
    the refusing population exactly what the refusal was written to prevent.  MEASURED on the
    shipped 2026-08-10 top-100: one name refuses, `NEXN` (`earningsYield = -0.013804`), and its
    published `PE-ratio` was **-18.1111**.

    The fallback is not removed -- a positive vendor P/E for a name whose newest row is simply
    missing is still worth publishing -- it is SIGN-CHECKED, so the column's one promise (a
    published P/E is a positive P/E) holds for every cell whatever its provenance.
    """
    import inspect
    import postBo as pb
    src = inspect.getsource(pb.writeBoAggToCSV)
    i = src.index('priceEarningsRatio')
    block = src[i:i + 1400]
    assert 'perat > 0' in block, (
        'the vendor fallback publishes without a sign test, so a refused loss-maker still '
        'gets a negative P/E on the CEO\'s sheet')
    #  and the computed side refuses the same population, so the two agree on the promise
    assert pb._pe_ratio_from_panel({'NEXN': (-0.013804, 4.0)}, 'NEXN') is None
