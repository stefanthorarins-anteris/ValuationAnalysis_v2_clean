"""Tests for the universe registry, the EURONEXT/OSE fix, and the curated TEST universe.

WHOLLY OFFLINE.  Not one test here touches the network: the exchange-code facts were
verified by live call on 2026-08-02 and RECORDED in `universes._VERIFIED_COUNTS`, and
these tests pin the recorded facts and the code that consumes them.  That distinction is
deliberate and worth stating plainly -- a green suite here means "the pipeline applies the
codes we verified", NOT "the codes are still correct today".  Re-verification is a live
operation (2 calls: `available-traded/list` and `financial-statement-symbol-lists`,
intersected, `type == 'stock'`, counted by `exchangeShortName`); the counts in
`_VERIFIED_COUNTS` are what to diff against.
"""

import os
import re

import numpy as np
import pandas as pd
import pytest

import configuration as cfg
import getData_gen as gdg
import universes as un
import utils
import verify_test_universe as vtu


# --------------------------------------------------------------------------- #
#  REGISTRY INTEGRITY                                                          #
# --------------------------------------------------------------------------- #
def test_every_universe_has_exactly_one_membership_basis():
    """exchanges XOR symbols XOR every_exchange.  Two bases set at once would make the
    wrapper's branch order decide the universe, which is how a definition becomes
    accidental."""
    for name, d in un.UNIVERSES.items():
        bases = [d['exchanges'] is not None, d['symbols'] is not None,
                 bool(d['every_exchange'])]
        assert sum(bases) == 1, '%s has %d membership bases, expected exactly 1' % (
            name, sum(bases))


def test_the_dead_exchange_codes_are_GONE_from_every_universe():
    """THE REGRESSION GUARD FOR THE ORIGINAL DEFECT.

    `EURONEXT` and `OSE` are not FMP exchangeShortName values.  Each matched ZERO rows,
    so four universes silently under-delivered and 1,046 statement-bearing common stocks
    were never fetched by any run.  Nothing in the code could notice: a filter matching
    nothing is indistinguishable from a filter over a small exchange.  This test is what
    makes their reintroduction impossible to do quietly.
    """
    for name, d in un.UNIVERSES.items():
        codes = set(d['exchanges'] or ())
        for dead in un.DEAD_CODES:
            assert dead not in codes, (
                '%s references the DEAD code %r, which matches zero FMP rows. '
                'Its real equivalents are %s.' % (name, dead, un.DEAD_CODES[dead]))


def test_every_wired_exchange_code_was_verified_to_match_live_rows():
    """No universe may reference a code that is absent from the verified-count table, or
    one whose verified count is zero -- that is precisely the EURONEXT/OSE shape."""
    for name, d in un.UNIVERSES.items():
        for code in (d['exchanges'] or ()):
            assert code in un._VERIFIED_COUNTS, (
                '%s references %r, which was never verified against the live exchange '
                'list. Verify it before wiring it.' % (name, code))
            assert un._VERIFIED_COUNTS[code] > 0, (
                '%s references %r, verified to match 0 statement-bearing stocks'
                % (name, code))


def test_the_dead_codes_replacements_all_carry_real_members():
    """The five codes the fix wired in must each be non-empty, and must sum to the
    1,046-name restoration the fix claims."""
    restored = ('PAR', 'AMS', 'BRU', 'LIS', 'OSL')
    for c in restored:
        assert un._VERIFIED_COUNTS[c] > 0
    assert sum(un._VERIFIED_COUNTS[c] for c in restored) == 1046


def test_each_label_states_the_count_its_definition_actually_implies():
    """The member count in a label is a factual claim and drifts silently otherwise --
    it did: stock_WW1_TV's label said 9,978 against a definition summing to 9,900.

    The labels state the PRE-FILTER definition sum, and say so, because the RESOLVED
    count is smaller (the instrument filter, delisted prune and sector filter all cut
    afterwards) and letting a pre-filter number read as a resolved one is exactly the
    kind of loose figure this project has been burned by.
    """
    checked = 0
    for name, d in un.UNIVERSES.items():
        exp = un.expected_count(name)
        if exp is None:
            continue                      # the FULL universe has no per-code sum
        if d['symbols'] is not None:
            #  An explicit-list universe is not exchange-defined; its label states BOTH
            #  the listed and the post-filter count, which is checked separately by
            #  test_the_test_universe_states_both_its_listed_and_its_effective_size.
            assert str(len(d['symbols'])) in d['label']
            continue
        m = re.search(r'\(([\d,]+) pre-filter\)', d['label'])
        assert m is not None, (
            '%s label must state its count as "(N pre-filter)" so the number cannot be '
            'read as a resolved member count: %r' % (name, d['label']))
        stated = int(m.group(1).replace(',', ''))
        assert stated == exp, ('%s label states %d, definition sums to %d'
                               % (name, stated, exp))
        checked += 1
    assert checked >= 6, 'expected to check every exchange-defined universe'


def test_fingerprints_are_distinct_and_stable():
    fps = {n: un.definition_fingerprint(n) for n in un.names()}
    assert len(set(fps.values())) == len(fps), 'fingerprint collision: %s' % fps
    #  PINNED.  A change here is a universe-definition change and must be a deliberate
    #  edit of this test, not a silent drift -- that is the whole point of the stamp.
    assert fps['stock_NA1_EU1'] == '43d60094b156'
    assert fps['stock_TEST1'] == '6f8b8825dc90'


def test_fingerprint_is_order_insensitive_but_membership_sensitive():
    d = un.UNIVERSES['stock_NA1']
    codes = d['exchanges']
    try:
        un.UNIVERSES['_probe'] = dict(d, exchanges=tuple(reversed(codes)))
        assert un.definition_fingerprint('_probe') == un.definition_fingerprint('stock_NA1')
        un.UNIVERSES['_probe'] = dict(d, exchanges=codes + ('AMEX',))
        assert un.definition_fingerprint('_probe') != un.definition_fingerprint('stock_NA1')
    finally:
        un.UNIVERSES.pop('_probe', None)


# --------------------------------------------------------------------------- #
#  CONTINUITY -- nothing on disk may orphan                                    #
# --------------------------------------------------------------------------- #
HISTORICAL_NAMES = ('stock_NA1', 'stock_US1', 'stock_WW1_TV', 'stock_NA1_EU1',
                    'stock_US1_EU1', 'stock_US1_EU2')


def test_every_historical_filter_name_still_resolves():
    """Artifact filenames (`Bometric_dic-fmp_<name>_*`) and resume files embed the NAME.
    Dropping or renaming one orphans every artifact that carries it."""
    for name in HISTORICAL_NAMES:
        assert name in un.UNIVERSES
        assert un.exchanges(name), '%s must still be exchange-defined' % name
        cfgd = cfg.getDataFetchConfiguration(['x', '-tickerfilter', name])
        assert cfgd['tickerfilter'] == name


def test_the_default_universe_is_unchanged():
    """It used to be `tickerfilterlist[3]` -- pinned by LIST POSITION, so any insertion
    would have moved it.  It is now named; this pins the value it names."""
    assert un.DEFAULT_UNIVERSE == 'stock_NA1_EU1'
    assert cfg.getDataFetchConfiguration([])['tickerfilter'] == 'stock_NA1_EU1'


def test_configuration_and_the_registry_agree_on_the_valid_names():
    """The two lists that used to drift.  A name accepted by the CLI but unknown to the
    wrapper fell through to `df = tickdf` and returned the whole ~50,000-name table."""
    with pytest.raises(Exception):
        cfg.getDataFetchConfiguration(['x', '-tickerfilter', 'stock_NOPE'])
    for name in un.names():
        assert cfg.getDataFetchConfiguration(
            ['x', '-tickerfilter', name])['tickerfilter'] == name


def test_resume_filename_format_is_unchanged_and_every_universe_is_resumable():
    """The format must stay `lastIndexOfRead_<ds>_<filter>.txt` (files exist on disk),
    and the whitelist must cover every universe -- stock_US1_EU1 and stock_US1_EU2 were
    CLI-valid but raised 'Not Implemented' on resume."""
    assert un.resume_filename('stock_NA1_EU1') == 'lastIndexOfRead_fmp_stock_NA1_EU1.txt'
    for name in un.names():
        assert un.resume_filename(name) in un.resume_filenames('fmp')
    for name in ('stock_US1_EU1', 'stock_US1_EU2'):
        assert un.resume_filename(name) in un.resume_filenames('fmp'), (
            '%s was previously unresumable; it must not regress' % name)


def test_every_resume_file_ALREADY_ON_DISK_is_still_accepted(tmp_path, monkeypatch):
    """The no-orphaning check, run against the real files in the repo rather than a
    list of names someone remembered to update."""
    here = os.path.dirname(os.path.abspath(un.__file__))
    on_disk = [f for f in os.listdir(here) if f.startswith('lastIndexOfRead_')]
    allowed = set(un.resume_filenames('fmp'))
    orphaned = [f for f in on_disk if f not in allowed]
    assert not orphaned, 'resume file(s) on disk no longer accepted: %s' % orphaned
    #  and the accept path actually works, in a scratch cwd so nothing is written here
    monkeypatch.chdir(tmp_path)
    for f in on_disk:
        (tmp_path / f).write_text('7')
        assert utils.get_lastIndexRead(f) == 7


def test_an_unknown_resume_filename_still_raises_with_a_useful_message():
    with pytest.raises(Exception) as e:
        utils.get_lastIndexRead('lastIndexOfRead_fmp_stock_MADEUP.txt')
    assert 'Not Implemented' in str(e.value)


# --------------------------------------------------------------------------- #
#  tickerfilterWrapper -- offline, against a fixture shaped like the real       #
#  available-traded/list INTERSECT statement-symbol-list table.                 #
# --------------------------------------------------------------------------- #
#  Symbols and suffixes are real ones from the live 2026-08-02 list, so the suffix
#  assertions below are assertions about FMP's actual convention.
FIXTURE_ROWS = [
    ('AAPL',      'Apple Inc.',                 'NASDAQ', 'stock'),
    ('BRK-A',     'Berkshire Hathaway Inc.',    'NYSE',   'stock'),
    ('TCL-A.TO',  'Transcontinental Inc.',      'TSX',    'stock'),
    ('SNR.L',     'Senior plc',                 'LSE',    'stock'),
    ('LIN.DE',    'Linde plc',                  'XETRA',  'stock'),
    ('BOL.ST',    'Boliden AB (publ)',          'STO',    'stock'),
    ('BRIM.IC',   'Brim hf.',                   'ICE',    'stock'),
    ('ALLUX.PA',  'Installux S.A.',             'PAR',    'stock'),
    ('NEDAP.AS',  'Nedap N.V.',                 'AMS',    'stock'),
    ('CAMB.BR',   'Campine N.V.',               'BRU',    'stock'),
    ('COR.LS',    'Corticeira Amorim',          'LIS',    'stock'),
    ('GYL.OL',    'Gyldendal ASA',              'OSL',    'stock'),
    ('7203.T',    'Toyota Motor Corporation',   'JPX',    'stock'),
    ('005930.KS', 'Samsung Electronics',        'KSC',    'stock'),
    #  The Korean PREFERRED of that common (2026-08-05).  Present so the Asia universes
    #  exercise the shape they were blocked on: same 5-digit root, same company name,
    #  6th character 5 instead of 0.  Nothing name- or suffix-based can see it; only
    #  carveOut's Korean canonicity marker can, which is what the Korea gate checks.
    ('005935.KS', 'Samsung Electronics',        'KSC',    'stock'),
    #  An ACCESS-EXCLUDED Asian venue (Taiwan), so "excluded codes leak nowhere" is a
    #  real assertion rather than a claim about an empty set.
    ('2330.TW',   'Taiwan Semiconductor',       'TAI',    'stock'),
    ('SIRI',      'Sirius XM Holdings',         'AMEX',   'stock'),
    ('OTCX',      'Some OTC Corp',              'OTC',    'stock'),
    ('SPY',       'SPDR S&P 500 ETF Trust',     'NYSE',   'etf'),
]


@pytest.fixture
def fixture_df():
    return pd.DataFrame(FIXTURE_ROWS,
                        columns=['symbol', 'name', 'exchangeShortName', 'type'])


@pytest.fixture
def offline_wrapper(monkeypatch, tmp_path):
    """Run tickerfilterWrapper with NO network and NO artifacts written into the repo."""
    monkeypatch.setattr(gdg, 'safe_get', lambda *a, **k: [])   # delisted page 0 -> empty
    monkeypatch.chdir(tmp_path)
    return gdg.tickerfilterWrapper


def _syms(df):
    return sorted(df['symbol'])


def test_the_corrected_european_codes_select_the_expected_names_and_suffixes(
        fixture_df, offline_wrapper):
    """THE FIX, demonstrated.  Paris/Amsterdam/Brussels/Lisbon/Oslo names now enter, and
    they carry exactly the suffixes FMP uses for those venues."""
    got = offline_wrapper(fixture_df, 'stock_NA1_EU1', 'all', -1, 'http://x', 'k')
    for sym, suffix in (('ALLUX.PA', '.PA'), ('NEDAP.AS', '.AS'), ('CAMB.BR', '.BR'),
                        ('COR.LS', '.LS'), ('GYL.OL', '.OL')):
        assert sym in _syms(got), '%s (restored code) is missing from stock_NA1_EU1' % sym
        assert sym.endswith(suffix)


def test_the_pre_fix_definition_would_have_selected_NONE_of_them(fixture_df):
    """The counterfactual, on the same fixture: the OLD code list picks zero European
    names beyond LSE/XETRA/STO/ICE.  This is what shipped for the life of the project."""
    old = un.UNIVERSES['stock_NA1_EU1']['was']
    assert 'EURONEXT' in old and 'OSE' in old
    picked = fixture_df[fixture_df['exchangeShortName'].isin(old)]
    for sym in ('ALLUX.PA', 'NEDAP.AS', 'CAMB.BR', 'COR.LS', 'GYL.OL'):
        assert sym not in set(picked['symbol'])
    assert set(fixture_df[fixture_df['exchangeShortName'].isin(
        ('EURONEXT', 'OSE'))]['symbol']) == set()


def test_us1_selects_only_us_and_us1_eu2_is_no_longer_us_only(fixture_df, offline_wrapper):
    """stock_US1_EU2's only non-US code was EURONEXT, so it WAS stock_US1 exactly.  The
    two must now differ."""
    us1 = set(_syms(offline_wrapper(fixture_df, 'stock_US1', 'all', -1, 'http://x', 'k')))
    eu2 = set(_syms(offline_wrapper(fixture_df, 'stock_US1_EU2', 'all', -1, 'http://x', 'k')))
    assert us1 == {'AAPL', 'BRK-A'}
    assert eu2 - us1 == {'ALLUX.PA', 'NEDAP.AS', 'CAMB.BR', 'COR.LS'}
    assert 'GYL.OL' not in eu2, 'US1_EU2 is Euronext-only; Oslo is a separate code'


def test_asia_appears_ONLY_in_the_asia_universes_and_amex_otc_nowhere(
        fixture_df, offline_wrapper):
    """REWRITTEN 2026-08-05, when Asia was wired.  It used to assert Asia leaked NOWHERE;
    the equivalent guard now is that Asia appears ONLY where it was deliberately added,
    and that the ACCESS-EXCLUDED Asian venues (India / China A / Taiwan) still appear
    nowhere at all.  AMEX and OTC remain unwired pending a CEO decision.
    """
    asia_universes = {n for n in un.names()
                      if set(un.exchanges(n) or ()) & set(un.ASIA_LIKELY_INVESTABLE)}
    assert asia_universes == {'stock_ASIA1', 'stock_NA1_EU1_ASIA1'}, (
        'an unexpected universe now wires likely-investable Asia: %s' % asia_universes)
    for name in un.names():
        if un.is_every_exchange(name):
            continue                          # FULL is *supposed* to contain everything
        got = set(_syms(offline_wrapper(fixture_df, name, 'all', -1, 'http://x', 'k')))
        #  Never wired anywhere: unwired US venues and the access-excluded Asian ones.
        for leak in ('SIRI', 'OTCX', '2330.TW'):
            assert leak not in got, '%s leaked %s' % (name, leak)
        for asian in ('7203.T', '005930.KS', '005935.KS'):
            if name in asia_universes:
                assert asian in got, '%s should contain %s' % (name, asian)
            else:
                assert asian not in got, '%s leaked %s' % (name, asian)


def test_the_access_excluded_asian_codes_are_recorded_and_wired_nowhere():
    """India, China A and Taiwan are the BULK of what Asia would add and are excluded on
    ACCESS grounds, not data grounds.  Recorded as data so the decision is legible, and
    asserted absent from every universe so it cannot be widened by accident."""
    assert set(un.ASIA_ACCESS_EXCLUDED) == {'NSE', 'BSE', 'SHH', 'SHZ', 'TAI', 'TWO'}
    for name, d in un.UNIVERSES.items():
        assert not (set(un.ASIA_ACCESS_EXCLUDED) & set(d['exchanges'] or ())), (
            '%s wires an access-excluded Asian code' % name)
    #  Taiwan's counts are this module's own live-verified figures, so they must agree.
    assert un.ASIA_CANDIDATE_CODES['TAI'] + un.ASIA_CANDIDATE_CODES['TWO'] == 2108


def test_the_full_universe_applies_no_exchange_filter(fixture_df, offline_wrapper):
    got = set(_syms(offline_wrapper(fixture_df, 'stock_FULL1', 'all', -1, 'http://x', 'k')))
    for sym in ('7203.T', '005930.KS', 'SIRI', 'OTCX', 'AAPL', 'GYL.OL'):
        assert sym in got
    assert 'SPY' not in got, 'type != stock must still be dropped'


def test_the_etf_row_is_dropped_by_every_universe(fixture_df, offline_wrapper):
    for name in un.names():
        got = set(_syms(offline_wrapper(fixture_df, name, 'all', -1, 'http://x', 'k')))
        assert 'SPY' not in got


def test_an_unknown_universe_RAISES_instead_of_returning_the_whole_world(
        fixture_df, offline_wrapper):
    """The old chain had no `else`, so an unrecognised name returned `tickdf` untouched:
    a 5x-oversized fetch with no warning."""
    with pytest.raises(Exception) as e:
        offline_wrapper(fixture_df, 'stock_TYPO', 'all', -1, 'http://x', 'k')
    assert 'not valid' in str(e.value)
    assert 'stock_NA1_EU1' in str(e.value), 'the error should list the valid universes'


def test_the_explicit_test_universe_selects_by_symbol_not_exchange(offline_wrapper):
    members = un.symbols('stock_TEST1')
    rows = [(s, 'Name of ' + s, 'NASDAQ', 'stock') for s in members[:20]]
    rows.append(('NOT_A_MEMBER', 'Other Co', 'NASDAQ', 'stock'))
    df = pd.DataFrame(rows, columns=['symbol', 'name', 'exchangeShortName', 'type'])
    got = set(_syms(offline_wrapper(df, 'stock_TEST1', 'all', -1, 'http://x', 'k')))
    assert 'NOT_A_MEMBER' not in got
    assert got == set(members[:20]) - {'MS-PE', 'GIPRW'}   # instrument filter still runs


# --------------------------------------------------------------------------- #
#  THE CURATED TEST UNIVERSE                                                   #
# --------------------------------------------------------------------------- #
def test_the_test_universe_is_STABLE_across_two_invocations():
    """The stability requirement, demonstrated.  Two independent resolutions must give
    byte-identical membership IN THE SAME ORDER and the same fingerprint -- a universe
    that re-derives membership from live data cannot do this, which is why the list is
    frozen rather than ruled."""
    a, b = un.symbols('stock_TEST1'), un.symbols('stock_TEST1')
    assert a == b
    assert list(a) == list(b)
    assert un.definition_fingerprint('stock_TEST1') == un.definition_fingerprint('stock_TEST1')
    #  and across a fresh import of the module -- i.e. not memoised state
    import importlib
    m = importlib.reload(un)
    assert tuple(m.symbols('stock_TEST1')) == tuple(a)
    assert m.definition_fingerprint('stock_TEST1') == un.definition_fingerprint('stock_TEST1')


def test_every_test_universe_member_has_a_rationale():
    """Requirement 3: a test universe whose reasons are lost becomes an arbitrary list
    nobody dares change."""
    for sym, tag, reason in un.test_universe_manifest():
        assert tag, '%s has no tag' % sym
        assert reason and len(reason) > 20, '%s has no usable rationale: %r' % (sym, reason)


def test_the_test_universe_has_no_duplicates_and_matches_its_registry_entry():
    manifest = un.test_universe_manifest()
    syms = [s for s, _t, _r in manifest]
    assert len(syms) == len(set(syms))
    assert tuple(syms) == tuple(un.symbols('stock_TEST1'))
    assert un.expected_count('stock_TEST1') == len(syms)


def test_the_test_universe_is_a_SUBSET_of_the_production_universe():
    """A test universe containing names production cannot see would test nothing.  Checked
    via the ticker suffix, which is deterministic per exchange on FMP."""
    allowed = set(un.exchanges('stock_NA1_EU1'))
    suffix_to_code = {'PA': 'PAR', 'AS': 'AMS', 'BR': 'BRU', 'LS': 'LIS', 'OL': 'OSL',
                      'L': 'LSE', 'DE': 'XETRA', 'ST': 'STO', 'TO': 'TSX', 'IC': 'ICE'}
    for sym in un.symbols('stock_TEST1'):
        if '.' not in sym:
            continue                    # suffix-less = US (NYSE/NASDAQ), both allowed
        suf = sym.rsplit('.', 1)[1]
        assert suf in suffix_to_code, '%s has an unrecognised suffix %r' % (sym, suf)
        assert suffix_to_code[suf] in allowed, (
            '%s sits on %s, which is not in stock_NA1_EU1' % (sym, suffix_to_code[suf]))


def test_the_test_universe_spans_every_restored_european_exchange():
    """It must exercise the fix, not merely coexist with it."""
    by_suffix = {}
    for sym in un.symbols('stock_TEST1'):
        if '.' in sym:
            by_suffix.setdefault(sym.rsplit('.', 1)[1], []).append(sym)
    for suf, code in (('PA', 'PAR'), ('AS', 'AMS'), ('BR', 'BRU'),
                      ('LS', 'LIS'), ('OL', 'OSL')):
        assert len(by_suffix.get(suf, [])) >= 1, (
            'no %s (%s) member -- the test universe would not exercise the fix' % (code, suf))


def test_the_test_universe_declares_both_reporting_frequencies_and_several_currencies():
    reasons = ' | '.join(r for _s, _t, r in un.test_universe_manifest())
    for token in ('semi-annual', 'quarterly'):
        assert token in reasons, 'no member declares %s' % token
    for ccy in ('EUR', 'NOK', 'ISK', 'SEK', 'CAD', 'GBp', 'IDR', 'KRW'):
        assert ccy in reasons, 'no member declares the %s reporting currency' % ccy


def test_the_test_universe_carries_every_required_edge_case_tag():
    tags = {t for _s, t, _r in un.test_universe_manifest()}
    for required in vtu.REQUIRED_TAGS:
        assert required in tags, 'no member tagged %r' % required


#  LIVE company names, as FMP returns them (2026-08-02).  They matter: the filter's
#  rule A reads the NAME, and FMP gives GIPRW the COMMON's name verbatim -- no "Warrants"
#  token -- which is why rule C, not rule A, is what removes it.
_FILTER_FIXTURE = {
    'MS-PE':     ('Morgan Stanley', 'NYSE'),
    'MS':        ('Morgan Stanley', 'NYSE'),
    'GIPRW':     ('Generation Income Properties, Inc.', 'NASDAQ'),
    'GIPR':      ('Generation Income Properties, Inc.', 'NASDAQ'),
    'INVE-A.ST': ('Investor AB (publ)', 'STO'),
    'INVE-B.ST': ('Investor AB (publ)', 'STO'),
}


def test_the_share_class_filter_removes_the_filter_remove_members_and_keeps_the_others():
    """Both directions of the instrument filter, on the members curated for it.  The
    must-KEEP side is the one that matters: an earlier version of that filter deleted
    real commons (Brookdale SENIOR Living, PREFERRED Bank, NOTE AB)."""
    manifest = un.test_universe_manifest()
    remove = [s for s, t, _r in manifest if t == 'filter-remove']
    keep = [s for s, t, _r in manifest if t == 'filter-keep']
    assert remove and keep
    #  GIPR is included because rule C is PAIRWISE (see below); it is a real member.
    syms = sorted(set(remove + keep + ['GIPR']))
    df = pd.DataFrame([(s,) + _FILTER_FIXTURE[s] + ('stock',) for s in syms],
                      columns=['symbol', 'name', 'exchangeShortName', 'type'])
    out = gdg.filter_non_common_instruments(df, verbose=False, log_csv=False)
    survivors = set(out['symbol'])
    for s in remove:
        assert s not in survivors, '%s should have been removed by the instrument filter' % s
    for s in keep:
        assert s in survivors, '%s is a COMMON and must survive the instrument filter' % s


def test_rule_C_is_PAIRWISE_so_the_warrants_sibling_must_stay_in_the_universe():
    """A GENUINE PROPERTY OF THE FILTER, found while building this list.

    `filter_non_common_instruments` rule C recognises GIPRW as GIPR + "W" only by
    comparing it to a SHORTER SAME-NAME SAME-EXCHANGE SIBLING.  Rules A and B cannot
    catch GIPRW at all: FMP gives the warrant the common's name verbatim (no "Warrants"
    token) and there is no `-P` suffix.  So dropping GIPR from the universe silently
    disables the only rule that removes GIPRW -- which means a universe SUBSET can be
    less well filtered than the full universe.  GIPR is therefore load-bearing here and
    this test is what stops a future re-curation of the cohort fill from removing it.
    """
    both = pd.DataFrame([('GIPR',) + _FILTER_FIXTURE['GIPR'],
                         ('GIPRW',) + _FILTER_FIXTURE['GIPRW']],
                        columns=['symbol', 'name', 'exchangeShortName'])
    both['type'] = 'stock'
    kept_both = set(gdg.filter_non_common_instruments(
        both.copy(), verbose=False, log_csv=False)['symbol'])
    assert kept_both == {'GIPR'}, 'with the sibling present the warrant must go'

    alone = both[both['symbol'] == 'GIPRW'].reset_index(drop=True)
    kept_alone = set(gdg.filter_non_common_instruments(
        alone.copy(), verbose=False, log_csv=False)['symbol'])
    assert kept_alone == {'GIPRW'}, (
        'documented behaviour: WITHOUT the sibling, rule C cannot fire and the warrant '
        'survives -- this is why GIPR must remain a member')

    assert 'GIPR' in un.symbols('stock_TEST1'), (
        'GIPR was dropped from the test universe; the GIPRW filter-remove slot no longer '
        'tests anything')


def test_the_test_universe_states_both_its_listed_and_its_effective_size():
    """142 listed, 140 fetched.  A run reporting "140 of 142" is CORRECT (two members
    exist to be filtered out) and must not be read as shrinkage -- so both numbers are
    recorded rather than one."""
    assert un.TEST_UNIVERSE_LISTED == 142
    assert un.TEST_UNIVERSE_EFFECTIVE == 140
    assert set(un.TEST_UNIVERSE_FILTERED_BY_DESIGN) == {'MS-PE', 'GIPRW'}
    assert (un.TEST_UNIVERSE_LISTED
            - len(un.TEST_UNIVERSE_FILTERED_BY_DESIGN)) == un.TEST_UNIVERSE_EFFECTIVE
    for s in un.TEST_UNIVERSE_FILTERED_BY_DESIGN:
        assert s in un.symbols('stock_TEST1')
    #  the fetch-cost claim, so "minutes not hours" is checkable arithmetic
    assert un.TEST_UNIVERSE_API_CALLS == 5 * 142 + 3
    assert un.TEST_UNIVERSE_WALLCLOCK_MIN == (8, 15)


@pytest.mark.skipif(vtu.newest_panel() is None,
                    reason='no saved Bometric panel on this machine (gitignored, ~140MB)')
def test_the_test_universe_covers_every_required_category_MEASURED_not_asserted():
    """The coverage claim, MEASURED against the repo's own classifiers.

    `gaps()` returns the categories that are NOT populated, so a decayed slot (a
    short-history IPO that has accreted past the gate, a cohort tag that moved when the
    sector map was rebuilt) surfaces as a named gap rather than as a green test that
    stopped looking.
    """
    cov = vtu.collect(verbose=False)
    problems = vtu.gaps(cov)
    assert not problems, 'test-universe coverage gaps: %s' % (problems,)
    #  the specific counts the curation promises
    assert cov['n_members'] == 142
    for cohort in vtu.REQUIRED_COHORTS:
        assert cov['cohort_counts'].get(cohort, 0) >= vtu.MIN_PER_COHORT
    assert cov['frequency_counts'].get('semiannual', 0) >= 20
    assert cov['frequency_counts'].get('quarterly', 0) >= 50
    assert len(cov['exchange_counts']) >= 12


# --------------------------------------------------------------------------- #
#  PROVENANCE STAMPING                                                         #
# --------------------------------------------------------------------------- #
def test_nrTaT_on_the_curated_universe_warns_that_it_discards_the_coverage(capsys):
    """`-nrTaT` re-introduces exactly the positional-prefix bias the curated universe
    exists to remove, so combining them must not be silent.  A WARNING, not a rejection:
    a deliberate 5-ticker smoke test of the test universe is legitimate."""
    cfg.getDataFetchConfiguration(['x', '-tickerfilter', 'stock_TEST1', '-nrTaT', '20'])
    out = capsys.readouterr().out
    assert 'positional prefix' in out and 'stock_TEST1' in out

    cfg.getDataFetchConfiguration(['x', '-tickerfilter', 'stock_US1', '-nrTaT', '20'])
    assert 'positional prefix' not in capsys.readouterr().out, (
        'the warning is about the CURATED list losing its coverage; it does not apply '
        'to an exchange-defined universe')


def test_the_stale_exchange_notes_file_is_marked_as_not_a_source():
    """`Notes/stockExchangeList.txt` calls Korea's KSC "Kuwait" and Iceland's ICE
    "Intercontinental Exchange".  It must carry its own warning, or the next person
    to need an exchange code will read it and be misled the same way."""
    here = os.path.dirname(os.path.abspath(un.__file__))
    p = os.path.join(here, 'Notes', 'stockExchangeList.txt')
    if not os.path.exists(p):
        pytest.skip('notes file absent')
    head = open(p, encoding='utf-8').read(2000)
    assert 'DO NOT USE THIS FILE AS A SOURCE OF EXCHANGE CODES' in head
    assert 'universes.py' in head


# --------------------------------------------------------------------------- #
#  THE SHARED SECTOR/INDUSTRY MAPS MUST SURVIVE A SUBSET RUN                    #
#                                                                               #
#  The regression these pin was INTRODUCED by the selectable-universe work and    #
#  its trigger was the README's own step 4.  `get_tickers` hands the FILTERED     #
#  universe to `ensure_sector_industry_maps`, and `buildSectorIndustryMaps` used   #
#  to `pd.to_pickle` over `sectorsdic_fmp.pickle` with no merge.  `-nrTaT 50`     #
#  never shrank `df` there (the cap applies downstream in getData_fmp), so before  #
#  stock_TEST1 existed the map was always authored from a full universe.          #
# --------------------------------------------------------------------------- #
import findAllSectors as fas
import carveOut as co


def test_a_subset_universe_NEVER_authors_the_shared_sector_maps(tmp_path, monkeypatch,
                                                               capsys):
    """A curated-subset run must not write these shared, universe-independent maps.

    A 142-symbol map applied to a later 10,693-name pool covers 1.3% of it -- non-empty,
    so it passes carveOut's empty-map abort while REIT/Mining leak wholesale.  Worse,
    `_normalize_sector_dic`'s <10-member floor is calibrated for the full universe, so
    on a ~142-name batch most sectors collapse to 'Unspecified' and the map is not just
    small but almost entirely SECTORLESS.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fas, 'buildSectorIndustryMaps',
                        lambda *a, **k: pytest.fail('a SUBSET run must never build the '
                                                    'shared sector/industry maps'))
    built = fas.ensure_sector_industry_maps(
        list(un.symbols('stock_TEST1')), 'https://x/', 'KEY',
        universe_is_subset=True, universe_name='stock_TEST1')
    assert built is False
    out = capsys.readouterr().out
    assert 'SKIPPED' in out and 'SUBSET' in out
    assert 'stock_TEST1' in out
    #  and nothing was written
    assert not list(tmp_path.glob('sectorsdic_fmp.pickle'))
    assert not list(tmp_path.glob('industrydic_fmp_*.pickle'))


def test_get_tickers_flags_the_subset_universe_to_the_map_hook(monkeypatch, tmp_path):
    """The wiring, not just the helper: the flag must actually be set for stock_TEST1
    and clear for an exchange-defined universe."""
    seen = {}
    rows = [('AAPL', 'Apple Inc.', 'NASDAQ', 'stock'),
            ('MCFT', 'MasterCraft Boat Holdings, Inc.', 'NASDAQ', 'stock'),
            ('GYL.OL', 'Gyldendal ASA', 'OSL', 'stock')]
    at = [dict(symbol=s, name=n, price=1.0, exchange=e, exchangeShortName=e, type=t)
          for s, n, e, t in rows]
    monkeypatch.setattr(gdg, 'safe_get',
                        lambda url, *a, **k: (at if 'available-traded' in url else
                                              ([{'symbol': s} for s, _n, _e, _t in rows]
                                               if 'symbol-lists' in url else [])))
    monkeypatch.setattr(fas, 'ensure_sector_industry_maps',
                        lambda syms, *a, **k: seen.update(k) or None)
    monkeypatch.chdir(tmp_path)
    gdg.get_tickers('fmp', 'https://x/', 'KEY', [], 'stock_TEST1', sfilt='all', mcapf=-1, fn='')
    assert seen.get('universe_is_subset') is True
    assert seen.get('universe_name') == 'stock_TEST1'
    seen.clear()
    gdg.get_tickers('fmp', 'https://x/', 'KEY', [], 'stock_NA1_EU1', sfilt='all', mcapf=-1, fn='')
    assert seen.get('universe_is_subset') is False


def test_building_the_maps_MERGES_and_can_never_shrink_them(tmp_path, monkeypatch):
    """The no-shrink property, end to end through the real writer.

    Even a legitimate full-universe rebuild must not delete symbols it did not happen to
    cover this time -- the artifact is shared and the next reader may score a bigger pool.
    """
    monkeypatch.chdir(tmp_path)
    pd.to_pickle({'Real Estate': ['OLD1', 'OLD2'], 'Basic Materials': ['OLD3']},
                 'sectorsdic_fmp.pickle')
    pd.to_pickle({'OLD1': 'REITs', 'OLD3': 'Gold'}, 'industrydic_fmp_2020-01-01.pickle')

    profiles = [{'symbol': 'NEW%d' % i, 'sector': 'Technology', 'industry': 'Software'}
                for i in range(12)]
    profiles.append({'symbol': 'OLD1', 'sector': 'Technology', 'industry': 'Software'})
    monkeypatch.setattr(fas, '_fetch_profiles_batched',
                        lambda *a, **k: (profiles, 1, True, set()))

    sec, ind = fas.buildSectorIndustryMaps(['x'], 'https://x/', 'KEY')
    on_disk = pd.read_pickle('sectorsdic_fmp.pickle')
    syms = {s for v in on_disk.values() for s in v}
    assert {'OLD2', 'OLD3'} <= syms, 'pre-existing symbols were DELETED by a rebuild'
    assert 'NEW0' in syms
    #  OLD1 was re-fetched into a different sector -> it moves, and does NOT duplicate
    assert 'OLD1' not in on_disk.get('Real Estate', [])
    assert 'OLD1' in on_disk['Technology']
    assert sum(v.count('OLD1') for v in on_disk.values()) == 1
    ind_disk = pd.read_pickle(sorted(tmp_path.glob('industrydic_fmp_*.pickle'))[-1])
    assert ind_disk['OLD3'] == 'Gold', 'an industry entry was dropped by a rebuild'
    assert ind_disk['OLD1'] == 'Software', 'a re-fetched entry should win'


def test_a_full_universe_still_builds_the_maps(tmp_path, monkeypatch):
    """The guard must not break the legitimate self-heal path it sits in front of."""
    monkeypatch.chdir(tmp_path)
    called = {}
    monkeypatch.setattr(fas, 'buildSectorIndustryMaps',
                        lambda *a, **k: called.setdefault('yes', True))
    assert fas.ensure_sector_industry_maps(['A', 'B'], 'https://x/', 'KEY',
                                           universe_is_subset=False) is True
    assert called.get('yes') is True


def test_both_maps_present_is_still_an_idempotent_no_op(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pd.to_pickle({'Technology': ['A']}, 'sectorsdic_fmp.pickle')
    pd.to_pickle({'A': 'Software'}, 'industrydic_fmp_2020-01-01.pickle')
    monkeypatch.setattr(fas, 'buildSectorIndustryMaps',
                        lambda *a, **k: pytest.fail('must not rebuild when both present'))
    assert fas.ensure_sector_industry_maps(['A'], 'https://x/', 'KEY') is False
    assert fas.ensure_sector_industry_maps(['A'], 'https://x/', 'KEY',
                                           universe_is_subset=True) is False


# --------------------------------------------------------------------------- #
#  THE CONSUMER-SIDE COVERAGE GUARD                                            #
# --------------------------------------------------------------------------- #
def test_the_coverage_thresholds_sit_between_the_measured_healthy_and_poisoned_cases():
    """Grounded, not picked.  Live maps cover 87.1% of the 2026-01-08 panel's 9,012
    sources; a map authored by the 142-name test universe would cover 1.6%."""
    assert co.SECTOR_COVERAGE_ABORT_BELOW < co.SECTOR_COVERAGE_WARN_BELOW
    poisoned = 142 / 9012.0
    assert poisoned < co.SECTOR_COVERAGE_ABORT_BELOW, 'the poisoned case must ABORT'
    assert co.SECTOR_COVERAGE_WARN_BELOW < co.SECTOR_COVERAGE_HEALTHY_REF, (
        'a healthy run (87.1%) must not trip the warning')


def _carve_inputs(n_pool, n_covered):
    pool = ['S%04d' % i for i in range(n_pool)]
    bo = pd.DataFrame({'source': pool, 'BoScore': [1.0] * n_pool})
    sector_dic = {'Real Estate': pool[:n_covered]}
    tickers = pd.DataFrame({'symbol': pool, 'name': ['N' + s for s in pool]})
    cdx = pd.DataFrame({'source': pool, 'date': pd.Timestamp('2025-01-01'),
                        'marketCap': 1e8, 'totalStockholdersEquity': 5e7,
                        'totalAssets': 1e8, 'revenue': 5e7,
                        'weightedAverageShsOut': 1e6, 'netIncome': 1e6})
    return bo, cdx, tickers, sector_dic


def test_carve_ABORTS_when_the_sector_map_covers_a_fraction_of_the_pool(tmp_path,
                                                                       monkeypatch):
    """THE BLOCKING BUG'S CONSUMER-SIDE BACKSTOP.  A non-empty but wrong-sized map used
    to sail through, because the only guard tested for EMPTY."""
    monkeypatch.chdir(tmp_path)
    bo, cdx, tickers, sector_dic = _carve_inputs(1000, 14)     # 1.4% -- the real case
    pd.to_pickle(sector_dic, 'sectorsdic_fmp.pickle')
    monkeypatch.setattr(co, '_load_sector_map',
                        lambda *a, **k: {s: 'Real Estate' for s in sector_dic['Real Estate']})
    monkeypatch.setattr(co, '_load_industry_map', lambda *a, **k: {})
    with pytest.raises(RuntimeError) as e:
        co.partition_universe(bo, cdx, tickers)
    assert 'covers only' in str(e.value)
    assert '1.4' in str(e.value)


def test_carve_WARNS_but_proceeds_on_thin_coverage_and_is_QUIET_when_healthy(
        tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(co, '_load_industry_map', lambda *a, **k: {})

    bo, cdx, tickers, _ = _carve_inputs(1000, 600)             # 60% -> warn, proceed
    monkeypatch.setattr(co, '_load_sector_map',
                        lambda *a, **k: {'S%04d' % i: 'Real Estate' for i in range(600)})
    co.partition_universe(bo, cdx, tickers)
    assert 'THIN SECTOR-MAP COVERAGE' in capsys.readouterr().out

    monkeypatch.setattr(co, '_load_sector_map',
                        lambda *a, **k: {'S%04d' % i: 'Real Estate' for i in range(871)})
    co.partition_universe(bo, cdx, tickers)                     # 87.1% -- the real norm
    out = capsys.readouterr().out
    assert 'THIN SECTOR-MAP COVERAGE' not in out, 'a healthy run must not warn'
    assert 'coverage: 871 of 1000' in out


# --------------------------------------------------------------------------- #
#  THE INSTRUMENT FILTER IS UNIVERSE-INDEPENDENT (review item 3)                 #
# --------------------------------------------------------------------------- #
def test_the_instrument_filter_is_as_strong_on_a_SUBSET_as_on_the_WHOLE(offline_wrapper):
    """Rule C is PAIRWISE, so filtering the already-narrowed frame made the filter's
    strength depend on which universe was active.  Suffix maps 1:1 to exchange code with
    one exception -- every US ticker has NO suffix, so NYSE/NASDAQ/AMEX/OTC/PNK share the
    empty one and an enabling sibling can sit on an unwired exchange.

    The real instance (measured 2026-08-02): ZCARW, a Zoomcar warrant on NASDAQ, whose
    enabling common ZCAR trades on OTC.  Every universe admitted it, because no universe
    wires OTC.  Filtering the FULL table first removes it from all of them.
    """
    rows = [('ZCAR', 'Zoomcar Holdings, Inc.', 'OTC', 'stock'),
            ('ZCARW', 'Zoomcar Holdings, Inc.', 'NASDAQ', 'stock'),
            ('AAPL', 'Apple Inc.', 'NASDAQ', 'stock')]
    df = pd.DataFrame(rows, columns=['symbol', 'name', 'exchangeShortName', 'type'])
    #  stock_US1 does NOT include OTC, so under the old order ZCAR was absent from the
    #  frame when rule C ran and the warrant survived.
    got = set(offline_wrapper(df, 'stock_US1', 'all', -1, 'http://x', 'k')['symbol'])
    assert 'ZCARW' not in got, (
        'the warrant survived: the instrument filter is still being applied AFTER '
        'membership selection, so its completeness depends on the active universe')
    assert got == {'AAPL'}

    #  and the property generalises: for every universe, membership selection and the
    #  instrument filter COMMUTE, because the filter no longer sees the narrowed frame.
    full_kept = set(gdg.filter_non_common_instruments(
        df.copy(), verbose=False, log_csv=False)['symbol'])
    for name in un.names():
        resolved = set(offline_wrapper(df, name, 'all', -1, 'http://x', 'k')['symbol'])
        assert resolved <= full_kept, (
            '%s kept a name the full-table filter removes' % name)


# --------------------------------------------------------------------------- #
#  ISSUER-GROUP CLOSURE (review item 4)                                        #
# --------------------------------------------------------------------------- #
#  NOTE FOR THE NEXT READER: an intermediate version of the test below tried to verify the
#  closure claims by REGEX-SCANNING the rationale prose for ticker-shaped tokens, with a
#  skip-list of English words that look like tickers.  That approach is a dead end and was
#  removed rather than left lying around: a rationale legitimately names symbols it does NOT
#  close (SMSN.L cites the excluded SMSD.L and the Seoul ticker 005930.KS as evidence), so no
#  token scan can separate a claim from a counter-example.  The claim is DATA now
#  (`universes.DEDUP_PARTNER_CLOSES`).


def test_the_dedup_partners_are_all_present_and_each_names_what_it_closes():
    """carveOut's dedup edges are PAIRWISE OVER THE POOL, so a member whose same-issuer
    sibling is absent is deduped differently here than in production.  A name-based audit
    found SIXTEEN such members; 15 partners now close them, and the groups that cannot be
    closed without re-importing an instrument line are DECLARED.

    Checked here: every partner is a member, is tagged as one, and its rationale names
    the member whose group it closes -- and that named member is itself a member. That is
    what stops a partner being dropped later and leaving an unhonoured claim behind."""
    manifest = un.test_universe_manifest()
    members = {s for s, _t, _r in manifest}
    partners = [(s, r) for s, t, r in manifest if t == 'dedup-partner']
    assert len(partners) == 15, 'expected 15 closure partners, found %d' % len(partners)

    #  THE PREVIOUS VERSION OF THIS BLOCK WAS A NO-OP: a loop whose every branch was
    #  `continue`, with NO assert, under a docstring claiming it verified membership.
    #  A decorative guard is worse than none -- the class it was written to catch (a
    #  partner asserting a closure the list does not deliver) recurred, and the suite
    #  stayed green through it.
    #
    #  It is now checked against DATA (`DEDUP_PARTNER_CLOSES`), not by regex-scanning the
    #  English. A rationale legitimately names symbols it does NOT close -- SMSN.L cites
    #  the excluded SMSD.L and the Seoul ticker 005930.KS as evidence -- so no token scan
    #  can distinguish a claim from a counter-example.
    assert set(un.DEDUP_PARTNER_CLOSES) == {s for s, _r in partners}, (
        'DEDUP_PARTNER_CLOSES and the dedup-partner tags disagree: %s'
        % (set(un.DEDUP_PARTNER_CLOSES) ^ {s for s, _r in partners}))

    unhonoured = []
    for sym, reason in partners:
        assert reason and len(reason) > 30, '%s has no usable rationale' % sym
        closes = un.DEDUP_PARTNER_CLOSES[sym]
        if closes is None:
            #  a partner that closes nothing must SAY so, and the member it sits beside
            #  must be declared open -- otherwise it is a silent non-closure
            assert 'does NOT close' in reason or 'closes NOTHING' in reason, (
                '%s closes nothing but its rationale does not say so: %r' % (sym, reason))
            continue
        if closes not in members:
            unhonoured.append((sym, closes, 'the member it closes is not in the list'))
        elif closes in un.TEST_UNIVERSE_OPEN_GROUPS:
            unhonoured.append((sym, closes,
                               'claims to CLOSE a group that is DECLARED OPEN'))
    assert not unhonoured, (
        'dedup-partner(s) assert a closure the list does not deliver -- exactly the HTFC '
        'defect: %s' % unhonoured)


def test_the_claimed_cross_listing_dedup_cases_actually_have_their_partners():
    """M1: AI.PA and SHELL.AS were documented as exercising cross-listing dedup while
    NONE of their partners was a member, so no dedup fired on any restored code at all --
    a claim the list did not honour."""
    members = set(un.symbols('stock_TEST1'))
    for member, partners in (('AI.PA', ('AIL.DE',)),
                             ('SHELL.AS', ('SHEL.L', 'SHEL')),
                             ('DWD.DE', ('MS', '0QYU.L')),
                             ('8TI.DE', ('STLA', 'STLAP.PA')),
                             ('WDC', ('0QZF.L', 'WDC.DE')),
                             ('LIN', ('LIN.DE',)),
                             ('IHG.L', ('IHG',)),
                             ('CGI.TO', ('CGI.L',)),
                             ('NEDAP.AS', ('0NNU.L',))):
        assert member in members
        for p in partners:
            assert p in members, (
                '%s claims dedup but its partner %s is absent -- production would '
                'collapse this issuer and the test universe would not' % (member, p))


def test_dedup_now_fires_on_a_RESTORED_exchange():
    """The point of the closure work: at least one complete issuer group must SPAN a
    newly-restored code, or the restored exchanges are never dedup-tested."""
    members = set(un.symbols('stock_TEST1'))
    restored_groups = [('AI.PA', 'AIL.DE'), ('SHELL.AS', 'SHEL.L'),
                       ('STLAP.PA', 'STLA'), ('NEDAP.AS', '0NNU.L')]
    fired = [g for g in restored_groups if all(x in members for x in g)]
    assert fired, 'no complete issuer group spans a restored exchange'


def test_the_known_non_common_lines_were_REMOVED_from_the_reference_list():
    """M3: WHLRD (a Series D preferred) and BWNB (senior notes) sat in the list as
    ORDINARY COHORT MEMBERS carrying their sibling common's fundamentals.  They survive
    in PRODUCTION too -- pre-existing filter gaps -- but they must not silently populate
    cohorts in the reference list, or its cohort counts are not trustworthy."""
    members = set(un.symbols('stock_TEST1'))
    for junk in ('WHLRD', 'BWNB'):
        assert junk not in members, (
            '%s is a non-common line and must not be a cohort member' % junk)
    #  the commons they hang off ARE legitimate members
    assert 'WHLR' in members and 'BW' in members


def test_a_CURATED_panel_can_never_stand_in_as_the_production_pool():
    """THE REGRESSION GUARD for a live trap (fixed 2026-08-04).

    `derive_divergent` reconstructs "the pool production sees" from a panel's own
    `Tickers_df`.  Handed a CURATED-universe panel it derives that pool from the very ~140
    names under test and then reconciles the test universe against it -- circular, and it
    reports every declared open group as over-declared because production's siblings cannot
    exist in a 126-name panel.  It fired for real the first time a `stock_TEST1` run wrote a
    panel, which the README recommends as the standard iteration route.

    Two things are pinned, because fixing only the first does NOT fix the bug:
      * "newest" means newest BY DATE (it sorted lexicographically, and `stock_TEST1` beats
        `stock_NA1_EU1` on a string compare whatever the dates);
      * and the production-only selector EXCLUDES curated panels -- which is the operative
        fix, since a date sort alone still picks the newer curated panel.
    """
    #  the discriminator is the REGISTRY's own, not a hard-coded name
    assert vtu.is_explicit_list_panel('Bometric_dic-fmp_stock_TEST1_all_2026-08-04_len126.pickle')
    assert not vtu.is_explicit_list_panel(
        'Bometric_dic-fmp_stock_NA1_EU1_all_2026-01-08_len9012.pickle')
    #  ... and the nested names resolve to the LONGEST match, not the first
    assert vtu.panel_universe(
        'Bometric_dic-fmp_stock_NA1_EU1_all_2026-01-08_len9012.pickle') == 'stock_NA1_EU1'
    assert vtu.panel_universe(
        'Bometric_dic-fmp_stock_NA1_all_2026-01-08_len9012.pickle') == 'stock_NA1'
    #  every universe with an explicit ticker list is treated as curated, by construction
    for name in un.names():
        path = 'Bometric_dic-fmp_%s_all_2026-01-08_len1.pickle' % name
        assert vtu.is_explicit_list_panel(path) == (un.symbols(name) is not None), name
    prod = vtu.newest_panel(production_only=True)
    if prod is not None:
        assert not vtu.is_explicit_list_panel(prod)


#  PRODUCTION-ONLY panel, matching what `derive_divergent` now selects (fix, 2026-08-04).
#  Skipping on `newest_panel()` was wrong for THIS test: a machine holding only a curated
#  `stock_TEST1` panel would not skip, and would then fail with 'reconciliation unavailable' or
#  -- worse, before the fix -- with a circular over-declaration.  The guard must ask for the
#  kind of panel the derivation actually needs.
@pytest.mark.skipif(vtu.newest_panel(production_only=True) is None,
                    reason='no saved PRODUCTION Bometric panel on this machine '
                           '(gitignored, ~140MB; a curated test-universe panel cannot '
                           'stand in -- see verify_test_universe.derive_divergent)')
def test_the_declared_open_groups_RECONCILE_with_a_fresh_derivation_both_ways():
    """THE GUARD THAT REPLACES THE NO-OP.

    Re-derives the divergent set with `carveOut._issuer_components` -- the function the
    pipeline actually dedups with -- and reconciles it against the declaration in BOTH
    directions:
      * derived-but-not-declared is a HIDDEN divergence (the defect direction);
      * declared-but-not-derived is equally misleading, and is only tolerated for the
        entries whose evidence is live-list membership rather than a fundamentals
        fingerprint (`OPEN_GROUPS_LIVE_LIST_ONLY`).
    It also requires the declared SIBLING SETS to match the derivation, because a
    declaration naming the wrong sibling does not describe the divergence it claims to.

    The first closure pass declared ONE group and was wrong in three places; the guard
    that was supposed to catch that asserted nothing.
    """
    problems = vtu.reconcile_open_groups()
    assert not problems, 'open-group reconciliation failed: %s' % (problems,)


def test_the_two_evidence_bases_partition_the_declared_groups():
    """Every declared group must be attributed to one evidence base -- fingerprint-derived
    or live-list-only.  An unattributed entry is one nobody can check."""
    declared = set(un.TEST_UNIVERSE_OPEN_GROUPS)
    fp = set(un.OPEN_GROUPS_FINGERPRINT_DERIVABLE)
    ll = set(un.OPEN_GROUPS_LIVE_LIST_ONLY)
    assert fp | ll == declared, 'unattributed declared group(s): %s' % (declared - fp - ll)
    assert not (fp & ll), 'a group cannot be in both evidence bases: %s' % (fp & ll)


def test_the_mis_added_instrument_lines_are_gone_from_the_partner_set():
    """HTFC (a Horizon baby bond) and SMSD.L (the Samsung PREFERRED GDR) were added as
    `dedup-partner`s under a header claiming each was "never an instrument line".  SMSD.L
    is the Korean-preferred trap the Asia note warns about -- 29% below its common on
    identical fundamentals -- imported into the reference list by hand."""
    members = set(un.symbols('stock_TEST1'))
    for junk in ('HTFC', 'SMSD.L'):
        assert junk not in members, '%s is an instrument line and must not be a member' % junk
    #  the commons they were hanging off stay, and are declared open in consequence
    for keep in ('HRZN', 'SMSN.L', 'BC94.L'):
        assert keep in members
        assert keep in un.TEST_UNIVERSE_OPEN_GROUPS


def test_the_partner_header_no_longer_claims_none_are_instrument_lines():
    """The header asserted 'never an instrument line' while two of them were.  A comment
    that overstates is the failure mode this repo has been bitten by repeatedly."""
    import inspect
    src = inspect.getsource(un)
    assert 'never an instrument line' not in src or 'THAT WAS FALSE' in src, (
        'the corrected header must not leave the false claim standing unqualified')
    assert 'SMSD.L -- the Samsung Electronics PREFERRED GDR' in src
    assert 'HTFC   -- a Horizon Technology Finance BABY BOND' in src


def test_the_one_group_that_cannot_be_closed_is_DECLARED_not_silent():
    """Wheeler's group is {WHLR, WHLRD, WHLRL} and BOTH siblings are non-common lines the
    filter misses.  Closing it would import junk into the cohorts; dropping WHLR would
    cost a real small-cap REIT.  So the divergence is recorded rather than hidden."""
    members = set(un.symbols('stock_TEST1'))
    assert un.TEST_UNIVERSE_OPEN_GROUPS, 'the residual divergence must be declared'
    for member, spec in un.TEST_UNIVERSE_OPEN_GROUPS.items():
        assert member in members
        siblings, reason = spec[:-1], spec[-1]
        assert reason and len(reason) > 40
        for sib in siblings:
            assert sib not in members, (
                '%s is declared an OPEN-group sibling but is actually a member' % sib)


def test_the_share_class_filter_gaps_are_recorded_as_data():
    """Item 2 + M3.  Pre-existing gaps, found while auditing the reference list -- which
    is what a reference list is for.  Recorded as data so they are testable and cannot
    decay into prose, and so acting on them is a wiring change rather than a re-audit."""
    gaps = gdg.SHARE_CLASS_FILTER_KNOWN_GAPS
    assert set(gaps) == {'unwhitelisted-single-letter-tail', 'truncated-company-name',
                         'continental-convention'}
    flat = [row for rows in gaps.values() for row in rows]
    assert len(flat) >= 5
    for row in flat:
        assert len(row) == 3 and all(row), row
    #  every one of these SURVIVES the filter today -- that is the finding
    names = {'WHLRD': 'Wheeler Real Estate Investment Trust, Inc.',
             'WHLRL': 'Wheeler Real Estate Investment Trust, Inc.',
             'WHLR': 'Wheeler Real Estate Investment Trust, Inc.',
             'BWNB': 'Babcock & Wilcox Enterprises, I',
             'BW': 'Babcock & Wilcox Enterprises, Inc.',
             'CBE.PA': 'Robertet S.A.', 'RBT.PA': 'Robertet S.A.',
             'PREVA.AS': 'Value8 N.V.', 'VALUE.AS': 'Value8 N.V.'}
    df = pd.DataFrame([(s, n, 'NYSE', 'stock') for s, n in names.items()],
                      columns=['symbol', 'name', 'exchangeShortName', 'type'])
    survivors = set(gdg.filter_non_common_instruments(
        df, verbose=False, log_csv=False)['symbol'])
    for row in flat:
        assert row[0] in survivors, (
            '%s is recorded as a filter GAP but the filter now catches it -- if the '
            'filter was widened, update SHARE_CLASS_FILTER_KNOWN_GAPS' % row[0])


# --------------------------------------------------------------------------- #
#  THE ISIN DETECTOR -- built, not wired (review item 2)                        #
# --------------------------------------------------------------------------- #
def test_the_isin_detector_reproduces_its_recorded_findings_and_its_false_positive():
    """1,046 Continental names were restored and the filter removed EXACTLY ZERO of them
    -- equally consistent with "clean venues" and "blind filter".  It is the blind
    filter: two admitted non-common lines (Robertet's certificat d'investissement at
    -17.9%, Value8's preference shares at -29.9%) and two literal same-ISIN duplicates.

    NEITHER of the first two is reachable by any symbol-shape rule -- `CBE` shares no
    prefix with `RBT`, `PREVA` none with `VALUE` -- so ISIN is the only discriminator.

    The detector is NOT wired, and this test pins WHY: it fires on Heineken N.V. vs
    Heineken HOLDING N.V., two genuinely separate issuers, on the very first universe it
    was tested against.  Deleting a common is the expensive error.
    """
    syms = ['CBE.PA', 'RBT.PA', 'PREVA.AS', 'VALUE.AS', 'HAFNI.OL', 'HAFNIO.OL',
            'HEIA.AS', 'HEIO.AS']
    names = ['Robertet S.A.', 'Robertet S.A.', 'Value8 N.V.', 'Value8 N.V.',
             'Hafnia Limited', 'Hafnia Limited', 'Heineken N.V.', 'Heineken Holding N.V.']
    isins = ['FR0000045601', 'FR0000039091', 'NL0015118803', 'NL0010661864',
             'SGXZ53070850', 'SGXZ53070850', 'NL0000009165', 'NL0000008977']
    dup, multi = gdg.isin_same_issuer_groups(syms, names, isins)

    assert dup == {'SGXZ53070850': ['HAFNI.OL', 'HAFNIO.OL']}
    got = {frozenset(s for s, _i in pairs) for pairs in multi.values()}
    assert frozenset({'CBE.PA', 'RBT.PA'}) in got
    assert frozenset({'PREVA.AS', 'VALUE.AS'}) in got
    assert frozenset({'HEIA.AS', 'HEIO.AS'}) in got, (
        'the Heineken false positive is the documented reason this rule is NOT wired; '
        'if it stopped firing, the reason needs rewriting')

    #  and it is wired into NOTHING
    import inspect
    src = inspect.getsource(gdg)
    assert src.count('isin_same_issuer_groups') == 1, (
        'isin_same_issuer_groups appears %d times -- it should appear ONCE, as its own '
        'def and nowhere else. It must remain unwired until the CEO decides, because it '
        'has a known false positive on Heineken.' % src.count('isin_same_issuer_groups'))


def test_the_isin_findings_are_recorded_as_data():
    f = gdg.ISIN_DETECTOR_VERIFIED_FINDINGS
    assert ('CBE.PA', 'RBT.PA', -17.9) in f['non_common_admitted']
    assert ('PREVA.AS', 'VALUE.AS', -29.9) in f['non_common_admitted']
    assert ('HAFNI.OL', 'HAFNIO.OL') in f['duplicate_isin']
    assert ('HEIA.AS', 'HEIO.AS') in f['known_false_positive']


# --------------------------------------------------------------------------- #
#  PER-CODE RESOLVED FLOOR (review item 6)                                     #
# --------------------------------------------------------------------------- #
def test_a_dead_exchange_code_is_caught_per_code_not_per_universe():
    """Only ZERO was special-cased, which is exactly why EURONEXT/OSE hid: the UNIVERSE
    still resolved to thousands of names.  Losing OSL is 224 of 11,497 = 1.9%, while the
    NORMAL shortfall from the instrument filter and delisted prune is 7-12% -- so a
    universe-level ratio can never see it.  Per code, a dead code loses 100% of itself."""
    #  a healthy run: measured per-code shortfalls, none above the natural worst (13.5%)
    healthy = {'NYSE': 1964, 'NASDAQ': 3429, 'TSX': 616, 'LSE': 2250, 'XETRA': 684,
               'STO': 684, 'ICE': 20, 'PAR': 577, 'AMS': 103, 'BRU': 107, 'LIS': 35,
               'OSL': 224}
    assert un.check_resolved_counts('stock_NA1_EU1', healthy) == []

    #  OSL dies -- a 1.9% hit on the universe, invisible to any total
    dead = dict(healthy, OSL=0)
    problems = un.check_resolved_counts('stock_NA1_EU1', dead)
    assert [p[0] for p in problems] == ['OSL']
    assert problems[0][1] == 224 and problems[0][2] == 0
    universe_level_shortfall = 1 - (sum(dead.values()) / float(sum(healthy.values())))
    assert universe_level_shortfall < 0.03, (
        'the point of per-code checking: this loss is ~2% of the universe and would be '
        'buried under the 7-12% normal attrition')


def test_the_floor_threshold_clears_the_worst_natural_shortfall():
    """Grounded, not picked: NYSE loses 13.5% to the instrument filter (preferreds
    concentrate there); a dead code loses 100%."""
    assert un.RESOLVED_WORST_NATURAL_SHORTFALL < un.RESOLVED_SHORTFALL_WARN_ABOVE < 1.0
    assert un.RESOLVED_SHORTFALL_WARN_ABOVE - un.RESOLVED_WORST_NATURAL_SHORTFALL > 0.2


def test_the_floor_is_silent_for_universes_with_no_per_code_expectation():
    assert un.check_resolved_counts('stock_TEST1', {}) == []
    assert un.check_resolved_counts('stock_FULL1', {}) == []


def test_configdic_carries_the_universe_definition_stamp():
    """The filter NAME alone is no longer sufficient provenance, because four names now
    denote a different membership than they did before 2026-08-02."""
    c = cfg.getDataFetchConfiguration(['x', '-tickerfilter', 'stock_NA1_EU1'])
    assert c['universe'] == 'stock_NA1_EU1'
    assert c['universe_fingerprint'] == un.definition_fingerprint('stock_NA1_EU1')
    assert c['universe_definition_changed'] is True
    assert 'EURONEXT' in c['universe_previous_exchanges']
    assert 'PAR' in c['universe_exchanges'] and 'OSL' in c['universe_exchanges']
    assert c['universe_codes_verified'] == '2026-08-02'
    assert c['universe_expected_count'] == 11497


def test_the_stamp_is_json_able_so_tooling_need_not_import_this_module():
    import json
    for name in un.names():
        json.dumps(un.provenance(name))


def test_an_unchanged_universe_is_stamped_as_unchanged():
    c = cfg.getDataFetchConfiguration(['x', '-tickerfilter', 'stock_US1'])
    assert c['universe_definition_changed'] is False
    assert c['universe_previous_exchanges'] is None


# --------------------------------------------------------------------------- #
#  PROVENANCE ON EVERY ARTIFACT AND EVERY PATH (review item 5)                  #
# --------------------------------------------------------------------------- #
import Sbocker as sb


def test_the_fetch_path_stamp_is_the_active_universe():
    c = cfg.getDataFetchConfiguration(['x', '-tickerfilter', 'stock_NA1_EU1'])
    stamp = sb.universe_provenance_for_run(c)
    assert stamp['universe'] == 'stock_NA1_EU1'
    assert stamp['universe_fingerprint'] == un.definition_fingerprint('stock_NA1_EU1')
    assert set(stamp) == set(sb._UNIVERSE_KEYS)


def test_the_LOAD_path_takes_the_LOADED_panels_universe_not_the_active_flag(capsys):
    """THE CORRECTNESS POINT OF ITEM 5.  `-loadbometric` re-scores a panel fetched for
    whatever universe built it.  Stamping the CURRENT `-tickerfilter` onto loaded data
    would MANUFACTURE provenance -- the exact failure the stamp exists to prevent,
    committed by the stamp itself."""
    c = cfg.getDataFetchConfiguration(['x', '-tickerfilter', 'stock_US1'])
    loaded = {'universe': 'stock_NA1_EU1',
              'universe_fingerprint': un.definition_fingerprint('stock_NA1_EU1'),
              'universe_label': 'x', 'universe_exchanges': ['NYSE'],
              'universe_symbols': None, 'universe_every_exchange': False,
              'universe_expected_count': 11497, 'universe_definition_changed': True,
              'universe_previous_exchanges': ['EURONEXT'],
              'universe_codes_verified': '2026-08-02', 'universe_note': 'x'}
    stamp = sb.universe_provenance_for_run(c, loaded=loaded)
    assert stamp['universe'] == 'stock_NA1_EU1', 'the DATA is the loaded panel'
    assert stamp['universe_fingerprint'] == loaded['universe_fingerprint']
    out = capsys.readouterr().out
    assert 'UNIVERSE MISMATCH' in out, (
        'the filenames are built from -tickerfilter, so a disagreement must be loud')


def test_the_mismatch_banner_compares_FINGERPRINTS_not_just_names(capsys):
    """THE CASE THE FINGERPRINT EXISTS FOR, and the banner used to be blind to it: two
    panels both named `stock_NA1_EU1`, one from either side of the 2026-08-02 restoration,
    differing by 1,046 members.  Same name -> the old name-only comparison said nothing."""
    c = cfg.getDataFetchConfiguration(['x', '-tickerfilter', 'stock_NA1_EU1'])
    pre_fix = {'universe': 'stock_NA1_EU1',            # SAME NAME
               'universe_fingerprint': 'deadbeef0000',  # DIFFERENT DEFINITION
               'universe_label': 'pre-restoration', 'universe_exchanges': ['EURONEXT'],
               'universe_symbols': None, 'universe_every_exchange': False,
               'universe_expected_count': 10451, 'universe_definition_changed': False,
               'universe_previous_exchanges': None,
               'universe_codes_verified': None, 'universe_note': 'x'}
    stamp = sb.universe_provenance_for_run(c, loaded=pre_fix)
    out = capsys.readouterr().out
    assert 'UNIVERSE MISMATCH' in out, (
        'same name / different fingerprint went unreported -- the comparison is still on '
        'the name')
    assert 'DEFINITION (same name' in out
    assert stamp['universe_fingerprint'] == 'deadbeef0000', 'the loaded stamp still wins'

    #  and an exact match stays quiet
    same = dict(pre_fix, universe_fingerprint=c['universe_fingerprint'])
    sb.universe_provenance_for_run(c, loaded=same)
    assert 'UNIVERSE MISMATCH' not in capsys.readouterr().out


def test_an_UNSTAMPED_loaded_panel_is_marked_unknown_not_guessed(capsys):
    """A panel built before 2026-08-02 has no stamp, and its membership cannot be
    recovered from the artifact.  An honest 'unknown' beats a confident wrong answer:
    a reader comparing two runs can at least see that one basis is unestablished."""
    c = cfg.getDataFetchConfiguration(['x', '-tickerfilter', 'stock_NA1_EU1'])
    stamp = sb.universe_provenance_for_run(c, loaded={'tickerfilter': 'stock_NA1_EU1'})
    assert stamp['universe_fingerprint'] == 'unknown-unstamped-panel'
    assert stamp['universe_fingerprint'] != un.definition_fingerprint('stock_NA1_EU1'), (
        'an unstamped panel must NOT inherit the current filter identity')
    assert 'NO UNIVERSE STAMP' in capsys.readouterr().out


def test_the_postRank_stamp_keys_match_the_configdic_stamp_keys():
    """postRank is what the BACKTEST reads -- the artifact most likely to be compared
    across runs -- and it carried only `tickerfilter`, a name whose meaning changed."""
    c = cfg.getDataFetchConfiguration(['x', '-tickerfilter', 'stock_NA1_EU1'])
    for k in sb._UNIVERSE_KEYS:
        assert k in c, '%s must reach configdic so postRank can copy it from resdic' % k
    assert 'universe_fingerprint' in sb._UNIVERSE_KEYS


def test_the_provenance_sidecar_is_written_and_names_every_deliverable(tmp_path,
                                                                      monkeypatch):
    """The human-readable deliverables are named only by a universe NAME, so a sidecar
    carries the fingerprint for the whole set -- including the XLSX, which is produced by
    a 160 KB generator that threading a stamp through would put at risk for no extra
    information."""
    import json
    import postBo as pb
    monkeypatch.chdir(tmp_path)
    resdic = {
        'ntopagg': 2, 'ntopxlsx': 2,
        'postRank': pd.DataFrame({'source': ['A', 'B'], 'AggScore': [1.0, 0.5]}),
        'cdx_df': pd.DataFrame({'source': ['A', 'B'], 'date': pd.Timestamp('2025-01-01')}),
        'SLmeanMscore': pd.DataFrame(), 'SLmeanCscore': pd.DataFrame(),
        'baseurl': 'http://x/', 'api_key': 'k',
        'tickerfilter': 'stock_TEST1', 'datasource': 'fmp',
        'universe': 'stock_TEST1',
        'universe_fingerprint': un.definition_fingerprint('stock_TEST1'),
        'universe_symbols': list(un.symbols('stock_TEST1')),
        'universe_resolved_members': 142,
    }
    #  every API-touching sub-stage is stubbed; we are testing the sidecar only
    monkeypatch.setattr(pb, 'writeBoAggToCSV', lambda *a, **k: None)
    monkeypatch.setattr(pb.ff, 'buildForensicFlagTable', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pb.ff, 'writeForensicFlagsCSV', lambda *a, **k: None)
    monkeypatch.setattr(pb, 'createPresentation', lambda *a, **k: None, raising=False)
    try:
        pb.writeResWrapper(resdic)
    except Exception:
        pass                     # later stages need far more of the pipeline; irrelevant
    hits = sorted(tmp_path.glob('RunProvenance-*_fmp_stock_TEST1.json'))
    assert hits, 'no universe-provenance sidecar was written'
    prov = json.load(open(hits[0]))
    assert prov['universe'] == 'stock_TEST1'
    assert prov['universe_fingerprint'] == un.definition_fingerprint('stock_TEST1')
    assert len(prov['universe_symbols']) == 142
    assert prov['deliverables'], 'the sidecar must name the files it describes'


def test_the_sidecar_is_gitignored_because_it_names_universe_members():
    here = os.path.dirname(os.path.abspath(un.__file__))
    gi = open(os.path.join(here, '.gitignore'), encoding='utf-8').read()
    assert 'RunProvenance-*.json' in gi


def test_the_banner_shouts_about_a_changed_definition():
    b = un.run_banner('stock_NA1_EU1', resolved_count=11497)
    assert 'CHANGED MEANING' in b
    assert 'EURONEXT' in b and 'OSE' in b
    assert 'universe_fingerprint' in b or un.definition_fingerprint('stock_NA1_EU1') in b
    assert 'RESOLVED members      : 11497' in b


def test_the_banner_shouts_that_test_universe_scores_are_not_production_scores():
    b = un.run_banner('stock_TEST1')
    assert 'NOT PRODUCTION SCORES' in b
    assert 'POOL COMPOSITION' in b
    b2 = un.run_banner('stock_US1')
    assert 'NOT PRODUCTION SCORES' not in b2
    assert 'CHANGED MEANING' not in b2


def test_the_banner_warns_about_korea_on_the_full_universe():
    b = un.run_banner('stock_FULL1')
    assert 'Korea' in b
    assert 'OTC' in b


# --------------------------------------------------------------------------- #
#  ASIA -- WIRED 2026-08-05, AND ONLY BECAUSE THE DEDUP BLOCKER CLOSED          #
# --------------------------------------------------------------------------- #
def test_asia_is_wired_with_the_likely_investable_set_only():
    """The blocker was DEDUP, never data.  It closed, so Asia is wired -- but only the
    venues the CEO judged likely investable, and the note must still record that the
    blocker existed and what remains unverified about it."""
    codes = set(un.asia_codes())
    assert {'JPX', 'HKSE', 'KSC', 'KOE', 'ASX', 'SES'} <= codes
    for c in codes:
        assert un.ASIA_CANDIDATE_CODES[c] > 0, '%s must be recorded as data-available' % c
    assert set(un.ASIA_LIKELY_INVESTABLE) == {'JPX', 'HKSE', 'KSC', 'KOE', 'ASX', 'SES'}
    assert (set(un.ASIA_LIKELY_INVESTABLE) & set(un.ASIA_ACCESS_EXCLUDED)) == set()
    assert set(un.exchanges('stock_ASIA1')) == set(un.ASIA_LIKELY_INVESTABLE)
    assert (set(un.exchanges('stock_NA1_EU1_ASIA1'))
            == set(un.exchanges('stock_NA1_EU1')) | set(un.ASIA_LIKELY_INVESTABLE))
    #  The blocker note must record BOTH that it closed and what is still unproven --
    #  "closed" alone would read as "Korea is verified", which it is not.
    b = un.ASIA_BLOCKER.lower()
    assert 'dedup' in b
    assert 'closed' in b
    assert 'unverified' in b, (
        'ASIA_BLOCKER no longer states the open residual: whether FMP serves Korean '
        'preferreds their issuer\'s STATEMENTS is unverified until a Korea fetch is '
        'regression-tested. Deleting that sentence would turn an open gate into an '
        'implied pass.')


def test_a_korean_universe_does_not_resolve_without_the_dedup_marker(
        fixture_df, offline_wrapper):
    """THE DEPENDENCY, ENFORCED.  Korea is admissible only because canonical-choice dedup
    exists, so a universe wiring KSC/KOE must REFUSE to resolve if the Korean canonicity
    marker is gone -- otherwise 196 preferred lines at 30-60% discounts enter a cheapness
    screen.  Nobody can enable Korea by editing the registry alone."""
    import carveOut as _co
    real = _co._non_canonical_tag
    try:
        del _co._non_canonical_tag
        with pytest.raises(Exception) as e:
            offline_wrapper(fixture_df, 'stock_ASIA1', 'all', -1, 'http://x', 'k')
        assert 'Korea' in str(e.value)
    finally:
        _co._non_canonical_tag = real
    #  ...and with the marker present it resolves, preferred line included (dedup drops it
    #  later, at the carve/ranking stage -- membership is not where that happens).
    got = set(_syms(offline_wrapper(fixture_df, 'stock_ASIA1', 'all', -1,
                                    'http://x', 'k')))
    assert {'005930.KS', '005935.KS', '7203.T'} <= got


def test_the_unwired_but_available_codes_are_recorded_rather_than_forgotten():
    """AMEX in particular: 256 statement-bearing US commons that no universe includes,
    including the one called "US ONLY"."""
    assert 'AMEX' in un.US_NOT_WIRED
    assert un._VERIFIED_COUNTS['AMEX'] == 256
    assert 'AMEX' in un.note('stock_US1')
    for c in un.EUROPE_NOT_WIRED + un.US_NOT_WIRED:
        assert c in un._VERIFIED_COUNTS
        for name, d in un.UNIVERSES.items():
            if un.is_every_exchange(name):
                continue
            assert c not in (d['exchanges'] or ())
