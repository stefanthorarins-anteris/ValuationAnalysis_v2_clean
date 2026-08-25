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

import json
import os
import re
from datetime import datetime, timedelta

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


@pytest.fixture(scope='module')
def live_capture_names():
    """symbol -> name from the LIVE 2026-08-04 FMP capture.  WHOLLY OFFLINE (a saved
    pickle); skips rather than passing vacuously if the capture is not on this machine."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'available_traded_raw_2026-08-04.pickle')
    if not os.path.exists(p):
        pytest.skip('live capture absent: %s' % os.path.basename(p))
    d = pd.read_pickle(p)
    return dict(zip(d['symbol'], d['name']))


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
    #  stock_CUR3K added 2026-08-06: it wires KSC/KOE DELIBERATELY, because the Korean
    #  preferred families are one of the dedup cases the ~3,000-name shakedown run exists
    #  to observe. It is Korea-gated like the other two.
    #  stock_CUR6K added 2026-08-21 for the same reason and MORE of it: it raises Korea
    #  25% -> 40% precisely because the one still-unverified half of the Asia work (does an
    #  FMP Korean preferred carry its ISSUER'S statements, so the family GROUPS at all --
    #  ASIA_BLOCKER residual) is answered by having more whole FAMILIES, not more US names.
    asia_universes = {n for n in un.names()
                      if set(un.exchanges(n) or ()) & set(un.ASIA_LIKELY_INVESTABLE)}
    assert asia_universes == {'stock_ASIA1', 'stock_NA1_EU1_ASIA1', 'stock_CUR3K',
                              'stock_CUR6K'}, (
        'an unexpected universe now wires likely-investable Asia: %s' % asia_universes)
    for name in un.names():
        if un.is_every_exchange(name):
            continue                          # FULL is *supposed* to contain everything
        got = set(_syms(offline_wrapper(fixture_df, name, 'all', -1, 'http://x', 'k')))
        #  Never wired anywhere: unwired US venues and the access-excluded Asian ones.
        for leak in ('SIRI', 'OTCX', '2330.TW'):
            assert leak not in got, '%s leaked %s' % (name, leak)
        #  KEYED ON THE CODE, not on "is an Asia universe" (tightened 2026-08-06).  The
        #  loose form assumed every Asia-wiring universe wires ALL of them, which
        #  stock_CUR3K breaks on purpose: it wires KOREA ONLY, because the Korean preferred
        #  families are the dedup case it exists to observe and Japan is not.  Asserting
        #  per-code is strictly stronger anyway -- it now catches a universe that wires JPX
        #  and returns no Japanese name, which the old form could not.
        for asian, code in (('7203.T', 'JPX'), ('005930.KS', 'KSC'), ('005935.KS', 'KSC')):
            if code in (un.exchanges(name) or ()):
                assert asian in got, '%s wires %s but returned no %s' % (name, code, asian)
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
def test_the_test_universe_covers_every_required_category_MEASURED_not_asserted(_isolated_absent_map_state):
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


def test_ALL_FOUR_maps_present_and_fresh_is_an_idempotent_no_op(tmp_path, monkeypatch):
    """The idempotent skip is correct -- but it must be gated on ALL FOUR artifacts
    `buildSectorIndustryMaps` writes, not the two it happened to write first.

    THIS TEST USED TO ASSERT THE BUG (fixed 2026-08-07).  It seeded only sectorsdic
    + industrydic and demanded `False`, which is exactly the condition under which
    isindic/volavgdic can never be created -- so the test PROTECTED the defect that
    shipped `isin_map_n: 0` / `volavg_map_n: 0` and sent 19 issuer groups to the
    alphabetical tiebreak.  It now seeds all four, DATED TODAY so the freshness
    trigger does not fire, and the skip is asserted on that state.

    A FIFTH CONDITION JOINED THE GATE ON 2026-08-14: the CAPTURE SCHEMA.  Presence, age and
    coverage are all properties of the ARTIFACTS; none of them can see a code change to WHICH
    FIELDS the writer pulls, which is how two shipped capture changes silently failed to land
    on 2026-08-10.  So the skip now also requires the stamp on disk to match today's declared
    field set -- and this test seeds it, because "all four maps are present and fresh" is no
    longer the whole statement of "there is nothing to do".  An ABSENT stamp is CHANGED, not
    unchanged, so the un-stamped case is asserted separately below.
    """
    monkeypatch.chdir(tmp_path)
    today = datetime.now().strftime('%Y-%m-%d')
    pd.to_pickle({'Technology': ['A']}, 'sectorsdic_fmp.pickle')
    pd.to_pickle({'A': 'Software'}, f'industrydic_fmp_{today}.pickle')
    pd.to_pickle({'A': 'US0000000001'}, f'isindic_fmp_{today}.pickle')
    pd.to_pickle({'A': {'volAvg': 1000, 'asof': today}}, f'volavgdic_fmp_{today}.pickle')
    fas.write_capture_schema(str(tmp_path), verbose=False)
    monkeypatch.setattr(fas, 'buildSectorIndustryMaps',
                        lambda *a, **k: pytest.fail('must not rebuild when all four present'))
    assert fas.ensure_sector_industry_maps(['A'], 'https://x/', 'KEY') is False
    assert fas.ensure_sector_industry_maps(['A'], 'https://x/', 'KEY',
                                           universe_is_subset=True) is False


def test_a_CAPTURE_SCHEMA_CHANGE_rebuilds_even_when_every_other_condition_says_skip(
        tmp_path, monkeypatch, capsys):
    """THE REGRESSION GUARD FOR THE 2026-08-10 LOSS, at the GATE (not the helper) level.

    Same state as the test above -- all four maps present, fresh, coverage fine -- but the
    stamp on disk records a DIFFERENT captured field set.  That is exactly the 2026-08-10
    machine after a capture change shipped, and the gate must now rebuild ON ITS OWN, with no
    `-force_rebuild_maps` and nobody remembering.  The banner must say TRIGGERED (the gate's
    own judgement), not FORCED.
    """
    monkeypatch.chdir(tmp_path)
    today = datetime.now().strftime('%Y-%m-%d')
    pd.to_pickle({'Technology': ['A']}, 'sectorsdic_fmp.pickle')
    pd.to_pickle({'A': 'Software'}, f'industrydic_fmp_{today}.pickle')
    pd.to_pickle({'A': 'US0000000001'}, f'isindic_fmp_{today}.pickle')
    pd.to_pickle({'A': {'volAvg': 1000, 'asof': today}}, f'volavgdic_fmp_{today}.pickle')
    stale_fields = [f for f in fas.PROFILE_CAPTURE_FIELDS if f != 'mktCap']
    with open(tmp_path / fas.PROFILE_CAPTURE_SCHEMA_FILE, 'w', encoding='utf-8') as fh:
        json.dump({'fingerprint': fas.profile_capture_fingerprint(stale_fields),
                   'fields': stale_fields, 'asof': today}, fh)
    built = []
    monkeypatch.setattr(fas, 'buildSectorIndustryMaps',
                        lambda *a, **k: built.append(True) or ({}, {}))
    assert fas.ensure_sector_industry_maps(['A'], 'https://x/', 'KEY') is True
    out = capsys.readouterr().out
    assert built, 'a capture-schema change must rebuild by itself'
    assert 'CAPTURE SCHEMA' in out and 'mktCap' in out, (
        'the log must name WHICH field changed, not merely that something did')
    assert 'TRIGGERED' in out and 'FORCED by' not in out


def test_a_MISSING_isin_or_volavg_map_does_NOT_count_as_cached(tmp_path, monkeypatch):
    """THE REGRESSION GUARD FOR THE ACTUAL DEFECT.  With sectorsdic + industrydic
    present but isindic/volavgdic absent -- the exact state of the machine that
    produced the 2026-08-07 CUR3K run -- the gate must NOT report 'cached'.  It must
    reach the build (here: the subset refusal, which is the non-API branch)."""
    monkeypatch.chdir(tmp_path)
    today = datetime.now().strftime('%Y-%m-%d')
    pd.to_pickle({'Technology': ['A']}, 'sectorsdic_fmp.pickle')
    pd.to_pickle({'A': 'Software'}, f'industrydic_fmp_{today}.pickle')

    built = {'n': 0}

    def _fake_build(*a, **k):
        built['n'] += 1
        return {}, {}

    monkeypatch.setattr(fas, 'buildSectorIndustryMaps', _fake_build)
    fas.ensure_sector_industry_maps(['A'], 'https://x/', 'KEY')
    assert built['n'] == 1, (
        'sectorsdic+industrydic present must NOT short-circuit the gate while '
        'isindic/volavgdic are missing -- that is the bug that made the ISIN and '
        'volAvg dedup tiebreaks permanently unavailable')


def test_a_STALE_sector_map_does_NOT_count_as_cached(tmp_path, monkeypatch):
    """Presence is not freshness.  The shipped run carved against a 2025-12-10 map,
    240 days old, and nothing said so."""
    monkeypatch.chdir(tmp_path)
    old = (datetime.now() - timedelta(days=fas.MAP_STALE_DAYS + 30)).strftime('%Y-%m-%d')
    pd.to_pickle({'Technology': ['A']}, 'sectorsdic_fmp.pickle')
    for stem in ('industrydic', 'isindic', 'volavgdic'):
        pd.to_pickle({'A': 'x'}, f'{stem}_fmp_{old}.pickle')

    built = {'n': 0}
    monkeypatch.setattr(fas, 'buildSectorIndustryMaps',
                        lambda *a, **k: (built.__setitem__('n', built['n'] + 1), ({}, {}))[1])
    fas.ensure_sector_industry_maps(['A'], 'https://x/', 'KEY')
    assert built['n'] == 1, 'a map older than MAP_STALE_DAYS must warrant a rebuild'


def test_LOW_sector_coverage_does_NOT_count_as_cached(tmp_path, monkeypatch):
    """The 2026-08-07 map covered 84.2% of the universe (41.1% on KOSDAQ) and was
    still treated as a healthy cache.  Coverage below the floor must trigger."""
    monkeypatch.chdir(tmp_path)
    today = datetime.now().strftime('%Y-%m-%d')
    pd.to_pickle({'Technology': ['A']}, 'sectorsdic_fmp.pickle')
    for stem in ('industrydic', 'isindic', 'volavgdic'):
        pd.to_pickle({'A': 'x'}, f'{stem}_fmp_{today}.pickle')

    built = {'n': 0}
    monkeypatch.setattr(fas, 'buildSectorIndustryMaps',
                        lambda *a, **k: (built.__setitem__('n', built['n'] + 1), ({}, {}))[1])
    # 1 of 5 symbols mapped = 20% coverage, far below the floor.
    fas.ensure_sector_industry_maps(['A', 'B', 'C', 'D', 'E'], 'https://x/', 'KEY')
    assert built['n'] == 1, (
        'sector coverage below MIN_SECTOR_COVERAGE_PCT must warrant a rebuild -- '
        'an existing map that covers a fraction of the universe is not a cache hit')


# --------------------------------------------------------------------------- #
#  THE DELISTED PRUNE MUST NAME WHAT IT REMOVES  (2026-08-09)                   #
#                                                                               #
#  It was the last gate that removed universe members with no count, no banner   #
#  and no shipped artifact -- and the end-of-run reconciliation cannot cover for #
#  it, because the prune runs UPSTREAM of Tickers_df so `resolved` is already    #
#  post-prune.                                                                   #
# --------------------------------------------------------------------------- #
def _get_tickers_with_delist(tmp_path, monkeypatch, delisted, rows=None):
    rows = rows or [('AAPL', 'Apple Inc.', 'NASDAQ', 'stock'),
                    ('MCFT', 'MasterCraft Boat Holdings, Inc.', 'NASDAQ', 'stock'),
                    ('GYL.OL', 'Gyldendal ASA', 'OSL', 'stock')]
    at = [dict(symbol=s, name=n, price=1.0, exchange=e, exchangeShortName=e, type=t)
          for s, n, e, t in rows]

    def _fake_get(url, *a, **k):
        if 'available-traded' in url:
            return at
        if 'symbol-lists' in url:
            return [{'symbol': s} for s, _n, _e, _t in rows]
        if 'delisted-companies' in url:
            return [{'symbol': s} for s in delisted]
        return []

    monkeypatch.setattr(gdg, 'safe_get', _fake_get)
    monkeypatch.setattr(fas, 'ensure_sector_industry_maps', lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    out = gdg.get_tickers('fmp', 'https://x/', 'KEY', [], 'stock_NA1_EU1',
                          sfilt='all', mcapf=-1, fn='')
    return out


def test_the_delisted_prune_NAMES_every_symbol_it_removes(tmp_path, monkeypatch, capsys):
    """The house invariant: a name that leaves the universe is named, with its reason, in
    a shipped artifact.  Both channels are checked -- the console banner an operator reads
    and the CSV that travels."""
    df = _get_tickers_with_delist(tmp_path, monkeypatch, ['MCFT', 'NOT_IN_UNIVERSE'])
    assert 'MCFT' not in set(df['symbol']), 'the prune must still prune'
    assert {'AAPL', 'GYL.OL'} <= set(df['symbol'])

    out = capsys.readouterr().out
    assert 'DELISTED PRUNE' in out
    assert 'MCFT' in out, 'a removed symbol was not NAMED on the console'
    assert '1 of 3' in out, 'the count of removed rows is not stated'

    rec = sorted(tmp_path.glob('DelistedPrune_*.csv'))
    assert rec, 'no shipped record of the prune was written'
    got = pd.read_csv(rec[-1])
    assert list(got['symbol']) == ['MCFT'], (
        'the record must name what WE removed, not the raw vendor list -- '
        'NOT_IN_UNIVERSE was on the vendor list and was never ours to remove')
    assert (got['reason'] == 'delisted_prune').all()
    assert int(got['vendor_list_size'].iloc[0]) == 2


def test_the_prune_banner_STATES_the_page_0_limitation(tmp_path, monkeypatch, capsys):
    """Keeping page-0-only is a separate (deferred) question; reading the prune as
    COMPLETE is the defect.  The limitation is stated on the console AND inside the CSV,
    because the two travel to different readers."""
    _get_tickers_with_delist(tmp_path, monkeypatch, ['MCFT'])
    out = capsys.readouterr().out
    assert 'page=0' in out and 'PARTIAL' in out
    got = pd.read_csv(sorted(tmp_path.glob('DelistedPrune_*.csv'))[-1])
    assert 'PAGE 0 ONLY' in got['coverage_note'].iloc[0]


def test_the_prune_reports_the_ZERO_case_too(tmp_path, monkeypatch, capsys):
    """"0 removed" is a measurement; silence is not, and the two were indistinguishable.
    A run where the gate removed nothing must still say so -- otherwise a broken endpoint
    returning [] looks exactly like a clean universe."""
    df = _get_tickers_with_delist(tmp_path, monkeypatch, [])
    assert len(df) == 3
    out = capsys.readouterr().out
    assert 'DELISTED PRUNE' in out and '0 of 3' in out
    assert 'REMOVED: none' in out
    assert 'page=0' in out, 'the scope caveat must print even on the zero case'


def test_the_prune_record_is_reachable_by_the_transfer_allowlist(tmp_path, monkeypatch):
    """The artifact only counts if it SHIPS -- and since 2026-08-10 (CEO) that means it is
    at the REPO ROOT with its OWN top-level pattern, not inside `output/`.

    The ruling's evidence: on the 2026-08-10 run `output/` did not reach Drive at all, so
    riding a directory that "ships whole" lost the record for that date while every
    root-level artifact survived.  This test therefore pins BOTH halves -- the file is
    written where glob.glob can see it, and the manifest carries a pattern that matches it."""
    _get_tickers_with_delist(tmp_path, monkeypatch, ['MCFT'])
    _rec = tmp_path / ('DelistedPrune_%s.csv' % datetime.today().strftime('%Y-%m-%d'))
    assert _rec.exists(), 'the prune record is not at the repo root'
    assert not (tmp_path / 'output' / _rec.name).exists(), (
        'the prune record is still being written into output/, which is the directory that '
        'did not travel on 2026-08-10')
    assert sorted(tmp_path.glob('delisted_tickers_*.csv')), 'raw vendor list not written'
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Sbocker.py'),
              encoding='utf-8', errors='ignore') as _f:
        src = _f.read()
    assert "'delisted_tickers_*.csv'" in src, (
        'the raw vendor list has no transfer pattern -- it is top-level and dated, so a '
        'glob genuinely matches it (unlike one aimed into output/)')
    assert "'DelistedPrune_*.csv'" in src, (
        'the prune record is at root now and MUST have its own top-level manifest pattern; '
        'riding output/ is exactly what failed on 2026-08-10')


# --------------------------------------------------------------------------- #
#  CAPTURE-ONLY PROFILE FIELDS  (2026-08-08 price/currency, 2026-08-09 +5)      #
#                                                                               #
#  The whole safety argument for capture-now-wire-later is TWO claims:           #
#    (1) the fields actually LAND, in the volavgdic entry, under the SAME asof;   #
#    (2) capturing them changes NOTHING, because the single loader every consumer #
#        goes through ignores every key but volAvg/asof.                          #
#  Neither was covered before this block -- 90b0d5f's message asserted the        #
#  enriched shape was tested, and no test in the suite built one.                 #
# --------------------------------------------------------------------------- #
#  A payload shaped like the measured 2026-08-09 v3/profile sample: an LSE line
#  quoting in GBp (11 of 100 did), a US line, and a name whose profile omits the new
#  fields entirely -- the vendor-drops-a-field case, which must give None, not KeyError.
_PROFILE_ROWS = [
    {'symbol': 'AAA.L', 'sector': 'Technology', 'industry': 'Software',
     'isin': 'GB00AAA00001', 'volAvg': 123456, 'price': 412.5, 'currency': 'GBp',
     'isActivelyTrading': True, 'exchange': 'London Stock Exchange',
     'exchangeShortName': 'LSE', 'country': 'GB', 'beta': 1.234,
     #  the 2026-08-14 wave.  `mktCap` deliberately does NOT equal price x any round share
     #  count: it is the TRADED LINE's own cap, and the whole reason to capture it is that
     #  mktCap/price need not agree with the statement's weightedAverageShsOut.
     'mktCap': 1236789000, 'ipoDate': '1998-06-19', 'companyName': 'AAA Holdings plc',
     'isAdr': False, 'isEtf': False, 'isFund': False,
     'cik': '0000012345', 'cusip': 'G0000A101', 'fullTimeEmployees': '4321'},
    {'symbol': 'BBB', 'sector': 'Technology', 'industry': 'Software',
     'isin': 'US00BBB00001', 'volAvg': 987654, 'price': 31.2, 'currency': 'USD',
     'isActivelyTrading': True, 'exchange': 'NASDAQ Global Select',
     'exchangeShortName': 'NASDAQ', 'country': 'US', 'beta': -0.965,
     'mktCap': 8400000000, 'ipoDate': '2012-01-31', 'companyName': 'BBB Inc.',
     'isAdr': False, 'isEtf': False, 'isFund': False,
     'cik': '0000054321', 'cusip': '00BBB0101', 'fullTimeEmployees': '150'},
    {'symbol': 'CCC', 'sector': 'Technology', 'industry': 'Software',
     'isin': 'US00CCC00001', 'volAvg': 10},
]
#  DERIVED FROM THE DECLARATION, never hand-listed alongside it (2026-08-14).  A hand-listed
#  copy is how the rebuild gate came to be checking a SUBSET of what the writer writes, and a
#  test that carries its own stale copy of the field set stops testing the field set.
_EXTRA_KEYS = (('price', 'currency', 'isActivelyTrading', 'exchange',
                'exchangeShortName', 'country', 'beta')
               + tuple(fas.PROFILE_EXTRA_CAPTURE_FIELDS))


def _build_with_profiles(tmp_path, monkeypatch, rows):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fas, '_fetch_profiles_batched',
                        lambda *a, **k: (list(rows), 1, True, set()))
    fas.buildSectorIndustryMaps(['x'], 'https://x/', 'KEY')
    return pd.read_pickle(sorted(tmp_path.glob('volavgdic_fmp_*.pickle'))[-1])


def test_the_capture_only_profile_fields_LAND_under_the_SAME_asof(tmp_path, monkeypatch):
    """Claim (1).  All seven capture-only fields must be IN the volAvg entry -- not in
    sidecar artifacts -- because a traded value, a venue and a beta are only jointly
    meaningful at ONE point in time, and separate artifacts would reintroduce exactly the
    stale-vs-fresh comparison the per-entry `asof` was added to prevent."""
    on_disk = _build_with_profiles(tmp_path, monkeypatch, _PROFILE_ROWS)
    e = on_disk['AAA.L']
    assert e['volAvg'] == 123456
    assert e['price'] == 412.5 and e['currency'] == 'GBp'
    assert e['isActivelyTrading'] is True
    assert e['exchange'] == 'London Stock Exchange'
    assert e['exchangeShortName'] == 'LSE'      # a SEPARATE field from `exchange`
    assert e['country'] == 'GB' and e['beta'] == 1.234
    #  the 2026-08-14 wave, in the SAME entry under the SAME asof for the same reason.
    assert e['mktCap'] == 1236789000 and e['ipoDate'] == '1998-06-19'
    assert e['companyName'] == 'AAA Holdings plc'
    assert e['isAdr'] is False and e['isEtf'] is False and e['isFund'] is False, (
        'a FALSE boolean must survive as False -- an absent key and a False value are '
        'different facts and a consumer must be able to tell them apart')
    assert e['cik'] == '0000012345' and e['cusip'] == 'G0000A101'
    assert e['fullTimeEmployees'] == '4321'
    #  ONE date for all of them -- the property that makes sharing the entry the point.
    assert set(e) == {'asof'} | {'volAvg'} | set(_EXTRA_KEYS)
    assert e['asof'] == on_disk['BBB']['asof'] == on_disk['CCC']['asof']
    assert e['asof'] == datetime.today().strftime('%Y-%m-%d')
    #  A profile that OMITS the fields yields None, never a KeyError and never a
    #  fabricated default -- absence must stay legible as absence.
    assert all(on_disk['CCC'][k] is None for k in _EXTRA_KEYS)


def test_capturing_the_extra_fields_changes_NOTHING_a_consumer_sees(tmp_path, monkeypatch):
    """Claim (2), and it is what makes 'capture only' a PROPERTY rather than an intention.

    `carveOut._load_volavg_map` is the SINGLE seam through which this pickle reaches every
    consumer (grepped: nothing else opens volavgdic_fmp_*.pickle), so if the loader's
    output is identical with and without the extra keys, no consumer can be affected.

    NON-VACUITY: the same assertion is made against a bare {'volAvg','asof'} pickle built
    from the same rows, so a loader that started leaking the extra keys through would fail
    here rather than silently agreeing with itself.
    """
    on_disk = _build_with_profiles(tmp_path, monkeypatch, _PROFILE_ROWS)
    enriched = co._load_volavg_map(
        str(sorted(tmp_path.glob('volavgdic_fmp_*.pickle'))[-1]))
    stripped_path = tmp_path / 'stripped.pickle'
    pd.to_pickle({s: {'volAvg': v['volAvg'], 'asof': v['asof']}
                  for s, v in on_disk.items()}, stripped_path)
    assert enriched == co._load_volavg_map(str(stripped_path))
    assert enriched['AAA.L'] == (123456, datetime.today().strftime('%Y-%m-%d'))
    #  and the loaded value is a 2-tuple, so nothing downstream can even reach a new field
    assert all(len(v) == 2 for v in enriched.values())
    #  non-vacuity: the enriched pickle really did carry the extra keys
    assert all(k in on_disk['AAA.L'] for k in _EXTRA_KEYS)


def test_isActivelyTrading_is_captured_but_is_NOT_wired_as_a_liveness_filter(tmp_path,
                                                                            monkeypatch):
    """MEASURED WARNING, pinned so it cannot be forgotten: on the 2026-08-09 sample
    `isActivelyTrading` was True on 100/100 rows INCLUDING all 39 sampled names that
    FAILED the previous fetch.  It therefore rejects nothing on this population, and a
    filter that never fires is worse than absent because it LOOKS like coverage.

    This test pins the ABSENCE of such a filter: a symbol with isActivelyTrading False
    must still be captured and must still load exactly like any other name.
    """
    rows = list(_PROFILE_ROWS) + [
        {'symbol': 'DEAD', 'sector': 'Technology', 'industry': 'Software',
         'volAvg': 5, 'isActivelyTrading': False, 'country': 'US'}]
    on_disk = _build_with_profiles(tmp_path, monkeypatch, rows)
    assert on_disk['DEAD']['isActivelyTrading'] is False, 'the False value must be kept'
    loaded = co._load_volavg_map(
        str(sorted(tmp_path.glob('volavgdic_fmp_*.pickle'))[-1]))
    assert loaded['DEAD'] == (5, datetime.today().strftime('%Y-%m-%d')), (
        'isActivelyTrading=False changed what a consumer sees -- it has been WIRED as a '
        'liveness/delisting filter, which the measured sample says it cannot support')


def test_the_volavg_pickle_still_has_exactly_ONE_reading_seam():
    """The assumption the test above RESTS ON, made checkable instead of assumed.

    "Capturing extra keys cannot affect a consumer" is only true while `_load_volavg_map`
    is the sole reader -- a second module doing its own `pd.read_pickle` on this artifact
    would see the raw entries and the argument silently lapses.  So the set of modules
    naming the artifact is pinned: findAllSectors WRITES it, carveOut LOADS it, Sbocker
    SHIPS it, and there is no fourth.

    *** WHAT THIS ASSERTION CANNOT TELL APART, recorded 2026-08-24 after it fired on a
    COMMENT. ***  It is a SUBSTRING SCAN over raw file text, so it counts any module that
    MENTIONS the artifact, not any module that READS it.  It went red when `postBo` acquired
    a prose comment naming the glob while explaining that the traded-value floor must not run
    on a point-in-time panel -- a note ABOUT the reader, in a module that does not read.  The
    scan stays (a crude over-broad guard on a capture-only guarantee is the right trade, and
    the noise is one comment every few months), and the comment was reworded rather than the
    module set widened: widening it to include `postBo` would have retired the guard for a
    module that might one day genuinely read the pickle.
    *** AND WHAT IT STRUCTURALLY CANNOT SEE AT ALL: a second reader INSIDE an already-listed
    module. ***  `carveOut` has had two independent loaders since 2026-08-13 --
    `_load_volavg_map` (narrow, capture-only, feeds the dedup tie-breaks) and
    `_load_volavg_profile_map` (wide, feeds `dollar_volume_frame`) -- and this test counts
    modules, so it has never had an opinion about them.  Two independent "newest on disk"
    resolutions that disagreed would be a vintage mismatch of exactly the kind this guard
    exists to prevent, one level down.  `test_the_two_volavg_loaders_resolve_to_the_SAME_file`
    below is the assertion that actually covers it.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent
    naming = {p.name for p in sorted(root.glob('*.py'))
              if not p.name.startswith('test_')
              and 'volavgdic_fmp_' in p.read_text(encoding='utf-8', errors='ignore')}
    assert naming == {'findAllSectors.py', 'carveOut.py', 'Sbocker.py'}, (
        'the volavgdic artifact gained or lost a naming module (%s) -- if a NEW module '
        'reads it directly, the capture-only guarantee no longer follows from '
        '_load_volavg_map alone' % sorted(naming))


def test_the_two_volavg_loaders_resolve_to_the_SAME_file(tmp_path, monkeypatch):
    """*** THE GAP THE MODULE-COUNT SEAM TEST STRUCTURALLY CANNOT SEE (2026-08-24). ***

    `carveOut` holds TWO independent readers of the volAvg capture: `_load_volavg_map`
    (narrow -- drops every profile field, which is what makes "capturing extra keys cannot
    affect the dedup tie-breaks" a property rather than a hope) and
    `_load_volavg_profile_map` (wide -- price and currency, which is what
    `dollar_volume_frame` needs).  Two readers is DELIBERATE and is the right design; two
    readers that each resolve "newest on disk" INDEPENDENTLY is the risk, because a run whose
    dedup used one capture and whose $1M/day floor used another would screen names on a
    different day's liquidity than it de-duplicated them with -- silently, and in the same
    vintage-mismatch class as the point-in-time lookahead the floor already had to close.

    Today they cannot disagree: the two resolution blocks are the same four lines (repo root
    first, then CWD, `sorted(...)[-1]`).  Nothing asserts that, so this does -- over a
    directory holding SEVERAL dated captures, which is the only state in which a divergence
    could show.

    WHAT THIS CANNOT DETECT: a disagreement caused by a file appearing BETWEEN the two calls,
    and whether the file both of them choose is the RIGHT one for the run (that is
    `profile_map_for_run`, which takes the run date explicitly and is a different question)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(co, '_MODULE_DIR', str(tmp_path))
    for d in ('2026-08-13', '2026-08-19', '2026-08-22'):
        pd.to_pickle({'AAA': {'volAvg': 1000, 'asof': d, 'price': 1.0, 'currency': 'USD'}},
                     tmp_path / ('volavgdic_fmp_%s.pickle' % d))
    narrow = co._load_volavg_map()
    wide = co._load_volavg_profile_map()
    assert narrow and wide, 'neither loader found a capture -- the fixture proves nothing'
    #  Both must have read the SAME (newest) capture.  The as-of date is the observable that
    #  says which file each one opened.
    assert narrow['AAA'][1] == wide['AAA']['asof'] == '2026-08-22', (
        'the two volAvg loaders resolved to DIFFERENT captures (narrow=%r, wide=%r) -- the '
        'de-dup and the traded-value floor would be reading two different days'
        % (narrow['AAA'][1], wide['AAA']['asof']))


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
#  COVERAGE SCOPE -- the PIT/backtest carve (2026-08-22)                        #
# --------------------------------------------------------------------------- #
#  THE BUG THESE PIN.  The guard above measures coverage over the WHOLE pool, which assumes
#  every pool member is a name the map COULD cover.  For the PIT/backtest carve that is false:
#  `depth_horizon_grid` ranks live survivors UNION delisted-registry entities, and the sector
#  map is built from FMP company PROFILES, which a delisted entity has none of.  So the guard
#  aborted the grid stage on the 08-20 (39.8%) and 08-22 (45.9%) runs -- with the SAME map
#  measuring 90.8% of the active universe and 100.0% of the LIVE carve on those same runs.
#  Uncovered was 2,088 in both runs on pools of 3,470 and 3,857: an invariant of the registry,
#  not a property of the map.
#
#  THE SHAPE OF THESE TESTS IS DELIBERATE.  Asserting "no exception" would pass on a guard that
#  had simply been deleted, so each one asserts (a) the abort still fires when the map really is
#  the wrong artifact FOR THE NAMES IT CAN SPEAK ABOUT, and (b) the leak the scope admits is
#  announced and counted, never silent.
def _pit_carve_inputs(n_live, n_dead, live_covered):
    """A PIT-shaped pool: live names (the map can cover them) + dead ones (it cannot).

    Mirrors the real proportions rather than a toy: 45% live / 55% dead is the 08-22 run
    (1,769 covered of 3,857), and the live half is FULLY covered, as the live carve measured.
    """
    live = ['L%04d' % i for i in range(n_live)]
    dead = ['D%04d' % i for i in range(n_dead)]
    pool = live + dead
    bo = pd.DataFrame({'source': pool, 'BoScore': [1.0] * len(pool)})
    tickers = pd.DataFrame({'symbol': pool, 'name': ['N' + s for s in pool]})
    cdx = pd.DataFrame({'source': pool, 'date': pd.Timestamp('2025-01-01'),
                        'marketCap': 1e8, 'totalStockholdersEquity': 5e7,
                        'totalAssets': 1e8, 'revenue': 5e7,
                        'weightedAverageShsOut': 1e6, 'netIncome': 1e6})
    smap = {s: 'Technology' for s in live[:live_covered]}
    return bo, cdx, tickers, set(live), smap


def test_whole_pool_measurement_still_ABORTS_on_a_pit_pool_when_no_scope_is_given(
        tmp_path, monkeypatch):
    """The pre-fix behaviour, kept as the control: with coverage_scope=None nothing changed.

    This is what the pipeline hit.  If this test ever stops raising, the fix has silently
    become a general relaxation of the guard instead of a scoped measurement."""
    monkeypatch.chdir(tmp_path)
    bo, cdx, tickers, live, smap = _pit_carve_inputs(1769, 2088, 1769)
    monkeypatch.setattr(co, '_load_sector_map', lambda *a, **k: smap)
    monkeypatch.setattr(co, '_load_industry_map', lambda *a, **k: {})
    with pytest.raises(RuntimeError) as e:
        co.partition_universe(bo, cdx, tickers)
    assert 'covers only' in str(e.value)
    assert '1769 of 3857' in str(e.value), str(e.value)


def test_scoping_coverage_to_the_live_sources_lets_the_pit_carve_RUN(tmp_path, monkeypatch,
                                                                    capsys):
    """The fix.  Same inputs, same map, scope = the live sources -> 100% and no abort."""
    monkeypatch.chdir(tmp_path)
    bo, cdx, tickers, live, smap = _pit_carve_inputs(1769, 2088, 1769)
    monkeypatch.setattr(co, '_load_sector_map', lambda *a, **k: smap)
    monkeypatch.setattr(co, '_load_industry_map', lambda *a, **k: {})

    part = co.partition_universe(bo, cdx, tickers, coverage_scope=live)
    out = capsys.readouterr().out

    d = part['diagnostics']
    assert d['sector_coverage'] == (1769, 1769), d['sector_coverage']
    assert d['n_coverage_out_of_scope'] == 2088, d['n_coverage_out_of_scope']
    assert 'coverage: 1769 of 1769' in out, out[-1200:]
    assert 'SCOPED' in out, out[-1200:]


def test_the_scoped_carve_ANNOUNCES_the_sectorless_leak_into_general(tmp_path, monkeypatch,
                                                                    capsys):
    """The COST of the scope, asserted as a fact about the returned pool -- not as a log line.

    The out-of-scope names are not excluded: with no sector they are labelled `general`.  A
    reader must not be able to mistake this pool for a sector-clean one, so the leak has to be
    (a) really in the pool and (b) named at runtime."""
    monkeypatch.chdir(tmp_path)
    bo, cdx, tickers, live, smap = _pit_carve_inputs(100, 40, 100)
    monkeypatch.setattr(co, '_load_sector_map', lambda *a, **k: smap)
    monkeypatch.setattr(co, '_load_industry_map', lambda *a, **k: {})

    #  dedup=False so the assertion is about LABELLING, not about issuer collapse: every
    #  synthetic name here carries the identical market cap, so the issuer de-dup legitimately
    #  folds them into one survivor and would hide the leak this test is measuring.
    part = co.partition_universe(bo, cdx, tickers, coverage_scope=live, dedup=False)
    out = capsys.readouterr().out

    gen = set(part['general']['source'])
    dead_in_general = {s for s in gen if s.startswith('D')}
    #  (a) the leak is real and total: every uncoverable name is in `general`.
    assert len(dead_in_general) == 40, len(dead_in_general)
    #  (b) and the run said so, on stdout, in the loud form.
    assert 'CANNOT BE CARVED' in out, out[-1500:]
    assert 'NOT sector-clean' in out, out[-1500:]
    assert '40' in out


def test_a_genuinely_poisoned_map_still_ABORTS_even_with_a_scope(tmp_path, monkeypatch):
    """The scope must not be a way to survive the failure the guard exists for.

    Same PIT pool, but the map now covers only 1.6% of the LIVE names -- the test-universe
    poisoning case.  Scoped or not, that is the wrong artifact and must refuse."""
    monkeypatch.chdir(tmp_path)
    bo, cdx, tickers, live, smap = _pit_carve_inputs(1000, 1000, 16)
    monkeypatch.setattr(co, '_load_sector_map', lambda *a, **k: smap)
    monkeypatch.setattr(co, '_load_industry_map', lambda *a, **k: {})
    with pytest.raises(RuntimeError) as e:
        co.partition_universe(bo, cdx, tickers, coverage_scope=live)
    assert 'covers only' in str(e.value)
    assert '16 of 1000' in str(e.value), str(e.value)


def test_thin_coverage_inside_the_scope_still_WARNS(tmp_path, monkeypatch, capsys):
    """Between abort and healthy the warning must still fire, measured over the scope."""
    monkeypatch.chdir(tmp_path)
    bo, cdx, tickers, live, smap = _pit_carve_inputs(1000, 500, 600)   # 60% of the live names
    monkeypatch.setattr(co, '_load_sector_map', lambda *a, **k: smap)
    monkeypatch.setattr(co, '_load_industry_map', lambda *a, **k: {})
    co.partition_universe(bo, cdx, tickers, coverage_scope=live)
    out = capsys.readouterr().out
    assert 'THIN SECTOR-MAP COVERAGE' in out, out[-1200:]
    assert '600 of 1000' in out, out[-1200:]


def test_a_scope_that_matches_nothing_is_REFUSED_not_passed_vacuously(tmp_path, monkeypatch):
    """The one way the scope could disable the guard rather than aim it.

    An empty intersection scores frac = 1.0 and would sail through both the abort and the
    warning, so a caller that passed the wrong key space would get a universe nothing had
    checked -- exactly the silent substitution the guard exists to prevent."""
    monkeypatch.chdir(tmp_path)
    bo, cdx, tickers, live, smap = _pit_carve_inputs(50, 50, 50)
    monkeypatch.setattr(co, '_load_sector_map', lambda *a, **k: smap)
    monkeypatch.setattr(co, '_load_industry_map', lambda *a, **k: {})
    with pytest.raises(ValueError) as e:
        co.partition_universe(bo, cdx, tickers, coverage_scope={'NOT_A_SOURCE'})
    assert 'matches NONE' in str(e.value), str(e.value)


def test_the_pit_carve_caller_actually_passes_the_live_sources(monkeypatch):
    """AST guard on the WIRING, because the fix is only a fix at the call site.

    carveOut's default is unchanged, so `depth_horizon_grid.rank_all_anchors` passing nothing
    would leave the grid stage exactly as broken while every carveOut test above passed."""
    import ast as _ast
    import inspect
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent / 'baseline_tools'
           / 'depth_horizon_grid.py').read_text(encoding='utf-8')
    tree = _ast.parse(src)
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == 'rank_all_anchors')
    calls = [c for c in _ast.walk(fn)
             if isinstance(c, _ast.Call)
             and getattr(c.func, 'id', None) == 'carve_general_universe']
    assert len(calls) == 1, 'expected exactly one carve call in rank_all_anchors'
    kw = {k.arg: k for k in calls[0].keywords}
    assert 'coverage_scope' in kw, (
        'rank_all_anchors calls carve_general_universe WITHOUT coverage_scope -- the PIT '
        'abort is back, whatever carveOut can do')
    assert getattr(kw['coverage_scope'].value, 'id', None) == 'live_sources', (
        'coverage_scope must be the LIVE sources; anything else re-measures the wrong set')


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
    #  The two properties are now separated.  The original proxy conflated them:
    #    - is_explicit_list_panel: True iff defined by an explicit ticker list
    #    - is_production_panel: True iff this is the production universe
    #  The proxy was insufficient because stock_CUR3K is curated by a SAMPLING RULE,
    #  not a list, so un.symbols('stock_CUR3K') returns None and it read as production.
    for name in un.names():
        path = 'Bometric_dic-fmp_%s_all_2026-01-08_len1.pickle' % name
        # Both properties must hold independently
        assert vtu.is_explicit_list_panel(path) == (un.symbols(name) is not None), name
        assert vtu.is_production_panel(path) == (name == un.DEFAULT_UNIVERSE), name
    prod = vtu.newest_panel(production_only=True)
    if prod is not None:
        # Production panel must NOT be explicit-list-curated
        assert not vtu.is_explicit_list_panel(prod)
        # And it MUST be the production universe
        assert vtu.is_production_panel(prod)


#  PRODUCTION-ONLY panel, matching what `derive_divergent` now selects (fix, 2026-08-04).
#  Skipping on `newest_panel()` was wrong for THIS test: a machine holding only a curated
#  `stock_TEST1` panel would not skip, and would then fail with 'reconciliation unavailable' or
#  -- worse, before the fix -- with a circular over-declaration.  The guard must ask for the
#  kind of panel the derivation actually needs.
@pytest.mark.skipif(vtu.newest_panel(production_only=True) is None,
                    reason='no saved PRODUCTION Bometric panel on this machine '
                           '(gitignored, ~140MB; a curated test-universe panel cannot '
                           'stand in -- see verify_test_universe.derive_divergent)')
def test_the_declared_open_groups_RECONCILE_with_a_fresh_derivation_both_ways(_isolated_absent_map_state):
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


# --------------------------------------------------------------------------- #
#  THE FLOOR KNOWS ABOUT SAMPLING (2026-08-07)                                 #
#                                                                              #
#  The floor compared a POST-SAMPLE count against a WHOLE-EXCHANGE number, so   #
#  stock_CUR3K reported 5 of its 8 codes 75-83% SHORT on every run -- all five  #
#  false.  A guard that cries wolf is a guard the operator learns to skip, and  #
#  then the next dead code hides in the noise of the guard built to find it.    #
# --------------------------------------------------------------------------- #
def _sampled_share(name, code):
    """What `code` delivers when the sample works exactly as designed."""
    return int(round(un.expected_resolved_count(name, code)))


def test_a_sampled_universe_is_silent_when_every_code_delivers_its_share():
    """THE FALSE-POSITIVE THIS FIX REMOVES.  NYSE/NASDAQ/LSE at 17% and KSC/KOE at 25%
    deliver a FRACTION of the venue by design; TSX/PAR/AMS are taken whole."""
    healthy = {c: _sampled_share('stock_CUR3K', c)
               for c in un.exchanges('stock_CUR3K')}
    #  the shares are the sampled ones, not the whole-exchange counts
    assert healthy['NYSE'] == 386 and healthy['NASDAQ'] == 658 and healthy['LSE'] == 383
    assert healthy['KSC'] == 214 and healthy['KOE'] == 100
    assert healthy['TSX'] == 662 and healthy['PAR'] == 577 and healthy['AMS'] == 103
    assert un.check_resolved_counts('stock_CUR3K', healthy) == []

    #  and the PRE-FIX comparison would have screamed on all five sampled codes
    prefix_short = sorted(
        c for c in un.exchanges('stock_CUR3K')
        if 1.0 - (healthy[c] / float(un._VERIFIED_COUNTS[c]))
        > un.RESOLVED_SHORTFALL_WARN_ABOVE)
    assert prefix_short == ['KOE', 'KSC', 'LSE', 'NASDAQ', 'NYSE']


def test_a_dead_SAMPLED_code_still_screams():
    """The entire value of the guard.  Scaling the expectation moves the THRESHOLD; it
    must not blunt the dead-code signal -- 0 names is 100% short of a sampled expectation
    exactly as it is 100% short of a verified one."""
    healthy = {c: _sampled_share('stock_CUR3K', c)
               for c in un.exchanges('stock_CUR3K')}
    for dead_code in ('KOE', 'NYSE', 'LSE', 'PAR'):
        dead = dict(healthy, **{dead_code: 0})
        problems = un.check_resolved_counts('stock_CUR3K', dead)
        assert [p[0] for p in problems] == [dead_code], (
            '%s died and the floor did not catch it' % dead_code)
        assert problems[0][2] == 0
        assert problems[0][3] == 1.0, 'a dead code is 100% short of its expectation'
        #  reported against the SAMPLED expectation, which is what the message now says
        assert problems[0][1] == _sampled_share('stock_CUR3K', dead_code)

    #  a code merely THINNED past the cut is caught too (not just the zero case)
    thinned = dict(healthy, NASDAQ=int(healthy['NASDAQ'] * 0.5))
    assert [p[0] for p in un.check_resolved_counts('stock_CUR3K', thinned)] == ['NASDAQ']


#  THE UNIVERSES THAT CARRY A `sample` DICT.  DECLARED rather than derived, so that adding
#  one is a deliberate edit HERE as well as in the registry -- several tests below assert
#  properties of the UNSAMPLED path, and a new sampled universe silently joining the set
#  would quietly shrink their coverage rather than fail.  stock_CUR3K 2026-08-06,
#  stock_CUR6K 2026-08-21.
SAMPLED_UNIVERSES = frozenset(('stock_CUR3K', 'stock_CUR6K'))


def test_the_declared_sampled_universes_are_exactly_the_ones_with_a_sample_dict():
    """Both directions.  A universe with a rate but not declared here would weaken every
    test that keys off SAMPLED_UNIVERSES; a name declared here that no longer samples is
    dead configuration -- the EURONEXT lesson applied to a test constant."""
    assert {n for n in un.names() if un.sample_rates(n)} == set(SAMPLED_UNIVERSES)


def test_sampling_changes_nothing_for_a_universe_with_no_sample_dict():
    """BIT-IDENTICAL, asserted rather than assumed: every universe outside
    SAMPLED_UNIVERSES has no `sample` dict, so its rate is 1.0 and the arithmetic must be
    the pre-fix one."""
    for name in un.names():
        if un.sample_rates(name):
            assert name in SAMPLED_UNIVERSES, (
                'a new sampled universe appeared -- re-check this test still covers the '
                'unsampled path, then add it to SAMPLED_UNIVERSES: %s' % name)
            continue
        for c in (un.exchanges(name) or ()):
            assert un.expected_resolved_count(name, c) == float(
                un._VERIFIED_COUNTS.get(c, 0))

    #  and the returned tuples match the PRE-FIX formula exactly, recomputed here
    for name in un.names():
        if un.sample_rates(name):
            continue
        codes = un.exchanges(name)
        if not codes:
            continue
        resolved = {c: int(un._VERIFIED_COUNTS.get(c, 0) * 0.9) for c in codes}
        resolved[codes[0]] = 0                       # one genuinely dead code
        prefix = []
        for c in codes:
            v = un._VERIFIED_COUNTS.get(c, 0)
            if v <= 0:
                continue
            r = int(resolved.get(c, 0))
            sf = 1.0 - (r / float(v))
            if sf > un.RESOLVED_SHORTFALL_WARN_ABOVE:
                prefix.append((c, v, r, sf))
        assert un.check_resolved_counts(name, resolved) == prefix, name


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


def test_the_provenance_sidecar_records_the_veto_regime(tmp_path, monkeypatch):
    """*** THE VISIBILITY RULE, AFTER THE FLIP (CEO, 2026-08-07). ***  The Stage-1 veto now
    ships ON for the general pool and ejects 58.4% of it.  While its default was OFF, turning it
    on WAS the visible event; with the default ON, only the artifact can say which regime
    produced a top-100 -- and without this stamp a vetoed and an un-vetoed run are
    INDISTINGUISHABLE on the one axis that changed the pool.  This test is where the old
    `assert sv.ENABLED is False` guarantee went.

    Also pins the distinction the counts alone cannot carry: a cohort the veto DECLINED to gate
    (`applies=False`) versus one it gated and found clean.  Both show `n_ejected == 0`."""
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
        'stage1_veto': {
            'general': {'pool': 'general', 'enabled': True, 'applies': True,
                        'not_applicable_reason': None, 'n_in': 1545, 'n_ejected': 902,
                        'n_out': 643, 'by_flag': {'uCurrentRatio': 300},
                        'ejected': ['UHS'], 'n_short_window': {'uInterestCoverage': 400},
                        'short_window': {'UHS': {'uInterestCoverage': 2}}},
            'REIT': {'pool': 'REIT', 'enabled': True, 'applies': False,
                     'not_applicable_reason': 'structurally undefined on this cohort',
                     'n_in': 49, 'n_ejected': 0, 'n_out': 49, 'by_flag': {},
                     'ejected': [], 'n_short_window': {}, 'short_window': {}},
        },
    }
    monkeypatch.setattr(pb, 'writeBoAggToCSV', lambda *a, **k: None)
    monkeypatch.setattr(pb.ff, 'buildForensicFlagTable', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pb.ff, 'writeForensicFlagsCSV', lambda *a, **k: None)
    monkeypatch.setattr(pb, 'createPresentation', lambda *a, **k: None, raising=False)
    try:
        pb.writeResWrapper(resdic)
    except Exception:
        pass                     # later stages need far more of the pipeline; irrelevant
    hits = sorted(tmp_path.glob('RunProvenance-*_fmp_stock_TEST1.json'))
    assert hits, 'no provenance sidecar was written'
    prov = json.load(open(hits[0]))
    v = prov['stage1_veto']
    assert v['status'] == 'applied' and v['enabled'] is True
    assert v['pools'] == ['general'], 'the sidecar must name the pools the veto actually gated'
    assert v['by_pool']['general']['n_in'] == 1545
    assert v['by_pool']['general']['n_ejected'] == 902
    assert v['by_pool']['general']['ejected_by_flag'] == {'uCurrentRatio': 300}
    assert v['by_pool']['REIT']['applies'] is False
    assert v['by_pool']['REIT']['not_applicable_reason'], (
        'a cohort with n_ejected == 0 and no reason reads as "the veto found it clean"')
    assert v['params']['window_rows'] == 8 and v['params']['eject_min_flags'] == 1
    #  Not carried: the per-source lists live in the postRank pickle (also transferred).
    assert 'ejected' not in v['by_pool']['general']


def test_the_veto_stamp_tells_did_not_run_apart_from_ejected_nobody():
    """THREE STATUSES THAT MUST NOT COLLAPSE.  `did_not_run` (the guarded block raised, so the
    pools are UN-VETOED) and `unknown` (an older resdic that carries no report at all) would
    both otherwise be readable as "the veto ran and found nothing" -- the exact confusion the
    stamp exists to stop.  Neither may raise: the key must survive to say it does not know."""
    import postBo as pb
    assert pb._veto_provenance({'stage1_veto': {}})['status'] == 'did_not_run'
    assert pb._veto_provenance({})['status'] == 'unknown'
    off = pb._veto_provenance({'stage1_veto': {
        'general': {'enabled': False, 'applies': True, 'n_in': 5, 'n_ejected': 0, 'n_out': 5}}})
    assert off['status'] == 'off' and off['enabled'] is False
    #  A malformed report is reported as unknown, never dropped -- a missing key reads as
    #  "no veto", which is a claim this run cannot make.
    assert pb._veto_provenance({'stage1_veto': {'general': 'not a dict'}})['status'] == 'unknown'


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


# --------------------------------------------------------------------------- #
#  THE AD-HOC CURATED ~3,000 UNIVERSE (stock_CUR3K, CEO 2026-08-06)             #
#                                                                               #
#  THE GUARD THE CEO ASKED FOR IN SO MANY WORDS: "a test asserting every           #
#  must-include ticker is present in the universe's resolved membership. The whole  #
#  point is that these cannot be dropped by a later edit to the base rule."         #
#  So the membership test below resolves through the REAL wrapper (offline, against  #
#  a fixture table) rather than checking the registry tuple against itself -- a       #
#  registry-only assertion would have passed even when the union was not applied.     #
# --------------------------------------------------------------------------- #
CUR3K = 'stock_CUR3K'


#  UNPINNED SIBLINGS OF PINNED SYMBOLS -- symbol -> the name it shares with its pinned
#  partner, taken from the live 2026-08-04 table.  These are in the fixture because they are
#  the SECOND HALF of the closure defect: the pins are by SYMBOL, not by GROUP, so five
#  pinned groups arrived incomplete even once the base rule closed upward.  Without these
#  rows the "no partial group" test would pass vacuously.
CUR3K_UNPINNED_SIBLINGS = {
    '0NZN.L':    'Robertet S.A.',                          # RBT.PA + CBE.PA are pinned
    'TROW':      'T. Rowe Price Group, Inc.',              # 0KNY.L is pinned
    'FGN':       'F&G Annuities & Life, Inc.',             # FG is pinned
    '000660.KS': 'SK hynix Inc.',                          # SKHY is pinned
    '0VCO.L':    'Peyto Exploration & Development Corp.',  # PEY.TO is pinned
}

#  The pinned symbols whose live name matches a sibling above, so the fixture holds the
#  group together the way the live table does.
CUR3K_PINNED_SIBLING_NAMES = {
    'RBT.PA': 'Robertet S.A.', 'CBE.PA': 'Robertet S.A.',
    '0KNY.L': 'T. Rowe Price Group, Inc.',
    'FG': 'F&G Annuities & Life, Inc.',
    'SKHY': 'SK hynix Inc.',
    'PEY.TO': 'Peyto Exploration & Development Corp.',
}


def _cur3k_fixture_rows():
    """A pre-filter table containing EVERY pinned symbol, the UNPINNED SIBLINGS of the five
    pinned groups that arrived incomplete, plus base-rule filler.

    Names matter as much as symbols here: the base rule samples on the NORMALISED ISSUER
    NAME, so a fixture that gave every row a unique name would not exercise the property
    the sample exists for."""
    rows = []
    for sym, tag, _why in un.curated3k_manifest():
        #  Give the members of a known group the SAME name, which is what they carry live.
        if sym in CUR3K_PINNED_SIBLING_NAMES:
            nm = CUR3K_PINNED_SIBLING_NAMES[sym]
        elif sym in ('UHS', '0LJL.L'):
            nm = 'Universal Health Services, Inc.'
        elif sym in ('AEM', 'AEM.TO', '0R2J.L'):
            nm = 'Agnico Eagle Mines Limited'
        elif sym in ('VALUE.AS', 'PREVA.AS'):
            nm = 'Value8 N.V.'
        elif sym in ('SMSN.L', 'SMSD.L', 'BC94.L', '005930.KS', '005935.KS'):
            nm = 'Samsung Electronics Co., Ltd.'
        elif sym in ('CIM', 'CIMN'):
            nm = 'Chimera Investment Corporation'
        else:
            nm = 'Issuer %s' % sym
        ex = _exchange_of(sym)
        rows.append({'symbol': sym, 'name': nm, 'type': 'stock',
                     'exchangeShortName': ex, 'tag': tag})
    #  The unpinned siblings, so the five incomplete pinned groups are OBSERVABLE here.
    for sym, nm in CUR3K_UNPINNED_SIBLINGS.items():
        rows.append({'symbol': sym, 'name': nm, 'type': 'stock',
                     'exchangeShortName': _exchange_of(sym), 'tag': 'unpinned-sibling'})
    #  Filler on every base-rule exchange, so the sample has something to sample.
    for code in un.exchanges(CUR3K):
        for i in range(60):
            rows.append({'symbol': 'F%s%02d' % (code, i), 'name': 'Filler %s %02d' % (code, i),
                         'type': 'stock', 'exchangeShortName': code, 'tag': 'filler'})
    return pd.DataFrame(rows)


def _exchange_of(sym):
    """The venue each pinned symbol actually sits on -- including the three that sit
    OUTSIDE the base rule, which is the case that proves the union is not exchange-bound."""
    if sym.endswith('.TO'):
        return 'TSX'
    if sym.endswith('.PA'):
        return 'PAR'
    if sym.endswith('.AS'):
        return 'AMS'
    if sym.endswith('.L'):
        return 'LSE'
    if sym.endswith('.KS'):
        return 'KSC'
    if sym.endswith('.ST'):
        return 'STO'          # EMBELL.ST -- NOT a base-rule exchange
    if sym.endswith('.DE'):
        return 'XETRA'        # EIN.DE / DRW3.DE -- NOT base-rule exchanges
    return 'NASDAQ'


def test_cur3k_every_must_include_symbol_RESOLVES_through_the_real_wrapper(
        offline_wrapper):
    """*** THE POINT OF THE MUST-INCLUDE LIST.  Each of these is a case that has been
    MEASURED and that the first post-2026-08-05 fetch exists to OBSERVE, so a later edit to
    a sample rate must not be able to drop one. ***
    Resolved through `tickerfilterWrapper` offline, so it tests the UNION as applied, not
    the registry tuple against itself."""
    fixture = _cur3k_fixture_rows()
    got = set(_syms(offline_wrapper(fixture, CUR3K, 'all', -1, 'http://x', 'k')))
    missing = sorted(set(un.CURATED3K_MUST_INCLUDE_SYMBOLS) - got)
    assert not missing, (
        '%d pinned case(s) did not resolve: %s. Every one is an observation this run was '
        'built to make -- do not relax this test, fix the membership rule.'
        % (len(missing), missing))


def test_cur3k_pins_reach_OUTSIDE_the_base_rule_exchanges(offline_wrapper):
    """The three pins on unwired venues are the reason the union draws from the FULL table
    rather than from the already-narrowed frame: observing them costs 3 names instead of
    the ~1,375 that wiring STO + XETRA would cost."""
    base = set(un.exchanges(CUR3K))
    outside = {s for s in un.CURATED3K_MUST_INCLUDE_SYMBOLS
               if _exchange_of(s) not in base}
    assert outside == {'EMBELL.ST', 'EIN.DE', 'DRW3.DE'}, outside
    got = set(_syms(offline_wrapper(_cur3k_fixture_rows(), CUR3K, 'all', -1,
                                    'http://x', 'k')))
    assert outside <= got


def test_cur3k_covers_every_venue_the_measured_dedup_cases_live_on():
    """"Must span the exchanges the known dedup cases live on" -- US, LSE including the
    0-prefixed IOB lines, Toronto, Paris, Amsterdam, and enough Korea to carry the
    preferred families.  That constraint alone is what rules out a single-region base."""
    codes = set(un.exchanges(CUR3K))
    assert {'NYSE', 'NASDAQ', 'LSE', 'TSX', 'PAR', 'AMS', 'KSC', 'KOE'} <= codes
    pins = set(un.CURATED3K_MUST_INCLUDE_SYMBOLS)
    assert {'0LJL.L', '0KNY.L', '0QQF.L', '0HQ7.L'} <= pins, 'the IOB lines'
    assert {'UHS', 'AEM', 'AEM.TO'} <= pins, 'the must-MERGE pairs'
    assert {'HEIA.AS', 'HEIO.AS'} <= pins, 'the must-NOT-merge pair'
    assert {'CBE.PA', 'PREVA.AS', 'SMSD.L'} <= pins, 'the three K-1 wrong picks'
    assert {'005930.KS', '005935.KS'} <= pins, 'the Korean preferred family'
    assert 'CIMN' in pins, 'the Chimera NOTES line'


def test_cur3k_the_seventeen_veto_ejects_and_the_six_controls_are_all_pinned():
    """A veto with no observed non-ejections is not an observed veto, so the 6 clean
    controls are pinned alongside the 17 ejects.  The 17 is an UPPER BOUND from the
    superseded five-flag set -- which is exactly why they are pinned: the current set
    differs and the difference is the observation."""
    pins = set(un.CURATED3K_MUST_INCLUDE_SYMBOLS)
    ejects = {'SBH', 'UHS', '0LJL.L', 'NWPX', 'BKE', 'BVFL', 'DDI', 'EMBELL.ST', 'FG',
              'GMR.L', 'JEL.L', 'MU', 'PBYI', 'PEY.TO', 'RFX.L', 'SKHY', 'STRT'}
    controls = {'0KNY.L', '0QQF.L', 'EIN.DE', 'NEXN', 'DRW3.DE', 'KFY'}
    assert len(ejects) == 17 and ejects <= pins, sorted(ejects - pins)
    assert len(controls) == 6 and controls <= pins, sorted(controls - pins)


#  A CROSS-RATE group: one line on a TAKE-ALL venue (TSX) and one on a SAMPLED venue (NYSE
#  at 170), with a bucket >= 170 so the OLD per-row rule kept the TSX line and dropped the
#  NYSE one.  These are the REAL measured cases, and their buckets are all >= 170 -- which
#  is why 218 groups split: 83% of names miss a 17% threshold.
CUR3K_CROSS_RATE_CASES = (
    #  (issuer name, [(symbol, exchange), ...])
    ('Bank of Montreal',            [('BMO', 'NYSE'), ('BMO.TO', 'TSX')]),
    ('BCE Inc.',                    [('BCE', 'NYSE'), ('BCE.TO', 'TSX')]),
    ('Agnico Eagle Mines Limited',  [('AEM', 'NYSE'), ('AEM.TO', 'TSX'), ('0R2J.L', 'LSE')]),
    ('Robertet S.A.',               [('RBT.PA', 'PAR'), ('CBE.PA', 'PAR'),
                                     ('0NZN.L', 'LSE')]),
)


def test_cur3k_a_CROSS_RATE_group_is_NEVER_SPLIT_by_the_base_rule(offline_wrapper):
    """*** THE PROPERTY THE PREVIOUS VERSION OF THIS TEST CLAIMED AND DID NOT CHECK.

    That version asserted only `issuer_in_sample(norm, rate) == (bucket < rate)` FOR A
    FIXED RATE -- a restatement of the function's own body -- and its two worked examples
    were `del names`, i.e. DEAD CODE that made the test LOOK like it checked a group.  It
    passed while the base rule split 218 groups, because closure is not a property of one
    rate: the bucket is a function of the NAME but the threshold was read off the
    EXCHANGE, and TSX/PAR/AMS are take-all while NYSE/NASDAQ/LSE keep buckets < 170. ***

    So this test builds groups that STRADDLE TWO RATES, resolves them through the REAL
    wrapper, and asserts the group arrives WHOLE.  Every fixture case has bucket >= 170, so
    it FAILS on the per-row rule -- which is what makes it a regression test rather than a
    restatement."""
    rates = un.sample_rates(CUR3K)
    rows = []
    for nm, lines in CUR3K_CROSS_RATE_CASES:
        norm = co._norm_issuer_name(nm)
        bucket = un.issuer_sample_bucket(norm)
        venue_rates = {rates.get(ex) for _s, ex in lines}
        assert len(venue_rates) > 1, (
            '%s no longer straddles two sample rates (%r) -- the fixture has stopped '
            'testing the cross-rate case, which is the ONLY case that can split'
            % (nm, venue_rates))
        assert bucket >= min(r for r in venue_rates if r is not None), (
            '%s (bucket %d) now passes the STRICTER rate on its own, so it would arrive '
            'whole even under the broken per-row rule -- pick a different case'
            % (nm, bucket))
        rows += [{'symbol': s, 'name': nm, 'type': 'stock', 'exchangeShortName': ex}
                 for s, ex in lines]
    #  Filler so the sample has a pool, and so this is not a table of four groups.
    for code in un.exchanges(CUR3K):
        rows += [{'symbol': 'F%s%02d' % (code, i), 'name': 'Filler %s %02d' % (code, i),
                  'type': 'stock', 'exchangeShortName': code} for i in range(40)]
    got = set(_syms(offline_wrapper(pd.DataFrame(rows), CUR3K, 'all', -1, 'http://x', 'k')))
    for nm, lines in CUR3K_CROSS_RATE_CASES:
        members = {s for s, _ex in lines}
        present = members & got
        assert present in (set(), members), (
            '%s arrived SPLIT: %s in, %s out. carveOut dedups PAIRWISE OVER THE POOL, so a '
            'half-present group is not a smaller test of dedup -- it is NO test of dedup, '
            'and it introduces a divergence from stock_NA1_EU1 where this group arrives '
            'whole.' % (nm, sorted(present), sorted(members - present)))


def test_cur3k_NO_multi_line_issuer_arrives_PARTIAL_from_the_real_wrapper(offline_wrapper):
    """The general form of the property, over the whole resolved membership rather than four
    hand-picked groups -- so a group shape nobody thought of cannot slip through.  Run over
    the pinned-case fixture, which is where the multi-line groups are.

    This covers BOTH halves of the defect: the base rule's per-row threshold AND the fact
    that the pins are by SYMBOL, not by GROUP (five pinned groups arrived incomplete, and
    the union adding one line of a group observes nothing)."""
    fixture = _cur3k_fixture_rows()
    got = set(_syms(offline_wrapper(fixture, CUR3K, 'all', -1, 'http://x', 'k')))
    partial = []
    for norm, g in fixture.groupby(fixture['name'].map(co._norm_issuer_name)):
        members = set(g['symbol'])
        if not norm or len(members) < 2:
            continue
        present = members & got
        if present and present != members:
            partial.append((norm, sorted(present), sorted(members - present)))
    assert not partial, (
        '%d multi-line issuer(s) arrived PARTIAL: %r' % (len(partial), partial))


def test_cur3k_the_five_INCOMPLETE_pinned_groups_now_arrive_WHOLE(offline_wrapper):
    """*** THE LOAD-BEARING CONSEQUENCE, `robertet s a` first. ***  Because the pins are by
    SYMBOL and not by GROUP, five pinned groups arrived incomplete -- and `robertet s a` is
    one of the THREE K-1 wrong-pick groups the survivor work exists to fix.  Observed as a
    2-member group where production sees 3, the term that decides it in this run need not be
    the term that decides it in production, and the observation does not transfer.  That is
    the un-diagnostic outcome the run was kept short to avoid, so it is pinned by test."""
    got = set(_syms(offline_wrapper(_cur3k_fixture_rows(), CUR3K, 'all', -1,
                                    'http://x', 'k')))
    for sib, nm in sorted(CUR3K_UNPINNED_SIBLINGS.items()):
        partners = sorted(set(CUR3K_PINNED_SIBLING_NAMES) &
                          {s for s, n in CUR3K_PINNED_SIBLING_NAMES.items() if n == nm})
        assert partners, 'fixture bookkeeping: %s has no pinned partner' % sib
        assert sib in got, (
            '%s (%s) did NOT arrive, so the pinned group %s is observed as %d line(s) where '
            'production sees %d -- the pin observes a group shape that does not exist '
            'upstream.' % (sib, nm, partners, len(partners), len(partners) + 1))
    #  Named explicitly, because this one decides whether "volAvg fixed Robertet" transfers.
    assert {'RBT.PA', 'CBE.PA', '0NZN.L'} <= got, (
        'the Robertet group must arrive at all THREE lines; with only RBT.PA + CBE.PA the '
        'ISIN plurality term ABSTAINS (2 members, 2 distinct ISINs) and a different term '
        'decides the survivor than the one that will decide it in production')


def test_cur3k_the_sample_rate_is_resolved_PER_ISSUER_not_PER_ROW():
    """The unit-level statement of the same thing, on the function that now owns the
    decision.  A take-all venue anywhere in the group takes the WHOLE group; otherwise the
    LEAST restrictive rate applies."""
    rates = un.sample_rates(CUR3K)
    assert un.most_permissive_rate({'NYSE', 'TSX'}, rates) is None, 'TSX is take-all'
    assert un.most_permissive_rate({'NYSE', 'LSE'}, rates) == 170
    assert un.most_permissive_rate({'NYSE', 'KSC'}, rates) == 250, 'the LESS restrictive'
    assert un.most_permissive_rate({'KSC', 'KOE'}, rates) == 250
    #  Empty -> take it: an issuer with no usable venue cannot be shown to be OUT, and
    #  dropping on missing metadata is how a filter silently shrinks a universe.
    assert un.most_permissive_rate(set(), rates) is None
    #  And the per-rate primitive is still a pure function of the NAME, which is the OTHER
    #  half of closure -- necessary, and on its own not sufficient.
    for nm in ('Universal Health Services, Inc.', 'Agnico Eagle Mines Limited',
               'Robertet S.A.', 'Samsung Electronics Co., Ltd.'):
        norm = co._norm_issuer_name(nm)
        b = un.issuer_sample_bucket(norm)
        assert 0 <= b < un.CURATED3K_SAMPLE_DENOMINATOR
        for rate in un.CURATED3K_SAMPLED.values():
            assert un.issuer_in_sample(norm, rate) == (b < rate)


def test_cur3k_sample_bucket_is_STABLE_not_process_salted():
    """SHA1, not `hash()`.  Python salts `hash()` of a str per process, so a hash()-based
    sample would resolve to a DIFFERENT universe on every invocation -- and silently.  The
    bucket is pinned to a literal so a change of hash function cannot pass unnoticed."""
    assert un.issuer_sample_bucket('robertet sa') == un.issuer_sample_bucket('robertet sa')
    #  A recomputed-by-hand value: sha1('issuer:' + name), first 8 hex digits, mod 1000.
    import hashlib as _h
    for nm in ('robertet sa', 'value8 nv', ''):
        want = int(_h.sha1(('issuer:' + nm).encode('utf-8')).hexdigest()[:8], 16) % 1000
        assert un.issuer_sample_bucket(nm) == want


def test_cur3k_an_unsampled_code_is_taken_WHOLE_not_sampled_at_zero():
    """Absence from the `sample` dict means NO SAMPLING.  If it meant "rate 0" a typo'd
    code would silently empty an exchange -- the EURONEXT/OSE defect in a new costume."""
    rates = un.sample_rates(CUR3K)
    for c in un.CURATED3K_TAKE_ALL:
        assert c not in rates
        assert un.issuer_in_sample('anything', rates.get(c)) is True
    assert un.issuer_in_sample('anything', 0) is False, \
        'an EXPLICIT zero must still mean "take nothing"'


def test_cur3k_sample_rates_name_no_code_the_universe_does_not_wire():
    """A rate keyed on an unwired code is dead configuration that reads as though it were
    doing something.  Checked for EVERY universe, not just this one."""
    for name in un.names():
        assert un.check_sample_rates(name) == [], name


def test_cur3k_expected_count_is_SCALED_by_the_sample_not_the_raw_sum():
    """The operator reads "expected members" to sanity-check the run LENGTH.  Reporting the
    unscaled 10,991 for a universe built to be ~3,000 would be actively misleading."""
    exp = un.expected_count(CUR3K)
    raw = sum(un._VERIFIED_COUNTS[c] for c in un.exchanges(CUR3K))
    assert raw > 10000, 'the fixture assumption changed'
    assert 2700 <= exp <= 3400, (
        'stock_CUR3K expects %d members; the CEO asked for roughly 3,000' % exp)
    #  `expected_count` now RETURNS the measured sizing for this universe (a per-code
    #  scaling cannot model per-issuer closure), so asserting exp == measured would be
    #  tautological.  What is still worth checking is that the OLD rate-scaled FORMULA has
    #  not drifted far from the measurement: they answer the same question two ways, and a
    #  large gap means one of them is stale.  The known post-closure gap is +135 (the
    #  formula understates, because upward closure keeps issuers for a reason that lives on
    #  a different exchange code).
    rates = un.sample_rates(CUR3K)
    formula = sum(
        un._VERIFIED_COUNTS.get(c, 0) if rates.get(c) is None
        else un._VERIFIED_COUNTS.get(c, 0) * rates[c] / un.CURATED3K_SAMPLE_DENOMINATOR
        for c in un.exchanges(CUR3K)) + len(un.CURATED3K_MUST_INCLUDE_SYMBOLS)
    gap = un.CURATED3K_ESTIMATED_MEMBERS - formula
    assert 0 < gap < 400, (
        'measured sizing %d vs rate-scaled formula %.0f (gap %+.0f). The gap must be '
        'POSITIVE -- upward closure can only ADD members -- and small; a large or negative '
        'gap means the measurement or the rates are stale.'
        % (un.CURATED3K_ESTIMATED_MEMBERS, formula, gap))


def test_cur3k_fingerprint_moves_when_a_RATE_or_a_PIN_moves():
    """A sample rate and a must-include list are part of what the NAME means.  Changing a
    rate from 17% to 20% changes membership as surely as adding an exchange code, so an
    artifact must not claim the same provenance across it."""
    base = un.definition_fingerprint(CUR3K)
    entry = un.UNIVERSES[CUR3K]
    old_sample, old_pins = entry['sample'], entry['must_include']
    try:
        entry['sample'] = dict(old_sample, NYSE=200)
        assert un.definition_fingerprint(CUR3K) != base, 'a rate change did not move it'
        entry['sample'] = old_sample
        entry['must_include'] = old_pins + ('ZZZZ',)
        assert un.definition_fingerprint(CUR3K) != base, 'a pin change did not move it'
    finally:
        entry['sample'], entry['must_include'] = old_sample, old_pins
    assert un.definition_fingerprint(CUR3K) == base


def test_cur3k_fingerprints_of_the_PRE_EXISTING_universes_are_UNTOUCHED():
    """The sample/pin basis is appended only when non-empty, so every universe that
    predates 2026-08-06 fingerprints to exactly the value it did before -- otherwise every
    existing artifact would look like it came from a different definition."""
    for name in un.names():
        #  WIDENED 2026-08-21 from `name == CUR3K` to the PROPERTY rather than the name.
        #  The fingerprint basis grows a `|sample/...` / `|must:...` segment exactly when a
        #  universe HAS rates or pins, so keying the skip on one hardcoded name meant the
        #  SECOND such universe (stock_CUR6K) failed a test whose whole point is about the
        #  universes that predate the feature.  `test_the_declared_sampled_universes_...`
        #  is what keeps the skipped set from growing silently.
        if un.sample_rates(name) or un.must_include(name):
            assert name in SAMPLED_UNIVERSES, name
            continue
        d = un.UNIVERSES[name]
        if d['symbols'] is not None:
            basis = 'symbols:' + ','.join(sorted(d['symbols']))
        elif d['every_exchange']:
            basis = 'every-exchange'
        else:
            basis = 'exchanges:' + ','.join(sorted(d['exchanges']))
        import hashlib as _h
        assert un.definition_fingerprint(name) == \
            _h.sha1(basis.encode('utf-8')).hexdigest()[:12], name


def test_cur3k_banner_says_the_pool_is_not_production():
    """~3,000 is a REAL pool -- a top-100 cut means something on it, which is why the CEO
    chose it over 142 -- but it is still not production's ~10,693, so no pooled statistic
    crosses over.  The banner must say so on every run, not the docstring."""
    b = un.run_banner(CUR3K)
    assert 'SAMPLED UNIVERSE' in b
    assert 'NOT COMPARABLE' in b
    assert 'universe_fingerprint' in b
    assert 'SAMPLED on issuer NAME' in b
    assert 'must-include' in b


def test_cur3k_is_KOREA_GATED_like_every_other_korea_universe(offline_wrapper):
    """It wires KSC/KOE, so `assert_korea_dedup_ready` must fire on it -- Korea cannot be
    enabled by adding a registry entry."""
    assert any(c in gdg.KOREA_EXCHANGE_CODES for c in un.exchanges(CUR3K))
    fixture = _cur3k_fixture_rows()
    import carveOut as _co
    real = _co._non_canonical_tag
    try:
        del _co._non_canonical_tag
        with pytest.raises(Exception) as e:
            offline_wrapper(fixture, CUR3K, 'all', -1, 'http://x', 'k')
        assert 'Korea' in str(e.value)
    finally:
        _co._non_canonical_tag = real


def test_cur3k_runtime_estimate_is_stated_on_the_SAME_basis_as_the_test_universe():
    """The CEO is choosing this universe specifically to keep the run SHORT, so the number
    has to be defensible: 5 statement calls per source + 3 for the universe build, at the
    per-call rate two independent runs imply (0.67-1.26 s/call from the 142-name run;
    0.80 s/call from the 12h production run over 10,737 sources ATTEMPTED -- kept 9,012
    plus 1,725 failed, because a failed source still costs its calls)."""
    assert un.CURATED3K_API_CALLS == 5 * un.CURATED3K_ESTIMATED_MEMBERS + 3
    lo, hi = un.CURATED3K_WALLCLOCK_HOURS
    assert lo < hi
    #  The stated band must actually follow from the stated per-call rates.
    #  TOLERANCE TIGHTENED 0.5 -> 0.05 h (2026-08-06, reviewer).  At +-0.5 h the guard was
    #  DECORATIVE: the band sat at (3.4, 4.9) -- the pre-group-closure 15,228-call figure --
    #  while the coded call count had moved to 16,293, i.e. (3.6, 5.2), and the check passed
    #  anyway because the drift was smaller than the tolerance.  0.05 h is one decimal place,
    #  which is the precision the band is actually stated to, so a stale band now FAILS here
    #  instead of being absorbed.  This is the cheap protection against the same class of
    #  staleness in `expected_count` (see the fingerprint note in that docstring).
    assert abs(lo - un.CURATED3K_API_CALLS * 0.80 / 3600.0) < 0.05, (
        'CURATED3K_WALLCLOCK_HOURS lo=%r does not follow from %d calls at 0.80 s/call '
        '(%.2f h) -- re-derive the band, do not widen the tolerance'
        % (lo, un.CURATED3K_API_CALLS, un.CURATED3K_API_CALLS * 0.80 / 3600.0))
    assert abs(hi - un.CURATED3K_API_CALLS * 1.15 / 3600.0) < 0.05, (
        'CURATED3K_WALLCLOCK_HOURS hi=%r does not follow from %d calls at 1.15 s/call '
        '(%.2f h) -- re-derive the band, do not widen the tolerance'
        % (hi, un.CURATED3K_API_CALLS, un.CURATED3K_API_CALLS * 1.15 / 3600.0))
    assert 3.0 <= lo and hi <= 5.5, 'the CEO was told ~4-5 h; keep the claim honest'
    #  The label an operator READS must carry the same band as the constant a script uses.
    assert '%.1f-%.1f h' % (lo, hi) in un.label(CUR3K), (
        'the registry label and CURATED3K_WALLCLOCK_HOURS state different fetch lengths')


def test_cur3k_cohort_estimate_sums_to_the_CODED_member_count():
    """*** THE PAIR THAT WAS STALE, NOW PINNED (2026-08-06, reviewer). ***
    `CURATED3K_COHORT_ESTIMATE` summed to 3,045 -- the PRE-group-closure member count --
    while `CURATED3K_ESTIMATED_MEMBERS` had moved to 3,258, so an operator adding the cohort
    column up got a universe 213 names smaller than the one the run builds.  The SHARES are
    the measured quantity and the counts are those shares applied to the member count, so
    the sum is a DERIVED figure and must track it."""
    tot = sum(un.CURATED3K_COHORT_ESTIMATE.values())
    assert tot == un.CURATED3K_ESTIMATED_MEMBERS, (
        'CURATED3K_COHORT_ESTIMATE sums to %d but the universe is sized at %d -- re-scale '
        'the cohort counts (the shares stay; only the base moves)'
        % (tot, un.CURATED3K_ESTIMATED_MEMBERS))
    #  The conclusion the table exists to support: `general` is comfortably above the
    #  top-100 cut, so the general pool's top-100 and the veto's survivor count are real.
    assert un.CURATED3K_COHORT_ESTIMATE['general'] > 100


#  Pins whose why-string DELIBERATELY does not name a company, so the identity check below
#  has nothing to compare.  DECLARED rather than silently absent, which is this file's
#  standing convention (see `test_the_unverified_gates_are_declared` in test_dedup_issuer):
#  each is a sibling row whose prose points at the group, and the named half of the group is
#  pinned separately.
CUR3K_WHY_STRINGS_WITHOUT_A_COMPANY_NAME = {
    '0KNY.L': 'names only the T. Rowe Price GROUP role (IOB + veto control); TROW is the '
              'unpinned sibling that carries the name',
    'BC94.L': 'says "the third Samsung line on LSE"; SMSN.L/SMSD.L carry the full name',
    'CIMN':   'says "a Chimera NOTES line"; CIM carries the full name',
}


def _fmp_name_token(fmp_name):
    """The leading DISTINCTIVE words of an FMP company name, for a substring check.

    Not a string equality, deliberately: the why-strings are prose and legitimately shorten
    "Sally Beauty Holdings, Inc." to "Sally Beauty Holdings".  Corporate form and the
    leading article are stripped (FMP writes "The Buckle, Inc." where the manifest writes
    "Buckle Inc"), then the first two remaining words are required -- enough to have caught
    BOTH real defects (SK hynix vs SkyHarbour, NWPX Infrastructure vs Northwest Pipe)
    without failing on an abbreviation."""
    s = str(fmp_name).strip()
    if s.lower().startswith('the '):
        s = s[4:]
    words = []
    for w in s.replace(',', ' ').split():
        if w.strip('.').lower() in ('inc', 'corp', 'corporation', 'ltd', 'limited', 'plc',
                                    'ab', 'publ', '(publ)', 'nv', 'n.v', 'sa', 's.a',
                                    'ag', 'kgaa', 'co', 'group', '&'):
            break
        words.append(w)
        if len(words) == 2:
            break
    return ' '.join(words) or s


def test_every_pin_why_string_names_the_company_FMP_ACTUALLY_RETURNS(live_capture_names):
    """*** THE MANIFEST IS WHAT THE OPERATOR READS, SO A WRONG NAME IS THE ONE STALE
    NUMBER THAT ACTUALLY MATTERS. ***
    `SKHY`'s why-string said "SK Growth Opportunities / SkyHarbour" -- two unrelated issuers
    whose tickers merely look similar -- while FMP returns "SK hynix Inc." for it.  That is
    precisely why the `000660.KS` sibling fires under group closure, so an operator checking
    the closure against the manifest would have read a CORRECT closure as a DEFECT.  All 40
    pins were then cross-checked against the live capture and `NWPX` was stale too (FMP
    renamed Northwest Pipe to "NWPX Infrastructure").
    Runs against the LIVE CAPTURE, i.e. the names FMP actually returns, not a hand table."""
    why = dict((s, w) for s, _t, w in un.curated3k_manifest())
    checked, absent = 0, []
    for sym, w in why.items():
        if sym in CUR3K_WHY_STRINGS_WITHOUT_A_COMPANY_NAME:
            continue
        nm = live_capture_names.get(sym)
        if not nm:
            absent.append(sym)
            continue
        checked += 1
        tok = _fmp_name_token(nm)
        assert tok.lower() in w.lower(), (
            '%s: FMP returns %r but the why-string never mentions %r -- %r'
            % (sym, nm, tok, w[:140]))
    assert checked >= 30, (
        'only %d of %d pins were checkable against the capture (absent: %s) -- too few to '
        'be evidence' % (checked, len(why), absent))


def test_the_pins_that_do_not_name_a_company_are_DECLARED():
    """The exclusion list above must not decay into a place to hide a wrong name: every
    entry has to be a real pin, and every pin NOT in it must be one this gate checked."""
    syms = {s for s, _t, _w in un.curated3k_manifest()}
    assert set(CUR3K_WHY_STRINGS_WITHOUT_A_COMPANY_NAME) <= syms
    assert all(v for v in CUR3K_WHY_STRINGS_WITHOUT_A_COMPANY_NAME.values()), \
        'every exclusion needs a stated reason'


def test_the_two_WRONG_company_names_cannot_come_back():
    """A named regression gate on the actual defect, independent of the token heuristic
    above: the two wrong names must not appear anywhere in the manifest."""
    blob = ' '.join(w for _s, _t, w in un.curated3k_manifest()).lower()
    for bad in ('skyharbour', 'sk growth opportunities', 'northwest pipe'):
        assert bad not in blob, (
            '%r is back in a CUR3K why-string -- it names the WRONG company' % bad)


# =========================================================================== #
#  stock_CUR6K -- THE ~6,000-NAME UNIVERSE WITH stock_CUR3K AS A STRICT SUBSET   #
#                                                                               #
#  THE HARD REQUIREMENT IS THE SUBSET PROPERTY, and it is asserted HERE against   #
#  the LIVE definitions replayed through the REAL wrapper -- not argued from the    #
#  construction.  The construction argument is in universes.py above                #
#  CURATED6K_TAKE_ALL and it is sound, but it is exactly the class of argument       #
#  this project has been burned by: the FIRST cut of the issuer sample also had a    #
#  sound-sounding closure argument and split 218 groups in practice (see              #
#  `most_permissive_rate`).  So the property is MEASURED.                            #
#                                                                               #
#  WHOLLY OFFLINE.  The replay runs over `available_traded_raw_2026-08-04.pickle`,   #
#  a saved live capture, with `safe_get` stubbed -- no network, no paid call.  The    #
#  tests skip rather than pass vacuously when the capture is not on the machine.      #
# =========================================================================== #
CUR6K = 'stock_CUR6K'


@pytest.fixture(scope='module')
def live_capture_table():
    """The full live 2026-08-04 available-traded capture as the pre-filter table.

    Module-scoped: the replay tests run the real wrapper over 68k rows and there is no
    reason to pay for the load more than once.
    """
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'available_traded_raw_2026-08-04.pickle')
    if not os.path.exists(p):
        pytest.skip('live capture absent: %s' % os.path.basename(p))
    return pd.read_pickle(p)


def _replay(table, name, monkeypatch, tmp_path):
    """Resolve `name` end-to-end through the REAL wrapper, offline.

    Deliberately NOT a reimplementation of the membership rule: the whole value of this
    replay is that it exercises the same code path a fetch does -- type filter,
    `filter_non_common_instruments` on the full table, the exchange filter, the issuer
    sample, the pin union with group closure, the delisted prune.  A hand-rolled
    reimplementation would prove the subset property of the TEST, not of the pipeline.
    """
    monkeypatch.setattr(gdg, 'safe_get', lambda *a, **k: [])   # delisted page 0 -> empty
    monkeypatch.chdir(tmp_path)
    return set(gdg.tickerfilterWrapper(table, name, 'all', -1, 'http://x', 'k')['symbol'])


def test_cur6k_stock_CUR3K_IS_A_STRICT_SUBSET_of_it__REPLAYED(
        live_capture_table, monkeypatch, tmp_path):
    """*** THE REQUIREMENT.  Every stock_CUR3K member is a stock_CUR6K member. ***

    WHY IT IS THE REQUIREMENT AND NOT A NICETY: it is what makes the two runs comparable.
    With it, every difference between a CUR3K run and a CUR6K run is attributable to the
    ADDED names -- same pinned dedup/veto cases, same measured names, same buckets.
    Without it the two runs are two unrelated experiments and every pooled statistic
    differs for a second, unattributable reason, which is the irreconcilable-beat-rate
    trap the CONTINUITY note at the top of universes.py exists to prevent.

    Replayed through the REAL wrapper over the live capture, both directions checked.
    """
    small = _replay(live_capture_table, CUR3K, monkeypatch, tmp_path)
    big = _replay(live_capture_table, CUR6K, monkeypatch, tmp_path)
    missing = sorted(small - big)
    assert not missing, (
        '%d stock_CUR3K member(s) are NOT in stock_CUR6K -- the subset property is BROKEN '
        'and the two universes are no longer comparable: %s%s'
        % (len(missing), ', '.join(missing[:25]),
           ' (+%d more)' % (len(missing) - 25) if len(missing) > 25 else ''))
    #  STRICT, not merely a subset: a "superset" that added nothing would satisfy the
    #  inclusion above and answer none of the CEO's question.
    assert len(big) > len(small), 'stock_CUR6K added nothing'
    #  And the replay must agree with the numbers the module PUBLISHES, or the banner, the
    #  label and the cost estimate are all quoting a universe that is not the one built.
    assert len(small) == un.CURATED3K_ESTIMATED_MEMBERS, (
        'stock_CUR3K replays to %d, module says %d' % (len(small),
                                                      un.CURATED3K_ESTIMATED_MEMBERS))
    assert len(big) == un.CURATED6K_ESTIMATED_MEMBERS, (
        'stock_CUR6K replays to %d, module says %d -- re-replay and update BOTH '
        'CURATED6K_SIZING and REPLAYED_MEMBER_COUNTS (fingerprint included)'
        % (len(big), un.CURATED6K_ESTIMATED_MEMBERS))


def test_cur6k_the_subset_test_above_HAS_TEETH(live_capture_table, monkeypatch, tmp_path):
    """THE ANTI-VACUITY GUARD.  A subset assertion passes trivially if the smaller set is
    empty, if the two definitions are identical, or if the resolution silently returns
    everything.  So: the added set must be large, and -- the part that actually gives the
    assertion teeth -- LOWERING a rate below stock_CUR3K's must BREAK the property.

    If this test could not break it, the one above would be proving nothing about the
    sample and everything about the pin union.
    """
    small = _replay(live_capture_table, CUR3K, monkeypatch, tmp_path)
    added = _replay(live_capture_table, CUR6K, monkeypatch, tmp_path) - small
    assert len(added) > 2000, 'only %d names added; the point was ~6,000 total' % len(added)

    #  A DELIBERATELY BROKEN definition: CUR6K's venues, but the US/LSE rate cut BELOW
    #  CUR3K's 170. The nesting argument says this must lose CUR3K members, and it must.
    entry = un.UNIVERSES[CUR6K]
    old = entry['sample']
    try:
        entry['sample'] = dict(old, NYSE=100, NASDAQ=100, LSE=100)
        broken = _replay(live_capture_table, CUR6K, monkeypatch, tmp_path)
    finally:
        entry['sample'] = old
    assert small - broken, (
        'LOWERING the US/LSE rate from 220 to 100 lost NO stock_CUR3K member, so the '
        'subset test above cannot distinguish a nested sample from an arbitrary one -- '
        'either the sample is not being applied or the pins are carrying the whole '
        'property')
    #  Sanity: the un-broken definition is still whole after the mutation is undone.
    assert not (small - _replay(live_capture_table, CUR6K, monkeypatch, tmp_path))


def test_cur6k_raising_a_rate_can_NEVER_drop_an_issuer():
    """THE MONOTONICITY PROPERTY, at the level of the two functions that decide it, so it
    holds for any future rate edit and not just for the pair of definitions on the table.

    `issuer_sample_bucket` is a pure function of the NORMALISED ISSUER NAME, so the bucket
    is the same number in both universes and only the THRESHOLD moves.
    `most_permissive_rate` returns None on any take-all venue and otherwise the MAX over
    the venues the issuer occupies -- monotone non-decreasing in BOTH the venue set and the
    per-code rates.  `issuer_in_sample` is `bucket < rate`.  So a rate rise can only ever
    ADD issuers.

    *** IF THIS FAILED, THE DESIGN WOULD BE WRONG AND THE ANSWER WOULD BE TO SAY SO, NOT
    TO WORK AROUND IT: a threshold sample whose selected set is not NESTED under a rate
    rise is not a sample, it is a reshuffle, and no two runs over it are comparable. ***
    """
    lo, hi = un.sample_rates(CUR3K), un.sample_rates(CUR6K)
    #  Precondition of the whole argument: same bucket scale, and every shared rate rose.
    assert un.CURATED6K_SAMPLE_DENOMINATOR == un.CURATED3K_SAMPLE_DENOMINATOR, (
        'the two universes must share the bucket denominator or `bucket < rate` is not '
        'nested and the subset property is an accident')
    for c, r in lo.items():
        assert hi.get(c, r) >= r, '%s rate FELL %d -> %d' % (c, r, hi[c])
    #  1. per-issuer: nesting of `issuer_in_sample` over 5,000 synthetic names.
    for i in range(5000):
        nm = 'issuer name %d' % i
        for c, r in lo.items():
            if un.issuer_in_sample(nm, r):
                assert un.issuer_in_sample(nm, hi[c]), (
                    '%r is in at rate %d but OUT at the higher rate %d' % (nm, r, hi[c]))
    #  2. most_permissive_rate: adding a venue never tightens, and a take-all venue wins.
    small_codes = set(un.exchanges(CUR3K))
    big_codes = set(un.exchanges(CUR6K))
    assert small_codes < big_codes, 'CUR6K must wire strictly more venues'
    for c in sorted(small_codes):
        r_small = un.most_permissive_rate((c,), lo)
        r_big = un.most_permissive_rate((c,), hi)
        assert r_big is None or (r_small is not None and r_big >= r_small), (
            '%s: effective rate tightened %r -> %r' % (c, r_small, r_big))
        #  and adding any CUR6K-only venue to the group can only loosen it further
        for extra in sorted(big_codes - small_codes):
            r_pair = un.most_permissive_rate((c, extra), hi)
            assert r_pair is None, (
                '%s + %s should be take-all (%s carries no rate) but resolved to %r'
                % (c, extra, extra, r_pair))


def test_cur6k_take_all_is_EXACTLY_production_minus_the_sampled_venues():
    """The definition is worth being able to state in one line -- "production, with
    NYSE/NASDAQ/LSE sampled and Korea added" -- and the prose above CURATED6K_TAKE_ALL
    states it.  Asserted so the prose and the tuple cannot drift apart."""
    prod = set(un.exchanges('stock_NA1_EU1'))
    assert set(un.CURATED6K_TAKE_ALL) == prod - {'NYSE', 'NASDAQ', 'LSE'}
    #  Every wired code is either take-all or sampled, nothing in between and nothing lost.
    assert set(un.exchanges(CUR6K)) == set(un.CURATED6K_TAKE_ALL) | set(un.CURATED6K_SAMPLED)
    #  It adds Korea on top of production, and nothing else outside it.
    assert set(un.exchanges(CUR6K)) - prod == set(un.korea_codes())


def test_cur6k_the_four_NEVER_FETCHED_venues_are_actually_in_it():
    """The stated main reason for the fetch is new-venue defect discovery, and four of the
    six added venues have never been through the fetch path at all (OSL/BRU/LIS/ICE were
    the EURONEXT/OSE dead-code defect, restored 2026-08-02, and no run since has covered
    them).  If they were not actually wired, the universe's whole rationale would be
    prose."""
    for code in ('XETRA', 'STO', 'OSL', 'BRU', 'LIS', 'ICE'):
        assert code in un.exchanges(CUR6K), code
        assert code not in un.sample_rates(CUR6K), '%s must be taken WHOLE' % code
        assert code in un.CURATED6K_SIZING and un.CURATED6K_SIZING[code] > 0, code
    #  and they were NOT in CUR3K in any meaningful quantity (STO/XETRA only via pins)
    for code in ('OSL', 'BRU', 'LIS', 'ICE'):
        assert code not in (un.exchanges(CUR3K) or ()), code


def test_cur6k_sizing_sums_to_the_member_count_and_covers_every_wired_code():
    """The per-code table is what an operator reads to see WHERE the added mass went, so a
    missing code or a sum that disagrees with the member count is a misleading table --
    the CURATED3K_COHORT_ESTIMATE defect (summed to 3,045 against a 3,258 universe) in a
    new place."""
    assert sum(un.CURATED6K_SIZING.values()) == un.CURATED6K_ESTIMATED_MEMBERS
    assert set(un.CURATED6K_SIZING) == set(un.exchanges(CUR6K)), (
        'CURATED6K_SIZING and the wired exchange set disagree: %s'
        % (set(un.CURATED6K_SIZING) ^ set(un.exchanges(CUR6K))))
    #  It must land where the CEO asked. "Roughly 6,000" is the target; a universe that
    #  quietly drifted to 7,500 is a different (and 25%-more-expensive) run.
    assert 5700 <= un.CURATED6K_ESTIMATED_MEMBERS <= 6300, (
        'stock_CUR6K sizes at %d; the CEO asked for roughly 6,000'
        % un.CURATED6K_ESTIMATED_MEMBERS)
    #  And it must be a real superset in SIZE, not just in membership.
    assert un.CURATED6K_ESTIMATED_MEMBERS > 1.5 * un.CURATED3K_ESTIMATED_MEMBERS


def test_cur6k_the_rate_scaled_FORMULA_gap_is_recorded_not_hidden():
    """`expected_count` returns the REPLAY for this universe, so the formula's answer is
    invisible unless it is written down.  It is worth writing down: the formula understates
    by 544 here against 135 for CUR3K, and that ratio IS the compromise story -- six more
    take-all venues means far more per-issuer upward closure, which is why the rates had to
    come down from 30% to 22% to hit ~6,000."""
    rates = un.sample_rates(CUR6K)
    formula = sum(
        un._VERIFIED_COUNTS.get(c, 0) if rates.get(c) is None
        else un._VERIFIED_COUNTS.get(c, 0) * rates[c] / un.CURATED6K_SAMPLE_DENOMINATOR
        for c in un.exchanges(CUR6K)) + len(un.must_include(CUR6K))
    gap = un.CURATED6K_ESTIMATED_MEMBERS - formula
    assert gap > 0, (
        'the gap must be POSITIVE -- upward closure can only ADD members. Got %+.0f '
        '(replay %d vs formula %.0f), which means the replay or the rates are stale'
        % (gap, un.CURATED6K_ESTIMATED_MEMBERS, formula))
    assert abs(gap - un.CURATED6K_FORMULA_GAP) < 25, (
        'CURATED6K_FORMULA_GAP says %d, the arithmetic says %+.0f'
        % (un.CURATED6K_FORMULA_GAP, gap))
    #  The claim that this universe closes MORE than CUR3K does, asserted rather than said.
    assert gap > (un.CURATED3K_ESTIMATED_MEMBERS - sum(
        un._VERIFIED_COUNTS.get(c, 0) if un.sample_rates(CUR3K).get(c) is None
        else un._VERIFIED_COUNTS.get(c, 0) * un.sample_rates(CUR3K)[c]
             / un.CURATED3K_SAMPLE_DENOMINATOR
        for c in un.exchanges(CUR3K)) - len(un.must_include(CUR3K)))


def test_cur6k_runtime_estimate_is_on_the_SAME_basis_as_CUR3K():
    """The CEO is deciding whether to spend ~1.85x CUR3K's fetch, so the number has to be
    defensible AND comparable: same 5-calls-per-source + 3 basis, same 0.80-1.15 s/call
    band, profile batches excluded from both.  Tolerance is one decimal place -- the
    precision the band is stated to -- for the reason the CUR3K version of this test
    records: at +-0.5 h the guard was decorative and absorbed a real drift."""
    assert un.CURATED6K_API_CALLS == 5 * un.CURATED6K_ESTIMATED_MEMBERS + 3
    lo, hi = un.CURATED6K_WALLCLOCK_HOURS
    assert lo < hi
    assert abs(lo - un.CURATED6K_API_CALLS * 0.80 / 3600.0) < 0.05, (
        'CURATED6K_WALLCLOCK_HOURS lo=%r does not follow from %d calls at 0.80 s/call '
        '(%.2f h) -- re-derive the band, do not widen the tolerance'
        % (lo, un.CURATED6K_API_CALLS, un.CURATED6K_API_CALLS * 0.80 / 3600.0))
    assert abs(hi - un.CURATED6K_API_CALLS * 1.15 / 3600.0) < 0.05, (
        'CURATED6K_WALLCLOCK_HOURS hi=%r does not follow from %d calls at 1.15 s/call '
        '(%.2f h)' % (hi, un.CURATED6K_API_CALLS, un.CURATED6K_API_CALLS * 1.15 / 3600.0))
    #  The label an operator READS must carry the same band as the constant a script uses.
    assert '%.1f-%.1f h' % (lo, hi) in un.label(CUR6K), (
        'the registry label and CURATED6K_WALLCLOCK_HOURS state different fetch lengths')
    #  And the BANNER's independently-derived line must agree with the constants, since it
    #  is computed from `expected_count` rather than looked up -- that is the point of
    #  deriving it, and it is only worth anything if the two actually match.
    assert '~{:,} statement calls'.format(un.CURATED6K_API_CALLS) in un.run_banner(CUR6K)
    assert '~%.1f-%.1f h' % (lo, hi) in un.run_banner(CUR6K)
    assert '~{:,} statement calls'.format(un.CURATED3K_API_CALLS) in un.run_banner(CUR3K)
    assert '~%.1f-%.1f h' % un.CURATED3K_WALLCLOCK_HOURS in un.run_banner(CUR3K)


def test_cur6k_inherits_the_pins_and_the_KOREA_GATE(offline_wrapper):
    """The pins are the measured dedup/veto cases; sharing them with CUR3K by IDENTITY (not
    by copy) is what makes "same pinned cases" a fact rather than a convention someone has
    to maintain in two places.  And Korea cannot be enabled by adding a registry entry."""
    assert un.must_include(CUR6K) == un.CURATED3K_MUST_INCLUDE_SYMBOLS
    assert un.must_include(CUR6K) == un.must_include(CUR3K)
    assert any(c in gdg.KOREA_EXCHANGE_CODES for c in un.exchanges(CUR6K))
    fixture = _cur3k_fixture_rows()
    import carveOut as _co
    real = _co._non_canonical_tag
    try:
        del _co._non_canonical_tag
        with pytest.raises(Exception) as e:
            offline_wrapper(fixture, CUR6K, 'all', -1, 'http://x', 'k')
        assert 'Korea' in str(e.value)
    finally:
        _co._non_canonical_tag = real


def test_cur6k_every_pin_RESOLVES_through_the_real_wrapper(live_capture_table, monkeypatch,
                                                          tmp_path):
    """The guard the CEO asked for on CUR3K, applied here too: no edit to the base rule may
    drop a pinned case.  Resolved through the REAL wrapper against the live capture rather
    than checked registry-against-itself, because a registry-only assertion passes even
    when the union is not applied at all."""
    got = _replay(live_capture_table, CUR6K, monkeypatch, tmp_path)
    absent = sorted(set(un.must_include(CUR6K)) - got)
    #  MS-PE and GIPRW are not pins here; every CUR6K pin should resolve, but a pin the
    #  INSTRUMENT FILTER removes stays removed by design, so compare against what CUR3K
    #  itself achieves rather than demanding 40/40 in the abstract.
    absent_in_cur3k = sorted(set(un.must_include(CUR3K))
                             - _replay(live_capture_table, CUR3K, monkeypatch, tmp_path))
    assert absent == absent_in_cur3k, (
        'stock_CUR6K resolves a DIFFERENT set of pins than stock_CUR3K (CUR6K missing %s, '
        'CUR3K missing %s) -- the pin union is behaving differently on the wider universe'
        % (absent, absent_in_cur3k))


def test_cur6k_the_PINS_REACH_OUTSIDE_property_is_now_vacuous_here_and_that_is_DECLARED():
    """CUR3K's pins reach OUTSIDE its base-rule exchanges (EMBELL.ST on STO, EIN.DE and
    DRW3.DE on XETRA), which is what proves the must-include union is not exchange-bound.
    CUR6K WIRES STO and XETRA, so all 40 of its pins sit inside its own exchange set and it
    exercises that path vacuously.

    Pinned as a FACT rather than left as a comment, because the tempting mistake is to read
    CUR6K as a superset of CUR3K's TEST coverage as well as of its membership.  It is not:
    this property is covered only by `test_cur3k_pins_reach_OUTSIDE_the_base_rule_exchanges`
    and that test must not be deleted on the grounds that CUR6K subsumes CUR3K.
    """
    codes = set(un.exchanges(CUR6K))
    outside = [s for s in un.must_include(CUR6K) if _exchange_of(s) not in codes]
    assert outside == [], (
        'a CUR6K pin now sits outside its exchange set (%s) -- the property is no longer '
        'vacuous here and this test should become a real assertion' % outside)
    #  ... and it is genuinely NON-vacuous on CUR3K, which is what still covers it.
    cur3k_codes = set(un.exchanges(CUR3K))
    assert [s for s in un.must_include(CUR3K) if _exchange_of(s) not in cur3k_codes], \
        'CUR3K no longer has an out-of-exchange pin, so NOTHING covers the union path'


# --------------------------------------------------------------------------- #
#  THE REPLAYED-COUNT FINGERPRINT GUARD (2026-08-21)                            #
#  Closes the staleness path the 2026-08-06 review named and deferred: a member   #
#  count stored against a NAME survives an edit to the definition and keeps        #
#  looking plausible.                                                             #
# --------------------------------------------------------------------------- #
def test_the_replayed_member_counts_MATCH_the_definitions_they_were_measured_against():
    """The stored fingerprints are the whole guard, so they must be current."""
    assert set(un.REPLAYED_MEMBER_COUNTS) == set(SAMPLED_UNIVERSES), (
        'a sampled universe has no replayed member count, so `expected_count` silently '
        'falls back to the formula that understates it: %s'
        % (set(SAMPLED_UNIVERSES) ^ set(un.REPLAYED_MEMBER_COUNTS)))
    for name, (fp, n) in un.REPLAYED_MEMBER_COUNTS.items():
        assert un.definition_fingerprint(name) == fp, (
            '%s: replay was measured against definition %s but the definition is now %s '
            '-- re-replay offline and update BOTH the fingerprint and the count'
            % (name, fp, un.definition_fingerprint(name)))
        assert un.replayed_count_is_stale(name) is None
        assert un.expected_count(name) == n


def test_a_STALE_replay_is_REFUSED_and_the_banner_SHOUTS_about_it():
    """The failure this closes is silent: the stored count survives a definition edit and
    still looks plausible, so nothing looks wrong.  Two things must happen instead --
    `expected_count` must fall back to the formula, and the banner must say the number
    changed kind (measurement -> model), because otherwise the operator just sees a slightly
    different figure."""
    entry = un.UNIVERSES[CUR6K]
    old = entry['sample']
    measured = un.expected_count(CUR6K)
    try:
        entry['sample'] = dict(old, NYSE=201)          # a real definition change
        stale = un.replayed_count_is_stale(CUR6K)
        assert stale is not None, 'a rate change did not invalidate the stored replay'
        assert stale[0] == un.REPLAYED_MEMBER_COUNTS[CUR6K][0]
        assert stale[1] == un.definition_fingerprint(CUR6K) != stale[0]
        assert un.expected_count(CUR6K) != measured, (
            'the STORED count was returned for a definition it was not measured against '
            '-- this is exactly the silent staleness the fingerprint key exists to stop')
        b = un.run_banner(CUR6K)
        assert 'REPLAYED MEMBER COUNT FOR THIS UNIVERSE IS STALE' in b
        assert 'RATE-SCALED FORMULA' in b
    finally:
        entry['sample'] = old
    assert un.replayed_count_is_stale(CUR6K) is None
    assert un.expected_count(CUR6K) == measured


# --------------------------------------------------------------------------- #
#  WHAT THE BANNER MUST SAY BEFORE A 7-10 HOUR FETCH                            #
# --------------------------------------------------------------------------- #
def test_cur6k_banner_states_the_SUBSET_the_COST_and_that_it_is_NOT_A_POWER_FIX():
    """Three things an operator would otherwise get wrong in a predictable direction, so
    they belong in the loudest place in the run rather than in a docstring."""
    b = un.run_banner(CUR6K)
    #  1. the subset property -- what the comparison to the CUR3K run is actually worth
    assert 'stock_CUR3K IS A STRICT SUBSET OF THIS UNIVERSE' in b
    assert 'attributable to the %d ADDED names' % (
        un.CURATED6K_ESTIMATED_MEMBERS - un.CURATED3K_ESTIMATED_MEMBERS) in b
    #  2. the cost, before it is incurred
    assert 'fetch cost (estimate)' in b
    assert 'A FLOOR' in b, (
        'the band comes from runs with smaller payloads; at -nrperiods 80 the payloads are '
        '~2.3x larger, so it must not read as a centre')
    #  3. that "more names" is NOT more statistical power -- the claim most likely to be
    #     read into a bigger universe, and the measurement says otherwise
    assert 'DOES **NOT** FIX THE POWER PROBLEM' in b
    assert 'independent 3-year WINDOWS' in b
    #  the generic sampled-universe warning still fires too
    assert 'SAMPLED UNIVERSE' in b and 'NOT COMPARABLE' in b
    assert 'universe_fingerprint' in b


def test_cur6k_banner_says_a_BACKTEST_over_it_does_not_COVER_it():
    """THE PRICE-COVERAGE CAVEAT.  The CEO has ruled that price coverage is an analysis
    tool and not a filter input, so the 1,386 XETRA+STO names are a deliberate inclusion,
    NOT a blocker -- these names score normally.  What the ruling does not cover is someone
    reading a BACKTEST over this universe as covering this universe.  It does not, and the
    banner has to say so on every run."""
    b = un.run_banner(CUR6K)
    assert 'SCOREABLE BUT NOT BACKTESTABLE' in b
    assert str(un.CURATED6K_UNBACKTESTABLE) in b
    assert '2021-12-31' in b
    assert 'STO' in b and 'XETRA' in b
    #  the pool a beat-rate may actually be quoted against
    assert str(un.CURATED6K_ESTIMATED_MEMBERS - un.CURATED6K_UNBACKTESTABLE) in b


def test_the_price_coverage_gap_is_recorded_as_DATA_and_its_total_follows():
    """Recorded as data rather than prose so the banner and any tooling read ONE source,
    and so the figures are attributable: they are supplied by the price-coverage work,
    measured at 2021-12-31, NOT measured by universes.py -- which is why they carry a date
    and a venue rather than sitting inside a sentence."""
    gap = un.CURATED6K_PRICE_COVERAGE_GAP
    assert set(gap) == {'XETRA', 'STO'}
    for code, (members, covered) in gap.items():
        assert code in un.exchanges(CUR6K), code
        assert members == un.CURATED6K_SIZING[code], (
            '%s: the price-coverage row says %d members but CURATED6K_SIZING says %d'
            % (code, members, un.CURATED6K_SIZING[code]))
        assert 0 <= covered < members
    assert un.CURATED6K_UNBACKTESTABLE == sum(m for m, _c in gap.values())
    #  It is a real fraction of the universe -- worth a banner -- but not most of it.
    assert 0.15 < un.CURATED6K_UNBACKTESTABLE / float(un.CURATED6K_ESTIMATED_MEMBERS) < 0.30


def test_the_sampled_banner_derives_its_top_100_share_instead_of_HARDCODING_3_PERCENT():
    """THE DEFECT A SECOND SAMPLED UNIVERSE EXPOSED (fixed 2026-08-21).  The sampled-universe
    warning said "a top-100 cut here is roughly the top 3% of the pool" as a literal.  That
    was measured for stock_CUR3K, but the block fires for EVERY sampled universe, so
    stock_CUR6K -- where the real figure is 1.7% -- would have been told 3%: a wrong number
    in the loudest place in the run, and wrong in the direction that makes the pool sound
    MORE distorted than it is.  Same class as the label counts that drifted."""
    for name in sorted(SAMPLED_UNIVERSES):
        exp = un.expected_count(name)
        want = '~%.1f%%' % (100.0 * 100.0 / exp)
        assert 'top %s of the pool' % want in un.run_banner(name), (
            '%s (%d members) should state a top-100 share of %s' % (name, exp, want))
    #  and the two universes must NOT be told the same share, which is the whole point
    assert (un.run_banner(CUR3K).count('top ~3.1% of the pool') == 1
            and un.run_banner(CUR6K).count('top ~1.7% of the pool') == 1)


def test_cur6k_the_dead_code_FLOOR_is_silent_on_the_real_replay_but_still_screams():
    """THE GUARD THAT MUST NOT CRY WOLF ON A SECOND SAMPLED UNIVERSE.

    This is the defect fixed on 2026-08-07: comparing a POST-SAMPLE count against a
    WHOLE-EXCHANGE number made five of stock_CUR3K's eight codes report 75-83% SHORT on
    EVERY run, all five false -- and a guard that fires five times a run is one the
    operator learns to skip, so the next genuinely dead code hides inside the noise of the
    guard built to find it.  stock_CUR6K has fourteen codes and two rate tiers, so it is
    the harder case.

    Asserted against the REPLAYED per-code counts, not against a synthetic healthy dict:
    a synthetic one is constructed to pass.
    """
    problems = un.check_resolved_counts(CUR6K, un.CURATED6K_SIZING)
    assert problems == [], (
        "the shortfall floor fires on stock_CUR6K's own replayed counts: %s" % problems)
    #  Every code must clear the floor with real margin, or the guard is one bad run from
    #  being noise. The floor UNDER-states by design (it cannot model upward closure), so
    #  the replay should sit at or above it.
    for c in un.exchanges(CUR6K):
        exp = un.expected_resolved_count(CUR6K, c)
        shortfall = 1.0 - (un.CURATED6K_SIZING[c] / exp)
        assert shortfall < 0.10, (
            '%s replays %d against a floor of %.0f (%.1f%% short) -- too close to the '
            '%.0f%% warn threshold for the guard to stay quiet on a normal run'
            % (c, un.CURATED6K_SIZING[c], exp, 100 * shortfall,
               100 * un.RESOLVED_SHORTFALL_WARN_ABOVE))
    #  AND THE DEAD-CODE SIGNAL SURVIVES, which is the whole point of the guard. Checked on
    #  a NEWLY-ADDED venue: if wiring six new codes had broken the floor for them, the four
    #  never-fetched venues would be exactly where a dead code could hide.
    for dead_code in ('OSL', 'BRU', 'LIS', 'ICE', 'XETRA', 'STO'):
        dead = dict(un.CURATED6K_SIZING, **{dead_code: 0})
        got = un.check_resolved_counts(CUR6K, dead)
        assert [g[0] for g in got] == [dead_code], (
            'a dead %s did not scream: %s' % (dead_code, got))
        assert got[0][3] == 1.0, '%s reported %.2f short, expected 100%%' % (dead_code,
                                                                            got[0][3])


def test_cur6k_lands_on_the_right_side_of_BOTH_panel_predicates():
    """The 2026-08-17 defect, checked for the NEW universe by name as well as by the generic
    loop in `test_a_CURATED_panel_can_never_stand_in_as_the_production_pool`.

    stock_CUR6K is curated by a SAMPLING RULE, so `un.symbols()` returns None for it exactly
    as it does for stock_CUR3K -- the shape that made a curated universe read as production.
    `is_explicit_list_panel` must be FALSE (it is not an explicit list) and
    `is_production_panel` must ALSO be false (it is not the production universe), and those
    two are different questions.
    """
    path = 'Bometric_dic-fmp_%s_all_2026-08-21_len6036.pickle' % CUR6K
    assert un.symbols(CUR6K) is None, 'CUR6K is rule-curated, not list-curated'
    assert vtu.panel_universe(path) == CUR6K, (
        'the panel filename must resolve to CUR6K and not to a name it contains as a '
        'substring -- panel_universe matches LONGEST first')
    assert vtu.is_explicit_list_panel(path) is False
    assert vtu.is_production_panel(path) is False
    #  and the two CUR names must not alias each other, which is the failure a shared
    #  prefix would cause
    assert vtu.panel_universe(
        'Bometric_dic-fmp_%s_all_2026-08-07_len3258.pickle' % CUR3K) == CUR3K


def test_cur6k_is_RESUMABLE_and_its_artifact_names_do_not_collide_with_cur3k(
        tmp_path, monkeypatch):
    """A newly-added universe must not be born unresumable -- the defect that left
    stock_US1_EU1 and stock_US1_EU2 raising 'Not Implemented' for -startfromlastindex after
    a partial multi-hour fetch had already been paid for.

    Exercised through the REAL `utils.get_lastIndexRead`, not just against
    `resume_filenames()`: the derivation is the thing that could break, and the whitelist
    check is inside that function.  `chdir(tmp_path)` because the function CREATES the
    resume file when the name is allowed, and a test must not drop one in the repo."""
    fn = un.resume_filename(CUR6K)
    assert fn == 'lastIndexOfRead_fmp_stock_CUR6K.txt'
    assert fn in un.resume_filenames()
    assert fn != un.resume_filename(CUR3K)
    monkeypatch.chdir(tmp_path)
    assert utils.get_lastIndexRead(fn) == 0, (
        'a fresh resume file for a newly-added universe must start at 0, not raise')
    with pytest.raises(Exception):
        utils.get_lastIndexRead('lastIndexOfRead_fmp_stock_NOT_A_UNIVERSE.txt')
    #  the fingerprint is distinct, so artifacts from the two runs can never be conflated
    assert un.definition_fingerprint(CUR6K) != un.definition_fingerprint(CUR3K)


def test_cur6k_provenance_stamp_carries_the_sample_and_the_pins():
    """The filter NAME is not sufficient provenance for a sampled universe: two runs at
    different rates carry the same name.  The stamp must carry the rates and the pins, and
    it must stay JSON-able so tooling need not import this module."""
    st = un.provenance(CUR6K)
    assert st['universe'] == CUR6K
    assert st['universe_sample'] == dict(un.CURATED6K_SAMPLED)
    assert st['universe_sample_denominator'] == un.CURATED6K_SAMPLE_DENOMINATOR
    assert len(st['universe_must_include']) == len(un.must_include(CUR6K))
    assert st['universe_expected_count'] == un.CURATED6K_ESTIMATED_MEMBERS
    assert st['universe_definition_changed'] is False
    json.dumps(st)


def test_the_recorded_subset_relationships_HOLD_against_the_live_definitions(
        live_capture_table, monkeypatch, tmp_path):
    """CONTAINS_AS_STRICT_SUBSET is read by `run_banner`, so it is a claim printed on every
    run.  Every entry is therefore replayed, both counts included -- a banner asserting a
    subset relationship that no longer holds is worse than no banner."""
    assert un.CONTAINS_AS_STRICT_SUBSET, 'nothing recorded; this test would pass vacuously'
    for big_name, (small_name, n_small, n_big) in un.CONTAINS_AS_STRICT_SUBSET.items():
        small = _replay(live_capture_table, small_name, monkeypatch, tmp_path)
        big = _replay(live_capture_table, big_name, monkeypatch, tmp_path)
        assert small <= big, '%s is NOT a subset of %s' % (small_name, big_name)
        assert small < big, '%s is not a STRICT subset of %s' % (small_name, big_name)
        assert (len(small), len(big)) == (n_small, n_big), (
            '%s/%s recorded as (%d, %d) but replays to (%d, %d)'
            % (small_name, big_name, n_small, n_big, len(small), len(big)))


# =========================================================================== #
#  -force_rebuild_maps  (CEO, 2026-08-10)                                      #
# =========================================================================== #
def test_force_rebuild_maps_BYPASSES_the_skip_conditions_and_SAYS_it_was_forced(
        tmp_path, monkeypatch, capsys):
    """*** WHY THE FLAG EXISTS. ***  On the 2026-08-10 run all four profile maps were present,
    none was 60 days stale and sector coverage was above the floor, so the gate skipped --
    CORRECTLY, by its own rules.  The consequence is that two capture changes that had already
    shipped never landed (`price`/`currency` from 90b0d5f; `isActivelyTrading` / `exchange` /
    `exchangeShortName` / `country` / `beta` from 1e9d353), every 2026-08-10 pick carries
    `volAvg_asof = 2026-08-07`, and nothing would force a rebuild until 2026-10-06.  A CODE
    change to WHAT the maps capture has no representation in a freshness rule about their AGE.

    BOTH HALVES ARE PINNED, because the second is the operational one: the rebuild happens,
    AND the banner distinguishes FORCED from TRIGGERED so a forced run is identifiable in the
    log after the fact.
    """
    import findAllSectors as fas
    monkeypatch.chdir(tmp_path)
    #  all four maps present and fresh -> every skip condition satisfied
    for name in ('sectorsdic_fmp.pickle', 'industrydic_fmp_2026-08-10.pickle',
                 'isindic_fmp_2026-08-10.pickle', 'volavgdic_fmp_2026-08-10.pickle'):
        pd.to_pickle({'Technology': ['A']} if name.startswith('sectors') else {'A': 'x'},
                     tmp_path / name)
    #  The CAPTURE-SCHEMA condition (2026-08-14) is the FIFTH skip condition, and this test is
    #  about the FLAG, so the schema is seeded as CURRENT to isolate it.  The schema trigger
    #  has its own test above -- and note it makes the flag less load-bearing than it was: the
    #  case this test's docstring describes now rebuilds on its own.  The flag stays for the
    #  reasons the code cannot see.
    fas.write_capture_schema(str(tmp_path), verbose=False)
    built = []
    monkeypatch.setattr(fas, 'buildSectorIndustryMaps',
                        lambda *a, **k: built.append(True) or ({}, {}))

    #  WITHOUT the flag: skipped, no build, and the log offers the flag by name.
    assert fas.ensure_sector_industry_maps(['A'], 'https://x/', 'KEY') is False
    out = capsys.readouterr().out
    assert not built and 'no rebuild, no API calls' in out
    assert '-force_rebuild_maps' in out, (
        'the skip banner must name the override, or an operator who needs it cannot find it')
    #  ...but it must NOT still offer the flag for the case the gate now handles ITSELF
    #  (review L-7, 2026-08-14).  A stale "pass the flag after a capture change" line builds a
    #  reflex flag around a gate that no longer needs one, and a reflex flag is how an operator
    #  stops reading the verdict.
    assert 'which no freshness rule can see' not in out, (
        'the skip banner still advises -force_rebuild_maps for a CAPTURE-SCHEMA change, which '
        'is now a skip condition the gate evaluates on its own')
    assert 'rebuilds on its own' in out

    #  WITH the flag: built, and the banner says FORCED rather than triggered.
    assert fas.ensure_sector_industry_maps(['A'], 'https://x/', 'KEY',
                                           force_rebuild=True) is True
    out = capsys.readouterr().out
    assert built, 'the flag did not bypass the skip conditions'
    assert 'FORCED' in out and 'TRIGGERED' not in out, (
        'a forced rebuild must be distinguishable from one the gate decided on its own')
    assert 'no gate condition fired' in out


def test_force_rebuild_maps_does_NOT_override_the_SUBSET_universe_refusal(
        tmp_path, monkeypatch, capsys):
    """The one thing the flag must NOT bypass.  A curated subset universe must never author
    these SHARED maps -- a 142-symbol map applied to a later 10,693-name pool covers ~1% of
    it, which is non-empty and so slips past carveOut's empty-map abort while REIT and Mining
    leak wholesale.  "The operator asked for it" is not a reason to write that map."""
    import findAllSectors as fas
    monkeypatch.chdir(tmp_path)
    built = []
    monkeypatch.setattr(fas, 'buildSectorIndustryMaps',
                        lambda *a, **k: built.append(True) or ({}, {}))
    assert fas.ensure_sector_industry_maps(['A'], 'https://x/', 'KEY',
                                           universe_is_subset=True,
                                           universe_name='stock_TEST1',
                                           force_rebuild=True) is False
    assert not built
    assert 'MAP BUILD SKIPPED' in capsys.readouterr().out


def test_the_force_rebuild_flag_is_THREADED_from_the_CLI_to_the_gate():
    """A flag that parses and reaches nothing is worse than no flag: it reports success and
    changes nothing.  The whole chain is checked by source, because the fetch path itself
    cannot be exercised without an API key."""
    import inspect
    import configuration as cfg
    import getData_gen as gdg
    ns, _ = cfg._build_parser().parse_known_args(['-force_rebuild_maps'])
    assert ns.force_rebuild_maps is not None
    assert 'force_rebuild_maps' in inspect.signature(gdg.get_tickers).parameters
    #  `get_tickers` is where the self-heal hook actually lives (NOT `tickerfilterWrapper`,
    #  which is a sibling call and never sees the maps) -- so that is the one function whose
    #  body has to carry the argument through.
    assert 'force_rebuild=bool(force_rebuild_maps)' in inspect.getsource(gdg.get_tickers)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Sbocker.py'),
              encoding='utf-8', errors='ignore') as f:
        assert "force_rebuild_maps=configdic.get('force_rebuild_maps')" in f.read()
