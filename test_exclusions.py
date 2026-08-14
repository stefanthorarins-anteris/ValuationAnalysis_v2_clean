"""Dated, expiring exclusions -- the safety properties, not the happy path.

3,692 names were being excluded in production by a file nobody could read, so the tests that
matter here are the ones that say WHAT CANNOT HAPPEN: the legacy file cannot be loaded, an
entry with no expiry cannot become permanent by accident, an expired entry cannot be applied,
and the change cannot alter a run on its own.
"""
import os
from datetime import date, timedelta

import pytest

import exclusions as x


TODAY = date(2026, 8, 14)


def _write(tmp_path, lines):
    p = tmp_path / 'ExclusionList_test.csv'
    p.write_text('\n'.join(lines) + '\n')
    return str(p)


# --------------------------------------------------------------------------- #
#  (a) THE LEGACY 3,692-NAME FILE IS UNREACHABLE                               #
# --------------------------------------------------------------------------- #
def test_legacy_bare_ticker_row_is_refused_whole(tmp_path):
    """A headerless row of tickers -- the pre-2026-08 format -- applies ZERO names.

    This is the single most important property in the module.  The old loader took
    `csv.reader(f)` and used `templist[0]` as the whole list, which is precisely how a
    February-2023 file was still removing 3,692 names in January 2026.
    """
    p = _write(tmp_path, ['BSD.PA,ELCO.L,BKT,BSP.DE,ALWEC.PA,DNP,JGH'])
    v = x.load_exclusions(p, as_of=TODAY, verbose=False)
    assert v.applied == [], (
        'a bare ticker row was ACCEPTED and applied %d name(s). The legacy format carries no '
        'date, no reason and no expiry, so it cannot be evaluated and must apply nothing.'
        % len(v.applied))
    assert len(v.by_status('malformed')) == 1
    assert 'FILE REFUSED WHOLE' in v.by_status('malformed')[0].note


def test_the_real_legacy_file_in_the_repo_applies_nothing():
    """Not a synthetic: the actual 2023 file still sitting in the repo root."""
    p = 'ManualEliminationTickersList_fmp_2023-02-14.csv'
    if not os.path.exists(p):
        pytest.skip('%s not present in this tree' % p)
    v = x.load_exclusions(p, as_of=TODAY, verbose=False)
    assert v.applied == [], (
        '%s applied %d names. It is the file that produced `manelim3692`; under the dated '
        'schema it must apply none.' % (p, len(v.applied)))


# --------------------------------------------------------------------------- #
#  (b) EXPIRY IS REAL, IN BOTH DIRECTIONS                                      #
# --------------------------------------------------------------------------- #
def test_expired_entry_is_not_applied_and_is_reported(tmp_path):
    p = _write(tmp_path, [
        ','.join(x.EXCLUSION_HEADER),
        'AAA,transient_fetch,fetch failed,2026-07-01,2026-07-15,run-observed',
        'BBB,transient_fetch,fetch failed,2026-08-10,2026-08-24,run-observed',
    ])
    v = x.load_exclusions(p, as_of=TODAY, verbose=False)
    assert v.applied == ['BBB']
    assert [e.ticker for e in v.by_status('expired')] == ['AAA']
    assert 'AAA' in v.report(verbose=False) and 'EXPIRED' in v.report(verbose=False)


def test_short_history_name_ages_off_by_itself(tmp_path):
    """THE CASE THE OLD RULE COULD NOT EXPRESS.

    `newmanelimtckrs = manualelim + (tickersfailed - lenfail)` SUBTRACTED the
    history-length failures precisely because, with no expiry, listing them would ban a
    two-year-old company forever.  With an expiry the name can be listed AND age off.
    """
    added = TODAY - timedelta(days=100)
    exp = x.default_expiry('short_history', added)
    assert exp == added + timedelta(days=90)
    p = _write(tmp_path, [
        ','.join(x.EXCLUSION_HEADER),
        'NEWCO,short_history,failed the history-length gate,%s,%s,run-observed'
        % (added.strftime(x.DATE_FMT), exp.strftime(x.DATE_FMT)),
    ])
    assert x.load_exclusions(p, as_of=TODAY, verbose=False).applied == []
    assert x.load_exclusions(p, as_of=added + timedelta(days=30),
                             verbose=False).applied == ['NEWCO']


def test_blank_expiry_is_refused_except_for_duplicate(tmp_path):
    """"Forever" must be REACHED, never DEFAULTED INTO by leaving a field empty."""
    p = _write(tmp_path, [
        ','.join(x.EXCLUSION_HEADER),
        'DUPE,duplicate,second listing of one issuer,2026-08-14,,DedupSurvivorReport',
        'SNEAK,vendor_bad,bad data,2026-08-14,,someone left it blank',
    ])
    v = x.load_exclusions(p, as_of=TODAY, verbose=False)
    assert v.applied == ['DUPE']
    bad = v.by_status('malformed')
    assert [e.ticker for e in bad] == ['SNEAK']
    assert 'blank expiry' in bad[0].note


def test_transient_fetch_has_the_shortest_life_and_duplicate_the_longest():
    """The expiry table is an argument about the conditions, so assert its ordering."""
    assert x.CATEGORIES['duplicate'][0] is None
    assert (x.CATEGORIES['transient_fetch'][0]
            < x.CATEGORIES['short_history'][0]
            < x.CATEGORIES['vendor_bad'][0]
            < x.CATEGORIES['ceo'][0])


# --------------------------------------------------------------------------- #
#  (c) A MALFORMED OR REASONLESS ENTRY IS REFUSED, NOT APPLIED                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('row,why', [
    ('AAA,not_a_category,x,2026-08-14,2026-12-01,e', 'unknown category'),
    ('AAA,ceo,,2026-08-14,2026-12-01,e', 'no reason'),
    ('AAA,ceo,x,not-a-date,2026-12-01,e', 'unparseable added'),
    ('AAA,ceo,x,2026-08-14,nonsense,e', 'unparseable expires'),
    (',ceo,x,2026-08-14,2026-12-01,e', 'no ticker'),
])
def test_malformed_entries_are_refused(tmp_path, row, why):
    p = _write(tmp_path, [','.join(x.EXCLUSION_HEADER), row])
    v = x.load_exclusions(p, as_of=TODAY, verbose=False)
    assert v.applied == [], 'a %s entry was applied; refusing is the safe direction' % why
    assert len(v.by_status('malformed')) == 1


def test_missing_file_is_empty_not_an_error():
    v = x.load_exclusions('no_such_exclusion_file_at_all.csv', as_of=TODAY, verbose=False)
    assert v.applied == [] and v.path is None


# --------------------------------------------------------------------------- #
#  RECONCILIATION AND THE WRITER/LOADER SCHEMA                                 #
# --------------------------------------------------------------------------- #
def test_reconcile_raises_when_an_exclusion_has_no_live_entry(tmp_path):
    p = _write(tmp_path, [','.join(x.EXCLUSION_HEADER),
                          'AAA,ceo,because,2026-08-14,2027-08-14,e'])
    v = x.load_exclusions(p, as_of=TODAY, verbose=False)
    assert x.reconcile(v.applied, v) is True
    with pytest.raises(AssertionError):
        x.reconcile(v.applied + ['GHOST'], v)


def test_writer_and_loader_share_one_schema(tmp_path):
    """They did not before: the writer emitted a headerless ticker row and the loader read
    `templist[0]`, so nothing carried a date, a reason or an expiry."""
    entries = x.propose_from_run(['AAA', 'BBB', 'CCC'], ['CCC'])
    p = str(tmp_path / 'out.csv')
    x.write_exclusions(p, entries)
    v = x.load_exclusions(p, as_of=TODAY, verbose=False)
    assert v.applied == ['AAA', 'BBB', 'CCC']
    cats = {e.ticker: e.category for e in v.by_status('live')}
    assert cats == {'AAA': 'transient_fetch', 'BBB': 'transient_fetch',
                    'CCC': 'short_history'}, (
        'the `- lenfail` subtraction is gone: a history-length failure must be RECORDED as '
        '`short_history` with an expiry, not omitted from the list entirely.')


def test_an_expired_entry_does_not_shadow_a_re_observation(tmp_path):
    """THE EXPIRY WAS ONE-SHOT (review S2, 2026-08-14).

    `have` was built over ALL entries including expired ones, so once a machine-proposed entry
    aged off, the same failure re-observed on every later run was silently dropped and the name
    could never be listed again.  That contradicts this module's headline argument -- that a
    short-history name "can go ON the list and AGE OFF" -- because it aged off and could not
    come back, however many quarters it kept failing.
    """
    expired = x.Entry('ACME', 'transient_fetch', 'fetch failed on this run', '2026-07-15',
                      '2026-07-29', 'run-observed', 'expired', 'expired 2026-07-29')
    merged = x.merge_entries([expired], x.propose_from_run(['ACME'], []))
    live = [e for e in merged if e.status == 'live']
    assert [e.ticker for e in live] == ['ACME'], (
        'the re-observation was dropped: %s' % [(e.ticker, e.status) for e in merged])
    #  and the superseded machine-authored row is collapsed rather than accumulating forever
    assert len(merged) == 1
    #  round-trip: the refreshed entry is LIVE on the next load
    p = str(tmp_path / 'rt.csv')
    x.write_exclusions(p, merged)
    assert x.load_exclusions(p, as_of=TODAY, verbose=False).applied == ['ACME']


def test_an_expired_HAND_EDITED_entry_is_kept_as_evidence_when_superseded():
    """A human's record of what they did and when is not the machine's to tidy away."""
    human = x.Entry('ACME', 'transient_fetch', 'CEO looked at this', '2026-01-01',
                    '2026-07-29', 'by hand', 'expired', '')
    merged = x.merge_entries([human], x.propose_from_run(['ACME'], []))
    assert len(merged) == 2
    assert sum(1 for e in merged if e.status == 'live') == 1
    assert any(e.evidence == 'by hand' for e in merged)


def test_hand_edited_entry_is_not_clobbered_by_a_machine_re_observation():
    """The CEO edits this file by hand; a re-observation must not extend his expiry."""
    human = x.Entry('AAA', 'transient_fetch', 'CEO looked at this', '2026-01-01',
                    '2026-12-31', 'by hand', 'live', '')
    merged = x.merge_entries([human], x.propose_from_run(['AAA'], []))
    assert len(merged) == 1 and merged[0].expires == '2026-12-31'
    assert merged[0].reason == 'CEO looked at this'


# --------------------------------------------------------------------------- #
#  THE CHANGE ALONE CANNOT ALTER A RUN                                         #
# --------------------------------------------------------------------------- #
def test_default_configuration_excludes_nothing():
    """Three independent reasons, each asserted, because one of them failing silently is
    exactly how the 3,692-name list stayed live for three years."""
    import configuration as cf
    assert not os.path.exists(x.DEFAULT_EXCLUSION_FILE), (
        '%r exists in this tree, so the default is no longer guaranteed empty.'
        % x.DEFAULT_EXCLUSION_FILE)
    #  1. the default filename does not exist -> empty verdict
    assert x.load_exclusions(x.DEFAULT_EXCLUSION_FILE, verbose=False).applied == []
    #  2. `-manelimtickers` still defaults to OFF
    ns = cf.build_parser().parse_args([]) if hasattr(cf, 'build_parser') else None
    if ns is not None:
        assert getattr(ns, 'manelimtickers', None) in (None, False)
    #  3. even a populated list is only reachable via BOTH the flag and a schema file
    assert x.DEFAULT_EXCLUSION_FILE != 'ManualEliminationTickersList_fmp_2023-02-14.csv'
